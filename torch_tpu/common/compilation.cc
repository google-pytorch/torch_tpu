/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "torch_tpu/common/compilation.h"

#include <map>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/translate/register.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/xla.pb.h"

namespace torch_tpu {

// Names of special XLA compiler options.
constexpr std::string_view kOptimizationLevelOption = "xla_optimization_level";
constexpr std::string_view kMemoryFittingLevelOption =
    "xla_memory_fitting_level";
constexpr std::string_view kAllowExcessPrecisionOption =
    "xla_allow_excess_precision";

namespace {
absl_nonnull std::unique_ptr<mlir::MLIRContext> MakeMlirContext() {
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  return context;
}
}  // namespace

ContextedModule::ContextedModule(std::unique_ptr<mlir::MLIRContext> context,
                                 mlir::OwningOpRef<mlir::ModuleOp> module)
    : context_(std::move(context)), module_(std::move(module)) {}

absl::StatusOr<ContextedModule> ContextedModule::Make(
    const MlirComputationBuilder& computation_builder) {
  auto context = MakeMlirContext();
  TT_ASSIGN_OR_RETURN(auto module, computation_builder(*context));
  return ContextedModule(std::move(context), std::move(module));
}

absl::StatusOr<SharedLoadedExecutable> Compile(
    xla::PjRtClient& client, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      std::move(executable_builder)(client, std::move(compile_options)));
  TT_RET_CHECK(executable, error::kInternal)
      << "compilation succeeded but returned a null executable.";
  return executable;
}

absl::StatusOr<LoadedExecutableBuilder>
MlirComputationBuilderToExecutableBuilder(
    const MlirComputationBuilder& computation_builder) {
  TT_ASSIGN_OR_RETURN(ContextedModule contexted_module,
                      ContextedModule::Make(computation_builder));
  return [contexted_module = std::move(contexted_module)](
             xla::PjRtClient& client, UniqueCompileOptions options) mutable
             -> absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> {
    return client.CompileAndLoad(
        std::move(contexted_module).ToMaybeOwningMlirModule(),
        std::move(*options));
  };
}

// Returns the compile option overrides for the current thread as a stack.
// Each element in the stack represents a set of overrides that have been
// pushed but not yet popped. Each time we push a set of overrides, it is
// merged with the existing overrides (later pushes take precedence).
//
// For example, given Python code:
//
//  with compiler.custom_compiler_options({
//      "foo": "1",
//      "bar": "2",
//  }):
//    f().to("cpu")  // triggers compilation
//    with compiler.custom_compiler_options({
//      "bar": "3",
//      "baz": "4",
//    }):
//      g().to("cpu")  // triggers compilation
//    h().to("cpu")  // triggers compilation
//
// 1. initially the stack is empty,
// 2. when we are in the outer custom_compiler_options context but not in
//    the inner one, the stack contains one element
//      { "foo": "1", "bar": "2" },
// 3. when we are in the inner custom_compiler_options context, the stack
//    contains two elements:
//      (top)    { "foo": "1", "bar": "3", "baz": "4" }
//      (bottom) { "foo": "1", "bar": "2" }
//
// Therefore f() and h() will be compiled with { "foo": "1", "bar": "2" },
// and g() will be compiled with { "foo": "1", "bar": "3", "baz": "4" }.
[[nodiscard]] static std::stack<CompilerOptionOverrides>&
GetMutableCompileOptionOverridesStack() {
  // User PyTorch code may set different overrides in different Python
  // threads, so this needs to be thread-local.
  static thread_local std::stack<CompilerOptionOverrides> overrides;
  return overrides;
}

// Returns the compile option overrides for the current thread, as set in
// the user Python code. Thread-safe.
[[nodiscard]] CompilerOptionOverrides static GetCompilerOptionOverridesFromPython() {
  const auto& stack = GetMutableCompileOptionOverridesStack();
  if (stack.empty()) {
    return {};
  }
  return stack.top();
}

// Updates `map` with the contents of `updates`. If a key appears in both `map`
// and `updates`, the value from `updates` is used.
template <typename Map>
static void UpdateMap(Map& map, Map updates) {
  for (auto& [key, value] : updates) {
    map[std::move(key)] = std::move(value);
  }
}

void PushCompilerOptionOverrides(CompilerOptionOverrides overrides) {
  // When we push the overrides, the new overrides are merged with the existing
  // overrides. The innermost map takes precedence.
  auto merged_overrides = GetCompilerOptionOverridesFromPython();
  UpdateMap(merged_overrides, std::move(overrides));
  GetMutableCompileOptionOverridesStack().push(std::move(merged_overrides));
}

void PopCompilerOptionOverrides() {
  auto& stack = GetMutableCompileOptionOverridesStack();
  ABSL_CHECK(!stack.empty())  // CRASH_OK
      << "Attempted to pop empty compiler option overrides stack. This is a "
         "bug in TorchTPU.";
  stack.pop();
}

// Fingerprints the given compiler option overrides into a single 64-bit value.
// This is used as part of the compilation cache key.
FingerprintType FingerprintCompilerOptions(
    const CompilerOptionOverrides& overrides) {
  FingerprintType fp = overrides.size();
  for (const auto& [key, value] : overrides) {
    fp = FingerprintCat(fp,
                        FingerprintCat(Fingerprint(key), Fingerprint(value)));
  }
  return fp;
}

static absl::StatusOr<xla::ExecutionOptions::EffortLevel> ParseEffortLevel(
    std::string_view level) {
  if (level == "O0") {
    return xla::ExecutionOptions::EFFORT_O0;
  }
  if (level == "O1") {
    return xla::ExecutionOptions::EFFORT_O1;
  }
  if (level == "O2") {
    return xla::ExecutionOptions::EFFORT_O2;
  }
  if (level == "O3") {
    return xla::ExecutionOptions::EFFORT_O3;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "unknown XLA compiler effort level: " << level;
}

// Returns the compile option overrides from the environment variable
// TORCH_TPU_INTERNAL_XLA_OPTIONS, in the format of "key1=value1 key2=value2
// ...".
//
// This function is memoized, so the environment variable is only read once.
static const CompilerOptionOverrides& GetCompilerOptionOverridesFromEnvVar() {
  static const absl::NoDestructor<CompilerOptionOverrides> overrides([] {
    CompilerOptionOverrides overrides;
    const auto& xla_flags = GetEnvOnce<kTorchTpuInternalXlaOptionsEnvVar>();
    if (xla_flags.has_value()) {
      for (std::string_view flag : absl::StrSplit(*xla_flags, ' ')) {
        if (flag.empty()) {
          continue;
        }
        std::pair<std::string, std::string> p =
            absl::StrSplit(flag, absl::MaxSplits('=', 1));
        overrides[p.first] = p.second;
      }
    }
    return overrides;
  }());
  return *overrides;
}

static absl::Status SetDefaultDeviceAssignment(
    xla::ExecutableBuildOptions& options) {
  TT_ASSIGN_OR_RETURN(const int num_devices,
                      PjrtBackend::GetInstance().GetGlobalDeviceCount());
  // These options are sensible for 1 partition and 1 replica per device,
  // which aligns with typical single-host parallelism in native PyTorch.
  // Every device runs identical HLO and collectives are included explicitly.
  // We will revisit this as we explore multiple hosts, shardy, etc.

  // Use of multiple partitions aligns with JAX collective behavior and is
  // required by Pallas/Mosaic.
  options.set_num_replicas(1);
  options.set_num_partitions(num_devices);
  xla::DeviceAssignment da(1, num_devices);
  for (int idx = 0; idx < num_devices; ++idx) {
    da(0, idx) = idx;
  }
  options.set_device_assignment(da);

  // Enable SPMD partitioning in order to compile Pallas / Mosaic
  // kernels with RDMAs. Enable Shardy as it is needed for ragged_dot. Since
  // TorchTPU generates StableHLO without sharding annotations, the partitioning
  // will be a no-op.
  options.set_use_spmd_partitioning(true);
  options.set_use_shardy_partitioner(true);

  return absl::OkStatus();
}

// Sets `options.env_option_overrides` to TPU-specific options, sorted by key
// in ascending order.
//
// Returns whether the current device is a TPU.
static absl::StatusOr<bool> SetTpuOptions(xla::CompileOptions& options) {
  const xla::PjRtClient* const client = PjrtBackend::GetInstance().GetClient();
  TT_RET_CHECK(client, error::kFailedPrecondition)
      << "PjRtClient must be initialized.";
  const bool is_tpu = client->platform_id() == xla::TpuId();
  if (is_tpu) {
    // WARNING: When assigning string values below, make sure to explicitly
    // pass std::string objects as opposed to using string literals, since the
    // C++ compiler used for the OSS version incorrectly casts const char* to
    // bool as opposed to building an std::string object.
    static const absl::NoDestructor<
        xla::CompileOptions::EnvironmentOptionOverrides>
        tpu_env_option_overrides({
            // Important: keep the keys sorted alphabetically - the override
            // merging logic relies on this.
            // go/keep-sorted start
            // Reduces TPU binary size by using calls for deduplicated HLOs.
            {"xla_tpu_enable_deduplicated_calls", std::string("ENABLED")},
            // Enable "safe" XLA scavenge mode (where "safe" is needed by
            // Pallas), which helps with a bit for performance, but also helps
            // kernels which specify their own scoped limit.
            {"xla_tpu_vmem_scavenging_mode", std::string("SAFE")},
            // go/keep-sorted end
        });
    options.env_option_overrides = *tpu_env_option_overrides;
  }
  return is_tpu;
}

// Return the overrides based on the compilation mode.
static CompilerOptionOverrides MakeCompilerOptionOverrides(
    const bool is_tpu, const CompilationMode mode,
    std::optional<CompilerOptionOverrides> explicit_overrides) {
  CompilerOptionOverrides overrides;
  if (is_tpu && mode == CompilationMode::kFastCompile) {
    // Use O1 for the eager mode and the default optimization level (O2) for
    // the torch.compile mode.
    overrides[std::string(kOptimizationLevelOption)] = "O1";
    // We want lowering parameters to be chosen according to the optimization
    // level, rather than being pulled out of the autofdo database.
    // TODO(b/456145756): Remove this once we figure out the interaction
    // between autofdo and optimization level.
    overrides["xla_tpu_autofdo"] = "false";
  }
  if (explicit_overrides.has_value()) {
    UpdateMap(overrides, std::move(*explicit_overrides));
  }
  return overrides;
}

// Updates `compile_options.env_option_overrides` with the given overrides.
//
// If a key in `overrides` is already present in the former, the corresponding
// value in the former is updated. Otherwise, the (key, value) pair is appended
// to the former.
//
// The special keys "xla_optimization_level" and "xla_memory_fitting_level"
// are translated separately.
absl::Status ApplyCompilerOptionOverrides(
    // Pass `overrides` by value as we need to move the values from the map
    // and the caller passes a temporary object.
    CompilerOptionOverrides overrides, xla::CompileOptions& compile_options) {
  ABSL_VLOG(1) << "Setting XLA compiler option override: start";
  const int original_size = compile_options.env_option_overrides.size();
  int cur_override_idx = 0;
  static_assert(
      std::is_same_v<decltype(overrides), std::map<std::string, std::string>>,
      "The override merge logic relies on the elements in `overrides` being "
      "sorted by key in ascending order. If the type of `overrides` changes, "
      "we must revisit the logic.");
  for (auto& [key, value] : overrides) {
    // `key` is a const std::string&. `value` is a std::string&.
    ABSL_VLOG(1) << "Setting XLA compiler option override: "
                 << absl::CEscape(key) << " = " << absl::CEscape(value);
    TT_RET_CHECK(!key.empty(), error::kInvalidArgument)
        << "XLA compiler option name must not be empty";
    if (key == kOptimizationLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_optimization_level(level);
    } else if (key == kMemoryFittingLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_memory_fitting_level(level);
    } else if (key == kAllowExcessPrecisionOption) {
      bool enabled;
      if (value == "true") {
        enabled = true;
      } else if (value == "false") {
        enabled = false;
      } else {
        return TT_ERROR(error::kInvalidArgument)
               << "invalid value for " << kAllowExcessPrecisionOption << ": "
               << value << ". Expected 'true' or 'false'.";
      }
      compile_options.executable_build_options.mutable_debug_options()
          ->set_xla_allow_excess_precision(enabled);
    } else {
      // Apply a new override by replacing the entry with the same key, if
      // present, or by appending the new (key, value) pair at the end fo the
      // options.
      bool replaced = false;
      // Since the keys in `overrides` and
      // `compile_options.env_option_overrides` are both sorted alphabetically,
      // to search for `key` in the latter, we can just continue from where we
      // left off in the previous iteration.
      //
      // We search up to the original size, as there's no need to check the
      // newly appended elements.
      for (; cur_override_idx < original_size; ++cur_override_idx) {
        auto& [existing_key, existing_value] =
            compile_options.env_option_overrides[cur_override_idx];
        if (existing_key == key) {
          replaced = true;
          existing_value = std::move(value);
          ++cur_override_idx;
          break;
        }
      }
      if (!replaced) {
        // This invalidates iterators on compile_options.env_option_overrides.
        // This is why we use an index instead of an iterator.
        compile_options.env_option_overrides.push_back(
            // This move is safe despite the move in the for-loop above, as it's
            // done when when replaced is false.
            std::make_pair(key, std::move(value)));
      }
    }
  }
  ABSL_VLOG(1) << "Setting XLA compiler option override: end";
  return absl::OkStatus();
}

CompilerOptionOverrides GetActiveCompilerOptionOverrides() {
  // When merging the overrides, the environment variable takes precedence.
  CompilerOptionOverrides overrides = GetCompilerOptionOverridesFromEnvVar();
  UpdateMap(overrides, GetCompilerOptionOverridesFromPython());
  return overrides;
}

absl::StatusOr<UniqueCompileOptions> MakeCompilerOptions(
    CompilationMode mode, std::optional<CompilerOptionOverrides> overrides) {
  auto compile_options = std::make_unique<xla::CompileOptions>();
  // Call mutable_debug_options to parse XLA_FLAGS into compile options.
  compile_options->executable_build_options.mutable_debug_options();
  TT_RETURN_IF_ERROR(
      SetDefaultDeviceAssignment(compile_options->executable_build_options));

  TT_ASSIGN_OR_RETURN(const bool is_tpu, SetTpuOptions(*compile_options));

  // Finally, override the default flags if needed.
  auto merged_overrides =
      MakeCompilerOptionOverrides(is_tpu, mode, std::move(overrides));
  TT_RETURN_IF_ERROR(ApplyCompilerOptionOverrides(std::move(merged_overrides),
                                                  *compile_options));

  return std::move(compile_options);
}

}  // namespace torch_tpu
