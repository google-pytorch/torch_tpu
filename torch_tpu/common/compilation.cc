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

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_split.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "torch_tpu/common/compilation_spec.h"
#include "torch_tpu/common/compile_options_key.h"
#include "torch_tpu/common/context_manager.h"
#include "torch_tpu/common/context_states.h"
#include "torch_tpu/common/env_vars.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/pjrt/pjrt_state.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/translate/register.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

namespace torch_tpu {
namespace internal {

// Fingerprint specialization for `xla::ExecutionOptions::EffortLevel`.
template <>
struct Fingerprint64Impl<xla::ExecutionOptions::EffortLevel,
                         /*kIsSmallIntegral=*/false> {
  [[nodiscard]] static FingerprintType Compute(
      const xla::ExecutionOptions::EffortLevel effort_level) {
    // The proto enum string names for EffortLevel are unique and stable as
    // they are used for parameter cache keys.
    return Fingerprint(xla::ExecutionOptions::EffortLevel_Name(effort_level));
  }
};

}  // namespace internal

namespace {

// Names of special XLA compiler options.
constexpr std::string_view kOptimizationLevelOption = "xla_optimization_level";
constexpr std::string_view kMemoryFittingLevelOption =
    "xla_memory_fitting_level";

// Represents the state of the "allow excess precision" compiler option.
enum class ExcessPrecisionState {
  // Use XLA_FLAGS or fall back to the default.
  kUnset,
  // Explicitly allow excess precision.
  kAllow,
  // Explicitly disallow excess precision.
  kDisallow,
};


// Returns the cached default `xla::ExecutableBuildOptions` instance. This
// function is memoized, so the XLA_FLAGS environment variable is parsed exactly
// once.
const xla::ExecutableBuildOptions& GetDefaultExecutableBuildOptions() {
  static const absl::NoDestructor<xla::ExecutableBuildOptions>
      default_executable_build_options([] {
        xla::ExecutableBuildOptions options;
        // Calling mutable_debug_options() triggers parsing XLA_FLAGS.
        options.mutable_debug_options();
        return options;
      }());
  return *default_executable_build_options;
}

}  // namespace

absl_nonnull std::unique_ptr<mlir::MLIRContext> MakeMlirContext() {
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::DialectRegistry registry;
  xla::RegisterMlirToHloDependentDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
  return context;
}

ContextedModule::ContextedModule(std::unique_ptr<mlir::MLIRContext> context,
                                 mlir::OwningOpRef<mlir::ModuleOp> module)
    : context_(std::move(context)), module_(std::move(module)) {}

absl::StatusOr<ContextedModule> ContextedModule::Make(
    const MlirComputationBuilder& computation_builder) {
  auto context = MakeMlirContext();
  TT_ASSIGN_OR_RETURN(auto module, computation_builder(*context));
  return ContextedModule(std::move(context), std::move(module));
}

absl::StatusOr<SharedLoadedExecutableWithMetadata>
LoadedExecutableWithMetadata::MakeShared(
    absl_nonnull std::unique_ptr<xla::PjRtLoadedExecutable> executable) {
  TT_RET_CHECK(executable, error::kInternal)
      << "cannot create SharedLoadedExecutableWithMetadata from null "
         "executable.";

  // Get the flattened output shapes from the executable.
  // Executables can return tuples, we want individual tensor shapes.
  TT_ASSIGN_OR_RETURN(std::vector<xla::Shape> output_shapes_vec,
                      executable->GetOutputShapes());
  std::vector<const xla::Shape*> output_shapes_flat_vec;
  for (const auto& output_shape : output_shapes_vec) {
    xla::ShapeUtil::FlattenTupleShape(output_shape, output_shapes_flat_vec);
  }

  std::vector<Shape> output_shapes;
  output_shapes.reserve(output_shapes_flat_vec.size());
  for (const auto* xla_shape : output_shapes_flat_vec) {
    TT_ASSIGN_OR_RETURN(Shape shape, MakeShape(*xla_shape));
    output_shapes.push_back(std::move(shape));
  }

  return std::shared_ptr<const LoadedExecutableWithMetadata>(
      new LoadedExecutableWithMetadata(std::move(executable),
                                       std::move(output_shapes)));
}

absl::StatusOr<SharedLoadedExecutableWithMetadata> Compile(
    xla::PjRtClient& client, LoadedExecutableBuilder executable_builder,
    UniqueCompileOptions compile_options) {
  TT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtLoadedExecutable> executable,
      std::move(executable_builder)(client, std::move(compile_options)));
  return LoadedExecutableWithMetadata::MakeShared(std::move(executable));
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

// Updates `map` with the contents of `updates`. If a key appears in both `map`
// and `updates`, the value from `updates` is used.
template <typename Map>
static void UpdateMap(Map& map, Map updates) {
  for (auto& [key, value] : updates) {
    map[std::move(key)] = std::move(value);
  }
}

absl::Status PushCompilerOptionOverrides(CompilerOptionOverrides overrides) {
  // When we push the overrides, the new overrides are merged with the existing
  // overrides. The innermost map takes precedence.
  auto merged_overrides =
      GetContextState<CustomCompilerOptionsContextState>({});
  UpdateMap(merged_overrides, std::move(overrides));
  PushContextState<CustomCompilerOptionsContextState>(
      std::move(merged_overrides));

  // TODO(b/502270689): return the actual status when users push invalid options
  // at context entry time.
  return absl::OkStatus();
}

void PopCompilerOptionOverrides() {
  PopContextState<CustomCompilerOptionsContextState>();
}

static std::atomic<ExcessPrecisionState> g_allow_excess_precision{
    ExcessPrecisionState::kUnset};

// TODO: b/498591854 - Remove this callback once allow_excess_precision is
// included in the cache key. This is a temporary solution to avoid a circular
// dependency between the low-level compilation.cc and the higher-level
// compilation_cache.cc. We cannot call cache eviction in compilation_cache
// directly from here. compilation_cache would fill this callback with the
// eviction function and we call this callback instead. This is called before
// the value of g_allow_excess_precision is changed.
std::atomic<void (*)()> g_excess_precision_change_callback{nullptr};

void SetAllowExcessPrecision(bool allow) {
  if (GetAllowExcessPrecision() != allow) {
    if (void (*callback)() = g_excess_precision_change_callback.load()) {
      callback();
    }
  }
  g_allow_excess_precision.store(allow ? ExcessPrecisionState::kAllow
                                       : ExcessPrecisionState::kDisallow);
}

bool GetAllowExcessPrecision() {
  switch (g_allow_excess_precision.load()) {
    case ExcessPrecisionState::kAllow:
      return true;
    case ExcessPrecisionState::kDisallow:
      return false;
    case ExcessPrecisionState::kUnset:
      // Fall back to XLA_FLAGS or default to true.
      // Use static to memoize the value to avoid reading the debug options
      // multiple times.
      static const bool allow_excess_precision = [] {
        const xla::DebugOptions& debug_options =
            GetDefaultExecutableBuildOptions().debug_options();
        if (debug_options.has_xla_allow_excess_precision()) {
          return debug_options.xla_allow_excess_precision();
        }
        // TODO: b/502610173 - Set to False when XLA_FLAGS is not set.
        return true;
      }();
      return allow_excess_precision;
  }
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
        std::pair<std::string, std::string>  // STD_PAIR_OK=dictated by absl.
            p = absl::StrSplit(flag, absl::MaxSplits('=', 1));
        overrides[p.first] = p.second;
      }
    }
    return overrides;
  }());
  return *overrides;
}

// Updates the `executable_build_options.debug_options` field of the given
// `options` with the settings derived from the XLA_FLAGS environment variable.
// LINT.IfChange
static void UpdateFromXlaFlags(xla::CompileOptions& options) {
  // Use the cached `ExecutableBuildOptions` to avoid parsing the XLA_FLAGS
  // environment variable multiple times when calling `mutable_debug_options`.
  options.executable_build_options = GetDefaultExecutableBuildOptions();

  xla::DebugOptions* debug_options =
      options.executable_build_options.mutable_debug_options();
  debug_options->set_xla_allow_excess_precision(GetAllowExcessPrecision());
}
// LINT.ThenChange(:xla_flags_fingerprint)

// Returns fingerprint of raw string value read from XLA_FLAGS environment
// variable. The environment variable is only read exactly once per process.
//
// It is intended not to replicate parsing XLA_FLAGS into `tsl::Flag`. The
// current approach is simpler and practically sufficient to capture XLA
// behavioral changes controlled by XLA_FLAGS. Semantically identical flags in a
// different order will yield different keys, but it is an acceptable trade-off.
//
// LINT.IfChange(xla_flags_fingerprint)
[[nodiscard]] static FingerprintType GetXlaFlagsFingerprint() {
  const std::optional<std::string>& xla_flags = GetEnvOnce<kXlaFlagsEnvVar>();
  return xla_flags.has_value() ? Fingerprint(*xla_flags) : FingerprintCat();
}
// LINT.ThenChange()

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
  // LINT.IfChange
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
  // LINT.ThenChange(:xla_executable_build_options_fingerprint)

  return absl::OkStatus();
}

// Returns fingerprint of `xla::ExecutableBuildOptions` fields that are used
// in `SetDefaultDeviceAssignment` and `ApplyCompilerOptionOverrides`.
//
// Note that `options.debug_options` is excluded because it is already included
// in the XLA_FLAGS fingerprint (via `GetXlaFlagsFingerprint`).
//
// LINT.IfChange(xla_executable_build_options_fingerprint)
[[nodiscard]] static FingerprintType GetXlaExecutableBuildOptionsFingerprint(
    const xla::ExecutableBuildOptions& options) {
  return FingerprintCat(
      // Device assignment related fields.
      options.num_replicas(), options.num_partitions(),
      options.use_spmd_partitioning(), options.use_shardy_partitioner(),
      // XLA execution effort levels.
      options.optimization_level(), options.memory_fitting_level());
}
// LINT.ThenChange()

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

static CompilerOptionOverrides MakeCompilerOptionOverrides(
    const bool is_tpu, const CompilationMode mode) {
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
  UpdateMap(overrides, GetCompilerOptionOverridesFromEnvVar());
  // When merging the overrides, the Python context manager takes precedence.
  UpdateMap(overrides, GetContextState<CustomCompilerOptionsContextState>({}));
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
    // LINT.IfChange
    if (key == kOptimizationLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_optimization_level(level);
    } else if (key == kMemoryFittingLevelOption) {
      TT_ASSIGN_OR_RETURN(const auto level, ParseEffortLevel(value));
      compile_options.executable_build_options.set_memory_fitting_level(level);
      // LINT.ThenChange(:xla_executable_build_options_fingerprint)
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

// LINT.IfChange
absl::StatusOr<UniqueCompileOptions> MakeCompilerOptions(CompilationMode mode) {
  auto compile_options = std::make_unique<xla::CompileOptions>();

  // Set options derived from XLA_FLAGS environment variable in the
  // `executable_build_options.debug_options` field.
  UpdateFromXlaFlags(*compile_options);

  // Set device-related options, primarily global device count, in the
  // `executable_build_options` field.
  TT_RETURN_IF_ERROR(
      SetDefaultDeviceAssignment(compile_options->executable_build_options));

  // Finally, override the default flags if needed. The overrides are eventually
  // reflected in the `env_option_overrides` field.
  TT_ASSIGN_OR_RETURN(const bool is_tpu, SetTpuOptions(*compile_options));
  TT_RETURN_IF_ERROR(ApplyCompilerOptionOverrides(
      MakeCompilerOptionOverrides(is_tpu, mode), *compile_options));

  return compile_options;
}
// LINT.ThenChange(:compile_options_key)

// LINT.IfChange(compile_options_key)
[[nodiscard]] CompileOptionsKey GetCompileOptionsKey(
    const xla::CompileOptions& options) {
  return CompileOptionsKey(FingerprintCat(
      // Fingerprint of XLA_FLAGS environment variable.
      GetXlaFlagsFingerprint(),
      // Fingerprint of XLA executable build options, mainly device-related
      // information.
      GetXlaExecutableBuildOptionsFingerprint(options.executable_build_options),
      // Fingerprint of resolved eventual compiler option overrides, including
      // TorchTPU defaults, TorchTPU-internal and explicit user overrides.
      Fingerprint(options.env_option_overrides)));
}
// LINT.ThenChange()

absl::StatusOr<CompilationSpecsByMode> MakeCompilationSpecs(
    CompilationMode mode) {
  CompilationSpecsByMode specs;

  TT_ASSIGN_OR_RETURN(UniqueCompileOptions compile_options,
                      MakeCompilerOptions(mode));
  specs.try_emplace(
      mode, std::move(compile_options),
      // TODO(b/502270689): use `GetCompileOptionsKey to generate fingerprint.
      CompileOptionsKey(0));

  return specs;
}

}  // namespace torch_tpu
