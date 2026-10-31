/*
 * Copyright 2026 Google LLC
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

#ifndef TORCH_TPU_CSRC_COMMON_ENV_VARS_H_
#define TORCH_TPU_CSRC_COMMON_ENV_VARS_H_

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "absl/base/no_destructor.h"
#include "absl/strings/numbers.h"
#include "c10/util/Exception.h"
#include "csrc/common/constexpr_map.h"
#include "csrc/common/symbol_stage.h"
#include "csrc/common/warnings.h"

namespace torch_tpu {

// All environment variables read or written by torch_tpu.
//
// If an env var is not introduced by TorchTPU, please document the source.
//
// Since std::getenv() and setenv() take a const char*, and conversion from
// std::string_view to const char* is expensive, we define these as C-string
// literals.

// go/keep-sorted start
inline constexpr char kAcceleratorTypeEnvVar[] =
    "ACCELERATOR_TYPE";  // Set by Google Cloud.
inline constexpr char kAllowMultipleLibtpuLoadEnvVar[] =
    "ALLOW_MULTIPLE_LIBTPU_LOAD";  // Read by libtpu.
inline constexpr char kCloudTpuTaskIdEnvVar[] =
    "CLOUD_TPU_TASK_ID";  // Read by Google Cloud.
// The index of this node within a Kubernetes Indexed Job. Read by TorchTPU
// launchers to derive NODE_RANK.
inline constexpr char kJobCompletionIndexEnvVar[] = "JOB_COMPLETION_INDEX";
inline constexpr char kLibtpuInitArgsEnvVar[] =
    "LIBTPU_INIT_ARGS";  // Set by Google Cloud.
inline constexpr char kLocalRankEnvVar[] = "LOCAL_RANK";    // Set by launchers.
inline constexpr char kMasterAddrEnvVar[] = "MASTER_ADDR";  // Set by launchers.
inline constexpr char kMasterPortEnvVar[] = "MASTER_PORT";  // Set by launchers.
// The MegaScale debug server port of this process. Read by libtpu, which
// defaults it to 8081 in every process.
inline constexpr char kMegascaleDebugPortEnvVar[] = "MEGASCALE_DEBUG_PORT";
// The MegaScale (DCN) port of this process in a multi-slice job. Read by
// libtpu, which defaults it to 8080 in every process.
inline constexpr char kMegascalePortEnvVar[] = "MEGASCALE_PORT";
// The number of nodes in a multi-host job. Set by launchers like torchrun.
inline constexpr char kNnodesEnvVar[] = "NNODES";
// The rank of this node within a multi-host job. Set by launchers like
// torchrun.
inline constexpr char kNodeRankEnvVar[] = "NODE_RANK";
// Set by the build system (e.g. bazel) to indicate the number of shards
// running on the same host.
inline constexpr char kNprocEnvVar[] = "NPROC";
inline constexpr char kRankEnvVar[] = "RANK";  // Set by launchers.
// Read by libtpu: if set to any value, new TPU PJRT clients skip MegaScale.
inline constexpr char kSkipMegascalePjrtClientEnvVar[] =
    "SKIP_MEGASCALE_PJRT_CLIENT";
// Standard temporary directory.
inline constexpr char kTmpdirEnvVar[] = "TMPDIR";
// If set to "0", PyTorch skips autoloading out-of-tree device backends.
// TorchTPU reads it and sets it to "0" while `torch_tpu._loader` imports
// `torch`, to break the autoload import cycle, then restores the original
// value. Owned by PyTorch:
// https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html
inline constexpr char kTorchDeviceBackendAutoloadEnvVar[] =
    "TORCH_DEVICE_BACKEND_AUTOLOAD";
// If set to "1", C++ stack traces are appended to error messages.
// Shared with PyTorch:
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html#pytorch-debug-environment-variables
inline constexpr char kTorchShowCppStacktracesEnvVar[] =
    "TORCH_SHOW_CPP_STACKTRACES";
// If set to "1" or "true", calling an internal `torch.tpu` device API that
// has a `torch.accelerator` equivalent (e.g. `torch.tpu.device_count`) from
// outside the PyTorch framework emits a one-time warning instead of raising a
// RuntimeError. Default is "false".
inline constexpr char kTorchTpuAllowLegacyDeviceApisEnvVar[] =
    "TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS";
// The first port used when deriving slicebuilder addresses for a multi-host
// job. If not set, 8070 is used.
inline constexpr char kTorchTpuBasePortEnvVar[] = "TORCH_TPU_BASE_PORT";
inline constexpr char kTorchTpuDeferAndFuseEnvVar[] =
    "TORCH_TPU_DEFER_AND_FUSE";  // Used to enable EagerMode::kDeferAndFuse by
                                 // default.
// The number of TPU devices on each host, used when deriving slicebuilder
// addresses for a multi-host job. If not set, 8 is used.
inline constexpr char kTorchTpuDevicesPerHostEnvVar[] =
    "TORCH_TPU_DEVICES_PER_HOST";
// If set to "1", enable expensive debug checks in TorchTPU. This catches
// more bugs in user code, but comes at a significant performance cost for
// some ops. The debug eager mode enables these checks by default, but can
// be overridden by setting this env var to "0".
inline constexpr char kTorchTpuEnableDebugChecksEnvVar[] =
    "TORCH_TPU_ENABLE_DEBUG_CHECKS";
// Specifies the fixed port number for the handshake server.
inline constexpr char kTorchTpuHandshakePortEnvVar[] =
    "TORCH_TPU_INTERNAL_HANDSHAKE_PORT";
// Specifies the repeated op detection mode. If not set, "safe" mode is used.
// Supported modes:
// - "safe": Uses GraphSplitMode::kSplitGraph when a repeated sequence is
// detected.
// - "aggressive": Uses GraphSplitMode::kFullGraph when a repeated sequence
// is detected.
// Other values are no-ops.
inline constexpr char kTorchTpuInternalDetectRepeatedOpsEnvVar[] =
    "TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS";
// If set to "1" or "true", disables input buffer donation for in-place ATen
// operations in eager mode. Default is false (buffer donation is enabled).
inline constexpr char kTorchTpuInternalDisableInplaceBufferDonationEnvVar[] =
    "TORCH_TPU_INTERNAL_DISABLE_INPLACE_BUFFER_DONATION";
// Specifies where to dump optimization pass IR artifacts (e.g. "print" or a
// directory path).
inline constexpr char kTorchTpuInternalDumpMlirPassesToEnvVar[] =
    "TORCH_TPU_INTERNAL_DUMP_MLIR_PASSES_TO";
// If set to "0" or "false", disables reassociating normalization weights.
// Default is "true".
inline constexpr char kTorchTpuInternalEnableReassociateNormWeightsEnvVar[] =
    "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS";
// How long, in seconds, to wait for the distributed handshake to complete. If
// not set, 60 is used.
inline constexpr char kTorchTpuInternalHandshakeTimeoutSEnvVar[] =
    "TORCH_TPU_INTERNAL_HANDSHAKE_TIMEOUT_S";
// If set to "0" or "false", disables forcing graph breaks for collective ops.
// This is useful for SPMD workloads where graph differences between ranks are
// not expected. Default is "true".
inline constexpr char kTorchTpuInternalMaterializeCollectiveTensorsEnvVar[] =
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS";
// If set to "1", all traversals which contain exactly one distributed operation
// (a "skinny collective") will be automatically pinned to the SparseCore.
// This will not override manually specified stream pinning.
// Default is "0", meaning that skinny collectives are not automatically pinned.
inline constexpr char kTorchTpuInternalSkinnyScOffloadEnvVar[] =
    "TORCH_TPU_INTERNAL_SKINNY_SC_OFFLOAD";
// If set to "1", forces the RNG state update to be isolated into a separate
// compilation unit.
// This may degrade performance, but improves error recovery due to failed
// compilation or execution, as it keeps the RNG generator in a "clean" state
// even in the presence of errors.
inline constexpr char kTorchTpuInternalSplitRngStateUpdate[] =
    "TORCH_TPU_INTERNAL_SPLIT_RNG_STATE_UPDATE";
// If set to "1" or "true", StaticCompiler creates one placeholder per shared
// input storage and rebuilds aliasing inputs as views of it, instead of one
// independent placeholder per input. Default is "false".
// TODO(dighamber): Temporary rollout gate; remove once storage-aware
// placeholders are always enabled.
inline constexpr char kTorchTpuInternalStorageAwarePlaceholdersEnvVar[] =
    "TORCH_TPU_INTERNAL_STORAGE_AWARE_PLACEHOLDERS";
// The PID of the earliest TorchTPU process in the current process' lineage (see
// csrc/common/process.h). Set by a TorchTPU process that has no TorchTPU
// ancestor process and inherited by its descendant processes so they can
// detect that they have a TorchTPU ancestor process.
inline constexpr char kTorchTpuInternalTopmostPidEnvVar[] =
    "TORCH_TPU_INTERNAL_TOPMOST_PID";
// If set to `"1"` or `"true"`, unbounded kernels are loaded instead of static
// ones if a compatible one exists.
inline constexpr char kTorchTpuInternalUseUnboundedDynamicKernelsEnvVar[] =
    "TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS";
// TorchTPU-internal XLA compiler option overrides, in the format of
// "key1=value1 key2=value2 ...". It is used to update `xla::CompileOptions`
// through `env_option_overrides`, and takes precedence over `debug_options` set
// via XLA_FLAGS.
inline constexpr char kTorchTpuInternalXlaOptionsEnvVar[] =
    "TORCH_TPU_INTERNAL_XLA_OPTIONS";
// Similar to CUDA_LAUNCH_BLOCKING, used to enable
// EagerMode::kDeferNeverAndLaunchBlocking by default.
inline constexpr char kTorchTpuLaunchBlockingEnvVar[] =
    "TORCH_TPU_LAUNCH_BLOCKING";
// The output directory for TPU profiler XPlane files.
inline constexpr char kTorchTpuProfilerOutputDirEnvVar[] =
    "TORCH_TPU_PROFILER_OUTPUT_DIR";
inline constexpr char kTorchTpuSlicebuilderAddressesEnvVar[] =
    "TORCH_TPU_SLICEBUILDER_ADDRESSES";  // Set by Torch TPU specific launchers.
// Specifies the stage at which ranks perform a handshake to check that they
// generated identical graphs containing collectives, so that such graphs can be
// compiled without splitting at the collectives.
// Supported modes: OFF, COMPILE_STAGE, DISPATCH_STAGE.
// Default is OFF.
inline constexpr char kTorchTpuSpmdHandshakeStageEnvVar[] =
    "TORCH_TPU_SPMD_HANDSHAKE_STAGE";
// The name of the tier-2 compilation cache. The special name "disabled" can be
// used to disable the tier-2 cache. If not set, TorchTPU decides whether to use
// the tier-2 cache or not based on the world size: if the world size is 1, the
// tier-2 cache is disabled; otherwise, it is enabled and the name is set to
// "default".
inline constexpr char kTorchTpuTier2CompilationCacheEnvVar[] =
    "TORCH_TPU_TIER2_COMPILATION_CACHE";
// The root path of the tier-3 compilation cache. If not set, the tier-3
// compilation cache is disabled.
inline constexpr char kTorchTpuTier3CompilationCacheRootEnvVar[] =
    "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT";
inline constexpr char kTorchTpuTopologyEnvVar[] =
    "TORCH_TPU_TOPOLOGY";  // Set by Torch TPU specific launchers.
// Identifies the XProf profiling session shared by all workers of a job. Set
// by Torch TPU specific launchers when it is not already set.
inline constexpr char kTorchTpuXprofSessionIdEnvVar[] =
    "TORCH_TPU_XPROF_SESSION_ID";
// If set, enables structured logging for tlparse.
inline constexpr char kTorchTraceEnvVar[] = "TORCH_TRACE";
// The rank of this process as seen by torchcomm. Takes precedence over RANK.
inline constexpr char kTorchcommRankEnvVar[] = "TORCHCOMM_RANK";
// The number of processes as seen by torchcomm. Takes precedence over
// WORLD_SIZE.
inline constexpr char kTorchcommSizeEnvVar[] = "TORCHCOMM_SIZE";
inline constexpr char kTpuChipsPerHostBoundsEnvVar[] =
    "TPU_CHIPS_PER_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuChipsPerProcessBoundsEnvVar[] =
    "TPU_CHIPS_PER_PROCESS_BOUNDS";
inline constexpr char kTpuHostBoundsEnvVar[] =
    "TPU_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuPremappedBufferSizeEnvVar[] =
    "TPU_PREMAPPED_BUFFER_SIZE";  // Premapped buffer size in bytes for PjRt
                                  // client.
inline constexpr char kTpuProcessAddressesEnvVar[] =
    "TPU_PROCESS_ADDRESSES";  // Set by Google Cloud.
inline constexpr char kTpuProcessBoundsEnvVar[] = "TPU_PROCESS_BOUNDS";
inline constexpr char kTpuProcessPortEnvVar[] =
    "TPU_PROCESS_PORT";  // Read by Google Cloud.
// Specifies which TPU chips are visible to this process. Read by libtpu.
// Note: TPU_VISIBLE_DEVICES is the source of truth for device visibility in
// TorchTPU. TPU_VISIBLE_CHIPS is overwritten to match TPU_VISIBLE_DEVICES
// because libtpu prioritizes non-empty TPU_VISIBLE_CHIPS over
// TPU_VISIBLE_DEVICES.
inline constexpr char kTpuVisibleChipsEnvVar[] = "TPU_VISIBLE_CHIPS";
// Comma-separated device paths of the TPUs that are visible to this process.
// Read by libtpu.
inline constexpr char kTpuVisibleDevicePathsEnvVar[] =
    "TPU_VISIBLE_DEVICE_PATHS";
// Specifies which TPU devices are visible to this process. Read by libtpu.
inline constexpr char kTpuVisibleDevicesEnvVar[] = "TPU_VISIBLE_DEVICES";
// The hostnames of every worker in a multi-host job, comma separated.
inline constexpr char kTpuWorkerHostnamesEnvVar[] =
    "TPU_WORKER_HOSTNAMES";  // Set by Google Cloud.
// How many devices are in this process group. Set by launchers like
// torchrun.
inline constexpr char kWorldSizeEnvVar[] = "WORLD_SIZE";
// Standard environment variable to configure XLA compiler behavior.
inline constexpr char kXlaFlagsEnvVar[] = "XLA_FLAGS";
// go/keep-sorted end

// Maps an env var to its symbol stage. nullopt means the stage is unknown.
inline constexpr auto kEnvVarToStage =
    MakeConstexprMap<std::string_view, std::optional<SymbolStage>>({
        // go/keep-sorted start
        {kAcceleratorTypeEnvVar, SymbolStage::InternalImplementation()},
        {kAllowMultipleLibtpuLoadEnvVar, SymbolStage::InternalImplementation()},
        {kCloudTpuTaskIdEnvVar, SymbolStage::InternalImplementation()},
        {kJobCompletionIndexEnvVar, SymbolStage::InternalApi()},
        // Marked as Stable because it is an established XLA/PJRT runtime
        // configuration knob widely used by production workloads. The TorchTPU
        // team is committed to honor this env var.
        {kLibtpuInitArgsEnvVar, SymbolStage::Stable()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kLocalRankEnvVar, SymbolStage::Stable()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kMasterAddrEnvVar, SymbolStage::Stable()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kMasterPortEnvVar, SymbolStage::Stable()},
        {kMegascaleDebugPortEnvVar, SymbolStage::InternalImplementation()},
        {kMegascalePortEnvVar, SymbolStage::InternalImplementation()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kNnodesEnvVar, SymbolStage::Stable()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kNodeRankEnvVar, SymbolStage::Stable()},
        // Marked as Internal because it is an internal build/test runner
        // environment variable used to adjust compilation thread concurrency,
        // controlling TorchTPU internal behavior rather than serving as a
        // public API contract.
        {kNprocEnvVar, SymbolStage::InternalApi()},
        // Marked as Stable because it is an established distributed launcher
        // environment variable widely used by production workloads. The
        // TorchTPU team is committed to honor this env var.
        {kRankEnvVar, SymbolStage::Stable()},
        {kSkipMegascalePjrtClientEnvVar, SymbolStage::InternalImplementation()},
        {kTmpdirEnvVar, SymbolStage::Stable()},
        {kTorchDeviceBackendAutoloadEnvVar, SymbolStage::Stable()},
        {kTorchShowCppStacktracesEnvVar, SymbolStage::Experimental()},
        {kTorchTpuAllowLegacyDeviceApisEnvVar, SymbolStage::Experimental()},
        {kTorchTpuBasePortEnvVar, SymbolStage::Experimental()},
        {kTorchTpuDeferAndFuseEnvVar, SymbolStage::Experimental()},
        {kTorchTpuDevicesPerHostEnvVar, SymbolStage::Experimental()},
        {kTorchTpuEnableDebugChecksEnvVar, SymbolStage::Experimental()},
        {kTorchTpuHandshakePortEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalDetectRepeatedOpsEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalDisableInplaceBufferDonationEnvVar,
         SymbolStage::InternalApi()},
        {kTorchTpuInternalDumpMlirPassesToEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalEnableReassociateNormWeightsEnvVar,
         SymbolStage::InternalApi()},
        {kTorchTpuInternalHandshakeTimeoutSEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalMaterializeCollectiveTensorsEnvVar,
         SymbolStage::InternalApi()},
        {kTorchTpuInternalSkinnyScOffloadEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalSplitRngStateUpdate, SymbolStage::InternalApi()},
        {kTorchTpuInternalStorageAwarePlaceholdersEnvVar,
         SymbolStage::InternalApi()},
        {kTorchTpuInternalTopmostPidEnvVar,
         SymbolStage::InternalImplementation()},
        {kTorchTpuInternalUseUnboundedDynamicKernelsEnvVar,
         SymbolStage::InternalApi()},
        {kTorchTpuInternalXlaOptionsEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuLaunchBlockingEnvVar, SymbolStage::Experimental()},
        {kTorchTpuProfilerOutputDirEnvVar, SymbolStage::Experimental()},
        {kTorchTpuSlicebuilderAddressesEnvVar, SymbolStage::Experimental()},
        {kTorchTpuSpmdHandshakeStageEnvVar, SymbolStage::Experimental()},
        {kTorchTpuTier2CompilationCacheEnvVar, SymbolStage::Experimental()},
        {kTorchTpuTier3CompilationCacheRootEnvVar, SymbolStage::Experimental()},
        {kTorchTpuTopologyEnvVar, SymbolStage::Experimental()},
        {kTorchTpuXprofSessionIdEnvVar, SymbolStage::Experimental()},
        {kTorchTraceEnvVar, SymbolStage::Experimental()},
        {kTorchcommRankEnvVar, SymbolStage::Experimental()},
        {kTorchcommSizeEnvVar, SymbolStage::Experimental()},
        {kTpuChipsPerHostBoundsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuChipsPerProcessBoundsEnvVar,
         SymbolStage::InternalImplementation()},
        {kTpuHostBoundsEnvVar, SymbolStage::InternalImplementation()},
        // Marked as Stable because it is an established XLA/PJRT runtime
        // configuration knob widely used by production workloads. The TorchTPU
        // team is committed to honor this env var.
        {kTpuPremappedBufferSizeEnvVar, SymbolStage::Stable()},
        {kTpuProcessAddressesEnvVar, SymbolStage::InternalImplementation()},
        {kTpuProcessBoundsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuProcessPortEnvVar, SymbolStage::InternalImplementation()},
        {kTpuVisibleChipsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuVisibleDevicePathsEnvVar, SymbolStage::InternalImplementation()},
        // Marked as Stable because it is the standard TPU runtime equivalent to
        // CUDA_VISIBLE_DEVICES that external production consumers and infra
        // rely on. The TorchTPU team is committed to honor this env var.
        {kTpuVisibleDevicesEnvVar, SymbolStage::Stable()},
        {kTpuWorkerHostnamesEnvVar, SymbolStage::InternalApi()},
        {kWorldSizeEnvVar, SymbolStage::Stable()},
        {kXlaFlagsEnvVar, SymbolStage::Experimental()},
        // go/keep-sorted end
    });

// Sets the environment variable with the given name to the given value.
template <const char* name>
void SetEnv(const std::string& value) {
  static_assert(kEnvVarToStage.contains(name),
                "Unknown environment variable. All env vars used by TorchTPU "
                "must be registered in kEnvVarToStage.");
  setenv(name, value.c_str(), /*overwrite=*/1);
}

// Returns the value of the environment variable with the given name, or
// std::nullopt if it is not set.
//
// This function is memoized, so the environment variable is only read once.
//
// We make the name a template parameter so that different env vars are
// memoized separately and the TT_WARN_ONCE() call works for each env var
// individually. The name must be one of the kFooEnvVar variables defined in
// this file.
template <const char* name>
[[nodiscard]] const std::optional<std::string>& GetEnvOnce() {
  static_assert(kEnvVarToStage.contains(name),
                "Unknown environment variable. All env vars used by TorchTPU "
                "must be registered in kEnvVarToStage.");
  constexpr auto maybe_stage = kEnvVarToStage[name];
  if constexpr (maybe_stage.has_value()) {
    static_assert(
        !maybe_stage->is_internal_implementation(),
        "Cannot read from an internal implementation env var. "
        "TorchTPU can only write to them. If TorchTPU really needs to "
        "read this env var, its stage in kEnvVarToStage must be changed.");
  }
  static const absl::NoDestructor<std::optional<std::string>> env_var(
      [maybe_stage]() -> std::optional<std::string> {
        const char* const env_var =  //
            std::getenv(name);       // GETENV_OK=implementing GetEnvOnce().
        if (env_var == nullptr) return std::nullopt;
        if (env_var[0] == '\0') return std::string();

        // The env var is set to a non-empty string. Warn the user if the env
        // var is experimental or deprecated.
        if (maybe_stage.has_value()) {
          if (maybe_stage->is_experimental()) {
            TT_WARN_ONCE(WarningCategory::kExperimentalApi, "the ", name,
                         " environment variable is an experimental feature "
                         "and may change or be removed without notice.");
          } else if (maybe_stage->is_deprecated()) {
            const auto deprecated_since = maybe_stage->version();
            TT_WARN_ONCE(WarningCategory::kDeprecatedApi, "the ", name,
                         " environment variable is deprecated since TorchTPU v",
                         deprecated_since,
                         " and will be removed in a future release.");
          }
        }
        return std::string(env_var);
      }());
  return *env_var;
}

// Returns the boolean value of the environment variable with the given name, or
// std::nullopt if it is not set or cannot be parsed as a Boolean.
//
// This function delegates to GetEnvOnce() to read the environment variable, and
// then parses the string into a Boolean using absl::SimpleAtob.
template <const char* name>
[[nodiscard]] std::optional<bool> GetBooleanEnvOnce() {
  // Compute the value once and memoize it.
  static const std::optional<bool> value = []() -> std::optional<bool> {
    const auto& env_var = GetEnvOnce<name>();
    if (!env_var.has_value()) {
      return std::nullopt;
    }
    bool result = false;
    if (absl::SimpleAtob(*env_var, &result)) {
      return result;
    }
    TT_WARN_ONCE(WarningCategory::kCorrectness,
                 "expected environment variable ", name,
                 " to be a Boolean (e.g., '1', '0', 'true', 'false'), got '",
                 *env_var, "'");
    return std::nullopt;
  }();
  return value;
}

// Returns the integer value of the environment variable with the given name, or
// std::nullopt if it is not set or cannot be parsed as an integer.
//
// This function delegates to GetEnvOnce() to read the environment variable, and
// then parses the string into an integer using absl::SimpleAtoi.
template <typename IntType, const char* name>
[[nodiscard]] std::optional<IntType> GetIntegerEnvOnce() {
  static_assert(std::is_integral_v<IntType> && !std::is_same_v<IntType, bool>,
                "IntType must be a non-boolean integral type.");
  // Compute the value once and memoize it.
  static const std::optional<IntType> value = []() -> std::optional<IntType> {
    const auto& env_var = GetEnvOnce<name>();
    if (!env_var.has_value()) {
      return std::nullopt;
    }
    IntType result = 0;
    if (absl::SimpleAtoi(*env_var, &result)) {
      return result;
    }
    TT_WARN_ONCE(WarningCategory::kCorrectness,
                 "expected environment variable ", name,
                 " to be an integer, got '", *env_var, "'");
    return std::nullopt;
  }();
  return value;
}

// Ensures that the rank is expanded in environment variables that depend on it.
// It may overwrite the env vars when called for the first time, and has no
// effect on subsequent calls.
void EnsureRankIsExpandedInEnvVars();

// Returns the value of the XLA_FLAGS env var, with the "${RANK}" placeholder
// expanded to the current rank.
//
// Returns std::nullopt if the env var is not set.
[[nodiscard]] const std::optional<std::string>&
ExpandAndGetXlaFlagsEnvVarOnce();

// Returns the value of the LIBTPU_INIT_ARGS env var, with the "${RANK}"
// placeholder expanded to the current rank.
//
// Returns std::nullopt if the env var is not set.
[[nodiscard]] const std::optional<std::string>&
ExpandAndGetLibtpuInitArgsEnvVarOnce();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_ENV_VARS_H_
