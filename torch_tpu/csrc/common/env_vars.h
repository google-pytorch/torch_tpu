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

#ifndef TORCH_TPU_COMMON_ENV_VARS_H_
#define TORCH_TPU_COMMON_ENV_VARS_H_

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#include "absl/base/no_destructor.h"
#include "c10/util/Exception.h"
#include "torch_tpu/csrc/common/constexpr_map.h"
#include "torch_tpu/csrc/common/symbol_stage.h"

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
inline constexpr char kLibtpuInitArgsEnvVar[] =
    "LIBTPU_INIT_ARGS";  // Set by Google Cloud.
inline constexpr char kLocalRankEnvVar[] = "LOCAL_RANK";    // Set by launchers.
inline constexpr char kMasterAddrEnvVar[] = "MASTER_ADDR";  // Set by launchers.
inline constexpr char kMasterPortEnvVar[] = "MASTER_PORT";  // Set by launchers.
// Set by the build system (e.g. bazel) to indicate the number of shards
// running on the same host.
inline constexpr char kNprocEnvVar[] = "NPROC";
inline constexpr char kRankEnvVar[] = "RANK";  // Set by launchers.
// Standard temporary directory.
inline constexpr char kTmpdirEnvVar[] = "TMPDIR";
// If set to "1", C++ stack traces are appended to error messages.
// Shared with PyTorch:
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html#pytorch-debug-environment-variables
inline constexpr char kTorchShowCppStacktracesEnvVar[] =
    "TORCH_SHOW_CPP_STACKTRACES";
// Specifies the fixed port number for the handshake server.
inline constexpr char kTorchTpuHandshakePortEnvVar[] =
    "TORCH_TPU_INTERNAL_HANDSHAKE_PORT";
// Specifies the repeated op detection mode. If not set, "safe" mode is used.
// Supported modes:
// - "safe": Uses MaterializationMode::kSplitGraph when a repeated sequence is
// detected.
// - "aggressive": Uses MaterializationMode::kFullGraph when a repeated sequence
// is detected.
// Other values are no-ops.
inline constexpr char kTorchTpuInternalDetectRepeatedOpsEnvVar[] =
    "TORCH_TPU_INTERNAL_DETECT_REPEATED_OPS";
// If set to "1", enable expensive debug checks in TorchTPU. This catches
// more bugs in user code, but comes at a significant performance cost for
// some ops. The debug eager mode enables these checks by default, but can
// be overridden by setting this env var to "0".
inline constexpr char kTorchTpuInternalEnableDebugChecksEnvVar[] =
    "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS";
// Specifies the internal handshake stage mode.
// Supported modes: OFF, COMPILE_STAGE, DISPATCH_STAGE.
// Default is OFF.
inline constexpr char kTorchTpuInternalHandshakeStageEnvVar[] =
    "TORCH_TPU_INTERNAL_HANDSHAKE_STAGE";
// If set to "0" or "false", disables forcing graph breaks for collective ops.
// This is useful for SPMD workloads where graph differences between ranks are
// not expected. Default is "true".
inline constexpr char kTorchTpuInternalMaterializeCollectiveTensorsEnvVar[] =
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS";
// If set to "1", forces the RNG state update to be isolated into a separate
// compilation unit.
// This may degrade performance, but improves error recovery due to failed
// compilation or execution, as it keeps the RNG generator in a "clean" state
// even in the presence of errors.
inline constexpr char kTorchTpuInternalSplitRngStateUpdate[] =
    "TORCH_TPU_INTERNAL_SPLIT_RNG_STATE_UPDATE";
// If unset or set to "1", schedule local compilation right away as a backup to
// tier-3 compilation cache read (whichever succeeds first will unblock
// execution). If set to "0", local compilation is done only after tier-3
// compilation cache read fails.
//
// This env var is only used when the tier-3 compilation cache is enabled.
inline constexpr char
    kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar[] =
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK";
// TorchTPU-internal XLA compiler option overrides, in the format of
// "key1=value1 key2=value2 ...". It is used to update `xla::CompileOptions`
// through `env_option_overrides`, and takes precedence over `debug_options` set
// via XLA_FLAGS.
inline constexpr char kTorchTpuInternalXlaOptionsEnvVar[] =
    "TORCH_TPU_INTERNAL_XLA_OPTIONS";
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
// If set, enables structured logging for tlparse.
inline constexpr char kTorchTraceEnvVar[] = "TORCH_TRACE";
inline constexpr char kTpuChipsPerHostBoundsEnvVar[] =
    "TPU_CHIPS_PER_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuChipsPerProcessBoundsEnvVar[] =
    "TPU_CHIPS_PER_PROCESS_BOUNDS";
inline constexpr char kTpuDeferAndFuse[] =
    "TPU_DEFER_AND_FUSE";  // Used to enable EagerMode::kDeferAndFuse by
                           // default.
inline constexpr char kTpuHostBoundsEnvVar[] =
    "TPU_HOST_BOUNDS";  // Read by Google Cloud.
inline constexpr char kTpuLaunchBlocking[] =
    "TPU_LAUNCH_BLOCKING";  // Similar to CUDA_LAUNCH_BLOCKING, used enable
                            // EagerMode::kDeferNeverAndLaunchBlocking by
                            // default.
inline constexpr char kTpuPremappedBufferSizeEnvVar[] =
    "TPU_PREMAPPED_BUFFER_SIZE";  // Premapped buffer size in bytes for PjRt
                                  // client.
inline constexpr char kTpuProcessAddressesEnvVar[] =
    "TPU_PROCESS_ADDRESSES";  // Set by Google Cloud.
inline constexpr char kTpuProcessBoundsEnvVar[] = "TPU_PROCESS_BOUNDS";
inline constexpr char kTpuProcessPortEnvVar[] =
    "TPU_PROCESS_PORT";  // Read by Google Cloud.
// The output directory for TPU profiler XPlane files.
inline constexpr char kTpuProfilerOutputDirEnvVar[] = "TPU_PROFILER_OUTPUT_DIR";
inline constexpr char kTpuSlicebuilderAddressesEnvVar[] =
    "TORCH_TPU_SLICEBUILDER_ADDRESSES";  // Set by Torch TPU specific launchers.
inline constexpr char kTpuTopologyEnvVar[] =
    "TORCH_TPU_TOPOLOGY";  // Set by Torch TPU specific launchers.
// Specifies which TPU chips are visible to this process. Read by libtpu.
// Note: TPU_VISIBLE_DEVICES is the source of truth for device visibility in
// TorchTPU. TPU_VISIBLE_CHIPS is overwritten to match TPU_VISIBLE_DEVICES
// because libtpu prioritizes non-empty TPU_VISIBLE_CHIPS over
// TPU_VISIBLE_DEVICES.
inline constexpr char kTpuVisibleChipsEnvVar[] = "TPU_VISIBLE_CHIPS";
// Specifies which TPU devices are visible to this process. Read by libtpu.
inline constexpr char kTpuVisibleDevicesEnvVar[] = "TPU_VISIBLE_DEVICES";
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
        {kLibtpuInitArgsEnvVar, std::nullopt},
        {kLocalRankEnvVar, std::nullopt},
        {kMasterAddrEnvVar, std::nullopt},
        {kMasterPortEnvVar, std::nullopt},
        {kNprocEnvVar, std::nullopt},
        {kRankEnvVar, std::nullopt},
        {kTmpdirEnvVar, std::nullopt},
        {kTorchShowCppStacktracesEnvVar, std::nullopt},
        {kTorchTpuHandshakePortEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalDetectRepeatedOpsEnvVar, std::nullopt},
        {kTorchTpuInternalEnableDebugChecksEnvVar, std::nullopt},
        {kTorchTpuInternalHandshakeStageEnvVar, SymbolStage::InternalApi()},
        {kTorchTpuInternalMaterializeCollectiveTensorsEnvVar, std::nullopt},
        {kTorchTpuInternalSplitRngStateUpdate, std::nullopt},
        {kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar,
         std::nullopt},
        {kTorchTpuInternalXlaOptionsEnvVar, std::nullopt},
        {kTorchTpuTier2CompilationCacheEnvVar, SymbolStage::Experimental()},
        {kTorchTpuTier3CompilationCacheRootEnvVar, SymbolStage::Experimental()},
        {kTorchTraceEnvVar, std::nullopt},
        {kTpuChipsPerHostBoundsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuChipsPerProcessBoundsEnvVar,
         SymbolStage::InternalImplementation()},
        {kTpuDeferAndFuse, std::nullopt},
        {kTpuHostBoundsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuLaunchBlocking, std::nullopt},
        {kTpuPremappedBufferSizeEnvVar, std::nullopt},
        {kTpuProcessAddressesEnvVar, SymbolStage::InternalImplementation()},
        {kTpuProcessBoundsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuProcessPortEnvVar, SymbolStage::InternalImplementation()},
        {kTpuProfilerOutputDirEnvVar, std::nullopt},
        {kTpuSlicebuilderAddressesEnvVar, std::nullopt},
        {kTpuTopologyEnvVar, std::nullopt},
        {kTpuVisibleChipsEnvVar, SymbolStage::InternalImplementation()},
        {kTpuVisibleDevicesEnvVar, std::nullopt},
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
// memoized separately and the TORCH_WARN_ONCE() call works for each env var
// individually. The name must be one of the kFooEnvVar variables defined in
// this file.
template <const char* name>
const std::optional<std::string>& GetEnvOnce() {
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
            TORCH_WARN_ONCE("the ", name,
                            " environment variable is an experimental feature "
                            "and may change or be removed without notice.");
          } else if (maybe_stage->is_deprecated()) {
            const auto deprecated_since = maybe_stage->version();
            TORCH_WARN_ONCE(
                "the ", name,
                " environment variable is deprecated since TorchTPU v",
                deprecated_since, " and will be removed in a future release.");
          }
        }
        return std::string(env_var);
      }());
  return *env_var;
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_ENV_VARS_H_
