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

#include "absl/base/no_destructor.h"

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
// Set by the Bazel build system during unit test execution.
inline constexpr char kTestTargetEnvVar[] = "TEST_TARGET";
// Set by Bazel test runner.
inline constexpr char kTestTmpdirEnvVar[] = "TEST_TMPDIR";
inline constexpr char kTestWorkspaceEnvVar[] = "TEST_WORKSPACE";
// Standard temporary directory.
inline constexpr char kTmpdirEnvVar[] = "TMPDIR";
// If set to "1", C++ stack traces are appended to error messages.
// Shared with PyTorch:
// https://docs.pytorch.org/docs/stable/debugging_environment_variables.html#pytorch-debug-environment-variables
inline constexpr char kTorchShowCppStacktracesEnvVar[] =
    "TORCH_SHOW_CPP_STACKTRACES";
// If set to "1", enable expensive debug checks in TorchTPU. This catches
// more bugs in user code, but comes at a significant performance cost for
// some ops. The debug eager mode enables these checks by default, but can
// be overridden by setting this env var to "0".
inline constexpr char kTorchTpuInternalEnableDebugChecksEnvVar[] =
    "TORCH_TPU_INTERNAL_ENABLE_DEBUG_CHECKS";
// The name of the tier-2 compilation cache. The special name "disabled" can be
// used to disable the tier-2 cache. If not set, TorchTPU decides whether to use
// the tier-2 cache or not based on the world size: if the world size is 1, the
// tier-2 cache is disabled; otherwise, it is enabled and the name is set to
// "default".
inline constexpr char kTorchTpuInternalTier2CompilationCacheEnvVar[] =
    "TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE";
// If unset or set to "1", schedule local compilation right away as a backup to
// tier-3 compilation cache read (whichever succeeds first will unblock
// execution). If set to "0", local compilation is done only after tier-3
// compilation cache read fails.
//
// This env var is only used when the tier-3 compilation cache is enabled.
inline constexpr char
    kTorchTpuInternalTier3CompilationCacheLocalBackupTaskEnvVar[] =
        "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK";
// The root path of the tier-3 compilation cache. If not set, the tier-3
// compilation cache is disabled.
inline constexpr char kTorchTpuInternalTier3CompilationCacheRootEnvVar[] =
    "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT";
// XLA compiler option overrides, in the format of "key1=value1 key2=value2
// ...".
inline constexpr char kTorchTpuInternalXlaOptionsEnvVar[] =
    "TORCH_TPU_INTERNAL_XLA_OPTIONS";
inline constexpr char kTpuChipsPerHostBoundsEnvVar[] =
    "TPU_CHIPS_PER_HOST_BOUNDS";  // Read by Google Cloud.
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
    "TPU_PREMAPPED_BUFFER_SIZE";  // Read when initializing the pjrt runtime.
inline constexpr char kTpuProcessAddressesEnvVar[] =
    "TPU_PROCESS_ADDRESSES";  // Set by Google Cloud.
inline constexpr char kTpuProcessPortEnvVar[] =
    "TPU_PROCESS_PORT";  // Read by Google Cloud.
// The output directory for TPU profiler XPlane files.
inline constexpr char kTpuProfilerOutputDirEnvVar[] = "TPU_PROFILER_OUTPUT_DIR";
inline constexpr char kTpuSlicebuilderAddressesEnvVar[] =
    "TORCH_TPU_SLICEBUILDER_ADDRESSES";  // Set by Torch TPU specific launchers.
inline constexpr char kTpuTopologyEnvVar[] =
    "TORCH_TPU_TOPOLOGY";  // Set by Torch TPU specific launchers.
inline constexpr char kTpuVisibleChipsEnvVar[] =
    "TPU_VISIBLE_CHIPS";  // Read by Google Cloud.
// How many devices are in this process group. Set by launchers like
// torchrun.
inline constexpr char kWorldSizeEnvVar[] = "WORLD_SIZE";
// go/keep-sorted end

// Sets the environment variable with the given name to the given value.
inline void SetEnv(const char* name, const std::string& value) {
  setenv(name, value.c_str(), /*overwrite=*/1);
}

// Returns the value of the environment variable with the given name, or
// std::nullopt if it is not set.
//
// This function is memoized, so the environment variable is only read once.
//
// We make the name a template parameter so that different env vars are
// memoized separately.
template <const char* name>
const std::optional<std::string>& GetEnvOnce() {
  static const absl::NoDestructor<std::optional<std::string>> env_var(
      []() -> std::optional<std::string> {
        const char* const env_var =  //
            std::getenv(name);       // GETENV_OK=implementing GetEnvOnce().
        if (env_var == nullptr) return std::nullopt;
        return std::string(env_var);
      }());
  return *env_var;
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_ENV_VARS_H_
