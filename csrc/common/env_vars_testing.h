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

// Test-only utilities for env_vars.h.

#ifndef TORCH_TPU_CSRC_COMMON_ENV_VARS_TESTING_H_
#define TORCH_TPU_CSRC_COMMON_ENV_VARS_TESTING_H_

namespace torch_tpu {

// Clears the values memoized by the GetEnvOnce() family in env_vars.h, which
// the Python-facing readers in `torch_tpu._internal.env` share, so that their
// next calls read the current environment and warn again about experimental,
// deprecated or invalid values. Also lets SafeSetEnv() and SafeUnsetEnv()
// change each env var until it is read again.
//
// References returned by GetEnvOnce() before the reset observe the new value
// once the env var is read again.
//
// REQUIRES: No env var is read concurrently.
//
// Defined in env_vars.cc, next to the state it resets.
void ResetEnvCacheForTesting();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_ENV_VARS_TESTING_H_
