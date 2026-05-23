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

#ifndef TORCH_TPU_OPS_RNG_UTILS_H_
#define TORCH_TPU_OPS_RNG_UTILS_H_

#include <mutex>
#include <utility>
#include <vector>

#include "ATen/core/Generator.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/device_gen_impl.h"
#include "torch_tpu/eager/tensor_to_buffer.h"

namespace torch_tpu {

// Dispatches an RNG operation, handling the common pattern of acquiring the
// generator, locking it, getting the state, invoking the operation, and
// updating the state. Returns all output buffers except the new RNG state.
//
// Parameters:
//   generator: Optional generator to use. If not provided, the default
//     device generator is used.
//   dispatch_func: A callable that takes an `at::Tensor` (the current RNG
//     state) and returns an `absl::StatusOr` of a container of DeviceBufferRef
//     (e.g., std::array or std::vector), where the first element is the new
//     RNG state.
template <typename DispatchFunc>
absl::StatusOr<std::vector<DeviceBufferRef>> DispatchRngOpGeneral(
    c10::optional<at::Generator> generator, DispatchFunc dispatch_func) {
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());

  // NOLINTNEXTLINE(build/c++11) - std::mutex required by PyTorch
  std::scoped_lock<std::mutex> lock(gen->mutex_);
  at::Tensor rng_input_state = gen->DeviceStateTensor();

  TT_ASSIGN_OR_RETURN(auto results_array,
                      std::move(dispatch_func)(rng_input_state));

  std::vector<DeviceBufferRef> results(
      std::make_move_iterator(results_array.begin()),
      std::make_move_iterator(results_array.end()));

  ABSL_CHECK(!results.empty())  // CRASH_OK
      << "RNG op must produce at least one output (the new state).";

  auto rng_output_state = MakeTensor(std::move(results[0]));
  TT_RETURN_IF_ERROR(gen->SetDeviceStateTensor(rng_output_state));

  results.erase(results.begin());
  return results;
}

// Like `DispatchRngOpGeneral` but expects the dispatch function to return
// exactly one output buffer (in addition to the new RNG state). Returns
// that single output buffer.
template <typename DispatchFunc>
absl::StatusOr<DeviceBufferRef> DispatchRngOpAndReturnBuffer(
    c10::optional<at::Generator> generator, DispatchFunc dispatch_func) {
  TT_ASSIGN_OR_RETURN(
      auto results, DispatchRngOpGeneral(generator, std::move(dispatch_func)));
  ABSL_CHECK_EQ(results.size(), 1)  // CRASH_OK
      << "Expected 1 output buffer, got " << results.size();
  return std::move(results[0]);
}

// Like `DispatchRngOpAndReturnBuffer` but assigns the output buffer to
// `result_tensor`.
template <typename DispatchFunc>
absl::Status DispatchRngOp(at::Tensor& result_tensor,
                           c10::optional<at::Generator> generator,
                           DispatchFunc dispatch_func) {
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef output_buf,
      DispatchRngOpAndReturnBuffer(generator, std::move(dispatch_func)));
  return AssignBufferToAtTensor(std::move(output_buf), result_tensor);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_RNG_UTILS_H_
