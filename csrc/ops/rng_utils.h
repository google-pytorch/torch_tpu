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

#ifndef TORCH_TPU_CSRC_OPS_RNG_UTILS_H_
#define TORCH_TPU_CSRC_OPS_RNG_UTILS_H_

#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "ATen/core/Generator.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/util/Optional.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_gen_impl.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

// Represents the number of elements and bit width consumed by an RNG operation.
struct RngUsage {
  int64_t num_elements = -1;
  int64_t bit_width = -1;
};

// Dispatches an RNG operation, handling locking, state retrieval, and
// advancement. `dispatch_func` must use the provided state tensor and return
// the output buffers. If `usage` is not provided, the RNG advancement
// parameters (num_elements and bit_width) are inferred from the first output
// buffer. Otherwise, the explicit `usage` values are used (required when the
// output type/shape does not match the actual RNG consumption, e.g., dropout).
template <typename DispatchFunc>
absl::StatusOr<std::vector<DeviceBufferRef>> DispatchRngOpGeneral(
    c10::optional<at::Generator> generator, DispatchFunc dispatch_func,
    const std::optional<RngUsage>& usage = std::nullopt) {
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());

  // NOLINTNEXTLINE(build/c++11) - std::mutex required by PyTorch
  std::scoped_lock<std::mutex> lock(gen->mutex_);
  TT_ASSIGN_OR_RETURN(auto results_array,
                      std::move(dispatch_func)(gen->DeviceStateTensor()));

  int64_t num_elements = -1;
  int64_t bit_width = -1;

  if (usage.has_value()) {
    num_elements = usage->num_elements;
    bit_width = usage->bit_width;
  } else {
    ABSL_CHECK(!results_array.empty())  // CRASH_OK
        << "Expected at least one output buffer to infer RNG parameters";
    const auto& first_result = results_array[0];
    num_elements = first_result.num_elements();
    bit_width = TorchEquivalentBitwidth(first_result.element_type());
  }

  // Advance the generator state when dispatch_func returns successfully.
  TT_RETURN_IF_ERROR(gen->AdvanceDeviceStateTensor(num_elements, bit_width));

  return std::vector<DeviceBufferRef>(
      std::make_move_iterator(results_array.begin()),
      std::make_move_iterator(results_array.end()));
}

// Helper that wraps `DispatchRngOpGeneral` and returns the single output buffer
// produced by `dispatch_func`. Supports optional explicit RNG advancement
// parameters; if not specified, they are inferred from the returned buffer.
template <typename DispatchFunc>
absl::StatusOr<DeviceBufferRef> DispatchRngOpAndReturnBuffer(
    c10::optional<at::Generator> generator, DispatchFunc dispatch_func,
    const std::optional<RngUsage>& usage = std::nullopt) {
  TT_ASSIGN_OR_RETURN(
      auto results,
      DispatchRngOpGeneral(generator, std::move(dispatch_func), usage));
  ABSL_CHECK_EQ(results.size(), 1)  // CRASH_OK
      << "Expected 1 output buffer, got " << results.size();
  return std::move(results[0]);
}

// Helper that wraps `DispatchRngOpAndReturnBuffer` and assigns the single
// output buffer to `result_tensor`. Supports optional explicit RNG advancement
// parameters; if not specified, they are inferred from the returned buffer.
template <typename DispatchFunc>
absl::Status DispatchRngOp(
    at::Tensor& result_tensor, c10::optional<at::Generator> generator,
    DispatchFunc dispatch_func,
    const std::optional<RngUsage>& usage = std::nullopt) {
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef output_buf,
      DispatchRngOpAndReturnBuffer(generator, std::move(dispatch_func), usage));
  return AssignBufferToAtTensor(std::move(output_buf), result_tensor);
}

// Helper that executes an RNG op with out-tensor via `DispatchOpOut`, handling
// locking, generator state retrieval, and advancement.
template <typename DispatchFunc>
absl::Status DispatchRngOpOut(
    at::Tensor& out, c10::optional<at::Generator> generator,
    mlir::ElementType out_dtype, DispatchFunc dispatch_func,
    const std::optional<RngUsage>& usage = std::nullopt) {
  auto gen = at::get_generator_or_default<DeviceGeneratorImpl>(
      generator, GetDefaultDeviceGenerator());

  std::scoped_lock<std::mutex> lock(gen->mutex_);
  TT_RETURN_IF_ERROR(std::move(dispatch_func)(gen->DeviceStateTensor()));

  int64_t num_elements = usage.has_value() ? usage->num_elements : out.numel();
  int64_t bit_width =
      usage.has_value() ? usage->bit_width : TorchEquivalentBitwidth(out_dtype);
  return gen->AdvanceDeviceStateTensor(num_elements, bit_width);
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_RNG_UTILS_H_
