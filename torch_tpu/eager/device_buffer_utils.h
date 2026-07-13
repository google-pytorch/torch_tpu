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

#ifndef TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_
#define TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/layout_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {

namespace internal {

// Parameters for creating a deferred DeviceBufferList.
struct DeferredOpParams {
  // The name of the operation, see op_names.h for possible values.
  OpName op_name;
  // The MLIR op builder for the operation. It must take a vector of MlirOps as
  // inputs and return a vector of MlirOps as outputs.
  MlirOpBuilder op_builder;
  // The cache keys for the operation. See cache_key.h for more details.
  OpParamCacheKeys op_param_cache_keys;
  // The inputs to the operation. May be empty for nullary ops, but must match
  // the number of inputs expected by the op builder.
  std::vector<DeviceBufferRef> inputs;
  // The output shapes and types of the operation. Must match the number of
  // outputs returned by the op builder.
  std::vector<Shape> output_shapes;
  // If specified, forces all inputs to be cast to this dtype before the op
  // builder is applied. The output types are still determined by the
  // output_shapes.
  std::optional<mlir::ElementType> computation_dtype = std::nullopt;
  // The split mode for the operation. See device_buffer.h for more details.
  OpSplitMode split_mode = OpSplitMode::kNone;
  // The indices of the donated inputs. This is primarily used by custom ops
  // (such as Pallas ops).
  Indices donated_indices = {};
};

// Don't use this directly when defining ops. Use DispatchOp<kNumInputs,
// kNumOutputs> instead, or one of the other public functions in this file like
// CreateEmptyDeviceBufferRef.
absl::StatusOr<std::vector<DeviceBufferRef>> CreateDeferredDeviceBufferList(
    DeferredOpParams&& params);

}  // namespace internal

// Creates a deferred DeviceBufferList containing a constant tensor value,
// based on a 1D byte array.
// Under the hood, this uses `mlir::DenseElementsAttr::getFromRawBuffer`,
// with special handling for booleans and complex numbers (which have
// differing layout requirements between PyTorch and XLA).
// This value will be "baked into" the MLIR for XLA compilation, allowing for
// greater optimization potential.
// However, this means that any future execution which have a different value
// for the constant will require a re-compilation.
absl::StatusOr<DeviceBufferRef> CreateConstantDeviceBufferRef(
    std::vector<char> cpu_tensor_data, Dimensions dimensions,
    mlir::ElementType element_type);

// Creates a DeviceBufferList as if by using the torch.empty() operation
// with fill_uninitialized_memory=True.
// This will be a DeferredOp that fills the buffer with NaNs for floats
// (including complex floats), and max for integers and booleans.
absl::StatusOr<DeviceBufferRef> CreateEmptyDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type);

// Creates a DeviceBufferList with a constant value of "no data".
// Equivalent to CreateConstant with an empty cpu_tensor_data vector.
absl::StatusOr<DeviceBufferRef> CreateZeroSizeDeviceBufferRef(
    Dimensions dimensions, mlir::ElementType element_type);

// Creates a DeviceBufferRef representing a view of a base DeviceBufferRef.
absl::StatusOr<DeviceBufferRef> CreateViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, const TpuLayout& tpu_layout,
    bool view_is_conj);

absl::StatusOr<DeviceBufferRef> CreateInverseViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, DeviceBufferRef write_buf,
    const TpuLayout& tpu_layout, bool view_is_conj);

}  // namespace torch_tpu

#endif  // TORCH_TPU_EAGER_DEVICE_BUFFER_UTILS_H_
