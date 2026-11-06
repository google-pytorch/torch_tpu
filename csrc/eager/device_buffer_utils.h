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

#ifndef TORCH_TPU_CSRC_EAGER_DEVICE_BUFFER_UTILS_H_
#define TORCH_TPU_CSRC_EAGER_DEVICE_BUFFER_UTILS_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/eager/device_assignment.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

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
  // Device assignment for the operation (e.g. rank-to-device mapping).
  DeviceAssignment device_assignment = {};
  // Whether creating the op may immediately trigger the eager materialization
  // that the current eager mode performs after creating an op: executing the
  // current stream in the kDeferNever modes, or inserting a graph break when
  // the repeated ops heuristic detects a loop in kDeferAndFuse.
  //
  // Set to false for the ops of a sequence that must be compiled into the same
  // graph, such as the ops of an in-place update that donates the destination
  // tensor's buffer before the tensor's c10::DataPtr is updated to the result
  // (see DispatchOpOut). The op is still recorded on the events queue, but the
  // eager materialization is postponed until the caller calls
  // RunPostponedMaterialization() at the end of the sequence. In kDeferAndFuse,
  // this also suspends the repeated ops heuristic's graph breaks on the current
  // stream until then, including for ops created later in the sequence with
  // `allow_materialization = true`.
  bool allow_materialization = true;
};

// Don't use this directly when defining ops. Use DispatchOp<kNumInputs,
// kNumOutputs> instead, or one of the other public functions in this file like
// CreateEmptyDeviceBufferRef.
absl::StatusOr<std::vector<DeviceBufferRef>> CreateDeferredDeviceBufferList(
    DeferredOpParams&& params);

// Runs the eager materialization that the current eager mode postponed for the
// deferred ops created with `allow_materialization = false` (see
// DeferredOpParams):
//  - kDeferNever: starts executing the current stream in the background.
//  - kDeferNeverAndLaunchBlocking: executes the current stream and waits for
//    the execution to complete.
//  - kDeferAndFuse: resumes the repeated ops heuristic's graph breaks on the
//    current stream, and inserts a graph break if the heuristic requested one
//    while they were suspended.
//  - Other modes: does nothing.
//
// Must be called once all the ops of the sequence have been created and their
// results have been assigned to their destination tensors.
absl::Status RunPostponedMaterialization();

// Computes the OpParamCacheKeys for CreateConstantDeviceBufferRefImpl.
absl::StatusOr<OpParamCacheKeys> ComputeConstantDeviceBufferRefOpParamCacheKeys(
    absl::Span<const char> cpu_tensor_data, const Dimensions& dimensions,
    mlir::ElementType element_type);

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

// Whether to donate the base buffer when creating a view.
enum class DonateBase {
  // The base buffer is donated to the view.
  // WARNING: if the base is donated, then it can only be used by other ops
  // that are compiled in the same traversal as the constructed view.
  // Materializing the view separately will invalidate the base and should only
  // be performed if all Tensor references to the base will be dropped before
  // returning to Python.
  kDonate,
  // The base buffer is not donated to the view and remains valid after the
  // view is materialized.
  kDoNotDonate,
};

// Creates a DeviceBufferRef representing a view of a base DeviceBufferRef.
absl::StatusOr<DeviceBufferRef> CreateViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, const TpuLayout& tpu_layout,
    bool view_is_conj, DonateBase donate_base);

absl::StatusOr<DeviceBufferRef> CreateInverseViewDeviceBufferRef(
    DeviceBufferRef base_buffer_ref, DeviceBufferRef write_buf,
    const TpuLayout& tpu_layout, bool view_is_conj, bool donate_base);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_EAGER_DEVICE_BUFFER_UTILS_H_
