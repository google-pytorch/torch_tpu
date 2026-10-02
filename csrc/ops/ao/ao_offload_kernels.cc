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

#include "csrc/ops/ao/ao_offload_kernels.h"

#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/util/Exception.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/utils.h"
#include "csrc/common/warnings.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/view_decomposition/contiguous_to_view.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {
namespace {

using mlir::stablehlo::CustomCallApiVersion;      // USING_DECL_OK
using mlir::stablehlo::CustomCallApiVersionAttr;  // USING_DECL_OK

mlir::MlirOp BuildPlacementOp(mlir::MlirOp input, std::string_view placement) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::NamedAttribute call_target = op_builder.getNamedAttr(
      "call_target_name",
      op_builder.getStringAttr("annotate_device_placement"));
  const mlir::NamedAttribute has_side_effect =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(true));
  const mlir::NamedAttribute api_version = op_builder.getNamedAttr(
      "api_version",
      CustomCallApiVersionAttr::get(
          &builder.getContext(), CustomCallApiVersion::API_VERSION_ORIGINAL));
  const mlir::NamedAttribute frontend = op_builder.getNamedAttr(
      "mhlo.frontend_attributes",
      op_builder.getDictionaryAttr({op_builder.getNamedAttr(
          "_xla_buffer_placement", op_builder.getStringAttr(placement))}));
  const mlir::RankedTensorType out_type = GetTensorTypeOrDie(input);

  auto op = mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(), {out_type}, {input.getValue()},
      {call_target, has_side_effect, api_version, frontend});
  return mlir::MlirOp(builder, op.getResult(0));
}

at::Tensor MaybeApplyStridedView(DeviceBufferRef device_buffer,
                                 at::OptionalIntArrayRef original_stride) {
  if (!original_stride.has_value()) {
    return MakeTensor(std::move(device_buffer));
  }

  TT_ASSIGN_OR_THROW(
      const at::Tensor result_tensor,
      ContiguousToView(std::move(device_buffer), *original_stride,
                       /*target_storage_offset=*/0));
  return result_tensor;
}

absl::StatusOr<mlir::MlirOp> ReshapeIfNeeded(mlir::MlirOp input,
                                             at::IntArrayRef input_shape,
                                             at::IntArrayRef target_shape) {
  if (input_shape == target_shape) {
    return input;
  }
  const Dimensions before = CopyIntVector(input_shape);
  const Dimensions after = CopyIntVector(target_shape);
  return ReshapeFromStaticDimensions(input, before, after);
}

}  // namespace

at::Tensor AtenAoOffload(const at::Tensor& tensor) {
  TT_KERNEL(OpName::kAoOffload, _, (tensor), {
    const auto op_builder =
        [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPlacementOp(input_op, "pinned_host");
    };

    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(tensor.scalar_type()));
    TT_ASSIGN_OR_THROW(
        DeviceBufferRef device_buffer,
        DispatchOp<1>(op_builder, tensor,
                      {.out_dtype = out_dtype,
                       .out_dims = tensor.sizes(),
                       .op_param_cache_keys = OpParamCacheKeys::Empty()}));
    device_buffer.set_is_pinned_host();

    return MakeTensor(std::move(device_buffer));
  });
}

at::Tensor AtenAoReload(const at::Tensor& tensor, at::Device device,
                        at::OptionalIntArrayRef original_size,
                        at::OptionalIntArrayRef original_stride) {
  TT_KERNEL(
      OpName::kAoReload, param_keys,
      (tensor, device, original_size, original_stride), {
        TT_CHECK_THROW(device.type() == GetPrivateUse1DeviceType(),
                       error::kInvalidArgument)
            << "expected 'tpu' device, got '" << device << "'";
        if (device.has_index() && tensor.device().has_index()) {
          TT_CHECK_THROW(device.index() == tensor.device().index(),
                         error::kInvalidArgument)
              << "expected reload device '" << tensor.device() << "', got '"
              << device << "'";
        }

        const Dimensions input_dims = CopyIntVector(tensor.sizes());
        const Dimensions output_dims =
            CopyIntVector(original_size.value_or(tensor.sizes()));

        const auto op_builder =
            [input_dims, output_dims](
                mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::MlirOp placement_op =
              BuildPlacementOp(input_op, "device");
          return ReshapeIfNeeded(placement_op, input_dims, output_dims);
        };

        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(tensor.scalar_type()));
        TT_ASSIGN_OR_THROW(
            DeviceBufferRef device_buffer,
            DispatchOp<1>(op_builder, tensor,
                          {.out_dtype = out_dtype,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys)}));

        return MaybeApplyStridedView(std::move(device_buffer), original_stride);
      });
}

at::Tensor AtenAoWaitTensor(
    const at::Tensor& tensor, const std::optional<at::Tensor>& keepalive,
    const std::optional<at::Tensor>& last_use_of_storage) {
  TT_KERNEL(
      OpName::kAoWaitTensor, _,
      (tensor, IgnoreInCacheKey(keepalive, "Doesn't affect wait_tensor"),
       IgnoreInCacheKey(last_use_of_storage, "Doesn't affect wait_tensor")),
      {
        // wait_tensor is an identity pass-through; XLA manages transfer
        // scheduling.
        TT_WARN_ONCE(WarningCategory::kTorchTpu,
                     "ao.wait_tensor is a no-op on TPU. XLA manages transfer "
                     "scheduling.");

        return tensor;
      });
}

}  // namespace torch_tpu
