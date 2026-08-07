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

#include "torch_tpu/ops/ao/ao_offload_kernels.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/util/Exception.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/device_type.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/view_decomposition/contiguous_to_view.h"

namespace torch_tpu {
namespace {

using mlir::stablehlo::CustomCallApiVersion;      // USING_DECL_OK
using mlir::stablehlo::CustomCallApiVersionAttr;  // USING_DECL_OK

enum class BufferPlacement {
  kPinnedHost,
  kDevice,
};

constexpr std::string_view GetPlacementString(BufferPlacement placement) {
  switch (placement) {
    case BufferPlacement::kPinnedHost:
      return "pinned_host";
    case BufferPlacement::kDevice:
      return "device";
  }
}

mlir::MlirOp BuildPlacementOp(mlir::MlirOp input, BufferPlacement placement) {
  mlir::MlirBuilder& builder = input.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType out_type = GetTensorTypeOrDie(input);

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
          "_xla_buffer_placement",
          op_builder.getStringAttr(GetPlacementString(placement)))}));

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

at::IntArrayRef GetReloadSizes(const at::Tensor& tensor,
                               at::OptionalIntArrayRef original_size) {
  return original_size.has_value() ? *original_size : tensor.sizes();
}

absl::StatusOr<mlir::MlirOp> ReshapeIfNeeded(mlir::MlirOp input,
                                             at::IntArrayRef target_shape) {
  const llvm::ArrayRef<int64_t> current_shape =
      GetTensorTypeOrDie(input).getShape();
  const llvm::ArrayRef<int64_t> target_shape_ref(target_shape.data(),
                                                 target_shape.size());
  if (current_shape == target_shape_ref) {
    return input;
  }
  const Dimensions before(current_shape.begin(), current_shape.end());
  const Dimensions after(target_shape.begin(), target_shape.end());
  return ReshapeFromStaticDimensions(input, before, after);
}

template <typename F>
DeviceBufferRef DispatchAoOp(F op_builder, const at::Tensor& tensor,
                             mlir::ElementType out_dtype,
                             at::IntArrayRef out_dims,
                             OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_THROW(
      DeviceBufferRef device_buffer,
      DispatchOp<1>(op_builder, tensor,
                    {.out_dtype = out_dtype,
                     .out_dims = out_dims,
                     .op_param_cache_keys = std::move(param_keys)}));
  return device_buffer;
}

}  // namespace

at::Tensor AtenAoOffload(const at::Tensor& tensor) {
  TT_KERNEL(OpName::kAoOffload, _, (tensor), {
    const auto op_builder =
        [](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildPlacementOp(input_op, BufferPlacement::kPinnedHost);
    };

    TT_ASSIGN_OR_THROW(const auto out_dtype,
                       ConvertTo<mlir::ElementType>(tensor.scalar_type()));
    DeviceBufferRef device_buffer =
        DispatchAoOp(op_builder, tensor, out_dtype, tensor.sizes(),
                     OpParamCacheKeys::Empty());

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

        const at::IntArrayRef output_dims =
            GetReloadSizes(tensor, original_size);

        const auto op_builder =
            [output_dims](
                mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
          const mlir::MlirOp placement_op =
              BuildPlacementOp(input_op, BufferPlacement::kDevice);
          return ReshapeIfNeeded(placement_op, output_dims);
        };

        TT_ASSIGN_OR_THROW(const auto out_dtype,
                           ConvertTo<mlir::ElementType>(tensor.scalar_type()));
        DeviceBufferRef device_buffer = DispatchAoOp(
            op_builder, tensor, out_dtype, output_dims, std::move(param_keys));

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
        TORCH_WARN_ONCE(
            "ao.wait_tensor is a no-op on TPU. XLA manages transfer "
            "scheduling.");

        return tensor;
      });
}

}  // namespace torch_tpu
