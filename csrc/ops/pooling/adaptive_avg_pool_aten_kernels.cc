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

#include "csrc/ops/pooling/adaptive_avg_pool_aten_kernels.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/string_view.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/device_buffer_utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/einsum/einsum_aten_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/pooling/avg_pool_aten_kernels.h"
#include "csrc/ops/pooling/pooling.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

enum class PoolingDirection : bool {
  kForward = false,
  kBackward = true,
};

c10::string_view GetAdaptiveAvgPoolEinsumEquation(
    const int64_t spatial_dim_count, const int64_t dim,
    const PoolingDirection direction) {
  if (spatial_dim_count == 2) {
    if (dim == 2) {
      return direction == PoolingDirection::kForward ? "nchw,oh->ncow"
                                                     : "ncow,oh->nchw";
    }
    return direction == PoolingDirection::kForward ? "nchw,ow->ncho"
                                                   : "ncho,ow->nchw";
  }
  if (dim == 2) {
    return direction == PoolingDirection::kForward ? "ncdhw,od->ncohw"
                                                   : "ncohw,od->ncdhw";
  }
  if (dim == 3) {
    return direction == PoolingDirection::kForward ? "ncdhw,oh->ncdow"
                                                   : "ncdow,oh->ncdhw";
  }
  return direction == PoolingDirection::kForward ? "ncdhw,ow->ncdho"
                                                 : "ncdho,ow->ncdhw";
}

// Builds the 1D adaptive average pooling weight matrix W of shape
// [out_size, in_size] in `compute_dtype`, where:
//   s_o = (o * in_size) / out_size
//   e_o = ((o + 1) * in_size + out_size - 1) / out_size
//   W[o, i] = 1.0 / (e_o - s_o)  if s_o <= i < e_o, else 0.0
absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool1dWeightMatrix(
    mlir::MlirBuilder& builder, const int64_t in_size, const int64_t out_size,
    const mlir::Type compute_dtype) {
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const Dimensions col_shape = {out_size, 1};
  const Dimensions mat_shape = {out_size, in_size};
  const mlir::RankedTensorType col_i32_type =
      mlir::RankedTensorType::get(col_shape, i32_type);
  const mlir::RankedTensorType mat_i32_type =
      mlir::RankedTensorType::get(mat_shape, i32_type);
  const mlir::RankedTensorType mat_calc_type =
      mlir::RankedTensorType::get(mat_shape, compute_dtype);

  mlir::MlirOp row_iota = mlir::stablehlo::Iota(builder, col_i32_type, 0);
  mlir::MlirOp in_size_const =
      MakeConstant(builder, in_size, i32_type, col_shape);
  mlir::MlirOp out_size_const =
      MakeConstant(builder, out_size, i32_type, col_shape);
  mlir::MlirOp one_i32 = MakeConstant(builder, 1, i32_type, col_shape);
  mlir::MlirOp out_minus_one_i32 =
      MakeConstant(builder, out_size - 1, i32_type, col_shape);

  // starts = (o * in_size) / out_size
  mlir::MlirOp starts_num = mlir::stablehlo::Mul(row_iota, in_size_const);
  mlir::MlirOp starts = mlir::stablehlo::Div(starts_num, out_size_const);

  // ends = ((o + 1) * in_size + out_size - 1) / out_size
  mlir::MlirOp row_plus_one = mlir::stablehlo::Add(row_iota, one_i32);
  mlir::MlirOp ends_prod = mlir::stablehlo::Mul(row_plus_one, in_size_const);
  mlir::MlirOp ends_num = mlir::stablehlo::Add(ends_prod, out_minus_one_i32);
  mlir::MlirOp ends = mlir::stablehlo::Div(ends_num, out_size_const);

  // inv_lengths = 1.0 / ConvertElementType(ends - starts, compute_dtype)
  mlir::MlirOp lengths = mlir::stablehlo::Subtract(ends, starts);
  mlir::MlirOp lengths_calc =
      mlir::stablehlo::ConvertElementType(lengths, compute_dtype);
  mlir::MlirOp one_calc = MakeConstant(builder, 1.0, compute_dtype, col_shape);
  mlir::MlirOp inv_lengths = mlir::stablehlo::Div(one_calc, lengths_calc);

  const Dimensions bcast_dims = {0, 1};
  mlir::MlirOp starts_mat = mlir::stablehlo::BroadcastInDim(
      mat_i32_type, starts, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp ends_mat = mlir::stablehlo::BroadcastInDim(
      mat_i32_type, ends, AsArrayRef<int64_t>(bcast_dims));
  mlir::MlirOp inv_lengths_mat = mlir::stablehlo::BroadcastInDim(
      mat_calc_type, inv_lengths, AsArrayRef<int64_t>(bcast_dims));

  // i in [0, in_size), shape [out_size, in_size]
  mlir::MlirOp col_iota = mlir::stablehlo::Iota(builder, mat_i32_type, 1);
  mlir::MlirOp ge_start = mlir::stablehlo::Compare(
      col_iota, starts_mat, mlir::stablehlo::ComparisonDirection::GE);
  mlir::MlirOp lt_end = mlir::stablehlo::Compare(
      col_iota, ends_mat, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp in_window = mlir::stablehlo::And(ge_start, lt_end);

  mlir::MlirOp zero_mat = MakeConstant(builder, 0.0, compute_dtype, mat_shape);
  return mlir::stablehlo::Select(in_window, inv_lengths_mat, zero_mat);
}

// Applies 1D adaptive average pooling (forward or backward) along `dim` via
// an MXU einsum contraction with the 1D adaptive pooling weight matrix.
absl::StatusOr<mlir::MlirOp> ApplyAdaptiveAvgPool1D(
    mlir::MlirBuilder& builder, mlir::MlirOp current,
    const int64_t spatial_dim_count, const int64_t dim, const int64_t in_size,
    const int64_t out_size, const PoolingDirection direction,
    const mlir::stablehlo::Precision precision) {
  if (in_size == out_size) {
    return current;
  }
  const mlir::Type compute_dtype = GetTensorTypeOrDie(current).getElementType();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp weight_mat,
                      BuildAdaptiveAvgPool1dWeightMatrix(
                          builder, in_size, out_size, compute_dtype));
  mlir::MlirOp contract_ops[] = {current, weight_mat};
  return BuildEinsumShlo(
      contract_ops,
      GetAdaptiveAvgPoolEinsumEquation(spatial_dim_count, dim, direction),
      precision);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool2dShlo(
    mlir::MlirOp input_op, const int64_t in_h, const int64_t in_w,
    const int64_t out_h, const int64_t out_w, const int64_t spatial_dim_count,
    const mlir::stablehlo::Precision precision) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input_op, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirBuilder& builder = batch_input.getBuilder();

  auto input_type = GetTensorTypeOrDie(batch_input);
  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;

  TT_ASSIGN_OR_RETURN(
      auto h_op, ApplyAdaptiveAvgPool1D(builder, batch_input, spatial_dim_count,
                                        h_dim, in_h, out_h,
                                        PoolingDirection::kForward, precision));
  TT_ASSIGN_OR_RETURN(
      auto w_op,
      ApplyAdaptiveAvgPool1D(builder, h_op, spatial_dim_count, w_dim, in_w,
                             out_w, PoolingDirection::kForward, precision));

  auto result = RemoveTrivialBatch(w_op, batch_input_info.original_dim_size,
                                   spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool3dShlo(
    mlir::MlirOp input_op, const int64_t in_d, const int64_t in_h,
    const int64_t in_w, const int64_t out_d, const int64_t out_h,
    const int64_t out_w, const int64_t spatial_dim_count,
    const mlir::stablehlo::Precision precision) {
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(input_op);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input_op, spatial_dim_count));
  mlir::MlirOp batch_input = batch_input_info.batch_input;
  mlir::MlirBuilder& builder = batch_input.getBuilder();

  auto input_type = GetTensorTypeOrDie(batch_input);
  const int64_t num_dims = input_type.getRank();
  const int64_t d_dim = num_dims - 3;
  const int64_t h_dim = num_dims - 2;
  const int64_t w_dim = num_dims - 1;

  TT_ASSIGN_OR_RETURN(
      auto d_op, ApplyAdaptiveAvgPool1D(builder, batch_input, spatial_dim_count,
                                        d_dim, in_d, out_d,
                                        PoolingDirection::kForward, precision));
  TT_ASSIGN_OR_RETURN(
      auto h_op,
      ApplyAdaptiveAvgPool1D(builder, d_op, spatial_dim_count, h_dim, in_h,
                             out_h, PoolingDirection::kForward, precision));
  TT_ASSIGN_OR_RETURN(
      auto w_op,
      ApplyAdaptiveAvgPool1D(builder, h_op, spatial_dim_count, w_dim, in_w,
                             out_w, PoolingDirection::kForward, precision));

  auto result = RemoveTrivialBatch(w_op, batch_input_info.original_dim_size,
                                   spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool2dBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input,
    const mlir::stablehlo::Precision precision) {
  const int64_t spatial_dim_count = 2;
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(grad_output);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));

  mlir::MlirBuilder& builder = grad_output.getBuilder();

  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_info,
                      CreateBatchInput(grad_output, spatial_dim_count));

  auto input_type = GetTensorTypeOrDie(batch_input_info.batch_input);
  auto grad_output_type = GetTensorTypeOrDie(batch_grad_info.batch_input);

  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;
  const int64_t in_h = input_type.getShape()[h_dim];
  const int64_t in_w = input_type.getShape()[w_dim];
  const int64_t out_h = grad_output_type.getShape()[h_dim];
  const int64_t out_w = grad_output_type.getShape()[w_dim];

  // input: [..., out_h, out_w] -> output: [..., out_h, in_w]
  TT_ASSIGN_OR_RETURN(auto w_op, ApplyAdaptiveAvgPool1D(
                                     builder, batch_grad_info.batch_input,
                                     spatial_dim_count, w_dim, in_w, out_w,
                                     PoolingDirection::kBackward, precision));

  // input: [..., out_h, in_w] -> output: [..., in_h, in_w]
  TT_ASSIGN_OR_RETURN(
      auto h_op,
      ApplyAdaptiveAvgPool1D(builder, w_op, spatial_dim_count, h_dim, in_h,
                             out_h, PoolingDirection::kBackward, precision));

  auto result = RemoveTrivialBatch(h_op, batch_input_info.original_dim_size,
                                   spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

absl::StatusOr<mlir::MlirOp> BuildAdaptiveAvgPool3dBackwardShlo(
    mlir::MlirOp grad_output, mlir::MlirOp input,
    const mlir::stablehlo::Precision precision) {
  const int64_t spatial_dim_count = 3;
  const mlir::ElementType orig_dtype = GetElementTypeOrDie(grad_output);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(orig_dtype));
  TT_ASSIGN_OR_RETURN(grad_output, CastIfNeeded(grad_output, compute_dtype));

  mlir::MlirBuilder& builder = grad_output.getBuilder();

  TT_ASSIGN_OR_RETURN(auto batch_input_info,
                      CreateBatchInput(input, spatial_dim_count));
  TT_ASSIGN_OR_RETURN(auto batch_grad_info,
                      CreateBatchInput(grad_output, spatial_dim_count));

  auto input_type = GetTensorTypeOrDie(batch_input_info.batch_input);
  auto grad_output_type = GetTensorTypeOrDie(batch_grad_info.batch_input);

  const int64_t d_dim = input_type.getRank() - 3;
  const int64_t h_dim = input_type.getRank() - 2;
  const int64_t w_dim = input_type.getRank() - 1;
  const int64_t in_d = input_type.getShape()[d_dim];
  const int64_t in_h = input_type.getShape()[h_dim];
  const int64_t in_w = input_type.getShape()[w_dim];
  const int64_t out_d = grad_output_type.getShape()[d_dim];
  const int64_t out_h = grad_output_type.getShape()[h_dim];
  const int64_t out_w = grad_output_type.getShape()[w_dim];

  // input: [..., out_d, out_h, out_w] -> output: [..., out_d, out_h, in_w]
  TT_ASSIGN_OR_RETURN(auto w_op, ApplyAdaptiveAvgPool1D(
                                     builder, batch_grad_info.batch_input,
                                     spatial_dim_count, w_dim, in_w, out_w,
                                     PoolingDirection::kBackward, precision));

  // input: [..., out_d, out_h, in_w] -> output: [..., out_d, in_h, in_w]
  TT_ASSIGN_OR_RETURN(
      auto h_op,
      ApplyAdaptiveAvgPool1D(builder, w_op, spatial_dim_count, h_dim, in_h,
                             out_h, PoolingDirection::kBackward, precision));

  // input: [..., out_d, in_h, in_w] -> output: [..., in_d, in_h, in_w]
  TT_ASSIGN_OR_RETURN(
      auto d_op,
      ApplyAdaptiveAvgPool1D(builder, h_op, spatial_dim_count, d_dim, in_d,
                             out_d, PoolingDirection::kBackward, precision));

  auto result = RemoveTrivialBatch(d_op, batch_input_info.original_dim_size,
                                   spatial_dim_count);
  return CastIfNeeded(result, orig_dtype);
}

absl::StatusOr<at::Tensor> MakeAdaptiveAvgPoolZeroOutput(
    const Dimensions& out_dims, const mlir::ElementType output_mlir_dtype,
    std::optional<std::reference_wrapper<at::Tensor>> out) {
  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef zero_buf,
      CreateZeroSizeDeviceBufferRef(out_dims, output_mlir_dtype));
  if (out.has_value()) {
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out->get(), out_dims));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(zero_buf), out->get()));
    return out->get();
  }
  return MakeTensor(std::move(zero_buf));
}

mlir::stablehlo::Precision GetAdaptiveAvgPoolPrecision(
    OpParamCacheKeys& param_keys) {
  return GetAndAddPrecisionWithDefaultTo(
      param_keys,
      mlir::stablehlo::Precision::
          HIGHEST);  // EXPLICIT_PRECISION_OK=adaptive_avg_pool
}

}  // namespace

absl::Status ValidateAdaptiveAvgPool2dInputs(const at::Tensor& self) {
  TT_RET_CHECK(self.scalar_type() != at::ScalarType::Bool &&
                   self.scalar_type() != at::ScalarType::Byte &&
                   self.scalar_type() != at::ScalarType::Char &&
                   self.scalar_type() != at::ScalarType::Short &&
                   self.scalar_type() != at::ScalarType::Int &&
                   self.scalar_type() != at::ScalarType::Long &&
                   self.scalar_type() != at::ScalarType::ComplexFloat,
               error::kInvalidArgument)
      << "expected input dtype to be none of (bool, uint8, int8, int16, int32, "
         "int64, complex64), got "
      << torch_tpu::ToString(self.scalar_type());

  const int64_t spatial_dim_count = 2;
  const int64_t num_dims = self.dim();
  TT_RET_CHECK(
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected input to be a " << spatial_dim_count + 1 << "-D or "
      << spatial_dim_count + 2 << "-D tensor, got " << num_dims << "-D tensor";

  for (int64_t i = num_dims - spatial_dim_count; i < num_dims; ++i) {
    TT_RET_CHECK(self.size(i) > 0, error::kInvalidArgument)
        << "expected input to have non-zero size for non-batch dimensions, got "
        << self.sizes() << " with dimension " << i << " being empty";
  }
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> AdaptiveAvgPool2dHelper(
    const at::Tensor& self, c10::SymIntArrayRef output_size,
    at::ScalarType output_dtype, OpParamCacheKeys param_keys,
    std::optional<std::reference_wrapper<at::Tensor>> out = std::nullopt) {
  TT_RET_CHECK(output_size.size() == 1 || output_size.size() == 2,
               error::kInvalidArgument)
      << "expected output_size to have 1 or 2 elements, got "
      << output_size.size();

  // The output size is either a single integer H or a tuple (H, W)
  const int64_t out_h = output_size[0].expect_int();
  const int64_t out_w =
      output_size.size() == 2 ? output_size[1].expect_int() : out_h;
  TT_RET_CHECK(out_h >= 0 && out_w >= 0, error::kInvalidArgument)
      << "expected output_size elements to be non-negative, got "
      << output_size;

  TT_RETURN_IF_ERROR(ValidateAdaptiveAvgPool2dInputs(self));

  const int64_t spatial_dim_count = 2;
  const int64_t num_dims = self.dim();
  TT_ASSIGN_OR_RETURN(const auto output_mlir_dtype,
                      ConvertTo<mlir::ElementType>(output_dtype));

  const int64_t in_h = self.size(num_dims - 2);
  const int64_t in_w = self.size(num_dims - 1);

  Dimensions out_dims = CopyIntVector(self.sizes());
  out_dims[num_dims - 2] = out_h;
  out_dims[num_dims - 1] = out_w;

  if (out_h == 0 || out_w == 0 || self.numel() == 0) {
    return MakeAdaptiveAvgPoolZeroOutput(out_dims, output_mlir_dtype, out);
  }

  if (in_h >= out_h && in_w >= out_w && in_h % out_h == 0 &&
      in_w % out_w == 0) {
    const int64_t stride_h = in_h / out_h;
    const int64_t stride_w = in_w / out_w;
    const int64_t kernel_h = in_h - (out_h - 1) * stride_h;
    const int64_t kernel_w = in_w - (out_w - 1) * stride_w;

    const Dimensions kernel_size = {kernel_h, kernel_w};
    const Dimensions stride = {stride_h, stride_w};
    const Dimensions padding = {0, 0};

    if (out.has_value()) {
      TT_RETURN_IF_ERROR(
          BuildAvgPoolOutNd(
              self, kernel_size, stride, padding, /*ceil_mode=*/false,
              /*count_include_pad=*/true, /*divisor_override=*/std::nullopt,
              out->get(), spatial_dim_count, std::move(param_keys))
              .status());
      return out->get();
    }
    TT_ASSIGN_OR_RETURN(
        auto buffer,
        BuildAvgPoolNd(self, kernel_size, stride, padding, /*ceil_mode=*/false,
                       /*count_include_pad=*/true,
                       /*divisor_override=*/std::nullopt, output_mlir_dtype,
                       out_dims, spatial_dim_count, std::move(param_keys),
                       // Override the OpName to kAvgPool2dOut, since we are
                       // lowering into it, anyway.
                       /*override_op_name=*/OpName::kAvgPool2dOut));
    return MakeTensor(std::move(buffer));
  } else {
    const mlir::stablehlo::Precision precision =
        GetAdaptiveAvgPoolPrecision(param_keys);
    auto op_builder = [in_h, in_w, out_h, out_w, precision](
                          mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
      return BuildAdaptiveAvgPool2dShlo(input, in_h, in_w, out_h, out_w,
                                        spatial_dim_count, precision);
    };

    if (out.has_value()) {
      TT_RETURN_IF_ERROR(
          DispatchOpOut<1>(std::move(op_builder), self, out->get(),
                           {.out_dtype = output_mlir_dtype,
                            .out_dims = std::move(out_dims),
                            .op_param_cache_keys = std::move(param_keys)}));
      return out->get();
    }
    TT_ASSIGN_OR_RETURN(
        auto buffer,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = output_mlir_dtype,
                       .out_dims = std::move(out_dims),
                       .op_param_cache_keys = std::move(param_keys)}));
    return MakeTensor(std::move(buffer));
  }
}

at::Tensor& AtenAdaptiveAvgPool2dOut(const at::Tensor& self,
                                     c10::SymIntArrayRef output_size,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool2dOut, param_keys, (self, output_size, out), {
        TT_THROW_IF_ERROR(AdaptiveAvgPool2dHelper(self, output_size,
                                                  out.scalar_type(),
                                                  std::move(param_keys), out)
                              .status());
        return out;
      });
}

at::Tensor AtenAdaptiveAvgPool2d(const at::Tensor& self,
                                 c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kAdaptiveAvgPool2d, param_keys, (self, output_size), {
    TT_ASSIGN_OR_THROW(auto result, AdaptiveAvgPool2dHelper(
                                        self, output_size, self.scalar_type(),
                                        std::move(param_keys)));
    return result;
  });
}

namespace {

absl::StatusOr<Dimensions> ValidateAdaptiveAvgPool3dOutputSize(
    c10::SymIntArrayRef output_size) {
  TT_RET_CHECK(output_size.size() == 1 || output_size.size() == 3,
               error::kInvalidArgument)
      << "expected output_size to have 1 or 3 elements, got "
      << output_size.size();
  const int64_t out_d = output_size[0].expect_int();
  const int64_t out_h =
      output_size.size() == 3 ? output_size[1].expect_int() : out_d;
  const int64_t out_w =
      output_size.size() == 3 ? output_size[2].expect_int() : out_d;
  TT_RET_CHECK(out_d >= 0 && out_h >= 0 && out_w >= 0, error::kInvalidArgument)
      << "expected output_size elements to be non-negative, got [" << out_d
      << ", " << out_h << ", " << out_w << "]";
  return Dimensions{out_d, out_h, out_w};
}

absl::Status ValidateAdaptiveAvgPool3dInputs(const at::Tensor& self) {
  const int64_t spatial_dim_count = 3;
  const int64_t num_dims = self.dim();
  TT_RET_CHECK(
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected input to be a " << spatial_dim_count + 1 << "-D or "
      << spatial_dim_count + 2 << "-D tensor, got " << num_dims << "-D tensor";

  for (int64_t i = 1; i < num_dims; ++i) {
    TT_RET_CHECK(self.size(i) > 0, error::kInvalidArgument)
        << "expected input to have non-zero size for non-batch dimensions, got "
        << self.sizes() << " with dimension " << i << " being empty";
  }

  return absl::OkStatus();
}

absl::Status ValidateAdaptiveAvgPool3dBackwardGradOutput(
    const at::Tensor& grad_output, const at::Tensor& self) {
  TT_RET_CHECK(grad_output.dim() == self.dim(), error::kInvalidArgument)
      << "expected grad_output to have " << self.dim() << " dimensions, got "
      << grad_output.dim();
  for (int64_t i = 1; i < grad_output.dim(); ++i) {
    TT_RET_CHECK(grad_output.size(i) > 0, error::kInvalidArgument)
        << "expected grad_output to have non-zero size for non-batch "
           "dimensions, got "
        << grad_output.sizes() << " with dimension " << i << " being empty";
  }
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> AdaptiveAvgPool3dHelper(
    const at::Tensor& self, c10::SymIntArrayRef output_size,
    at::ScalarType output_scalar_type, OpParamCacheKeys param_keys,
    std::optional<std::reference_wrapper<at::Tensor>> out = std::nullopt) {
  TT_ASSIGN_OR_RETURN(const Dimensions out_spatial,
                      ValidateAdaptiveAvgPool3dOutputSize(output_size));
  TT_RETURN_IF_ERROR(ValidateAdaptiveAvgPool3dInputs(self));

  const int64_t spatial_dim_count = 3;
  const int64_t num_dims = self.dim();
  const int64_t out_d = out_spatial[0];
  const int64_t out_h = out_spatial[1];
  const int64_t out_w = out_spatial[2];

  Dimensions expected_out_shape = CopyIntVector(self.sizes());
  expected_out_shape[num_dims - 3] = out_d;
  expected_out_shape[num_dims - 2] = out_h;
  expected_out_shape[num_dims - 1] = out_w;

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(output_scalar_type));

  if (out_d == 0 || out_h == 0 || out_w == 0 || self.numel() == 0) {
    return MakeAdaptiveAvgPoolZeroOutput(expected_out_shape, output_dtype, out);
  }

  TT_RET_CHECK(self.scalar_type() != at::ScalarType::Bool &&
                   self.scalar_type() != at::ScalarType::Byte &&
                   self.scalar_type() != at::ScalarType::Char &&
                   self.scalar_type() != at::ScalarType::Short &&
                   self.scalar_type() != at::ScalarType::Int &&
                   self.scalar_type() != at::ScalarType::Long &&
                   self.scalar_type() != at::ScalarType::ComplexFloat,
               error::kInvalidArgument)
      << "expected input dtype to be none of (bool, uint8, int8, "
         "int16, int32, int64, complex64), got "
      << torch_tpu::ToString(self.scalar_type());

  if (out.has_value()) {
    TT_RETURN_IF_ERROR(
        ResizeTensorIfShapeDiffers(out->get(), expected_out_shape));
  }

  const int64_t in_d = self.size(num_dims - 3);
  const int64_t in_h = self.size(num_dims - 2);
  const int64_t in_w = self.size(num_dims - 1);

  const bool use_std_avg_pool = in_d >= out_d && in_h >= out_h &&
                                in_w >= out_w && in_d % out_d == 0 &&
                                in_h % out_h == 0 && in_w % out_w == 0;
  TT_RETURN_IF_ERROR(param_keys.SetParam("std_avg_pool", use_std_avg_pool));
  if (use_std_avg_pool) {
    const int64_t stride_d = in_d / out_d;
    const int64_t stride_h = in_h / out_h;
    const int64_t stride_w = in_w / out_w;
    const int64_t kernel_d = in_d - (out_d - 1) * stride_d;
    const int64_t kernel_h = in_h - (out_h - 1) * stride_h;
    const int64_t kernel_w = in_w - (out_w - 1) * stride_w;

    const Dimensions kernel_size = {kernel_d, kernel_h, kernel_w};
    const Dimensions stride = {stride_d, stride_h, stride_w};
    const Dimensions padding = {0, 0, 0};

    if (out.has_value()) {
      TT_RETURN_IF_ERROR(BuildAvgPoolOutNd(self, kernel_size, stride, padding,
                                           /*ceil_mode=*/false,
                                           /*count_include_pad=*/true,
                                           /*divisor_override=*/std::nullopt,
                                           out->get(), spatial_dim_count,
                                           std::move(param_keys))
                             .status());
      return out->get();
    }
    TT_ASSIGN_OR_RETURN(
        auto buffer,
        BuildAvgPoolNd(self, kernel_size, stride, padding,
                       /*ceil_mode=*/false,
                       /*count_include_pad=*/true,
                       /*divisor_override=*/std::nullopt, output_dtype,
                       expected_out_shape, spatial_dim_count,
                       std::move(param_keys),
                       /*override_op_name=*/OpName::kAvgPool3dOut));
    return MakeTensor(std::move(buffer));
  }

  const mlir::stablehlo::Precision precision =
      GetAdaptiveAvgPoolPrecision(param_keys);
  auto op_builder = [in_d, in_h, in_w, out_d, out_h, out_w, precision](
                        mlir::MlirOp input) -> absl::StatusOr<mlir::MlirOp> {
    return BuildAdaptiveAvgPool3dShlo(input, in_d, in_h, in_w, out_d, out_h,
                                      out_w, spatial_dim_count, precision);
  };

  if (out.has_value()) {
    DispatchOpOptions<1> options = {
        .out_dtype = output_dtype,
        .out_dims = out->get().sizes(),
        .op_param_cache_keys = std::move(param_keys),
    };
    TT_RETURN_IF_ERROR(DispatchOpOut<1>(std::move(op_builder), self, out->get(),
                                        std::move(options)));
    return out->get();
  }
  TT_ASSIGN_OR_RETURN(
      auto buffer,
      DispatchOp<1>(std::move(op_builder), self,
                    {.out_dtype = output_dtype,
                     .out_dims = std::move(expected_out_shape),
                     .op_param_cache_keys = std::move(param_keys)}));
  return MakeTensor(std::move(buffer));
}

absl::Status AdaptiveAvgPool3dBackwardHelper(const at::Tensor& grad_output,
                                             const at::Tensor& self,
                                             OpParamCacheKeys param_keys,
                                             at::Tensor& grad_input) {
  TT_RETURN_IF_ERROR(ValidateAdaptiveAvgPool3dInputs(self));
  TT_RETURN_IF_ERROR(
      ValidateAdaptiveAvgPool3dBackwardGradOutput(grad_output, self));

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(grad_input.scalar_type()));
  const Dimensions out_dims = CopyIntVector(self.sizes());
  if (self.numel() == 0) {
    TT_ASSIGN_OR_RETURN(DeviceBufferRef zero_buf,
                        CreateZeroSizeDeviceBufferRef(out_dims, output_dtype));
    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(grad_input, out_dims));
    TT_RETURN_IF_ERROR(AssignBufferToAtTensor(std::move(zero_buf), grad_input));
    return absl::OkStatus();
  }

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(grad_input, out_dims));

  const int64_t spatial_dim_count = 3;
  const int64_t num_dims = self.dim();
  const int64_t in_d = self.size(num_dims - 3);
  const int64_t in_h = self.size(num_dims - 2);
  const int64_t in_w = self.size(num_dims - 1);
  const int64_t out_d = grad_output.size(num_dims - 3);
  const int64_t out_h = grad_output.size(num_dims - 2);
  const int64_t out_w = grad_output.size(num_dims - 1);

  const bool use_std_avg_pool = in_d >= out_d && in_h >= out_h &&
                                in_w >= out_w && in_d % out_d == 0 &&
                                in_h % out_h == 0 && in_w % out_w == 0;
  TT_RETURN_IF_ERROR(param_keys.SetParam("std_avg_pool", use_std_avg_pool));
  if (use_std_avg_pool) {
    const int64_t stride_d = in_d / out_d;
    const int64_t stride_h = in_h / out_h;
    const int64_t stride_w = in_w / out_w;
    const int64_t kernel_d = in_d - (out_d - 1) * stride_d;
    const int64_t kernel_h = in_h - (out_h - 1) * stride_h;
    const int64_t kernel_w = in_w - (out_w - 1) * stride_w;

    const Dimensions kernel_size = {kernel_d, kernel_h, kernel_w};
    const Dimensions stride = {stride_d, stride_h, stride_w};
    const Dimensions padding = {0, 0, 0};

    return BuildAvgPoolBackwardGradInputNd(
               grad_output, self, kernel_size, stride, padding,
               /*ceil_mode=*/false, /*count_include_pad=*/true,
               /*divisor_override=*/std::nullopt, grad_input, spatial_dim_count,
               std::move(param_keys))
        .status();
  }

  const mlir::stablehlo::Precision precision =
      GetAdaptiveAvgPoolPrecision(param_keys);
  auto op_builder = [precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    return BuildAdaptiveAvgPool3dBackwardShlo(inputs[0], inputs[1], precision);
  };

  DispatchOpOptions<1> options = {
      .out_dtype = output_dtype,
      .out_dims = self.sizes(),
      .op_param_cache_keys = std::move(param_keys),
  };
  return DispatchOpOut<2>(std::move(op_builder), {grad_output, self},
                          grad_input, std::move(options));
}

}  // namespace

at::Tensor& AtenAdaptiveAvgPool3dOut(const at::Tensor& self,
                                     c10::SymIntArrayRef output_size,
                                     at::Tensor& out) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool3dOut, param_keys, (self, output_size, out), {
        TT_THROW_IF_ERROR(AdaptiveAvgPool3dHelper(self, output_size,
                                                  out.scalar_type(),
                                                  std::move(param_keys), out)
                              .status());
        return out;
      });
}

at::Tensor AtenAdaptiveAvgPool3d(const at::Tensor& self,
                                 c10::SymIntArrayRef output_size) {
  TT_KERNEL(OpName::kAdaptiveAvgPool3d, param_keys, (self, output_size), {
    TT_ASSIGN_OR_THROW(auto result, AdaptiveAvgPool3dHelper(
                                        self, output_size, self.scalar_type(),
                                        std::move(param_keys)));
    return result;
  });
}

absl::Status ValidateAdaptiveAvgPoolBackwardGradOutput(
    const at::Tensor& grad_output, const at::Tensor& self) {
  TT_RET_CHECK(grad_output.dim() == self.dim(), error::kInvalidArgument)
      << "expected grad_output to have " << self.dim() << " dimensions, got "
      << grad_output.dim();
  for (int64_t i = 1; i < grad_output.dim(); ++i) {
    TT_RET_CHECK(grad_output.size(i) > 0, error::kInvalidArgument)
        << "expected grad_output to have non-zero size for non-batch "
           "dimensions, got "
        << grad_output.sizes() << " with dimension " << i << " being empty";
  }
  return absl::OkStatus();
}

at::Tensor AtenAdaptiveAvgPool2dBackward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool2dBackward, param_keys, (grad_output, self), {
        TT_THROW_IF_ERROR(ValidateAdaptiveAvgPool2dInputs(self));
        TT_THROW_IF_ERROR(
            ValidateAdaptiveAvgPoolBackwardGradOutput(grad_output, self));

        const int64_t spatial_dim_count = 2;
        const int64_t num_dims = self.dim();
        const int64_t in_h = self.size(num_dims - 2);
        const int64_t in_w = self.size(num_dims - 1);
        const int64_t out_h = grad_output.size(num_dims - 2);
        const int64_t out_w = grad_output.size(num_dims - 1);

        TT_ASSIGN_OR_THROW(
            at::Tensor grad_input,
            MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
        if (self.numel() == 0) {
          return grad_input;
        }

        const bool use_std_avg_pool = in_h >= out_h && in_w >= out_w &&
                                      in_h % out_h == 0 && in_w % out_w == 0;
        TT_THROW_IF_ERROR(
            param_keys.SetParam("std_avg_pool", use_std_avg_pool));
        if (use_std_avg_pool) {
          const int64_t stride_h = in_h / out_h;
          const int64_t stride_w = in_w / out_w;
          const int64_t kernel_h = in_h - (out_h - 1) * stride_h;
          const int64_t kernel_w = in_w - (out_w - 1) * stride_w;

          const Dimensions kernel_size = {kernel_h, kernel_w};
          const Dimensions stride = {stride_h, stride_w};
          const Dimensions padding = {0, 0};

          TT_THROW_IF_ERROR(BuildAvgPoolBackwardGradInputNd(
                                grad_output, self, kernel_size, stride, padding,
                                /*ceil_mode=*/false, /*count_include_pad=*/true,
                                /*divisor_override=*/std::nullopt, grad_input,
                                spatial_dim_count, std::move(param_keys))
                                .status());
          return grad_input;
        }

        TT_ASSIGN_OR_THROW(const auto output_dtype,
                           ConvertTo<mlir::ElementType>(self.scalar_type()));
        const mlir::stablehlo::Precision precision =
            GetAdaptiveAvgPoolPrecision(param_keys);

        auto op_builder = [precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
            -> absl::StatusOr<mlir::MlirOp> {
          return BuildAdaptiveAvgPool2dBackwardShlo(inputs[0], inputs[1],
                                                    precision);
        };

        TT_ASSIGN_OR_THROW(
            auto result,
            (DispatchOp<2>(std::move(op_builder), {grad_output, self},
                           {.out_dtype = output_dtype,
                            .out_dims = CopyIntVector(self.sizes()),
                            .op_param_cache_keys = std::move(param_keys)})));
        TT_THROW_IF_ERROR(
            AssignBufferToAtTensor(std::move(result), grad_input));
        return grad_input;
      });
}

at::Tensor& AtenAdaptiveAvgPool3dBackwardGradInput(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::Tensor& grad_input) {
  TT_KERNEL(OpName::kAdaptiveAvgPool3dBackwardGradInput, param_keys,
            (grad_output, self, grad_input), {
              TT_THROW_IF_ERROR(AdaptiveAvgPool3dBackwardHelper(
                  grad_output, self, std::move(param_keys), grad_input));
              return grad_input;
            });
}

at::Tensor AtenAdaptiveAvgPool3dBackward(const at::Tensor& grad_output,
                                         const at::Tensor& self) {
  TT_KERNEL(
      OpName::kAdaptiveAvgPool3dBackward, param_keys, (grad_output, self), {
        TT_THROW_IF_ERROR(ValidateAdaptiveAvgPool3dInputs(self));
        TT_THROW_IF_ERROR(
            ValidateAdaptiveAvgPool3dBackwardGradOutput(grad_output, self));
        TT_ASSIGN_OR_THROW(
            at::Tensor grad_input,
            MakeEmptyTensor(self.sizes(), self.scalar_type(), self.device()));
        TT_THROW_IF_ERROR(AdaptiveAvgPool3dBackwardHelper(
            grad_output, self, std::move(param_keys), grad_input));
        return grad_input;
      });
}
}  // namespace torch_tpu
