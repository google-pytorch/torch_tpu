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

#include "csrc/ops/norm/norm_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/linalg/vector_norm/pnorm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateOutDtype(const at::Tensor& self, const at::Tensor& out,
                              std::optional<at::ScalarType> dtype) {
  at::ScalarType expected_dtype =
      dtype.value_or(c10::toRealValueType(self.scalar_type()));
  if (!dtype.has_value() &&
      c10::isIntegralType(expected_dtype, /*includeBool=*/true)) {
    expected_dtype = c10::get_default_dtype_as_scalartype();
  }
  TT_RET_CHECK(out.scalar_type() == expected_dtype, error::kInvalidArgument)
      << "expected the output dtype to be " << ToString(expected_dtype)
      << ", got " << ToString(out.scalar_type());
  return absl::OkStatus();
}

absl::StatusOr<mlir::MlirOp> BuildZeroNormShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_type));
  const auto compute_mlir_type =
      mlir::getElementType(builder.getContext(), compute_dtype);

  if (IsComplexType(GetTensorTypeOrDie(input_op))) {
    input_op = mlir::stablehlo::Abs(input_op);
  }
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  mlir::MlirOp zero = MakeScalarConstant(builder, 0.0, compute_dtype);
  mlir::MlirOp zero_bcast =
      mlir::stablehlo::BroadcastInDim(GetTensorTypeOrDie(input_op), zero, {});
  mlir::MlirOp ne_op = mlir::stablehlo::Compare(
      input_op, zero_bcast, mlir::stablehlo::ComparisonDirection::NE);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp ne_cast, CastIfNeeded(ne_op, compute_dtype));

  mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, compute_dtype);
  const auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp reduction_res =
      BuildReductionShlo(ne_cast, reduce_dims, compute_mlir_type, init_val,
                         reduce_fn, reduction_mode);
  return CastIfNeeded(reduction_res, out_type);
}

absl::Status PNormOut(const at::Tensor& self, PromotedScalar& promoted_ord,
                      double ord, at::IntArrayRef dim, bool keepdim,
                      std::optional<at::ScalarType> dtype, at::Tensor& out,
                      OpParamCacheKeys op_cache_keys) {
  TT_RETURN_IF_ERROR(ValidateOutDtype(self, out, dtype));

  const at::ScalarType out_dtype = out.scalar_type();
  const ReductionMode reduction_mode =
      keepdim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(const Dimensions canonical_dims,
                      CanonicalizeDims(self, dim));

  const Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  TT_ASSIGN_OR_RETURN(
      const at::Tensor ord_tensor,
      promoted_ord.GetTensor(c10::toRealValueType(self.scalar_type())));

  if (ord == 0.0) {
    const auto op_builder =
        [canonical_dims, reduction_mode,
         element_type](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      return BuildZeroNormShlo(input_op, canonical_dims, reduction_mode,
                               element_type);
    };

    TT_ASSIGN_OR_RETURN(
        auto result_buf,
        DispatchOp<1>(std::move(op_builder), self,
                      {.out_dtype = element_type,
                       .out_dims = output_dims,
                       .op_param_cache_keys = std::move(op_cache_keys)}));

    TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, output_dims));
    return AssignBufferToAtTensor(std::move(result_buf), out);
  }

  const auto op_builder = [ord, canonical_dims, reduction_mode,
                           element_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input_op, ord_op] = inputs;
    return BuildPNormShlo(input_op, ord_op, ord, canonical_dims, reduction_mode,
                          element_type);
  };

  TT_ASSIGN_OR_RETURN(
      auto result_buf,
      DispatchOp<2>(std::move(op_builder), {self, ord_tensor},
                    {.out_dtype = element_type,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(op_cache_keys)}));

  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(out, output_dims));
  return AssignBufferToAtTensor(std::move(result_buf), out);
}

}  // namespace

at::Tensor& AtenNormOut(const at::Tensor& self,
                        const std::optional<at::Scalar>& p, at::IntArrayRef dim,
                        bool keepdim, at::Tensor& out) {
  const at::Scalar ord = p.value_or(2.0);
  PromotedScalar promoted_ord = PromoteScalar(ord);
  TT_KERNEL(OpName::kNormOut, op_cache_keys,
            (self, promoted_ord, dim, keepdim, out), {
              TT_THROW_IF_ERROR(op_cache_keys.SetParam("p", ord));
              TT_THROW_IF_ERROR(PNormOut(self, promoted_ord, ord.toDouble(),
                                         dim, keepdim, std::nullopt, out,
                                         std::move(op_cache_keys)));
              return out;
            });
}

at::Tensor& AtenNormDtypeOut(const at::Tensor& self,
                             const std::optional<at::Scalar>& p,
                             at::IntArrayRef dim, bool keepdim,
                             at::ScalarType dtype, at::Tensor& out) {
  const at::Scalar ord = p.value_or(2.0);
  PromotedScalar promoted_ord = PromoteScalar(ord);
  TT_KERNEL(OpName::kNormDtypeOut, op_cache_keys,
            (self, promoted_ord, dim, keepdim, dtype, out), {
              TT_THROW_IF_ERROR(op_cache_keys.SetParam("p", ord));
              TT_THROW_IF_ERROR(PNormOut(self, promoted_ord, ord.toDouble(),
                                         dim, keepdim, dtype, out,
                                         std::move(op_cache_keys)));
              return out;
            });
}

}  // namespace torch_tpu
