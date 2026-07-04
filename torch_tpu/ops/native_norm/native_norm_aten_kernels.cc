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

#include "torch_tpu/ops/native_norm/native_norm_aten_kernels.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fixed_size_span.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"

namespace torch_tpu {

namespace {

absl::Status CheckEmptyTensorNormIdentity(const at::Tensor& self, double p_val,
                                          at::IntArrayRef dim) {
  if (self.numel() == 0 && (p_val < 0.0 || p_val == INFINITY)) {
    bool is_all_reduce = dim.empty();
    TT_RET_CHECK(!is_all_reduce, error::kInvalidArgument)
        << "expected a non-empty tensor when p is " << p_val
        << " (which has no identity), got 0 elements";
    for (int64_t d : dim) {
      int64_t norm_d = d < 0 ? d + self.dim() : d;
      TT_RET_CHECK(self.size(norm_d) != 0, error::kInvalidArgument)
          << "expected dimension " << d << " to be non-empty when p is "
          << p_val << " (which has no identity), got size 0";
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<at::Tensor> NativeNormImpl(const at::Tensor& self,
                                          MaybePromotedScalar& promoted_p,
                                          at::IntArrayRef dim,
                                          ReductionMode reduction_mode,
                                          std::optional<at::ScalarType> dtype,
                                          OpParamCacheKeys& op_cache_keys) {
  TT_RETURN_IF_ERROR(CheckFloatOrComplex(self.scalar_type()));
  if (dtype.has_value()) {
    TT_RETURN_IF_ERROR(CheckFloatOrComplex(dtype.value()));
  }
  double p_val = promoted_p.scalar().toDouble();
  TT_RETURN_IF_ERROR(CheckEmptyTensorNormIdentity(self, p_val, dim));
  c10::ScalarType out_dtype = at::toRealValueType(
      dtype.has_value() ? dtype.value() : self.scalar_type());
  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));

  Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);

  TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  if (promoted_p.IsZero()) {
    auto zero_builder =
        [canonical_dims, reduction_mode,
         element_type](mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
      mlir::MlirOp zero = MakeConstantLike(input_op, 0);
      mlir::MlirOp mask = mlir::stablehlo::Compare(
          input_op, zero, mlir::stablehlo::ComparisonDirection::NE);
      // Sum in I64 to avoid overflow or precision loss on half-precision types.
      mlir::MlirOp mask_i64 =
          mlir::stablehlo::ConvertElementType(mask, mlir::ElementType::I64);
      TT_ASSIGN_OR_RETURN(mlir::MlirOp sum_op,
                          BuildSumShlo(mask_i64, canonical_dims, reduction_mode,
                                       mlir::ElementType::I64));
      if (GetElementTypeOrDie(sum_op) != element_type) {
        sum_op = mlir::stablehlo::ConvertElementType(sum_op, element_type);
      }
      return sum_op;
    };
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef result_buf,
        DispatchOp<1>(std::move(zero_builder), self,
                      {.out_dtype = element_type,
                       .out_dims = output_dims,
                       .op_param_cache_keys = std::move(op_cache_keys)}));
    return MakeTensor(std::move(result_buf));
  }

  // BuildPNormShlo generates specialized MLIR subgraphs for exact numeric
  // values (such as 1.0, 2.0, inf, -inf). We must inject `p_val` into the
  // cache key to prevent XLA from reusing graphs across different p-norms.
  TT_RETURN_IF_ERROR(op_cache_keys.SetParam("p", p_val));
  TT_ASSIGN_OR_RETURN(at::Tensor p_tensor, promoted_p.GetTensor(out_dtype));

  mlir::ElementType target_element_type = element_type;
  if (c10::isComplexType(self.scalar_type())) {
    TT_ASSIGN_OR_RETURN(
        target_element_type,
        ConvertTo<mlir::ElementType>(c10::toComplexType(out_dtype)));
  }

  auto op_builder = [p_val, canonical_dims, reduction_mode, element_type,
                     target_element_type](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [input_op, p_op] = inputs;
    if (GetElementTypeOrDie(input_op) != target_element_type) {
      input_op =
          mlir::stablehlo::ConvertElementType(input_op, target_element_type);
    }
    return BuildPNormShlo(input_op, p_op, p_val, canonical_dims, reduction_mode,
                          element_type);
  };

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef result_buf,
      DispatchOp<2>(std::move(op_builder), {self, p_tensor},
                    {.out_dtype = element_type,
                     .out_dims = output_dims,
                     .op_param_cache_keys = std::move(op_cache_keys)}));

  return MakeTensor(std::move(result_buf));
}

}  // namespace

at::Tensor AtenNativeNormScalarOptDimDtype(
    const at::Tensor& self, const std::optional<at::Scalar>& p,
    at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  // Why p defaults to 2.0: L2 (Euclidean) norm is the default vector norm in
  // PyTorch.
  // Why AvoidPromoting(ScalarValue::kZero): When p == 0, we compute the 0-norm
  // (count of non-zero elements) via a specialized StableHLO graph, so we avoid
  // promoting 0 to an unused cached scalar tensor.
  MaybePromotedScalar promoted_p =
      PromoteScalar(p.value_or(2.0)).AvoidPromoting(ScalarValue::kZero);
  TT_KERNEL(OpName::kNativeNormScalarOptDimDtype, op_cache_keys,
            (self, promoted_p, dim, keepdim, dtype), {
              ReductionMode reduction_mode =
                  keepdim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
              TT_ASSIGN_OR_THROW(
                  at::Tensor res,
                  NativeNormImpl(self, promoted_p, dim, reduction_mode, dtype,
                                 op_cache_keys));
              return res;
            });
}

at::Tensor AtenNativeNorm(const at::Tensor& self, const at::Scalar& p) {
  // Why AvoidPromoting(ScalarValue::kZero): Divert 0-norm to the non-zero
  // counting graph without creating an unused 0.0 scalar tensor.
  auto promoted_p = PromoteScalar(p).AvoidPromoting(ScalarValue::kZero);
  TT_KERNEL(OpName::kNativeNorm, op_cache_keys, (self, promoted_p), {
    TT_ASSIGN_OR_THROW(
        at::Tensor res,
        NativeNormImpl(self, promoted_p, /*dim=*/{}, ReductionMode::kDropDims,
                       /*dtype=*/std::nullopt, op_cache_keys));
    return res;
  });
}

}  // namespace torch_tpu
