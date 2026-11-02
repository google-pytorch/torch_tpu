/*
 * Copyright 2025 Google LLC
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

#include "csrc/ops/native_norm/native_norm_aten_kernels.h"

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/sum.h"
#include "absl/status/statusor.h"
#include "c10/core/DefaultDtype.h"
#include "c10/core/ScalarType.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/linalg/vector_norm/pnorm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

// Helper function that builds and dispatches the MLIR/SHLO operation to compute
// the vector p-norm of a tensor over specified dimensions.
absl::StatusOr<at::Tensor> PNorm(const at::Tensor& self,
                                 MaybePromotedScalar& promoted_ord, double ord,
                                 at::OptionalIntArrayRef dim,
                                 ReductionMode reduction_mode,
                                 std::optional<at::ScalarType> dtype,
                                 OpParamCacheKeys op_cache_keys) {
  c10::ScalarType out_dtype =
      dtype.value_or(c10::toRealValueType(self.scalar_type()));
  if (!dtype.has_value() &&
      c10::isIntegralType(out_dtype, /*includeBool=*/true)) {
    out_dtype = c10::get_default_dtype_as_scalartype();
  }

  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));

  const Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);

  TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  TT_ASSIGN_OR_RETURN(
      at::Tensor ord_tensor,
      promoted_ord.GetTensor(c10::toRealValueType(self.scalar_type())));

  auto op_builder = [ord, canonical_dims, reduction_mode,
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

  return MakeTensor(std::move(result_buf));
}

}  // namespace

// ATen kernel handler for aten::linalg_vector_norm (or native_norm) when only
// the scalar order p is provided without reducing over specific dimensions.
at::Tensor AtenNativeNormScalar(const at::Tensor& self, const at::Scalar& p) {
  MaybePromotedScalar promoted_ord =
      PromoteScalar(p).AvoidPromoting(ScalarValue::kZero);
  TT_KERNEL(OpName::kNativeNorm, op_cache_keys, (self, promoted_ord), {
    if (promoted_ord.IsZero()) {
      at::ScalarType out_dtype = c10::toRealValueType(self.scalar_type());
      if (c10::isIntegralType(out_dtype, /*includeBool=*/true)) {
        out_dtype = c10::get_default_dtype_as_scalartype();
      }
      return at::sum(self.ne(0), at::IntArrayRef{}, /*keepdim=*/false,
                     out_dtype);
    } else {
      TT_THROW_IF_ERROR(op_cache_keys.SetParam("p", p));
      TT_ASSIGN_OR_THROW(at::Tensor res,
                         PNorm(self, promoted_ord, p.toDouble(), std::nullopt,
                               ReductionMode::kDropDims, std::nullopt,
                               std::move(op_cache_keys)));
      return res;
    }
  });
}

// ATen kernel handler for aten::linalg_vector_norm with optional scalar order
// p, optional reduction dimensions, keepdim flag, and optional output dtype.
at::Tensor AtenNativeNormScalarOptDimDtype(
    const at::Tensor& self, const std::optional<at::Scalar>& p,
    at::IntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  at::Scalar ord = p.value_or(2.0);
  MaybePromotedScalar promoted_ord =
      PromoteScalar(ord).AvoidPromoting(ScalarValue::kZero);
  TT_KERNEL(OpName::kNativeNormScalarOptDimDtype, op_cache_keys,
            (self, promoted_ord, dim, keepdim, dtype), {
              if (promoted_ord.IsZero()) {
                at::ScalarType out_dtype =
                    dtype.value_or(c10::toRealValueType(self.scalar_type()));
                if (!dtype.has_value() &&
                    c10::isIntegralType(out_dtype, /*includeBool=*/true)) {
                  out_dtype = c10::get_default_dtype_as_scalartype();
                }
                return at::sum(self.ne(0), dim, keepdim, out_dtype);
              } else {
                TT_THROW_IF_ERROR(op_cache_keys.SetParam("p", ord));
                TT_ASSIGN_OR_THROW(
                    at::Tensor res,
                    PNorm(self, promoted_ord, ord.toDouble(), dim,
                          keepdim ? ReductionMode::kKeepDims
                                  : ReductionMode::kDropDims,
                          dtype, std::move(op_cache_keys)));
                return res;
              }
            });
}

}  // namespace torch_tpu
