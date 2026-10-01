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

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/sum.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/OptionalArrayRef.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/linalg/vector_norm/pnorm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reduction_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

absl::Status ValidateLinalgVectorNormInputs(const at::Tensor& self,
                                            const at::Scalar& ord,
                                            at::OptionalIntArrayRef dim,
                                            std::optional<at::ScalarType> dtype,
                                            const at::Tensor& out) {
  TT_RET_CHECK(IsPrivateUse1Device(self) && IsPrivateUse1Device(out),
               error::kInvalidArgument)
      << "expected self and out tensors to be on "
      << GetPrivateUse1DeviceDebugName() << ", got " << self.device() << " and "
      << out.device();
  TT_RET_CHECK(IsFloatingPoint(self) || IsComplex(self),
               error::kInvalidArgument)
      << "expected self dtype to be floating point or complex, got "
      << ToString(self.scalar_type());
  TT_RET_CHECK(!ord.isComplex(), error::kInvalidArgument)
      << "expected a non-complex scalar as the order of norm, got "
      << ToString(ord.type());

  if (dtype.has_value()) {
    const at::ScalarType target_dtype = *dtype;
    const bool valid_target_dtype =
        (c10::isFloatingType(target_dtype) || IsComplex(target_dtype)) &&
        IsComplex(self) == IsComplex(target_dtype) &&
        c10::promoteTypes(self.scalar_type(), target_dtype) == target_dtype;
    TT_RET_CHECK(valid_target_dtype, error::kInvalidArgument)
        << "expected input dtype " << ToString(self.scalar_type())
        << " to be convertible without narrowing to matching real/complex "
           "floating-point dtype, got "
        << ToString(target_dtype);
  }

  const double ord_val = ord.toDouble();
  if (self.numel() == 0 &&
      (ord_val < 0.0 || ord_val == std::numeric_limits<double>::infinity())) {
    TT_ASSIGN_OR_RETURN(const Dimensions canonical_dims,
                        CanonicalizeDims(self, dim));
    for (const int64_t d : canonical_dims) {
      TT_RET_CHECK(self.size(d) != 0, error::kInvalidArgument)
          << "cannot compute the " << ord_val
          << " norm on an empty tensor or dimension because the operation "
             "does not have an identity";
    }
  }
  return absl::OkStatus();
}

absl::Status PNormOut(const at::Tensor& self, MaybePromotedScalar& promoted_ord,
                      double ord, at::OptionalIntArrayRef dim, bool keepdim,
                      at::ScalarType expected_dtype, at::Tensor& out,
                      OpParamCacheKeys op_cache_keys) {
  ReductionMode reduction_mode =
      keepdim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));

  Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);

  TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(expected_dtype));

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

  return DispatchOpOut<2>(std::move(op_builder), {self, ord_tensor}, {out},
                          {.out_dtype = element_type,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(op_cache_keys)});
}

}  // namespace

at::Tensor& AtenLinalgVectorNormOut(const at::Tensor& self,
                                    const at::Scalar& ord,
                                    at::OptionalIntArrayRef dim, bool keepdim,
                                    std::optional<at::ScalarType> dtype,
                                    at::Tensor& out) {
  MaybePromotedScalar promoted_ord =
      PromoteScalar(ord).AvoidPromoting(ScalarValue::kZero);
  TT_KERNEL(
      OpName::kLinalgVectorNormOut, op_cache_keys,
      (self, promoted_ord, dim, keepdim, dtype, out), {
        // ord == inf, max(abs(x))
        // ord == -inf, min(abs(x))
        // ord == 0, sum(x != 0), count nonzero
        // ord == int or float, sum(abs(x)^{ord})^{(1 / ord)}
        TT_THROW_IF_ERROR(
            ValidateLinalgVectorNormInputs(self, ord, dim, dtype, out));

        const at::ScalarType expected_dtype =
            c10::toRealValueType(dtype.value_or(self.scalar_type()));
        TT_THROW_IF_ERROR(ValidateOutDtype(out, expected_dtype,
                                           /*allow_out_dtype_cast=*/false));

        if (promoted_ord.IsZero()) {
          // TODO: maybe convert to StableHLO, if more efficient.
          auto temp = at::sum(self.ne(0), dim, keepdim, expected_dtype);
          TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, temp.sizes()));
          out.copy_(temp);
        } else {
          TT_THROW_IF_ERROR(op_cache_keys.SetParam("ord", ord));
          TT_THROW_IF_ERROR(PNormOut(self, promoted_ord, ord.toDouble(), dim,
                                     keepdim, expected_dtype, out,
                                     std::move(op_cache_keys)));
        }
        return out;
      });
}

}  // namespace torch_tpu
