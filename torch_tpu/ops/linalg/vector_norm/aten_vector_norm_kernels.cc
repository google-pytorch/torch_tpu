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

#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/Scalar.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/sum.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/OptionalArrayRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/linalg/vector_norm/pnorm.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reduction_utils.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

absl::Status PNormOut(const at::Tensor& self, double ord,
                      at::OptionalIntArrayRef dim, bool keepdim,
                      std::optional<at::ScalarType> dtype, at::Tensor& out,
                      OpParamCacheKeys op_cache_keys) {
  c10::ScalarType out_dtype = dtype.value_or(out.scalar_type());
  ReductionMode reduction_mode =
      keepdim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));

  Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);

  TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  auto fn = [ord, canonical_dims, reduction_mode, element_type](
                mlir::MlirOp input_op) -> absl::StatusOr<mlir::MlirOp> {
    return BuildPNormShlo(input_op, ord, canonical_dims, reduction_mode,
                          element_type);
  };

  return UnaryOpOut(self, out, fn,
                    {.op_param_cache_keys = std::move(op_cache_keys),
                     .out_dtype = element_type,
                     .out_dims = std::move(output_dims)});
}

}  // namespace

at::Tensor& AtenLinalgVectorNormOut(const at::Tensor& self,
                                    const at::Scalar& ord,
                                    at::OptionalIntArrayRef dim, bool keepdim,
                                    std::optional<at::ScalarType> dtype,
                                    at::Tensor& out) {
  TT_KERNEL(
      OpName::kLinalgVectorNormOut, op_cache_keys,
      (self, ord, dim, keepdim, dtype, out), {
        // ord == inf, max(abs(x))
        // ord == -inf, min(abs(x))
        // ord == 0, sum(x != 0), count nonzero
        // ord == int or float, sum(abs(x)^{ord})^{(1 / ord)}

        if (ord.isIntegral(/*includeBool=*/false) && ord.toInt() == 0) {
          // TODO: maybe convert to StableHLO, if more efficient.
          auto temp = at::sum(self.ne(0), dim, keepdim, dtype);
          out.copy_(temp);
        } else {
          TT_THROW_IF_ERROR(PNormOut(self, ord.toDouble(), dim, keepdim, dtype,
                                     out, std::move(op_cache_keys)));
        }
        return out;
      });
}

}  // namespace torch_tpu
