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

#include "csrc/ops/index_reduce/index_reduce_aten_kernels.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/utils.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/ops/index_utils.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/scatter/scatter.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

absl::StatusOr<ScatterOp> GetReduceOp(const std::string_view reduce) {
  if (reduce == "prod") {
    return ScatterOp::kProd;
  } else if (reduce == "mean") {
    return ScatterOp::kMean;
  } else if (reduce == "amax") {
    return ScatterOp::kAmax;
  } else if (reduce == "amin") {
    return ScatterOp::kAmin;
  }
  return TT_ERROR(error::kInvalidArgument)
         << "Expected reduce to be one of prod, mean, amax or amin but got "
         << reduce;
}

absl::Status IndexReduce(const at::Tensor& self, int64_t dim,
                         const at::Tensor& index, const at::Tensor& source,
                         const std::string_view reduce, bool include_self,
                         at::Tensor& out, OpParamCacheKeys param_keys) {
  TT_ASSIGN_OR_RETURN(dim,
                      ValidateIndexInputsAndGetDim(self, dim, index, source));
  TT_ASSIGN_OR_RETURN(ScatterOp reduce_op, GetReduceOp(reduce));
  ScatterIncludeSelf include_self_enum =
      include_self ? ScatterIncludeSelf::kYes : ScatterIncludeSelf::kNo;

  at::ScalarType promoted_scalar_type =
      c10::promoteTypes(self.scalar_type(), out.scalar_type());
  TT_ASSIGN_OR_RETURN(const auto computation_dtype,
                      ConvertTo<mlir::ElementType>(promoted_scalar_type));

  auto index_reduce_op_builder =
      [dim, reduce_op, include_self_enum,
       computation_dtype](FixedSizeSpan<mlir::MlirOp, 3> inputs) {
        auto& [self, index, source] = inputs;
        return BuildScatterShlo(self, dim, index, source, reduce_op,
                                computation_dtype, include_self_enum);
      };

  TT_ASSIGN_OR_RETURN(const auto output_dtype,
                      ConvertTo<mlir::ElementType>(out.scalar_type()));
  Dimensions output_dims = CopyIntVector(self.sizes());

  return DispatchOpOut<3>(std::move(index_reduce_op_builder),
                          {self, index, source}, out,
                          {.out_dtype = output_dtype,
                           .out_dims = output_dims,
                           .op_param_cache_keys = std::move(param_keys)});
}

}  // namespace

at::Tensor& TpuAtenIndexReduceOut(const at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  const at::Tensor& source,
                                  std::string_view reduce, bool include_self,
                                  at::Tensor& out) {
  TT_KERNEL(OpName::kIndexReduceOut, param_keys,
            (self, dim, index, source, reduce, include_self, out), {
              TT_THROW_IF_ERROR(IndexReduce(self, dim, index, source, reduce,
                                            include_self, out,
                                            std::move(param_keys)));
              return out;
            });
}

}  // namespace torch_tpu
