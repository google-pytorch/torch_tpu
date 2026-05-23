// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "torch_tpu/ops/prod/prod_aten_kernels.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/ScalarType.h"
#include "absl/functional/bind_front.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/prod/prod.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {
namespace {

at::ScalarType InferProdDtype(at::ScalarType input_dtype,
                              std::optional<at::ScalarType> dtype) {
  if (dtype.has_value()) {
    // If the user specifies a dtype, use it.
    return dtype.value();
  }
  if (c10::isIntegralType(input_dtype, /*include_bool=*/true)) {
    return c10::ScalarType::Long;
  }
  // Otherwise, the output type is the same as the input type.
  return input_dtype;
}

Dimensions GetSizesAfterProd(at::IntArrayRef self_size,
                             std::optional<int64_t> dim, bool keep_dim) {
  Dimensions output_size;
  // if no dim is provided, then reduce all dimensions.
  if (dim.has_value()) {
    for (int64_t dim_pos = 0; dim_pos < self_size.size(); ++dim_pos) {
      if (dim_pos == dim.value()) {
        if (keep_dim) {
          // If keep_dim is true, then the output size of the reduced dimension
          // is 1.
          output_size.push_back(1);
        }
      } else {
        // If keep_dim is false, then the output size of the reduced dimension
        // is the size of the input tensor.
        output_size.push_back(self_size[dim_pos]);
      }
    }
  }
  return output_size;
}

absl::StatusOr<at::Tensor> AtenProdHelper(const at::Tensor& self,
                                          std::optional<int64_t> dim,
                                          bool keep_dim,
                                          std::optional<at::ScalarType> dtype,
                                          OpParamCacheKeys param_keys) {
  at::ScalarType inferred_dtype = InferProdDtype(self.scalar_type(), dtype);
  if (self.dim() == 0) {
    return self.to(inferred_dtype);
  }
  std::optional<int64_t> normalized_dim = std::nullopt;
  if (dim.has_value()) {
    TT_ASSIGN_OR_RETURN(normalized_dim, SafeWrapDim(*dim, self.dim()));
  }
  std::optional<mlir::ElementType> dtype_mlir_type = std::nullopt;
  if (dtype.has_value()) {
    TT_ASSIGN_OR_RETURN(  // ERROR_COV_INFEASIBLE=there is no way to trigger
                          // this error as PyTorch attempts to create empty
                          // tensor with this dtype and errors out early.
        dtype_mlir_type, ConvertTo<mlir::ElementType>(dtype.value()));
  }
  TT_ASSIGN_OR_RETURN(auto inferred_mlir_dtype,
                      ConvertTo<mlir::ElementType>(inferred_dtype));
  Dimensions output_dims =
      GetSizesAfterProd(self.sizes(), normalized_dim, keep_dim);
  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  TT_ASSIGN_OR_RETURN(
      auto result, UnaryOp(self,
                           absl::bind_front(BuildProdShlo, normalized_dim, mode,
                                            dtype_mlir_type),
                           {.op_param_cache_keys = std::move(param_keys),
                            .out_dtype = inferred_mlir_dtype,
                            .out_dims = std::move(output_dims)}));
  return result;
}

absl::Status AtenProdOutHelper(const at::Tensor& self,
                               std::optional<int64_t> dim, bool keep_dim,
                               std::optional<at::ScalarType> dtype,
                               at::Tensor& out, OpParamCacheKeys param_keys) {
  at::ScalarType inferred_dtype = InferProdDtype(self.scalar_type(), dtype);
  if (self.dim() == 0) {
    out = self.to(inferred_dtype);
    return absl::OkStatus();
  }
  std::optional<int64_t> normalized_dim = std::nullopt;
  if (dim.has_value()) {
    TT_ASSIGN_OR_RETURN(normalized_dim, SafeWrapDim(*dim, self.dim()));
  }
  std::optional<mlir::ElementType> dtype_element_type = std::nullopt;
  if (dtype.has_value()) {
    TT_ASSIGN_OR_RETURN(dtype_element_type,
                        ConvertTo<mlir::ElementType>(dtype.value()));
  }

  TT_ASSIGN_OR_RETURN(auto inferred_mlir_dtype,
                      ConvertTo<mlir::ElementType>(inferred_dtype));
  Dimensions output_dims =
      GetSizesAfterProd(self.sizes(), normalized_dim, keep_dim);
  ReductionMode mode =
      keep_dim ? ReductionMode::kKeepDims : ReductionMode::kDropDims;
  return UnaryOpOut(
      self, out,
      absl::bind_front(BuildProdShlo, normalized_dim, mode, dtype_element_type),
      {.op_param_cache_keys = std::move(param_keys),
       .out_dtype = inferred_mlir_dtype,
       .out_dims = std::move(output_dims)});
}

}  // namespace

at::Tensor AtenProd(const at::Tensor& self,
                    std::optional<at::ScalarType> dtype) {
  TT_KERNEL(OpName::kProd, param_keys, (self, dtype), {
    TT_ASSIGN_OR_THROW(
        at::Tensor result,
        AtenProdHelper(self, /*dim=*/std::nullopt,
                       /*keep_dim=*/false, dtype, std::move(param_keys)));
    return result;
  });
}

at::Tensor& AtenProdDimOut(const at::Tensor& self, int64_t dim, bool keep_dim,
                           std::optional<at::ScalarType> dtype,
                           at::Tensor& out) {
  TT_KERNEL(OpName::kProdDimOut, param_keys, (self, dim, keep_dim, dtype, out),
            {
              TT_THROW_IF_ERROR(AtenProdOutHelper(self, dim, keep_dim, dtype,
                                                  out, std::move(param_keys)));
              return out;
            });
}
}  // namespace torch_tpu
