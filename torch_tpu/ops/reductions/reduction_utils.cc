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

#include "torch_tpu/ops/reductions/reduction_utils.h"

#include <cstdint>
#include <string>
#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/OptionalArrayRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "torch_tpu/ops/reductions/reductions.h"
#include "torch_tpu/ops/reductions/sum.h"
#include "torch_tpu/ops/unary_aten_kernels.h"

namespace torch_tpu {

namespace {

MlirUnaryOpBuilder GetSumFunctional(Dimensions dim,
                                    ReductionMode reduction_mode,
                                    mlir::ElementType mlir_type) {
  // The returned lambda captures the dimensions vector by value (via move),
  // ensuring it lives as long as the lambda.
  return
      [dim = std::move(dim), reduction_mode, mlir_type](mlir::MlirOp input_op) {
        return BuildSumShlo(input_op, dim, reduction_mode, mlir_type);
      };
}

// All reduced dimensions are set to size 1.
Dimensions OutputSizesArrayKeepDim(
    at::IntArrayRef self_size, absl::Span<const int64_t> canonicalized_dims) {
  Dimensions output_size = CopyIntVector(self_size);
  for (const auto& d : canonicalized_dims) {
    output_size[d] = 1;
  }
  return output_size;
}

// All reduced dimensions are removed.
Dimensions OutputSizesArrayDiscardDim(
    at::IntArrayRef self_size, absl::Span<const int64_t> canonicalized_dims) {
  Dimensions output_size;
  absl::flat_hash_set<int64_t> dims_to_reduce(canonicalized_dims.begin(),
                                              canonicalized_dims.end());
  output_size.reserve(self_size.size() - dims_to_reduce.size());
  for (int d = 0; d < self_size.size(); ++d) {
    if (dims_to_reduce.find(d) == dims_to_reduce.end()) {
      output_size.push_back(self_size[d]);
    }
  }
  return output_size;
}

absl::StatusOr<OpParamCacheKeys> MakeSumCacheKeys(
    absl::Span<const int64_t> dims, ReductionMode reduction_mode) {
  const bool keep_dim = reduction_mode == ReductionMode::kKeepDims;
  return TT_MAKE_OP_PARAM_CACHE_KEYS(keep_dim, dims);
}

}  // namespace

Dimensions GetSizesAfterReduction(at::IntArrayRef self_size,
                                  ReductionMode reduction_mode,
                                  at::IntArrayRef canonicalized_dims) {
  return reduction_mode == ReductionMode::kKeepDims
             ? OutputSizesArrayKeepDim(self_size, canonicalized_dims)
             : OutputSizesArrayDiscardDim(self_size, canonicalized_dims);
}

absl::Status CheckFloatOrComplex(c10::ScalarType scalar_type) {
  TT_ASSIGN_OR_RETURN(const auto dtype,
                      ConvertTo<mlir::ElementType>(scalar_type));
  TT_RET_CHECK(
      c10::isFloatingType(scalar_type) || c10::isComplexType(scalar_type),
      error::kInvalidArgument)
      << "expected a floating point or complex dtype, got " << ToString(dtype);
  return absl::OkStatus();
}

absl::StatusOr<Dimensions> CanonicalizeDims(
    const at::Tensor& tensor, c10::OptionalArrayRef<int64_t> dim) {
  int64_t rank = tensor.dim();
  ABSL_CHECK(rank >= 0);  // CRASH_OK

  auto log_and_return =
      [&](Dimensions canonical_dims) -> absl::StatusOr<Dimensions> {
    if (ABSL_VLOG_IS_ON(3)) {
      const std::string dim_str =
          dim.has_value() ? absl::StrCat("[", absl::StrJoin(*dim, ","), "]")
                          : "nullopt";
      ABSL_VLOG(3) << "CanonicalizeDims: tensor.sizes=["
                   << absl::StrJoin(tensor.sizes(), ",") << "], dim=" << dim_str
                   << ", canonical_dims=[" << absl::StrJoin(canonical_dims, ",")
                   << "]";
    }
    return canonical_dims;
  };

  if (!dim.has_value() || dim->empty()) {
    // Reduce all dimensions: [0, 1, ..., rank - 1].
    Dimensions canonical_dims(rank);
    absl::c_iota(canonical_dims, 0);
    return log_and_return(std::move(canonical_dims));
  }

  if (rank == 0) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
        *dim == Dimensions{0} || *dim == Dimensions{-1}, error::kOutOfRange)
        << "dim must be 0 or -1 for a scalar, got " << *dim;
    return log_and_return({});
  }

  Dimensions canonical_dims;
  canonical_dims.reserve(dim->size());
  absl::flat_hash_set<int64_t> seen_dims;
  // Dim values can be between -rank and rank - 1.
  const int64_t min_value = -rank;
  const int64_t max_value = rank - 1;
  for (const auto& d : *dim) {
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
        d <= max_value && d >= min_value, error::kOutOfRange)
        << "dim must be in range [" << min_value << ", " << max_value
        << "], got " << d;
    int64_t pos_dim = d < 0 ? d + rank : d;
    TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=PyTorch catches this error first.
        seen_dims.insert(pos_dim).second, error::kInvalidArgument)
        << "dim must contain unique dimensions, got " << *dim << ", where "
        << pos_dim << " appears multiple times";
    canonical_dims.push_back(pos_dim);
  }

  return log_and_return(std::move(canonical_dims));
}

absl::StatusOr<int64_t> GetReductionFactor(const at::Tensor& input_tensor,
                                           c10::OptionalArrayRef<int64_t> dim) {
  int64_t reduction_size = 1;
  if (dim.has_value()) {
    TT_ASSIGN_OR_RETURN(Dimensions canonical_dims,
                        CanonicalizeDims(input_tensor, dim));
    for (int64_t d : canonical_dims) {
      TT_ASSIGN_OR_RETURN(
          reduction_size, SafeMultiply(reduction_size, input_tensor.size(d)),
          _.SetPrepend()
              << "overflow calculating reduction factor for tensor of size "
              << dim.value() << ": ");
    }
  } else {
    reduction_size = input_tensor.numel();
  }
  return reduction_size;
}

absl::StatusOr<at::Tensor> ApplySumReduction(const at::Tensor& self,
                                             c10::OptionalArrayRef<int64_t> dim,
                                             ReductionMode reduction_mode,
                                             c10::ScalarType out_dtype) {
  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));
  Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      MakeSumCacheKeys(canonical_dims, reduction_mode));
  TT_ASSIGN_OR_RETURN(mlir::ElementType mlir_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  return UnaryOp(
      self,
      GetSumFunctional(std::move(canonical_dims), reduction_mode, mlir_type),
      // Use kSum instead of the OpName passed to TT_KERNEL() as this is not for
      // implementing a specific op, but rather a general reduction utility.
      {.op_name = OpName::kSum,
       .op_param_cache_keys = std::move(param_keys),
       .out_dtype = mlir_type,
       .out_dims = std::move(output_dims)});
}

absl::Status ApplySumReductionOut(const at::Tensor& self, at::Tensor& out,
                                  c10::OptionalArrayRef<int64_t> dim,
                                  ReductionMode reduction_mode,
                                  c10::ScalarType out_dtype) {
  TT_ASSIGN_OR_RETURN(Dimensions canonical_dims, CanonicalizeDims(self, dim));
  Dimensions output_dims =
      GetSizesAfterReduction(self.sizes(), reduction_mode, canonical_dims);
  TT_ASSIGN_OR_RETURN(auto param_keys,
                      MakeSumCacheKeys(canonical_dims, reduction_mode));
  TT_ASSIGN_OR_RETURN(mlir::ElementType mlir_type,
                      ConvertTo<mlir::ElementType>(out_dtype));

  return UnaryOpOut(
      self, out,
      GetSumFunctional(std::move(canonical_dims), reduction_mode, mlir_type),
      // Use kSumIntListOut instead of the OpName passed to TT_KERNEL() as this
      // is not for implementing a specific op, but rather a general reduction
      // utility.
      {.op_name = OpName::kSumIntListOut,
       .op_param_cache_keys = std::move(param_keys),
       .out_dtype = mlir_type,
       .out_dims = std::move(output_dims)});
}

}  // namespace torch_tpu
