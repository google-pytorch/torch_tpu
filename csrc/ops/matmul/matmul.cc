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

#include "csrc/ops/matmul/matmul.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/unary.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

Dimensions MakeIotaDims(const int64_t count) {
  Dimensions dims(count);
  std::iota(dims.begin(), dims.end(), 0);
  return dims;
}

bool AreDimensionsEqual(const stablehlo::Dimensions& a,
                        const stablehlo::Dimensions& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size != b[i].size || a[i].boundOp != b[i].boundOp ||
        a[i].boundOpDim != b[i].boundOpDim) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<stablehlo::Dimensions> ComputeBroadcastBatchDims(
    const stablehlo::Dimensions& lhs_dims,
    const stablehlo::Dimensions& rhs_dims) {
  const int64_t lhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(lhs_dims.size()) - 2);
  const int64_t rhs_batch_rank =
      std::max<int64_t>(0, static_cast<int64_t>(rhs_dims.size()) - 2);
  const int64_t batch_rank = std::max(lhs_batch_rank, rhs_batch_rank);
  stablehlo::Dimensions batch_dims;
  batch_dims.reserve(batch_rank);
  for (int64_t i = 0; i < batch_rank; ++i) {
    const int64_t lhs_idx = i - (batch_rank - lhs_batch_rank);
    const int64_t rhs_idx = i - (batch_rank - rhs_batch_rank);
    if (lhs_idx < 0) {
      batch_dims.push_back(rhs_dims[rhs_idx]);
    } else if (rhs_idx < 0) {
      batch_dims.push_back(lhs_dims[lhs_idx]);
    } else {
      const auto& l_dim = lhs_dims[lhs_idx];
      const auto& r_dim = rhs_dims[rhs_idx];
      TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Checked before MLIR lowering.
          l_dim.size == r_dim.size || l_dim.size == 1 || r_dim.size == 1,
          error::kInvalidArgument)
          << "batch dimensions are not broadcastable: " << l_dim.size << " vs "
          << r_dim.size;
      if (l_dim.size == 1) {
        batch_dims.push_back(r_dim);
      } else if (r_dim.size == 1) {
        batch_dims.push_back(l_dim);
      } else {
        batch_dims.push_back(l_dim.boundOp.has_value() ? l_dim : r_dim);
      }
    }
  }
  return batch_dims;
}

absl::StatusOr<mlir::MlirOp> BroadcastMatrixBatchDims(
    mlir::MlirOp matrix, const stablehlo::Dimensions& batch_dims) {
  const stablehlo::Dimensions matrix_dims = GetDimensions(matrix);
  const int64_t rank = matrix_dims.size();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Internal invariant (callers ensure rank
                 // >= 2).
      rank >= 2, error::kInvalidArgument)
      << "expected matrix rank >= 2, got " << rank;
  stablehlo::Dimensions target_dims = batch_dims;
  target_dims.push_back(matrix_dims[rank - 2]);
  target_dims.push_back(matrix_dims[rank - 1]);
  if (AreDimensionsEqual(matrix_dims, target_dims)) {
    return matrix;
  }
  return BroadcastIfNeeded(matrix, target_dims);
}

mlir::MlirOp EmitDotGeneral(mlir::MlirOp lhs, mlir::MlirOp rhs,
                            absl::Span<const int64_t> lhs_batching,
                            absl::Span<const int64_t> rhs_batching,
                            absl::Span<const int64_t> lhs_contracting,
                            absl::Span<const int64_t> rhs_contracting,
                            const stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const bool is_i64 = lhs_type.getElementType().isInteger(64);
  if (is_i64) {
    // XLA doesn't support matmul with i64, so we convert them to f64.
    lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
  }
  mlir::MLIRContext& ctx = lhs.getContext();
  const auto dot_dimension_numbers = stablehlo::DotDimensionNumbersAttr::get(
      &ctx, AsArrayRef(lhs_batching), AsArrayRef(rhs_batching),
      AsArrayRef(lhs_contracting), AsArrayRef(rhs_contracting));
  const mlir::ArrayAttr precision_config =
      stablehlo::PrecisionConfigAttr::get(&ctx, {precision, precision});
  mlir::MlirOp result =
      stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers, precision_config);
  if (is_i64) {
    result = stablehlo::ConvertElementType(result, mlir::ElementType::I64);
  }
  return result;
}

absl::StatusOr<mlir::MlirOp> ReduceBroadcastedBatchDims(
    mlir::MlirOp grad_bcast, mlir::MlirOp target_op) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad_bcast);
  const mlir::RankedTensorType target_type = GetTensorTypeOrDie(target_op);
  const auto grad_shape = grad_type.getShape();
  const auto target_shape = target_type.getShape();
  if (grad_shape == target_shape) {
    return grad_bcast;
  }
  const int64_t bcast_batch_rank = grad_type.getRank() - 2;
  const int64_t target_batch_rank = target_type.getRank() - 2;
  const int64_t rank_diff = bcast_batch_rank - target_batch_rank;
  Dimensions reduce_dims;
  Dimensions squeeze_dims;
  for (int64_t i = 0; i < bcast_batch_rank; ++i) {
    const int64_t target_idx = i - rank_diff;
    if (target_idx < 0) {
      if (grad_shape[i] != 1) {
        reduce_dims.push_back(i);
      }
      squeeze_dims.push_back(i);
    } else if (target_shape[target_idx] == 1 && grad_shape[i] != 1) {
      reduce_dims.push_back(i);
    }
  }
  mlir::MlirOp result = grad_bcast;
  if (reduce_dims == squeeze_dims) {
    if (!reduce_dims.empty()) {
      TT_ASSIGN_OR_RETURN(
          result, BuildSumShlo(result, reduce_dims, ReductionMode::kDropDims));
    }
    return result;
  }
  if (!reduce_dims.empty()) {
    TT_ASSIGN_OR_RETURN(
        result, BuildSumShlo(result, reduce_dims, ReductionMode::kKeepDims));
  }
  if (!squeeze_dims.empty()) {
    TT_ASSIGN_OR_RETURN(result, Squeeze(result, squeeze_dims));
  }
  return result;
}

absl::StatusOr<mlir::MlirOp> EmitBatchedGradDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs, mlir::MlirOp grad, mlir::MlirOp target,
    const int64_t matrix_contract_dim, const bool has_dynamic_dims,
    const stablehlo::Precision precision) {
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  const mlir::RankedTensorType target_type = GetTensorTypeOrDie(target);
  const auto grad_shape = grad_type.getShape();
  const auto target_shape = target_type.getShape();
  const int64_t batch_rank = grad_type.getRank() - 2;
  if (!has_dynamic_dims) {
    Dimensions batch_indices;
    Dimensions contract_dims;
    const int64_t target_batch_rank = target_type.getRank() - 2;
    const int64_t rank_diff = batch_rank - target_batch_rank;
    for (int64_t i = 0; i < batch_rank; ++i) {
      const int64_t target_idx = i - rank_diff;
      if (target_idx < 0 ||
          (target_shape[target_idx] == 1 && grad_shape[i] != 1)) {
        contract_dims.push_back(i);
      } else {
        batch_indices.push_back(i);
      }
    }
    contract_dims.push_back(matrix_contract_dim);
    mlir::MlirOp result =
        EmitDotGeneral(lhs, rhs, batch_indices, batch_indices, contract_dims,
                       contract_dims, precision);
    if (GetTensorTypeOrDie(result).getShape() != target_shape) {
      return stablehlo::Reshape(result, target_shape);
    }
    return result;
  }
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  mlir::MlirOp grad_bcast =
      EmitDotGeneral(lhs, rhs, batch_indices, batch_indices,
                     /*lhs_contracting=*/{matrix_contract_dim},
                     /*rhs_contracting=*/{matrix_contract_dim}, precision);
  return ReduceBroadcastedBatchDims(grad_bcast, target);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildMatmulShlo(
    mlir::MlirOp lhs, mlir::MlirOp rhs, const stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);
  const int64_t lhs_rank = lhs_type.getRank();
  const int64_t rhs_rank = rhs_type.getRank();
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank constraint is checked before MLIR
                 // lowering.
      lhs_rank >= 1 && rhs_rank >= 1, error::kInvalidArgument)
      << "both arguments to matmul need to be at least 1D, but they are "
      << lhs_rank << "D and " << rhs_rank << "D";

  // When lhs is 1D or rhs is <= 2D, DotGeneral directly produces the exact
  // output shape without any batch broadcasting or reshaping.
  if (lhs_rank == 1 || rhs_rank <= 2) {
    const int64_t lhs_contract_dim = lhs_rank - 1;
    const int64_t rhs_contract_dim = std::max<int64_t>(0, rhs_rank - 2);
    return EmitDotGeneral(lhs, rhs, /*lhs_batching=*/{}, /*rhs_batching=*/{},
                          /*lhs_contracting=*/{lhs_contract_dim},
                          /*rhs_contracting=*/{rhs_contract_dim}, precision);
  }

  // General batched case: lhs_rank >= 2 && rhs_rank >= 3.
  const stablehlo::Dimensions lhs_dims = GetDimensions(lhs);
  const stablehlo::Dimensions rhs_dims = GetDimensions(rhs);
  TT_ASSIGN_OR_RETURN(const stablehlo::Dimensions batch_dims,
                      ComputeBroadcastBatchDims(lhs_dims, rhs_dims));
  TT_ASSIGN_OR_RETURN(lhs, BroadcastMatrixBatchDims(lhs, batch_dims));
  TT_ASSIGN_OR_RETURN(rhs, BroadcastMatrixBatchDims(rhs, batch_dims));

  const int64_t batch_rank = batch_dims.size();
  const Dimensions batch_indices = MakeIotaDims(batch_rank);
  return EmitDotGeneral(lhs, rhs, batch_indices, batch_indices,
                        /*lhs_contracting=*/{batch_rank + 1},
                        /*rhs_contracting=*/{batch_rank}, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradSelfShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  if (self_type.getNumElements() == 0) {
    return MakeZeroSizedTensor(self.getBuilder(), self_type.getElementType(),
                               self_type.getShape());
  }
  if (grad_type.getNumElements() == 0 || other_type.getNumElements() == 0) {
    return MakeConstantLike(self, 0);
  }
  const int64_t self_rank = self_type.getRank();
  const int64_t other_rank = other_type.getRank();
  const int64_t grad_rank = grad_type.getRank();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp conj_other, BuildConjPhysicalShlo(other));

  if (self_rank == 1 && other_rank == 1) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_bcast, BroadcastIfNeeded(grad, self));
    return stablehlo::Mul(grad_bcast, conj_other);
  }
  if (self_rank == 1) {
    // self: [K], other: [...B, K, N], grad: [...B, N].
    // Contract all dimensions of grad with [...B, N] of conj_other.
    const Dimensions grad_contract = MakeIotaDims(grad_rank);
    Dimensions other_contract;
    other_contract.reserve(other_rank - 1);
    for (int64_t i = 0; i < other_rank - 2; ++i) {
      other_contract.push_back(i);
    }
    other_contract.push_back(other_rank - 1);
    return EmitDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                          /*rhs_batching=*/{}, grad_contract, other_contract,
                          precision);
  }
  if (other_rank == 1) {
    // self: [...B, M, K], other: [K], grad: [...B, M].
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_unsqueezed,
                        Unsqueeze(grad, grad_rank));
    TT_ASSIGN_OR_RETURN((auto [grad_bcast, other_bcast]),
                        ApplyBroadcastIfNeeded(grad_unsqueezed, conj_other));
    return stablehlo::Mul(grad_bcast, other_bcast);
  }
  if (other_rank == 2) {
    // self: [...B, M, K], other: [K, N], grad: [...B, M, N].
    return EmitDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                          /*rhs_batching=*/{},
                          /*lhs_contracting=*/{grad_rank - 1},
                          /*rhs_contracting=*/{1}, precision);
  }
  const bool has_dynamic_dims = grad_type.getNumDynamicDims() > 0 ||
                                self_type.getNumDynamicDims() > 0 ||
                                other_type.getNumDynamicDims() > 0;
  if (self_rank == 2 && !has_dynamic_dims) {
    // self: [M, K], other: [...B, K, N], grad: [...B, M, N].
    // Contract [...B] and N directly inside DotGeneral.
    const int64_t batch_rank = grad_rank - 2;
    Dimensions contract_dims = MakeIotaDims(batch_rank);
    contract_dims.push_back(batch_rank + 1);
    return EmitDotGeneral(grad, conj_other, /*lhs_batching=*/{},
                          /*rhs_batching=*/{}, contract_dims, contract_dims,
                          precision);
  }

  // General batched case: self_rank >= 2 && other_rank >= 3.
  const stablehlo::Dimensions grad_dims = GetDimensions(grad);
  const stablehlo::Dimensions batch_dims(grad_dims.begin(),
                                         grad_dims.end() - 2);
  TT_ASSIGN_OR_RETURN(conj_other,
                      BroadcastMatrixBatchDims(conj_other, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  return EmitBatchedGradDotGeneral(grad, conj_other, grad, self, batch_rank + 1,
                                   has_dynamic_dims, precision);
}

absl::StatusOr<mlir::MlirOp> BuildMatmulGradOtherShlo(
    mlir::MlirOp grad, mlir::MlirOp self, mlir::MlirOp other,
    const stablehlo::Precision precision) {
  const mlir::RankedTensorType self_type = GetTensorTypeOrDie(self);
  const mlir::RankedTensorType other_type = GetTensorTypeOrDie(other);
  const mlir::RankedTensorType grad_type = GetTensorTypeOrDie(grad);
  if (other_type.getNumElements() == 0) {
    return MakeZeroSizedTensor(other.getBuilder(), other_type.getElementType(),
                               other_type.getShape());
  }
  if (grad_type.getNumElements() == 0 || self_type.getNumElements() == 0) {
    return MakeConstantLike(other, 0);
  }
  const int64_t self_rank = self_type.getRank();
  const int64_t other_rank = other_type.getRank();
  const int64_t grad_rank = grad_type.getRank();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp conj_self, BuildConjPhysicalShlo(self));

  if (self_rank == 1 && other_rank == 1) {
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_bcast,
                        BroadcastIfNeeded(grad, other));
    return stablehlo::Mul(conj_self, grad_bcast);
  }
  if (self_rank == 1) {
    // self: [K], other: [...B, K, N], grad: [...B, N].
    TT_ASSIGN_OR_RETURN(mlir::MlirOp self_unsqueezed, Unsqueeze(conj_self, 1));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp grad_unsqueezed,
                        Unsqueeze(grad, grad_rank - 1));
    TT_ASSIGN_OR_RETURN(
        (auto [self_bcast, grad_bcast]),
        ApplyBroadcastIfNeeded(self_unsqueezed, grad_unsqueezed));
    return stablehlo::Mul(self_bcast, grad_bcast);
  }
  if (other_rank == 1) {
    // self: [...B, M, K], other: [K], grad: [...B, M].
    const Dimensions contract_dims = MakeIotaDims(grad_rank);
    return EmitDotGeneral(conj_self, grad, /*lhs_batching=*/{},
                          /*rhs_batching=*/{}, contract_dims, contract_dims,
                          precision);
  }
  const bool has_dynamic_dims = grad_type.getNumDynamicDims() > 0 ||
                                self_type.getNumDynamicDims() > 0 ||
                                other_type.getNumDynamicDims() > 0;
  if (other_rank == 2 && (self_rank == 2 || !has_dynamic_dims)) {
    // self: [...B, M, K], other: [K, N], grad: [...B, M, N].
    // Contract [...B, M] directly inside DotGeneral.
    const Dimensions contract_dims = MakeIotaDims(self_rank - 1);
    return EmitDotGeneral(conj_self, grad, /*lhs_batching=*/{},
                          /*rhs_batching=*/{}, contract_dims, contract_dims,
                          precision);
  }

  // General batched case: self_rank >= 2 && other_rank >= 2.
  const stablehlo::Dimensions grad_dims = GetDimensions(grad);
  const stablehlo::Dimensions batch_dims(grad_dims.begin(),
                                         grad_dims.end() - 2);
  TT_ASSIGN_OR_RETURN(conj_self,
                      BroadcastMatrixBatchDims(conj_self, batch_dims));
  const int64_t batch_rank = batch_dims.size();
  return EmitBatchedGradDotGeneral(conj_self, grad, grad, other, batch_rank,
                                   has_dynamic_dims, precision);
}

}  // namespace torch_tpu
