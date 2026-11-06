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

#include "csrc/ops/linalg/vector_norm/pnorm.h"

#include <cmath>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/binary.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/reductions/reductions.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

absl::StatusOr<mlir::MlirOp> ApplyNormPower(mlir::MlirOp op,
                                            mlir::MlirOp ord_op, double ord,
                                            bool is_root) {
  const double abs_ord = std::abs(ord);
  const mlir::ElementType dtype = GetElementTypeOrDie(op);
  const mlir::MlirOp one = MakeScalarConstant(op.getBuilder(), 1.0, dtype);
  if (abs_ord == 1.0) {
    return ord < 0.0 ? BuildDivShlo(one, op) : op;
  }
  if (abs_ord == 2.0) {
    op = is_root ? mlir::stablehlo::Sqrt(op) : mlir::stablehlo::Mul(op, op);
    return ord < 0.0 ? BuildDivShlo(one, op) : op;
  }
  TT_ASSIGN_OR_RETURN(ord_op, CastIfNeeded(ord_op, dtype));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp exp,
                      is_root ? BuildDivShlo(one, ord_op) : ord_op);
  return BuildPowShlo(op, exp);
}

bool HasZeroReductionDim(mlir::RankedTensorType type,
                         absl::Span<const int64_t> reduce_dims) {
  llvm::ArrayRef<int64_t> shape = type.getShape();
  const int64_t rank = static_cast<int64_t>(shape.size());
  for (int64_t dim : reduce_dims) {
    int64_t canonical_dim = dim < 0 ? dim + rank : dim;
    if (canonical_dim >= 0 && canonical_dim < rank &&
        shape[canonical_dim] == 0) {
      return true;
    }
  }
  return false;
}

// GENERIC: semantic 1-element Lp norm reduces to absolute value.
[[nodiscard]] bool IsSingleElementReduction(
    mlir::RankedTensorType type, absl::Span<const int64_t> reduce_dims) {
  const int64_t rank = type.getRank();
  if (rank == 0) {
    return true;
  }
  if (reduce_dims.empty()) {
    return false;
  }
  llvm::ArrayRef<int64_t> shape = type.getShape();
  for (int64_t dim : reduce_dims) {
    int64_t canonical_dim = dim < 0 ? dim + rank : dim;
    if (canonical_dim < 0 || canonical_dim >= rank ||
        shape[canonical_dim] != 1) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<mlir::MlirOp> BuildZeroNormShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type,
    mlir::ElementType compute_dtype) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::Type compute_mlir_type =
      mlir::getElementType(builder.getContext(), compute_dtype);
  mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, compute_dtype);

  if (IsComplexType(GetTensorTypeOrDie(input_op))) {
    input_op = mlir::stablehlo::Abs(input_op);
  }
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));
  mlir::MlirOp zero_bcast = mlir::stablehlo::BroadcastInDim(
      GetTensorTypeOrDie(input_op), init_val, {});
  mlir::MlirOp ne_op = mlir::stablehlo::Compare(
      input_op, zero_bcast, mlir::stablehlo::ComparisonDirection::NE);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp ne_cast, CastIfNeeded(ne_op, compute_dtype));
  const auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
  };
  mlir::MlirOp reduction_res =
      BuildReductionShlo(ne_cast, reduce_dims, compute_mlir_type, init_val,
                         reduce_fn, reduction_mode);
  return CastIfNeeded(reduction_res, out_type);
}

// GENERIC: semantic 1-element Lp vector norm reduces to absolute value.
absl::StatusOr<mlir::MlirOp> BuildSingleElementPNormShlo(
    mlir::MlirOp input_op, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type) {
  mlir::MlirOp abs_op = mlir::stablehlo::Abs(input_op);
  TT_ASSIGN_OR_RETURN(abs_op, CastIfNeeded(abs_op, out_type));
  if (reduction_mode == ReductionMode::kDropDims &&
      GetTensorTypeOrDie(abs_op).getRank() > 0) {
    return Squeeze(abs_op, reduce_dims);
  }
  return abs_op;
}

absl::StatusOr<mlir::MlirOp> BuildInfNormShlo(
    mlir::MlirOp input_op, double ord, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type,
    mlir::ElementType compute_dtype) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::Type compute_mlir_type =
      mlir::getElementType(builder.getContext(), compute_dtype);
  const bool is_pos_inf = (ord == INFINITY);
  mlir::MlirOp init_val =
      MakeScalarConstant(builder, is_pos_inf ? 0.0 : INFINITY, compute_dtype);
  auto reduce_fn = [compute_mlir_type, is_pos_inf](mlir::RegionBuilder& rb) {
    if (is_pos_inf) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    } else {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::MinOp>(
          compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
    }
  };
  mlir::MlirOp reduction_res =
      BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                         reduce_fn, reduction_mode);
  return CastIfNeeded(reduction_res, out_type);
}

absl::StatusOr<mlir::MlirOp> BuildGeneralPNormShlo(
    mlir::MlirOp input_op, mlir::MlirOp ord_op, double ord,
    absl::Span<const int64_t> reduce_dims, ReductionMode reduction_mode,
    mlir::ElementType out_type, mlir::ElementType compute_dtype) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  const mlir::Type compute_mlir_type =
      mlir::getElementType(builder.getContext(), compute_dtype);
  mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, compute_dtype);
  mlir::MlirOp orig_abs_input = input_op;
  auto reduce_fn = [compute_mlir_type](mlir::RegionBuilder& rb) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::AddOp>(
        compute_mlir_type, rb.getRegion(), rb.getOpBuilder());
  };

  if (std::isnan(ord) &&
      HasZeroReductionDim(GetTensorTypeOrDie(input_op), reduce_dims)) {
    mlir::MlirOp reduction_res =
        BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                           reduce_fn, reduction_mode);
    return CastIfNeeded(reduction_res, out_type);
  }

  TT_ASSIGN_OR_RETURN(input_op,
                      ApplyNormPower(input_op, ord_op, ord, /*is_root=*/false));

  mlir::MlirOp reduction_res =
      BuildReductionShlo(input_op, reduce_dims, compute_mlir_type, init_val,
                         reduce_fn, reduction_mode);

  TT_ASSIGN_OR_RETURN(reduction_res, ApplyNormPower(reduction_res, ord_op, ord,
                                                    /*is_root=*/true));

  if (ord < 0.0) {
    // If ord < 0 and any element along reduce_dims is 0, pow(0, ord) = inf,
    // which may not reduce back to 0 cleanly; explicitly select 0.
    mlir::MlirOp zero_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(orig_abs_input), init_val, {});
    mlir::MlirOp is_zero = mlir::stablehlo::Compare(
        orig_abs_input, zero_bcast, mlir::stablehlo::ComparisonDirection::EQ);
    const mlir::Type pred_mlir_type =
        mlir::getElementType(builder.getContext(), mlir::ElementType::PRED);
    mlir::MlirOp false_val =
        MakeScalarConstant(builder, false, mlir::ElementType::PRED);
    const auto or_reduce_fn = [pred_mlir_type](mlir::RegionBuilder& rb) {
      mlir::stablehlo::buildReduceBody<mlir::stablehlo::OrOp>(
          pred_mlir_type, rb.getRegion(), rb.getOpBuilder());
    };
    mlir::MlirOp has_zero =
        BuildReductionShlo(is_zero, reduce_dims, pred_mlir_type, false_val,
                           or_reduce_fn, reduction_mode);
    mlir::MlirOp zero_res = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(reduction_res), init_val, {});
    reduction_res = mlir::stablehlo::Select(has_zero, zero_res, reduction_res);
  }

  return CastIfNeeded(reduction_res, out_type);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildPNormShlo(
    mlir::MlirOp input_op, double ord, absl::Span<const int64_t> reduce_dims,
    ReductionMode reduction_mode, mlir::ElementType out_type) {
  mlir::MlirBuilder& builder = input_op.getBuilder();
  mlir::MlirOp ord_op = MakeScalarConstant(builder, ord, out_type);
  return BuildPNormShlo(input_op, ord_op, ord, reduce_dims, reduction_mode,
                        out_type);
}

absl::StatusOr<mlir::MlirOp> BuildPNormShlo(
    mlir::MlirOp input_op, mlir::MlirOp ord_op, double ord,
    absl::Span<const int64_t> reduce_dims, ReductionMode reduction_mode,
    mlir::ElementType out_type) {
  TT_ASSIGN_OR_RETURN(const mlir::ElementType compute_dtype,
                      InferComputationDtype(out_type));

  if (ord == 0.0) {
    return BuildZeroNormShlo(input_op, reduce_dims, reduction_mode, out_type,
                             compute_dtype);
  }

  // GENERIC: semantic 1-element Lp vector norm reduces to absolute value.
  if (ord != 0.0 &&
      IsSingleElementReduction(GetTensorTypeOrDie(input_op), reduce_dims)) {
    return BuildSingleElementPNormShlo(input_op, reduce_dims, reduction_mode,
                                       out_type);
  }

  input_op = mlir::stablehlo::Abs(input_op);
  TT_ASSIGN_OR_RETURN(input_op, CastIfNeeded(input_op, compute_dtype));

  if (std::isinf(ord)) {
    return BuildInfNormShlo(input_op, ord, reduce_dims, reduction_mode,
                            out_type, compute_dtype);
  }

  return BuildGeneralPNormShlo(input_op, ord_op, ord, reduce_dims,
                               reduction_mode, out_type, compute_dtype);
}

}  // namespace torch_tpu
