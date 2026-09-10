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

#include "csrc/ops/linalg/linalg_svd_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/ScalarType.h"
#include "c10/util/string_view.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/all_any/all_any.h"
#include "csrc/ops/eye/eye_lib.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/linalg/qr/qr_lib.h"
#include "csrc/ops/linalg/vector_norm/pnorm.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/nullary_aten_kernels.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/sort/sort.h"
#include "csrc/ops/unary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"

namespace torch_tpu {
namespace {

// Disambiguate the `Transpose` enum from the free `Transpose` op builder
// function in the `mlir::stablehlo` namespace.
using StablehloTranspose = enum mlir::stablehlo::Transpose;

template <typename T, typename... Dims>
Dimensions ConcatBatchAndDims(const T& batch_dims, Dims... dims) {
  Dimensions shape(batch_dims.begin(), batch_dims.end());
  (shape.push_back(dims), ...);
  return shape;
}

absl::StatusOr<mlir::MlirOp> BuildTransposeShlo(mlir::MlirOp op) {
  const int64_t rank = GetTensorTypeOrDie(op).getRank();
  Indices permutation = GetAllDimensions(op);
  std::swap(permutation[rank - 2], permutation[rank - 1]);

  if (GetElementTypeOrDie(op) != mlir::ElementType::COMPLEXF64) {
    return mlir::stablehlo::Transpose(op, permutation);
  }

  mlir::MlirOp real = mlir::stablehlo::Real(op);
  mlir::MlirOp imag = mlir::stablehlo::Imag(op);
  mlir::MlirOp real_t = mlir::stablehlo::Transpose(real, permutation);
  mlir::MlirOp imag_t = mlir::stablehlo::Transpose(imag, permutation);
  return mlir::stablehlo::Complex(real_t, imag_t);
}

absl::StatusOr<mlir::MlirOp> BuildConjugateTransposeShlo(mlir::MlirOp op) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp transposed, BuildTransposeShlo(op));
  return BuildConjPhysicalShlo(transposed);
}

absl::StatusOr<mlir::MlirOp> BuildMatMulShlo(mlir::MlirOp lhs,
                                             mlir::MlirOp rhs) {
  // Batch dimensions should be the same for both lhs and rhs.
  const Indices all_dims = GetAllDimensions(lhs);
  const auto batch_dims = llvm::ArrayRef<int64_t>(all_dims).drop_back(2);
  const int64_t lhs_rank = GetTensorTypeOrDie(lhs).getRank();
  const int64_t rhs_rank = GetTensorTypeOrDie(rhs).getRank();

  const auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &lhs.getContext(),
      /*lhs_batching_dimensions=*/batch_dims,
      /*rhs_batching_dimensions=*/batch_dims,
      /*lhs_contracting_dimensions=*/{lhs_rank - 1},
      /*rhs_contracting_dimensions=*/{rhs_rank - 2});

  return mlir::stablehlo::DotGeneral(lhs, rhs, dot_dims);
}

mlir::stablehlo::CustomCallOp BuildCustomCallOp(
    mlir::MlirBuilder& builder, llvm::StringRef call_target_name,
    llvm::ArrayRef<mlir::Type> result_types,
    llvm::ArrayRef<mlir::Value> operands, llvm::StringRef backend_config = "") {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();

  std::vector<mlir::NamedAttribute> attributes;
  attributes.reserve(3 + (backend_config.empty() ? 0 : 1));
  attributes.push_back(op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr(call_target_name)));
  attributes.push_back(op_builder.getNamedAttr("has_side_effect",
                                               op_builder.getBoolAttr(false)));
  attributes.push_back(op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_ORIGINAL)));
  if (!backend_config.empty()) {
    attributes.push_back(op_builder.getNamedAttr(
        "backend_config", op_builder.getStringAttr(backend_config)));
  }

  const mlir::Location loc = builder.getLoc();
  return mlir::stablehlo::CustomCallOp::create(op_builder, loc, result_types,
                                               operands, attributes);
}

absl::StatusOr<MlirOpResults<2>> BuildEighShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(input);

  TT_ASSIGN_OR_RETURN(const mlir::ElementType a_elem_type,
                      ConvertTo<mlir::ElementType>(a_type.getElementType()));
  const mlir::ElementType w_elem_type = RealComponentOf(a_elem_type);

  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::Type w_mlir_type =
      mlir::getElementType(builder.getContext(), w_elem_type);

  const llvm::ArrayRef<int64_t> a_shape = a_type.getShape();
  const auto w_type =
      mlir::RankedTensorType::get(a_shape.drop_back(1), w_mlir_type);

  // We use similar values as `jax.lax.eigh`:
  // lower=true(1), sort_eigenvalues=false(0), max_iter=100, tol=1e-6 for single
  // precision, 1e-12 for double precision.
  const char* kBackendConfig = (w_elem_type == mlir::ElementType::F64)
                                   ? "1,0,100,1e-12"
                                   : "1,0,100,1e-6";
  mlir::stablehlo::CustomCallOp eigh_op =
      BuildCustomCallOp(builder, "Eigh",
                        /*result_types=*/{a_type, w_type},
                        /*operands=*/{input.getValue()}, kBackendConfig);

  return {{mlir::MlirOp(builder, eigh_op.getResult(0)),
           mlir::MlirOp(builder, eigh_op.getResult(1))}};
}

absl::StatusOr<mlir::MlirOp> ConvertToComplexIfNeeded(mlir::MlirOp scalar,
                                                      mlir::MlirOp target) {
  const mlir::RankedTensorType target_type = GetTensorTypeOrDie(target);
  if (!llvm::isa<mlir::ComplexType>(target_type.getElementType())) {
    return scalar;
  }

  const mlir::RankedTensorType scalar_type = GetTensorTypeOrDie(scalar);
  if (llvm::isa<mlir::ComplexType>(scalar_type.getElementType())) {
    return scalar;
  }

  mlir::MlirBuilder& builder = scalar.getBuilder();
  const mlir::MlirOp zero = MakeConstantLike(scalar, 0.0);
  const auto complex_type = mlir::RankedTensorType::get(
      scalar_type.getShape(), target_type.getElementType());
  return mlir::MlirOp(
      builder, mlir::stablehlo::ComplexOp::create(
                   builder.getOpBuilder(), builder.getLoc(), complex_type,
                   scalar.getValue(), zero.getValue())
                   .getResult());
}

// Coefficients for the QDWH (QR-based Dynamically Weighted Halley) iteration.
struct QdwhCoefs {
  // Coefficient 'a' used in the Halley iteration formula.
  std::vector<double> a;
  // Coefficient 'b' used to compute the scaling factor e = b / c.
  std::vector<double> b;
  // Coefficient 'c' used in the denominator of the Halley iteration.
  std::vector<double> c;
  // Indicates whether to use Cholesky decomposition (1) or QR decomposition (0)
  // to solve the linear system at each iteration.
  //
  // We use std::vector<int8_t> instead of std::vector<bool> because
  // std::vector<bool> is bit-packed and cannot provide a contiguous pointer
  // (via .data()) which is required to interface with MLIR's ArrayRef.
  std::vector<int8_t> use_cholesky;
};

// Precomputes the QDWH coefficients. This matches JAX's behavior which also
// precomputes these during compilation.
//
// Adapted from JAX's TPU QDWH implementation in linalg/qdwh.py.
QdwhCoefs ComputeQdwhCoefs(double epsilon, int64_t max_iterations) {
  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  // Indicates whether to use Cholesky decomposition (1) or QR decomposition (0)
  // to solve the linear system at each iteration.
  //
  // We use std::vector<int8_t> instead of std::vector<bool> because
  // std::vector<bool> is bit-packed and cannot provide a contiguous pointer
  // (via .data()) which is required to interface with MLIR's ArrayRef.
  std::vector<int8_t> use_cholesky;

  a.reserve(max_iterations);
  b.reserve(max_iterations);
  c.reserve(max_iterations);
  use_cholesky.reserve(max_iterations);

  double l = epsilon;
  const double l_tolerance = 10.0 * epsilon / 2.0;
  for (int64_t k = 0; k < max_iterations; ++k) {
    if (l + l_tolerance >= 1.0) {
      a.push_back(3.0);
      b.push_back(1.0);
      c.push_back(3.0);
      use_cholesky.push_back(1);
    } else {
      const double l2 = l * l;
      const double dd = std::pow(4.0 * (1.0 / l2 - 1.0) / l2, 1.0 / 3.0);
      const double sqd = std::sqrt(1.0 + dd);
      const double a_val =
          sqd + std::sqrt(2.0 - dd + 2.0 * (2.0 - l2) / (l2 * sqd));
      const double b_val = (a_val - 1.0) * (a_val - 1.0) / 4.0;
      const double c_val = a_val + b_val - 1.0;

      a.push_back(a_val);
      b.push_back(b_val);
      c.push_back(c_val);
      use_cholesky.push_back(c_val <= 100.0 ? 1 : 0);

      l = l * (a_val + b_val * l2) / (1.0 + c_val * l2);
    }
  }
  return {a, b, c, use_cholesky};
}

// Extracts a scalar (0-rank) tensor from a 1D tensor of coefficients at a
// dynamic index.
absl::StatusOr<mlir::MlirOp> GetScalarAtIndex(mlir::MlirBuilder& builder,
                                              mlir::MlirOp consts,
                                              mlir::MlirOp index,
                                              mlir::Type mlir_type) {
  const auto slice_type = mlir::RankedTensorType::get({1}, mlir_type);
  const mlir::MlirOp slice_op = builder.create<mlir::stablehlo::DynamicSliceOp>(
      slice_type, consts.getValue(), mlir::ValueRange{index.getValue()},
      builder.getOpBuilder().getDenseI64ArrayAttr({1}));
  const auto scalar_type = mlir::RankedTensorType::get({}, mlir_type);
  const mlir::MlirOp reshape_op = builder.create<mlir::stablehlo::ReshapeOp>(
      scalar_type, slice_op.getValue());
  return reshape_op;
}

// Computes the maximum reduction along specified dimensions.
absl::StatusOr<mlir::MlirOp> BuildMaxReduction(mlir::MlirOp input,
                                               absl::Span<const int64_t> dims) {
  const mlir::Type element_type = GetElementTypeOrSelf(input);
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::MlirOp init_val = MakeScalarConstant(builder, 0.0, element_type);
  const auto reduce_fn = [element_type](mlir::RegionBuilder& body) {
    mlir::stablehlo::buildReduceBody<mlir::stablehlo::MaxOp>(
        element_type, body.getRegion(), body.getOpBuilder());
  };
  return BuildReductionShlo(input, dims, element_type, init_val, reduce_fn);
}

// Computes the maximum absolute column sum (1-norm) of a matrix, preserving
// batch dims.
absl::StatusOr<mlir::MlirOp> BuildMatrix1Norm(mlir::MlirOp x) {
  // Sum along rows (dimension rank-2) to get column sums.
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp abs_x, BuildAbsShlo(x));
  const int64_t rank = GetTensorTypeOrDie(x).getRank();
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp col_sums,
                      BuildSumShlo(abs_x, {rank - 2}));
  // Max over the columns (dimension rank-2 after reduction) to get 1-norm.
  return BuildMaxReduction(col_sums, {rank - 2});
}

// Computes the maximum absolute row sum (inf-norm) of a matrix, preserving
// batch dims.
absl::StatusOr<mlir::MlirOp> BuildMatrixInfNorm(mlir::MlirOp x) {
  // Sum along columns (dimension rank-1) to get row sums.
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp abs_x, BuildAbsShlo(x));
  const int64_t rank = GetTensorTypeOrDie(x).getRank();
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp row_sums,
                      BuildSumShlo(abs_x, {rank - 1}));
  // Max over the rows (dimension rank-2 after reduction) to get inf-norm.
  return BuildMaxReduction(row_sums, {rank - 2});
}

absl::StatusOr<mlir::MlirOp> BroadcastScalar(
    mlir::MlirOp scalar, llvm::ArrayRef<int64_t> target_shape) {
  return Broadcast(scalar, target_shape, {});
}

absl::StatusOr<mlir::MlirOp> BuildScaleTensor(mlir::MlirOp tensor,
                                              mlir::MlirOp scale) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp scale_val,
                      ConvertToComplexIfNeeded(scale, tensor));
  const mlir::RankedTensorType tensor_type = GetTensorTypeOrDie(tensor);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scale_bcast,
                      BroadcastScalar(scale_val, tensor_type.getShape()));
  return mlir::stablehlo::Mul(scale_bcast, tensor);
}

// Helper to construct stablehlo::TriangularSolveOp.
// This is needed in QDWH iteration to compute the next iteration step.
absl::StatusOr<mlir::MlirOp> BuildTriangularSolveShlo(
    mlir::MlirOp a, mlir::MlirOp b, bool left_side, bool lower,
    StablehloTranspose transpose_a, bool unit_diagonal) {
  mlir::MlirBuilder& builder = a.getBuilder();
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::stablehlo::TransposeAttr transpose_attr =
      mlir::stablehlo::TransposeAttr::get(&builder.getContext(), transpose_a);
  const mlir::MlirOp solve_op =
      builder.create<mlir::stablehlo::TriangularSolveOp>(
          GetTensorTypeOrDie(b), a.getValue(), b.getValue(),
          op_builder.getBoolAttr(left_side), op_builder.getBoolAttr(lower),
          op_builder.getBoolAttr(unit_diagonal), transpose_attr);
  return solve_op;
}

// Helper to construct Eye(N) and broadcast to (batch_dims..., N, N)
absl::StatusOr<mlir::MlirOp> BuildBatchedEye(
    mlir::MlirBuilder& builder, mlir::ElementType element_type, int64_t n,
    llvm::ArrayRef<int64_t> batch_shape) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp eye,
                      BuildEyeShlo(builder, element_type, n, n));
  const Dimensions target_shape = ConcatBatchAndDims(batch_shape, n, n);
  const int64_t rank = target_shape.size();
  return Broadcast(eye, target_shape, {rank - 2, rank - 1});
}

// Performs a single QDWH iteration step using QR decomposition.
//
// The calculation steps are:
//   1. scaled_u = sqrt(c) * U
//   2. Y = [scaled_u; I] (vertical concatenation)
//   3. Q = QR(Y) (reduced)
//   4. Partition Q into [Q1; Q2] (top M rows and bottom N rows)
//   5. U_next = e * U + ((a - e) / sqrt(c)) * Q1 @ Q2.H
//
// Input U is the current polar estimate (shape ..., M, N).
// Scalars e (b/c), a_minus_e (a-e), and c are derived from precomputed
// QDWH coefficients for the current loop iteration.
absl::StatusOr<mlir::MlirOp> BuildQdwhIterationQR(mlir::MlirOp u,
                                                  mlir::MlirOp e,
                                                  mlir::MlirOp a_minus_e,
                                                  mlir::MlirOp c) {
  // sqrt_c = Sqrt(c)
  mlir::MlirOp sqrt_c = mlir::stablehlo::Sqrt(c);
  // scaled_u = sqrt(c) * U  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp scaled_u, BuildScaleTensor(u, sqrt_c));

  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u);
  const int64_t rank = u_type.getRank();
  const llvm::ArrayRef<int64_t> u_shape = u_type.getShape();
  const int64_t n = u_shape[rank - 1];
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(u_type.getElementType()));
  mlir::MlirBuilder& builder = u.getBuilder();
  // eye_bcast = Eye(N) broadcasted to batch shape  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp eye_bcast,
      BuildBatchedEye(builder, element_type, n, u_shape.drop_back(2)));

  // Y = Concat([scaled_u, eye_bcast], axis=rank-2)
  // This constructs the matrix Y = [sqrt(c)*U; I] (shape: ..., M+N, N)
  const mlir::MlirOp y = builder.create<mlir::stablehlo::ConcatenateOp>(
      mlir::ValueRange{scaled_u.getValue(), eye_bcast.getValue()},
      builder.getOpBuilder().getI64IntegerAttr(rank - 2));

  // Q, _ = QR(Y)
  // We only need Q (reduced QR, shape: ..., M+N, N)
  TT_ASSIGN_OR_RETURN(const MlirOpResults<2> qr_results,
                      BuildQrShlo(y, "reduced"));
  const mlir::MlirOp q = qr_results[0];

  // Slice Q1 = Q[..., :m, :] and Q2 = Q[..., m:, :]
  // Q1 has shape: ..., M, N, Q2 has shape: ..., N, N
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp q1,
                      BuildMaybeSlice(q, rank - 2, 0, -n));
  const int64_t m = u_shape[rank - 2];
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp q2,
                      BuildMaybeSlice(q, rank - 2, -m, 0));

  // Q2_H = conj(Q2.T) (conjugate transpose)  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp q2_h, BuildConjugateTransposeShlo(q2));

  // q1_q2_h = Q1 @ Q2_H  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp q1_q2_h, BuildMatMulShlo(q1, q2_h));

  // a_minus_e_by_sqrt_c = a_minus_e / sqrt_c
  const mlir::MlirOp a_minus_e_by_sqrt_c =
      mlir::stablehlo::Div(a_minus_e, sqrt_c);

  // term2 represents the second term of the QDWH update formula, computed as:
  // (a_minus_e / sqrt(c)) * (Q1 @ Q2.H)  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp term2,
                      BuildScaleTensor(q1_q2_h, a_minus_e_by_sqrt_c));

  // e_u = e * U  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp e_u, BuildScaleTensor(u, e));

  return mlir::stablehlo::Add(e_u, term2);
}

// Performs a single QDWH iteration step using Cholesky decomposition.
// The calculation steps are:
//   1. X = c * U.H @ U + I
//   2. L = Cholesky(X) (lower triangular, X = L @ L.H)
//   3. Solve L @ L.H @ Z2 = U.H for Z2 using two triangular solves:
//      a. Solve conj(L) @ Z1 = U.T
//      b. Solve L.H @ Z2 = conj(Z1)
//   4. Z = Z2.H
//   5. U_next = e * U + (a - e) * Z
// Input U is the current polar estimate (shape ..., M, N).
// Scalars e (b/c), a_minus_e (a-e), and c are derived from precomputed
// QDWH coefficients for the current loop iteration.
absl::StatusOr<mlir::MlirOp> BuildQdwhIterationCholesky(mlir::MlirOp u,
                                                        mlir::MlirOp e,
                                                        mlir::MlirOp a_minus_e,
                                                        mlir::MlirOp c) {
  // U_H = conj(U.T) (conjugate transpose)  (shape: ..., N, M)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_h, BuildConjugateTransposeShlo(u));

  // U_H_U = U_H @ U  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_h_u, BuildMatMulShlo(u_h, u));

  // scaled_u_h_u = c * U_H @ U  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp scaled_u_h_u, BuildScaleTensor(u_h_u, c));

  mlir::MlirBuilder& builder = u.getBuilder();
  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u);
  const int64_t rank = u_type.getRank();
  const llvm::ArrayRef<int64_t> u_shape = u_type.getShape();
  const int64_t n = u_shape[rank - 1];

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(u_type.getElementType()));
  // eye_bcast = Eye(N) broadcasted to batch shape  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp eye_bcast,
      BuildBatchedEye(builder, element_type, n, u_shape.drop_back(2)));

  // X = c * (U.H @ U) + Eye(N)  (shape: ..., N, N)
  const mlir::MlirOp x = mlir::stablehlo::Add(scaled_u_h_u, eye_bcast);

  // Computes the Cholesky decomposition of X.
  // Since we request the lower triangular factor (lower = true), this computes
  // Y such that X = Y @ Y.H.
  // (shape of Y: ..., N, N)
  const mlir::MlirOp y = builder.create<mlir::stablehlo::CholeskyOp>(
      x.getValue(), /*lower=*/builder.getOpBuilder().getBoolAttr(true));

  // y_conj = conj(Y)  (shape: ..., N, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp y_conj, BuildConjPhysicalShlo(y));

  // Solve 1: Solve conj(Y) @ Z1 = U.T  (Z1 shape: ..., N, M)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_t, BuildTransposeShlo(u));
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp z1,
      BuildTriangularSolveShlo(y_conj, u_t, /*left_side=*/true, /*lower=*/true,
                               StablehloTranspose::NO_TRANSPOSE,
                               /*unit_diagonal=*/false));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp z1_conj, BuildConjPhysicalShlo(z1));

  // Solve 2: Solve Y.H @ Z2 = z1_conj  (Z2 shape: ..., N, M)
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp z2,
      BuildTriangularSolveShlo(y, z1_conj, /*left_side=*/true, /*lower=*/true,
                               StablehloTranspose::ADJOINT,
                               /*unit_diagonal=*/false));

  // z = conj(Z2.T)  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp z, BuildConjugateTransposeShlo(z2));

  // term2 represents the second term of the QDWH update formula:
  // (a_k - e_k) * U_k * (c_k * U_k.H @ U_k + I)^-1
  // here computed as: (a_k - e_k) * Z (where Z is computed via triangular
  // solves)
  // (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp term2, BuildScaleTensor(z, a_minus_e));

  // e_u = e * U  (shape: ..., M, N)
  TT_ASSIGN_OR_RETURN(mlir::MlirOp e_u, BuildScaleTensor(u, e));

  return mlir::stablehlo::Add(e_u, term2);
}

// Returns the epsilon for a given floating-point element type.
double GetEpsilon(mlir::ElementType type) {
  switch (type) {
    case mlir::ElementType::F64:
      return std::numeric_limits<double>::epsilon();
    case mlir::ElementType::F32:
      return std::numeric_limits<float>::epsilon();
    case mlir::ElementType::BF16:
      return static_cast<double>(std::numeric_limits<c10::BFloat16>::epsilon());
    case mlir::ElementType::F16:
      return static_cast<double>(std::numeric_limits<c10::Half>::epsilon());
    default:
      return std::numeric_limits<float>::epsilon();
  }
}

// If the input norm is zero, returns 1.0. Otherwise, returns alpha_inverse.
absl::StatusOr<mlir::MlirOp> BuildSafeAlphaInverse(mlir::MlirOp one_norm,
                                                   mlir::MlirOp alpha_inverse) {
  mlir::MlirOp zero = MakeConstantLike(one_norm, 0.0);
  mlir::MlirOp one_norm_is_zero = mlir::stablehlo::Compare(
      one_norm, zero, mlir::stablehlo::ComparisonDirection::EQ);
  mlir::MlirOp one = MakeConstantLike(one_norm, 1.0);
  return mlir::stablehlo::Select(one_norm_is_zero, one, alpha_inverse);
}

// Calculates the scaling factor:
//   alpha = sqrt(||A||_1 * ||A||_inf)
// and returns:
//   A_scaled = A / alpha = A * (rsqrt(||A||_1) * rsqrt(||A||_inf))
//
// If ||A||_1 == 0, scales by 1.0.
absl::StatusOr<mlir::MlirOp> BuildQdwhInitialScaling(mlir::MlirOp input) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp one_norm, BuildMatrix1Norm(input));
  mlir::MlirOp rsqrt_one = mlir::stablehlo::Rsqrt(one_norm);

  TT_ASSIGN_OR_RETURN(mlir::MlirOp inf_norm, BuildMatrixInfNorm(input));
  mlir::MlirOp rsqrt_inf = mlir::stablehlo::Rsqrt(inf_norm);

  const mlir::MlirOp alpha_inverse = mlir::stablehlo::Mul(rsqrt_one, rsqrt_inf);

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp alpha_inverse_safe,
                      BuildSafeAlphaInverse(one_norm, alpha_inverse));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp alpha_inverse_complex,
                      ConvertToComplexIfNeeded(alpha_inverse_safe, input));

  const llvm::ArrayRef<int64_t> input_shape =
      GetTensorTypeOrDie(input).getShape();
  Dimensions bcast_dimensions = GetAllDimensions(input);
  bcast_dimensions.resize(bcast_dimensions.size() - 2);
  TT_ASSIGN_OR_RETURN(
      auto alpha_inverse_complex_bcast,
      Broadcast(alpha_inverse_complex, input_shape, bcast_dimensions));

  return mlir::stablehlo::Mul(input, alpha_inverse_complex_bcast);
}

// Applies Newton-Schulz refinement to the polar factor U:
//   U_refined = 1.5 * U - 0.5 * U @ (U.H @ U)
absl::StatusOr<mlir::MlirOp> BuildNewtonSchulzRefinement(
    mlir::MlirOp u, mlir::ElementType element_type) {
  mlir::MlirBuilder& builder = u.getBuilder();
  const mlir::ElementType real_element_type = RealComponentOf(element_type);

  const mlir::MlirOp one_half =
      MakeScalarConstant(builder, 1.5, real_element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp term_1, BuildScaleTensor(u, one_half));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_h, BuildConjugateTransposeShlo(u));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_h_u, BuildMatMulShlo(u_h, u));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_u_h_u, BuildMatMulShlo(u, u_h_u));

  const mlir::MlirOp half = MakeScalarConstant(builder, 0.5, real_element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp term_2, BuildScaleTensor(u_u_h_u, half));

  return mlir::stablehlo::Subtract(term_1, term_2);
}

absl::StatusOr<mlir::MlirOp> BuildHermitianFactor(
    mlir::MlirOp u_refined, mlir::MlirOp input,
    mlir::ElementType element_type) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_refined_h,
                      BuildConjugateTransposeShlo(u_refined));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp u_h_input,
                      BuildMatMulShlo(u_refined_h, input));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp u_h_input_h,
                      BuildConjugateTransposeShlo(u_h_input));
  const mlir::MlirOp sum_h = mlir::stablehlo::Add(u_h_input, u_h_input_h);

  mlir::MlirBuilder& builder = u_refined.getBuilder();
  const mlir::ElementType real_element_type = RealComponentOf(element_type);
  const mlir::MlirOp half_refine =
      MakeScalarConstant(builder, 0.5, real_element_type);
  return BuildScaleTensor(sum_h, half_refine);
}

template <typename IterationFn>
absl::Status BuildQdwhBranch(mlir::MlirBuilder& parent_builder,
                             mlir::Region& branch_region, mlir::MlirOp parent_u,
                             mlir::MlirOp parent_e,
                             mlir::MlirOp parent_a_minus_e,
                             mlir::MlirOp parent_c, IterationFn iteration_fn) {
  mlir::RegionBuilder branch_builder(parent_builder, branch_region);
  mlir::MlirOp branch_u = mlir::swap(branch_builder, parent_u);
  mlir::MlirOp branch_e = mlir::swap(branch_builder, parent_e);
  mlir::MlirOp branch_a_minus_e = mlir::swap(branch_builder, parent_a_minus_e);
  mlir::MlirOp branch_c = mlir::swap(branch_builder, parent_c);
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp u_next,
      iteration_fn(branch_u, branch_e, branch_a_minus_e, branch_c));
  mlir::stablehlo::Return(branch_builder, {u_next});
  return absl::OkStatus();
}

// Config options and precomputed constants for the QDWH loop body.
struct QdwhLoopOptions {
  // The element type of the matrix being processed.
  mlir::ElementType element_type;
  // Tolerance value used to determine convergence of the iteration.
  double tolerance;
  // The shape of the batch dimensions of the input tensor.
  llvm::ArrayRef<int64_t> batch_shape;
  // The rank of the input tensor.
  int64_t rank;
  // Precomputed 'a' coefficients.
  mlir::MlirOp a_consts;
  // Precomputed 'b' coefficients.
  mlir::MlirOp b_consts;
  // Precomputed 'c' coefficients.
  mlir::MlirOp c_consts;
  // Precomputed boolean flags indicating whether to use Cholesky iteration.
  mlir::MlirOp use_chol_consts;
};

// Checks convergence by comparing the norm of the difference between successive
// U estimates against the tolerance.
absl::StatusOr<mlir::MlirOp> BuildQdwhConvergenceCheck(
    mlir::MlirBuilder& builder, mlir::MlirOp parent_u_next,
    mlir::MlirOp parent_u_current, mlir::MlirOp parent_active_mask,
    mlir::MlirOp use_cholesky, const QdwhLoopOptions& options) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::Location loc = builder.getLoc();

  const mlir::RankedTensorType cond_type =
      GetTensorTypeOrDie(parent_active_mask);

  // Selects convergence status. Early exit is only checked during Cholesky
  // iterations. During QR (conditioning phase), convergence is not expected,
  // so we propagate the previous status to avoid redundant norm calculations.
  //
  // This matches JAX's implementation (linalg/qdwh.py), which splits the
  // iterations into a QR loop (with convergence check disabled) and a
  // Cholesky loop (with convergence check enabled).
  auto if_op = mlir::stablehlo::IfOp::create(
      op_builder, loc, mlir::TypeRange{cond_type}, use_cholesky.getValue());

  // True branch: compute norm and check convergence.
  {
    mlir::RegionBuilder true_rb(builder, if_op->getRegion(0));
    mlir::MlirOp u_next = mlir::swap(true_rb, parent_u_next);
    mlir::MlirOp u_current = mlir::swap(true_rb, parent_u_current);

    const mlir::MlirOp diff = mlir::stablehlo::Subtract(u_next, u_current);
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp abs_diff, BuildAbsShlo(diff));

    const Indices reduce_dims = {options.rank - 2, options.rank - 1};
    TT_ASSIGN_OR_RETURN(
        const mlir::MlirOp norm_op,
        BuildPNormShlo(abs_diff, 2.0, reduce_dims, ReductionMode::kDropDims,
                       options.element_type));

    const mlir::MlirOp tolerance_op =
        MakeScalarConstant(true_rb, options.tolerance, options.element_type);
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp tolerance_bcast,
                        BroadcastScalar(tolerance_op, options.batch_shape));

    // Shape `(batch_dims...)`. True if the corresponding matrix has not
    // converged.
    const mlir::MlirOp new_active_mask(
        true_rb, mlir::stablehlo::CompareOp::create(
                     true_rb.getOpBuilder(), true_rb.getLoc(),
                     norm_op.getValue(), tolerance_bcast.getValue(),
                     mlir::stablehlo::ComparisonDirection::GT)
                     .getResult());

    mlir::stablehlo::Return(true_rb, {new_active_mask});
  }

  // False branch: propagate previous status.
  {
    mlir::RegionBuilder false_rb(builder, if_op->getRegion(1));
    const mlir::MlirOp active_mask = mlir::swap(false_rb, parent_active_mask);
    mlir::stablehlo::Return(false_rb, {active_mask});
  }

  return mlir::MlirOp(builder, if_op.getResult(0));
}

// Builds the body of the QDWH convergence loop.
//
// Computes QDWH coefficients, conditionally executes QR or Cholesky
// iterations, checks convergence, and increments the loop counter.
//
// Arguments:
//   region_builder: The builder for the loop body region.
//   i_op: The current loop iteration counter (i32 scalar).
//   u_op: The current polar factor estimate U (tensor).
//   active_mask: Boolean tensor of shape `(batch_dims...)` tracking which
//                batch elements have not yet converged (are still active).
//   options: Loop-invariant configurations and precomputed constant tensors.
//
// Returns:
//   A vector containing {i_next, u_next, active_mask_next}.
absl::StatusOr<std::vector<mlir::MlirOp>> BuildQdwhLoopBody(
    mlir::RegionBuilder& region_builder, mlir::MlirOp i_op, mlir::MlirOp u_op,
    mlir::MlirOp active_mask, const QdwhLoopOptions& options) {
  mlir::MlirBuilder& builder = region_builder;
  const mlir::Type mlir_type =
      mlir::getElementType(builder.getContext(), options.element_type);

  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp a,
      GetScalarAtIndex(builder, options.a_consts, i_op, mlir_type));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp b,
      GetScalarAtIndex(builder, options.b_consts, i_op, mlir_type));
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp c,
      GetScalarAtIndex(builder, options.c_consts, i_op, mlir_type));

  // Compute e and a_minus_e before branching.
  mlir::MlirOp e = mlir::stablehlo::Div(b, c);
  const mlir::MlirOp a_minus_e = mlir::stablehlo::Subtract(a, e);

  mlir::OpBuilder& op_builder = region_builder.getOpBuilder();
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp use_cholesky,
                      GetScalarAtIndex(builder, options.use_chol_consts, i_op,
                                       op_builder.getI1Type()));
  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u_op);
  const mlir::Location loc = region_builder.getLoc();
  auto if_op = mlir::stablehlo::IfOp::create(
      op_builder, loc, mlir::TypeRange{u_type}, use_cholesky.getValue());

  TT_RETURN_IF_ERROR(BuildQdwhBranch(builder, if_op->getRegion(0), u_op, e,
                                     a_minus_e, c, BuildQdwhIterationCholesky));
  TT_RETURN_IF_ERROR(BuildQdwhBranch(builder, if_op->getRegion(1), u_op, e,
                                     a_minus_e, c, BuildQdwhIterationQR));

  const mlir::MlirOp u_next(builder, if_op.getResult(0));

  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp active_mask_next,
      BuildQdwhConvergenceCheck(builder, u_next, u_op, active_mask,
                                use_cholesky, options));

  mlir::MlirOp one_i32 =
      MakeScalarConstant(builder, 1, builder.getOpBuilder().getI32Type());
  const mlir::MlirOp i_next = mlir::stablehlo::Add(i_op, one_i32);

  return std::vector<mlir::MlirOp>{i_next, u_next, active_mask_next};
}

absl::StatusOr<MlirOpResults<2>> BuildQdwhShlo(
    mlir::MlirOp input, int64_t max_iterations,
    std::optional<double> epsilon = std::nullopt) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType element_type,
      ConvertTo<mlir::ElementType>(input_type.getElementType()));

  const mlir::ElementType real_element_type = RealComponentOf(element_type);
  const double epsilon_val = epsilon.value_or(GetEpsilon(real_element_type));
  // Reuse the same tolerance as JAX.
  const double l_tolerance_val = 5.0 * epsilon_val;
  const double tolerance = std::cbrt(l_tolerance_val);

  // Precompute coefficients.
  const QdwhCoefs coeffs = ComputeQdwhCoefs(epsilon_val, max_iterations);

  const mlir::Type real_mlir_type =
      mlir::getElementType(builder.getContext(), real_element_type);
  const auto const_type =
      mlir::RankedTensorType::get({max_iterations}, real_mlir_type);

  const mlir::MlirOp a_consts = MakeConstant(builder, coeffs.a, const_type);
  const mlir::MlirOp b_consts = MakeConstant(builder, coeffs.b, const_type);
  const mlir::MlirOp c_consts = MakeConstant(builder, coeffs.c, const_type);

  const mlir::Type i1_type = builder.getOpBuilder().getI1Type();
  const auto use_chol_type =
      mlir::RankedTensorType::get({max_iterations}, i1_type);
  const mlir::MlirOp use_chol_consts =
      MakeConstant(builder, coeffs.use_cholesky, use_chol_type);

  const mlir::MlirOp zero_i32 =
      MakeScalarConstant(builder, 0, builder.getOpBuilder().getI32Type());

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_init,
                      BuildQdwhInitialScaling(input));

  const llvm::ArrayRef<int64_t> input_shape = input_type.getShape();
  const int64_t rank = input_type.getRank();
  const mlir::MlirOp true_const =
      MakeScalarConstant(builder, true, builder.getOpBuilder().getI1Type());
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp active_mask_init,
                      BroadcastIfNeeded(true_const, input_shape.drop_back(2)));

  auto while_op = builder.createUnwrapped<mlir::stablehlo::WhileOp>(
      mlir::TypeRange{zero_i32.getType(), u_init.getType(),
                      active_mask_init.getType()},
      mlir::ValueRange{zero_i32.getValue(), u_init.getValue(),
                       active_mask_init.getValue()});

  {
    mlir::RegionBuilder cond_rb(builder, while_op.getCond());
    const auto args = mlir::stablehlo::Arguments(cond_rb, while_op);
    const mlir::MlirOp i_op = args[0];
    const mlir::MlirOp active_mask = args[2];

    const mlir::MlirOp max_iter_op = MakeScalarConstant(
        cond_rb, max_iterations, cond_rb.getOpBuilder().getI32Type());
    mlir::MlirOp i_lt_max(cond_rb, mlir::stablehlo::CompareOp::create(
                                       cond_rb.getOpBuilder(), cond_rb.getLoc(),
                                       i_op.getValue(), max_iter_op.getValue(),
                                       mlir::stablehlo::ComparisonDirection::LT)
                                       .getResult());

    // Check if any matrix in the batch is still active (not converged).
    // For batched inputs, we reduce the active mask over all batch dimensions.
    mlir::MlirOp any_active;
    if (input_shape.size() <= 2) {
      any_active = active_mask;
    } else {
      TT_ASSIGN_OR_RETURN(
          any_active, BuildAnyShlo(active_mask, GetAllDimensions(active_mask),
                                   ReductionMode::kDropDims));
    }

    const mlir::MlirOp cond = mlir::stablehlo::And(any_active, i_lt_max);
    mlir::stablehlo::Return(cond_rb, {cond});
  }

  {
    mlir::RegionBuilder body_rb(builder, while_op.getBody());
    const auto args = mlir::stablehlo::Arguments(body_rb, while_op);

    const QdwhLoopOptions options{
        .element_type = real_element_type,
        .tolerance = tolerance,
        .batch_shape = input_shape.drop_back(2),
        .rank = rank,
        .a_consts = a_consts,
        .b_consts = b_consts,
        .c_consts = c_consts,
        .use_chol_consts = use_chol_consts,
    };
    TT_ASSIGN_OR_RETURN(
        const auto body_results,
        BuildQdwhLoopBody(body_rb, args[0], args[1], args[2], options));

    mlir::stablehlo::Return(body_rb, body_results);
  }

  builder.getOpBuilder().setInsertionPointAfter(while_op);

  const mlir::MlirOp u_final(builder, while_op.getResult(1));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_refined,
                      BuildNewtonSchulzRefinement(u_final, element_type));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp h,
                      BuildHermitianFactor(u_refined, input, element_type));
  return {{u_refined, h}};
}

constexpr mlir::ElementType kCastDtypeForIntegerInput = mlir::ElementType::F64;

absl::StatusOr<mlir::ElementType> GetOutDtype(const at::Tensor& self) {
  return IsInteger(self) ? kCastDtypeForIntegerInput
                         : ConvertTo<mlir::ElementType>(self.scalar_type());
}

mlir::MlirOp MaybeCastInput(mlir::MlirOp self_op, bool input_is_integer) {
  if (!input_is_integer) {
    return self_op;
  }

  return mlir::stablehlo::ConvertElementType(self_op,
                                             kCastDtypeForIntegerInput);
}

// Returns a slice of the input tensor with size 0 along the specified
// dimension.
absl::StatusOr<mlir::MlirOp> MakeEmptySlice(mlir::MlirOp op,
                                            int64_t dimension) {
  const mlir::RankedTensorType input_tensor_type = GetTensorTypeOrDie(op);
  const llvm::ArrayRef<int64_t> input_shape = input_tensor_type.getShape();
  const Dimensions left_dim(input_tensor_type.getRank(), 0);
  Dimensions right_dim(input_shape.begin(), input_shape.end());
  right_dim[dimension] = 0;
  const Dimensions strides(input_tensor_type.getRank(), 1);
  return mlir::stablehlo::Slice(op, left_dim, right_dim, strides);
}

absl::StatusOr<MlirOpResults<2>> BuildSortEigenPairs(
    mlir::MlirOp s, mlir::MlirOp v, mlir::ElementType element_type) {
  const int64_t s_rank = GetTensorTypeOrDie(s).getRank();

  const SortShloOutputs sorted_outputs = BuildSortShlo(
      s, /*stable=*/true, /*dim=*/s_rank - 1, /*descending=*/true);
  const mlir::MlirOp s_sorted = sorted_outputs.values;
  const mlir::MlirOp sort_idx = sorted_outputs.indices;

  const mlir::RankedTensorType v_type = GetTensorTypeOrDie(v);
  const int64_t rank = v_type.getRank();
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp unsqueezed_idx,
                      Unsqueeze(sort_idx, rank - 2));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp broadcasted_idx,
                      BroadcastIfNeeded(unsqueezed_idx, v));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp v_sorted,
                      BuildGatherShlo(v, /*dim=*/rank - 1, broadcasted_idx,
                                      /*sparse_grad=*/false, element_type));

  return MlirOpResults<2>{{s_sorted, v_sorted}};
}

struct SvdPreprocessResult {
  mlir::MlirOp a_svd;
  std::optional<mlir::MlirOp> q_op;
  std::optional<mlir::MlirOp> u_out_null;
};

// If complete QR is requested (full_matrices=true and m > n), performs complete
// QR and slices the results. Otherwise, if the matrix is sufficiently tall (m
// > 1.15 * n), performs reduced QR to reduce the size of the matrix before SVD.
absl::StatusOr<SvdPreprocessResult> BuildSvdPreprocess(mlir::MlirOp a,
                                                       bool full_matrices) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(a);
  const llvm::ArrayRef<int64_t> a_shape = a_type.getShape();
  const int64_t rank = a_type.getRank();
  const int64_t m = a_shape[rank - 2];
  const int64_t n = a_shape[rank - 1];

  std::optional<mlir::MlirOp> q_op;
  std::optional<mlir::MlirOp> u_out_null;
  mlir::MlirOp a_svd = a;

  if (full_matrices && m > n) {
    TT_ASSIGN_OR_RETURN(const auto qr_results, BuildQrShlo(a, "complete"));
    const mlir::MlirOp q_full = qr_results[0];
    const mlir::MlirOp r_full = qr_results[1];

    // Extract the first 'n' columns of Q_full (shape m x n) to form Q.
    TT_ASSIGN_OR_RETURN(q_op, BuildMaybeSlice(q_full, rank - 1, 0, n - m));
    // Extract the remaining 'm - n' columns of Q_full (shape m x (m - n))
    // to form U_out_null.
    TT_ASSIGN_OR_RETURN(u_out_null, BuildMaybeSlice(q_full, rank - 1, -n, 0));
    // Extract the top 'n' rows of R_full (shape n x n) to form R.
    TT_ASSIGN_OR_RETURN(a_svd, BuildMaybeSlice(r_full, rank - 2, 0, n - m));
  } else if (m > 1.15 * n) {
    // The constant 1.15 is the same as in JAX's linalg.svd.
    TT_ASSIGN_OR_RETURN(const auto qr_results, BuildQrShlo(a, "reduced"));
    q_op = qr_results[0];
    a_svd = qr_results[1];
  }

  return SvdPreprocessResult{a_svd, q_op, u_out_null};
}

// Reconstructs the final U matrix from the SVD of the preprocessed matrix.
// If QR reduction was used (A = Q @ R), computes U = Q @ U.
// If full_matrices=True, appends: U = [U, U_null].
absl::StatusOr<mlir::MlirOp> BuildSvdUReconstruction(
    mlir::MlirOp u_svd, std::optional<mlir::MlirOp> q_op,
    std::optional<mlir::MlirOp> u_out_null) {
  mlir::MlirOp u_reconstructed = u_svd;
  if (q_op.has_value()) {
    TT_ASSIGN_OR_RETURN(u_reconstructed,
                        BuildMatMulShlo(*q_op, u_reconstructed));
  }
  if (u_out_null.has_value()) {
    mlir::MlirBuilder& builder = u_svd.getBuilder();
    const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u_svd);
    const int64_t rank = u_type.getRank();
    u_reconstructed = mlir::stablehlo::Concatenate(
        builder, {u_reconstructed, *u_out_null}, rank - 1);
  }
  return u_reconstructed;
}

struct SvdPostprocessResult {
  mlir::MlirOp u;
  mlir::MlirOp vh;
};

// Postprocesses SVD outputs by swapping U and V if the input was transposed.
absl::StatusOr<SvdPostprocessResult> BuildSvdPostprocess(mlir::MlirOp u_svd,
                                                         mlir::MlirOp v_sorted,
                                                         bool transpose_input) {
  if (transpose_input) {
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp vh_final,
                        BuildConjugateTransposeShlo(u_svd));
    return SvdPostprocessResult{v_sorted, vh_final};
  }

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp vh_final,
                      BuildConjugateTransposeShlo(v_sorted));
  return SvdPostprocessResult{u_svd, vh_final};
}

// Extracts the diagonal elements of a (..., N, N) tensor to (..., N).
//
// Example for a 3x3 matrix (no batch):
// Input:
//   op = [[1, 2, 3],
//         [4, 5, 6],
//         [7, 8, 9]]
// Steps:
//   1. Generate column indices: index_bcast = [[0], [1], [2]] (shape: 3x1)
//   2. Gather elements along columns = [[1], [5], [9]] (shape: 3x1)
//   3. Reshape to 1D: [1, 5, 9] (shape: 3)
absl::StatusOr<mlir::MlirOp> BuildGetDiagonal(mlir::MlirOp op) {
  mlir::MlirBuilder& builder = op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(op);
  const int64_t rank = type.getRank();
  const llvm::ArrayRef<int64_t> shape = type.getShape();
  const int64_t n = shape[rank - 1];

  const auto iota_type =
      mlir::RankedTensorType::get({n}, builder.getOpBuilder().getI32Type());
  const mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp index_2d, Unsqueeze(iota, 1));

  Dimensions target_index_shape(shape.begin(), shape.end() - 2);
  target_index_shape.reserve(shape.size());
  target_index_shape.push_back(n);
  target_index_shape.push_back(1);

  Indices broadcast_dimensions = {rank - 2, rank - 1};
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp index_bcast,
      Broadcast(index_2d, target_index_shape, broadcast_dimensions));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp gather_res,
                      BuildGatherShlo(op, rank - 1, index_bcast,
                                      /*sparse_grad=*/false, element_type));

  return Squeeze(gather_res, {rank - 1});
}

// Computes QR decomposition and returns Q with corrected column signs.
// Column signs are flipped to ensure the diagonal of R is non-negative,
// which makes the QR decomposition consistent.
// Steps:
//   1. [Q, R] = QR(input)
//   2. diag_R = Diagonal(R)
//   3. signs = diag_R >= 0 ? 1 : -1  (vector of size N)
//   4. Q_corrected = Q * Bcast(signs)
//      Each column i of Q (size M x N) is multiplied by signs[i].
absl::StatusOr<mlir::MlirOp> BuildQrCorrectedSigns(
    mlir::MlirOp u_svd_rb, const mlir::Type& real_mlir_type) {
  TT_ASSIGN_OR_RETURN(const auto qr_results, BuildQrShlo(u_svd_rb, "reduced"));
  mlir::MlirOp q = qr_results[0];
  const mlir::MlirOp r = qr_results[1];
  TT_ASSIGN_OR_RETURN(mlir::MlirOp diag_r, BuildGetDiagonal(r));

  mlir::MlirOp diag_r_real = diag_r;
  const mlir::RankedTensorType diag_r_type = GetTensorTypeOrDie(diag_r);
  mlir::MlirBuilder& builder = u_svd_rb.getBuilder();
  if (IsComplexType(diag_r_type)) {
    // Note: Householder QR guarantees diag_r is real. CompareOp requires real
    // float operands, so we extract Real(diag_r) for comparison.
    diag_r_real = mlir::stablehlo::Real(diag_r);
  }

  const mlir::MlirOp zero = MakeConstantLike(diag_r_real, 0.0);
  const mlir::MlirOp diag_r_ge_zero(
      builder,
      mlir::stablehlo::CompareOp::create(
          builder.getOpBuilder(), builder.getLoc(), diag_r_real.getValue(),
          zero.getValue(), mlir::stablehlo::ComparisonDirection::GE)
          .getResult());
  const mlir::MlirOp one = MakeConstantLike(diag_r_real, 1.0);
  const mlir::MlirOp minus_one = MakeConstantLike(diag_r_real, -1.0);

  const mlir::MlirOp signs = builder.create<mlir::stablehlo::SelectOp>(
      diag_r_ge_zero.getValue(), one.getValue(), minus_one.getValue());
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp signs_complex,
                      ConvertToComplexIfNeeded(signs, q));

  const int64_t rank = GetTensorTypeOrDie(u_svd_rb).getRank();
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp signs_unsqueezed,
                      Unsqueeze(signs_complex, rank - 2));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp signs_bcast,
                      BroadcastIfNeeded(signs_unsqueezed, q));
  return mlir::stablehlo::Mul(q, signs_bcast);
}

// Slices the input tensor along its last dimension from start to end.
absl::StatusOr<mlir::MlirOp> BuildSliceLastDimension(mlir::MlirOp op,
                                                     int64_t start,
                                                     int64_t end) {
  const mlir::RankedTensorType op_type = GetTensorTypeOrDie(op);
  const int64_t op_rank = op_type.getRank();
  const llvm::ArrayRef<int64_t> op_shape = op_type.getShape();
  Dimensions left(op_rank, 0);
  left[op_rank - 1] = start;
  Dimensions right(op_shape.begin(), op_shape.end());
  right[op_rank - 1] = end;
  const Dimensions strides(op_rank, 1);
  return mlir::stablehlo::Slice(op, left, right, strides);
}

// Checks if the matrix is rank-deficient using the formula:
//   s_min <= N * eps * s_max
// where:
//   - s_min is the smallest singular value.
//   - s_max is the largest singular value.
//   - N is the dimension (number of singular values).
//   - eps is the precision (epsilon).
// Returns a boolean indicating if any batch element is rank-deficient.
absl::StatusOr<mlir::MlirOp> BuildCheckRankDeficiency(
    mlir::MlirOp s_sorted, absl::Span<const int64_t> batch_shape,
    mlir::ElementType element_type) {
  mlir::MlirBuilder& builder = s_sorted.getBuilder();
  const mlir::RankedTensorType s_type = GetTensorTypeOrDie(s_sorted);
  const int64_t rank = s_type.getRank();
  const llvm::ArrayRef<int64_t> s_shape = s_type.getShape();
  const int64_t n = s_shape[rank - 1];

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp s_max_slice,
                      BuildSliceLastDimension(s_sorted, 0, 1));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp s_min_slice,
                      BuildSliceLastDimension(s_sorted, n - 1, n));
  const mlir::ElementType real_element_type = RealComponentOf(element_type);
  const int64_t squeeze_dim = batch_shape.size();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp s_max, Squeeze(s_max_slice, {squeeze_dim}));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp s_min,
                      Squeeze(s_min_slice, {squeeze_dim}));
  const double epsilon_val = GetEpsilon(real_element_type);
  const mlir::MlirOp n_epsilon =
      MakeScalarConstant(builder, n * epsilon_val, real_element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp n_epsilon_bcast,
                      BroadcastScalar(n_epsilon, batch_shape));
  const mlir::MlirOp threshold = mlir::stablehlo::Mul(n_epsilon_bcast, s_max);

  const mlir::MlirOp need_correction(
      builder,
      mlir::stablehlo::CompareOp::create(
          builder.getOpBuilder(), builder.getLoc(), s_min.getValue(),
          threshold.getValue(), mlir::stablehlo::ComparisonDirection::LE)
          .getResult());

  mlir::MlirOp need_correction_any = need_correction;
  if (!batch_shape.empty()) {
    TT_ASSIGN_OR_RETURN(
        need_correction_any,
        BuildAnyShlo(need_correction, GetAllDimensions(need_correction),
                     ReductionMode::kDropDims));
  }

  return need_correction_any;
}

absl::StatusOr<mlir::MlirOp> BuildCorrectRankDeficiency(
    mlir::MlirOp u_svd, mlir::MlirOp s_sorted,
    const mlir::ElementType element_type) {
  mlir::MlirBuilder& builder = u_svd.getBuilder();
  const mlir::RankedTensorType u_type = GetTensorTypeOrDie(u_svd);
  const llvm::ArrayRef<int64_t> u_shape = u_type.getShape();
  const Dimensions batch_shape(u_shape.begin(), u_shape.end() - 2);

  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp need_correction_any,
      BuildCheckRankDeficiency(s_sorted, batch_shape, element_type));

  const mlir::ElementType real_element_type = RealComponentOf(element_type);
  const mlir::Type real_mlir_type =
      mlir::getElementType(builder.getContext(), real_element_type);

  auto if_op = mlir::stablehlo::IfOp::create(
      builder.getOpBuilder(), builder.getLoc(), mlir::TypeRange{u_type},
      need_correction_any.getValue());

  // True branch: If rank deficiency is detected in any batch element, apply the
  // QR correction to the entire batch (matching JAX).
  {
    mlir::RegionBuilder true_rb(builder, if_op->getRegion(0));
    mlir::MlirOp u_svd_rb = mlir::swap(true_rb, u_svd);
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_corrected,
                        BuildQrCorrectedSigns(u_svd_rb, real_mlir_type));
    mlir::stablehlo::Return(true_rb, {u_corrected});
  }

  // False branch: If no rank deficiency is detected, return the original U
  // unchanged.
  {
    mlir::RegionBuilder false_rb(builder, if_op->getRegion(1));
    mlir::MlirOp u_svd_rb = mlir::swap(false_rb, u_svd);
    mlir::stablehlo::Return(false_rb, {u_svd_rb});
  }

  return mlir::MlirOp(builder, if_op.getResult(0));
}

struct DiagonalShiftResult {
  mlir::MlirOp shift_matrix;
  mlir::MlirOp delta;
};

// Constructs a batched diagonal shift matrix delta * I of shape `shape`,
// where delta = h_norm * shift_epsilon.
absl::StatusOr<DiagonalShiftResult> BuildDiagonalShiftMatrix(
    mlir::MlirOp h_norm, mlir::ElementType element_type,
    llvm::ArrayRef<int64_t> shape) {
  mlir::MlirBuilder& builder = h_norm.getBuilder();
  const mlir::ElementType real_element_type = RealComponentOf(element_type);

  const double shift_epsilon =
      (real_element_type == mlir::ElementType::F64) ? 1e-12 : 1e-6;
  const mlir::MlirOp shift =
      MakeScalarConstant(builder, shift_epsilon, real_element_type);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp shift_bcast,
                      BroadcastIfNeeded(shift, h_norm));
  const mlir::MlirOp delta = mlir::stablehlo::Mul(h_norm, shift_bcast);

  const int64_t n = shape.back();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp eye, BuildBatchedEye(builder, element_type,
                                                        n, shape.drop_back(2)));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp delta_complex,
                      ConvertToComplexIfNeeded(delta, eye));
  const Dimensions batch_dims = GetAllDimensions(delta);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp delta_bcast,
                      Broadcast(delta_complex, shape, batch_dims));

  const mlir::MlirOp shift_matrix = mlir::stablehlo::Mul(delta_bcast, eye);
  return DiagonalShiftResult{shift_matrix, delta};
}

// Recovers unshifted singular values by subtracting delta from the regularized
// eigenvalues s_reg and clamping to 0 to avoid negative values due to
// precision.
absl::StatusOr<mlir::MlirOp> BuildRecoverSingularValues(mlir::MlirOp s_reg,
                                                        mlir::MlirOp delta) {
  mlir::MlirBuilder& builder = s_reg.getBuilder();
  const Dimensions batch_dims = GetAllDimensions(delta);
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp delta_bcast,
      Broadcast(delta, GetTensorTypeOrDie(s_reg).getShape(), batch_dims));
  mlir::MlirOp subtracted = mlir::stablehlo::Subtract(s_reg, delta_bcast);
  const mlir::MlirOp zero =
      MakeScalarConstant(builder, 0.0, GetElementTypeOrSelf(subtracted));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_bcast,
                      BroadcastIfNeeded(zero, subtracted));
  return mlir::stablehlo::Max(subtracted, zero_bcast);
}

// Regularizes H with a diagonal shift: H_reg = H + delta * I, to prevent NaNs
// in BuildEighShlo on rank-deficient inputs.
absl::StatusOr<MlirOpResults<2>> BuildEighWithDiagonalShift(mlir::MlirOp h) {
  const mlir::RankedTensorType h_type = GetTensorTypeOrDie(h);
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(h_type.getElementType()));

  TT_ASSIGN_OR_RETURN(mlir::MlirOp h_norm, BuildMatrix1Norm(h));
  const llvm::ArrayRef<int64_t> h_shape = h_type.getShape();
  TT_ASSIGN_OR_RETURN(DiagonalShiftResult shift_result,
                      BuildDiagonalShiftMatrix(h_norm, element_type, h_shape));
  const mlir::MlirOp h_reg = mlir::stablehlo::Add(h, shift_result.shift_matrix);

  TT_ASSIGN_OR_RETURN(const auto eigh_results, BuildEighShlo(h_reg));
  const mlir::MlirOp v = eigh_results[0];
  const mlir::MlirOp s_reg = eigh_results[1];

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp s,
                      BuildRecoverSingularValues(s_reg, shift_result.delta));
  return MlirOpResults<2>{{v, s}};
}

absl::StatusOr<MlirOpResults<3>> BuildSvdTallAndSquare(mlir::MlirOp input,
                                                       bool compute_uv) {
  TT_ASSIGN_OR_RETURN(const auto qdwh_results,
                      // 10 iterations matches JAX's default.
                      BuildQdwhShlo(input, /*max_iterations=*/10));
  const mlir::MlirOp h = qdwh_results[1];

  TT_ASSIGN_OR_RETURN(const auto eigh_results, BuildEighWithDiagonalShift(h));
  const mlir::MlirOp v = eigh_results[0];
  const mlir::MlirOp s = eigh_results[1];

  if (!compute_uv) {
    const int64_t s_rank = GetTensorTypeOrDie(s).getRank();
    const SortShloOutputs sorted_s_outputs = BuildSortShlo(
        s, /*stable=*/true, /*dim=*/s_rank - 1, /*descending=*/true);
    const mlir::MlirOp s_sorted = sorted_s_outputs.values;

    const int64_t rank = GetTensorTypeOrDie(input).getRank();
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_dummy,
                        MakeEmptySlice(input, rank - 1));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp v_dummy,
                        MakeEmptySlice(v, rank - 1));
    return MlirOpResults<3>{{u_dummy, s_sorted, v_dummy}};
  }

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      GetElementType(input));
  TT_ASSIGN_OR_RETURN(const auto sort_results,
                      BuildSortEigenPairs(s, v, element_type));
  const mlir::MlirOp s_sorted = sort_results[0];
  const mlir::MlirOp v_sorted = sort_results[1];

  const mlir::MlirOp u_polar = qdwh_results[0];
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_svd,
                      BuildMatMulShlo(u_polar, v_sorted));

  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp u_corrected,
      BuildCorrectRankDeficiency(u_svd, s_sorted, element_type));

  return MlirOpResults<3>{{u_corrected, s_sorted, v_sorted}};
}

/// Returns an element-wise boolean mask indicating which elements of the input
// tensor are finite (not NaN and not Inf).
absl::StatusOr<mlir::MlirOp> BuildIsFiniteShlo(mlir::MlirOp input) {
  const mlir::Type mlir_type = GetElementTypeOrSelf(input);

  if (llvm::isa<mlir::ComplexType>(mlir_type)) {
    mlir::MlirOp real_part = mlir::stablehlo::Real(input);
    mlir::MlirOp imag_part = mlir::stablehlo::Imag(input);
    mlir::MlirOp real_finite = mlir::stablehlo::IsFinite(real_part);
    mlir::MlirOp imag_finite = mlir::stablehlo::IsFinite(imag_part);
    return mlir::stablehlo::And(real_finite, imag_finite);
  } else if (llvm::isa<mlir::FloatType>(mlir_type)) {
    return mlir::stablehlo::IsFinite(input);
  }
  return MakeConstantLike(input, true, mlir::ElementType::PRED);
}

// Returns a boolean tensor of shape (batch_shape) indicating if all elements
// in each batch matrix are finite (i.e., not NaN and not Inf).
absl::StatusOr<mlir::MlirOp> BuildAllFinite(mlir::MlirOp input) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp is_finite, BuildIsFiniteShlo(input));
  const int64_t rank = GetTensorTypeOrDie(is_finite).getRank();
  return BuildAllShlo(is_finite, {rank - 2, rank - 1},
                      ReductionMode::kDropDims);
}

// Returns a tensor of the same shape and type as the input, but filled with NaN
// values (for float and complex types). Returns the input unchanged for other
// types.
absl::StatusOr<mlir::MlirOp> BuildNanLike(mlir::MlirOp op) {
  const mlir::Type mlir_type = GetElementTypeOrSelf(op);
  const double nan_val = std::numeric_limits<double>::quiet_NaN();

  if (llvm::isa<mlir::ComplexType>(mlir_type)) {
    const mlir::Type real_mlir_type =
        llvm::cast<mlir::ComplexType>(mlir_type).getElementType();
    mlir::MlirOp nan_real = MakeConstantLike(op, nan_val, real_mlir_type);
    return mlir::stablehlo::Complex(nan_real, nan_real);
  } else if (llvm::isa<mlir::FloatType>(mlir_type)) {
    return MakeConstantLike(op, nan_val, mlir_type);
  }

  return op;
}

// Conditionally replaces the output tensor with NaNs if the corresponding batch
// element in the `is_finite` mask is false.
absl::StatusOr<mlir::MlirOp> PropagateNanIfNeeded(mlir::MlirOp output,
                                                  mlir::MlirOp is_finite) {
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp nan_output, BuildNanLike(output));
  mlir::MlirBuilder& builder = output.getBuilder();
  const Indices all_dims = GetAllDimensions(is_finite);
  TT_ASSIGN_OR_RETURN(
      const mlir::MlirOp is_finite_bcast,
      Broadcast(is_finite, GetTensorTypeOrDie(output).getShape(), all_dims));
  return builder.create<mlir::stablehlo::SelectOp>(
      is_finite_bcast.getValue(), output.getValue(), nan_output.getValue());
}

absl::StatusOr<MlirOpResults<3>> BuildSvdShlo(mlir::MlirOp input,
                                              bool full_matrices,
                                              bool compute_uv) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  const int64_t rank = input_type.getRank();
  const llvm::ArrayRef<int64_t> input_shape = input_type.getShape();

  const int64_t m = input_shape[rank - 2];
  const int64_t n = input_shape[rank - 1];

  const bool transpose_input = m < n;
  mlir::MlirOp a = input;
  if (transpose_input) {
    TT_ASSIGN_OR_RETURN(a, BuildConjugateTransposeShlo(input));
  }

  TT_ASSIGN_OR_RETURN(const SvdPreprocessResult preprocess_result,
                      BuildSvdPreprocess(a, full_matrices));

  TT_ASSIGN_OR_RETURN(
      const auto svd_results,
      BuildSvdTallAndSquare(preprocess_result.a_svd, compute_uv));
  const mlir::MlirOp u = svd_results[0];
  const mlir::MlirOp s = svd_results[1];
  const mlir::MlirOp v = svd_results[2];

  if (!compute_uv) {
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp is_finite, BuildAllFinite(input));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp s_final,
                        PropagateNanIfNeeded(s, is_finite));

    TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_dummy,
                        MakeEmptySlice(input, rank - 1));
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp vh_dummy,
                        MakeEmptySlice(input, rank - 2));
    return MlirOpResults<3>{{u_dummy, s_final, vh_dummy}};
  }

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_reconstructed,
                      BuildSvdUReconstruction(u, preprocess_result.q_op,
                                              preprocess_result.u_out_null));

  TT_ASSIGN_OR_RETURN(const auto postprocess_result,
                      BuildSvdPostprocess(u_reconstructed, v, transpose_input));

  // NaN propagation logic (matching JAX).
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp is_finite, BuildAllFinite(input));

  TT_ASSIGN_OR_RETURN(const mlir::MlirOp u_final,
                      PropagateNanIfNeeded(postprocess_result.u, is_finite));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp s_final,
                      PropagateNanIfNeeded(s, is_finite));
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp vh_final,
                      PropagateNanIfNeeded(postprocess_result.vh, is_finite));

  return MlirOpResults<3>{{u_final, s_final, vh_final}};
}

struct SvdOutDims {
  Dimensions u;
  Dimensions s;
  Dimensions vh;
};

SvdOutDims GetSvdOutDims(const at::Tensor& self, bool full_matrices,
                         bool compute_uv) {
  const int64_t rank = self.dim();
  const int64_t m = self.size(rank - 2);
  const int64_t n = self.size(rank - 1);
  const int64_t k = std::min(m, n);

  const at::IntArrayRef batch_dims = self.sizes().slice(0, rank - 2);

  const int64_t u_cols = compute_uv ? (full_matrices ? m : k) : 0;
  const int64_t vh_rows = compute_uv ? (full_matrices ? n : k) : 0;

  return {ConcatBatchAndDims(batch_dims, m, u_cols),
          ConcatBatchAndDims(batch_dims, k),
          ConcatBatchAndDims(batch_dims, vh_rows, n)};
}

absl::StatusOr<at::Tensor> DowncastFromComplexDouble(const at::Tensor& self) {
  const auto to_c64_builder =
      [](mlir::MlirOp op) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp real = mlir::stablehlo::Real(op);
    const mlir::MlirOp imag = mlir::stablehlo::Imag(op);

    mlir::MlirOp real_f32 =
        mlir::stablehlo::ConvertElementType(real, mlir::ElementType::F32);
    mlir::MlirOp imag_f32 =
        mlir::stablehlo::ConvertElementType(imag, mlir::ElementType::F32);

    return mlir::stablehlo::Complex(real_f32, imag_f32);
  };

  TT_ASSIGN_OR_RETURN(
      const DeviceBufferRef c64_buf,
      DispatchOp<1>(std::move(to_c64_builder), self,
                    {.op_name = OpName::kToCopy,
                     .out_dtype = mlir::ElementType::COMPLEXF32,
                     .out_dims = self.sizes(),
                     .op_param_cache_keys = OpParamCacheKeys::Empty()}));

  return MakeTensor(c64_buf, self.device().index());
}

absl::StatusOr<DeviceBufferRef> UpcastToComplexDouble(
    const at::Tensor& tensor, const Dimensions& target_dims) {
  const auto upcast_c128 = [](mlir::MlirOp op) -> absl::StatusOr<mlir::MlirOp> {
    const mlir::MlirOp real = mlir::stablehlo::Real(op);
    const mlir::MlirOp imag = mlir::stablehlo::Imag(op);

    mlir::MlirOp real_f64 =
        mlir::stablehlo::ConvertElementType(real, mlir::ElementType::F64);
    mlir::MlirOp imag_f64 =
        mlir::stablehlo::ConvertElementType(imag, mlir::ElementType::F64);

    return mlir::stablehlo::Complex(real_f64, imag_f64);
  };

  return DispatchOp<1>(upcast_c128, tensor,
                       {.op_name = OpName::kToCopy,
                        .out_dtype = mlir::ElementType::COMPLEXF64,
                        .out_dims = target_dims,
                        .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

absl::StatusOr<DeviceBufferRef> UpcastToDouble(const at::Tensor& tensor,
                                               const Dimensions& target_dims) {
  const auto upcast_f64 = [](mlir::MlirOp op) -> absl::StatusOr<mlir::MlirOp> {
    return mlir::stablehlo::ConvertElementType(op, mlir::ElementType::F64);
  };

  return DispatchOp<1>(upcast_f64, tensor,
                       {.op_name = OpName::kToCopy,
                        .out_dtype = mlir::ElementType::F64,
                        .out_dims = target_dims,
                        .op_param_cache_keys = OpParamCacheKeys::Empty()});
}

absl::StatusOr<DeviceBufferRefArray<3>> UpcastComplexDoubleOutputs(
    const DeviceBufferRefArray<3>& c64_results, const SvdOutDims& expected_dims,
    int64_t device_index) {
  const at::Tensor u_c64 = MakeTensor(c64_results[0], device_index);
  const at::Tensor s_f32 = MakeTensor(c64_results[1], device_index);
  const at::Tensor vh_c64 = MakeTensor(c64_results[2], device_index);

  TT_ASSIGN_OR_RETURN(const DeviceBufferRef u_buf,
                      UpcastToComplexDouble(u_c64, expected_dims.u));
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef s_buf,
                      UpcastToDouble(s_f32, expected_dims.s));
  TT_ASSIGN_OR_RETURN(const DeviceBufferRef vh_buf,
                      UpcastToComplexDouble(vh_c64, expected_dims.vh));

  return DeviceBufferRefArray<3>{u_buf, s_buf, vh_buf};
}

absl::StatusOr<DeviceBufferRefArray<3>> Svd(const at::Tensor& self,
                                            bool full_matrices, bool compute_uv,
                                            OpParamCacheKeys param_keys) {
  const SvdOutDims expected_dims =
      GetSvdOutDims(self, full_matrices, compute_uv);

  // Due to lack of native complex128 support, compute SVD in complex64 and
  // upcast outputs back to preserve user dtype.
  if (self.scalar_type() == at::ScalarType::ComplexDouble) {
    TT_ASSIGN_OR_RETURN(const at::Tensor self_c64,
                        DowncastFromComplexDouble(self));
    TT_ASSIGN_OR_RETURN(
        const DeviceBufferRefArray<3> c64_results,
        Svd(self_c64, full_matrices, compute_uv, std::move(param_keys)));
    return UpcastComplexDoubleOutputs(c64_results, expected_dims,
                                      self.device().index());
  }

  const bool input_is_integer = IsInteger(self);
  const auto op_builder =
      [full_matrices, compute_uv, input_is_integer](
          mlir::MlirOp self_op) -> absl::StatusOr<MlirOpResults<3>> {
    const mlir::MlirOp cast_self_op = MaybeCastInput(self_op, input_is_integer);
    return BuildSvdShlo(cast_self_op, full_matrices, compute_uv);
  };

  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype, GetOutDtype(self));

  return DispatchOp<1, 3>(
      std::move(op_builder), {self},
      {.out_dtypes = {out_dtype, RealComponentOf(out_dtype), out_dtype},
       .out_dims_list = {expected_dims.u, expected_dims.s, expected_dims.vh},
       .op_param_cache_keys = std::move(param_keys)});
}

absl::Status HandleEmptyInputSvd(const at::Tensor& self, bool full_matrices,
                                 bool compute_uv, at::Tensor& u, at::Tensor& s,
                                 at::Tensor& vh) {
  if (!compute_uv || !full_matrices) {
    return absl::OkStatus();
  }

  const int64_t m = self.size(-2);
  const int64_t n = self.size(-1);
  if ((m == 0) == (n == 0)) {
    return absl::OkStatus();
  }

  at::Tensor& target = (m == 0) ? vh : u;
  const int64_t dim = (m == 0) ? n : m;
  TT_ASSIGN_OR_RETURN(const mlir::ElementType dtype,
                      ConvertTo<mlir::ElementType>(target.scalar_type()));
  const at::IntArrayRef batch = self.sizes().slice(0, self.dim() - 2);
  return ApplyNullaryOpOut(
      target,
      [dtype, dim, batch](mlir::MlirBuilder& builder) {
        return BuildBatchedEye(builder, dtype, dim, batch);
      },
      target.scalar_type(), target.sizes(), OpParamCacheKeys::Empty());
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> AtenLinalgSvdU(
    const at::Tensor& self, bool full_matrices, bool compute_uv,
    std::optional<c10::string_view> driver, at::Tensor& u, at::Tensor& s,
    at::Tensor& vh) {
  TT_KERNEL(
      OpName::kLinalgSvdU, param_keys,
      (self, full_matrices, compute_uv, driver, u, s, vh), {
        TT_CHECK_THROW(  // ERROR_COV_INFEASIBLE=PyTorch catches this error
                         // first; this check is here as a safeguard.
            self.dim() >= 2, error::kInvalidArgument)
            << "expected input to have at least 2 dimensions, got "
            << self.dim();

        if (driver.has_value() && !driver->empty() && *driver != "gesvd") {
          TT_CHECK_THROW(false, error::kPythonNotImplementedError)
              << "expected default driver ('gesvd'), got '" << *driver << "'";
        }

        TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                           GetOutDtype(self));
        const at::ScalarType expected_u_vh_dtype =
            ConvertTo<at::ScalarType>(out_dtype);
        const at::ScalarType expected_s_dtype =
            c10::toRealValueType(expected_u_vh_dtype);

        TT_CHECK_THROW(u.scalar_type() == expected_u_vh_dtype,
                       error::kInvalidArgument)
            << "expected out tensor to have dtype "
            << ToString(expected_u_vh_dtype) << ", got "
            << ToString(u.scalar_type());
        TT_CHECK_THROW(s.scalar_type() == expected_s_dtype,
                       error::kInvalidArgument)
            << "expected out tensor to have dtype "
            << ToString(expected_s_dtype) << ", got "
            << ToString(s.scalar_type());
        TT_CHECK_THROW(vh.scalar_type() == expected_u_vh_dtype,
                       error::kInvalidArgument)
            << "expected out tensor to have dtype "
            << ToString(expected_u_vh_dtype) << ", got "
            << ToString(vh.scalar_type());

        const SvdOutDims expected_dims =
            GetSvdOutDims(self, full_matrices, compute_uv);
        TT_THROW_IF_ERROR(ResizeTensor(u, expected_dims.u));
        TT_THROW_IF_ERROR(ResizeTensor(s, expected_dims.s));
        TT_THROW_IF_ERROR(ResizeTensor(vh, expected_dims.vh));

        if (self.numel() == 0) {
          TT_THROW_IF_ERROR(
              HandleEmptyInputSvd(self, full_matrices, compute_uv, u, s, vh));
          return {u, s, vh};
        }

        TT_ASSIGN_OR_THROW(
            const DeviceBufferRefArray<3> result_buffers,
            Svd(self, full_matrices, compute_uv, std::move(param_keys)));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[0], u));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[1], s));
        TT_THROW_IF_ERROR(AssignBufferToAtTensor(result_buffers[2], vh));

        return {u, s, vh};
      });
}

}  // namespace torch_tpu
