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

#include "csrc/ops/linalg/qr/qr_lib.h"

#include <algorithm>
#include <cstdint>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "c10/util/string_view.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/eye/eye_lib.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

mlir::stablehlo::CustomCallOp BuildCustomCallOp(
    mlir::MlirBuilder& builder, llvm::StringRef call_target_name,
    llvm::ArrayRef<mlir::Type> result_types,
    llvm::ArrayRef<mlir::Value> operands) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  mlir::NamedAttribute call_target_attr = op_builder.getNamedAttr(
      "call_target_name", op_builder.getStringAttr(call_target_name));
  mlir::NamedAttribute has_side_effect_attr =
      op_builder.getNamedAttr("has_side_effect", op_builder.getBoolAttr(false));
  auto api_version_attr = op_builder.getNamedAttr(
      "api_version",
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          &builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI));

  return mlir::stablehlo::CustomCallOp::create(
      op_builder, builder.getLoc(), result_types, operands,
      {call_target_attr, has_side_effect_attr, api_version_attr});
}

absl::StatusOr<MlirOpResults<2>> BuildGeqrfCustomCallOp(
    mlir::MlirOp& input, mlir::RankedTensorType out_a_type,
    mlir::RankedTensorType tau_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  auto qr_op = BuildCustomCallOp(builder, "Qr",
                                 /*result_types=*/{out_a_type, tau_type},
                                 /*operands=*/{input.getValue()});

  return {{mlir::MlirOp(builder, qr_op.getResult(0)),
           mlir::MlirOp(builder, qr_op.getResult(1))}};
}

absl::StatusOr<mlir::MlirOp> BuildHouseholderProductCustomCallOp(
    mlir::MlirOp& input, mlir::MlirOp& tau, mlir::RankedTensorType out_type) {
  mlir::MlirBuilder& builder = input.getBuilder();
  auto householder_op =
      BuildCustomCallOp(builder, "ProductOfElementaryHouseholderReflectors",
                        /*result_types=*/{out_type},
                        /*operands=*/{input.getValue(), tau.getValue()});

  return mlir::MlirOp(builder, householder_op.getResult(0));
}

Dimensions ConcatBatchAndMatrixDims(llvm::ArrayRef<int64_t> batch_dims,
                                    int64_t m, int64_t n) {
  Dimensions shape(batch_dims.begin(), batch_dims.end());
  shape.push_back(m);
  shape.push_back(n);
  return shape;
}

absl::StatusOr<MlirOpResults<2>> BuildEmptyQrShlo(
    mlir::MlirBuilder& builder, mlir::ElementType element_type, int64_t m,
    int64_t n, bool full_matrices, llvm::ArrayRef<int64_t> batch_dims,
    c10::string_view mode) {
  const int64_t k = full_matrices ? m : std::min(m, n);

  // Make Q. If mode is "r", Q is an empty matrix. Otherwise, Q is an identity
  // matrix with shape (m, k) broadcasted to the batch dimensions.
  mlir::MlirOp q;
  if (mode == "r") {
    const Dimensions empty_q_dims = {0};
    TT_ASSIGN_OR_RETURN(
        q, MakeConstant(builder, at::Scalar(0), element_type, empty_q_dims));
  } else {
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp eye_matrix,
                        BuildEyeShlo(builder, element_type, m, k));
    const Dimensions eye_dims = ConcatBatchAndMatrixDims(batch_dims, m, k);
    TT_ASSIGN_OR_RETURN(q, BroadcastIfNeeded(eye_matrix, eye_dims));
  }

  // Make R. R is a zero matrix with shape (k, n) broadcasted to the batch
  // dimensions.
  const Dimensions r_dims = ConcatBatchAndMatrixDims(batch_dims, k, n);
  TT_ASSIGN_OR_RETURN(const mlir::MlirOp r, MakeConstant(builder, at::Scalar(0),
                                                         element_type, r_dims));

  return {{q, r}};
}

absl::StatusOr<mlir::MlirOp> BuildPadShlo(
    mlir::MlirOp input, mlir::MlirOp padding_value,
    llvm::ArrayRef<int64_t> edge_padding_high) {
  Dimensions no_padding(GetTensorTypeOrDie(input).getRank(), 0);
  return mlir::stablehlo::Pad(input, padding_value,
                              /*edge_padding_low=*/no_padding,
                              edge_padding_high,
                              /*interior_padding=*/no_padding);
}

absl::StatusOr<mlir::MlirOp> BuildTriuShlo(mlir::MlirOp input,
                                           mlir::ElementType element_type,
                                           int64_t m, int64_t n) {
  mlir::MlirBuilder& builder = input.getBuilder();
  const auto iota_type =
      mlir::RankedTensorType::get({m, n}, builder.getOpBuilder().getI32Type());

  // Row indices:    0 0 0 ...
  //                 1 1 1 ...
  //                 2 2 2 ...
  //                 ...
  mlir::MlirOp rows =
      mlir::stablehlo::Iota(builder, iota_type, /*iota_dimension=*/0);

  // Column indices: 0 1 2 ...
  //                 0 1 2 ...
  //                 0 1 2 ...
  //                 ...
  mlir::MlirOp cols =
      mlir::stablehlo::Iota(builder, iota_type, /*iota_dimension=*/1);

  // Mask:           1 1 1 ...
  //                 0 1 1 ...
  //                 0 0 1 ...
  //                 ...
  const auto is_upper_triangle = mlir::stablehlo::Compare(
      rows, cols, mlir::stablehlo::ComparisonDirection::LE);

  const llvm::ArrayRef<int64_t> shape = GetTensorTypeOrDie(input).getShape();
  TT_ASSIGN_OR_RETURN(mlir::MlirOp full_mask,
                      BroadcastIfNeeded(is_upper_triangle, shape));
  mlir::MlirOp zero = MakeConstantLike(input, 0);

  return mlir::stablehlo::Select(full_mask, input, zero);
}

}  // namespace

absl::StatusOr<MlirOpResults<2>> BuildGeqrfShlo(mlir::MlirOp input) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(input);
  const int64_t a_rank = a_type.getRank();
  const llvm::ArrayRef<int64_t> a_shape = a_type.getShape();
  const int64_t m = a_shape[a_rank - 2];
  const int64_t n = a_shape[a_rank - 1];
  const int64_t k = std::min(m, n);

  // tau has shape (..., k)
  Dimensions tau_dims(a_shape.begin(), a_shape.end() - 2);
  tau_dims.push_back(k);
  const auto tau_type =
      mlir::RankedTensorType::get(tau_dims, a_type.getElementType());

  mlir::MlirBuilder& builder = input.getBuilder();

  // Empty case: return zero matrix for a and zero vector for tau.
  // Bypassing the custom call prevents unnecessary backend overhead and avoids
  // potential codegen issues for empty tensor FFI parameters.
  if (m == 0 || n == 0) {
    TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                        GetElementType(input));
    TT_ASSIGN_OR_RETURN(mlir::MlirOp zero_sized,
                        MakeZeroSizedTensor(builder, element_type));
    mlir::MlirOp a = mlir::stablehlo::Reshape(zero_sized, a_shape);
    mlir::MlirOp tau = mlir::stablehlo::Reshape(zero_sized, tau_dims);
    return {{a, tau}};
  }

  return BuildGeqrfCustomCallOp(input, /*out_a_type=*/a_type, tau_type);
}

absl::StatusOr<MlirOpResults<2>> BuildQrShlo(mlir::MlirOp input,
                                             c10::string_view mode) {
  const mlir::RankedTensorType a_type = GetTensorTypeOrDie(input);
  const int64_t a_rank = a_type.getRank();
  const llvm::ArrayRef<int64_t> a_shape = a_type.getShape();
  const int64_t m = a_shape[a_rank - 2];
  const int64_t n = a_shape[a_rank - 1];

  mlir::MlirBuilder& builder = input.getBuilder();
  const bool full_matrices = mode == "complete";
  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      GetElementType(input));

  // Empty case: return identity matrix for Q and zero matrix for R.
  const llvm::ArrayRef<int64_t> batch_dims = a_shape.drop_back(2);
  if (m == 0 || n == 0) {
    return BuildEmptyQrShlo(builder, element_type, m, n, full_matrices,
                            batch_dims, mode);
  }

  TT_ASSIGN_OR_RETURN(const MlirOpResults<2> geqrf, BuildGeqrfShlo(input));
  mlir::MlirOp packed_qr = geqrf[0];
  mlir::MlirOp taus = geqrf[1];

  mlir::MlirOp q;
  if (mode == "r") {
    const Dimensions empty_q_dims = {0};
    TT_ASSIGN_OR_RETURN(
        q, MakeConstant(builder, at::Scalar(0), element_type, empty_q_dims));
  } else {
    mlir::MlirOp resized_qr;
    if (m < n) {
      // Remove the last n - m columns.
      TT_ASSIGN_OR_RETURN(resized_qr,
                          BuildMaybeSlice(packed_qr, /*dimension=*/a_rank - 1,
                                          /*left_pad=*/0, /*right_pad=*/m - n));
    } else if (full_matrices) {
      // Pad with m - n zero columns.
      TT_ASSIGN_OR_RETURN(const mlir::MlirOp zero,
                          MakeConstant(builder, at::Scalar(0), element_type));
      Dimensions padding(a_rank, 0);
      padding[a_rank - 1] = m - n;
      TT_ASSIGN_OR_RETURN(resized_qr, BuildPadShlo(packed_qr, zero, padding));
    } else {
      resized_qr = packed_qr;
    }

    const Dimensions q_dims = ConcatBatchAndMatrixDims(
        batch_dims, m, full_matrices ? m : std::min(m, n));
    const auto q_type =
        mlir::RankedTensorType::get(q_dims, a_type.getElementType());

    TT_ASSIGN_OR_RETURN(
        q, BuildHouseholderProductCustomCallOp(resized_qr, taus, q_type));
  }

  mlir::MlirOp r;
  if (m > n && !full_matrices) {
    // Remove the last m - n rows.
    TT_ASSIGN_OR_RETURN(r,
                        BuildMaybeSlice(packed_qr, /*dimension=*/a_rank - 2,
                                        /*left_pad=*/0, /*right_pad=*/n - m));
  } else {
    r = packed_qr;
  }
  TT_ASSIGN_OR_RETURN(
      r, BuildTriuShlo(r, element_type, m > n && !full_matrices ? n : m, n));

  return {{q, r}};
}

}  // namespace torch_tpu
