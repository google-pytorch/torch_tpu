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

#include "csrc/ops/contraction_utils.h"

#include "absl/status/statusor.h"
#include "c10/core/ScalarType.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {
namespace {

namespace stablehlo = mlir::stablehlo;

mlir::MlirOp AddShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  return stablehlo::Add(lhs, rhs);
}

mlir::MlirOp SubShlo(mlir::MlirOp lhs, mlir::MlirOp rhs) {
  return stablehlo::Subtract(lhs, rhs);
}

mlir::MlirOp ZeroNaNs(mlir::MlirOp input) {
  mlir::MlirOp is_finite = stablehlo::IsFinite(input);
  mlir::MlirOp zero = MakeConstantLike(input, 0.0);
  return stablehlo::Select(is_finite, input, zero);
}

mlir::MlirOp SingleDotF32(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::ArrayAttr precision_config) {
  return stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                               precision_config,
                               /*preferredElementType=*/mlir::ElementType::F32);
}

}  // namespace

mlir::MlirOp GetBf16Part(mlir::MlirOp x_f32, int part_idx) {
  mlir::MlirOp residual = x_f32;
  for (int i = 0; i < part_idx; ++i) {
    mlir::MlirOp part_f32 =
        stablehlo::ReducePrecision(residual, /*exponent_bits=*/8,
                                   /*mantissa_bits=*/7);
    residual = ZeroNaNs(SubShlo(residual, part_f32));
  }
  mlir::MlirOp target_f32 =
      stablehlo::ReducePrecision(residual, /*exponent_bits=*/8,
                                 /*mantissa_bits=*/7);
  return stablehlo::ConvertElementType(target_f32, mlir::ElementType::BF16);
}

namespace {

bool IsFloat8Type(mlir::Type type) {
  return mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E4M3FNUZType,
                   mlir::Float8E5M2Type, mlir::Float8E5M2FNUZType,
                   mlir::Float8E8M0FNUType>(type);
}

bool IsAcBackwardModule(mlir::MlirOp op) {
  if (mlir::Region* region = op.getValue().getParentRegion()) {
    if (auto module_op = region->getParentOfType<mlir::ModuleOp>()) {
      if (std::optional<llvm::StringRef> name = module_op.getSymName()) {
        return name->ends_with("_ac_bwd");
      }
    }
  }
  return false;
}

mlir::MlirOp ChainDotPass(
    mlir::MlirOp acc, mlir::MlirOp lhs_f32, mlir::MlirOp rhs_f32,
    int lhs_part_idx, int rhs_part_idx,
    stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::ArrayAttr precision_config) {
  mlir::MlirOp lhs_part = GetBf16Part(lhs_f32, lhs_part_idx);
  mlir::MlirOp rhs_part = GetBf16Part(rhs_f32, rhs_part_idx);
  mlir::MlirOp dot =
      SingleDotF32(lhs_part, rhs_part, dot_dimension_numbers, precision_config);
  return AddShlo(acc, dot);
}

mlir::MlirOp BuildSplitBf16DotF32(
    mlir::MlirOp lhs_f32, mlir::MlirOp rhs_f32, int num_parts,
    stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::ArrayAttr precision_config) {
  if (IsAcBackwardModule(lhs_f32)) {
    return SingleDotF32(lhs_f32, rhs_f32, dot_dimension_numbers,
                        precision_config);
  }
  if (num_parts == 2) {
    // 4-pass exact BF16 split for F16 inputs (i <= 1, j <= 1), with per-pass
    // BF16 extraction fused directly into each DotGeneral loop.
    mlir::MlirOp lower_sum =
        SingleDotF32(GetBf16Part(lhs_f32, 1), GetBf16Part(rhs_f32, 1),
                     dot_dimension_numbers, precision_config);
    lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 0, 1,
                             dot_dimension_numbers, precision_config);
    lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 1, 0,
                             dot_dimension_numbers, precision_config);
    lower_sum = ZeroNaNs(lower_sum);
    mlir::MlirOp main_dot =
        SingleDotF32(GetBf16Part(lhs_f32, 0), GetBf16Part(rhs_f32, 0),
                     dot_dimension_numbers, precision_config);
    return AddShlo(main_dot, lower_sum);
  }

  // num_parts == 3: 6-pass BF16 split for F32 inputs (i + j <= 2), accumulated
  // from lowest to highest magnitude with per-pass BF16 extraction so XLA fuses
  // extraction into each DotGeneral loop with zero intermediate BF16 buffers.
  mlir::MlirOp lower_sum =
      SingleDotF32(GetBf16Part(lhs_f32, 1), GetBf16Part(rhs_f32, 1),
                   dot_dimension_numbers, precision_config);
  lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 2, 0,
                           dot_dimension_numbers, precision_config);
  lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 0, 2,
                           dot_dimension_numbers, precision_config);
  lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 1, 0,
                           dot_dimension_numbers, precision_config);
  lower_sum = ChainDotPass(lower_sum, lhs_f32, rhs_f32, 0, 1,
                           dot_dimension_numbers, precision_config);
  lower_sum = ZeroNaNs(lower_sum);
  mlir::MlirOp main_dot =
      SingleDotF32(GetBf16Part(lhs_f32, 0), GetBf16Part(rhs_f32, 0),
                   dot_dimension_numbers, precision_config);
  return AddShlo(main_dot, lower_sum);
}

}  // namespace

absl::StatusOr<mlir::MlirOp> CastOrConvertTo(
    mlir::MlirOp op, mlir::ElementType expected_output_type) {
  const mlir::RankedTensorType current_type = GetTensorTypeOrDie(op);
  TT_ASSIGN_OR_RETURN(
      const mlir::ElementType actual_element_type,
      ConvertTo<mlir::ElementType>(current_type.getElementType()));
  if (actual_element_type == expected_output_type) {
    return op;
  }
  const auto expected_scalar_type =
      ConvertTo<at::ScalarType>(expected_output_type);
  if (c10::isIntegralType(expected_scalar_type, /*includeBool=*/true)) {
    return stablehlo::ConvertElementType(op, expected_output_type);
  }
  return CastIfNeeded(op, expected_output_type);
}

absl::StatusOr<mlir::MlirOp> BuildAccurateDotGeneralAccumulator(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::RankedTensorType rhs_type = GetTensorTypeOrDie(rhs);

  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &lhs.getContext(), {precision, precision});
  mlir::Type elem_type = lhs_type.getElementType();

  // XLA doesn't support matmul with i64, so we convert them to f64.
  bool is_any_i64 =
      elem_type.isInteger(64) || rhs_type.getElementType().isInteger(64);
  if (is_any_i64) {
    lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::F64);
    rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::F64);
    return stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                                 precision_config);
  }

  if (IsFloat8Type(elem_type) || IsFloat8Type(rhs_type.getElementType())) {
    if (!elem_type.isBF16()) {
      lhs = stablehlo::ConvertElementType(lhs, mlir::ElementType::BF16);
    }
    if (!rhs_type.getElementType().isBF16()) {
      rhs = stablehlo::ConvertElementType(rhs, mlir::ElementType::BF16);
    }
    return SingleDotF32(lhs, rhs, dot_dimension_numbers, precision_config);
  }

  if (elem_type.isBF16()) {
    return SingleDotF32(lhs, rhs, dot_dimension_numbers, precision_config);
  }

  if (elem_type.isF16()) {
    mlir::MlirOp lhs_f32 =
        stablehlo::ConvertElementType(lhs, mlir::ElementType::F32);
    mlir::MlirOp rhs_f32 =
        stablehlo::ConvertElementType(rhs, mlir::ElementType::F32);
    return BuildSplitBf16DotF32(lhs_f32, rhs_f32, /*num_parts=*/2,
                                dot_dimension_numbers, precision_config);
  }

  if (elem_type.isF32()) {
    return BuildSplitBf16DotF32(lhs, rhs, /*num_parts=*/3,
                                dot_dimension_numbers, precision_config);
  }

  if (IsComplexType(lhs_type)) {
    mlir::MlirOp lhs_re = stablehlo::Real(lhs);
    if (GetTensorTypeOrDie(lhs_re).getElementType().isF32()) {
      mlir::MlirOp lhs_im = stablehlo::Imag(lhs);
      mlir::MlirOp rhs_re = stablehlo::Real(rhs);
      mlir::MlirOp rhs_im = stablehlo::Imag(rhs);

      mlir::MlirOp re_re =
          BuildSplitBf16DotF32(lhs_re, rhs_re, /*num_parts=*/3,
                               dot_dimension_numbers, precision_config);
      mlir::MlirOp im_im =
          BuildSplitBf16DotF32(lhs_im, rhs_im, /*num_parts=*/3,
                               dot_dimension_numbers, precision_config);
      mlir::MlirOp re_im =
          BuildSplitBf16DotF32(lhs_re, rhs_im, /*num_parts=*/3,
                               dot_dimension_numbers, precision_config);
      mlir::MlirOp im_re =
          BuildSplitBf16DotF32(lhs_im, rhs_re, /*num_parts=*/3,
                               dot_dimension_numbers, precision_config);

      mlir::MlirOp out_re = SubShlo(re_re, im_im);
      mlir::MlirOp out_im = AddShlo(re_im, im_re);
      return stablehlo::Complex(out_re, out_im);
    }
  }

  return stablehlo::DotGeneral(lhs, rhs, dot_dimension_numbers,
                               precision_config);
}

absl::StatusOr<mlir::MlirOp> BuildAccurateDotGeneral(
    mlir::MlirOp lhs, mlir::MlirOp rhs,
    stablehlo::DotDimensionNumbersAttr dot_dimension_numbers,
    mlir::ElementType out_dtype, stablehlo::Precision precision) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp acc,
                      BuildAccurateDotGeneralAccumulator(
                          lhs, rhs, dot_dimension_numbers, precision));
  return CastOrConvertTo(acc, out_dtype);
}

}  // namespace torch_tpu
