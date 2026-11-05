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

#include <complex>
#include <cstdint>
#include <limits>
#include <optional>

// We disable clang-format here to force the StableHLO Reference headers to
// be included before the TorchTPU StablehloBuilder headers. This prevents
// a namespace collision where both packages attempt to define 'Tuple'.
// clang-format off
#include "mlir/IR/Value.h"
#include "stablehlo/reference/Api.h"
#include "stablehlo/reference/Configuration.h"
// clang-format on

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/status_test_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::ExplainMatchResult;
using ::testing::HasSubstr;

constexpr int64_t kDyn = mlir::ShapedType::kDynamic;
constexpr mlir::stablehlo::Precision kDefaultPrecision =
    mlir::stablehlo::Precision::DEFAULT;  // EXPLICIT_PRECISION_OK=unit test

template <typename T>
struct EvaluatedTensor {
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<T> values;
};

template <typename T>
absl::StatusOr<EvaluatedTensor<T>> EvaluateModule(mlir::ModuleOp module) {
  if (mlir::failed(module.verify())) {
    return TT_ERROR(error::kInternal) << "MLIR module verification failed";
  }
  const mlir::stablehlo::InterpreterConfiguration config;
  const llvm::SmallVector<mlir::DenseElementsAttr> empty_inputs;
  const mlir::FailureOr<llvm::SmallVector<mlir::DenseElementsAttr>>
      eval_result = mlir::stablehlo::evalModule(module, empty_inputs, config);
  if (mlir::failed(eval_result) || eval_result->size() != 1) {
    return TT_ERROR(error::kInternal) << "StableHLO interpreter failed";
  }
  const mlir::DenseElementsAttr res_attr = (*eval_result)[0];
  const llvm::ArrayRef<int64_t> shape_ref = res_attr.getType().getShape();
  EvaluatedTensor<T> out;
  out.shape.assign(shape_ref.begin(), shape_ref.end());
  const auto vals = res_attr.getValues<T>();
  out.values.assign(vals.begin(), vals.end());
  return out;
}

MATCHER_P2(MatchesTensor, expected_shape, expected_values, "") {
  return ExplainMatchResult(ElementsAreArray(expected_shape), arg.shape,
                            result_listener) &&
         ExplainMatchResult(ElementsAreArray(expected_values), arg.values,
                            result_listener);
}

template <typename T>
auto EvaluatesTo(llvm::ArrayRef<int64_t> expected_shape,
                 llvm::ArrayRef<T> expected_values) {
  return IsOkAndHolds(MatchesTensor(
      llvm::SmallVector<int64_t>(expected_shape.begin(), expected_shape.end()),
      llvm::SmallVector<T>(expected_values.begin(), expected_values.end())));
}

MATCHER_P3(HasBoundedTensorType, expected_shape, expected_bounds,
           expected_elem_type, "") {
  const mlir::RankedTensorType type =
      mlir::cast<mlir::RankedTensorType>(arg.getType());
  if (type.getElementType() != expected_elem_type) {
    *result_listener << "element type mismatch";
    return false;
  }
  if (!ExplainMatchResult(ElementsAreArray(expected_shape), type.getShape(),
                          result_listener)) {
    return false;
  }
  const auto encoding =
      mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
          type.getEncoding());
  if (!encoding) {
    *result_listener << "missing TypeExtensionsAttr bounds encoding";
    return false;
  }
  return ExplainMatchResult(ElementsAreArray(expected_bounds),
                            encoding.getBounds(), result_listener);
}

MATCHER_P5(HasDotGeneralOperandBounds, expected_lhs_shape, expected_lhs_bounds,
           expected_rhs_shape, expected_rhs_bounds, expected_elem_type, "") {
  auto dot_op =
      arg.getValue().template getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  if (!dot_op) {
    *result_listener << "defining op is not stablehlo.dot_general";
    return false;
  }
  return ExplainMatchResult(
             HasBoundedTensorType(expected_lhs_shape, expected_lhs_bounds,
                                  expected_elem_type),
             dot_op.getLhs(), result_listener) &&
         ExplainMatchResult(
             HasBoundedTensorType(expected_rhs_shape, expected_rhs_bounds,
                                  expected_elem_type),
             dot_op.getRhs(), result_listener);
}

// Extracts and validates the shared `stablehlo.set_dimension_size` runtime
// size Value applied to both `lhs` (at `lhs_dim`) and `rhs` (at `rhs_dim`) on
// a `stablehlo.dot_general` op.
std::optional<mlir::Value> GetDotGeneralSharedReconciledSize(
    const mlir::MlirOp& op, int64_t lhs_dim, int64_t rhs_dim,
    testing::MatchResultListener* result_listener) {
  auto dot_op =
      op.getValue().template getDefiningOp<mlir::stablehlo::DotGeneralOp>();
  if (!dot_op) {
    *result_listener << "defining op is not stablehlo.dot_general";
    return std::nullopt;
  }
  auto lhs_sds =
      dot_op.getLhs()
          .template getDefiningOp<mlir::stablehlo::SetDimensionSizeOp>();
  auto rhs_sds =
      dot_op.getRhs()
          .template getDefiningOp<mlir::stablehlo::SetDimensionSizeOp>();
  if (!lhs_sds || !rhs_sds) {
    *result_listener
        << "dot_general operands are not defined by set_dimension_size";
    return std::nullopt;
  }
  if (static_cast<int64_t>(lhs_sds.getDimension()) != lhs_dim ||
      static_cast<int64_t>(rhs_sds.getDimension()) != rhs_dim) {
    *result_listener << "unexpected set_dimension_size dimensions: "
                     << lhs_sds.getDimension() << " and "
                     << rhs_sds.getDimension();
    return std::nullopt;
  }
  if (lhs_sds.getSize() != rhs_sds.getSize()) {
    *result_listener
        << "lhs and rhs do not share the same reconciled runtime size value";
    return std::nullopt;
  }
  return lhs_sds.getSize();
}

MATCHER_P4(MatchesDotGeneralReconciledMinOfTwoDynamicSizes, expected_lhs,
           lhs_dim, expected_rhs, rhs_dim, "") {
  const std::optional<mlir::Value> size_val =
      GetDotGeneralSharedReconciledSize(arg, lhs_dim, rhs_dim, result_listener);
  if (!size_val.has_value()) {
    return false;
  }
  auto min_op = size_val->getDefiningOp<mlir::stablehlo::MinOp>();
  if (!min_op) {
    *result_listener << "reconciled runtime size is not stablehlo.minimum";
    return false;
  }
  auto lhs_gds =
      min_op.getLhs()
          .template getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
  auto rhs_gds =
      min_op.getRhs()
          .template getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
  if (!lhs_gds || !rhs_gds) {
    *result_listener
        << "minimum operands are not both stablehlo.get_dimension_size";
    return false;
  }
  return lhs_gds.getOperand() == expected_lhs.getValue() &&
         static_cast<int64_t>(lhs_gds.getDimension()) == lhs_dim &&
         rhs_gds.getOperand() == expected_rhs.getValue() &&
         static_cast<int64_t>(rhs_gds.getDimension()) == rhs_dim;
}

// Matches a `stablehlo.dot_general` whose reconciled runtime size on `(lhs_dim,
// rhs_dim)` is `stablehlo.minimum(get_dimension_size(expected_lhs, lhs_dim),
// get_dimension_size(expected_rhs, rhs_dim))`.
auto HasDotGeneralReconciledMinOfTwoDynamicSizes(
    const mlir::MlirOp& expected_lhs, int64_t lhs_dim,
    const mlir::MlirOp& expected_rhs, int64_t rhs_dim) {
  return MatchesDotGeneralReconciledMinOfTwoDynamicSizes(expected_lhs, lhs_dim,
                                                         expected_rhs, rhs_dim);
}

MATCHER_P5(MatchesDotGeneralReconciledClampedDynamicSize, lhs_dim, rhs_dim,
           expected_dyn_op, dyn_dim, expected_bound, "") {
  const std::optional<mlir::Value> size_val =
      GetDotGeneralSharedReconciledSize(arg, lhs_dim, rhs_dim, result_listener);
  if (!size_val.has_value()) {
    return false;
  }
  auto min_op = size_val->getDefiningOp<mlir::stablehlo::MinOp>();
  if (!min_op) {
    *result_listener << "reconciled runtime size is not clamped via minimum";
    return false;
  }
  auto gds_op =
      min_op.getLhs()
          .template getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
  auto const_op =
      min_op.getRhs().template getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (!gds_op || !const_op) {
    *result_listener << "expected minimum(get_dimension_size, constant)";
    return false;
  }
  const auto dense_attr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!dense_attr || !dense_attr.isSplat() ||
      dense_attr.getSplatValue<int32_t>() != expected_bound) {
    *result_listener << "clamp constant does not equal " << expected_bound;
    return false;
  }
  return gds_op.getOperand() == expected_dyn_op.getValue() &&
         static_cast<int64_t>(gds_op.getDimension()) == dyn_dim;
}

// Matches a `stablehlo.dot_general` whose reconciled runtime size on `(lhs_dim,
// rhs_dim)` is `stablehlo.minimum(get_dimension_size(expected_dyn_op, dyn_dim),
// constant(expected_bound))`.
auto HasDotGeneralReconciledClampedDynamicSize(
    int64_t lhs_dim, int64_t rhs_dim, const mlir::MlirOp& expected_dyn_op,
    int64_t dyn_dim, int32_t expected_bound) {
  return MatchesDotGeneralReconciledClampedDynamicSize(
      lhs_dim, rhs_dim, expected_dyn_op, dyn_dim, expected_bound);
}

MATCHER_P4(MatchesDotGeneralReconciledUnclampedDynamicSize, lhs_dim, rhs_dim,
           expected_dyn_op, dyn_dim, "") {
  const std::optional<mlir::Value> size_val =
      GetDotGeneralSharedReconciledSize(arg, lhs_dim, rhs_dim, result_listener);
  if (!size_val.has_value()) {
    return false;
  }
  auto gds_op = size_val->getDefiningOp<mlir::stablehlo::GetDimensionSizeOp>();
  if (!gds_op) {
    *result_listener
        << "expected unclamped stablehlo.get_dimension_size runtime size";
    return false;
  }
  return gds_op.getOperand() == expected_dyn_op.getValue() &&
         static_cast<int64_t>(gds_op.getDimension()) == dyn_dim;
}

// Matches a `stablehlo.dot_general` whose reconciled runtime size on `(lhs_dim,
// rhs_dim)` is directly `get_dimension_size(expected_dyn_op, dyn_dim)` without
// a redundant `stablehlo.minimum` clamp.
auto HasDotGeneralReconciledUnclampedDynamicSize(
    int64_t lhs_dim, int64_t rhs_dim, const mlir::MlirOp& expected_dyn_op,
    int64_t dyn_dim) {
  return MatchesDotGeneralReconciledUnclampedDynamicSize(
      lhs_dim, rhs_dim, expected_dyn_op, dyn_dim);
}

mlir::MlirOp MakeDenseI32Constant(mlir::MlirBuilder& builder,
                                  llvm::ArrayRef<int64_t> shape,
                                  llvm::ArrayRef<int32_t> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI32Type());
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::MlirOp MakeDenseI64Constant(mlir::MlirBuilder& builder,
                                  llvm::ArrayRef<int64_t> shape,
                                  llvm::ArrayRef<int64_t> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI64Type());
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::MlirOp MakeDenseI8Constant(mlir::MlirBuilder& builder,
                                 llvm::ArrayRef<int64_t> shape,
                                 llvm::ArrayRef<int8_t> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI8Type());
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::MlirOp MakeDenseBoolConstant(mlir::MlirBuilder& builder,
                                   llvm::ArrayRef<int64_t> shape,
                                   llvm::ArrayRef<bool> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type =
      mlir::RankedTensorType::get(shape, op_builder.getI1Type());
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::MlirOp MakeDenseComplexF32Constant(
    mlir::MlirBuilder& builder, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<std::complex<float>> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type = mlir::RankedTensorType::get(
      shape, mlir::ComplexType::get(op_builder.getF32Type()));
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::MlirOp MakeDenseComplexF64Constant(
    mlir::MlirBuilder& builder, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<std::complex<double>> values) {
  mlir::OpBuilder& op_builder = builder.getOpBuilder();
  const mlir::RankedTensorType type = mlir::RankedTensorType::get(
      shape, mlir::ComplexType::get(op_builder.getF64Type()));
  return mlir::stablehlo::Constant(builder,
                                   mlir::DenseElementsAttr::get(type, values));
}

mlir::RankedTensorType MakeBoundedTensorType(mlir::MLIRContext* ctx,
                                             llvm::ArrayRef<int64_t> shape,
                                             llvm::ArrayRef<int64_t> bounds,
                                             mlir::Type elem_type) {
  const mlir::DenseI64ArrayAttr bounds_attr =
      mlir::DenseI64ArrayAttr::get(ctx, bounds);
  const mlir::stablehlo::TypeExtensionsAttr encoding =
      mlir::stablehlo::TypeExtensionsAttr::get(ctx, bounds_attr);
  return mlir::RankedTensorType::get(shape, elem_type, encoding);
}

class MatmulBuilderTest : public testing::Test {
 protected:
  MatmulBuilderTest()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  mlir::ModuleBuilder& module_builder() { return module_builder_; }

  template <typename Fn>
  absl::StatusOr<EvaluatedTensor<int32_t>> RunI32Module(Fn&& build_fn) {
    mlir::ModuleBuilder local_builder(context_, mlir::unknownLoc(context_));
    mlir::func::FunctionBuilder fn(local_builder, "main");
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp out, build_fn(fn));
    mlir::func::Return(fn, out);
    const mlir::OwningOpRef<mlir::ModuleOp> module = local_builder.build();
    return EvaluateModule<int32_t>(*module);
  }

  template <typename T, typename Fn>
  absl::StatusOr<EvaluatedTensor<T>> RunTypedModule(Fn&& build_fn) {
    mlir::ModuleBuilder local_builder(context_, mlir::unknownLoc(context_));
    mlir::func::FunctionBuilder fn(local_builder, "main");
    TT_ASSIGN_OR_RETURN(const mlir::MlirOp out, build_fn(fn));
    mlir::func::Return(fn, out);
    const mlir::OwningOpRef<mlir::ModuleOp> module = local_builder.build();
    return EvaluateModule<T>(*module);
  }

 private:
  mlir::MLIRContext context_;
  mlir::ModuleBuilder module_builder_;
};

TEST_F(MatmulBuilderTest, Forward1Dx1D) {
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
        const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {3}, {4, 5, 6});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({}, {32}));

  // 1D operands contract along axis 0 even if kTransposeLastTwo is passed.
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {2}, {3, 4});
                const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {2}, {5, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_lhs = MatrixTranspose::kTransposeLastTwo,
                     .transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({}, {39}));
}

TEST_F(MatmulBuilderTest, Forward2Dx1DAndTransposedLhs) {
  // Standard [2, 3] @ [3] -> [2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2}, {14, 32}));

  // Transposed lhs [3, 2]^T @ [3] -> [2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 4, 2, 5, 3, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_lhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2}, {14, 32}));
}

TEST_F(MatmulBuilderTest, Forward1Dx2DAndTransposedRhs) {
  // Standard [3] @ [3, 2] -> [2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2}, {22, 28}));

  // Transposed rhs [3] @ [2, 3]^T -> [2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 3, 5, 2, 4, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2}, {22, 28}));
}

TEST_F(MatmulBuilderTest, Forward2Dx2DAllTransposeCombinations) {
  // 1. kNone, kNone: [2, 3] @ [3, 2] -> [2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2}, {22, 28, 49, 64}));

  // 2. transpose_lhs only: [3, 2]^T @ [3, 2] -> [2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 4, 2, 5, 3, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_lhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2}, {22, 28, 49, 64}));

  // 3. transpose_rhs only: [2, 3] @ [2, 3]^T -> [2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 3, 5, 2, 4, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2}, {22, 28, 49, 64}));

  // 4. Both transposed: [3, 2]^T @ [2, 3]^T -> [2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 4, 2, 5, 3, 6});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 3, 5, 2, 4, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_lhs = MatrixTranspose::kTransposeLastTwo,
                     .transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2}, {22, 28, 49, 64}));
}

TEST_F(MatmulBuilderTest, Forward1Dx3DAnd2Dx3D) {
  // 1D x 3D: [3] @ [2, 3, 2] -> [2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                const mlir::MlirOp rhs = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2}, {4, 5, 5, 5}));

  // 2D x 3D: [2, 3] @ [2, 3, 2] -> [2, 2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                const mlir::MlirOp rhs = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {4, 5, 10, 11, 2, 4, 8, 10}));
}

TEST_F(MatmulBuilderTest, Forward3Dx1DFoldedAndUnfoldedVariants) {
  // 1. Default folded 3D x 1D ([2, 2, 3] @ [3] -> [2, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2}, {14, 32, 4, 2}));

  // 2. Unfolded via LeadingDimFolding::kNever
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.leading_dim_folding = LeadingDimFolding::kNever});
              }),
              EvaluatesTo<int32_t>({2, 2}, {14, 32, 4, 2}));

  // 3. Unfolded via transpose_lhs = kTransposeLastTwo ([2, 3, 2] @ [3] -> [2,
  // 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 4, 2, 5, 3, 6, 1, 0, 0, 1, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.transpose_lhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2}, {14, 32, 4, 2}));
}

TEST_F(MatmulBuilderTest,
       Forward3Dx2DEvaluatesAndAvoidsReshapesWhenTransposedOrUnfolded) {
  // 1. Folded default [2, 2, 3] @ [3, 2] -> [2, 2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4}));

  // 2. transpose_rhs = kTransposeLastTwo ([2, 2, 3] @ [2, 3]^T -> [2, 2, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs_t =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 3, 5, 2, 4, 6});
                return BuildMatmulShlo(
                    lhs, rhs_t, kDefaultPrecision,
                    {.transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4}));

  // 3. LeadingDimFolding::kNever ([2, 2, 3] @ [3, 2] -> [2, 2, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.leading_dim_folding = LeadingDimFolding::kNever});
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4}));

  // Verify zero reshapes for both transpose_rhs and LeadingDimFolding::kNever.
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::RankedTensorType lhs_type =
      mlir::RankedTensorType::get({2, 3, 4}, op_builder.getF32Type());
  const mlir::RankedTensorType rhs_transposed_type =
      mlir::RankedTensorType::get({5, 4}, op_builder.getF32Type());
  const mlir::RankedTensorType rhs_normal_type =
      mlir::RankedTensorType::get({4, 5}, op_builder.getF32Type());
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs_t = mlir::func::Argument(fn, rhs_transposed_type);
  const mlir::MlirOp rhs_n = mlir::func::Argument(fn, rhs_normal_type);
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_t,
      BuildMatmulShlo(lhs, rhs_t, kDefaultPrecision,
                      {.transpose_rhs = MatrixTranspose::kTransposeLastTwo}));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_n,
      BuildMatmulShlo(lhs, rhs_n, kDefaultPrecision,
                      {.leading_dim_folding = LeadingDimFolding::kNever}));
  mlir::func::Return(fn, {out_t, out_n});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  int reshape_count = 0;
  int dot_general_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
      ++reshape_count;
    } else if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
    }
  });
  EXPECT_EQ(reshape_count, 0);
  EXPECT_EQ(dot_general_count, 2);
}

TEST_F(MatmulBuilderTest,
       ForwardAndBackward3Dx3DUnbroadcastedEvaluatesAndHasZeroReshapes) {
  // Numerically evaluate B=2 unbroadcasted 3D x 3D forward, GradSelf,
  // GradOther.
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 0});
                const mlir::MlirOp rhs = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {4, 5, 10, 11, 4, 4, 5, 1}));

  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp self = MakeDenseI32Constant(
            fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 0});
        const mlir::MlirOp other = MakeDenseI32Constant(
            fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2});
        return BuildMatmulGradSelfShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 2, 3}, {1, 2, 3, 3, 4, 7, 2, 1, 0, 1, 0, 2}));

  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp self = MakeDenseI32Constant(
            fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 0});
        const mlir::MlirOp other = MakeDenseI32Constant(
            fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2});
        return BuildMatmulGradOtherShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 3, 2},
                           {13, 18, 17, 24, 21, 30, 2, 1, 0, 3, 1, 0}));

  // Verify B=1 3D bmm and backward emit zero reshapes.
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::RankedTensorType lhs_type =
      mlir::RankedTensorType::get({1, 3, 4}, op_builder.getF32Type());
  const mlir::RankedTensorType rhs_type =
      mlir::RankedTensorType::get({1, 4, 5}, op_builder.getF32Type());
  const mlir::RankedTensorType grad_type =
      mlir::RankedTensorType::get({1, 3, 5}, op_builder.getF32Type());
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, rhs_type);
  const mlir::MlirOp grad = mlir::func::Argument(fn, grad_type);
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs, rhs, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad, lhs, rhs, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad, lhs, rhs, kDefaultPrecision));
  mlir::func::Return(fn, {out, grad_self, grad_other});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  int reshape_count = 0;
  int dot_general_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ReshapeOp>(op)) {
      ++reshape_count;
    } else if (mlir::isa<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
    }
  });
  EXPECT_EQ(reshape_count, 0);
  EXPECT_EQ(dot_general_count, 3);
}

TEST_F(MatmulBuilderTest, ForwardAndBackward3Dx3DBroadcastBothDirections) {
  // Case A: lhs broadcasted ([1, 2, 3] @ [2, 3, 2] -> [2, 2, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
                const mlir::MlirOp rhs = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {4, 5, 10, 11, 2, 4, 8, 10}));

  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp other = MakeDenseI32Constant(
            fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 2, 3},
                                       other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({1, 2, 3}, {3, 2, 3, 3, 6, 7}));

  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(
                    fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradOtherShlo(grad, self,
                                                /*other_static_dims=*/{2, 3, 2},
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 3, 2},
                                   {13, 18, 17, 24, 21, 30, 1, 4, 2, 5, 3, 6}));

  // Case B: rhs broadcasted ([2, 2, 3] @ [1, 3, 2] -> [2, 2, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {1, 3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4}));

  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp other =
            MakeDenseI32Constant(fn, {1, 3, 2}, {1, 2, 3, 4, 5, 6});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 2, 3},
                                       other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 2, 3},
                           {5, 11, 17, 11, 25, 39, 1, 3, 5, 2, 4, 6}));

  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(
                    fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
                const mlir::MlirOp self = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                return BuildMatmulGradOtherShlo(grad, self,
                                                /*other_static_dims=*/{1, 3, 2},
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({1, 3, 2}, {14, 18, 17, 25, 22, 30}));

  // Case C: lhs [2, 3, 2]^T (effective [2, 2, 3]) @ rhs [1, 2, 3]^T (effective
  // [1, 3, 2]) -> [2, 2, 2] via general broadcasted EmitStaticBatchedMatmul,
  // exercising both GetLhsFreeMatrixDim and GetRhsFreeMatrixDim with
  // kTransposeLastTwo.
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs_t = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 4, 2, 5, 3, 6, 1, 0, 0, 1, 1, 0});
                const mlir::MlirOp rhs_t =
                    MakeDenseI32Constant(fn, {1, 2, 3}, {1, 3, 5, 2, 4, 6});
                return BuildMatmulShlo(
                    lhs_t, rhs_t, kDefaultPrecision,
                    MatmulShloOptions{
                        .transpose_lhs = MatrixTranspose::kTransposeLastTwo,
                        .transpose_rhs = MatrixTranspose::kTransposeLastTwo});
              }),
              EvaluatesTo<int32_t>({2, 2, 2}, {22, 28, 49, 64, 6, 8, 3, 4}));
}

TEST_F(MatmulBuilderTest,
       ForwardAndBackward5DNonDegenerateInterleavedAndCoalescedBatches) {
  // 5D x 5D with non-degenerate M=2, K=2, N=2 and interleaved batch axes:
  //   lhs shape:  [2, 1, 2, 2, 2]  (axes: kShared=2, kRhsOnly=1, kLhsOnly=2)
  //   rhs shape:  [2, 2, 1, 2, 2]  (axes: kShared=2, kRhsOnly=2, kLhsOnly=1)
  //   out shape:  [2, 2, 2, 2, 2]
  // Any transpose bug between M, N, and free batch axes will corrupt elements.
  const llvm::SmallVector<int32_t> lhs_vals = {// b0=0, b2=0: [[1, 2], [3, 4]]
                                               1, 2, 3, 4,
                                               // b0=0, b2=1: [[0, 1], [1, 0]]
                                               0, 1, 1, 0,
                                               // b0=1, b2=0: [[2, 0], [0, 3]]
                                               2, 0, 0, 3,
                                               // b0=1, b2=1: [[1, 1], [2, 3]]
                                               1, 1, 2, 3};
  const llvm::SmallVector<int32_t> rhs_vals = {// b0=0, b1=0: [[1, 0], [0, 1]]
                                               1, 0, 0, 1,
                                               // b0=0, b1=1: [[2, 1], [1, 3]]
                                               2, 1, 1, 3,
                                               // b0=1, b1=0: [[0, 2], [3, 1]]
                                               0, 2, 3, 1,
                                               // b0=1, b1=1: [[1, 2], [0, 1]]
                                               1, 2, 0, 1};
  const llvm::SmallVector<int32_t> expected_fwd = {
      // b0=0, b1=0, b2=0: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
      1, 2, 3, 4,
      // b0=0, b1=0, b2=1: [[0,1],[1,0]] @ I = [[0,1],[1,0]]
      0, 1, 1, 0,
      // b0=0, b1=1, b2=0: [[1,2],[3,4]] @ [[2,1],[1,3]] = [[4,7],[10,15]]
      4, 7, 10, 15,
      // b0=0, b1=1, b2=1: [[0,1],[1,0]] @ [[2,1],[1,3]] = [[1,3],[2,1]]
      1, 3, 2, 1,
      // b0=1, b1=0, b2=0: [[2,0],[0,3]] @ [[0,2],[3,1]] = [[0,4],[9,3]]
      0, 4, 9, 3,
      // b0=1, b1=0, b2=1: [[1,1],[2,3]] @ [[0,2],[3,1]] = [[3,3],[9,7]]
      3, 3, 9, 7,
      // b0=1, b1=1, b2=0: [[2,0],[0,3]] @ [[1,2],[0,1]] = [[2,4],[0,3]]
      2, 4, 0, 3,
      // b0=1, b1=1, b2=1: [[1,1],[2,3]] @ [[1,2],[0,1]] = [[1,3],[2,7]]
      1, 3, 2, 7};

  EXPECT_THAT(RunI32Module([&](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI32Constant(fn, {2, 1, 2, 2, 2}, lhs_vals);
                const mlir::MlirOp rhs =
                    MakeDenseI32Constant(fn, {2, 2, 1, 2, 2}, rhs_vals);
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 2, 2, 2}, expected_fwd));

  // Backward on the same 5D shapes with grad = expected_fwd:
  // For b0=0, b2=0: grad[0,0,0]=[[1,2],[3,4]], grad[0,1,0]=[[4,7],[10,15]]
  //   rhs[0,0,0]^T = [[1,0],[0,1]], rhs[0,1,0]^T = [[2,1],[1,3]]
  //   grad_self[0,0,0] = [[1,2],[3,4]]@I + [[4,7],[10,15]]@[[2,1],[1,3]]
  //                    = [[1,2],[3,4]] + [[15,25],[35,55]] = [[16,27],[38,59]]
  // For b0=0, b2=1: grad[0,0,1]=[[0,1],[1,0]], grad[0,1,1]=[[1,3],[2,1]]
  //   grad_self[0,0,1] = [[0,1],[1,0]] + [[5,10],[5,5]] = [[5,11],[6,5]]
  // For b0=1, b2=0: grad[1,0,0]=[[0,4],[9,3]], grad[1,1,0]=[[2,4],[0,3]]
  //   rhs[1,0,0]^T = [[0,3],[2,1]], rhs[1,1,0]^T = [[1,0],[2,1]]
  //   grad_self[1,0,0] = [[8,4],[6,30]] + [[10,4],[6,3]] = [[18,8],[12,33]]
  // For b0=1, b2=1: grad[1,0,1]=[[3,3],[9,7]], grad[1,1,1]=[[1,3],[2,7]]
  //   grad_self[1,0,1] = [[6,12],[14,34]] + [[7,3],[16,7]] = [[13,15],[30,41]]
  const llvm::SmallVector<int32_t> expected_grad_self = {
      16, 27, 38, 59, 5, 11, 6, 5, 18, 8, 12, 33, 13, 15, 30, 41};
  EXPECT_THAT(RunI32Module([&](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2, 2, 2, 2}, expected_fwd);
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {2, 1, 2, 2, 2}, lhs_vals);
                const mlir::MlirOp other =
                    MakeDenseI32Constant(fn, {2, 2, 1, 2, 2}, rhs_vals);
                return BuildMatmulGradSelfShlo(grad, self, other,
                                               kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 1, 2, 2, 2}, expected_grad_self));

  // GradOther: sum over b2 (axis 2):
  // b0=0, b1=0: lhs[0,0,0]^T@grad[0,0,0] + lhs[0,0,1]^T@grad[0,0,1]
  //   = [[1,3],[2,4]]@[[1,2],[3,4]] + [[0,1],[1,0]]@[[0,1],[1,0]]
  //   = [[10,14],[14,20]] + [[1,0],[0,1]] = [[11,14],[14,21]]
  // b0=0, b1=1: [[1,3],[2,4]]@[[4,7],[10,15]] + [[0,1],[1,0]]@[[1,3],[2,1]]
  //   = [[34,52],[48,74]] + [[2,1],[1,3]] = [[36,53],[49,77]]
  // b0=1, b1=0: [[2,0],[0,3]]@[[0,4],[9,3]] + [[1,2],[1,3]]@[[3,3],[9,7]]
  //   = [[0,8],[27,9]] + [[21,17],[30,24]] = [[21,25],[57,33]]
  // b0=1, b1=1: [[2,0],[0,3]]@[[2,4],[0,3]] + [[1,2],[1,3]]@[[1,3],[2,7]]
  //   = [[4,8],[0,9]] + [[5,17],[7,24]] = [[9,25],[7,33]]
  const llvm::SmallVector<int32_t> expected_grad_other = {
      11, 14, 14, 21, 36, 53, 49, 77, 21, 25, 57, 33, 9, 25, 7, 33};
  EXPECT_THAT(RunI32Module([&](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2, 2, 2, 2}, expected_fwd);
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {2, 1, 2, 2, 2}, lhs_vals);
                const mlir::MlirOp other =
                    MakeDenseI32Constant(fn, {2, 2, 1, 2, 2}, rhs_vals);
                return BuildMatmulGradOtherShlo(grad, self, other,
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 2, 1, 2, 2}, expected_grad_other));
}

TEST_F(MatmulBuilderTest,
       Backward1Dx1DAnd1Dx2DAnd2Dx1DAnd2Dx2DGradSelfAndGradOther) {
  // 1. 1D x 1D: self=[1,2,3], other=[4,5,6], grad=2
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseI32Constant(fn, {}, {2});
        const mlir::MlirOp self = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
        const mlir::MlirOp other = MakeDenseI32Constant(fn, {3}, {4, 5, 6});
        return BuildMatmulGradSelfShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({3}, {8, 10, 12}));

  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseI32Constant(fn, {}, {2});
        const mlir::MlirOp self = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
        const mlir::MlirOp other = MakeDenseI32Constant(fn, {3}, {4, 5, 6});
        return BuildMatmulGradOtherShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({3}, {2, 4, 6}));

  // 2. 1D x 2D: self=[3], other=[3, 2], grad=[2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2}, {2, 3});
                const mlir::MlirOp other =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{3},
                                               other, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3}, {8, 18, 28}));

  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2}, {2, 3});
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {3}, {1, 2, 3});
                return BuildMatmulGradOtherShlo(grad, self,
                                                /*other_static_dims=*/{3, 2},
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3, 2}, {2, 3, 4, 6, 6, 9}));

  // 3. 2D x 1D: self=[2, 3], other=[3], grad=[2]
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2}, {2, 3});
        const mlir::MlirOp other = MakeDenseI32Constant(fn, {3}, {4, 5, 6});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 3}, other,
                                       kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 3}, {8, 10, 12, 12, 15, 18}));

  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2}, {2, 3});
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradOtherShlo(
                    grad, self, /*other_static_dims=*/{3}, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3}, {14, 19, 24}));

  // 4. 2D x 2D: self=[2, 3], other=[3, 2], grad=[2, 2]
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
                const mlir::MlirOp other =
                    MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradSelfShlo(grad,
                                               /*self_static_dims=*/{2, 3},
                                               other, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 3}, {5, 11, 17, 11, 25, 39}));

  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradOtherShlo(grad, self,
                                                /*other_static_dims=*/{3, 2},
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3, 2}, {13, 18, 17, 24, 21, 30}));
}

TEST_F(MatmulBuilderTest, Backward3Dx2DAnd2Dx3DGradSelfAndGradOther) {
  // 1. 3D x 2D GradSelf ([2, 2, 2] @ [3, 2]^T -> [2, 2, 3])
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp self = MakeDenseI32Constant(
            fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
        const mlir::MlirOp other =
            MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
        return BuildMatmulGradSelfShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 2, 3},
                           {5, 11, 17, 11, 25, 39, 1, 3, 5, 2, 4, 6}));

  // 2. 3D x 2D GradOther ([2, 2, 3]^T @ [2, 2, 2] -> [3, 2])
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp self = MakeDenseI32Constant(
            fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
        const mlir::MlirOp other =
            MakeDenseI32Constant(fn, {3, 2}, {1, 2, 3, 4, 5, 6});
        return BuildMatmulGradOtherShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({3, 2}, {14, 18, 17, 25, 22, 30}));

  // 3. 2D x 3D GradSelf ([2, 2, 2] @ [2, 3, 2]^T summed over batch -> [2, 3])
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
        const mlir::MlirOp other = MakeDenseI32Constant(
            fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 3}, other,
                                       kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 3}, {3, 2, 3, 3, 6, 7}));

  // 4. 2D x 3D GradOther ([2, 3]^T @ [2, 2, 2] -> [2, 3, 2])
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(
                    fn, {2, 2, 2}, {1, 2, 3, 4, 1, 0, 0, 1});
                const mlir::MlirOp self =
                    MakeDenseI32Constant(fn, {2, 3}, {1, 2, 3, 4, 5, 6});
                return BuildMatmulGradOtherShlo(grad, self,
                                                /*other_static_dims=*/{2, 3, 2},
                                                kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 3, 2},
                                   {13, 18, 17, 24, 21, 30, 1, 4, 2, 5, 3, 6}));
}

TEST_F(MatmulBuilderTest, Backward1Dx3DAnd3Dx1DGradSelfAndGradOther) {
  // 1D x 3D GradSelf
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
                const mlir::MlirOp other = MakeDenseI32Constant(
                    fn, {2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0});
                return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{3},
                                               other, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3}, {7, 10, 3}));

  // 1D x 3D GradOther
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
        const mlir::MlirOp self = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
        return BuildMatmulGradOtherShlo(grad, self,
                                        /*other_static_dims=*/{2, 3, 2},
                                        kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 3, 2}, {1, 2, 2, 4, 3, 6, 3, 4, 6, 8, 9, 12}));

  // 3D x 1D GradSelf
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
        const mlir::MlirOp other = MakeDenseI32Constant(fn, {3}, {1, 2, 3});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 2, 3},
                                       other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 2, 3}, {1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12}));

  // 3D x 1D GradOther
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad =
                    MakeDenseI32Constant(fn, {2, 2}, {1, 2, 3, 4});
                const mlir::MlirOp self = MakeDenseI32Constant(
                    fn, {2, 2, 3}, {1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0});
                return BuildMatmulGradOtherShlo(
                    grad, self, /*other_static_dims=*/{3}, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({3}, {12, 16, 18}));
}

TEST_F(MatmulBuilderTest, ForwardComplexAllConjugateCombinations) {
  // lhs = [1+2i, 3-i], rhs = [2+i, 1+4i]
  // 1. conjugate_lhs only: conj([1+2i, 3-i]) . [2+i, 1+4i]
  //    = (1-2i)(2+i) + (3+i)(1+4i) = (4-3i) + (-1+13i) = 3+10i
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                               {.conjugate_lhs = ConjugateMode::kConjugate});
      }),
      EvaluatesTo<std::complex<float>>({}, {{3.0f, 10.0f}}));

  // 2. conjugate_rhs only: [1+2i, 3-i] . conj([2+i, 1+4i]) = conj(3+10i) =
  // 3-10i
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                               {.conjugate_rhs = ConjugateMode::kConjugate});
      }),
      EvaluatesTo<std::complex<float>>({}, {{3.0f, -10.0f}}));

  // 3. Both conjugated: conj([1+2i, 3-i]) . conj([2+i, 1+4i])
  //    = (1-2i)(2-i) + (3+i)(1-4i) = (0-5i) + (7-11i) = 7-16i
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                               {.conjugate_lhs = ConjugateMode::kConjugate,
                                .conjugate_rhs = ConjugateMode::kConjugate});
      }),
      EvaluatesTo<std::complex<float>>({}, {{7.0f, -16.0f}}));
}

TEST_F(MatmulBuilderTest, BackwardComplexWirtingerGradSelfAndGradOther) {
  // 2D x 2D Wirtinger GradSelf: grad @ conj(other)^T
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseComplexF32Constant(fn, {1, 1}, {{1.0f, 1.0f}});
        const mlir::MlirOp other = MakeDenseComplexF32Constant(
            fn, {2, 1}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 2}, other,
                                       kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<float>>({1, 2}, {{3.0f, 1.0f}, {5.0f, -3.0f}}));

  // 2D x 2D Wirtinger GradOther: conj(self)^T @ grad
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseComplexF32Constant(fn, {1, 1}, {{1.0f, 1.0f}});
        const mlir::MlirOp self = MakeDenseComplexF32Constant(
            fn, {1, 2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        return BuildMatmulGradOtherShlo(grad, self,
                                        /*other_static_dims=*/{2, 1},
                                        kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<float>>({2, 1}, {{3.0f, -1.0f}, {2.0f, 4.0f}}));

  // 1D x 1D Wirtinger GradSelf & GradOther: grad * conj(other), conj(self) *
  // grad
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad =
            MakeDenseComplexF32Constant(fn, {}, {{1.0f, 1.0f}});
        const mlir::MlirOp self =
            MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        const mlir::MlirOp other =
            MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulGradSelfShlo(grad, self, other, kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<float>>({2}, {{3.0f, 1.0f}, {5.0f, -3.0f}}));
}

TEST_F(MatmulBuilderTest,
       ComplexF64ForwardAndBackwardNumericalAndStructuralVerification) {
  // 1. Numerically evaluate C128 3D x 2D forward:
  //    lhs [1, 1, 2] = [[[1+2i, 3-i]]], rhs [2, 1] = [[2+i], [1+4i]]
  //    (1+2i)(2+i) + (3-i)(1+4i) = (0+5i) + (7+11i) = 7+16i
  EXPECT_THAT(
      RunTypedModule<std::complex<double>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs = MakeDenseComplexF64Constant(
            fn, {1, 1, 2}, {{1.0, 2.0}, {3.0, -1.0}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF64Constant(fn, {2, 1}, {{2.0, 1.0}, {1.0, 4.0}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<double>>({1, 1, 1}, {{7.0, 16.0}}));

  // 2. Numerically evaluate C128 backward with kept broadcast reduction
  //    (`reduce_dims != squeeze_dims`):
  //    self [1, 1, 2], other [2, 2, 1], grad [2, 1, 1]
  //    Batch 0: grad={{1+i}}, other={{2+i}, {1+4i}}
  //      -> grad @ conj(other)^T = [(1+i)(2-i), (1+i)(1-4i)] = [3+i, 5-3i]
  //    Batch 1: grad={{2+0i}}, other={{1-i}, {0+2i}}
  //      -> grad @ conj(other)^T = [2(1+i), 2(0-2i)] = [2+2i, 0-4i]
  //    Sum over batch axis 0 -> [[[5+3i, 5-7i]]]
  EXPECT_THAT(
      RunTypedModule<std::complex<double>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseComplexF64Constant(
            fn, {2, 1, 1}, {{1.0, 1.0}, {2.0, 0.0}});
        const mlir::MlirOp other = MakeDenseComplexF64Constant(
            fn, {2, 2, 1}, {{2.0, 1.0}, {1.0, 4.0}, {1.0, -1.0}, {0.0, 2.0}});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 1, 2},
                                       other, kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<double>>({1, 1, 2}, {{5.0, 3.0}, {5.0, -7.0}}));

  // 3. Numerically evaluate C128 2D x 3D forward and backward (`GradSelf` with
  //    dropped broadcast reduction `reduce_dims == squeeze_dims`, and
  //    `GradOther` broadcasting 2D `self` to 3D):
  //    self [1, 2], other [2, 2, 1], grad [2, 1, 1]
  EXPECT_THAT(
      RunTypedModule<std::complex<double>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF64Constant(fn, {1, 2}, {{1.0, 2.0}, {3.0, -1.0}});
        const mlir::MlirOp rhs = MakeDenseComplexF64Constant(
            fn, {2, 2, 1}, {{2.0, 1.0}, {1.0, 4.0}, {1.0, -1.0}, {0.0, 2.0}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<double>>({2, 1, 1}, {{7.0, 16.0}, {5.0, 7.0}}));

  EXPECT_THAT(
      RunTypedModule<std::complex<double>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseComplexF64Constant(
            fn, {2, 1, 1}, {{1.0, 1.0}, {2.0, 0.0}});
        const mlir::MlirOp other = MakeDenseComplexF64Constant(
            fn, {2, 2, 1}, {{2.0, 1.0}, {1.0, 4.0}, {1.0, -1.0}, {0.0, 2.0}});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{1, 2}, other,
                                       kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<double>>({1, 2}, {{5.0, 3.0}, {5.0, -7.0}}));

  EXPECT_THAT(
      RunTypedModule<std::complex<double>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseComplexF64Constant(
            fn, {2, 1, 1}, {{1.0, 1.0}, {2.0, 0.0}});
        const mlir::MlirOp self =
            MakeDenseComplexF64Constant(fn, {1, 2}, {{1.0, 2.0}, {3.0, -1.0}});
        return BuildMatmulGradOtherShlo(
            grad, self, /*other_static_dims=*/{2, 2, 1}, kDefaultPrecision);
      }),
      EvaluatesTo<std::complex<double>>(
          {2, 2, 1}, {{3.0, -1.0}, {2.0, 4.0}, {2.0, -4.0}, {6.0, 2.0}}));

  // 4. Verify C128 preferred_element_type = F32 downcasts output to COMPLEXF32.
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF64Constant(fn, {2}, {{1.0, 2.0}, {3.0, -1.0}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF64Constant(fn, {2}, {{2.0, 1.0}, {1.0, 4.0}});
        return BuildTypedDotGeneral(
            lhs, rhs, /*lhs_batching=*/{}, /*rhs_batching=*/{},
            /*lhs_contracting=*/{0}, /*rhs_contracting=*/{0}, kDefaultPrecision,
            mlir::ElementType::F32);
      }),
      EvaluatesTo<std::complex<float>>({}, {{7.0f, 16.0f}}));

  // 5. Structural verification: 3D x 2D C128 and bounded-dynamic C128 backward
  //    emit zero C128 reshapes and zero SliceOps.
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::ComplexType c128_type =
      mlir::ComplexType::get(op_builder.getF64Type());
  const mlir::MlirOp lhs_3d = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 3, 4}, c128_type));
  const mlir::MlirOp rhs_2d =
      mlir::func::Argument(fn, mlir::RankedTensorType::get({4, 5}, c128_type));
  const mlir::MlirOp grad_3d = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 3, 5}, c128_type));
  const mlir::MlirOp lhs_1 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({1, 3, 4}, c128_type));
  const mlir::MlirOp rhs_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 4, 5},
                                {8, kDyn, kDyn}, c128_type));
  const mlir::MlirOp grad_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 3, 5},
                                {8, kDyn, kDyn}, c128_type));
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out,
                          BuildMatmulShlo(lhs_3d, rhs_2d, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self,
      BuildMatmulGradSelfShlo(grad_3d, lhs_3d, rhs_2d, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other,
      BuildMatmulGradOtherShlo(grad_3d, lhs_3d, rhs_2d, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self_dyn,
      BuildMatmulGradSelfShlo(grad_dyn, lhs_1, rhs_dyn, kDefaultPrecision));
  mlir::func::Return(fn, {out, grad_self, grad_other, grad_self_dyn});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  int slice_count = 0;
  int c128_reshape_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::SliceOp>(op)) {
      ++slice_count;
    } else if (auto reshape_op =
                   mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op)) {
      if (auto complex_t = mlir::dyn_cast<mlir::ComplexType>(
              reshape_op.getType().getElementType());
          complex_t && complex_t.getElementType().isF64()) {
        ++c128_reshape_count;
      }
    }
  });
  EXPECT_EQ(slice_count, 0);
  EXPECT_EQ(c128_reshape_count, 0);
}

TEST_F(MatmulBuilderTest,
       ForwardI64LargeValuesPreserve53BitPrecisionAndSupportTargetDtype) {
  // 2^24 + 1 = 16777217 cannot be represented exactly in f32 or bf16, but is
  // exact in f64 with Precision::HIGHEST:
  //   16777217 * 1 + 16777217 * 1 = 33554434.
  constexpr int64_t kLarge = (1LL << 24) + 1;
  EXPECT_THAT(RunTypedModule<int64_t>([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseI64Constant(fn, {1, 2}, {kLarge, kLarge});
                const mlir::MlirOp rhs =
                    MakeDenseI64Constant(fn, {2, 1}, {1, 1});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int64_t>({1, 1}, {2 * kLarge}));

  // i64 with target_element_type = F32 converts the result to F32.
  EXPECT_THAT(RunTypedModule<float>([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI64Constant(fn, {2}, {3, 4});
                const mlir::MlirOp rhs = MakeDenseI64Constant(fn, {2}, {5, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.target_element_type = mlir::ElementType::F32});
              }),
              EvaluatesTo<float>({}, {39.0f}));
}

TEST_F(MatmulBuilderTest,
       ForwardSub64BitIntegerBoolAndComplexTargetElementTypePromotion) {
  // 1. i8 with target_element_type = F32 pre-converts operands to F32.
  EXPECT_THAT(RunTypedModule<float>([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI8Constant(fn, {2}, {3, 4});
                const mlir::MlirOp rhs = MakeDenseI8Constant(fn, {2}, {5, 6});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.target_element_type = mlir::ElementType::F32});
              }),
              EvaluatesTo<float>({}, {39.0f}));

  // 2. bool (i1) with target_element_type = F32 pre-converts to F32.
  EXPECT_THAT(RunTypedModule<float>([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs =
                    MakeDenseBoolConstant(fn, {3}, {true, false, true});
                const mlir::MlirOp rhs =
                    MakeDenseBoolConstant(fn, {3}, {true, true, true});
                return BuildMatmulShlo(
                    lhs, rhs, kDefaultPrecision,
                    {.target_element_type = mlir::ElementType::F32});
              }),
              EvaluatesTo<float>({}, {2.0f}));

  // 3. complex<f32> with target_element_type = F32 maps preferred_element_type
  //    to COMPLEXF32.
  EXPECT_THAT(
      RunTypedModule<std::complex<float>>([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp lhs =
            MakeDenseComplexF32Constant(fn, {2}, {{1.0f, 2.0f}, {3.0f, -1.0f}});
        const mlir::MlirOp rhs =
            MakeDenseComplexF32Constant(fn, {2}, {{2.0f, 1.0f}, {1.0f, 4.0f}});
        return BuildMatmulShlo(lhs, rhs, kDefaultPrecision,
                               {.target_element_type = mlir::ElementType::F32});
      }),
      EvaluatesTo<std::complex<float>>({}, {{7.0f, 16.0f}}));

  // 4. BF16 with target_element_type = F32 uses preferred_element_type on
  //    DotGeneralOp without pre-converting operands.
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::RankedTensorType lhs_type =
      mlir::RankedTensorType::get({2, 3}, op_builder.getBF16Type());
  const mlir::RankedTensorType rhs_type =
      mlir::RankedTensorType::get({3, 4}, op_builder.getBF16Type());
  const mlir::MlirOp lhs = mlir::func::Argument(fn, lhs_type);
  const mlir::MlirOp rhs = mlir::func::Argument(fn, rhs_type);
  const MatmulPrecisionPair asym_prec{
      mlir::stablehlo::Precision::HIGHEST,   // EXPLICIT_PRECISION_OK=unit test
      mlir::stablehlo::Precision::DEFAULT};  // EXPLICIT_PRECISION_OK=unit test
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out,
      BuildMatmulShlo(lhs, rhs, asym_prec,
                      {.target_element_type = mlir::ElementType::F32}));
  mlir::func::Return(fn, out);
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  const mlir::RankedTensorType result_type =
      mlir::cast<mlir::RankedTensorType>(out.getType());
  EXPECT_TRUE(result_type.getElementType().isF32());
  int convert_count = 0;
  int dot_general_count = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::stablehlo::ConvertOp>(op)) {
      ++convert_count;
    } else if (auto dot_op =
                   mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op)) {
      ++dot_general_count;
      const std::optional<mlir::ArrayAttr> prec_attr =
          dot_op.getPrecisionConfig();
      ASSERT_TRUE(prec_attr.has_value());
      ASSERT_EQ(prec_attr->size(), 2u);
      EXPECT_EQ(
          mlir::cast<mlir::stablehlo::PrecisionAttr>((*prec_attr)[0])
              .getValue(),
          mlir::stablehlo::Precision::HIGHEST);  // EXPLICIT_PRECISION_OK=test
      EXPECT_EQ(
          mlir::cast<mlir::stablehlo::PrecisionAttr>((*prec_attr)[1])
              .getValue(),
          mlir::stablehlo::Precision::DEFAULT);  // EXPLICIT_PRECISION_OK=test
    }
  });
  EXPECT_EQ(convert_count, 0);
  EXPECT_EQ(dot_general_count, 1);
}

TEST_F(MatmulBuilderTest, ZeroExtentForwardAndAllFourBackwardZeroGradBranches) {
  // 1. Forward zero contracting dim K=0 ([2, 0] @ [0, 3] -> [2, 3] of zeros)
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp lhs = MakeDenseI32Constant(fn, {2, 0}, {});
                const mlir::MlirOp rhs = MakeDenseI32Constant(fn, {0, 3}, {});
                return BuildMatmulShlo(lhs, rhs, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 3}, {0, 0, 0, 0, 0, 0}));

  // 2. Backward Branch 1: target_static_dims has 0, target_op is nullopt
  EXPECT_THAT(
      RunI32Module([](mlir::func::FunctionBuilder& fn) {
        const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 0, 4}, {});
        const mlir::MlirOp other = MakeDenseI32Constant(
            fn, {2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        return BuildMatmulGradSelfShlo(grad, /*self_static_dims=*/{2, 0, 3},
                                       other, kDefaultPrecision);
      }),
      EvaluatesTo<int32_t>({2, 0, 3}, {}));

  // 3. Backward Branch 3: target_static_dims non-zero ([2, 3]), other has 0,
  //    target_op is nullopt -> static zero-filled constant [2, 3].
  EXPECT_THAT(RunI32Module([](mlir::func::FunctionBuilder& fn) {
                const mlir::MlirOp grad = MakeDenseI32Constant(fn, {2, 0}, {});
                const mlir::MlirOp other = MakeDenseI32Constant(fn, {3, 0}, {});
                return BuildMatmulGradSelfShlo(grad,
                                               /*self_static_dims=*/{2, 3},
                                               other, kDefaultPrecision);
              }),
              EvaluatesTo<int32_t>({2, 3}, {0, 0, 0, 0, 0, 0}));

  // 4. Forward dynamic + zero-sized M=0, Backward Branch 2 (target_static_dims
  //    has 0 + bounded target_op), and Backward Branch 4 (target non-zero +
  //    bounded target_op -> chlo.constant_like).
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::Type i1_type = op_builder.getI1Type();
  const mlir::Type f32_type = op_builder.getF32Type();
  const mlir::MlirOp lhs_i1 = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 0, 3},
                                {4, kDyn, kDyn}, i1_type));
  const mlir::MlirOp rhs_i1 = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 3, 5},
                                {4, kDyn, kDyn}, i1_type));
  const mlir::MlirOp self_f32_non_empty = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 2, 3},
                                {4, kDyn, kDyn}, f32_type));
  const mlir::MlirOp self_f32_zero = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 0, 3},
                                {4, kDyn, kDyn}, f32_type));
  const mlir::MlirOp other_f32_zero = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 3, 0},
                                {4, kDyn, kDyn}, f32_type));
  const mlir::MlirOp grad_f32_zero = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 2, 0},
                                {4, kDyn, kDyn}, f32_type));
  const mlir::MlirOp grad_f32_zero_m = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 0, 0},
                                {4, kDyn, kDyn}, f32_type));

  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out_i1,
                          BuildMatmulShlo(lhs_i1, rhs_i1, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self_zero_target,
      BuildMatmulGradSelfShlo(grad_f32_zero_m, self_f32_zero, other_f32_zero,
                              kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self_non_empty_target,
      BuildMatmulGradSelfShlo(grad_f32_zero, self_f32_non_empty, other_f32_zero,
                              kDefaultPrecision));
  mlir::func::Return(
      fn, {out_i1, grad_self_zero_target, grad_self_non_empty_target});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));

  EXPECT_THAT(out_i1, HasBoundedTensorType(
                          llvm::ArrayRef<int64_t>{kDyn, 0, 5},
                          llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, i1_type));
  EXPECT_THAT(
      grad_self_zero_target,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 0, 3},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
  EXPECT_THAT(
      grad_self_non_empty_target,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 2, 3},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
}

TEST_F(MatmulBuilderTest,
       BoundedDynamicScalar1Dx1DAndMultiAxisReconciliationAndBroadcasts) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::Type f32_type = op_builder.getF32Type();
  const mlir::Type i64_type = op_builder.getI64Type();

  // 1. 1D x 1D bounded dynamic f32 and i64 produce rank-0 scalar with no
  //    TypeExtensionsAttr encoding.
  const mlir::MlirOp vec_f32 = mlir::func::Argument(
      fn,
      MakeBoundedTensorType(op_builder.getContext(), {kDyn}, {30}, f32_type));
  const mlir::MlirOp vec_i64 = mlir::func::Argument(
      fn,
      MakeBoundedTensorType(op_builder.getContext(), {kDyn}, {30}, i64_type));
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out_1d_f32,
                          BuildMatmulShlo(vec_f32, vec_f32, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out_1d_i64,
                          BuildMatmulShlo(vec_i64, vec_i64, kDefaultPrecision));
  const mlir::RankedTensorType out_f32_t =
      mlir::cast<mlir::RankedTensorType>(out_1d_f32.getType());
  const mlir::RankedTensorType out_i64_t =
      mlir::cast<mlir::RankedTensorType>(out_1d_i64.getType());
  EXPECT_EQ(out_f32_t.getRank(), 0);
  EXPECT_EQ(out_f32_t.getEncoding(), nullptr);
  EXPECT_EQ(out_i64_t.getRank(), 0);
  EXPECT_TRUE(out_i64_t.getElementType().isInteger(64));
  EXPECT_EQ(out_i64_t.getEncoding(), nullptr);

  // 2. Multi-axis bounded dynamic [?<=8, ?<=6, 3] @ [?<=4, 3, 5] ->
  //    [?<=4, ?<=6, 5]: verifies ResizeBoundedDimension on axis 0 preserves
  //    the dynamic bound on axis 1.
  const mlir::MlirOp lhs_multi_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, kDyn, 3},
                                {8, 6, kDyn}, f32_type));
  const mlir::MlirOp rhs_b4 = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 3, 5},
                                {4, kDyn, kDyn}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_multi_dyn,
      BuildMatmulShlo(lhs_multi_dyn, rhs_b4, kDefaultPrecision));
  EXPECT_THAT(
      out_multi_dyn,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, kDyn, 5},
                           llvm::ArrayRef<int64_t>{4, 6, kDyn}, f32_type));

  // 3. Bounded dynamic ?<=1 broadcasting against ?<=4:
  //    [?<=1, 2, 3] @ [?<=4, 3, 5] -> [?<=4, 2, 5]
  const mlir::MlirOp lhs_b1_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 2, 3},
                                {1, kDyn, kDyn}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_b1_bcast,
      BuildMatmulShlo(lhs_b1_dyn, rhs_b4, kDefaultPrecision));
  EXPECT_THAT(
      out_b1_bcast,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 2, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));

  // 4. Bounded dynamic real f32 backward with broadcast reduction:
  //    grad [?<=4, 2, 5], self [1, 2, 3], other [?<=4, 3, 5]
  //    -> grad_self [1, 2, 3] (static), grad_other [?<=4, 3, 5] (bounded).
  const mlir::MlirOp self_1_static = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({1, 2, 3}, f32_type));
  const mlir::MlirOp grad_b4 = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 2, 5},
                                {4, kDyn, kDyn}, f32_type));
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp grad_self_reduced,
                          BuildMatmulGradSelfShlo(grad_b4, self_1_static,
                                                  rhs_b4, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp grad_other_b4,
                          BuildMatmulGradOtherShlo(grad_b4, self_1_static,
                                                   rhs_b4, kDefaultPrecision));
  EXPECT_THAT(GetStaticShape(grad_self_reduced), ElementsAreArray({1, 2, 3}));
  EXPECT_THAT(
      grad_other_b4,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 3, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));

  mlir::func::Return(fn, {out_1d_f32, out_1d_i64, out_multi_dyn, out_b1_bcast,
                          grad_self_reduced, grad_other_b4});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();
  ASSERT_TRUE(mlir::succeeded(module->verify()));
}

TEST_F(MatmulBuilderTest, ReconcilesBoundedDynamicContractingDimensions) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  mlir::MLIRContext* ctx = op_builder.getContext();
  const mlir::Type f32_type = op_builder.getF32Type();
  const mlir::MlirOp lhs_k8 = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {2, kDyn}, {kDyn, 8}, f32_type));
  const mlir::MlirOp rhs_k4 = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 5}, {4, kDyn}, f32_type));
  const mlir::MlirOp lhs_k4_static =
      mlir::func::Argument(fn, mlir::RankedTensorType::get({2, 4}, f32_type));
  const mlir::MlirOp rhs_k8 = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 5}, {8, kDyn}, f32_type));
  const mlir::MlirOp rhs_k4_static =
      mlir::func::Argument(fn, mlir::RankedTensorType::get({4, 5}, f32_type));

  const mlir::MlirOp lhs_k4 = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {2, kDyn}, {kDyn, 4}, f32_type));

  TT_ASSERT_OK_AND_ASSIGN(const mlir::MlirOp out_k_dyn_dyn,
                          BuildMatmulShlo(lhs_k8, rhs_k4, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_k_static_dyn,
      BuildMatmulShlo(lhs_k4_static, rhs_k8, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_k_dyn_static,
      BuildMatmulShlo(lhs_k8, rhs_k4_static, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_k_dyn_eq_static,
      BuildMatmulShlo(lhs_k4, rhs_k4_static, kDefaultPrecision));
  mlir::func::Return(fn, {out_k_dyn_dyn, out_k_static_dyn, out_k_dyn_static,
                          out_k_dyn_eq_static});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  ASSERT_TRUE(mlir::succeeded(module->verify()));
  EXPECT_THAT(GetStaticShape(out_k_dyn_dyn), ElementsAreArray({2, 5}));
  EXPECT_THAT(GetStaticShape(out_k_static_dyn), ElementsAreArray({2, 5}));
  EXPECT_THAT(GetStaticShape(out_k_dyn_static), ElementsAreArray({2, 5}));
  EXPECT_THAT(GetStaticShape(out_k_dyn_eq_static), ElementsAreArray({2, 5}));
  EXPECT_THAT(out_k_dyn_dyn, HasDotGeneralOperandBounds(
                                 llvm::ArrayRef<int64_t>{2, kDyn},
                                 llvm::ArrayRef<int64_t>{kDyn, 4},
                                 llvm::ArrayRef<int64_t>{kDyn, 5},
                                 llvm::ArrayRef<int64_t>{4, kDyn}, f32_type));
  EXPECT_THAT(out_k_dyn_dyn, HasDotGeneralReconciledMinOfTwoDynamicSizes(
                                 lhs_k8, /*lhs_dim=*/1, rhs_k4, /*rhs_dim=*/0));
  EXPECT_THAT(
      out_k_static_dyn,
      HasDotGeneralOperandBounds(llvm::ArrayRef<int64_t>{2, kDyn},
                                 llvm::ArrayRef<int64_t>{kDyn, 4},
                                 llvm::ArrayRef<int64_t>{kDyn, 5},
                                 llvm::ArrayRef<int64_t>{4, kDyn}, f32_type));
  EXPECT_THAT(out_k_static_dyn,
              HasDotGeneralReconciledClampedDynamicSize(
                  /*lhs_dim=*/1, /*rhs_dim=*/0, rhs_k8, /*dyn_dim=*/0,
                  /*expected_bound=*/4));
  EXPECT_THAT(
      out_k_dyn_static,
      HasDotGeneralOperandBounds(llvm::ArrayRef<int64_t>{2, kDyn},
                                 llvm::ArrayRef<int64_t>{kDyn, 4},
                                 llvm::ArrayRef<int64_t>{kDyn, 5},
                                 llvm::ArrayRef<int64_t>{4, kDyn}, f32_type));
  EXPECT_THAT(out_k_dyn_static,
              HasDotGeneralReconciledClampedDynamicSize(
                  /*lhs_dim=*/1, /*rhs_dim=*/0, lhs_k8, /*dyn_dim=*/1,
                  /*expected_bound=*/4));
  EXPECT_THAT(out_k_dyn_eq_static,
              HasDotGeneralReconciledUnclampedDynamicSize(
                  /*lhs_dim=*/1, /*rhs_dim=*/0, lhs_k4, /*dyn_dim=*/1));
}

TEST_F(MatmulBuilderTest,
       ReconcilesMixedStaticAndBoundedDynamicBatchDimensions) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  mlir::MLIRContext* ctx = op_builder.getContext();
  const mlir::Type f32_type = op_builder.getF32Type();
  const mlir::MlirOp lhs_b4_static = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4, 2, 3}, f32_type));
  const mlir::MlirOp rhs_b8_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 3, 5}, {8, kDyn, kDyn}, f32_type));
  const mlir::MlirOp lhs_b8_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 2, 3}, {8, kDyn, kDyn}, f32_type));
  const mlir::MlirOp rhs_b4_static = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4, 3, 5}, f32_type));

  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_b_static_dyn,
      BuildMatmulShlo(lhs_b4_static, rhs_b8_dyn, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_b_dyn_static,
      BuildMatmulShlo(lhs_b8_dyn, rhs_b4_static, kDefaultPrecision));
  mlir::func::Return(fn, {out_b_static_dyn, out_b_dyn_static});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  ASSERT_TRUE(mlir::succeeded(module->verify()));
  EXPECT_THAT(
      out_b_static_dyn,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 2, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
  EXPECT_THAT(
      out_b_dyn_static,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 2, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
}

TEST_F(MatmulBuilderTest,
       BroadcastsMixedRank2Dx3DBoundedDynamicBatchDimensions) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  mlir::MLIRContext* ctx = op_builder.getContext();
  const mlir::Type f32_type = op_builder.getF32Type();
  const mlir::MlirOp lhs_2d =
      mlir::func::Argument(fn, mlir::RankedTensorType::get({2, 3}, f32_type));
  const mlir::MlirOp rhs_b4_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 3, 5}, {4, kDyn, kDyn}, f32_type));
  const mlir::MlirOp grad_b4_dyn = mlir::func::Argument(
      fn, MakeBoundedTensorType(ctx, {kDyn, 2, 5}, {4, kDyn, kDyn}, f32_type));

  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp out_2d_3d_dyn,
      BuildMatmulShlo(lhs_2d, rhs_b4_dyn, kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_self_2d,
      BuildMatmulGradSelfShlo(grad_b4_dyn, lhs_2d, rhs_b4_dyn,
                              kDefaultPrecision));
  TT_ASSERT_OK_AND_ASSIGN(
      const mlir::MlirOp grad_other_3d_dyn,
      BuildMatmulGradOtherShlo(grad_b4_dyn, lhs_2d, rhs_b4_dyn,
                               kDefaultPrecision));
  mlir::func::Return(fn, {out_2d_3d_dyn, grad_self_2d, grad_other_3d_dyn});
  mlir::OwningOpRef<mlir::ModuleOp> module = module_builder().build();

  ASSERT_TRUE(mlir::succeeded(module->verify()));
  EXPECT_THAT(
      out_2d_3d_dyn,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 2, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
  EXPECT_THAT(GetStaticShape(grad_self_2d), ElementsAreArray({2, 3}));
  EXPECT_THAT(
      grad_other_3d_dyn,
      HasBoundedTensorType(llvm::ArrayRef<int64_t>{kDyn, 3, 5},
                           llvm::ArrayRef<int64_t>{4, kDyn, kDyn}, f32_type));
}

TEST_F(MatmulBuilderTest, RejectsZeroRankAndMismatchedElementTypes) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::MlirOp scalar = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({}, op_builder.getF32Type()));
  const mlir::MlirOp vec_f32 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4}, op_builder.getF32Type()));
  const mlir::MlirOp vec_i32 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4}, op_builder.getI32Type()));

  EXPECT_THAT(BuildMatmulShlo(scalar, vec_f32, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("both arguments to matmul need to be at least "
                                 "1D")));
  EXPECT_THAT(BuildMatmulShlo(vec_f32, scalar, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("both arguments to matmul need to be at least "
                                 "1D")));
  EXPECT_THAT(BuildMatmulShlo(vec_f32, vec_i32, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected both arguments to have the same "
                                 "element type")));

  // Backward rank-0 and dtype mismatch checks:
  EXPECT_THAT(BuildMatmulGradSelfShlo(scalar, absl::Span<const int64_t>{},
                                      vec_f32, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("both arguments to matmul backward need to be "
                                 "at least 1D")));
  EXPECT_THAT(BuildMatmulGradSelfShlo(vec_f32, /*self_static_dims=*/{4},
                                      vec_i32, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected grad and matmul operands to have "
                                 "the same element type")));
  EXPECT_THAT(BuildMatmulGradSelfShlo(vec_f32, /*self_static_dims=*/{4},
                                      vec_f32, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("expected grad rank 0 for 1D x 1D matmul "
                                 "backward, got 1")));
}

TEST_F(MatmulBuilderTest, RejectsInvalidDotGeneralDimensionNumbers) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::MlirOp lhs_3d = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 3, 4}, op_builder.getF32Type()));
  const mlir::MlirOp rhs_3d = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 4, 5}, op_builder.getF32Type()));

  EXPECT_THAT(BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{0},
                                   /*rhs_batching=*/{}, /*lhs_contracting=*/{2},
                                   /*rhs_contracting=*/{1}, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("batching dimension counts must match")));
  EXPECT_THAT(BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{0},
                                   /*rhs_batching=*/{0},
                                   /*lhs_contracting=*/{2},
                                   /*rhs_contracting=*/{}, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("contracting dimension counts must match")));
  EXPECT_THAT(BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{-1},
                                   /*rhs_batching=*/{0},
                                   /*lhs_contracting=*/{2},
                                   /*rhs_contracting=*/{1}, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("lhs batch dimension -1 out of bounds")));
  EXPECT_THAT(BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{0, 0},
                                   /*rhs_batching=*/{0, 1},
                                   /*lhs_contracting=*/{2},
                                   /*rhs_contracting=*/{2}, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("duplicate lhs batch dimension 0")));
  EXPECT_THAT(
      BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{0},
                           /*rhs_batching=*/{0}, /*lhs_contracting=*/{0},
                           /*rhs_contracting=*/{1}, kDefaultPrecision),
      StatusIs(error::kInvalidArgument,
               HasSubstr("cannot be both batch/contracting or duplicated")));
  EXPECT_THAT(BuildTypedDotGeneral(lhs_3d, rhs_3d, /*lhs_batching=*/{0},
                                   /*rhs_batching=*/{0},
                                   /*lhs_contracting=*/{5},
                                   /*rhs_contracting=*/{1}, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("lhs contracting dimension 5 out of bounds")));
}

TEST_F(MatmulBuilderTest,
       RejectsMismatchedStaticAndBoundedDimensionsInForwardAndBackward) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::Type f32_type = op_builder.getF32Type();

  // 1. Static contracting mismatch rejected even when M=0 ([2, 0, 3] @ [2, 4,
  // 5]).
  const mlir::MlirOp lhs_zero_m = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 0, 3}, f32_type));
  const mlir::MlirOp rhs_bad_k = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 4, 5}, f32_type));
  EXPECT_THAT(BuildMatmulShlo(lhs_zero_m, rhs_bad_k, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("contracting dimensions do not match: 3 vs "
                                 "4")));

  // 2. Static batch mismatch ([2, 3, 4] @ [3, 4, 5]).
  const mlir::MlirOp lhs_b2 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, 3, 4}, f32_type));
  const mlir::MlirOp rhs_b3 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({3, 4, 5}, f32_type));
  EXPECT_THAT(BuildMatmulShlo(lhs_b2, rhs_b3, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("batch dimensions are not broadcastable: 2 vs "
                                 "3")));

  // 3. Static size S=8 exceeding bounded dynamic upper bound B=4 in
  //    BuildTypedDotGeneral and BuildMatmulShlo.
  const mlir::MlirOp lhs_static_8 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({8, 2, 3}, f32_type));
  const mlir::MlirOp rhs_dyn_4 = mlir::func::Argument(
      fn, MakeBoundedTensorType(op_builder.getContext(), {kDyn, 3, 5},
                                {4, kDyn, kDyn}, f32_type));
  EXPECT_THAT(
      BuildTypedDotGeneral(lhs_static_8, rhs_dyn_4, /*lhs_batching=*/{0},
                           /*rhs_batching=*/{0}, /*lhs_contracting=*/{2},
                           /*rhs_contracting=*/{1}, kDefaultPrecision),
      StatusIs(error::kInvalidArgument,
               HasSubstr("batch static dimension size 8 exceeds bounded "
                         "dynamic upper bound 4")));
  EXPECT_THAT(
      BuildTypedDotGeneral(rhs_dyn_4, lhs_static_8, /*lhs_batching=*/{0},
                           /*rhs_batching=*/{0}, /*lhs_contracting=*/{1},
                           /*rhs_contracting=*/{2}, kDefaultPrecision),
      StatusIs(error::kInvalidArgument,
               HasSubstr("batch static dimension size 8 exceeds bounded "
                         "dynamic upper bound 4")));
  EXPECT_THAT(BuildMatmulShlo(lhs_static_8, rhs_dyn_4, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("batch dimensions are not broadcastable: 8 vs "
                                 "4")));
  const mlir::MlirOp rhs_static_8 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({8, 5, 2}, f32_type));
  EXPECT_THAT(BuildMatmulShlo(rhs_dyn_4, rhs_static_8, kDefaultPrecision),
              StatusIs(error::kInvalidArgument,
                       HasSubstr("batch dimensions are not broadcastable: 4 vs "
                                 "8")));

  // 4. Backward incompatible batch dimensions (grad [4, 2, 5], self [3, 2, 4],
  //    other [4, 4, 5]).
  const mlir::MlirOp grad_b4 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4, 2, 5}, f32_type));
  const mlir::MlirOp other_b4 = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({4, 4, 5}, f32_type));
  EXPECT_THAT(
      BuildMatmulGradSelfShlo(grad_b4, /*self_static_dims=*/{3, 2, 4}, other_b4,
                              kDefaultPrecision),
      StatusIs(error::kInvalidArgument,
               HasSubstr("backward batch dimensions are not compatible")));
}

TEST_F(MatmulBuilderTest, RejectsDimensionProductOverflow) {
  mlir::func::FunctionBuilder fn(module_builder(), "main");
  mlir::OpBuilder& op_builder = fn.getOpBuilder();
  const mlir::Type f32_type = op_builder.getF32Type();
  const int64_t kHuge = std::numeric_limits<int64_t>::max() / 2 + 2;

  // 1. Overflow when folding leading dimensions in 3D x 2D.
  const mlir::MlirOp lhs_huge = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({kHuge, 3, 4}, f32_type));
  const mlir::MlirOp rhs_2d =
      mlir::func::Argument(fn, mlir::RankedTensorType::get({4, 5}, f32_type));
  EXPECT_THAT(
      BuildMatmulShlo(lhs_huge, rhs_2d, kDefaultPrecision),
      StatusIs(error::kInvalidArgument, HasSubstr("overflows as int64")));

  // 2. Overflow when coalescing adjacent kShared batch dimensions in 5D x 5D
  //    backward with broadcast reduction.
  const mlir::MlirOp grad_5d_huge = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, kHuge, 3, 2, 5}, f32_type));
  const mlir::MlirOp rhs_5d_huge = mlir::func::Argument(
      fn, mlir::RankedTensorType::get({2, kHuge, 3, 4, 5}, f32_type));
  EXPECT_THAT(
      BuildMatmulGradSelfShlo(grad_5d_huge,
                              /*self_static_dims=*/{1, kHuge, 3, 2, 4},
                              rhs_5d_huge, kDefaultPrecision),
      StatusIs(error::kInvalidArgument, HasSubstr("overflows as int64")));
}

}  // namespace
}  // namespace torch_tpu
