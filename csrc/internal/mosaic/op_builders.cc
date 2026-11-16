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

#include "csrc/internal/mosaic/op_builders.h"

#include <cstdint>
#include <memory>

#include "absl/log/absl_check.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/IR/FuncDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRefDialect.h"
#include "mlir/Dialect/SCF/IR/SCFDialect.h"
#include "mlir/Dialect/Tensor/IR/TensorDialect.h"
#include "mlir/Dialect/Vector/IR/VectorDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

namespace torch_tpu {

static_assert(kDefaultSerializationVersion <=
              mlir::tpu::MosaicSerdePass::kVersion);

mlir::Value CreateMatmul(mlir::ImplicitLocOpBuilder& builder, mlir::Value lhs,
                         mlir::Value rhs, const MatmulOptions& options) {
  auto dimension_numbers = mlir::tpu::defaultDimensionNumbers(
      builder, options.transpose_lhs, options.transpose_rhs);

  auto lhs_type = mlir::cast<mlir::VectorType>(lhs.getType());
  auto rhs_type = mlir::cast<mlir::VectorType>(rhs.getType());

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  mlir::SmallVector<int64_t> output_shape = {
      lhs_shape[options.transpose_lhs ? 1 : 0],
      rhs_shape[options.transpose_rhs ? 0 : 1]};

  auto accumulator_type =
      mlir::VectorType::get(output_shape, builder.getF32Type());
  mlir::Value accumulator = mlir::arith::ConstantOp::create(
      builder, accumulator_type,
      mlir::DenseElementsAttr::get(accumulator_type, 0.0f));

  mlir::tpu::ContractPrecisionAttr precision_attr =
      options.precision.has_value()
          ? mlir::tpu::ContractPrecisionAttr::get(builder.getContext(),
                                                  *options.precision)
          : nullptr;

  // Hard-set transpose_<lhs/rhs> as they are ignored when dimension_numbers is
  // provided.
  return mlir::tpu::MatmulOp::create(
      builder, accumulator_type, lhs, rhs, accumulator,
      /*transpose_lhs=*/false, /*transpose_rhs=*/false, precision_attr,
      dimension_numbers, false);
}

mlir::Value CreateIotaOp(mlir::ImplicitLocOpBuilder& builder, mlir::Type type,
                         int32_t dimension) {
  return mlir::tpu::IotaOp::create(builder, type, {dimension});
}

mlir::Value CreateRepeatOp(mlir::ImplicitLocOpBuilder& builder,
                           mlir::Value input, int64_t dimension,
                           int64_t target_dim_size) {
  mlir::VectorType input_type = mlir::cast<mlir::VectorType>(input.getType());
  int64_t input_dim_size = input_type.getDimSize(dimension);

  ABSL_CHECK_EQ(target_dim_size % input_dim_size, 0);  // CRASH_OK
  int64_t num_repeats = target_dim_size / input_dim_size;
  if (num_repeats == 1) {
    return input;
  }

  mlir::SmallVector<int64_t> output_shape(input_type.getShape());
  output_shape[dimension] = target_dim_size;
  mlir::VectorType output_type =
      mlir::VectorType::get(output_shape, input_type.getElementType());
  mlir::SmallVector<mlir::Value> operands(num_repeats, input);
  return mlir::tpu::ConcatenateOp::create(builder, output_type, operands,
                                          dimension);
}

mlir::Value CreateReciprocal(mlir::ImplicitLocOpBuilder& b, mlir::Value input) {
  return mlir::tpu::ReciprocalOp::create(b, input, /*approx=*/false,
                                         /*full_range=*/false);
}

mlir::MemRefType GetVmemMemRefType(mlir::MLIRContext* context,
                                   mlir::ArrayRef<int64_t> shape,
                                   mlir::Type element_type) {
  return mlir::MemRefType::get(
      shape, element_type, nullptr,
      mlir::tpu::MemorySpaceAttr::get(context, mlir::tpu::MemorySpace::kVmem));
}

mlir::MemRefType GetSmemMemRefType(mlir::MLIRContext* context,
                                   mlir::ArrayRef<int64_t> shape,
                                   mlir::Type element_type) {
  return mlir::MemRefType::get(
      shape, element_type, nullptr,
      mlir::tpu::MemorySpaceAttr::get(context, mlir::tpu::MemorySpace::kSmem));
}

mlir::MemRefType GetHbmMemRefType(mlir::MLIRContext* context,
                                  mlir::ArrayRef<int64_t> shape,
                                  mlir::Type element_type) {
  return mlir::MemRefType::get(
      shape, element_type, nullptr,
      mlir::tpu::MemorySpaceAttr::get(context, mlir::tpu::MemorySpace::kHbm));
}

mlir::MemRefType GetSemaphoreMemRefType(mlir::MLIRContext* context,
                                        mlir::ArrayRef<int64_t> shape,
                                        mlir::Type element_type) {
  return mlir::MemRefType::get(
      shape, element_type, nullptr,
      mlir::tpu::MemorySpaceAttr::get(context,
                                      mlir::tpu::MemorySpace::kSemaphoreMem));
}

mlir::Attribute GetDimensionSemanticsAttr(mlir::MLIRContext* context,
                                          DimensionSemantics semantics) {
  switch (semantics) {
    case DimensionSemantics::kParallel:
      return mlir::tpu::DimensionSemanticsAttr::get(
          context, mlir::tpu::DimensionSemantics::parallel);
    case DimensionSemantics::kArbitrary:
      return mlir::tpu::DimensionSemanticsAttr::get(
          context, mlir::tpu::DimensionSemantics::arbitrary);
  }
}

std::unique_ptr<mlir::MLIRContext> CreateMlirContextWithDialects() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  mlir::scf::SCFDialect, mlir::tpu::TPUDialect,
                  mlir::math::MathDialect, mlir::tensor::TensorDialect,
                  mlir::chlo::ChloDialect, mlir::stablehlo::StablehloDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  context->allowUnregisteredDialects();
  return context;
}

void SetTcCoreTypeAttr(mlir::func::FuncOp fn) {
  fn->setAttr("tpu.core_type", mlir::tpu::CoreTypeAttr::get(
                                   fn.getContext(), mlir::tpu::CoreType::kTc));
}

mlir::LogicalResult SerializeMosaicKernel(mlir::ModuleOp module,
                                          int serialization_version) {
  mlir::PassManager pm(module.getContext());
  if (mlir::failed(mlir::verify(module))) {
    return mlir::failure();
  }

  pm.addPass(mlir::tpu::createMosaicSerdePass(mlir::tpu::MosaicSerdePassOptions{
      .serialize = true, .target_version = serialization_version}));
  return pm.run(module);
}

}  // namespace torch_tpu
