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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

namespace mlir::torch_tpu {

static_assert(kDefaultSerializationVersion <= tpu::MosaicSerdePass::kVersion);

Value CreateMatmul(ImplicitLocOpBuilder& builder, Value lhs, Value rhs,
                   const MatmulOptions& options) {
  auto dimension_numbers = tpu::defaultDimensionNumbers(
      builder, options.transpose_lhs, options.transpose_rhs);

  auto lhs_type = mlir::cast<VectorType>(lhs.getType());
  auto rhs_type = mlir::cast<VectorType>(rhs.getType());

  auto lhs_shape = lhs_type.getShape();
  auto rhs_shape = rhs_type.getShape();

  SmallVector<int64_t> output_shape = {
      lhs_shape[options.transpose_lhs ? 1 : 0],
      rhs_shape[options.transpose_rhs ? 0 : 1]};

  auto accumulator_type = VectorType::get(output_shape, builder.getF32Type());
  Value accumulator =
      arith::ConstantOp::create(builder, accumulator_type,
                                DenseElementsAttr::get(accumulator_type, 0.0f));

  // We may want to consider setting this based on input types.
  // We leave it to the default for now to keep numerical compatibility.
  tpu::ContractPrecisionAttr precision_attr = nullptr;

  // Hard-set transpose_<lhs/rhs> as they are ignored when dimension_numbers is
  // provided.
  return tpu::MatmulOp::create(builder, accumulator_type, lhs, rhs, accumulator,
                               /*transpose_lhs=*/false, /*transpose_rhs=*/false,
                               precision_attr, dimension_numbers, false);
}

Value CreateIotaOp(ImplicitLocOpBuilder& builder, Type type,
                   int32_t dimension) {
  return tpu::IotaOp::create(builder, type, {dimension});
}

Value CreateRepeatOp(ImplicitLocOpBuilder& builder, Value input,
                     int64_t dimension, int64_t target_dim_size) {
  VectorType input_type = mlir::cast<VectorType>(input.getType());
  int64_t input_dim_size = input_type.getDimSize(dimension);

  ABSL_CHECK_EQ(target_dim_size % input_dim_size, 0);  // CRASH_OK
  int64_t num_repeats = target_dim_size / input_dim_size;

  SmallVector<int64_t> output_shape(input_type.getShape());
  output_shape[dimension] = target_dim_size;
  VectorType output_type =
      VectorType::get(output_shape, input_type.getElementType());

  return tpu::RepeatOp::create(builder, output_type, input, dimension,
                               num_repeats);
}

Value CreateReciprocal(ImplicitLocOpBuilder& b, Value input) {
  return tpu::ReciprocalOp::create(b, input, /*approx=*/false,
                                   /*full_range=*/false);
}

mlir::MemRefType GetVmemMemRefType(MLIRContext* context,
                                   ArrayRef<int64_t> shape, Type element_type) {
  return MemRefType::get(
      shape, element_type, nullptr,
      tpu::MemorySpaceAttr::get(context, tpu::MemorySpace::kVmem));
}

mlir::Attribute GetDimensionSemanticsAttr(MLIRContext* context,
                                          DimensionSemantics semantics) {
  switch (semantics) {
    case DimensionSemantics::kParallel:
      return tpu::DimensionSemanticsAttr::get(
          context, tpu::DimensionSemantics::parallel);
    case DimensionSemantics::kArbitrary:
      return tpu::DimensionSemanticsAttr::get(
          context, tpu::DimensionSemantics::arbitrary);
  }
}

std::unique_ptr<MLIRContext> CreateMlirContextWithDialects() {
  DialectRegistry registry;
  registry
      .insert<func::FuncDialect, memref::MemRefDialect, vector::VectorDialect,
              scf::SCFDialect, tpu::TPUDialect, math::MathDialect>();
  auto context = std::make_unique<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  context->allowUnregisteredDialects();
  return context;
}

void SetTcCoreTypeAttr(func::FuncOp fn) {
  fn->setAttr("tpu.core_type",
              tpu::CoreTypeAttr::get(fn.getContext(), tpu::CoreType::kTc));
}

LogicalResult SerializeMosaicKernel(ModuleOp module,
                                    int serialization_version) {
  PassManager pm(module.getContext());
  if (failed(verify(module))) {
    return failure();
  }

  pm.addPass(tpu::createMosaicSerdePass(tpu::MosaicSerdePassOptions{
      .serialize = true, .target_version = serialization_version}));
  return pm.run(module);
}

}  // namespace mlir::torch_tpu
