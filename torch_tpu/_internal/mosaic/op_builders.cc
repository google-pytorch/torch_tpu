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

#include "torch_tpu/_internal/mosaic/op_builders.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/dialect/tpu/transforms/serde.h"

namespace mlir::torch_tpu {

namespace {
tpu::ContractPrecisionAttr getContractPrecisionAttr(
    OpBuilder& builder, ContractPrecision precision) {
  switch (precision) {
    case ContractPrecision::kF32:
      return tpu::ContractPrecisionAttr::get(builder.getContext(),
                                             tpu::ContractPrecision::kFP32);
    case ContractPrecision::kBF16:
      return tpu::ContractPrecisionAttr::get(builder.getContext(),
                                             tpu::ContractPrecision::kBF16);
  }
}

}  // namespace

Value CreateContractionOp(OpBuilder& builder, Location loc, Value lhs,
                          Value rhs, Value acc,
                          ArrayRef<ArrayRef<AffineExpr>> indexingMaps,
                          ArrayRef<vector::IteratorType> iteratorTypes,
                          std::optional<ContractPrecision> precision) {
  auto contraction = vector::ContractionOp::create(builder, loc, lhs, rhs, acc,
                                                   indexingMaps, iteratorTypes);
  if (precision.has_value()) {
    contraction->setAttr("precision",
                         getContractPrecisionAttr(builder, *precision));
  }
  return contraction;
}

Value CreateContractionOp(ImplicitLocOpBuilder& builder, Value lhs, Value rhs,
                          Value acc,
                          ArrayRef<ArrayRef<AffineExpr>> indexingMaps,
                          ArrayRef<vector::IteratorType> iteratorTypes,
                          std::optional<ContractPrecision> precision) {
  return CreateContractionOp(builder, builder.getLoc(), lhs, rhs, acc,
                             indexingMaps, iteratorTypes, precision);
}

Value CreateIotaOp(ImplicitLocOpBuilder& builder, Type type,
                   int32_t dimension) {
  return tpu::IotaOp::create(builder, type, {dimension});
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

stablehlo::CustomCallOp CreateCustomCallOp(OpBuilder& builder, Location loc,
                                           std::string_view kernel_mlir,
                                           ValueRange inputs,
                                           TypeRange output_types) {
  std::string base64_kernel = absl::Base64Escape(kernel_mlir);
  std::string backend_config_json = absl::StrFormat(
      R"({
        "custom_call_config": {
          "body": "%s",
          "needs_layout_passes": true,
          "serialization_format": 1,
        },
        "device_type": "DEVICE_TYPE_TENSORCORE",
      })",
      base64_kernel);

  auto get_default_layout = [&builder](Type type) {
    int64_t rank = cast<RankedTensorType>(type).getRank();
    auto layout_range = llvm::reverse(llvm::seq(rank));
    SmallVector<int64_t> layout(layout_range.begin(), layout_range.end());
    auto layout_type = RankedTensorType::get({rank}, builder.getIndexType());
    return DenseIntElementsAttr::get(layout_type, layout);
  };

  SmallVector<Attribute> input_layouts;
  input_layouts.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_layouts.push_back(get_default_layout(input.getType()));
  }

  SmallVector<Attribute> output_layouts;
  output_layouts.reserve(output_types.size());
  for (const auto& output_type : output_types) {
    output_layouts.push_back(get_default_layout(output_type));
  }

  return stablehlo::CustomCallOp::create(
      builder, loc, output_types, inputs,
      builder.getStringAttr("tpu_custom_call"), builder.getBoolAttr(false),
      builder.getStringAttr(backend_config_json),
      stablehlo::CustomCallApiVersionAttr::get(
          builder.getContext(),
          stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
      builder.getArrayAttr({}), builder.getArrayAttr(input_layouts),
      builder.getArrayAttr(output_layouts),
      /*output_operand_aliases=*/nullptr,
      /*result_tilings=*/nullptr);
}

void SetTcCoreTypeAttr(func::FuncOp fn) {
  fn->setAttr("tpu.core_type",
              tpu::CoreTypeAttr::get(fn.getContext(), tpu::CoreType::kTc));
}

LogicalResult SerializeMoasicKernel(ModuleOp module) {
  PassManager pm(module.getContext());
  if (failed(verify(module))) {
    return failure();
  }
  pm.addPass(tpu::createMosaicSerdePass(tpu::MosaicSerdePassOptions{
      .serialize = true, .target_version = tpu::MosaicSerdePass::kVersion}));
  return pm.run(module);
}

}  // namespace mlir::torch_tpu
