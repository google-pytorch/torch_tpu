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

#ifndef TORCH_TPU_INTERNAL_MOSAIC_OP_BUILDERS_H_
#define TORCH_TPU_INTERNAL_MOSAIC_OP_BUILDERS_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::torch_tpu {

enum class ContractPrecision {
  kF32,
  kBF16,
};

enum class DimensionSemantics {
  kParallel,
  kArbitrary,
};

// Helper to build a vector::ContractionOp.
Value CreateContractionOp(
    ImplicitLocOpBuilder& builder, Value lhs, Value rhs, Value acc,
    ArrayRef<ArrayRef<AffineExpr>> indexingMaps,
    ArrayRef<vector::IteratorType> iteratorTypes,
    std::optional<ContractPrecision> precision = std::nullopt);

// Helper to build a tpu::IotaOp.
Value CreateIotaOp(ImplicitLocOpBuilder& builder, Type type, int32_t dimension);

// Helper to build a memref type with vmem tpu::Vmem memory space.
mlir::MemRefType GetVmemMemRefType(MLIRContext* context,
                                   ArrayRef<int64_t> shape, Type element_type);

// Helper to build a tpu:: dimension semantics attribute.
mlir::Attribute GetDimensionSemanticsAttr(MLIRContext* context,
                                          DimensionSemantics semantics);

// Helper to create an MLIR context with the dialects needed for TorchTPU
// kernels.
std::unique_ptr<MLIRContext> CreateMlirContextWithDialects();

// Helper to create a stablehlo::CustomCallOp with mosaic kernel.
stablehlo::CustomCallOp CreateCustomCallOp(OpBuilder& builder, Location loc,
                                           std::string_view kernel_mlir,
                                           ValueRange inputs,
                                           TypeRange output_types);

void SetTcCoreTypeAttr(func::FuncOp fn);

LogicalResult SerializeMoasicKernel(ModuleOp module);

}  // namespace mlir::torch_tpu

#endif  // TORCH_TPU_INTERNAL_MOSAIC_OP_BUILDERS_H_
