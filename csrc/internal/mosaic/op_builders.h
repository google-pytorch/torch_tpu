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

#ifndef TORCH_TPU_CSRC_INTERNAL_MOSAIC_OP_BUILDERS_H_
#define TORCH_TPU_CSRC_INTERNAL_MOSAIC_OP_BUILDERS_H_

#include <cstdint>
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::torch_tpu {

constexpr int kDefaultSerializationVersion = 13;

enum class DimensionSemantics {
  kParallel,
  kArbitrary,
};

struct MatmulOptions {
  bool transpose_lhs = false;
  bool transpose_rhs = false;
};

// Creates a matmul for two 2D vectors.
Value CreateMatmul(ImplicitLocOpBuilder& builder, Value lhs, Value rhs,
                   const MatmulOptions& options = MatmulOptions());

// Helper to build a tpu::IotaOp.
Value CreateIotaOp(ImplicitLocOpBuilder& builder, Type type, int32_t dimension);

// Helper to build a tpu::RepeatOp.
Value CreateRepeatOp(ImplicitLocOpBuilder& builder, Value input,
                     int64_t dimension, int64_t target_dim_size);

Value CreateReciprocal(ImplicitLocOpBuilder& b, Value input);

// Helper to build a memref type with vmem tpu::Vmem memory space.
mlir::MemRefType GetVmemMemRefType(MLIRContext* context,
                                   ArrayRef<int64_t> shape, Type element_type);

// Helper to build a tpu:: dimension semantics attribute.
mlir::Attribute GetDimensionSemanticsAttr(MLIRContext* context,
                                          DimensionSemantics semantics);

// Helper to create an MLIR context with the dialects needed for TorchTPU
// kernels.
std::unique_ptr<MLIRContext> CreateMlirContextWithDialects();

// Add tpu.core_type = #tpu.core_type<kTc> attribute to the function.
void SetTcCoreTypeAttr(func::FuncOp fn);

// Serialize the given module to a string.
// This is required to have backwards / forwards compatibility with different
// versions of libtpu.
LogicalResult SerializeMosaicKernel(
    ModuleOp module, int serialization_version = kDefaultSerializationVersion);

}  // namespace mlir::torch_tpu

#endif  // TORCH_TPU_CSRC_INTERNAL_MOSAIC_OP_BUILDERS_H_
