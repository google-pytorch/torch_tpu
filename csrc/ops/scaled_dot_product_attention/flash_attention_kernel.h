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

#ifndef TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_
#define TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_

#include "absl/status/statusor.h"
#include "csrc/ops/scaled_dot_product_attention/flash_attention_config.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::torch_tpu {

// Creates the flash attention kernel for the given configuration.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateKernel(
    MLIRContext* context, const FlashAttnConfig& config, const Tiling& tiling);

// Creates the flash attention backward DKV kernel for the given configuration.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBackwardDkvKernel(
    MLIRContext* context, const FlashAttnConfig& config, const Tiling& tiling);

// Creates the flash attention backward DQ kernel for the given configuration.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateBackwardDqKernel(
    MLIRContext* context, const FlashAttnConfig& config, const Tiling& tiling);

}  // namespace mlir::torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_SCALED_DOT_PRODUCT_ATTENTION_FLASH_ATTENTION_KERNEL_H_
