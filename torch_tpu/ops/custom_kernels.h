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

#ifndef TORCH_TPU_OPS_CUSTOM_KERNELS_H_
#define TORCH_TPU_OPS_CUSTOM_KERNELS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

// Registers a custom kernel. This can be any kernel that has been
// serialized into an MLIR module string. The most common patterns will be
// either importing from a pre-exported kernel library, or importing from a
// user-generated kernel authoring framework (e.g. Pallas) in the same
// process as torch_tpu.
//
// Args:
//  - name: The name of the custom kernel (user-provided).
//  - kwargs: a string representation of the keyword arguments to the custom
//    kernel. This is used to differentiate between otherwise-identical
//    custom kernels; for example, if the same kernel is imported with
//    different tiling behavior on the same input/output shapes.
//  - mlir_module_string: The serialized representation of the MLIR
//    for an mlir::ModuleOp. This must be a single MLIR module, with a @main
//    function, which may have dynamic shapes (e.g. tensor<?xf32>).
//
// Returns true if the kernel was newly inserted, or false if it already
// existed.
bool RegisterCustomKernel(std::string_view name, std::string_view kwargs,
                          std::string_view mlir_module_string);

// Looks up a registered custom kernel with the given name and kwargs.
// Returns true if a kernel was found, false otherwise.
// This can be used to avoid re-tracing kernels if the argument shapes and
// keyword arguments are the same.
bool LookupCustomKernel(std::string_view name, std::string_view kwargs);

// Calls a registered custom kernel with the given inputs.
// If this is the first call to this kernel in this builder, it will be loaded
// and specialized. Otherwise, it will reuse the previously-loaded function.
// Returns a NotFound error if a kernel has not been registered with this name
// and kwargs.
absl::StatusOr<DynamicMlirOpResults> CallCustomKernel(
    mlir::MlirBuilder& builder, absl::Span<const mlir::MlirOp> inputs,
    std::string_view name, std::string_view kwargs);

// Build a specialized MLIR kernel from a custom dynamically shaped MLIR
// kernel. These can be exported from any framework that produces dynamically
// shaped StableHLO, and calls are refined to the concrete argument
// shapes at runtime.
//
// Given an MLIR kernel with dynamic shapes (tensor<?xf32>) and concrete input
// shapes, produces a fully statically shaped kernel. The custom kernel should
// live in its own module, this builder will inline the kernel into the module
// being built.
//
// For full details on dynamism and refinement, see
// https://openxla.org/stablehlo/dynamism.
absl::StatusOr<DynamicMlirOpResults> BuildSpecializedMlirKernel(
    mlir::MlirBuilder& builder, mlir::ModuleOp custom_kernel,
    absl::Span<mlir::MlirOp> inputs);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_CUSTOM_KERNELS_H_
