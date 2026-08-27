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

#ifndef TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_H_
#define TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace torch_tpu {

absl::StatusOr<mlir::MlirOp> BuildConvolution(
    mlir::MlirOp input, mlir::MlirOp weight, std::optional<mlir::MlirOp> bias,
    absl::Span<const int64_t> stride, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> dilation, bool transposed,
    absl::Span<const int64_t> output_padding, int64_t groups,
    absl::Span<const int64_t> output_dims, mlir::ElementType output_dtype,
    mlir::stablehlo::Precision precision);

absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardInput(
    mlir::MlirOp grad_output, mlir::MlirOp weight,
    absl::Span<const int64_t> input_dims, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    int64_t groups, bool transposed, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype, mlir::stablehlo::Precision precision);

absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardWeight(
    mlir::MlirOp input, mlir::MlirOp grad_output,
    absl::Span<const int64_t> weight_dims, absl::Span<const int64_t> stride,
    absl::Span<const int64_t> padding, absl::Span<const int64_t> dilation,
    int64_t groups, bool transposed, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype, mlir::stablehlo::Precision precision);

absl::StatusOr<mlir::MlirOp> BuildConvolutionBackwardBias(
    mlir::MlirOp grad_output, absl::Span<const int64_t> output_padding,
    mlir::ElementType output_dtype);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_CONVOLUTION_CONVOLUTION_H_
