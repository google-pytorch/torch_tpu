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

#include "csrc/internal/mosaic/kiln_custom_call.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_format.h"
#include "csrc/common/error_utils.h"
#include "csrc/internal/mosaic/kiln/kiln.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_tpu {
namespace {

// Row-major default layout attribute for a ranked tensor type.
mlir::DenseIntElementsAttr DefaultLayout(mlir::OpBuilder& builder,
                                         mlir::Type type) {
  const int64_t rank = mlir::cast<mlir::RankedTensorType>(type).getRank();
  auto layout_range = llvm::reverse(llvm::seq(rank));
  llvm::SmallVector<int64_t> layout(layout_range.begin(), layout_range.end());
  auto layout_type =
      mlir::RankedTensorType::get({rank}, builder.getIndexType());
  return mlir::DenseIntElementsAttr::get(layout_type, layout);
}

}  // namespace

absl::StatusOr<mlir::stablehlo::CustomCallOp> CreateKilnCustomCallOp(
    mlir::OpBuilder& builder, mlir::Location loc, kiln::Kernel kernel,
    mlir::ValueRange inputs, mlir::TypeRange output_types,
    std::optional<int64_t> vmem_limit_bytes, bool needs_hlo_passes) {
  TT_RETURN_IF_ERROR(kernel.Verify());
  TT_ASSIGN_OR_RETURN(const std::string body, std::move(kernel).Serialize());

  std::string scoped_memory_configs;
  if (vmem_limit_bytes.has_value()) {
    scoped_memory_configs = absl::StrFormat(
        R"(,
        "scoped_memory_configs": [
          {
            "memory_space": 1,
            "offset": 0,
            "size": %d
          }
        ])",
        *vmem_limit_bytes);
  }
  const std::string backend_config_json = absl::StrFormat(
      R"({
        "custom_call_config": {
          "body": "%s",
          "needs_layout_passes": true,
          "needs_hlo_passes": %v,
          "serialization_format": 1
        },
        "device_type": "DEVICE_TYPE_TENSORCORE"%s
      })",
      absl::Base64Escape(body), needs_hlo_passes, scoped_memory_configs);

  llvm::SmallVector<mlir::Attribute> input_layouts;
  input_layouts.reserve(inputs.size());
  for (const mlir::Value input : inputs) {
    input_layouts.push_back(DefaultLayout(builder, input.getType()));
  }
  llvm::SmallVector<mlir::Attribute> output_layouts;
  output_layouts.reserve(output_types.size());
  for (const mlir::Type output_type : output_types) {
    output_layouts.push_back(DefaultLayout(builder, output_type));
  }

  return mlir::stablehlo::CustomCallOp::create(
      builder, loc, output_types, inputs,
      builder.getStringAttr("tpu_custom_call"),
      /*has_side_effect=*/builder.getBoolAttr(false),
      builder.getStringAttr(backend_config_json),
      mlir::stablehlo::CustomCallApiVersionAttr::get(
          builder.getContext(),
          mlir::stablehlo::CustomCallApiVersion::API_VERSION_STATUS_RETURNING),
      /*called_computations=*/builder.getArrayAttr({}),
      builder.getArrayAttr(input_layouts), builder.getArrayAttr(output_layouts),
      /*output_operand_aliases=*/nullptr,
      /*result_tilings=*/nullptr);
}

}  // namespace torch_tpu
