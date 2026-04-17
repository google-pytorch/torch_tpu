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

#include "torch_tpu/ops/view_decomposition/pad_primitive.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

namespace {

absl::Status ValidatePad(const PadPrimitive& pad, const size_t expected_rank) {
  TT_RET_CHECK(pad.pad_dims.size() == expected_rank, error::kInvalidArgument)
      << "pad has wrong number of dimensions. Expected " << expected_rank
      << " but got " << pad.pad_dims.size();
  for (const auto& pad_dim : pad.pad_dims) {
    if (pad_dim.low_padding < 0 || pad_dim.high_padding < 0 ||
        pad_dim.interior_padding < 0) {
      return TT_ERROR(error::kInvalidArgument)
             << "pad has negative padding: " << pad;
    }
  }
  return absl::OkStatus();
}
}  // namespace

bool operator==(const PadDimension& lhs, const PadDimension& rhs) {
  return lhs.low_padding == rhs.low_padding &&
         lhs.high_padding == rhs.high_padding &&
         lhs.interior_padding == rhs.interior_padding;
}

bool operator==(const PadPrimitive& lhs, const PadPrimitive& rhs) {
  return lhs.pad_dims == rhs.pad_dims;
}

std::ostream& operator<<(std::ostream& os, const PadDimension& dim) {
  os << "(low=" << dim.low_padding << ", high=" << dim.high_padding
     << ", interior=" << dim.interior_padding << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const PadPrimitive& pad) {
  os << "pad" << ToString(pad.pad_dims) << "";
  return os;
}

absl::StatusOr<bool> UpdateLayoutImpl(StridedLayout& layout,
                                      const PadPrimitive& pad) {
  TT_RETURN_IF_ERROR(ValidatePad(pad, layout.strided_dims.size()));
  bool updated = false;
  // Padding resets the storage offset to 0, as it requires a copy to
  // contiguous.
  StridedLayout new_layout{.storage_offset = 0};
  new_layout.strided_dims.reserve(pad.pad_dims.size());
  for (auto i = 0; i < pad.pad_dims.size(); ++i) {
    const auto& pad_dim = pad.pad_dims[i];
    const auto& input_dim = layout.strided_dims[i];

    if (pad_dim.low_padding != 0 || pad_dim.high_padding != 0 ||
        pad_dim.interior_padding != 0) {
      updated = true;
    }
    const int64_t new_size = input_dim.size + pad_dim.low_padding +
                             pad_dim.high_padding +
                             (input_dim.size - 1) * pad_dim.interior_padding;
    new_layout.strided_dims.push_back(
        StridedDimension{.size = new_size, .stride = 1});
  }
  if (!updated) {
    return false;
  }
  // Padding resets strides to contiguous.
  if (new_layout.strided_dims.size() > 1) {
    for (int i = new_layout.strided_dims.size() - 2; i >= 0; --i) {
      new_layout.strided_dims[i].stride =
          new_layout.strided_dims[i + 1].stride *
          new_layout.strided_dims[i + 1].size;
    }
  }
  layout = std::move(new_layout);
  return true;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(mlir::MlirOp input,
                                               const PadPrimitive& pad) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_RETURN_IF_ERROR(ValidatePad(pad, input_type.getRank()));

  // Restructure from list-of-tuples to tuple-of-lists to match StableHLO API.
  Indices low_padding;
  low_padding.reserve(pad.pad_dims.size());
  Indices interior_padding;
  interior_padding.reserve(pad.pad_dims.size());
  Indices high_padding;
  high_padding.reserve(pad.pad_dims.size());
  for (const auto& pad_dim : pad.pad_dims) {
    low_padding.push_back(pad_dim.low_padding);
    interior_padding.push_back(pad_dim.interior_padding);
    high_padding.push_back(pad_dim.high_padding);
  }

  mlir::Type input_element_type = input_type.getElementType();
  mlir::MlirOp zero_pad_value;
  if (input_element_type.isInteger()) {  // also includes PRED/boolean
    zero_pad_value =
        MakeScalarConstant(input.getBuilder(), 0, input_element_type);
  } else {
    zero_pad_value =
        MakeScalarConstant(input.getBuilder(), 0.0, input_element_type);
  }
  return mlir::stablehlo::Pad(input, zero_pad_value, low_padding, high_padding,
                              interior_padding);
}

}  // namespace torch_tpu
