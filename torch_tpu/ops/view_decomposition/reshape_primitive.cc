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

#include "torch_tpu/ops/view_decomposition/reshape_primitive.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/view_decomposition/strided_layout.h"
#include "torch_tpu/ops/view_decomposition/view_primitive_error_utils.h"

namespace torch_tpu {

namespace {

// A maximally-contiguous block of dimensions in a strided layout.
// Any two sequential dimensions in a strided layout which have the property
// that stride[i] == size[i+1] * stride[i+1] can be viewed as a single dimension
// with size = size[i] * size[i+1] and stride = stride[i+1].
// We refer to this as a "contiguity block"; a fully-contiguous tensor will
// have a single block with minor_stride = 1.
struct ContiguityBlock {
  int64_t num_elements = 0;
  int64_t minor_stride = 1;
};

// Returns the sequence of contiguity blocks in the given strided layout.
// This never fails because the worst case is one contiguity block per
// dimension, which is still valid.
std::vector<ContiguityBlock> GetContiguityBlocks(const StridedLayout& layout) {
  if (layout.strided_dims.empty()) {
    return {};
  }

  std::vector<ContiguityBlock> blocks;

  // Initialize the first block to be the first dimension with size > 1 to
  // skip arbitrary size-1 prefix dimensions.
  int64_t current_num_elements = layout.strided_dims[0].size;
  int64_t current_minor_stride = layout.strided_dims[0].stride;
  int64_t start_index = 1;
  while (current_num_elements == 1 &&
         start_index < layout.strided_dims.size()) {
    current_num_elements *= layout.strided_dims[start_index].size;
    current_minor_stride = layout.strided_dims[start_index].stride;
    start_index++;
  }

  for (int64_t i = start_index; i < layout.strided_dims.size(); ++i) {
    const auto& dim = layout.strided_dims[i];
    if (dim.size == 1) {
      // Size-1 dimensions have arbitrary strides, always merge with previous
      // (which doesn't increase the size or stride).
      continue;
    }
    if (current_minor_stride == dim.size * dim.stride) {
      // Contiguity criteria is met; extend the current block.
      current_num_elements *= dim.size;
      current_minor_stride = dim.stride;
      continue;
    }

    // End the previous block and start a new one.
    blocks.push_back(ContiguityBlock{.num_elements = current_num_elements,
                                     .minor_stride = current_minor_stride});
    current_num_elements = dim.size;
    current_minor_stride = dim.stride;
  }
  blocks.push_back(ContiguityBlock{.num_elements = current_num_elements,
                                   .minor_stride = current_minor_stride});
  return blocks;
}

bool IsNoOpReshape(const ReshapePrimitive& reshape,
                   const StridedLayout& layout) {
  if (reshape.new_sizes.size() != layout.strided_dims.size()) {
    return false;
  }
  for (int i = 0; i < reshape.new_sizes.size(); ++i) {
    if (reshape.new_sizes[i] != layout.strided_dims[i].size) {
      return false;
    }
  }
  return true;
}

void CheckNumberOfElements(const ReshapePrimitive& reshape,
                           const StridedLayout& layout) {
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout.strided_dims.size(), reshape.base_sizes.size())
      << "expected ReshapePrimitive base shape size to be "
      << layout.strided_dims.size() << ", which is the input rank, got "
      << reshape.base_sizes.size() << GetUpdateLayoutBugSuffix(reshape, layout);

  for (int i = 0; i < layout.strided_dims.size(); ++i) {
    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        layout.strided_dims[i].size, reshape.base_sizes[i])
        << "expected ReshapePrimitive base size at index " << i << " to be "
        << layout.strided_dims[i].size
        << ", which is the input size at that dimension, got "
        << reshape.base_sizes[i] << GetUpdateLayoutBugSuffix(reshape, layout);
  }
  int64_t layout_num_elements = 1;
  for (const auto& dim : layout.strided_dims) {
    layout_num_elements *= dim.size;
  }
  int64_t reshape_num_elements = 1;
  for (const auto& size : reshape.new_sizes) {
    reshape_num_elements *= size;
  }
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      layout_num_elements, reshape_num_elements)
      << "expected ReshapePrimitive total number of elements to be "
      << layout_num_elements
      << ", which is the input total number of elements, got "
      << reshape_num_elements << GetUpdateLayoutBugSuffix(reshape, layout);
  ABSL_CHECK_GT(  // CRASH_OK=Internal error on view decomposition.
      reshape_num_elements, 0)
      << "expected ReshapePrimitive total number of elements to be > 0, got "
      << reshape_num_elements << GetUpdateLayoutBugSuffix(reshape, layout);
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const ReshapePrimitive& reshape) {
  os << "reshape(base_sizes=" << absl::StrJoin(reshape.base_sizes, ",")
     << ", new_sizes=" << absl::StrJoin(reshape.new_sizes, ",") << ")";
  return os;
}

bool UpdateLayout(StridedLayout& layout, const ReshapePrimitive& reshape) {
  // Check for a no-op reshape
  if (IsNoOpReshape(reshape, layout)) {
    return false;
  }

  // Validate element count matches and is non-zero.
  CheckNumberOfElements(reshape, layout);

  // Not a no-op, actually need to do the reshape
  // Storage offset is unchanged.
  StridedLayout new_layout{.storage_offset = layout.storage_offset};

  // New sizes are given by the reshape reshape.
  new_layout.strided_dims.reserve(reshape.new_sizes.size());
  for (int i = 0; i < reshape.new_sizes.size(); ++i) {
    new_layout.strided_dims.push_back(
        StridedDimension{.size = reshape.new_sizes[i], .stride = 0});
  }

  // Solving for the new strides requires tracking the contiguity-like blocks.
  auto blocks = GetContiguityBlocks(layout);
  size_t block_index = 0;
  for (int64_t i = 0; i < new_layout.strided_dims.size(); ++i) {
    auto& new_dim = new_layout.strided_dims[i];

    if (new_dim.size == 1) {
      // Size-1 dimensions have arbitrary strides, and we can have as many
      // trailing size-1 dimensions as necessary.
      while (block_index < blocks.size() &&
             blocks[block_index].num_elements == 1) {
        ++block_index;
      }
      new_dim.stride = 1;
      continue;
    }

    ABSL_CHECK_LT(  // CRASH_OK=Internal error on view decomposition.
        block_index, blocks.size())
        << "expected the ReshapePrimitive the output of sizes "
        << ToString(reshape.new_sizes)
        << " to have only size 1 dimensions after reaching the end of the "
           "input contiguity blocks, got output size "
        << new_dim.size << " at index " << i
        << GetUpdateLayoutBugSuffix(reshape, layout);

    auto& current_block = blocks[block_index];
    if (new_dim.size == current_block.num_elements) {
      // Used up the elements in the current block.
      new_dim.stride = current_block.minor_stride;
      ++block_index;
      continue;
    }

    ABSL_CHECK_GT(  // CRASH_OK=Internal error on view decomposition.
        current_block.num_elements, new_dim.size)
        << "expected ReshapePrimitive output dimension size at index " << i
        << " to be < " << current_block.num_elements
        << ", which is the number of elements in the current contiguity block, "
           "got "
        << new_dim.size << GetUpdateLayoutBugSuffix(reshape, layout);
    ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
        current_block.num_elements % new_dim.size, 0)
        << "expected ReshapePrimitive the output dimension size "
        << new_dim.size << " to divide " << current_block.num_elements
        << ", which is the contiguity block number of elements, got "
        << new_dim.size << " does not divide " << current_block.num_elements
        << GetUpdateLayoutBugSuffix(reshape, layout);

    // Split the current block into two blocks to satisfy new_dim.size.
    // This converts a block with A*B elements and stride S into two blocks:
    //   1. A elements, stride B*S
    //   2. B elements, stride S
    current_block.num_elements /= new_dim.size;
    new_dim.stride = current_block.num_elements * current_block.minor_stride;
  }
  layout = std::move(new_layout);
  return true;
}

ReshapePrimitive Merge(ReshapePrimitive first, ReshapePrimitive second) {
  // The Reshape reshape themselves don't have stride metadata so we can't
  // validate the contiguity-like property here, but we can check that the
  // element counts match.
  int64_t current_num_elements = 1;
  for (int64_t size : first.new_sizes) {
    current_num_elements *= size;
  }
  int64_t to_merge_num_elements = 1;
  for (int64_t size : second.new_sizes) {
    to_merge_num_elements *= size;
  }
  ABSL_CHECK_EQ(  // CRASH_OK=Internal error on view decomposition.
      current_num_elements, to_merge_num_elements)
      << "expected sequential ReshapePrimitives element count to match, got "
      << current_num_elements << " != " << to_merge_num_elements
      << GetViewPrimitiveErrorSuffix(first);

  // Sequential reshapes from A -> B and B -> C merge to A -> C.
  second.base_sizes = std::move(first.base_sizes);
  return second;
}

absl::StatusOr<mlir::MlirOp> ViewPrimitiveShlo(
    mlir::MlirOp input, const ReshapePrimitive& reshape) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  TT_ASSIGN_OR_RETURN(auto result,
                      ReshapeFromStaticDimensions(input, reshape.base_sizes,
                                                  reshape.new_sizes));
  if (input_type.hasStaticShape()) {
    const mlir::RankedTensorType output_type = GetTensorTypeOrDie(result);
    // Verify that static shaped reshapes are exactly what PT dictates them to
    // be.
    ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
        input_type.getShape().equals(reshape.base_sizes))
        << "expected the ReshapePrimitive input shape to be equal "
        << ToString(reshape.base_sizes)
        << ", which is the expected input shape by the primitive, got "
        << ToString(input_type.getShape())
        << GetViewPrimitiveShloErrorSuffix(reshape, input_type.getShape());

    ABSL_CHECK(  // CRASH_OK=Internal error on view decomposition.
        output_type.getShape().equals(reshape.new_sizes))
        << "expected the ReshapePrimitive output shape to be equal "
        << ToString(reshape.new_sizes)
        << ", which is the expected output shape by the primitive, got "
        << ToString(output_type.getShape())
        << GetViewPrimitiveShloErrorSuffix(reshape, output_type.getShape());
  }
  return result;
}

}  // namespace torch_tpu
