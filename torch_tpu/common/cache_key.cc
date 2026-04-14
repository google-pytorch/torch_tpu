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

#include "torch_tpu/common/cache_key.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/core/Device.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace torch_tpu {

namespace internal {

absl::StatusOr<std::string> FormatParamCacheKey(const at::Scalar value) {
  std::string key;
  if (value.isFloatingPoint()) {
    key = LosslessToString(value.toDouble());
  } else if (value.isIntegral(/*include_bool=*/false)) {
    key = absl::StrCat(value.toLong());
  } else if (value.isBoolean()) {
    key = value.toBool() ? "1" : "0";
  } else if (value.isComplex()) {
    key = LosslessToString(value.toComplexDouble());
  } else {
    return TT_ERROR(error::kInvalidArgument)
           << "Unable to create key for scalar type.";
  }
  return absl::StrCat(key, ":", c10::toString(value.type()));
}

std::string FormatParamCacheKey(const c10::SymInt& value) {
  return ToString(value);
}

std::string FormatParamCacheKey(c10::SymIntArrayRef value) {
  return FormatParamCacheKey(absl::MakeConstSpan(value));
}

std::string FormatParamCacheKey(const c10d::ReduceOp value) {
  const c10d::ReduceOp::RedOpType reduce_op_type = value;
  return ToString(reduce_op_type);
}

std::string FormatParamCacheKey(const std::string_view value) {
  return std::string(value);
}

std::string FormatParamCacheKey(const absl::Span<const int64_t> value) {
  return absl::StrCat("[", absl::StrJoin(value, ","), "]");
}

std::string FormatParamCacheKey(const at::Layout value) {
  return ToString(value);
}

std::string FormatParamCacheKey(const at::MemoryFormat value) {
  return ToString(value);
}

std::string FormatParamCacheKey(const at::Device value) {
  // The device string may contain `:`, which may cause ambiguity when
  // parsing the cache key, so we quote it.
  return FormatParamCacheKey(value.str());
}

std::string_view ParseNextArgName(std::string_view& args_str) {
  // Find the first ',' that is not inside parentheses, or the end of the
  // string.
  size_t name_end = std::string_view::npos;
  int depth = 0;
  for (size_t i = 0; i < args_str.size(); ++i) {
    if (args_str[i] == '(') {
      ++depth;
    } else if (args_str[i] == ')') {
      --depth;
    } else if (args_str[i] == ',' && depth == 0) {
      name_end = i;
      break;
    }
  }

  if (name_end == std::string_view::npos) {
    name_end = args_str.size();
  }

  // Extract the first argument name, removing whitespace from both sides.
  std::string_view name = args_str.substr(0, name_end);
  name = absl::StripAsciiWhitespace(name);

  // Remove the name and the following comma (if any) from args_str.
  args_str.remove_prefix(name_end == args_str.size() ? name_end : name_end + 1);
  return name;
}

}  // namespace internal

PromotedScalar::PromotedScalar(Promoter promoter, at::Scalar scalar)
    : state_(std::make_unique<State>(State{.promoter = std::move(promoter),
                                           .scalar = std::move(scalar)})) {}

absl::StatusOr<at::Tensor> PromotedScalar::GetTensor(
    std::optional<at::ScalarType> scalar_type_opt) {
  state_->tensor_used = true;
  return state_->promoter(state_->scalar, scalar_type_opt);
}

std::string PromotedScalar::ToString() const {
  return torch_tpu::ToString(scalar());
}

absl::StatusOr<OpParamCacheKeys> OpParamCacheKeys::Builder::operator*() {
  if (first_error_.ok()) {
    return std::move(param_keys_);
  }
  // Clear the builder state before returning the error.
  auto result = first_error_;
  first_error_ = absl::OkStatus();
  param_keys_.name_to_value_.clear();
  return std::move(result);
}

std::string CompilationCacheKey::CompactFormat() const {
  return absl::StrFormat("%016x_%016x", shapeless_key.key, dimensions_key.key);
}

std::ostream& operator<<(std::ostream& os, CompilationCacheKey key) {
  os << "CompilationCacheKey{shapeless_key=" << std::hex
     << key.shapeless_key.key << ", dimensions_key=" << key.dimensions_key.key
     << std::dec << "}";
  return os;
}

DimensionsKey::DimensionsKey(
    const ShapeDynamismMetadata& shape_dynamism_metadata) {
  // These need to work with FingerprintCat
  std::vector<int64_t> upper_bounds;  // INT_VEC_OK
  std::vector<int64_t> lower_bounds;  // INT_VEC_OK
  upper_bounds.reserve(shape_dynamism_metadata.input_dimension_bounds().size());
  lower_bounds.reserve(shape_dynamism_metadata.input_dimension_bounds().size());

  for (const auto& dimension_bounds :
       shape_dynamism_metadata.input_dimension_bounds()) {
    upper_bounds.push_back(dimension_bounds.upper);
    lower_bounds.push_back(dimension_bounds.lower);
  }
  key = FingerprintCat(upper_bounds, lower_bounds);
}

ShapeDynamismMetadata::ShapeDynamismMetadata(absl::Span<const Shape> shapes) {
  for (const Shape& shape : shapes) {
    const int64_t tensor_start_dim = input_dimension_bounds_.size();
    for (int64_t dim : shape.dimensions()) {
      input_dimension_bounds_.push_back({dim, dim});
    }
    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      const int64_t dynamic_dim_index =
          tensor_start_dim + dynamic_dim.dimension;
      input_dimension_bounds_[dynamic_dim_index] = {dynamic_dim.lower_bound,
                                                    dynamic_dim.upper_bound};
    }
  }
}

namespace {
bool IsShapeCompatibleWithBounds(const Shape& shape,
                                 absl::Span<const DimensionBounds> bounds) {
  TT_CHECK_THROW(shape.dimensions().size() == bounds.size(), error::kInternal)
      << "Shape and bounds spans must have the same size.";
  for (int i = 0; i < shape.dimensions().size(); ++i) {
    const int64_t dim = shape.dimensions()[i];
    if (dim < bounds[i].lower || dim > bounds[i].upper) {
      return false;
    }
  }
  return true;
}

std::string GetIncompatibilityErrorMsg(
    int64_t index, const Shape& shape,
    absl::Span<const DimensionBounds> bounds) {
  return absl::StrCat(
      "Input shape is incompatible with dynamism metadata at index ", index,
      ". ", "input shapes: ", ToString(shape.dimensions()),
      " dynamism bounds: ",
      absl::StrJoin(
          bounds, ",", [](std::string* out, const DimensionBounds& bounds) {
            absl::StrAppend(out, "[", bounds.lower, ",", bounds.upper, "]");
          }));
}

Shape GetPaddingShape(const Shape& shape,
                      absl::Span<const DimensionBounds> bounds) {
  TT_CHECK_THROW(shape.dimensions().size() == bounds.size(), error::kInternal)
      << "Shape and bounds spans must have the same size.";
  Shape padding_shape(shape.dimensions(), shape.dtype());
  for (int i = 0; i < shape.dimensions().size(); ++i) {
    TT_CHECK_THROW(padding_shape.dimensions()[i] >= bounds[i].lower &&
                       padding_shape.dimensions()[i] <= bounds[i].upper,
                   error::kInternal)
        << GetIncompatibilityErrorMsg(i, shape, bounds);
    if (bounds[i].lower != bounds[i].upper) {
      padding_shape.dynamic_dimensions().push_back(
          {.dimension = i,
           .lower_bound = bounds[i].lower,
           .upper_bound = bounds[i].upper});
    }
  }
  return padding_shape;
}
}  // namespace

OpParamCacheKeys::OpParamCacheKeys(OpParamCacheKeys&& other)
    : valid_(other.valid_), name_to_value_(std::move(other.name_to_value_)) {
  ABSL_CHECK(other.valid_)  // CRASH_OK
      << "Moving from an invalid OpParamCacheKeys object. This is a TorchTPU "
         "bug.";
  other.valid_ = false;
}

OpParamCacheKeys& OpParamCacheKeys::operator=(OpParamCacheKeys&& other) {
  ABSL_CHECK(other.valid_)  // CRASH_OK
      << "Assigning from an invalid OpParamCacheKeys object. This is a "
         "TorchTPU bug.";
  if (this == &other) {
    return *this;
  }
  valid_ = true;
  name_to_value_ = std::move(other.name_to_value_);
  other.valid_ = false;
  return *this;
}

const OpParamCacheKeys::Map& OpParamCacheKeys::GetMapOrDie() const {
  ABSL_CHECK(valid_)  // CRASH_OK
      << "Accessing a moved-from OpParamCacheKeys object. This is a "
         "TorchTPU bug.";
  return name_to_value_;
}

bool ShapeDynamismMetadata::IsStaticShapeCompatible(
    absl::Span<const Shape> shapes) const {
  int64_t flattened_size = 0;
  for (const Shape& shape : shapes) {
    flattened_size += shape.dimensions().size();
  }
  if (flattened_size != input_dimension_bounds_.size()) {
    return false;
  }
  int64_t index = 0;
  absl::Span<const DimensionBounds> bounds = input_dimension_bounds_;
  for (const Shape& shape : shapes) {
    int64_t span_size = shape.dimensions().size();
    if (!IsShapeCompatibleWithBounds(shape, bounds.subspan(index, span_size))) {
      return false;
    }
    index += span_size;
  }
  return true;
}

std::vector<Shape> ShapeDynamismMetadata::GetPaddingShapes(
    absl::Span<const Shape> shapes) const {
  TT_CHECK_THROW(IsStaticShapeCompatible(shapes), error::kInternal)
      << "Input shapes are incompatible with dynamism metadata.";
  std::vector<Shape> padding_shapes;
  padding_shapes.reserve(shapes.size());
  int index = 0;
  absl::Span<const DimensionBounds> bounds = input_dimension_bounds_;
  for (const Shape& shape : shapes) {
    int64_t span_size = shape.dimensions().size();
    padding_shapes.push_back(
        GetPaddingShape(shape, bounds.subspan(index, span_size)));
    index += span_size;
  }
  return padding_shapes;
}

CompilationCacheKey ShapeDynamismMetadata::GetPadModuleCacheKey(
    absl::Span<const Shape> shapes) const {
  GraphSignature graph;

  for (const Shape& shape : shapes) {
    graph.AddInput(shape.dimensions(), shape.dtype());
  }

  absl::Span<const DimensionBounds> bounds = input_dimension_bounds_;
  int bounds_index = 0;
  for (int i = 0; i < shapes.size(); ++i) {
    const Shape& shape = shapes[i];
    const int64_t span_size = shape.dimensions().size();
    auto shape_bounds = bounds.subspan(bounds_index, span_size);
    bounds_index += span_size;

    bool has_dynamic_dimensions = std::any_of(
        shape_bounds.begin(), shape_bounds.end(),
        [](const DimensionBounds& b) { return b.lower != b.upper; });

    if (!has_dynamic_dimensions) {
      graph.AddGraphOutput(i);
      continue;
    }

    Dimensions padded_dimensions = shape.dimensions();
    for (int d = 0; d < span_size; ++d) {
      if (shape_bounds[d].lower != shape_bounds[d].upper) {
        padded_dimensions[d] = shape_bounds[d].upper;
      }
    }

    const int padded_tensor_index = graph.AddOp(
        OpName::kPadUninitialized_, OpParamCacheKeys::Empty(),
        /*donated_inputs=*/{}, [&](GraphSignature::OpSignatureBuilder& op) {
          op.AddInput(i);
          op.AddOutput(padded_dimensions, shape.dtype());
        });
    graph.AddGraphOutput(padded_tensor_index);

    for (int d = 0; d < span_size; ++d) {
      if (shape_bounds[d].lower != shape_bounds[d].upper) {
        TT_ASSIGN_OR_THROW(
            OpParamCacheKeys params,
            *OpParamCacheKeysBuilder().SetParam("dimension_index", d));

        const int dim_size_index = graph.AddOp(
            OpName::kGetDimensionSize, params, /*donated_inputs=*/{},
            [i](GraphSignature::OpSignatureBuilder& op) {
              op.AddInput(i);
              op.AddOutput({1}, mlir::ElementType::I32);
            });
        graph.AddGraphOutput(dim_size_index);
      }
    }
  }

  return graph.cache_key();
}

int GraphSignature::AddTensor(absl::Span<const int64_t> dimensions,
                              mlir::ElementType dtype) {
  int tensor_index = next_tensor_index_++;
  for (int64_t dim : dimensions) {
    tensor_dimensions_.push_back(dim);
  }
  tensor_dimensions_starts_.push_back(tensor_dimensions_.size());
  tensor_element_types_.push_back(dtype);
  return tensor_index;
}

int GraphSignature::AddInput(absl::Span<const int64_t> dimensions,
                             mlir::ElementType dtype) {
  ABSL_CHECK(  // CRASH_OK=we enforce adding all inputs before any ops.
      !has_ops_)
      << "Cannot add inputs after ops have been added.";
  int tensor_index = AddTensor(dimensions, dtype);
  num_inputs_++;
  // Update the end index for the "implicit" ops block covering just inputs.
  // This is index 0 of `op_outputs_indices_`.
  op_outputs_indices_[0] = next_tensor_index_;
  return tensor_index;
}

void GraphSignature::OpSignatureBuilder::AddInput(int topological_index) {
  graph_->op_inputs_indices_.push_back(topological_index);
}

int GraphSignature::OpSignatureBuilder::AddOutput(
    absl::Span<const int64_t> dimensions, mlir::ElementType dtype) {
  return graph_->AddTensor(dimensions, dtype);
}

int GraphSignature::AddOp(
    OpName op_name, const OpParamCacheKeys& op_param_cache_keys,
    absl::Span<const int64_t> donated_inputs,
    absl::FunctionRef<void(OpSignatureBuilder&)> builder) {
  if (!has_ops_) {
    has_ops_ = true;
  }

  op_names_.push_back(op_name);

  for (const auto& [key, value] : op_param_cache_keys) {
    op_param_cache_keys_.push_back({key, value});
  }
  op_param_cache_keys_starts_.push_back(op_param_cache_keys_.size());

  const int first_output_index = next_tensor_index_;
  const int first_input_vec_index = op_inputs_indices_.size();

  OpSignatureBuilder op_builder(this);
  builder(op_builder);

  op_inputs_starts_.push_back(op_inputs_indices_.size());

  for (int64_t donated_input : donated_inputs) {
    int op_input_index =
        op_inputs_indices_[first_input_vec_index + donated_input];
    if (donated_indices_set_.insert(op_input_index).second) {
      donated_indices_.push_back(op_input_index);
    }
  }

  op_outputs_indices_.push_back(next_tensor_index_);

  return first_output_index;
}

void GraphSignature::AddGraphOutput(int index) {
  graph_output_indices_.push_back(index);
}

CompilationCacheKey GraphSignature::cache_key() const {
  auto sorted_donated_indices = donated_indices_;
  std::sort(sorted_donated_indices.begin(), sorted_donated_indices.end());

  const ShapelessKey shapeless_key = {FingerprintCat(
      graph_output_indices_, tensor_dimensions_starts_, tensor_element_types_,
      sorted_donated_indices, op_inputs_starts_, op_inputs_indices_, op_names_,
      op_param_cache_keys_starts_, op_param_cache_keys_, op_outputs_indices_)};
  const DimensionsKey dimensions_key(tensor_dimensions_);
  return {
      .shapeless_key = shapeless_key,
      .dimensions_key = dimensions_key,
  };
}

}  // namespace torch_tpu
