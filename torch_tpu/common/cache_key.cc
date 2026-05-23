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

#include "ATen/core/ATen_fwd.h"
#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/core/Device.h"
#include "c10/core/Scalar.h"  // IWYU pragma: keep for c10::Scalar
#include "c10/core/SymInt.h"
#include "c10/core/SymIntArrayRef.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/fingerprint_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/op_names.h"

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

std::string FormatParamCacheKey(const MaybePromotedScalar& value) {
  if (value.ValueMatchesExclude()) {
    if (value.IsZero()) {
      return "0";
    }
    if (value.IsOne()) {
      return "1";
    }
    ABSL_LOG(FATAL)  // CRASH_OK
        << "Unexpected excluded scalar value: " << ToString(value.scalar());
  }
  return "";
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

bool IsScalarEqualTo(const at::Scalar& scalar, const int value) {
  if (scalar.isFloatingPoint()) {
    return scalar.toDouble() == value;
  }
  if (scalar.isIntegral(/*include_bool=*/true)) {
    return scalar.toLong() == value;
  }
  if (scalar.isComplex()) {
    return scalar.toComplexDouble() == static_cast<double>(value);
  }
  return false;
}

void AppendPromotedScalarPointers(
    std::vector<const PromotedScalarState* absl_nonnull>& promoted_scalars,
    const MaybePromotedScalar& arg) {
  if (!arg.ValueMatchesExclude()) {
    promoted_scalars.push_back(arg.state().get());
  }
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

MaybePromotedScalar PromotedScalar::AvoidPromoting(ScalarValue exclude) && {
  return MaybePromotedScalar(std::move(*this), exclude);
}

MaybePromotedScalar PromotedScalar::AvoidPromoting(ScalarValue exclude1,
                                                   ScalarValue exclude2) && {
  return MaybePromotedScalar(std::move(*this), exclude1, exclude2);
}

bool MaybePromotedScalar::MatchesAny(
    absl::Span<const ScalarValue> values) const {
  for (const auto& value : values) {
    if (value == ScalarValue::kZero && IsZero()) {
      return true;
    }
    if (value == ScalarValue::kOne && IsOne()) {
      return true;
    }
  }
  return false;
}

MaybePromotedScalar::MaybePromotedScalar(PromotedScalar s, ScalarValue exclude)
    : promoted_scalar_(std::move(s)),
      value_matches_exclude_(MatchesAny({exclude})) {}

MaybePromotedScalar::MaybePromotedScalar(PromotedScalar s, ScalarValue exclude1,
                                         ScalarValue exclude2)
    : promoted_scalar_(std::move(s)),
      value_matches_exclude_(MatchesAny({exclude1, exclude2})) {}

bool MaybePromotedScalar::IsZero() const {
  return internal::IsScalarEqualTo(promoted_scalar_.scalar(), 0);
}

bool MaybePromotedScalar::IsOne() const {
  return internal::IsScalarEqualTo(promoted_scalar_.scalar(), 1);
}

absl::StatusOr<at::Tensor> MaybePromotedScalar::GetTensor(
    std::optional<at::ScalarType> scalar_type_opt) {
  ABSL_CHECK(!value_matches_exclude_)  // CRASH_OK
      << "GetTensor() called when the scalar value " << ToString()
      << " should not be promoted.";
  return promoted_scalar_.GetTensor(scalar_type_opt);
}

[[nodiscard]] std::string MaybePromotedScalar::ToString() const {
  return promoted_scalar_.ToString();
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
  return absl::StrFormat("%016x_%016x_%016x", graph_key_.shapeless_key().key(),
                         graph_key_.dimensions_key().key(),
                         compile_options_key_.key());
}

std::ostream& operator<<(std::ostream& os, CompilationCacheKey key) {
  os << "CompilationCacheKey{shapeless_key=" << std::hex
     << key.graph_key().shapeless_key().key()
     << ", dimensions_key=" << key.graph_key().dimensions_key().key()
     << ", compile_options_key=" << key.compile_options_key().key() << std::dec
     << "}";
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
  key_ = FingerprintCat(upper_bounds, lower_bounds);
}

Dimensions GetUpperBounds(absl::Span<const DimensionBounds> bounds) {
  Dimensions dims;
  dims.reserve(bounds.size());
  for (auto& bound : bounds) {
    dims.push_back(bound.upper);
  }
  return dims;
}

ShapeDynamismMetadata::ShapeDynamismMetadata(
    absl::Span<const Shape> input_shapes,
    absl::Span<const Shape> output_shapes) {
  for (const Shape& shape : input_shapes) {
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

  for (const Shape& shape : output_shapes) {
    const int64_t tensor_start_dim = output_dimension_bounds_.size();
    for (int64_t dim : shape.dimensions()) {
      output_dimension_bounds_.push_back({dim, dim});
    }
    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      const int64_t dynamic_dim_index =
          tensor_start_dim + dynamic_dim.dimension;
      output_dimension_bounds_[dynamic_dim_index] = {dynamic_dim.lower_bound,
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

void LogShapes(absl::Span<const Shape> shapes, std::string_view prefix) {
  for (const auto& shape : shapes) {
    ABSL_VLOG(3) << prefix << " shape: " << ToString(shape.dimensions()) << " "
                 << ToString(shape.dtype());
  }
}

GraphKey PadModuleCacheKey(absl::Span<const Shape> dynamic_shapes) {
  ABSL_VLOG(3) << "[PadModuleCacheKey] Creating a cache key for a pad "
               << "module with " << dynamic_shapes.size();

  if (ABSL_VLOG_IS_ON(3)) {
    LogShapes(dynamic_shapes, "[PadModuleCacheKey]");
  }
  GraphSignature graph;

  for (const Shape& shape : dynamic_shapes) {
    graph.AddInput(shape.dimensions(), shape.dtype());
  }

  for (int i = 0; i < dynamic_shapes.size(); ++i) {
    const Shape& shape = dynamic_shapes[i];

    if (shape.dynamic_dimensions().empty()) {
      graph.AddGraphOutput(i);
      continue;
    }

    Dimensions padded_dimensions = shape.dimensions();
    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      padded_dimensions[dynamic_dim.dimension] = dynamic_dim.upper_bound;
    }

    const int padded_tensor_index = graph.AddOp(
        OpName::kPadUninitialized_, OpParamCacheKeys::Empty(),
        /*donated_inputs=*/{}, [&](GraphSignature::OpSignatureBuilder& op) {
          op.AddInput(i);
          op.AddOutput(padded_dimensions, shape.dtype());
        });
    graph.AddGraphOutput(padded_tensor_index);

    for (const auto& dynamic_dim : shape.dynamic_dimensions()) {
      TT_ASSIGN_OR_THROW(OpParamCacheKeys params,
                         *OpParamCacheKeysBuilder().SetParam(
                             "dimension_index", dynamic_dim.dimension));
      const int dim_size_index =
          graph.AddOp(OpName::kGetDimensionSize, params, /*donated_inputs=*/{},
                      [i](GraphSignature::OpSignatureBuilder& op) {
                        op.AddInput(i);
                        op.AddOutput({1}, mlir::ElementType::I32);
                      });
      graph.AddGraphOutput(dim_size_index);
    }
  }
  return graph.GetKey();
}

GraphKey SliceModuleCacheKey(
    absl::Span<const Dimensions> target_shapes,
    absl::Span<const Dimensions> padded_shapes,
    absl::Span<const mlir::ElementType> element_types) {
  ABSL_VLOG(3) << "[SliceModuleCacheKey] Creating a slice module cache key "
                  "for dynamism with "
               << target_shapes.size() << " shapes:";
  GraphSignature graph;
  std::vector<bool> requires_slicing(target_shapes.size(), false);
  TT_CHECK_THROW(target_shapes.size() == padded_shapes.size(), error::kInternal)
      << "target shapes and padded shapes must have the same size, "
      << "got " << target_shapes.size() << " and " << padded_shapes.size();

  for (int i = 0; i < target_shapes.size(); ++i) {
    const Dimensions& target_dims = target_shapes[i];
    const Dimensions padded_dims = padded_shapes[i];
    graph.AddInput(padded_dims, element_types[i]);
    for (int d = 0; d < target_dims.size(); ++d) {
      if (padded_dims[d] != target_dims[d]) {
        requires_slicing[i] = true;
        break;
      }
    }
  }
  for (int i = 0; i < target_shapes.size(); ++i) {
    if (!requires_slicing[i]) {
      graph.AddGraphOutput(i);
      continue;
    }
    int op_index = graph.AddOp(
        OpName::kSlice, OpParamCacheKeys::Empty(),
        /*donated_inputs=*/{}, [&](GraphSignature::OpSignatureBuilder& op) {
          op.AddInput(i);
          op.AddOutput(target_shapes[i], element_types[i]);
        });
    graph.AddGraphOutput(op_index);
  }
  return graph.GetKey();
}

GraphKey ShapeDynamismMetadata::GetSliceModuleCacheKey(
    absl::Span<const Shape> shapes) const {
  ABSL_VLOG(3) << "[GetSliceModuleCacheKey] Creating a slice module cache key "
                  "for dynamism with "
               << shapes.size() << " shapes:";
  std::vector<Dimensions> target_shapes;
  target_shapes.reserve(shapes.size());
  std::vector<Dimensions> padded_shapes;
  padded_shapes.reserve(shapes.size());
  std::vector<mlir::ElementType> element_types;
  element_types.reserve(shapes.size());
  absl::Span<const DimensionBounds> bounds = output_dimension_bounds_;
  int bounds_index = 0;
  for (int i = 0; i < shapes.size(); ++i) {
    const Shape& shape = shapes[i];
    element_types.push_back(shape.dtype());
    target_shapes.push_back(shape.dimensions());
    auto shape_bounds = bounds.subspan(bounds_index, shape.dimensions().size());
    Dimensions padded_dims = shape.dimensions();
    bounds_index += shape.dimensions().size();

    for (int d = 0; d < shape.dimensions().size(); ++d) {
      if (shape_bounds[d].lower != shape_bounds[d].upper) {
        padded_dims[d] = shape_bounds[d].upper;
      }
    }
    padded_shapes.push_back(std::move(padded_dims));
  }
  return SliceModuleCacheKey(target_shapes, padded_shapes, element_types);
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

GraphKey GraphSignature::GetKey() const {
  auto sorted_donated_indices = donated_indices_;
  std::sort(sorted_donated_indices.begin(), sorted_donated_indices.end());

  const ShapelessKey shapeless_key(FingerprintCat(
      graph_output_indices_, tensor_dimensions_starts_, tensor_element_types_,
      sorted_donated_indices, op_inputs_starts_, op_inputs_indices_, op_names_,
      op_param_cache_keys_starts_, op_param_cache_keys_, op_outputs_indices_));
  const DimensionsKey dimensions_key(tensor_dimensions_);
  return GraphKey(shapeless_key, dimensions_key);
}

}  // namespace torch_tpu
