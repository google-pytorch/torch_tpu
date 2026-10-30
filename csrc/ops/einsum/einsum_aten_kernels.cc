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

#include "csrc/ops/einsum/einsum_aten_kernels.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/TensorBody.h"
#include "ATen/ops/diagonal_scatter.h"
#include "ATen/ops/einsum.h"
#include "ATen/ops/einsum_native.h"
#include "ATen/ops/real.h"
#include "ATen/ops/zeros.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymInt.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/string_view.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fake_tensor_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/gather/gather.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/precision_context.h"
#include "csrc/ops/reductions/reductions.h"
#include "csrc/ops/reductions/sum.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

namespace {

constexpr uint8_t kNumLetters = 'z' - 'a' + 1;
constexpr uint8_t kTotalLabels = kNumLetters * 2;
constexpr uint8_t kEllipsis = kTotalLabels;

uint8_t LabelToSubscript(unsigned char label) {
  return std::isupper(label) ? label - 'A' : label - 'a' + kNumLetters;
}

char SubscriptToLabel(uint8_t s) {
  return static_cast<char>(s < kNumLetters ? s + 'A' : s + 'a' - kNumLetters);
}

// Records a repeated-label diagonal extraction step within a single operand.
// In forward, dimension `dim` (which duplicates `prev_dim`, with `prev_dim <
// dim`) is extracted along the diagonal into `prev_dim` and removed. In
// backward, `at::diagonal_scatter` embeds the gradient back into a zero tensor
// of shape `shape_before`.
struct DiagonalStep {
  Dimensions shape_before;
  int64_t prev_dim = 0;
  int64_t dim = 0;
  int64_t diag_size = 0;
};

struct ContractionStep {
  int64_t idx_a = 0;
  int64_t idx_b = 1;
};

struct ParsedEinsum {
  // Per-operand labels before diagonal deduplication (1-to-1 with original
  // operand dimensions).
  std::vector<Indices> raw_operand_labels;
  // Per-operand labels after diagonal deduplication (1-to-1 with
  // `dedup_operand_shapes`).
  std::vector<Indices> dedup_operand_labels;
  // Original operand shapes.
  std::vector<Dimensions> orig_operand_shapes;
  // Operand shapes after diagonal extraction.
  std::vector<Dimensions> dedup_operand_shapes;
  // Per-operand diagonal extraction steps.
  std::vector<std::vector<DiagonalStep>> diag_steps;
  // Output labels in `[0, total_expanded_labels)`.
  Indices rhs_labels;
  // Resolved broadcast dimension size for each label in
  // `[0, total_expanded_labels)`.
  Dimensions label_size;
  // Output tensor dimensions.
  Dimensions output_dims;
  // Product of `output_dims` (0 if any output dimension is 0).
  int64_t output_num_elements = 1;
  // True if any input operand has 0 elements (`numel() == 0`).
  bool any_operand_zero_elements = false;
  // Number of dimensions covered by `...` across all operands.
  int64_t ell_num_dim = 0;
  // Total number of unique letter labels + `ell_num_dim`.
  int64_t total_unique_labels = 0;
  // Character mapping `[0, total_expanded_labels) -> [A-Za-z]` for backward
  // equation synthesis (valid when `total_unique_labels <= kTotalLabels`).
  std::vector<char> label_to_char;
  // Contraction pairs `(idx_a, idx_b)` with `idx_a < idx_b`.
  std::vector<ContractionStep> contraction_steps;
  // True if an explicit `path` was provided (controls whether contracted
  // intermediates are appended to the back or inserted at the front of `ops`).
  bool has_path = false;
  // Original scalar types for each operand.
  std::vector<at::ScalarType> operand_dtypes;
  // Resolved output scalar type after pairwise promotions and contractions.
  at::ScalarType output_dtype = at::ScalarType::Undefined;
};

absl::Status ParseSingleLhsChar(unsigned char label, size_t& i,
                                c10::string_view lhs, size_t num_ops,
                                size_t& curr_op, bool& ell_in_input,
                                std::vector<std::vector<uint8_t>>& op_labels) {
  switch (label) {
    case ' ':
      return absl::OkStatus();

    case '.':
      TT_RET_CHECK(!ell_in_input, error::kInvalidArgument)
          << "expected at most one ellipsis ('...') for operand " << curr_op
          << ", got '.' after an ellipsis was already found";
      TT_RET_CHECK(i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.',
                   error::kInvalidArgument)
          << "expected '.' for operand " << curr_op
          << " to be part of an ellipsis ('...'), got '.' that is not part "
             "of any ellipsis";
      op_labels[curr_op].push_back(kEllipsis);
      ell_in_input = true;
      return absl::OkStatus();

    case ',':
      ++curr_op;
      TT_RET_CHECK(curr_op < num_ops, error::kInvalidArgument)
          << "expected the number of operands to match the equation, got "
             "fewer operands ("
          << num_ops << ") than specified in the equation";
      ell_in_input = false;
      return absl::OkStatus();

    default:
      TT_RET_CHECK(std::isalpha(label), error::kInvalidArgument)
          << "expected subscript at index " << i
          << " in the equation string to be in [a-zA-Z], got '"
          << static_cast<char>(label) << "'";
      op_labels[curr_op].push_back(LabelToSubscript(label));
      return absl::OkStatus();
  }
}

absl::StatusOr<std::vector<std::vector<uint8_t>>> ParseLhsSubscripts(
    c10::string_view lhs, size_t num_ops) {
  std::vector<std::vector<uint8_t>> op_labels(num_ops);
  bool ell_in_input = false;
  size_t curr_op = 0;
  for (size_t i = 0; i < lhs.length(); ++i) {
    TT_RETURN_IF_ERROR(ParseSingleLhsChar(lhs[i], i, lhs, num_ops, curr_op,
                                          ell_in_input, op_labels));
  }
  TT_RET_CHECK(curr_op == num_ops - 1, error::kInvalidArgument)
      << "expected " << (curr_op + 1)
      << " operand(s) as specified in the equation, got " << num_ops;
  return op_labels;
}

absl::Status CountSingleOperandLabels(size_t op_idx,
                                      const Dimensions& operand_shape,
                                      const std::vector<uint8_t>& labels,
                                      Dimensions& label_count,
                                      int64_t& ell_num_dim) {
  const int64_t ndims = static_cast<int64_t>(operand_shape.size());
  int64_t nlabels = static_cast<int64_t>(labels.size());
  bool has_ellipsis = false;

  for (const uint8_t label : labels) {
    if (label == kEllipsis) {
      --nlabels;
      has_ellipsis = true;
      ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
    } else {
      ++label_count[label];
    }
  }

  if (has_ellipsis) {
    TT_RET_CHECK(nlabels <= ndims, error::kInvalidArgument)
        << "expected the number of subscripts in the equation for operand "
        << op_idx << " to be at most the number of dimensions (" << ndims
        << "), got " << nlabels;
  } else {
    TT_RET_CHECK(nlabels == ndims, error::kInvalidArgument)
        << "expected the number of subscripts in the equation for operand "
        << op_idx << " to match the number of dimensions (" << ndims
        << "), got " << nlabels << " with no ellipsis given";
  }
  return absl::OkStatus();
}

absl::Status CountLabelsAndValidateDims(
    absl::Span<const Dimensions> operand_shapes,
    const std::vector<std::vector<uint8_t>>& op_labels, Dimensions& label_count,
    int64_t& ell_num_dim) {
  for (size_t i = 0; i < operand_shapes.size(); ++i) {
    TT_RETURN_IF_ERROR(CountSingleOperandLabels(
        i, operand_shapes[i], op_labels[i], label_count, ell_num_dim));
  }
  return absl::OkStatus();
}

Indices ParseImplicitRhsSubscripts(const Dimensions& label_count,
                                   int64_t ell_num_dim) {
  Indices rhs_labels;
  for (int64_t j = 0; j < ell_num_dim; ++j) {
    rhs_labels.push_back(kTotalLabels + j);
  }
  for (uint8_t label = 0; label < kTotalLabels; ++label) {
    if (label_count[label] == 1) {
      rhs_labels.push_back(label);
    }
  }
  return rhs_labels;
}

absl::Status ParseSingleRhsChar(
    unsigned char label, size_t& i, c10::string_view rhs, size_t lhs_size,
    const Dimensions& label_count, int64_t ell_num_dim, int64_t& perm_index,
    bool& ell_in_output, Dimensions& label_perm_index, Indices& rhs_labels) {
  switch (label) {
    case ' ':
      return absl::OkStatus();

    case '.':
      TT_RET_CHECK(!ell_in_output, error::kInvalidArgument)
          << "expected at most one ellipsis ('...') in the output, got '.' "
             "after an ellipsis was already found";
      TT_RET_CHECK(i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.',
                   error::kInvalidArgument)
          << "expected '.' in the output to be part of an ellipsis "
             "('...'), got '.' that is not part of any ellipsis";
      perm_index += ell_num_dim;
      ell_in_output = true;
      for (int64_t j = 0; j < ell_num_dim; ++j) {
        rhs_labels.push_back(kTotalLabels + j);
      }
      return absl::OkStatus();

    default: {
      TT_RET_CHECK(std::isalpha(label), error::kInvalidArgument)
          << "expected subscript at index " << (lhs_size + 2 + i)
          << " in the equation string to be in [a-zA-Z], got '"
          << static_cast<char>(label) << "'";
      const uint8_t index = LabelToSubscript(label);
      TT_RET_CHECK(label_count[index] > 0 && label_perm_index[index] == -1,
                   error::kInvalidArgument)
          << "expected output subscript '" << static_cast<char>(label)
          << "' to appear in at least one input operand and at most once "
             "in the output, got "
          << (label_perm_index[index] > -1
                  ? "a repeated output subscript"
                  : "a subscript that does not appear in any input operand");
      label_perm_index[index] = perm_index++;
      rhs_labels.push_back(index);
      return absl::OkStatus();
    }
  }
}

absl::StatusOr<Indices> ParseExplicitRhsSubscripts(
    c10::string_view rhs, size_t lhs_size, const Dimensions& label_count,
    int64_t ell_num_dim) {
  Dimensions label_perm_index(kTotalLabels, -1);
  int64_t perm_index = 0;
  bool ell_in_output = false;
  Indices rhs_labels;
  for (size_t i = 0; i < rhs.length(); ++i) {
    TT_RETURN_IF_ERROR(ParseSingleRhsChar(
        rhs[i], i, rhs, lhs_size, label_count, ell_num_dim, perm_index,
        ell_in_output, label_perm_index, rhs_labels));
  }
  return rhs_labels;
}

absl::Status ResolveOperandEllipsisDims(size_t op_idx, int64_t ndim,
                                        int64_t ell_num_dim,
                                        const Dimensions& curr_shape,
                                        int64_t& dim,
                                        Dimensions& first_seen_dim,
                                        ParsedEinsum& parsed) {
  for (int64_t j = ell_num_dim - ndim; j < ell_num_dim; ++j) {
    const int64_t exp_label = kTotalLabels + j;
    const int64_t d_size = curr_shape[dim];
    if (d_size != 1) {
      TT_RET_CHECK(parsed.label_size[exp_label] == 1 ||
                       parsed.label_size[exp_label] == d_size,
                   error::kInvalidArgument)
          << "expected dimension " << dim << " covered by ellipsis in operand "
          << op_idx << " to broadcast with previously seen ellipsis size "
          << parsed.label_size[exp_label] << ", got size " << d_size;
      parsed.label_size[exp_label] = d_size;
    }
    first_seen_dim[exp_label] = dim++;
    parsed.raw_operand_labels[op_idx].push_back(exp_label);
    parsed.dedup_operand_labels[op_idx].push_back(exp_label);
  }
  return absl::OkStatus();
}

absl::Status ResolveSingleOperandSubscript(size_t op_idx, int64_t orig_ndim,
                                           uint8_t s, size_t num_op_labels,
                                           int64_t ell_num_dim,
                                           Dimensions& curr_shape, int64_t& dim,
                                           Dimensions& first_seen_dim,
                                           ParsedEinsum& parsed) {
  if (s == kEllipsis) {
    const int64_t ndim = orig_ndim - (static_cast<int64_t>(num_op_labels) - 1);
    return ResolveOperandEllipsisDims(op_idx, ndim, ell_num_dim, curr_shape,
                                      dim, first_seen_dim, parsed);
  }
  if (first_seen_dim[s] == -1) {
    const int64_t d_size = curr_shape[dim];
    if (d_size != 1) {
      TT_RET_CHECK(parsed.label_size[s] == 1 || parsed.label_size[s] == d_size,
                   error::kInvalidArgument)
          << "expected subscript '" << SubscriptToLabel(s) << "' for operand "
          << op_idx << " to broadcast with previously seen size "
          << parsed.label_size[s] << ", got size " << d_size;
      parsed.label_size[s] = d_size;
    }
    first_seen_dim[s] = dim++;
    parsed.raw_operand_labels[op_idx].push_back(s);
    parsed.dedup_operand_labels[op_idx].push_back(s);
    return absl::OkStatus();
  }

  const int64_t prev_dim = first_seen_dim[s];
  const int64_t d_size = curr_shape[dim];
  const int64_t prev_size = curr_shape[prev_dim];
  TT_RET_CHECK(d_size == prev_size, error::kInvalidArgument)
      << "expected repeated subscript '" << SubscriptToLabel(s)
      << "' for operand " << op_idx << " to have matching dimension sizes, got "
      << d_size << " and " << prev_size;
  parsed.diag_steps[op_idx].push_back(DiagonalStep{.shape_before = curr_shape,
                                                   .prev_dim = prev_dim,
                                                   .dim = dim,
                                                   .diag_size = prev_size});
  curr_shape.erase(curr_shape.begin() + dim);
  parsed.raw_operand_labels[op_idx].push_back(s);
  return absl::OkStatus();
}

absl::Status ResolveSingleOperandDimensions(size_t op_idx,
                                            const Dimensions& orig_shape,
                                            const std::vector<uint8_t>& labels,
                                            int64_t total_expanded_labels,
                                            int64_t ell_num_dim,
                                            ParsedEinsum& parsed) {
  Dimensions curr_shape = orig_shape;
  parsed.orig_operand_shapes[op_idx] = curr_shape;
  if (absl::c_linear_search(curr_shape, 0)) {
    parsed.any_operand_zero_elements = true;
  }

  const int64_t orig_ndim = static_cast<int64_t>(orig_shape.size());
  Dimensions first_seen_dim(total_expanded_labels, -1);
  int64_t dim = 0;
  for (const uint8_t s : labels) {
    TT_RETURN_IF_ERROR(ResolveSingleOperandSubscript(
        op_idx, orig_ndim, s, labels.size(), ell_num_dim, curr_shape, dim,
        first_seen_dim, parsed));
  }
  parsed.dedup_operand_shapes[op_idx] = std::move(curr_shape);
  return absl::OkStatus();
}

absl::Status BuildContractionSteps(
    size_t num_ops, at::OptionalIntArrayRef path,
    std::vector<ContractionStep>& contraction_steps) {
  int64_t active_ops = static_cast<int64_t>(num_ops);
  size_t path_idx = 0;
  while (active_ops > 1) {
    int64_t i = 0;
    int64_t j = 1;
    if (path.has_value()) {
      i = (*path)[path_idx++];
      j = (*path)[path_idx++];
      if (j < i) {
        std::swap(i, j);
      }
      TT_RET_CHECK(i != j && i >= 0 && j < active_ops, error::kInvalidArgument)
          << "expected distinct contraction operand indices in [0, "
          << active_ops << "), got (" << i << ", " << j << ")";
    }
    contraction_steps.push_back(ContractionStep{.idx_a = i, .idx_b = j});
    --active_ops;
  }
  return absl::OkStatus();
}

void AssignEllipsisChars(const Dimensions& label_count, int64_t ell_num_dim,
                         std::vector<char>& label_to_char) {
  uint8_t next_unused = 0;
  for (int64_t j = 0; j < ell_num_dim; ++j) {
    while (next_unused < kTotalLabels && label_count[next_unused] > 0) {
      ++next_unused;
    }
    label_to_char[kTotalLabels + j] = SubscriptToLabel(next_unused++);
  }
}

void PopulateLabelToCharMap(const Dimensions& label_count, int64_t ell_num_dim,
                            int64_t total_expanded_labels,
                            ParsedEinsum& parsed) {
  int64_t used_letter_count = 0;
  for (uint8_t s = 0; s < kTotalLabels; ++s) {
    if (label_count[s] > 0) {
      ++used_letter_count;
    }
  }
  parsed.total_unique_labels = used_letter_count + ell_num_dim;
  if (parsed.total_unique_labels <= kTotalLabels) {
    parsed.label_to_char.resize(total_expanded_labels, '\0');
    for (uint8_t s = 0; s < kTotalLabels; ++s) {
      parsed.label_to_char[s] = SubscriptToLabel(s);
    }
    AssignEllipsisChars(label_count, ell_num_dim, parsed.label_to_char);
  }
}

absl::StatusOr<ParsedEinsum> ParseEinsumEquationFromShapes(
    c10::string_view equation, absl::Span<const Dimensions> operand_shapes,
    at::OptionalIntArrayRef path) {
  TT_RET_CHECK(!operand_shapes.empty(), error::kInvalidArgument)
      << "expected at least one operand, got 0";
  const size_t num_ops = operand_shapes.size();

  if (path.has_value()) {
    const size_t path_size = num_ops == 1 ? 1 : (num_ops - 1) * 2;
    TT_RET_CHECK(path->size() == path_size, error::kInvalidArgument)
        << "expected contraction path given in path parameter to have size "
        << path_size << ", got " << path->size();
  }

  const size_t arrow_pos = equation.find("->");
  const c10::string_view lhs = equation.substr(0, arrow_pos);
  TT_ASSIGN_OR_RETURN(const std::vector<std::vector<uint8_t>> op_labels,
                      ParseLhsSubscripts(lhs, num_ops));

  Dimensions label_count(kTotalLabels, 0);
  int64_t ell_num_dim = 0;
  TT_RETURN_IF_ERROR(CountLabelsAndValidateDims(operand_shapes, op_labels,
                                                label_count, ell_num_dim));

  Indices rhs_labels;
  if (arrow_pos == c10::string_view::npos) {
    rhs_labels = ParseImplicitRhsSubscripts(label_count, ell_num_dim);
  } else {
    TT_ASSIGN_OR_RETURN(rhs_labels, ParseExplicitRhsSubscripts(
                                        equation.substr(arrow_pos + 2),
                                        lhs.size(), label_count, ell_num_dim));
  }

  const int64_t total_expanded_labels = kTotalLabels + ell_num_dim;
  ParsedEinsum parsed;
  parsed.raw_operand_labels.resize(num_ops);
  parsed.dedup_operand_labels.resize(num_ops);
  parsed.orig_operand_shapes.resize(num_ops);
  parsed.dedup_operand_shapes.resize(num_ops);
  parsed.diag_steps.resize(num_ops);
  parsed.rhs_labels = std::move(rhs_labels);
  parsed.label_size.assign(total_expanded_labels, 1);
  parsed.ell_num_dim = ell_num_dim;
  parsed.has_path = path.has_value();

  for (size_t i = 0; i < num_ops; ++i) {
    TT_RETURN_IF_ERROR(ResolveSingleOperandDimensions(
        i, operand_shapes[i], op_labels[i], total_expanded_labels, ell_num_dim,
        parsed));
  }

  TT_RETURN_IF_ERROR(
      BuildContractionSteps(num_ops, path, parsed.contraction_steps));

  parsed.output_dims.reserve(parsed.rhs_labels.size());
  parsed.output_num_elements = 1;
  for (const int64_t c : parsed.rhs_labels) {
    const int64_t sz = parsed.label_size[c];
    parsed.output_dims.push_back(sz);
    parsed.output_num_elements *= sz;
  }

  PopulateLabelToCharMap(label_count, ell_num_dim, total_expanded_labels,
                         parsed);
  return parsed;
}

std::vector<bool> BuildInRhsMask(const ParsedEinsum& parsed) {
  std::vector<bool> in_rhs(parsed.label_size.size(), false);
  for (const int64_t c : parsed.rhs_labels) {
    in_rhs[c] = true;
  }
  return in_rhs;
}

std::vector<bool> BuildKeepLabelMask(const std::vector<bool>& in_rhs,
                                     const std::vector<Indices>& op_labels) {
  std::vector<bool> keep_label = in_rhs;
  for (const Indices& rem_labels : op_labels) {
    for (const int64_t c : rem_labels) {
      keep_label[c] = true;
    }
  }
  return keep_label;
}

struct ContractionClassification {
  Indices batch_labels;
  Indices contract_labels;
  Indices free_lhs_labels;
  Indices free_rhs_labels;
  Dimensions lhs_batch_dims;
  Dimensions rhs_batch_dims;
  Dimensions lhs_contract_dims;
  Dimensions rhs_contract_dims;
};

void ClassifyLhsContractionLabels(const Indices& lhs_labels,
                                  const Indices& rhs_labels_op,
                                  const std::vector<bool>& keep_label,
                                  ContractionClassification& cls) {
  for (size_t d_a = 0; d_a < lhs_labels.size(); ++d_a) {
    const int64_t c = lhs_labels[d_a];
    const auto it_b = absl::c_find(rhs_labels_op, c);
    if (it_b != rhs_labels_op.end()) {
      const int64_t d_b =
          static_cast<int64_t>(std::distance(rhs_labels_op.begin(), it_b));
      if (keep_label[c]) {
        cls.batch_labels.push_back(c);
        cls.lhs_batch_dims.push_back(static_cast<int64_t>(d_a));
        cls.rhs_batch_dims.push_back(d_b);
      } else {
        cls.contract_labels.push_back(c);
        cls.lhs_contract_dims.push_back(static_cast<int64_t>(d_a));
        cls.rhs_contract_dims.push_back(d_b);
      }
    } else {
      cls.free_lhs_labels.push_back(c);
    }
  }
}

ContractionClassification ClassifyContractionLabels(
    const Indices& lhs_labels, const Indices& rhs_labels_op,
    const std::vector<bool>& keep_label) {
  ContractionClassification cls;
  ClassifyLhsContractionLabels(lhs_labels, rhs_labels_op, keep_label, cls);
  for (const int64_t c : rhs_labels_op) {
    if (!absl::c_linear_search(lhs_labels, c)) {
      cls.free_rhs_labels.push_back(c);
    }
  }
  return cls;
}

Indices BuildDefaultStepOutLabels(const ContractionClassification& cls) {
  Indices out_labels = cls.batch_labels;
  out_labels.insert(out_labels.end(), cls.free_lhs_labels.begin(),
                    cls.free_lhs_labels.end());
  out_labels.insert(out_labels.end(), cls.free_rhs_labels.begin(),
                    cls.free_rhs_labels.end());
  return out_labels;
}

Indices FilterBroadcastSqueezedLabels(const Dimensions& dedup_shape,
                                      const Dimensions& label_size,
                                      const Indices& labels) {
  Indices remaining_labels;
  remaining_labels.reserve(labels.size());
  for (size_t d = 0; d < labels.size(); ++d) {
    const int64_t c = labels[d];
    if (!(dedup_shape[d] == 1 && label_size[c] != 1)) {
      remaining_labels.push_back(c);
    }
  }
  return remaining_labels;
}

Dimensions CountPostSqueezeLabels(size_t num_labels,
                                  const std::vector<Indices>& op_labels) {
  Dimensions post_squeeze_count(num_labels, 0);
  for (const Indices& labels : op_labels) {
    for (const int64_t c : labels) {
      ++post_squeeze_count[c];
    }
  }
  return post_squeeze_count;
}

Indices FilterPreReducedLabels(const std::vector<bool>& in_rhs,
                               const Dimensions& post_squeeze_count,
                               const Indices& labels) {
  Indices remaining_labels;
  remaining_labels.reserve(labels.size());
  for (const int64_t c : labels) {
    if (in_rhs[c] || post_squeeze_count[c] != 1) {
      remaining_labels.push_back(c);
    }
  }
  return remaining_labels;
}

std::vector<Indices> BuildInitialContractionLabels(
    const ParsedEinsum& parsed, const std::vector<bool>& in_rhs) {
  const size_t num_ops = parsed.dedup_operand_labels.size();
  std::vector<Indices> op_labels(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    op_labels[i] = FilterBroadcastSqueezedLabels(
        parsed.dedup_operand_shapes[i], parsed.label_size,
        parsed.dedup_operand_labels[i]);
  }
  const Dimensions post_squeeze_count =
      CountPostSqueezeLabels(parsed.label_size.size(), op_labels);
  for (size_t i = 0; i < num_ops; ++i) {
    op_labels[i] =
        FilterPreReducedLabels(in_rhs, post_squeeze_count, op_labels[i]);
  }
  return op_labels;
}

Indices FilterKeptPairOperandLabels(const std::vector<bool>& keep_label,
                                    const Indices& other_labels,
                                    const Indices& labels) {
  Indices kept_labels;
  kept_labels.reserve(labels.size());
  for (const int64_t c : labels) {
    if (keep_label[c] || absl::c_linear_search(other_labels, c)) {
      kept_labels.push_back(c);
    }
  }
  return kept_labels;
}

absl::Status ResolveSingleContractionStepDtype(
    const ContractionStep& step, bool has_path, const std::vector<bool>& in_rhs,
    std::vector<Indices>& op_labels, std::vector<at::ScalarType>& op_dtypes) {
  Indices lhs_labels = std::move(op_labels[step.idx_a]);
  Indices rhs_labels_op = std::move(op_labels[step.idx_b]);
  const at::ScalarType lhs_dtype = op_dtypes[step.idx_a];
  const at::ScalarType rhs_dtype = op_dtypes[step.idx_b];

  op_labels.erase(op_labels.begin() + step.idx_b);
  op_labels.erase(op_labels.begin() + step.idx_a);
  op_dtypes.erase(op_dtypes.begin() + step.idx_b);
  op_dtypes.erase(op_dtypes.begin() + step.idx_a);

  const std::vector<bool> keep_label = BuildKeepLabelMask(in_rhs, op_labels);
  lhs_labels =
      FilterKeptPairOperandLabels(keep_label, rhs_labels_op, lhs_labels);
  rhs_labels_op =
      FilterKeptPairOperandLabels(keep_label, lhs_labels, rhs_labels_op);

  const ContractionClassification cls =
      ClassifyContractionLabels(lhs_labels, rhs_labels_op, keep_label);
  if (!cls.contract_labels.empty()) {
    TT_RET_CHECK(lhs_dtype == rhs_dtype, error::kInvalidArgument)
        << "expected all tensors to have the same dtype, got "
        << ToString(lhs_dtype) << " and " << ToString(rhs_dtype);
  }
  const at::ScalarType step_dtype =
      cls.contract_labels.empty() ? c10::promoteTypes(lhs_dtype, rhs_dtype)
                                  : lhs_dtype;
  Indices out_labels = BuildDefaultStepOutLabels(cls);
  if (has_path) {
    op_labels.push_back(std::move(out_labels));
    op_dtypes.push_back(step_dtype);
  } else {
    op_labels.insert(op_labels.begin(), std::move(out_labels));
    op_dtypes.insert(op_dtypes.begin(), step_dtype);
  }
  return absl::OkStatus();
}

absl::Status ValidateAndResolveOperandDtypes(at::TensorList operands,
                                             ParsedEinsum& parsed) {
  parsed.operand_dtypes.reserve(operands.size());
  bool all_same_dtype = true;
  const at::ScalarType first_dtype = operands[0].scalar_type();
  for (const at::Tensor& op : operands) {
    TT_RET_CHECK(!IsBool(op), error::kPythonNotImplementedError)
        << "expected non-bool dtype, got " << ToString(op.scalar_type());
    const at::ScalarType dt = op.scalar_type();
    parsed.operand_dtypes.push_back(dt);
    if (dt != first_dtype) {
      all_same_dtype = false;
    }
  }
  if (all_same_dtype) {
    parsed.output_dtype = first_dtype;
    return absl::OkStatus();
  }

  const std::vector<bool> in_rhs = BuildInRhsMask(parsed);
  std::vector<Indices> op_labels =
      BuildInitialContractionLabels(parsed, in_rhs);
  std::vector<at::ScalarType> op_dtypes = parsed.operand_dtypes;
  for (const ContractionStep& step : parsed.contraction_steps) {
    TT_RETURN_IF_ERROR(ResolveSingleContractionStepDtype(
        step, parsed.has_path, in_rhs, op_labels, op_dtypes));
  }
  parsed.output_dtype = op_dtypes[0];
  return absl::OkStatus();
}

// Parses and validates the einsum equation, operand shapes/dtypes, and
// optional contraction path.
absl::StatusOr<ParsedEinsum> ParseEinsumEquation(c10::string_view equation,
                                                 at::TensorList operands,
                                                 at::OptionalIntArrayRef path) {
  std::vector<Dimensions> operand_shapes;
  operand_shapes.reserve(operands.size());
  for (const at::Tensor& op : operands) {
    operand_shapes.emplace_back(op.sizes().begin(), op.sizes().end());
  }
  TT_ASSIGN_OR_RETURN(ParsedEinsum parsed, ParseEinsumEquationFromShapes(
                                               equation, operand_shapes, path));
  TT_RETURN_IF_ERROR(ValidateAndResolveOperandDtypes(operands, parsed));
  return parsed;
}

// Extracts the diagonal along `(dim1, dim2)` (where `0 <= dim1 < dim2 < rank`)
// into `dim1` and removes `dim2` via `Iota + Unsqueeze + Broadcast +
// BuildGatherShlo + Squeeze`.
absl::StatusOr<mlir::MlirOp> BuildExtractDiagonalShlo(mlir::MlirOp op,
                                                      int64_t dim1,
                                                      int64_t dim2,
                                                      int64_t diag_size) {
  mlir::MlirBuilder& builder = op.getBuilder();
  const mlir::RankedTensorType type = GetTensorTypeOrDie(op);
  const llvm::ArrayRef<int64_t> shape = type.getShape();
  const int64_t n =
      mlir::ShapedType::isDynamic(shape[dim1]) ? diag_size : shape[dim1];
  if (n == 1) {
    return Squeeze(op, {dim2});
  }

  const auto iota_type =
      mlir::RankedTensorType::get({n}, builder.getOpBuilder().getI32Type());
  mlir::MlirOp iota = mlir::stablehlo::Iota(builder, iota_type, 0);
  TT_ASSIGN_OR_RETURN(mlir::MlirOp index_2d, Unsqueeze(iota, 1));

  Dimensions target_index_shape(shape.begin(), shape.end());
  target_index_shape[dim1] = n;
  target_index_shape[dim2] = 1;

  const Indices broadcast_dimensions = {dim1, dim2};
  TT_ASSIGN_OR_RETURN(
      mlir::MlirOp index_bcast,
      Broadcast(index_2d, target_index_shape, broadcast_dimensions));

  TT_ASSIGN_OR_RETURN(const mlir::ElementType element_type,
                      ConvertTo<mlir::ElementType>(type.getElementType()));
  TT_ASSIGN_OR_RETURN(mlir::MlirOp gather_res,
                      BuildGatherShlo(op, dim2, index_bcast,
                                      /*sparse_grad=*/false, element_type));

  return Squeeze(gather_res, {dim2});
}

// Emits `stablehlo.dot_general` with `i64 -> f64 -> i64` emulation (since XLA
// does not natively support `i64` DotGeneral on MXU/VPU) and configurable
// precision.
mlir::MlirOp BuildDotGeneral(mlir::MlirOp lhs, mlir::MlirOp rhs,
                             absl::Span<const int64_t> lhs_batch_dims,
                             absl::Span<const int64_t> rhs_batch_dims,
                             absl::Span<const int64_t> lhs_contract_dims,
                             absl::Span<const int64_t> rhs_contract_dims,
                             mlir::stablehlo::Precision precision) {
  const mlir::RankedTensorType lhs_type = GetTensorTypeOrDie(lhs);
  const mlir::Type orig_elem_type = lhs_type.getElementType();
  const bool is_i64 = orig_elem_type.isInteger(64);
  mlir::stablehlo::Precision effective_precision = precision;
  if (is_i64) {
    const mlir::Type f64_type =
        mlir::getElementType(lhs.getContext(), mlir::ElementType::F64);
    lhs = mlir::stablehlo::ConvertElementType(lhs, f64_type);
    rhs = mlir::stablehlo::ConvertElementType(rhs, f64_type);
    effective_precision =
        mlir::stablehlo::Precision::HIGHEST;  // EXPLICIT_PRECISION_OK=i64
  }

  const auto dot_dims = mlir::stablehlo::DotDimensionNumbersAttr::get(
      &lhs.getContext(), AsArrayRef(lhs_batch_dims), AsArrayRef(rhs_batch_dims),
      AsArrayRef(lhs_contract_dims), AsArrayRef(rhs_contract_dims));
  const auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      &lhs.getContext(), {effective_precision, effective_precision});
  mlir::MlirOp dot_result =
      mlir::stablehlo::DotGeneral(lhs, rhs, dot_dims, precision_config);
  if (is_i64) {
    return mlir::stablehlo::ConvertElementType(dot_result, orig_elem_type);
  }
  return dot_result;
}

absl::Status ExtractOperandDiagonals(const ParsedEinsum& parsed,
                                     std::vector<mlir::MlirOp>& ops) {
  for (size_t i = 0; i < ops.size(); ++i) {
    for (const DiagonalStep& step : parsed.diag_steps[i]) {
      TT_ASSIGN_OR_RETURN(
          ops[i], BuildExtractDiagonalShlo(ops[i], step.prev_dim, step.dim,
                                           step.diag_size));
    }
  }
  return absl::OkStatus();
}

absl::Status SqueezeSingleOperandBroadcastDims(const Dimensions& dedup_shape,
                                               const Dimensions& label_size,
                                               mlir::MlirOp& op,
                                               Indices& labels) {
  Dimensions squeeze_dims;
  Indices remaining_labels;
  remaining_labels.reserve(labels.size());
  for (size_t d = 0; d < labels.size(); ++d) {
    const int64_t c = labels[d];
    if (dedup_shape[d] == 1 && label_size[c] != 1) {
      squeeze_dims.push_back(static_cast<int64_t>(d));
    } else {
      remaining_labels.push_back(c);
    }
  }
  if (!squeeze_dims.empty()) {
    TT_ASSIGN_OR_RETURN(op, Squeeze(op, squeeze_dims));
    labels = std::move(remaining_labels);
  }
  return absl::OkStatus();
}

absl::Status SqueezeBroadcastDims(const ParsedEinsum& parsed,
                                  std::vector<mlir::MlirOp>& ops,
                                  std::vector<Indices>& op_labels) {
  for (size_t i = 0; i < ops.size(); ++i) {
    TT_RETURN_IF_ERROR(SqueezeSingleOperandBroadcastDims(
        parsed.dedup_operand_shapes[i], parsed.label_size, ops[i],
        op_labels[i]));
  }
  return absl::OkStatus();
}

absl::Status ReduceSingleOperandUnkeptDims(const std::vector<bool>& in_rhs,
                                           const Dimensions& post_squeeze_count,
                                           mlir::MlirOp& op, Indices& labels) {
  Dimensions sum_dims;
  Indices remaining_labels;
  remaining_labels.reserve(labels.size());
  for (size_t d = 0; d < labels.size(); ++d) {
    const int64_t c = labels[d];
    if (!in_rhs[c] && post_squeeze_count[c] == 1) {
      sum_dims.push_back(static_cast<int64_t>(d));
    } else {
      remaining_labels.push_back(c);
    }
  }
  if (!sum_dims.empty()) {
    TT_ASSIGN_OR_RETURN(op,
                        BuildSumShlo(op, sum_dims, ReductionMode::kDropDims));
    labels = std::move(remaining_labels);
  }
  return absl::OkStatus();
}

absl::Status PreReduceSingleOperandLabels(const ParsedEinsum& parsed,
                                          const std::vector<bool>& in_rhs,
                                          std::vector<mlir::MlirOp>& ops,
                                          std::vector<Indices>& op_labels) {
  const Dimensions post_squeeze_count =
      CountPostSqueezeLabels(parsed.label_size.size(), op_labels);
  for (size_t i = 0; i < ops.size(); ++i) {
    TT_RETURN_IF_ERROR(ReduceSingleOperandUnkeptDims(in_rhs, post_squeeze_count,
                                                     ops[i], op_labels[i]));
  }
  return absl::OkStatus();
}

absl::Status SumUnkeptSingleOperandDims(const std::vector<bool>& keep_label,
                                        const Indices& other_labels,
                                        mlir::MlirOp& op, Indices& labels) {
  Dimensions sum_dims;
  Indices kept_labels;
  kept_labels.reserve(labels.size());
  for (size_t d = 0; d < labels.size(); ++d) {
    const int64_t c = labels[d];
    if (!keep_label[c] && !absl::c_linear_search(other_labels, c)) {
      sum_dims.push_back(static_cast<int64_t>(d));
    } else {
      kept_labels.push_back(c);
    }
  }
  if (!sum_dims.empty()) {
    TT_ASSIGN_OR_RETURN(op,
                        BuildSumShlo(op, sum_dims, ReductionMode::kDropDims));
    labels = std::move(kept_labels);
  }
  return absl::OkStatus();
}

absl::Status PromoteUncontractedPairOperands(mlir::MlirOp& lhs_op,
                                             mlir::MlirOp& rhs_op) {
  const mlir::ElementType lhs_elem = GetElementTypeOrDie(lhs_op);
  const mlir::ElementType rhs_elem = GetElementTypeOrDie(rhs_op);
  if (lhs_elem == rhs_elem) {
    return absl::OkStatus();
  }
  const at::ScalarType promoted_dtype = c10::promoteTypes(
      ConvertTo<at::ScalarType>(lhs_elem), ConvertTo<at::ScalarType>(rhs_elem));
  TT_ASSIGN_OR_RETURN(const mlir::ElementType target_elem,
                      ConvertTo<mlir::ElementType>(promoted_dtype));
  TT_ASSIGN_OR_RETURN(lhs_op,
                      ComplexSafeConvertElementType(lhs_op, target_elem));
  TT_ASSIGN_OR_RETURN(rhs_op,
                      ComplexSafeConvertElementType(rhs_op, target_elem));
  return absl::OkStatus();
}

// On the final contraction step, if swapping `(lhs_op, rhs_op)` makes
// `DotGeneral`'s output label order (`batch + free_rhs + free_lhs`) match
// `rhs_labels` directly, swaps the operands to eliminate the final `Transpose`.
void MaybeSwapFinalDotGeneralOperands(const Indices& rhs_labels,
                                      mlir::MlirOp& lhs_op,
                                      mlir::MlirOp& rhs_op,
                                      ContractionClassification& cls,
                                      Indices& out_labels) {
  if (out_labels == rhs_labels) {
    return;
  }
  Indices swapped_out_labels = cls.batch_labels;
  swapped_out_labels.insert(swapped_out_labels.end(),
                            cls.free_rhs_labels.begin(),
                            cls.free_rhs_labels.end());
  swapped_out_labels.insert(swapped_out_labels.end(),
                            cls.free_lhs_labels.begin(),
                            cls.free_lhs_labels.end());
  if (swapped_out_labels == rhs_labels) {
    std::swap(lhs_op, rhs_op);
    std::swap(cls.lhs_batch_dims, cls.rhs_batch_dims);
    std::swap(cls.lhs_contract_dims, cls.rhs_contract_dims);
    out_labels = std::move(swapped_out_labels);
  }
}

absl::Status ExecuteContractionStep(const ContractionStep& step,
                                    const ParsedEinsum& parsed,
                                    const std::vector<bool>& in_rhs,
                                    mlir::stablehlo::Precision precision,
                                    std::vector<mlir::MlirOp>& ops,
                                    std::vector<Indices>& op_labels) {
  mlir::MlirOp lhs_op = ops[step.idx_a];
  mlir::MlirOp rhs_op = ops[step.idx_b];
  Indices lhs_labels = std::move(op_labels[step.idx_a]);
  Indices rhs_labels_op = std::move(op_labels[step.idx_b]);

  ops.erase(ops.begin() + step.idx_b);
  ops.erase(ops.begin() + step.idx_a);
  op_labels.erase(op_labels.begin() + step.idx_b);
  op_labels.erase(op_labels.begin() + step.idx_a);

  const std::vector<bool> keep_label = BuildKeepLabelMask(in_rhs, op_labels);
  TT_RETURN_IF_ERROR(SumUnkeptSingleOperandDims(keep_label, rhs_labels_op,
                                                lhs_op, lhs_labels));
  TT_RETURN_IF_ERROR(SumUnkeptSingleOperandDims(keep_label, lhs_labels, rhs_op,
                                                rhs_labels_op));

  ContractionClassification cls =
      ClassifyContractionLabels(lhs_labels, rhs_labels_op, keep_label);
  Indices out_labels = BuildDefaultStepOutLabels(cls);

  if (cls.contract_labels.empty()) {
    TT_RETURN_IF_ERROR(PromoteUncontractedPairOperands(lhs_op, rhs_op));
  }
  if (ops.empty()) {
    MaybeSwapFinalDotGeneralOperands(parsed.rhs_labels, lhs_op, rhs_op, cls,
                                     out_labels);
  }
  mlir::MlirOp step_result =
      BuildDotGeneral(lhs_op, rhs_op, cls.lhs_batch_dims, cls.rhs_batch_dims,
                      cls.lhs_contract_dims, cls.rhs_contract_dims, precision);

  if (parsed.has_path) {
    ops.push_back(step_result);
    op_labels.push_back(std::move(out_labels));
  } else {
    ops.insert(ops.begin(), step_result);
    op_labels.insert(op_labels.begin(), std::move(out_labels));
  }
  return absl::OkStatus();
}

mlir::MlirOp TransposeToOutputOrder(mlir::MlirOp result,
                                    const Indices& final_labels,
                                    const Indices& rhs_labels) {
  if (final_labels == rhs_labels) {
    return result;
  }
  Dimensions perm;
  perm.reserve(rhs_labels.size());
  for (const int64_t target_c : rhs_labels) {
    const auto it = absl::c_find(final_labels, target_c);
    perm.push_back(
        static_cast<int64_t>(std::distance(final_labels.begin(), it)));
  }
  return mlir::stablehlo::Transpose(result, perm);
}

// Lowers `aten::einsum` directly to StableHLO (`BuildExtractDiagonalShlo`,
// broadcast `Squeeze`, unary `BuildSumShlo`, pairwise `BuildDotGeneral`, and
// a single final `Transpose` when needed).
// Note: Conjugation (`tensor.is_conj()`) is already materialized by
// `GetBuffers` during view decomposition before `BuildEinsumShlo` is invoked.
absl::StatusOr<mlir::MlirOp> BuildEinsumShlo(
    absl::Span<const mlir::MlirOp> inputs, mlir::MlirBuilder& builder,
    const ParsedEinsum& parsed, mlir::ElementType out_dtype,
    mlir::stablehlo::Precision precision) {
  if (parsed.output_num_elements == 0) {
    return MakeZeroSizedTensor(builder, out_dtype,
                               AsArrayRef<int64_t>(parsed.output_dims));
  }
  if (parsed.any_operand_zero_elements) {
    return MakeConstant(builder, at::Scalar(0), out_dtype,
                        AsArrayRef<int64_t>(parsed.output_dims));
  }

  std::vector<mlir::MlirOp> ops(inputs.begin(), inputs.end());
  std::vector<Indices> op_labels = parsed.dedup_operand_labels;

  // Step 1: Extract diagonals for any operand with repeated labels.
  TT_RETURN_IF_ERROR(ExtractOperandDiagonals(parsed, ops));

  // Step 2: Squeeze size-1 dimensions that broadcast against a dimension of
  // size != 1 in another operand (`dedup_shape[d] == 1 && label_size[c] != 1`).
  TT_RETURN_IF_ERROR(SqueezeBroadcastDims(parsed, ops, op_labels));

  // Step 3: Unary pre-reduction (`BuildSumShlo`) for any label `c` that appears
  // in only 1 operand (post-squeeze) and is not in `rhs_labels`.
  const std::vector<bool> in_rhs = BuildInRhsMask(parsed);
  TT_RETURN_IF_ERROR(
      PreReduceSingleOperandLabels(parsed, in_rhs, ops, op_labels));

  // Step 4: Pairwise contractions along `contraction_steps` via `DotGeneral`.
  for (const ContractionStep& step : parsed.contraction_steps) {
    TT_RETURN_IF_ERROR(ExecuteContractionStep(step, parsed, in_rhs, precision,
                                              ops, op_labels));
  }

  // Step 5: Emit at most one final `Transpose` if `final_labels` differs from
  // `parsed.rhs_labels`.
  return TransposeToOutputOrder(ops[0], op_labels[0], parsed.rhs_labels);
}

bool IsPythonWrapperSubclass(const at::Tensor& tensor) {
  return tensor.defined() && tensor.key_set().has(c10::DispatchKey::Python) &&
         (!tensor.has_storage() ||
          tensor.storage().device_type() != c10::DeviceType::Meta);
}

bool HasSymbolicSizes(const at::Tensor& tensor) {
  for (const c10::SymInt& s : tensor.sym_sizes()) {
    if (s.is_symbolic()) {
      return true;
    }
  }
  return false;
}

bool ShouldFallbackToNativeEinsum(at::TensorList tensors) {
  if (IsFunctorchActive()) {
    return true;
  }
  for (const at::Tensor& t : tensors) {
    if (!t.defined()) {
      continue;
    }
    if (IsPythonWrapperSubclass(t) || torch::autograd::isFwGradDefined(t)) {
      return true;
    }
  }
  return false;
}

bool AnyOperandHasSymbolicSizes(at::TensorList tensors) {
  for (const at::Tensor& t : tensors) {
    if (t.defined() && HasSymbolicSizes(t)) {
      return true;
    }
  }
  return false;
}

template <typename T>
auto MakeAutogradNode() {
  if constexpr (std::is_base_of_v<c10::intrusive_ptr_target, T>) {
    return c10::make_intrusive<T>();
  } else {
    return std::shared_ptr<T>(new T(), [](T* ptr) { deleteNode(ptr); });
  }
}

std::string LabelsToSubscriptString(const Indices& labels,
                                    const std::vector<char>& label_to_char) {
  std::string sub;
  sub.reserve(labels.size());
  for (const int64_t c : labels) {
    sub.push_back(label_to_char[c]);
  }
  return sub;
}

std::vector<at::Tensor> UnpackSavedOperands(
    const std::vector<torch::autograd::SavedVariable>& saved_operands,
    const std::vector<bool>& is_saved) {
  const size_t num_ops = saved_operands.size();
  std::vector<at::Tensor> unpacked_operands(num_ops);
  for (size_t j = 0; j < num_ops; ++j) {
    if (is_saved[j]) {
      unpacked_operands[j] = saved_operands[j].unpack();
    }
  }
  return unpacked_operands;
}

std::string BuildBackwardLhs(size_t target_k, const at::Tensor& grad_out,
                             at::ScalarType bwd_dtype,
                             const std::string& grad_out_sub,
                             const std::vector<at::Tensor>& unpacked_operands,
                             const std::vector<std::string>& raw_subs,
                             const ParsedEinsum& parsed,
                             std::vector<at::Tensor>& bwd_operands,
                             std::vector<bool>& in_bwd_lhs) {
  const size_t num_ops = parsed.orig_operand_shapes.size();
  bwd_operands.reserve(num_ops);
  bwd_operands.push_back(
      grad_out.scalar_type() == bwd_dtype ? grad_out : grad_out.to(bwd_dtype));

  std::string bwd_lhs = grad_out_sub;
  for (const int64_t c : parsed.rhs_labels) {
    in_bwd_lhs[c] = true;
  }

  for (size_t j = 0; j < num_ops; ++j) {
    if (j == target_k) {
      continue;
    }
    const at::Tensor& other_op = unpacked_operands[j];
    at::Tensor op_j = other_op.is_complex() ? other_op.conj() : other_op;
    if (op_j.scalar_type() != bwd_dtype) {
      op_j = op_j.to(bwd_dtype);
    }
    bwd_operands.push_back(std::move(op_j));
    bwd_lhs.push_back(',');
    bwd_lhs.append(raw_subs[j]);
    for (const int64_t c : parsed.raw_operand_labels[j]) {
      in_bwd_lhs[c] = true;
    }
  }
  return bwd_lhs;
}

std::string BuildBackwardRhs(const Indices& dedup_labels,
                             const Dimensions& dedup_shape,
                             const std::vector<bool>& in_bwd_lhs,
                             const std::vector<char>& label_to_char,
                             std::vector<bool>& dim_kept) {
  std::string bwd_rhs;
  bwd_rhs.reserve(dedup_labels.size());
  for (size_t d = 0; d < dedup_labels.size(); ++d) {
    const int64_t c = dedup_labels[d];
    if (in_bwd_lhs[c] && dedup_shape[d] != 1) {
      bwd_rhs.push_back(label_to_char[c]);
      dim_kept[d] = true;
    }
  }
  return bwd_rhs;
}

at::Tensor RestoreAndScatterGrad(at::Tensor grad_k,
                                 const std::vector<bool>& dim_kept,
                                 const Dimensions& dedup_shape,
                                 const std::vector<DiagonalStep>& diag_steps) {
  for (size_t d = 0; d < dim_kept.size(); ++d) {
    if (!dim_kept[d]) {
      grad_k = grad_k.unsqueeze(static_cast<int64_t>(d));
    }
  }
  if (grad_k.sizes() != at::IntArrayRef(dedup_shape)) {
    grad_k = grad_k.expand(dedup_shape);
  }

  for (int64_t s = static_cast<int64_t>(diag_steps.size()) - 1; s >= 0; --s) {
    const DiagonalStep& step = diag_steps[s];
    grad_k =
        at::diagonal_scatter(at::zeros(step.shape_before, grad_k.options()),
                             grad_k.movedim(step.prev_dim, -1),
                             /*offset=*/0, step.prev_dim, step.dim);
  }
  return grad_k;
}

at::ScalarType ResolveBackwardEinsumDtype(
    size_t target_k, at::ScalarType grad_out_dtype,
    const std::vector<at::Tensor>& unpacked_operands) {
  at::ScalarType bwd_dtype = grad_out_dtype;
  for (size_t j = 0; j < unpacked_operands.size(); ++j) {
    if (j != target_k) {
      bwd_dtype =
          c10::promoteTypes(bwd_dtype, unpacked_operands[j].scalar_type());
    }
  }
  return bwd_dtype;
}

at::Tensor CastGradToTargetDtype(at::Tensor grad_k,
                                 at::ScalarType target_dtype) {
  if (grad_k.scalar_type() == target_dtype) {
    return grad_k;
  }
  if (grad_k.is_complex() && !c10::isComplexType(target_dtype)) {
    grad_k = at::real(grad_k);
  }
  if (grad_k.scalar_type() != target_dtype) {
    grad_k = grad_k.to(target_dtype);
  }
  return grad_k;
}

at::Tensor ComputeOperandGrad(size_t k, const at::Tensor& grad_out,
                              const std::string& grad_out_sub,
                              const std::vector<at::Tensor>& unpacked_operands,
                              const std::vector<std::string>& raw_subs,
                              const ParsedEinsum& parsed) {
  const Dimensions& orig_shape = parsed.orig_operand_shapes[k];
  const at::ScalarType target_dtype = parsed.operand_dtypes[k];
  if (grad_out.numel() == 0 || parsed.any_operand_zero_elements) {
    return at::zeros(orig_shape, grad_out.options().dtype(target_dtype));
  }

  const at::ScalarType bwd_dtype =
      ResolveBackwardEinsumDtype(k, grad_out.scalar_type(), unpacked_operands);
  std::vector<at::Tensor> bwd_operands;
  std::vector<bool> in_bwd_lhs(parsed.label_size.size(), false);
  const std::string bwd_lhs =
      BuildBackwardLhs(k, grad_out, bwd_dtype, grad_out_sub, unpacked_operands,
                       raw_subs, parsed, bwd_operands, in_bwd_lhs);

  const Indices& dedup_labels = parsed.dedup_operand_labels[k];
  const Dimensions& dedup_shape = parsed.dedup_operand_shapes[k];
  std::vector<bool> dim_kept(dedup_labels.size(), false);
  const std::string bwd_rhs = BuildBackwardRhs(
      dedup_labels, dedup_shape, in_bwd_lhs, parsed.label_to_char, dim_kept);

  const std::string bwd_eq = bwd_lhs + "->" + bwd_rhs;
  at::Tensor grad_k = at::einsum(bwd_eq, bwd_operands);
  grad_k = RestoreAndScatterGrad(std::move(grad_k), dim_kept, dedup_shape,
                                 parsed.diag_steps[k]);
  return CastGradToTargetDtype(std::move(grad_k), target_dtype);
}

// Custom autograd node (`torch::autograd::Node`) for `aten::einsum`.
// Computes each input gradient via a single inverted `at::einsum` call (which
// lowers directly to `stablehlo.dot_general`), avoiding the intermediate
// `permute`/`reshape`/`view` autograd nodes of `at::native::einsum`.
struct EinsumBackwardNode : public torch::autograd::Node {
  std::vector<torch::autograd::SavedVariable> saved_operands;
  std::vector<bool> is_saved;
  ParsedEinsum parsed;

  std::string name() const override { return "EinsumBackward0"; }

  torch::autograd::variable_list apply(
      torch::autograd::variable_list&& grads) override {
    const size_t num_ops = parsed.orig_operand_shapes.size();
    torch::autograd::variable_list grad_inputs(num_ops);
    if (grads.empty() || !grads[0].defined()) {
      return grad_inputs;
    }
    const at::Tensor& grad_out = grads[0];

    const std::vector<at::Tensor> unpacked_operands =
        UnpackSavedOperands(saved_operands, is_saved);
    const std::string grad_out_sub =
        LabelsToSubscriptString(parsed.rhs_labels, parsed.label_to_char);

    std::vector<std::string> raw_subs(num_ops);
    for (size_t j = 0; j < num_ops; ++j) {
      raw_subs[j] = LabelsToSubscriptString(parsed.raw_operand_labels[j],
                                            parsed.label_to_char);
    }

    for (size_t k = 0; k < num_ops; ++k) {
      if (task_should_compute_output(k)) {
        grad_inputs[k] = ComputeOperandGrad(
            k, grad_out, grad_out_sub, unpacked_operands, raw_subs, parsed);
      }
    }

    return grad_inputs;
  }

  void release_variables() override {
    for (torch::autograd::SavedVariable& saved_op : saved_operands) {
      saved_op.reset_data();
    }
  }
};

}  // namespace

absl::StatusOr<mlir::MlirOp> BuildEinsumShlo(
    const absl::Span<const mlir::MlirOp> inputs,
    const c10::string_view equation,
    const mlir::stablehlo::Precision precision) {
  std::vector<Dimensions> operand_shapes;
  operand_shapes.reserve(inputs.size());
  for (const mlir::MlirOp& op : inputs) {
    const llvm::ArrayRef<int64_t> shape = GetTensorTypeOrDie(op).getShape();
    operand_shapes.emplace_back(shape.begin(), shape.end());
  }
  TT_ASSIGN_OR_RETURN(const ParsedEinsum parsed,
                      ParseEinsumEquationFromShapes(equation, operand_shapes,
                                                    /*path=*/std::nullopt));
  at::ScalarType promoted_dtype =
      ConvertTo<at::ScalarType>(GetElementTypeOrDie(inputs[0]));
  for (size_t i = 1; i < inputs.size(); ++i) {
    promoted_dtype = c10::promoteTypes(
        promoted_dtype,
        ConvertTo<at::ScalarType>(GetElementTypeOrDie(inputs[i])));
  }
  TT_ASSIGN_OR_RETURN(const mlir::ElementType out_dtype,
                      ConvertTo<mlir::ElementType>(promoted_dtype));
  mlir::MlirOp first_input = inputs[0];
  mlir::MlirBuilder& builder = first_input.getBuilder();
  return BuildEinsumShlo(inputs, builder, parsed, out_dtype, precision);
}

at::Tensor AtenEinsumAutograd(c10::string_view equation, at::TensorList tensors,
                              at::OptionalIntArrayRef path) {
  if (ShouldFallbackToNativeEinsum(tensors)) {
    return at::native::einsum(equation, tensors, path);
  }

  const size_t num_ops = tensors.size();
  size_t num_requires_grad = 0;
  size_t single_grad_idx = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    if (tensors[i].defined() &&
        torch::autograd::compute_requires_grad(tensors[i])) {
      ++num_requires_grad;
      single_grad_idx = i;
    }
  }
  const bool any_requires_grad = num_requires_grad > 0;

  if (!any_requires_grad) {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::einsum(equation, tensors, path);
  }

  if (AnyOperandHasSymbolicSizes(tensors)) {
    return at::native::einsum(equation, tensors, path);
  }

  // Selective `SavedVariable` saving: operand `j` is only needed during
  // backward if at least one OTHER operand `k != j` requires a gradient.
  std::vector<torch::autograd::SavedVariable> saved_operands(num_ops);
  std::vector<bool> is_saved(num_ops, false);
  for (size_t j = 0; j < num_ops; ++j) {
    const bool needed_by_other =
        (num_requires_grad > 1) ||
        (num_requires_grad == 1 && j != single_grad_idx);
    if (needed_by_other) {
      saved_operands[j] =
          torch::autograd::SavedVariable(tensors[j], /*is_output=*/false);
      is_saved[j] = true;
    }
  }

  at::Tensor output = ([&]() {  // REF_CAPTURE_OK=called locally
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::einsum(equation, tensors, path);
  })();

  absl::StatusOr<ParsedEinsum> parsed =
      ParseEinsumEquation(equation, tensors, path);
  if (!parsed.ok() || parsed->total_unique_labels > kTotalLabels) {
    return at::native::einsum(equation, tensors, path);
  }

  auto grad_fn = MakeAutogradNode<EinsumBackwardNode>();
  grad_fn->set_next_edges(torch::autograd::collect_next_edges(tensors));
  grad_fn->saved_operands = std::move(saved_operands);
  grad_fn->is_saved = std::move(is_saved);
  grad_fn->parsed = *std::move(parsed);
  torch::autograd::set_history(torch::autograd::flatten_tensor_args(output),
                               grad_fn);
  return output;
}

at::Tensor AtenEinsum(c10::string_view equation, at::TensorList tensors,
                      at::OptionalIntArrayRef path) {
  TT_KERNEL(OpName::kEinsum, param_keys, (equation, tensors, path), {
    TT_ASSIGN_OR_THROW(ParsedEinsum parsed,
                       ParseEinsumEquation(equation, tensors, path));
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(parsed.output_dtype));
    const mlir::stablehlo::Precision current_precision =
        GetAndAddPrecisionTo(param_keys);

    Dimensions output_dims = parsed.output_dims;
    const std::vector<at::Tensor> inputs_vec(tensors.begin(), tensors.end());

    auto op_builder =
        [parsed = std::move(parsed), out_dtype, current_precision](
            absl::Span<mlir::MlirOp> inputs,
            mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
      return BuildEinsumShlo(inputs, builder, parsed, out_dtype,
                             current_precision);
    };

    TT_ASSIGN_OR_THROW(DeviceBufferRef result_buffer,
                       (DispatchOp<kDynamicSize>(
                           std::move(op_builder), inputs_vec,
                           {.out_dtype = out_dtype,
                            .out_dims = std::move(output_dims),
                            .op_param_cache_keys = std::move(param_keys)})));
    return MakeTensor(std::move(result_buffer));
  });
}

}  // namespace torch_tpu
