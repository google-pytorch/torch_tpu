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

#include "torch_tpu/eager/op_dispatcher.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/shape.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/eager_mode.h"
#include "torch_tpu/eager/materialize.h"
#include "torch_tpu/eager/materialize_new.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/copy_from/cpu_to_tpu.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "tsl/profiler/lib/traceme.h"

ABSL_FLAG(std::string, torch_tpu_internal_detect_repeated_ops, "safe",
          "Look for repeated sequences of ops and compile them as a whole. "
          "Possible values are \"safe\", \"aggressive\", or \"\" if not used");
ABSL_DECLARE_FLAG(bool, torch_tpu_internal_enable_new_materialization);

namespace torch_tpu {
namespace {

constexpr int kMinRepeatedSubsequenceLength = 10;
constexpr int kMaxRepeatedSubsequenceLength = 128;

// The result of trying to skip a list of (potentially) zero-sized outputs.
struct SkipIfAllZeroSizedResult {
  // The inputs were not all zero-sized, so we can't skip.
  struct NotAllZeroSized {};
  // We successfully skipped the op.
  struct Skipped {
    std::vector<DeviceBufferRef> zero_sized_buffers;
  };

  std::variant<NotAllZeroSized, Skipped> result;
};

// If all outputs are zero-sized, we can avoid dispatching the op entirely,
// and just return a vector of appropriately zero-sized DeviceBufferRefs.
absl::StatusOr<SkipIfAllZeroSizedResult> SkipIfAllZeroSized(
    absl::Span<const mlir::ElementType> output_dtypes,
    absl::Span<const absl::Span<const int64_t>> output_dims) {
  // Intentionally delay allocation; the most common case is that we have at
  // least one non-zero-sized output.
  std::vector<DeviceBufferRef> zero_sized_buffers;
  int num_outputs = output_dtypes.size();
  for (int i = 0; i < num_outputs; ++i) {
    const auto& output_dim_set = output_dims[i];
    // Scalars (rank-0 outputs) are not zero-sized.
    bool zero_sized = false;
    // If a tensor has any zero-sized dimension, it has zero elements.
    for (int64_t dim : output_dim_set) {
      if (dim == 0) {
        zero_sized = true;
        break;
      }
    }
    if (!zero_sized) {
      // Found a non-zero-sized output. We need to actually dispatch.
      return SkipIfAllZeroSizedResult{
          SkipIfAllZeroSizedResult::NotAllZeroSized{}};
    }
    // Found a zero-sized output. Push a zero-sized DeviceBufferRef onto the
    // list.
    if (zero_sized_buffers.capacity() == 0) {
      // Only allocate once we know there's at least one zero-sized output.
      zero_sized_buffers.reserve(num_outputs);
    }
    ABSL_CHECK(zero_sized);  // CRASH_OK
    DeviceBufferRef zero_sized_buffer =
        // Shouldn't fail because we just checked that the output shape is
        // zero-sized.
        DeviceBufferList::CreateZeroSize(CopyIntVector(output_dim_set),
                                         output_dtypes[i])
            .value();
    zero_sized_buffers.push_back(std::move(zero_sized_buffer));
  }
  // If we made it here, all outputs are zero-sized.
  return SkipIfAllZeroSizedResult{SkipIfAllZeroSizedResult::Skipped{
      .zero_sized_buffers = std::move(zero_sized_buffers)}};
}

// Internal helper for MakeTensor that returns the
// DeviceBufferRef that will get wrapped into the at::Tensor.
absl::StatusOr<DeviceBufferRef> MakeBuffer(
    const at::Scalar& scalar,
    c10::optional<at::ScalarType> scalar_type_opt = std::nullopt) {
  at::ScalarType scalar_type = scalar_type_opt.value_or(scalar.type());
  if (scalar_type == at::ScalarType::ComplexDouble) {
    // TPU does not support complex128. Use complex64 instead.
    scalar_type = at::ScalarType::ComplexFloat;
  }
  if (GetEagerMode() != EagerMode::kInternalDeferAll) {
    // Variable execution mode: materialize the scalar to a DeviceBufferRef.
    // This treats the scalar as an argument rather than a constant, which
    // decreases compiler specialization and improves code reuse.
    auto hashable_scalar = HashableScalar{
        .scalar = scalar,
        .scalar_type = scalar_type,
    };

    // See if we've already created a DeviceBufferRef for this scalar; if we
    // have, don't recreate it.
    //
    // We make this thread_local to avoid synchronization overhead. This means
    // that we may miss some opportunities to reuse scalars across threads,
    // but it might be a net win overall given that scalar buffers are cheap
    // to create.
    static thread_local absl::flat_hash_map<HashableScalar, DeviceBufferRef>
        scalar_map;
    if (auto it = scalar_map.find(hashable_scalar); it != scalar_map.end()) {
      return it->second;
    }

    // New scalar, create a DeviceBufferRef for it.
    // Create a TraceMe so that the host-to-device copy is visible in xprof.
    tsl::profiler::TraceMe t(
        [scalar] { return absl::StrCat("ScalarBuffer_", ToString(scalar)); });

    // Upgrade the scalar to a tensor on CPU, using CPU-optimized code paths.
    at::Tensor scalar_tensor =
        c10::scalar_to_tensor(scalar, /*device=*/at::kCPU);
    // Then cast the tensor to the appropriate type on CPU, if necessary.
    if (scalar_tensor.scalar_type() != scalar_type) {
      scalar_tensor = scalar_tensor.to(scalar_type);
    }

    // Then copy it to device as a materialized DeviceBufferRef.
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef buf_ref,
        CopyCpuToTpuBuffer(scalar_tensor, /*non_blocking=*/true));
    scalar_map.insert({hashable_scalar, buf_ref});
    return buf_ref;
  }

  // Constant execution mode: create an mlir::Constant op to represent the
  // value. This ensures that the compiler is able to specialize for this value
  // and produce an optimized executable, provided this value is consistent
  // on future executions.
  TT_ASSIGN_OR_RETURN(auto scalar_element_type,
                      ConvertTo<mlir::ElementType>(scalar_type));
  Shape shape(Dimensions{}, scalar_element_type);

  TT_ASSIGN_OR_RETURN(auto op_param_cache_keys,
                      TT_MAKE_OP_PARAM_CACHE_KEYS(scalar));
  auto op_builder =
      [scalar, scalar_element_type](
          mlir::MlirBuilder& builder) -> absl::StatusOr<mlir::MlirOp> {
    return MakeConstant(builder, scalar, scalar_element_type);
  };

  return DispatchOp<0>(std::move(op_builder),
                       /*inputs=*/{},
                       // Override the op name as this is a general utility
                       // rather than a specific op.
                       {.op_name = OpName::kScalarTensor,
                        .out_dtype = scalar_element_type,
                        .out_dims = Dimensions{},
                        .op_param_cache_keys = std::move(op_param_cache_keys)});
}

}  // namespace

namespace internal {

// Information about an injected op dispatch failure. Used for internal testing
// only.
struct OpDispatchFailure {
  // The base name of the op that is forced to fail. If empty, no op is forced
  // to fail.
  std::string op_base_name;
  // The failure message of the injected op failure.
  std::string failure_message;
};

// Returns the injected op dispatch failure.
[[nodiscard]] static OpDispatchFailure& GetOpDispatchFailure() {
  static absl::NoDestructor<OpDispatchFailure> failure;
  return *failure;
}

void SetOpDispatchFailure(std::string op_base_name,
                          std::string failure_message) {
  GetOpDispatchFailure() = {std::move(op_base_name),
                            std::move(failure_message)};
}

class OpWindow {
 public:
  OpWindow(size_t min_repeated_sequence_size, size_t max_repeated_sequence_size)
      : min_repeated_sequence_size_(min_repeated_sequence_size),
        // We need a history that is at least 2 times the maximum repeated
        // sequence size we want to detect.
        max_history_size_(2 * max_repeated_sequence_size) {}

  void Append(const DeferredOp& op) {
    // Here we store only a quick hash of the op. At the op level, this will
    // lead to a high probability of collision, however, that probability
    // decreases quickly as we match longer sequences.
    history_.push_back(op.Hash());
    if (history_.size() > max_history_size_) {
      history_.pop_front();
    }
    // Increase the number of ops we have encountered so far.
    pos_++;
    // However, for very long running op sequences, we wrap the counters around
    // to avoid over overflow.
    const auto wrap = 2 * max_history_size_;
    if (pos_ > wrap) {
      pos_ -= wrap;
      if (next_valid_pos_ > wrap) {
        next_valid_pos_ -= wrap;
      } else {
        next_valid_pos_ = pos_;
      }
    }
  }

  // Return true if the last dispatched op is the end of a repeated suffix in
  // the sequence of ops that have been dispatched so far.
  bool FindRepeatedSequence() {
    if (pos_ < next_valid_pos_) {
      return false;
    }

    // Skip if the accumulated ops are not enough.
    auto n = history_.size();
    if (n < 2 * min_repeated_sequence_size_) {
      return false;
    }

    // If last time we found a repeated sequence of size S, chances are we are
    // going to find another one of the same size. Hence, speculatively check
    // that size first, and only later check for repeated sequences of all other
    // sizes.
    if (last_repeated_sequence_size_ > 0 &&
        // Ensure we have sufficient elements in the history to check for the
        // match.
        n >= 2 * last_repeated_sequence_size_) {
      auto i = (n - 1 - last_repeated_sequence_size_);
      if (IsMatch(i)) {
        return true;
      }
    }

    if (ABSL_VLOG_IS_ON(1)) {
      std::ostringstream os;
      for (auto x : history_) {
        os << static_cast<char>(x | 0x20);  // Make the value printable.
      }
      ABSL_VLOG(1) << os.str();
    }

    // Here we perform an exhaustive search for all possible sizes.
    for (auto i = n - 2; i >= n / 2; --i) {
      if (IsMatch(i)) {
        ABSL_VLOG(1) << std::string(i, ' ') << "^";
        return true;
      }
    }
    return false;
  }

 private:
  // Return true if there is a repeated sequence in the history at position i,
  // specifically if history[2i+2-n:i] = history[i+1:n-1].
  bool IsMatch(size_t i) {
    if (history_[i] == history_.back()) {
      auto n = history_.size();
      size_t repeated_sequence_size = (n - i - 1);
      if (repeated_sequence_size > min_repeated_sequence_size_) {
        ABSL_CHECK((i - repeated_sequence_size + 1) >= 0);  // CRASH_OK
        bool has_repeated_sequence =
            std::equal(history_.begin() + (i - repeated_sequence_size + 1),
                       history_.begin() + i + 1,
                       history_.end() - repeated_sequence_size, history_.end());
        if (has_repeated_sequence) {
          ABSL_VLOG(1) << "Found repeated subsequence of length "
                       << repeated_sequence_size << ", pos=" << pos_;
          // Now that we have found a repeated sequence, we need to skip the
          // next search to `repeated_sequence_size` ops ahead.
          next_valid_pos_ = pos_ + repeated_sequence_size;
          last_repeated_sequence_size_ = repeated_sequence_size;
          return true;
        }
      }
    }
    return false;
  }

  size_t min_repeated_sequence_size_;
  size_t max_history_size_;
  std::deque<size_t> history_;
  size_t pos_ = 0;
  size_t next_valid_pos_ = 0;
  size_t last_repeated_sequence_size_ = 0;
};

absl::StatusOr<std::vector<DeviceBufferRef>> DynamicDispatchOp(
    MlirOpBuilder op_builder, std::vector<DeviceBufferRef> inputs,
    DispatchOpOptions<kDynamicSize> options) {
  ABSL_CHECK(options.op_name.has_value())  // CRASH_OK
      << "DynamicDispatchOp() called without op_name in options.";
  const OpName op_name = options.op_name.value();
  ABSL_VLOG(1) << "DispatchOp " << op_name;
  if (ABSL_VLOG_IS_ON(3)) {
    std::stringstream inputs_ss;
    for (int i = 0; i < inputs.size(); ++i) {
      inputs_ss << " input " << i << ": " << ToString(inputs[i].dimensions())
                << " dtype=" << ToString(inputs[i].element_type());
    }
    ABSL_VLOG(3) << inputs_ss.str();
  }

  // Fail if the op is forced to fail for testing.
  const std::string_view op_base_name = ToBaseName(op_name);
  const auto& op_failure = GetOpDispatchFailure();
  if (op_failure.op_base_name == op_base_name) {
    return TT_ERROR(error::kInternal) << op_failure.failure_message;
  }

  const auto& out_dtypes = options.out_dtypes;
  const auto& out_dims_list = options.out_dims_list;
  const int num_outputs = out_dtypes.size();
  ABSL_CHECK_EQ(num_outputs,  // CRASH_OK=indicates invalid kernel
                out_dims_list.size())
      << "Mismatching output sizes, num_outputs = " << num_outputs
      << ", out_dims_list.size() = " << out_dims_list.size();

  // Check if we can take a shortcut; if all outputs are zero-sized, then there
  // is no need to either defer or execute the op, as there will never be any
  // data to collect.
  TT_ASSIGN_OR_RETURN(SkipIfAllZeroSizedResult skip_result,
                      SkipIfAllZeroSized(out_dtypes, out_dims_list));
  if (std::holds_alternative<SkipIfAllZeroSizedResult::Skipped>(
          skip_result.result)) {
    return std::move(
        std::get<SkipIfAllZeroSizedResult::Skipped>(skip_result.result)
            .zero_sized_buffers);
  }

  if (options.computation_dtype) {
    op_builder = [computation_dtype = *options.computation_dtype,
                  op_builder = std::move(op_builder)](
                     mlir::MlirBuilder& builder,
                     absl::Span<mlir::MlirOp> inputs)
        -> absl::StatusOr<DynamicMlirOpResults> {
      mlir::Type computation_type =
          mlir::getElementType(builder.getContext(), computation_dtype);
      std::vector<mlir::MlirOp> casted_inputs;
      casted_inputs.reserve(inputs.size());
      for (auto input : inputs) {
        const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
        if (input_type.getElementType() != computation_type) {
          input = mlir::stablehlo::ConvertElementType(input, computation_dtype);
        }
        casted_inputs.push_back(input);
      }
      return op_builder(builder, absl::MakeSpan(casted_inputs));
    };
  }

  // Always create a deferred node to define the graph; we may or may not
  // then immediately execute it.
  std::vector<Shape> output_shapes;
  output_shapes.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_shapes.emplace_back(CopyIntVector(out_dims_list[i]), out_dtypes[i]);
  }
  TT_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferRef> results,
      DeviceBufferList::CreateDeferred(
          op_name, std::move(op_builder), std::move(inputs),
          std::move(options.op_param_cache_keys), std::move(output_shapes),
          options.split_mode, std::move(options.donated_indices)));

  auto eager_mode = GetEagerMode();
  if ((eager_mode == EagerMode::kDeferNever) ||
      (eager_mode == EagerMode::kDeferNeverAndLaunchBlocking)) {
    auto& device_buffer_list = results[0].device_buffer_list();
    TT_RETURN_IF_ERROR(Materialize(device_buffer_list));
    if (eager_mode == EagerMode::kDeferNeverAndLaunchBlocking) {
      TT_RETURN_IF_ERROR(device_buffer_list->Synchronize());
    }

  } else if (eager_mode != EagerMode::kInternalDeferAll) {
    if (absl::GetFlag(FLAGS_torch_tpu_internal_enable_new_materialization)) {
      TT_RETURN_IF_ERROR(OnNewOpDispatch(results[0].device_buffer_list()));
    }

    const std::string& detect_repeated_ops =
        absl::GetFlag(FLAGS_torch_tpu_internal_detect_repeated_ops);
    if (!detect_repeated_ops.empty()) {
      // Note that view operations (like reshapes and transposes) don't go
      // through the op_dispatcher sequence. So the heuristic below considers
      // only non-view ops.
      //
      // We make this thread_local to avoid op sequences from different threads
      // to interfere with each other.
      static thread_local OpWindow op_window(kMinRepeatedSubsequenceLength,
                                             kMaxRepeatedSubsequenceLength);
      const DeferredOp* absl_nullable op = results[0].deferred_op();
      ABSL_CHECK(op != nullptr);  // CRASH_OK
      op_window.Append(*op);
      if (op_window.FindRepeatedSequence()) {
        auto materialization_mode = (detect_repeated_ops == "aggressive")
                                        ? MaterializationMode::kFullGraph
                                        : MaterializationMode::kSplitGraph;
        TT_RETURN_IF_ERROR(
            Materialize(results[0].device_buffer_list(), materialization_mode));
      }
    }
  }

  return results;
}

std::string FormatParamCacheKey(const std::optional<PromotedScalar>& value) {
  return value.has_value() ? "s" : "";
}

}  // namespace internal

absl::StatusOr<at::Tensor> MakeTensor(
    const at::Scalar& scalar, c10::optional<at::ScalarType> scalar_type_opt) {
  TT_ASSIGN_OR_RETURN(  // TODO: Test by forcing "scalar_tensor" to fail.
      DeviceBufferRef buffer, MakeBuffer(scalar, scalar_type_opt));
  return MakeTensor(std::move(buffer));
}

PromotedScalar PromoteScalar(at::Scalar scalar) {
  return PromotedScalar(
      [](const at::Scalar& scalar,
         std::optional<at::ScalarType> scalar_type_opt)
          -> absl::StatusOr<at::Tensor> {
        TT_ASSIGN_OR_RETURN(DeviceBufferRef buffer,
                            MakeBuffer(scalar, scalar_type_opt));
        return MakeTensor(std::move(buffer));
      },
      std::move(scalar));
}

std::optional<PromotedScalar> PromoteScalar(std::optional<at::Scalar> scalar) {
  if (!scalar.has_value()) {
    return std::nullopt;
  }
  return PromoteScalar(scalar.value());
}

std::vector<PromotedScalar> PromoteScalar(at::ArrayRef<at::Scalar> scalars) {
  std::vector<PromotedScalar> promoted_scalars;
  promoted_scalars.reserve(scalars.size());
  for (const auto& scalar : scalars) {
    promoted_scalars.push_back(PromoteScalar(scalar));
  }
  return promoted_scalars;
}

}  // namespace torch_tpu
