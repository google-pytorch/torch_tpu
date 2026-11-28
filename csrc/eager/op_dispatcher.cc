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

#include "csrc/eager/op_dispatcher.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/ScalarOps.h"
#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "c10/util/Optional.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/device_type.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/dtype.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/common/shape.h"
#include "csrc/common/to_string.h"
#include "csrc/common/utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/eager_mode.h"
#include "csrc/eager/events_queue.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/resize/resize_aten_kernels.h"
#include "csrc/ops/stride/stride_helper.h"
#include "csrc/pjrt/pjrt_transfer.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/headeronly/core/DeviceType.h"
#include "torch/headeronly/core/ScalarType.h"
#include "tsl/profiler/lib/traceme.h"

namespace torch_tpu {

namespace {

// Number of scalar device buffer misses after which MakeBuffer's per-thread
// cache turns over a generation; see the cache in MakeBuffer.
constexpr size_t kMaxCachedScalarBuffers = 1024;

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

  // In compiled mode, variable scalar parameters are identified as FX graph
  // inputs. All non-variable scalars are therefore constants and can be
  // specialized to the output program to achieve constant folding.
  // In eager modes (DeferNever, DeferAndFuse, or DeferAll) scalars are treated
  // as arguments to increase code reuse, as the benefit of constant folding
  // is not justified at the risk of over-specialization and recompilation.
  const bool constant_mode =
      GetEagerMode() == EagerMode::kInternalCompileFxGraph;

  if (!constant_mode) {
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
    //
    // RATIONALE FOR TWO GENERATIONS:
    // A single-generation cache with a hard limit (e.g., 1024 entries) causes a
    // performance cliff: when the working set exceeds the limit, every miss
    // triggers a full cache clear. During that clear, scalars used before and
    // after the boundary end up on the graph input AND in the cache, breaking
    // deduplication and forcing recompilation (#6037). Two generations solve
    // this: scalars always survive at least 1024 misses after their last use,
    // so a working set up to 2048 distinct values avoids the cliff, stale values
    // are still freed (after two generations), and the common hit path is a
    // single lookup in the fresh map.
    //
    // IMPLEMENTATION: A new scalar is `fresh`; once `fresh` has received
    // kMaxCachedScalarBuffers misses, its entries become `stale` and the
    // previous `stale` entries are dropped. A hit in `stale` moves the scalar
    // back to `fresh`. The two maps together hold at most 2*kMaxCachedScalarBuffers
    // entries. Pending ops hold their own references to the buffers they use,
    // so dropping a generation cannot free a buffer in flight.
    static thread_local  // CPP_THREAD_LOCAL_OK=unrelated to Python threads.
        absl::flat_hash_map<HashableScalar, DeviceBufferRef>
            fresh_scalars;
    static thread_local  // CPP_THREAD_LOCAL_OK=unrelated to Python threads.
        absl::flat_hash_map<HashableScalar, DeviceBufferRef>
            stale_scalars;
    if (auto it = fresh_scalars.find(hashable_scalar);
        it != fresh_scalars.end()) {
      return it->second;
    }
    if (auto it = stale_scalars.find(hashable_scalar);
        it != stale_scalars.end()) {
      DeviceBufferRef buf_ref = std::move(it->second);
      stale_scalars.erase(it);
      fresh_scalars.insert({hashable_scalar, buf_ref});
      return buf_ref;
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

    // Copy from the tensor to a bytes vector to enable async transfer.
    std::vector<uint8_t> scalar_bytes(scalar_tensor.storage().nbytes());
    std::memcpy(scalar_bytes.data(), scalar_tensor.data_ptr(),
                scalar_tensor.storage().nbytes());

    // Then asynchronously copy it to device as a materialized DeviceBufferRef.
    TT_ASSIGN_OR_RETURN(mlir::ElementType element_type,
                        ConvertTo<mlir::ElementType>(scalar_type));
    auto transfer = HostToDeviceTransfer::FromBytesVector(
        std::move(scalar_bytes), element_type,
        /*dimensions=*/{});
    TT_ASSIGN_OR_RETURN(DeviceBufferRef buf_ref,
                        ScheduleHostToDeviceTransfer(std::move(transfer)));
    // Every cached scalar keeps its device buffer alive, so a stream of
    // distinct values would grow device memory without bound. When the fresh
    // generation is full, it becomes the stale one and the stale one is
    // dropped.
    if (fresh_scalars.size() >= kMaxCachedScalarBuffers) {
      stale_scalars = std::move(fresh_scalars);
      fresh_scalars.clear();
    }
    fresh_scalars.insert({hashable_scalar, buf_ref});
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

// How the base buffer of an output tensor can be donated by DispatchOpOut.
//
// The base buffer of `out` can be donated whenever an op of the dispatch both
// reads it and produces its replacement:
//   1. A full overwrite that reads its base (`x.add_(1.0)`): the op reads the
//      base through an input that aliases `out`, and `out` is replaced by the
//      op's result.
//   2. A partial overwrite that reads its base (`x[0].add_(1.0)`): the op reads
//      the base through an input view, and the view-inversion write-back reads
//      it again to produce the updated base.
//   3. A partial overwrite that doesn't read its base (`x[0].copy_(y)`): the op
//      doesn't read the base, but the write-back still does.
//   4. A full overwrite that doesn't read its base (`x.copy_(y)`): nothing
//      reads the base, so it is simply replaced and there's nothing to donate.
enum class BaseDonation {
  // The base buffer can't be donated.
  kNone,
  // The base can only be donated to the op, through an input that aliases
  // `out` with the same shape and dtype (cases 1 and 2).
  kToOp,
  // `out` is a partial view of its base, so the base can also be donated to
  // the write-back when no input matches (cases 2 and 3).
  kToOpOrWriteBack,
};

// Converts the on-device `dims` of a dispatch output of `dtype` (see
// DispatchOpOptions) to PyTorch sizes: the packed dimension of a packed
// sub-byte dtype (e.g. `float4_e2m1fn_x2`) holds `TorchPackingFactor` values
// per PyTorch element. Returns std::nullopt if `dims` can't be converted.
std::optional<Dimensions> ToTorchSizes(absl::Span<const int64_t> dims,
                                       mlir::ElementType dtype) {
  Dimensions sizes = CopyIntVector(dims);
  const int64_t packing_factor = TorchPackingFactor(dtype);
  if (packing_factor > 1) {
    if (sizes.empty() || sizes.back() % packing_factor != 0) {
      return std::nullopt;
    }
    sizes.back() /= packing_factor;
  }
  return sizes;
}

// Returns true if the dispatch writes `out` in place: `out` is non-empty and
// the dispatch produces its shape and dtype (otherwise `out` is resized or
// reallocated). Packed sub-byte outputs (e.g. `float4_e2m1fn_x2`) expand their
// packed dimension on device, so they must also be contiguous with no storage
// offset.
bool IsWrittenInPlace(const at::Tensor& out, mlir::ElementType out_dtype,
                      at::IntArrayRef destination_dims) {
  if (out.numel() == 0 || out.sizes() != destination_dims) {
    return false;
  }
  const auto current_out_dtype =
      ConvertTo<mlir::ElementType>(out.scalar_type());
  if (!current_out_dtype.ok() || *current_out_dtype != out_dtype) {
    return false;
  }
  return !IsPackedSubByteType(out_dtype) ||
         (out.is_contiguous() && out.storage_offset() == 0 && out.dim() > 0 &&
          out.stride(out.dim() - 1) == 1);
}

// Returns true if a write of `out_dtype` may update `base_buf` in place.
bool IsBaseWritableInPlace(const DeviceBufferRef& base_buf,
                           mlir::ElementType out_dtype) {
  // Because XLA tensors are immutable, TorchTPU uses copy-on-write semantics.
  // This means that if a tensor is cloned, we construct a new `c10::Storage`
  // but retain the same underlying device buffer.
  // If there is later an in-place write while multiple references exist, we
  // perform the operation out-of-place, as a copy is required.
  // If there is only one reference, then it is safe to donate the buffer.
  if (base_buf.live_data_ptrs() > 1) {
    return false;
  }
  // Writing a non-boolean value to a boolean tensor is allowed, but requires
  // upcasting the storage from single-bit PRED to avoid data loss. As such
  // this is performed out-of-place.
  return base_buf.element_type() != mlir::ElementType::PRED ||
         out_dtype == mlir::ElementType::PRED;
}

// Returns true if the packed sub-byte output `out` covers all of `base_buf`.
bool CoversPackedBase(const at::Tensor& out, const DeviceBufferRef& base_buf,
                      int64_t packing_factor) {
  return !base_buf.dimensions().empty() &&
         out.size(out.dim() - 1) * packing_factor ==
             base_buf.dimensions().back() &&
         out.nbytes() * packing_factor == base_buf.size_bytes();
}

// Returns true if writing `out` reads `base_buf` in a view-inversion
// write-back: `out` covers only part of `base_buf`, as a full write replaces
// (or permutes) the result directly (see AssignBufferToAtTensor). Packed
// sub-byte bases and overlapping views (which CreateInverseViewDeviceBufferRef
// rejects) are not written back.
bool IsWrittenBackIntoBase(const at::Tensor& out, mlir::ElementType out_dtype,
                           const DeviceBufferRef& base_buf) {
  if (IsPackedSubByteType(base_buf.element_type())) {
    return false;
  }
  const absl::StatusOr<TpuLayout> tpu_layout = ResolveTpuLayout(out);
  if (!tpu_layout.ok() ||
      IsOverlapping(tpu_layout->sizes, tpu_layout->strides)) {
    return false;
  }
  const auto out_bits =
      SafeMultiply(out.numel(), TorchEquivalentBitwidth(out_dtype));
  const auto base_bits =
      SafeMultiply(base_buf.num_elements(),
                   TorchEquivalentBitwidth(base_buf.element_type()));
  return out_bits.ok() && base_bits.ok() && *out_bits < *base_bits;
}

// Returns how the base buffer of `out` can be donated, if the dispatch
// produces a result of `out_dtype` and shape `destination_dims` for it.
BaseDonation GetBaseDonation(const at::Tensor& out, mlir::ElementType out_dtype,
                             at::IntArrayRef destination_dims) {
  if (!IsWrittenInPlace(out, out_dtype, destination_dims)) {
    return BaseDonation::kNone;
  }
  // Note: We explicitly guard the base buffer checks with
  // `GetPrivateUse1DeviceType()` because standalone C++ unit tests (e.g. in
  // op_dispatcher_test) test ATen layout, contiguity, and offset properties
  // using CPU tensors. In standalone C++ test binaries without Python runtime
  // extensions, ATen view kernels like `aten::as_strided` are not registered
  // for `PrivateUse1`. Guarding with `GetPrivateUse1DeviceType()` allows those
  // C++ unit tests to run on CPU tensors without `GetBaseBuffer` failing, while
  // strictly enforcing exclusive ownership for real TPU tensors in production.
  if (out.device().type() != GetPrivateUse1DeviceType()) {
    return BaseDonation::kToOp;
  }
  const absl::StatusOr<DeviceBufferRef> base_buf = GetBaseBuffer(out);
  if (!base_buf.ok() || !IsBaseWritableInPlace(*base_buf, out_dtype)) {
    return BaseDonation::kNone;
  }
  const int64_t packing_factor = TorchPackingFactor(out_dtype);
  if (packing_factor > 1) {
    return CoversPackedBase(out, *base_buf, packing_factor)
               ? BaseDonation::kToOp
               : BaseDonation::kNone;
  }
  // A base that is still an unmaterialized constant (e.g. from `torch.empty()`)
  // has no device buffer to reuse. DeferNever deliberately leaves such a
  // tensor and its (metadata-only) write-back deferred, to preserve constant
  // folding, so don't donate it to the write-back, which would materialize it.
  if (base_buf->is_constant() && base_buf->is_deferred()) {
    return BaseDonation::kToOp;
  }
  return IsWrittenBackIntoBase(out, out_dtype, *base_buf)
             ? BaseDonation::kToOpOrWriteBack
             : BaseDonation::kToOp;
}

// Returns true if input `in` can donate the base buffer of `out` to the op: it
// must alias `out` with the same shape and dtype (an in-place operation). The
// base buffer itself must already be donatable (see GetBaseDonation).
bool IsMatchingInputDonor(const at::Tensor& in, const at::Tensor& out,
                          mlir::ElementType out_dtype,
                          at::IntArrayRef destination_dims) {
  if (!in.defined() || !out.is_alias_of(in) || in.sizes() != destination_dims) {
    return false;
  }
  const auto in_dtype = ConvertTo<mlir::ElementType>(in.scalar_type());
  if (!in_dtype.ok() || *in_dtype != out_dtype) {
    return false;
  }
  // Non-contiguous or sliced packed sub-byte views (e.g. `float4_e2m1fn_x2`)
  // cannot safely donate their underlying packed storage buffer in-place.
  return !IsPackedSubByteType(*in_dtype) ||
         (in.is_contiguous() && in.storage_offset() == 0 &&
          in.stride(in.dim() - 1) == 1);
}

// Marks input `donated_idx` and all inputs that alias it as donated, since a
// buffer can be donated only once. Undefined inputs (e.g. omitted optional
// tensors) alias nothing.
void MarkDonatedAndAliases(size_t donated_idx,
                           absl::Span<const at::Tensor> inputs,
                           std::vector<bool>& input_donated) {
  const at::Tensor& donated_in = inputs[donated_idx];
  for (size_t k = 0; k < inputs.size(); ++k) {
    if (inputs[k].defined() && inputs[k].is_alias_of(donated_in)) {
      input_donated[k] = true;
    }
  }
}

// Returns the index of the first input not yet donated that can donate the
// base buffer of `out` to the op (see IsMatchingInputDonor), if any.
std::optional<size_t> FindMatchingInputDonor(
    const at::Tensor& out, mlir::ElementType out_dtype,
    at::IntArrayRef destination_dims, absl::Span<const at::Tensor> inputs,
    const std::vector<bool>& input_donated) {
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    if (!input_donated[in_idx] &&
        IsMatchingInputDonor(inputs[in_idx], out, out_dtype,
                             destination_dims)) {
      return in_idx;
    }
  }
  return std::nullopt;
}

// Marks the inputs not yet donated that alias `out` as donated, so that
// GetBuffers creates their views with DonateBase::kDonate, fusing the view
// reads, the op, and the write-back of `out` into one HLO module.
void MarkInputsAliasingOutputDonated(const at::Tensor& out,
                                     absl::Span<const at::Tensor> inputs,
                                     std::vector<bool>& input_donated,
                                     Indices& donated_indices) {
  for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
    if (!input_donated[in_idx] && inputs[in_idx].defined() &&
        inputs[in_idx].is_alias_of(out)) {
      donated_indices.push_back(static_cast<int64_t>(in_idx));
      input_donated[in_idx] = true;
    }
  }
}

}  // namespace

namespace internal {

[[nodiscard]] OpDispatchFailure& GetOpDispatchFailure() {
  static absl::NoDestructor<OpDispatchFailure> failure;
  return *failure;
}

void SetOpDispatchFailure(std::string op_base_name,
                          std::string failure_message) {
  GetOpDispatchFailure() = {std::move(op_base_name),
                            std::move(failure_message)};
}

std::string EncodeParamCacheKey(const std::optional<PromotedScalar>& value) {
  return value.has_value() ? "s" : "";
}

// Automatically detects and configures buffer donations from `inputs` into
// multiple `outputs` in eager mode.
//
// Donation is only attempted if enabled via IsInplaceBufferDonationEnabled()
// and in eager modes that support in-place buffer donation (kDeferNever,
// kDeferNeverAndLaunchBlocking, kDeferAndFuse, and kInternalDeferAll).
//
// For every output, its base buffer may only be donated if (GetBaseDonation):
// 1. Shape and data type: the output is non-empty and the op produces the
//    output's shape and element type, so the output is updated in place
//    rather than resized or reallocated.
// 2. Unique ownership: the base buffer is not shared with other non-alias
//    tensors (live_data_ptrs == 1), preventing donation from destroying
//    buffers shared via shallow-copy clone semantics.
// 3. No upcast: a non-boolean write into a PRED base requires an out-of-place
//    upcast to UI8.
// 4. Packed sub-byte types (e.g. `float4_e2m1fn_x2`): the output is contiguous
//    and covers its whole storage buffer.
//
// The base buffer is then donated in one of two ways:
// - To the op, through an input donor (IsMatchingInputDonor), which must:
//   a. alias the output (is_alias_of),
//   b. have the output's shape and element type, and
//   c. for packed sub-byte types, be contiguous with no storage offset.
//   The donor may be a view (e.g. a slice or a transpose), in which case its
//   base buffer is donated to the ops that read the view and write the result
//   back to the base (e.g. `x.add_(1.0)` or `x[0].add_(1.0)`).
// - Otherwise, if the output is a partial, non-overlapping view of its base
//   and no op split separates the op from the write-back, to the
//   view-inversion write-back, which reads the base (e.g. `x[0].copy_(y)` or
//   `torch.add(a, b, out=x[0])`). Inputs that alias the output are marked as
//   donated too, so their view reads are fused into the same HLO module.
//
// Single donation: in XLA / StableHLO, each device buffer can be donated to at
// most ONE output buffer. Once an input is donated, it and all its storage
// aliases (is_alias_of) are marked as donated to prevent any duplicate buffer
// donation across multiple outputs.
bool AutoDonateInPlaceBuffers(
    absl::Span<const at::Tensor> outputs,
    absl::Span<const mlir::ElementType> out_dtypes,
    absl::Span<const absl::Span<const int64_t>> out_dims_list,
    absl::Span<const at::Tensor> inputs, OpSplitMode split_mode,
    Indices& donated_indices, std::vector<uint8_t>& output_donated) {
  if (!donated_indices.empty() || !IsInplaceBufferDonationEnabled() ||
      !SupportsInplaceBufferDonation(GetEagerMode())) {
    return false;
  }

  std::vector<bool> input_donated(inputs.size(), false);
  for (size_t out_idx = 0; out_idx < outputs.size(); ++out_idx) {
    const at::Tensor& out = outputs[out_idx];
    if (!out.defined()) {
      continue;
    }
    const auto out_dtype = out_dtypes[out_idx];
    const std::optional<Dimensions> torch_dims =
        ToTorchSizes(out_dims_list[out_idx], out_dtype);
    if (!torch_dims.has_value()) {
      continue;
    }
    const at::IntArrayRef destination_dims(*torch_dims);
    const BaseDonation base_donation =
        GetBaseDonation(out, out_dtype, destination_dims);
    if (base_donation == BaseDonation::kNone) {
      continue;
    }

    // Donate the base to the op through a matching input (cases 1 and 2).
    if (const std::optional<size_t> in_idx = FindMatchingInputDonor(
            out, out_dtype, destination_dims, inputs, input_donated);
        in_idx.has_value()) {
      donated_indices.push_back(static_cast<int64_t>(*in_idx));
      output_donated[out_idx] = true;
      MarkDonatedAndAliases(*in_idx, inputs, input_donated);
      continue;
    }

    // Otherwise, donate the base of a partial view to the write-back (cases 2
    // and 3), along with inputs that alias the output (e.g.
    // `x[4:12].copy_(x[0:8])`).
    if (base_donation == BaseDonation::kToOpOrWriteBack &&
        split_mode == OpSplitMode::kNone) {
      output_donated[out_idx] = true;
      MarkInputsAliasingOutputDonated(out, inputs, input_donated,
                                      donated_indices);
    }
  }
  return !donated_indices.empty() ||
         absl::c_any_of(output_donated,
                        [](uint8_t donated) { return donated != 0; });
}

bool CanDonateBaseToWriteBack(const at::Tensor& out,
                              mlir::ElementType out_dtype) {
  return out.defined() && IsInplaceBufferDonationEnabled() &&
         SupportsInplaceBufferDonation(GetEagerMode()) &&
         GetBaseDonation(out, out_dtype, out.sizes()) ==
             BaseDonation::kToOpOrWriteBack;
}

absl::Status AssignBufferToOutput(const at::Tensor& output,
                                  DeviceBufferRef result_buffer,
                                  bool enable_autodonation) {
  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Output tensors are always defined.
      output.defined(), error::kInvalidArgument)
      << "expected defined output tensor";
  Dimensions logical_dims = CopyIntVector(result_buffer.dimensions());
  const int64_t packing_factor =
      TorchPackingFactor(result_buffer.element_type());
  if (packing_factor > 1) {
    ABSL_CHECK(!logical_dims.empty());                      // CRASH_OK
    ABSL_CHECK_EQ(logical_dims.back() % packing_factor, 0)  // CRASH_OK
        << "Last dimension of packed sub-byte tensor ("
        << ToString(result_buffer.element_type())
        << ") must be divisible by packing factor " << packing_factor
        << ", got " << logical_dims.back();
    // A packed dimension of exactly `packing_factor` becomes a trailing
    // dimension of size 1 rather than being dropped: PyTorch also stores it as
    // one container element (e.g. a `float4_e2m1fn_x2` tensor of shape
    // `[..., 1]` holds a device buffer of shape `[..., 2]`, see MakeTensor), so
    // an output of the same shape is not resized.
    logical_dims.back() /= packing_factor;
  }
  TT_RETURN_IF_ERROR(ResizeTensorIfShapeDiffers(output, logical_dims));
  return AssignBufferToAtTensor(std::move(result_buffer), output,
                                enable_autodonation);
}

absl::Status AssignBuffersToOutputs(
    absl::Span<const at::Tensor> outputs,
    absl::Span<DeviceBufferRef> result_buffers,
    absl::Span<const uint8_t> enable_autodonation) {
  ABSL_CHECK_EQ(outputs.size(), result_buffers.size())  // CRASH_OK
      << "Mismatch between number of output tensors and result buffers.";
  for (size_t i = 0; i < outputs.size(); ++i) {
    TT_RETURN_IF_ERROR(AssignBufferToOutput(
        outputs[i], std::move(result_buffers[i]), enable_autodonation[i]));
  }
  return absl::OkStatus();
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
