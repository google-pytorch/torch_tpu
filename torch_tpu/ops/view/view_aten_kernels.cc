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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ATen/TensorUtils.h"
#include "ATen/core/ATen_fwd.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "c10/core/DispatchKey.h"
#include "c10/core/DispatchKeySet.h"
#include "c10/core/ScalarType.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/core/TensorImpl.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Optional.h"
#include "c10/util/intrusive_ptr.h"
#include "c10/util/typeid.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/aten_utils.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/to_string.h"
#include "torch_tpu/common/utils.h"
#include "torch_tpu/ops/as_strided/as_strided_aten_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_names.h"

// Both _reshape_alias() and view() require kernel registrations according to
// https://docs.pytorch.org/docs/main/accelerator/operators.html
//
// view() is expected to:
//   1. Check that the requested size is view-compatible with the input shape.
//   2. If it is, then create an aliasing view of the c10::Storage with the
//      requested shape; otherwise, raise an exception
// This is required because torch assumes that devices may have specific
// requirements for view compatibility.
//
// torch_tpu does not have any specific view compatibility requirements, so
// we simply use PyTorch's view stride computation logic and forward the result
// to AtenAsStrided().
//
// _reshape_alias() is expected to be identical to view(), except that it
// doesn't need to check for view compatibility; it is not a user-facing API,
// and PyTorch guarantees it will only be called when view compatibility has
// already been verified.
//
// Some other device implementations (for example, Vulkan device in
// aten/src/ATen/native/vulkan/ops/Shape.cpp) simply reuse the implementation
// of view() for _reshape_alias(), which redundantly applies compatibility
// checks. For efficiency, we do not do this, and skip the compatibility check
// in _reshape_alias().

namespace torch_tpu {

namespace {

// Convenience function for building an error message with the same format.
//
// This function wraps a few other functions for creating error messages of the
// same form. In summary, it applies the following steps:
//
//   1. Filter the indices [0, size) using `pred`
//   2. Uses the filtered indices to build the "got" part of the error message
//     - Appends values followed with their respective index
//
// Example:
//   arr = [0, 2, 5, 1]
//
//   size = 4
//   singular = "bad element"
//   plural = "bad elements"
//   pred = lambda i: arr[i] > 1
//   to_value = lambda i: arr[i]
//
//   output = "2 bad elements: 2 at index 1 and 5 at index 3"
//   error = "expected ..., got 2 bad elements: 2 at index 1 and 5 at index 3"
template <typename Predicate, typename ToValue>
std::string FilterIndicesAndBuildErrorStr(const size_t size,
                                          const std::string_view singular,
                                          const std::string_view plural,
                                          const Predicate& pred,
                                          const ToValue& to_value) {
  auto bad_indices = FilterIndices(size, pred);
  return absl::StrCat(FormatCount(bad_indices.size(), singular, plural), ": ",
                      GetValueAtIndexErrorStr(bad_indices, to_value));
}

std::string GetInvalidDimensionIndexStr(absl::Span<const int64_t> sizes) {
  return FilterIndicesAndBuildErrorStr(
      sizes.size(),
      /* singular= */ "invalid size",
      /* plural= */ "invalid sizes",
      /* pred= */ [sizes](const int64_t i) { return sizes[i] < -1; },
      /* to_value= */ [sizes](const int64_t i) { return sizes[i]; });
}

std::string DropLastAndGetOddStridesStr(absl::Span<const int64_t> strides) {
  // Drop last element.
  strides.remove_suffix(1);

  return FilterIndicesAndBuildErrorStr(
      strides.size(),
      /* singular= */ "odd stride",
      /* plural= */ "odd strides",
      /* pred= */ [strides](const int64_t i) { return strides[i] % 2 != 0; },
      /* to_value= */ [strides](const int64_t i) { return strides[i]; });
}

// Returns a string of the comma-separated indices corresponding to -1 elements.
std::string GetNegativeOneDimensionsIndexStr(absl::Span<const int64_t> sizes) {
  const auto negative_one_indices = FilterIndices(
      sizes.size(),
      /* predicate= */ [sizes](const int64_t i) { return sizes[i] == -1; });

  return SequenceToReadableStr(
      absl::MakeConstSpan(negative_one_indices),
      /* value_to_string= */ [](const int64_t i) { return ToString(i); });
}

// torch passes the target size for view (and reshape) as SymInts.
// SymInts are used in torch's tracing logic; they record any writes against
// them, and then c10::guard_int is used to resolve them to a concrete, readable
// value. We need to read them here to dispatch MLIR ops with correct shapes.
// Additionally, torch allows one dimension to be specified as -1, which means
// "infer", by keeping the total number of elements the same between input
// and output.
absl::StatusOr<Dimensions> ViewSymIntsToDimensions(
    const at::Tensor& self, c10::SymIntArrayRef size_sym) {
  Dimensions out_sizes(size_sym.size());
  absl::c_transform(size_sym, out_sizes.begin(), [](const c10::SymInt& sym) {
    return sym.guard_int(__FILE__, __LINE__);
  });

  TT_RET_CHECK(
      absl::c_all_of(out_sizes, [](const int64_t size) { return size >= -1; }),
      error::kInvalidArgument)
      << "expected the given sizes " << ToString(out_sizes)
      << " to be >= -1, got " << GetInvalidDimensionIndexStr(out_sizes);

  const int64_t negative_one_count = absl::c_count(out_sizes, -1);
  TT_RET_CHECK(negative_one_count <= 1, error::kInvalidArgument)
      << "expected the given sizes " << ToString(out_sizes)
      << " to have up to 1 element equal to -1 (inferred dimension), got "
      << negative_one_count << " occurrences of -1 at indices "
      << GetNegativeOneDimensionsIndexStr(out_sizes);

  // Compute the number of elements in the output.
  int64_t out_numel = 1;

  for (const auto size : out_sizes) {
    if (size == -1) {
      continue;
    }

    TT_ASSIGN_OR_RETURN(
        out_numel, SafeMultiply(out_numel, size),
        _.SetOverride() << "the number of elements in the output view of shape "
                        << ToString(out_sizes)
                        << " cannot overflow a 64-bit integer");
  }

  const int64_t numel = self.sym_numel().guard_int(__FILE__, __LINE__);

  // Infer the -1 dimension if there is one.
  const auto negative_one_dimension_it = absl::c_find(out_sizes, -1);
  if (negative_one_dimension_it != out_sizes.end()) {
    TT_RET_CHECK(out_numel != 0, error::kInvalidArgument)
        << "cannot infer the dimension for a 0-element view of shape "
        << ToString(out_sizes)
        << " because it's ambiguous, i.e. it could be of any value";
    TT_RET_CHECK(numel % out_numel == 0, error::kInvalidArgument)
        << "expected the number of elements in the output view of shape "
        << ToString(out_sizes)
        << " to be a multiple of the number of elements in the input of shape "
        << ToString(self.sizes())
        << " in the presence of an inferred dimension (-1), got " << out_numel
        << ", which is not a multiple of " << numel;

    const int64_t negative_one_dimension =
        std::distance(out_sizes.begin(), negative_one_dimension_it);
    const int64_t inferred_dim_size = numel / out_numel;
    out_sizes[negative_one_dimension] = inferred_dim_size;

    TT_ASSIGN_OR_RETURN(
        out_numel, SafeMultiply(out_numel, inferred_dim_size),
        _.SetOverride()
            << "cannot infer the dimension for a view of shape "
            << ToString(out_sizes)
            << " because the number of elements would overflow a 64-bit "
               "integer");
  }

  TT_RET_CHECK(numel == out_numel, error::kInvalidArgument)
      << "expected the input of shape " << ToString(self.sizes())
      << " to have the same number of elements as the output of shape "
      << ToString(out_sizes) << ", got " << numel << " vs. " << out_numel;

  return out_sizes;
}

}  // namespace

at::Tensor AtenReshapeAlias(const at::Tensor& self,
                            c10::SymIntArrayRef size_sym,
                            c10::SymIntArrayRef stride_sym) {
  TT_KERNEL(
      OpName::kReshapeAlias, _,
      (self, IgnoreInCacheKey(size_sym, "Legacy usage"),
       IgnoreInCacheKey(stride_sym, "Legacy usage")),
      { return AtenAsStrided(self, size_sym, stride_sym, c10::nullopt); });
}

at::Tensor AtenView(const at::Tensor& self, c10::SymIntArrayRef size_sym) {
  TT_KERNEL(
      OpName::kView, _, (self, IgnoreInCacheKey(size_sym, "Legacy usage")), {
        // view is allowed to have one dimension as "-1", which needs to be
        // resolved to a positive shape.
        TT_ASSIGN_OR_THROW(Dimensions new_size,
                           ViewSymIntsToDimensions(self, size_sym));

        // Use PyTorch's default logic for view stride computation.
        std::optional<std::vector<int64_t>> new_stride =  // INT_VEC_OK=c10 API
            at::detail::computeStride(self.sizes(), self.strides(), new_size);
        TT_CHECK_THROW(new_stride.has_value(), error::kInvalidArgument)
            << "cannot create a view of shape " << ToString(new_size)
            << " from the input tensor of shape " << ToString(self.sizes())
            << " and strides " << ToString(self.strides())
            << "; consider creating a new tensor using reshape() instead of "
               "taking "
               "a view";

        // Redispatch to AtenAsStrided() with the resolved shape and strides.
        c10::SymIntArrayRef new_size_inferred_sym =
            c10::fromIntArrayRefKnownNonNegative(new_size);
        c10::SymIntArrayRef new_stride_sym =
            c10::fromIntArrayRefKnownNonNegative(*new_stride);
        at::Tensor tensor = AtenAsStrided(self, new_size_inferred_sym,
                                          new_stride_sym, c10::nullopt);
        if (self.is_conj() && !tensor.is_conj()) {
          tensor = tensor.conj();
        }
        return tensor;
      });
}

at::Tensor AtenViewAsReal(const at::Tensor& self) {
  TT_KERNEL(OpName::kViewAsReal, _, (self), {
    TT_CHECK_THROW(IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype to be complex, got "
        << ToString(self.scalar_type());

    Dimensions output_dims = CopyIntVector(self.sizes());
    output_dims.push_back(2);
    Strides output_strides;
    output_strides.reserve(self.dim() + 1);
    for (auto stride : self.strides()) {
      output_strides.push_back(stride * 2);
    }
    output_strides.push_back(1);
    const int64_t output_storage_offset = self.storage_offset() * 2;
    at::ScalarType output_dtype = c10::toRealValueType(self.scalar_type());
    auto output_type_meta = caffe2::TypeMeta::fromScalarType(output_dtype);

    c10::Storage storage_copy = self.storage();
    at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
        std::move(storage_copy),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), output_type_meta));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        output_dims, output_strides, output_storage_offset);
    return tensor;
  });
}

at::Tensor AtenViewAsComplex(const at::Tensor& self) {
  TT_KERNEL(OpName::kViewAsComplex, _, (self), {
    TT_CHECK_THROW(!IsHalf(self), error::kInvalidArgument)
        << "float16 dtype is not yet supported";

    TT_CHECK_THROW(IsFloatOrDouble(self), error::kInvalidArgument)
        << "expected the input dtype to be float32 or float64, got "
        << ToString(self.scalar_type());

    TT_CHECK_THROW(self.dim() > 0, error::kInvalidArgument)
        << "expected the input to be a tensor, got a scalar";

    TT_CHECK_THROW(self.size(-1) == 2, error::kInvalidArgument)
        << "expected the size of the last dimension of the input tensor to be "
           "2, got "
        << self.size(-1);

    TT_CHECK_THROW(self.stride(-1) == 1, error::kInvalidArgument)
        << "expected the stride of the last dimension of the input tensor to "
           "be 1, got "
        << self.stride(-1);

    const absl::Span<const int64_t> strides = self.strides();

    TT_CHECK_THROW(
        absl::c_all_of(strides.first(strides.size() - 1),
                       [](int64_t stride) { return stride % 2 == 0; }),
        error::kInvalidArgument)
        << "expected the input strides " << ToString(strides)
        << " to be even numbers (except in the last dimension), got "
        << DropLastAndGetOddStridesStr(strides);

    Dimensions output_dims = CopyIntVector(self.sizes());
    output_dims.pop_back();
    Strides output_strides;
    output_strides.reserve(self.dim() - 1);
    for (auto i = 0; i < self.dim() - 1; ++i) {
      output_strides.push_back(strides[i] / 2);
    }
    const int64_t output_storage_offset = self.storage_offset() / 2;
    at::ScalarType output_dtype = c10::toComplexType(self.scalar_type());
    auto output_type_meta = caffe2::TypeMeta::fromScalarType(output_dtype);

    c10::Storage storage_copy = self.storage();
    at::Tensor tensor(c10::make_intrusive<c10::TensorImpl>(
        std::move(storage_copy),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), output_type_meta));
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
        output_dims, output_strides, output_storage_offset);
    return tensor;
  });
}

}  // namespace torch_tpu
