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

// Layout-pinned tensor allocation for torch_tpu.
//
// `allocate_with_layout(cpu_tensor, minor_to_major)` copies the contents of
// `cpu_tensor` to a TPU device buffer whose underlying XLA Layout has the
// requested `minor_to_major` axis order. This lets callers persist
// `nn.Parameter`s in a layout that matches the consumer kernel's
// `entry_computation_layout`, so the per-step relayout that today shows up
// as `tt_jit_as_strided` / `copy_bitcast_fusion` modules can be DCE'd by XLA.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ATen/core/TensorBody.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/pjrt/pjrt_utils.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xla/layout.h"

namespace torch_tpu {
namespace py = pybind11;

namespace {

absl::StatusOr<at::Tensor> AllocateWithLayoutImpl(
    const at::Tensor& cpu_tensor, const std::vector<int64_t>& minor_to_major) {
  if (!cpu_tensor.is_cpu()) {
    return absl::InvalidArgumentError(
        "allocate_with_layout: source tensor must be on CPU.");
  }
  if (static_cast<size_t>(minor_to_major.size()) !=
      static_cast<size_t>(cpu_tensor.dim())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "allocate_with_layout: minor_to_major has ", minor_to_major.size(),
        " entries but tensor has ", cpu_tensor.dim(), " dimensions."));
  }
  // Verify it is a permutation of [0, rank).
  std::vector<bool> seen(minor_to_major.size(), false);
  for (int64_t axis : minor_to_major) {
    if (axis < 0 ||
        static_cast<size_t>(axis) >= minor_to_major.size() ||
        seen[axis]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "allocate_with_layout: minor_to_major must be a permutation of "
          "[0, rank), got [",
          absl::StrJoin(minor_to_major, ","), "]."));
    }
    seen[axis] = true;
  }

  // Force a contiguous CPU source so `data_ptr()` reflects the row-major
  // logical layout that TpuMallocAndMemcpyHtoD assumes.
  at::Tensor contiguous_src = cpu_tensor.contiguous();

  TT_ASSIGN_OR_RETURN(
      const auto dtype,
      ConvertTo<mlir::ElementType>(contiguous_src.scalar_type()));

  if (contiguous_src.numel() == 0) {
    Dimensions zero_sized_dimensions(contiguous_src.sizes().begin(),
                                     contiguous_src.sizes().end());
    TT_ASSIGN_OR_RETURN(
        DeviceBufferRef ref,
        DeviceBufferList::CreateZeroSize(std::move(zero_sized_dimensions),
                                         dtype));
    return MakeTensor(std::move(ref));
  }

  // Build the requested xla::Layout. PJRT's BufferFromHostBuffer takes the
  // device-side layout (minor-to-major) directly.
  xla::Layout layout(minor_to_major);

  ABSL_VLOG(1) << "[allocate_with_layout] dims=["
               << absl::StrJoin(contiguous_src.sizes(), ",")
               << "] minor_to_major=[" << absl::StrJoin(minor_to_major, ",")
               << "] dtype=" << c10::toString(contiguous_src.scalar_type());

  TT_ASSIGN_OR_RETURN(
      DeviceBufferRef tpu_buf,
      TpuMallocAndMemcpyHtoD(contiguous_src.data_ptr(), dtype,
                             contiguous_src.sizes(),
                             /*backing_tensor=*/std::nullopt,
                             /*device_layout=*/&layout));
  return MakeTensor(std::move(tpu_buf));
}

}  // namespace

PYBIND11_MODULE(layout_impl, m) {
  m.doc() =
      "Layout-pinned TPU tensor allocation. See `_internal.layout` for the "
      "Python API.";

  m.def(
      "allocate_with_layout",
      [](const at::Tensor& cpu_tensor,
         const std::vector<int64_t>& minor_to_major) -> at::Tensor {
        TT_ASSIGN_OR_THROW(at::Tensor result,
                           AllocateWithLayoutImpl(cpu_tensor, minor_to_major));
        return result;
      },
      py::arg("cpu_tensor"), py::arg("minor_to_major"),
      R"docstring(Allocate a TPU tensor with a specific XLA layout.

Args:
  cpu_tensor: a CPU torch.Tensor whose data is copied to TPU.
  minor_to_major: list of ints — the XLA layout minor-to-major axis order.
      For example, for an `f32[E,K,1,N]` tensor that the consumer kernel
      expects with layout `{1,3,2,0:T(...)}`, pass `[1, 3, 2, 0]`.

Returns:
  A TPU `torch.Tensor` whose underlying PJRT buffer has the requested
  XLA layout.
)docstring");
}

}  // namespace torch_tpu
