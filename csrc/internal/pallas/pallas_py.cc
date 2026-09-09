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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/string_view.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/layout_utils.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/custom_kernels.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type

namespace torch_tpu {
namespace py = pybind11;
namespace {

void PyRegisterCustomKernel(c10::string_view name, c10::string_view kernel_key,
                            c10::string_view mlir_module_string) {
  if (RegisterCustomKernel(name, kernel_key, mlir_module_string)) {
    ABSL_VLOG(1) << "Registered new custom kernel: name=" << name
                 << ", kernel_key=" << kernel_key;
  } else {
    ABSL_VLOG(2) << "Custom kernel already registered: name=" << name
                 << ", kernel_key=" << kernel_key;
  }
}

bool PyLookupCustomKernel(c10::string_view name, c10::string_view kernel_key) {
  return LookupCustomKernel(name, kernel_key);
}

// Dispatches a previously registered custom kernel (e.g., Pallas or XLA
// MLIR custom call) on the TPU eager runtime and returns the output tensors.
//
// Contract:
// - Preconditions:
//   - The custom kernel identified by `name` and `kernel_key` must already be
//     registered in the runtime symbol registry (via RegisterCustomKernel).
//   - All tensors in `inputs` must reside on the TPU device or be compatible
//     with TorchTPU device buffer conversion.
//   - `output_shapes` must specify the expected layout, dtype, and shape of
//     each output produced by the kernel.
//   - Elements of `donate_argnums` must be valid 0-based indices into `inputs`.
// - Postconditions:
//   - Returns a vector of freshly constructed at::Tensor objects representing
//     the execution results, matching `output_shapes` in count, shape, and
//     dtype.
//   - If `output_shapes` is empty, returns an empty vector without dispatching
//     any MLIR operation or executing on hardware.
// - Side Effects:
//   - Schedules asynchronous kernel execution on the TPU hardware/runtime.
//   - Input buffers corresponding to indices in `donate_argnums` may be
//     donated (reused or mutated in place by XLA) during execution, making
//     their prior tensor contents invalid or updated.
//
// Parameters:
//   name: The registered base name of the custom kernel symbol (e.g.,
//     "add_kernel").
//   kernel_key: A unique fingerprint of the lowered MLIR module.
//   inputs: Vector of input tensors to be passed into the kernel.
//   output_shapes: Vector of dummy/placeholder tensors whose shapes and dtypes
//     define the expected output tensor layouts from kernel execution.
//   donate_argnums: 0-based indices of leaf input tensors whose underlying
//     device buffers can be donated/aliased for output memory reuse.
//
// Example usage from Python:
//   ```python
//   import torch
//   from torch_tpu._internal.pallas import tpu_torch_pallas
//
//   kernel_name = "my_custom_add"
//   mlir_bytes = lower_to_mlir(block_size=128)  # Generates serialized MLIR
//   kernel_key = compute_fingerprint(mlir_bytes)
//
//   # 1. Register the kernel if it hasn't been registered yet for this key:
//   if not tpu_torch_pallas.lookup_custom_kernel(kernel_name, kernel_key):
//     tpu_torch_pallas.register_custom_kernel(
//         kernel_name,
//         kernel_key,
//         serialized_mlir_module=mlir_bytes,
//     )
//
//   # 2. Call the custom kernel:
//   x = torch.randn(1024, 1024, device="xla")
//   y = torch.randn(1024, 1024, device="xla")
//   out_placeholder = torch.empty_like(x)
//
//   results = tpu_torch_pallas.call_custom_kernel(
//       kernel_name,
//       kernel_key,
//       inputs=[x, y],
//       output_shapes=[out_placeholder],
//       donate_argnums=[0],  # Donate buffer x for output aliasing if possible.
//   )
//   out = results[0]
//   ```
std::vector<at::Tensor> PyCallCustomKernel(
    c10::string_view name, c10::string_view kernel_key,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& output_shapes,
    // Use std::vector as pybind has built-in support for it.
    const std::vector<int64_t>& donate_argnums  // INT_VEC_OK
) {
  TT_KERNEL(
      OpName::kCustomKernel, op_param_cache_keys,
      (name, kernel_key, inputs, output_shapes, donate_argnums), {
        if (output_shapes.empty()) {
          return {};
        }
        auto custom_op_builder = [name = std::string(name),
                                  kernel_key = std::string(kernel_key)](
                                     absl::Span<const mlir::MlirOp> inputs,
                                     mlir::MlirBuilder& builder) {
          return CallCustomKernel(builder, inputs, name, kernel_key);
        };

        std::vector<mlir::ElementType> output_dtypes;
        std::vector<absl::Span<const int64_t>> output_dims_list;
        output_dtypes.reserve(output_shapes.size());
        output_dims_list.reserve(output_shapes.size());

        std::vector<Dimensions> resolved_dims;
        resolved_dims.reserve(output_shapes.size());

        for (const auto& output_shape : output_shapes) {
          TT_ASSIGN_OR_THROW(auto layout, ResolveTpuLayout(output_shape));
          output_dtypes.push_back(layout.element_type);
          resolved_dims.push_back(std::move(layout.sizes));
        }
        for (const auto& dims : resolved_dims) {
          output_dims_list.push_back(dims);
        }

        DispatchOpOptions<kDynamicSize> options{
            .out_dtypes = output_dtypes,
            .out_dims_list = output_dims_list,
            .computation_dtype = std::nullopt,
            .op_param_cache_keys = std::move(op_param_cache_keys)};

        // Note that this DeferredOp needs its inputs to be marked with
        // jax.buffer_donor if they are leaf inputs to the MLIR module.
        options.donated_indices.assign(donate_argnums.begin(),
                                       donate_argnums.end());

        absl::StatusOr<std::vector<DeviceBufferRef>> results_status =
            DispatchOp<kDynamicSize, kDynamicSize>(std::move(custom_op_builder),
                                                   inputs, std::move(options));
        TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> results, results_status,
                           _.SetOverride()
                               << ::torch_tpu::AdaptExternalErrorMessage(
                                      results_status.status().message()));

        std::vector<at::Tensor> result_tensors;
        result_tensors.reserve(results.size());
        for (auto& result : results) {
          result_tensors.push_back(MakeTensor(std::move(result)));
        }
        return result_tensors;
      });
}

}  // namespace

PYBIND11_MODULE(tpu_torch_pallas, m) {
  m.def("register_custom_kernel", PyRegisterCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"),
        py::kw_only(),  // Everything after this is keyword-only
        py::arg("serialized_mlir_module"));
  m.def("lookup_custom_kernel", PyLookupCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"));
  m.def("call_custom_kernel", PyCallCustomKernel,  //
        py::arg("name"), py::arg("kernel_key"),
        py::kw_only(),  // Everything after this is keyword-only
        py::arg("inputs"), py::arg("output_shapes"),
        py::arg("donate_argnums") = std::vector<int64_t>{});  // INT_VEC_OK
}

}  // namespace torch_tpu
