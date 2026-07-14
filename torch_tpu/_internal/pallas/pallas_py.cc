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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/error_utils.h"
#include "torch_tpu/common/layout_utils.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/eager/op_dispatcher.h"
#include "torch_tpu/eager/tensor_to_buffer.h"
#include "torch_tpu/ops/custom_kernels.h"
#include "torch_tpu/ops/macros/kernel.h"
#include "torch_tpu/ops/op_builder_utils.h"
#include "torch_tpu/ops/op_names.h"

namespace torch_tpu {
namespace py = pybind11;
namespace {

void PyRegisterCustomKernel(c10::string_view name, c10::string_view kernel_key,
                            c10::string_view mlir_module_string) {
  if (RegisterCustomKernel(name, kernel_key, mlir_module_string)) {
    ABSL_VLOG(1) << "Registered new custom kernel: name=" << name
                 << ", kwargs=" << kernel_key;
  } else {
    ABSL_VLOG(2) << "Custom kernel already registered: name=" << name
                 << ", kwargs=" << kernel_key;
  }
}

bool PyLookupCustomKernel(c10::string_view name, c10::string_view kernel_key) {
  return LookupCustomKernel(name, kernel_key);
}

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
