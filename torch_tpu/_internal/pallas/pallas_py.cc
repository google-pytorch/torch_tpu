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
#include <unordered_map>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "c10/util/string_view.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "torch/extension.h"  // IWYU pragma: keep for aten::Tensor pybind type
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/common/dimension_types.h"
#include "torch_tpu/common/dtype.h"
#include "torch_tpu/common/error_utils.h"
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
    const c10::string_view name, const c10::string_view kernel_key,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& output_shapes,
    // Use std::vector as pybind has built-in support for it.
    const std::vector<int64_t>& donate_argnums,  // INT_VEC_OK
    bool inplace_donate,
    // Parallel (input_index, output_index) donation pairs from the module's
    // tf.aliasing_output attrs; a donated input's aliased output generally
    // sits at a different tuple position. Empty means identity pairing over
    // donate_argnums (single-output kernels donating input 0).
    const std::vector<int64_t>& donate_alias_inputs,   // INT_VEC_OK
    const std::vector<int64_t>& donate_alias_outputs) {  // INT_VEC_OK
  TT_KERNEL(
      OpName::kCustomKernel, op_param_cache_keys,
      (name, kernel_key, inputs, output_shapes, donate_argnums,
       IgnoreInCacheKey(
           inplace_donate,
           "Only affects host-side output wrapping, not op compilation."),
       IgnoreInCacheKey(
           donate_alias_inputs,
           "Only affects host-side output wrapping, not op compilation."),
       IgnoreInCacheKey(
           donate_alias_outputs,
           "Only affects host-side output wrapping, not op compilation.")),
      {
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
        for (const auto& output_shape : output_shapes) {
          TT_ASSIGN_OR_THROW(
              const auto output_dtype,
              ConvertTo<mlir::ElementType>(output_shape.scalar_type()));
          output_dtypes.push_back(output_dtype);
          output_dims_list.push_back(output_shape.sizes());
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
        TT_ASSIGN_OR_THROW(std::vector<DeviceBufferRef> results,
                           results_status);

        // When inplace_donate is set, each donated input tensor's storage is
        // rebound to its aliased output's buffer below. The pairing comes
        // from the module's tf.aliasing_output attrs (donate_alias_inputs/outputs);
        // without pairs it defaults to identity over donate_argnums. The
        // returned output is still a fresh at::Tensor wrapping the same
        // buffer — both reads and writes to the input go to the kernel's
        // output buffer. See `_internal/pallas/pallas.py` JaxCallable
        // for the motivating workload (block-major KV cache threaded
        // through every attention layer).
        TT_CHECK_THROW(
            donate_alias_inputs.size() == donate_alias_outputs.size(),
            error::kInvalidArgument)
            << "donate_alias_inputs/outputs must pair up";
        absl::flat_hash_map<int64_t, int64_t> rebind_input_by_output;
        if (inplace_donate) {
          if (!donate_alias_inputs.empty()) {
            for (size_t p = 0; p < donate_alias_inputs.size(); ++p) {
              rebind_input_by_output[donate_alias_outputs[p]] =
                  donate_alias_inputs[p];
            }
          } else {
            for (const int64_t idx : donate_argnums) {
              rebind_input_by_output[idx] = idx;
            }
          }
        }

        std::vector<at::Tensor> result_tensors;
        result_tensors.reserve(results.size());
        for (auto i = 0; i < results.size(); ++i) {
          const int64_t out_idx = static_cast<int64_t>(i);
          if (const auto it = rebind_input_by_output.find(out_idx);
              it != rebind_input_by_output.end()) {
            const int64_t in_idx = it->second;
            TT_CHECK_THROW(in_idx < static_cast<int64_t>(inputs.size()),
                           error::kInvalidArgument)
                << "inplace_donate=True requires the donated index " << in_idx
                << " to be a valid input index (size " << inputs.size() << ")";
            // Two strong refs to the same buffer: one stays with the freshly
            // returned at::Tensor; the other rebinds the donated input's
            // storage. We MUST NOT also return `inputs[in_idx]` — that
            // would alias an input and trip torch.library's
            // `_c_check_aliasing_constraint`. The point of the rebind is
            // that subsequent reads of the input observe the kernel's
            // writes without the caller having to `input.copy_(output)`.
            DeviceBufferRef rebind_buf = results[i];
            TT_THROW_IF_ERROR(
                AssignBufferToAtTensor(std::move(rebind_buf), inputs[in_idx]));
          }
          result_tensors.push_back(MakeTensor(std::move(results[i])));
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
        py::arg("donate_argnums") = std::vector<int64_t>{},  // INT_VEC_OK
        py::arg("inplace_donate") = false,
        py::arg("donate_alias_inputs") = std::vector<int64_t>{},   // INT_VEC_OK
        py::arg("donate_alias_outputs") = std::vector<int64_t>{});  // INT_VEC_OK
}

}  // namespace torch_tpu
