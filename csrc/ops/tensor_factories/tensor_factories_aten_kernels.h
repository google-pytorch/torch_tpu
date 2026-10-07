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

#ifndef TORCH_TPU_CSRC_OPS_TENSOR_FACTORIES_TENSOR_FACTORIES_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_TENSOR_FACTORIES_TENSOR_FACTORIES_ATEN_KERNELS_H_

#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/core/Device.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"

namespace torch_tpu {

at::Tensor AtenZeros(at::IntArrayRef size,
                     std::optional<at::ScalarType> dtype_opt,
                     std::optional<at::Layout> layout_opt,
                     std::optional<at::Device> device_opt,
                     std::optional<bool> pin_memory_opt);

at::Tensor AtenOnes(at::IntArrayRef size,
                    std::optional<at::ScalarType> dtype_opt,
                    std::optional<at::Layout> layout_opt,
                    std::optional<at::Device> device_opt,
                    std::optional<bool> pin_memory_opt);

at::Tensor AtenFull(at::IntArrayRef size, const at::Scalar& fill_value,
                    std::optional<at::ScalarType> dtype_opt,
                    std::optional<at::Layout> layout_opt,
                    std::optional<at::Device> device_opt,
                    std::optional<bool> pin_memory_opt);

at::Tensor AtenZerosLike(const at::Tensor& self,
                         std::optional<at::ScalarType> dtype_opt,
                         std::optional<at::Layout> layout_opt,
                         std::optional<at::Device> device_opt,
                         std::optional<bool> pin_memory_opt,
                         std::optional<at::MemoryFormat> memory_format_opt);

at::Tensor AtenOnesLike(const at::Tensor& self,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt,
                        std::optional<at::MemoryFormat> memory_format_opt);

at::Tensor AtenFullLike(const at::Tensor& self, const at::Scalar& fill_value,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt,
                        std::optional<at::MemoryFormat> memory_format_opt);

at::Tensor AtenNewZeros(const at::Tensor& self, at::IntArrayRef size,
                        std::optional<at::ScalarType> dtype_opt,
                        std::optional<at::Layout> layout_opt,
                        std::optional<at::Device> device_opt,
                        std::optional<bool> pin_memory_opt);

at::Tensor AtenNewOnes(const at::Tensor& self, at::IntArrayRef size,
                       std::optional<at::ScalarType> dtype_opt,
                       std::optional<at::Layout> layout_opt,
                       std::optional<at::Device> device_opt,
                       std::optional<bool> pin_memory_opt);

at::Tensor AtenNewFull(const at::Tensor& self, at::IntArrayRef size,
                       const at::Scalar& fill_value,
                       std::optional<at::ScalarType> dtype_opt,
                       std::optional<at::Layout> layout_opt,
                       std::optional<at::Device> device_opt,
                       std::optional<bool> pin_memory_opt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_TENSOR_FACTORIES_TENSOR_FACTORIES_ATEN_KERNELS_H_
