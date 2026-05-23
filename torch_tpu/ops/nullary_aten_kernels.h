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

#ifndef TORCH_TPU_OPS_NULLARY_ATEN_KERNELS_H_
#define TORCH_TPU_OPS_NULLARY_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "c10/core/Device.h"
#include "c10/core/SymIntArrayRef.h"
#include "c10/util/Optional.h"
#include "torch/headeronly/core/Layout.h"
#include "torch/headeronly/core/MemoryFormat.h"
#include "torch/headeronly/core/ScalarType.h"
#include "torch_tpu/common/cache_key.h"
#include "torch_tpu/eager/device_buffer.h"
#include "torch_tpu/ops/op_builder_utils.h"

namespace torch_tpu {

at::Tensor ApplyNullaryOp(MlirNullaryOpBuilder op_builder,
                          c10::ScalarType out_dtype, at::IntArrayRef out_dims,
                          OpParamCacheKeys op_param_cache_keys,
                          OpSplitMode split_mode = OpSplitMode::kNone);

absl::Status ApplyNullaryOpOut(at::Tensor& out, MlirNullaryOpBuilder op_builder,
                               c10::ScalarType out_dtype,
                               at::IntArrayRef out_dims,
                               OpParamCacheKeys op_param_cache_keys,
                               OpSplitMode split_mode = OpSplitMode::kNone);

// Implements AtenEmptyMemoryFormat() without going through TT_KERNEL
// dispatching.
absl::StatusOr<at::Tensor> MakeEmptyMemoryFormat(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt);

at::Tensor AtenEmptyMemoryFormat(
    at::IntArrayRef size,                     //
    c10::optional<at::ScalarType> dtype_opt,  // Defaults to the global default.
    c10::optional<at::Layout> layout_opt,     // Defaults to Strided.
    c10::optional<at::Device> device_opt,     // Defaults to the global default.
    c10::optional<bool> pin_memory_opt,       // Defaults to false.
    c10::optional<at::MemoryFormat>
        memory_format_opt);  // Defaults to Contiguous.

// Creates an empty tensor with the given size, dtype, and device. This tensor
// is strided and contiguous.
absl::StatusOr<at::Tensor> MakeEmptyTensor(
    at::IntArrayRef size, c10::ScalarType dtype,
    c10::optional<at::Device> device_opt);

at::Tensor AtenEmptyStrided(c10::SymIntArrayRef size_sym,
                            c10::SymIntArrayRef stride_sym,
                            c10::optional<at::ScalarType> dtype_opt,
                            c10::optional<at::Layout> layout_opt,
                            c10::optional<at::Device> device_opt,
                            c10::optional<bool> pin_memory_opt);

at::Tensor AtenEfficientZeroTensor(at::IntArrayRef size,
                                   c10::optional<at::ScalarType> dtype_opt,
                                   c10::optional<at::Layout> layout_opt,
                                   c10::optional<at::Device> device_opt,
                                   c10::optional<bool> pin_memory_opt);

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_NULLARY_ATEN_KERNELS_H_
