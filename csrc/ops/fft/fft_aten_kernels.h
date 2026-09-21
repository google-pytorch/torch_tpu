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

#ifndef TORCH_TPU_CSRC_OPS_FFT_FFT_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_FFT_FFT_ATEN_KERNELS_H_

#include <cstdint>
#include <optional>

#include "ATen/core/ATen_fwd.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/string_view.h"

namespace torch_tpu {

at::Tensor AtenFftR2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool onesided);
at::Tensor& AtenFftR2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool onesided,
                          at::Tensor& out);

at::Tensor AtenFftC2c(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, bool forward);
at::Tensor& AtenFftC2cOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, bool forward, at::Tensor& out);
at::Tensor AtenFftC2r(const at::Tensor& self, at::IntArrayRef dim,
                      int64_t normalization, int64_t last_dim_size);
at::Tensor& AtenFftC2rOut(const at::Tensor& self, at::IntArrayRef dim,
                          int64_t normalization, int64_t last_dim_size,
                          at::Tensor& out);

at::Tensor AtenFftFft(const at::Tensor& self, std::optional<c10::SymInt> n,
                      int64_t dim, std::optional<c10::string_view> norm);
at::Tensor& AtenFftFftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                          int64_t dim, std::optional<c10::string_view> norm,
                          at::Tensor& out);

at::Tensor AtenFftIfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                       int64_t dim, std::optional<c10::string_view> norm);
at::Tensor& AtenFftIfftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                           int64_t dim, std::optional<c10::string_view> norm,
                           at::Tensor& out);

at::Tensor AtenFftRfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                       int64_t dim, std::optional<c10::string_view> norm);
at::Tensor& AtenFftRfftOut(const at::Tensor& self, std::optional<c10::SymInt> n,
                           int64_t dim, std::optional<c10::string_view> norm,
                           at::Tensor& out);

at::Tensor AtenFftIrfft(const at::Tensor& self, std::optional<c10::SymInt> n,
                        int64_t dim, std::optional<c10::string_view> norm);
at::Tensor& AtenFftIrfftOut(const at::Tensor& self,
                            std::optional<c10::SymInt> n, int64_t dim,
                            std::optional<c10::string_view> norm,
                            at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_FFT_FFT_ATEN_KERNELS_H_
