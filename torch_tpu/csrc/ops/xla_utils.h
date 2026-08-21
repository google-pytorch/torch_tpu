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

#ifndef TORCH_TPU_OPS_XLA_UTILS_H_
#define TORCH_TPU_OPS_XLA_UTILS_H_

#include "c10/util/BFloat16.h"
#include "c10/util/Half.h"
#include "torch/headeronly/util/BFloat16.h"
#include "torch/headeronly/util/Half.h"
#include "torch/headeronly/util/complex.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"  // IWYU pragma: keep for xla::PrimitiveType

namespace torch_tpu {

inline xla::half ToXlaHalf(at::Half value) { return xla::half(value.x); }

inline xla::bfloat16 ToXlaBFloat16(at::BFloat16 value) {
  return xla::bfloat16(value.x);
}

inline xla::complex128 ToXlaComplex128(c10::complex<double> value) {
  return {value.real(), value.imag()};
}

inline xla::complex64 ToXlaComplex64(c10::complex<float> value) {
  return {value.real(), value.imag()};
}

}  // namespace torch_tpu

#endif  // TORCH_TPU_OPS_XLA_UTILS_H_
