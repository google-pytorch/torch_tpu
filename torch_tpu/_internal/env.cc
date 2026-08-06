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
#include "pybind11/pybind11.h"
#include "torch_tpu/common/native_scan_support.h"
#include "torch_tpu/common/utils.h"

namespace torch_tpu {

PYBIND11_MODULE(env, m) {
  m.attr("IS_INTERNAL_TORCH_TPU") = static_cast<bool>(TT_IS_INTERNAL_TORCH_TPU);
#if defined(NDEBUG)
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = true;
#else
  m.attr("TORCH_TPU_IS_OPTIMIZED_BUILD") = false;
#endif
  // Gates whether cumulative ops emit the native scan emitter (chlo.ScanOp);
  // set from Python at import time based on the libtpu version (b/529376045).
  m.def("set_native_scan_emitter_supported", &SetNativeScanEmitterSupported,
        pybind11::arg("supported"));
  m.def("native_scan_emitter_supported", &NativeScanEmitterSupported);
}

}  // namespace torch_tpu
