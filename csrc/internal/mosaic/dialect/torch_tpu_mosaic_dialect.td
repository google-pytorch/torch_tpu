/* Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
==============================================================================*/

#ifndef TORCH_TPU_INTERNAL_MOSAIC_DIALECT_TORCH_TPU_TD
#define TORCH_TPU_INTERNAL_MOSAIC_DIALECT_TORCH_TPU_TD

include "mlir/IR/OpBase.td"

def TorchTpuMosaic_Dialect : Dialect {
  let name = "torch_tpu_mosaic";
  let cppNamespace = "::mlir::torch_tpu_mosaic";
  let summary = "Thin wrapper around the Mosaic dialect for TorchTPU use cases.";
}

#endif  // TORCH_TPU_INTERNAL_MOSAIC_DIALECT_TORCH_TPU_TD
