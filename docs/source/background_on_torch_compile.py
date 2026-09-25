# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # Background on `torch.compile`

    When you use
    [`torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
    with the default `inductor` backend on CUDA, one of the optimizations is fusing
    ops. Without fusion, a sequence of operations like addition followed by ReLU
    would require multiple reads and writes to High Bandwidth Memory (HBM) in a
    synchronous, serial fashion. The addition would load data, compute, and write
    back to HBM. After the addition op completes, the ReLU op would then read that
    data back, compute, and write again. Fusion allows data to be loaded once,
    processed for both operations, and written back once.

    *Note: The code below is designed for CUDA (GPUs) and is presented statically here as this environment is configured for TPUs.*

    Below is the implementation from `pointwise_fusion_cuda.py`. This script
    demonstrates a full fusion within a single Triton kernel for a simple operation:

    ```python
    import os
    import sys

    # Set the flags to print generated code
    os.environ["TORCH_COMPILE_LOGS"] = "output_code"
    os.environ["TORCH_LOGS"] = "output_code"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "0"

    import torch

    @torch.compile(backend="inductor")
    def fwd(x, y):
      return torch.nn.functional.relu(torch.add(x, y))

    def main():
      if not torch.cuda.is_available():
        print("CUDA not available. This test requires a GPU.")
        return 0

      x = torch.randn(1024, device="cuda")
      y = torch.randn(1024, device="cuda")

      fwd(x, y)

      print("Success.")
      return 0

    if __name__ == "__main__":
      sys.exit(main())
    ```

    Inspecting the logs for this test (enabled via `TORCH_LOGS="output_code"`), you
    find a single Triton kernel named `triton_poi_fused_add_relu_0`. Even without
    becoming a Triton expert, you can get a basic idea of what this kernel is doing.
    Moving data from HBM to registers is explicit, and the computation happens in a
    fused way: there is no intermediate write to HBM.

    ```python
    @triton.jit
    def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
        # ...
        tmp0 = tl.load(in_ptr0 + (x0), xmask) # Load x from HBM to registers
        tmp1 = tl.load(in_ptr1 + (x0), xmask) # Load y from HBM to registers
        tmp2 = tmp0 + tmp1 # Pointwise addition
        tmp3 = tl.full([1], 0, tl.int32) # Create tensor of zeros in registers
        tmp4 = triton_helpers.maximum(tmp3, tmp2) # Apply ReLU in registers
        tl.store(out_ptr0 + (x0), tmp4, xmask) # Store result to HBM
    ```

    > 💡 **Note**: The reason this example uses a pointwise op rather than a matmul:
    > Inductor will often avoid attempting to fuse matmuls because the native
    > library, cuBLAS, is already highly optimized.
    """)
  return


if __name__ == "__main__":
  app.run()
