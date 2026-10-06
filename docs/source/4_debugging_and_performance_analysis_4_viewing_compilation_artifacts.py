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
    # Background on `torch.compile` on TPUs via XLA

    TorchTPU is the backend for PyTorch to run on Google TPUs. TorchTPU uses XLA to
    compile PyTorch rather than Inductor. XLA stands for "Accelerated Linear
    Algebra" and is an [open-source ML compiler](https://openxla.org/xla).

    The path from your Python code to assembly on a TPU is as follows:

    1.  Your Python code will require minor changes (e.g. setting "TPU" as the
        device).
    2.  PyTorch's dispatcher lowers your Python code to ATen and c10d (no change).
    3.  TorchTPU lowers the ATen and c10d ops to **StableHLO**, a dialect of
        [MLIR](https://mlir.llvm.org/) that is device agnostic.
    4.  XLA lowers the StableHLO to **HLO** ops.
    5.  XLA optimizes the HLO over several passes.
    6.  XLA lowers the HLO ops to device-specific **LLO** (Low-Level Ops) operators.
    7.  XLA optimizes the LLO over several passes, targeting the specific TPU
        hardware.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 1. Setting up HLO & LLO Dumps

    To inspect how XLA compiles a pointwise addition and ReLU operation, we can configure environment variables to dump the intermediate HLO ops and LLO ops.

    #### HLO Dump Configuration:
    ```python
    XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
    os.environ["XLA_FLAGS"] = f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
    ```

    #### LLO Dump Configuration:
    ```python
    LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"
    flags = [ f"--xla_jf_dump_to={LLO_DUMP_TO}", "--xla_jf_dump_llo_text=true" ]
    sys.argv.extend(flags)
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)
    ```

    Let's run the compiled pointwise operation and dump the generated HLO/LLO logs.
    """)
  return


@app.cell
def _():
  import os
  import sys
  from typing import Final

  # Setup compilation dumps directories
  LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"
  flags = [
      f"--xla_jf_dump_to={LLO_DUMP_TO}",
      "--xla_jf_dump_llo_text=true",
  ]
  sys.argv.extend(flags)
  os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)

  XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
  os.environ["XLA_FLAGS"] = (
      f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
  )

  from absl import app as absl_app
  import torch

  return LLO_DUMP_TO, XLA_DUMP_TO, torch


@app.cell
def _(torch):
  @torch.compile(backend="tpu")
  def fwd(x, y):
    return torch.nn.functional.relu(torch.add(x, y))

  return (fwd,)


@app.function
def dump_dir(directory):
  import os

  entries = os.listdir(directory) if os.path.exists(directory) else []
  print(f"\n=== Files in {directory} (Total: {len(entries)}) ===")

  # Only print the filenames to keep standard output fast
  for f in sorted(entries):
    print(f)

  print(
      "\n💡 Tip: All compilation files are written to disk. You can read any"
      " file interactively using: open(os.path.join(directory,"
      " filename)).read()"
  )


@app.cell
def _(LLO_DUMP_TO: "Final[str]", XLA_DUMP_TO: "Final[str]", fwd, torch):
  def main():
    tpu_available = False
    try:
      x = torch.randn(1024).to("tpu")
      y = torch.randn(1024).to("tpu")
      tpu_available = True
    except Exception:
      pass

    if not tpu_available:
      print(
          "TPU not available. Run this notebook on a TPU VM to generate live"
          " compilation dumps."
      )
      return 0

    res = fwd(x, y)
    _ = res.cpu()

    dump_dir(XLA_DUMP_TO)
    dump_dir(LLO_DUMP_TO)

    print("Success.")
    return 0

  main()
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 2. Analyzing the HLO (Before Optimizations)

    The output section will list several generated files in `/tmp/xla_dump`.

    Look for the module file named `module_XXXX.tt_jit_pointwise_fusion_xla_...before_optimizations.txt`. (Note that exact filenames vary based on your specific environment and line numbering).

    Opening the `before_optimizations.txt` file, you will find the initial unoptimized HLO representation. Notice that it explicitly outlines an `add` and a `maximum` (which represents the ReLU):

    ```hlo
    ENTRY %main.1 (Arg_0.1: f32[1024], Arg_1.1: f32[1024]) -> f32[1024] {
      %Arg_0.1 = f32[1024]{0} parameter(0)
      %Arg_1.1 = f32[1024]{0} parameter(1)
      %add.1 = f32[1024]{0} add(%Arg_0.1, %Arg_1.1), metadata={op_name="add/add" stack_frame_id=40}
      %constant.1 = f32[] constant(0)
      %relu.2 = f32[1024]{0} broadcast(%constant.1), dimensions={}, metadata={op_name="relu/relu" stack_frame_id=40}
      ROOT %relu.3 = f32[1024]{0} maximum(%add.1, %relu.2), metadata={op_name="relu/relu" stack_frame_id=40}
    }
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 3. Analyzing the HLO (After Codegen)

    The XLA compiler will then optimize this HLO. Looking inside the optimized codegen output file (`*after_codegen.txt`), you will see that XLA has successfully fused the two operations together into a single block assigned to a loop fusion computation variable: `add_maximum_fusion`.

    ```hlo
    %fused_computation (param_0.2: f32[1024], param_1.2: f32[1024]) -> f32[1024] {
      %param_0.2 = f32[1024]{0:T(1024)} parameter(0)
      %param_1.2 = f32[1024]{0:T(1024)} parameter(1)
      %add.0 = f32[1024]{0:T(1024)} add(%param_0.2, %param_1.2), metadata={op_name="add/add" stack_frame_id=40}
      %constant.0 = f32[]{:T(128)} constant(0)
      %relu.1 = f32[1024]{0:T(1024)} broadcast(%constant.0), dimensions={}, metadata={op_name="relu/relu" stack_frame_id=40}
      ROOT %relu.0 = f32[1024]{0:T(1024)} maximum(%add.0, %relu.1), metadata={op_name="relu/relu" stack_frame_id=40}
    }

    ENTRY %main.1 (Arg_0.1: f32[1024], Arg_1.1: f32[1024]) -> f32[1024] {
      %Arg_1.1 = f32[1024]{0:T(1024)} parameter(1), backend_config={"flag_configs":[],"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[]}
      %Arg_0.1 = f32[1024]{0:T(1024)} parameter(0), backend_config={"flag_configs":[],"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[]}
      ROOT %add_maximum_fusion = f32[1024]{0:T(1024)} fusion(%Arg_0.1, %Arg_1.1), kind=kLoop, calls=%fused_computation, metadata={op_name="relu/relu" stack_frame_id=40}, backend_config={"flag_configs":[],"window_config":{"kernel_window_bounds":[],"output_window_bounds":["1"],"input_window_bounds":[],"estimated_cycles":"1840","iteration_bounds":["1"],"cost_model_type":"COST_MODEL_TYPE_INVALID","ml_estimated_microseconds":0,"is_mask":false,"pad_output_on_minor_dim":"0","pad_input_on_minor_dim":"0","estimated_vmem_bytes":"0","estimated_bundle_count":"0","estimated_scoped_vmem_bytes":"0"},"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"12288"}],"retry_config":{"retry_count":"0"},"aliasing_operands":{"lists":[]}}
    }
    ```

    The key indicator here is `kind=kLoop` and the reference `calls=%fused_computation`, meaning both operators compile into a single memory-efficient loop.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## 4. Analyzing the LLO assembly

    In the LLO dump folder `/tmp/llo_dump`, search for files containing the `add_maximum_fusion` token. Look for the file ending in `add_maximum_fusion-01-original.txt` to view the physical TPU assembly instructions.

    Without needing to be a TPU architecture expert, you can isolate the exact vector addition (`vadd`) and vector maximum (`vmax`) instructions executing in TPU registers:

    ```assembly
    === File: 1777396851540540302-add_maximum_fusion-01-original.txt ===
    // [Enable stack traces via -xla_jf_collect_llo_stack_trace or -xla_jf_debug_level=2]
    $region0: #{add_maximum_fusion}
      #allocation6 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
      %s0 = inlined_call_operand.hbm [shape: f32[1024], index: 0, kind: input, shape index: {}] /* operand 0 */
      %s1 = inlined_call_operand.hbm [shape: f32[1024], index: 1, kind: input, shape index: {}] /* operand 1 */
      %s2 = inlined_call_operand.hbm [shape: f32[1024], index: 2, kind: output, shape index: {}] /* operand 2 */
      $region1: #{add_maximum_fusion} parent=0
        #allocation0 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 0']
        #allocation1 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
        #allocation2 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
        #allocation3 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 1']
        #allocation4 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
        #allocation5 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 2']
        %3 = vsyncpa [#allocation1], 0
        %4 = vsyncpa [#allocation4], 0
        %5 = vsyncpa [#allocation2], 0
        %7 = vsyncadd [#allocation1], 0
        %s9 = sshll.u32 %s0, 4
        %s10 = int_to_ptr.hbm [resolvable:$true] %s9
        %s11 = sshll.u32 [#allocation0], 4
        %s12 = int_to_ptr.vmem [resolvable:$true] %s11
        %14 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s10, /*size_in_granules=*/128, /*vmem=*/%s12, /*dst_syncflagno=*/[#allocation1]
        ...
        %25 = dma.done [#allocation1], 128 /* pipeline-emitter-dma-wait */
        %27 = dma.done [#allocation4], 128 /* pipeline-emitter-dma-wait */
        %v28 = vld [vmem:[#allocation0] sm:$0xff]
        %v29 = vld [vmem:[#allocation3] sm:$0xff]
        %30 = xla_tuple %v28, %v29
        %31 = xla_tuple %30
        %v32 = vadd.f32 %v28, %v29                   <-- Vector Addition
        %33 = xla_tuple %v32
        %34 = xla_tuple %v32, 0.0
        %35 = xla_tuple %34
        %v36 = vmax.f32 %v32, 0.0                   <-- Vector Maximum (ReLU)
        %37 = xla_tuple %v36
        %38 = vst [vmem:[#allocation5] sm:$0xff] /*vst_source=*/%v36
        %40 = vsyncadd [#allocation2], 0
        ...
        %49 = dma.done [#allocation2], 128 /* pipeline-emitter-dma-wait */
        %50 = vsyncpa [#allocation1], 1
        %51 = vsyncpa [#allocation4], 1
        %52 = vsyncpa [#allocation2], 1
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## Conclusion

    The heart of creating highly optimized custom TPU kernels is gainful control of this final outputted LLO, regardless of whether you write kernels using StableHLO, JAX-level ops, or Pallas.

    In the next chapters, we will explore how to build these custom ops using PyTorch's custom op API and low-level TPU kernel DSLs.
    """)
  return


if __name__ == "__main__":
  app.run()
