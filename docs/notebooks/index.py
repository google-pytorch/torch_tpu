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

# pylint: skip-file  # ruff: excluded from toml

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
    # 🚀 TorchTPU Interactive Portal
    Your hub for using PyTorch on Google's TPUs.

    💡 Run `marimo edit tutorials/` to browse all notebooks
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### 🏁 Navigation
    [Tutorials](#tutorials) | [How-to Guides](#guides) | [Reference](#reference) | [Explanation](#explanation)
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ### ⚙️ Installation — Set up TorchTPU on your TPU VM

    Install TorchTPU and its dependencies on your TPU VM. Components must be installed in this order to ensure the PyTorch dispatcher and TPU drivers align correctly.

    #### Step 1: Initialize Python 3.12

    ```bash
    sudo apt update && sudo apt install -y python3.12 python3.12-venv
    python3.12 -m venv env
    source env/bin/activate
    ```

    #### Step 2: Install TorchTPU & Dependencies

    ```bash
    # 1. Authenticate to get access to whl
    pip install keyrings.google-artifactregistry-auth
    gcloud auth login
    gcloud auth application-default login

    # 2. Install torch_tpu, PyTorch 2.11 CPU will install automatically, do not install manually.
    pip install --pre --index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu

    # 3. Install optional utilities
    pip install portpicker marimo scikit-learn pandas

    # 4. Optional for Pallas kernels
    pip install jax

    # 5. Optional for xProf
    pip install tensorboard
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    <div id="tutorials"></div>

    ### 🎓 Tutorials
    Step-by-step lessons that teach core concepts hands-on.

    1.  **[Get Started with TorchTPU](?file=get_started_tpu.py)**
        *Device init, deferred execution, and materialization*
    2.  **[Your First Port (ViT on MNIST)](?file=porting_vit_mnist.py)**
        *Ghosting shims, Linear patching, and pure-tensor pipelines*
    3.  **[The Debugger's Toolkit](?file=debugger_toolkit.py)**
        *OpTracer, format_model, and graph visualization*
    4.  **[Distributed Training 101](?file=distributed_training_101.py)**
        *Multi-core scaling (DDP) and the TPU environment orchestrator*
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    <div id="guides"></div>

    ### 🛠️ How-to Guides
    Focused recipes for solving specific problems.

    *   **[Numerical Parity Workflow](?file=numerical_parity.py)**
        *Validating TPU outputs match CPU within tolerance*
    *   **[Scaling to Large Models (FSDP)](?file=scaling_fsdp.py)**
        *Sharding models across TPU chips with FSDP*
    *   **[Advanced torch.compile on TPU](?file=advanced_compile.py)**
        *Graph breaks, backends, and compilation debugging*
    *   **[Mixed Precision (BFloat16 & AMP)](?file=mixed_precision.py)**
        *Casting, autocasting, and precision diagnostics*
    *   **[Custom Kernels with Pallas](?file=custom_pallas_kernels.py)**
        *Writing custom TPU kernels with JAX Pallas*
    *   **[Performance Profiling](?file=performance_profiling.py)**
        *Capturing and analyzing TPU performance traces*
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    <div id="reference"></div>

    ### 📖 Reference
    Lookup tables and quick-reference materials.

    *   **[API Mapping Table](?file=api_mapping.py)**
        *PyTorch ↔ TorchTPU API equivalents*
    *   **[Environment Variables](?file=environment_variables.py)**
        *All env vars that control TorchTPU behavior*
    *   **[Error Code Glossary](?file=error_glossary.py)**
        *Common errors, what they mean, and how to fix them*
    *   **[Hardware Alignment Cheat Sheet](?file=hardware_alignment.py)**
        *MXU tile sizes, padding rules, and performance cliffs*
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    <div id="explanation"></div>

    ### 🧠 Explanation
    Deep dives into how TorchTPU works under the hood.

    *   **[The XLA & StableHLO Pipeline](?file=xla_stablehlo_pipeline.py)**
        *How PyTorch ops become TPU machine code*
    *   **[The Compilation Cache](?file=compilation_cache.py)**
        *How compiled graphs are cached and reused*
    *   **[Eager Mode](?file=eager_mode.py)**
        *Guide to Fused Eager execution and debugging modes.*
    *   **[Precision Management](?file=precision_management.py)**
        *Guide to controlling floating-point precision on TPU.*
    *   **[Strict SPMD Synchronization](?file=spmd_synchronization.py)**
        *Why all ranks must execute the same graph*
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    <div id="custom_ops"></div>

    ### 🚀 Advanced: Custom Ops and Kernels
    Extend PyTorch on TPU from basic composition to low-level Pallas kernels. This progressive guide walks through different mechanisms to extend PyTorch functionality, starting with the easiest composition approach and gradually building up towards the most powerful and complex TPU hardware-level vector kernels.

    *   **[Extending Modules via Composition](?file=extending_modules_and_functions_via_composition.py)**
        *Quantization Aware Training via module composition*
    *   **[Customizing Autograd via torch.autograd.Function](?file=customizing_autograd_via_torch_autograd_function.py)**
        *Defining custom backward passes with torch.autograd.Function*
    *   **[Background on ATen Ops](?file=background_on_aten_ops.py)**
        *Enabling aot_graphs logging to view intermediate ATen IR*
    *   **[Background on torch.compile](?file=background_on_torch_compile.py)**
        *Concept of operator fusion and memory bottlenecks*
    *   **[Background on torch.compile on TPUs via XLA](?file=background_on_torch_compile_on_tpus_via_xla.py)**
        *Dumping and inspecting TPU HLO & LLO vector instructions*
    *   **[Custom Ops via JAX Kernels](?file=quantized_sum.py)**
        *General custom ops concepts and wrapping JAX popcnt*
    *   **[Custom Ops via Pallas: Python Baseline](?file=quant.py)**
        *Quantization and packing unoptimized baseline in Python*
    *   **[Custom Ops via Pallas: TPU Kernel](?file=custom_ops_via_pallas.py)**
        *Writing high-performance binarization and packing vector kernels in Pallas*
    """)
  return


if __name__ == "__main__":
  app.run()
