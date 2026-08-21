---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

+++

# 🚀 TorchTPU Documentation
Your hub for using PyTorch on Google's TPUs.

💡 Every page is a Jupyter notebook — download it and run it on a TPU VM.

+++

### 🏁 Navigation
[Tutorials](#tutorials) | [How-to Guides](#guides) | [Reference](#reference) | [Explanation](#explanation)

+++

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
pip install portpicker jupyter scikit-learn pandas

# 4. Optional for Pallas kernels
pip install jax

# 5. Optional for xProf
pip install tensorboard
```

+++

(tutorials)=

### 🎓 Tutorials
Step-by-step lessons that teach core concepts hands-on.

1.  **[Get Started with TorchTPU](get_started_tpu.ipynb)**
    *Device init, deferred execution, and materialization*
2.  **[Your First Port (ViT on MNIST)](porting_vit_mnist.ipynb)**
    *Ghosting shims, Linear patching, and pure-tensor pipelines*
3.  **The Debugger's Toolkit** *(coming soon)*
    *OpTracer, format_model, and graph visualization*
4.  **[Distributed Training 101](distributed_training_101.ipynb)**
    *Multi-core scaling (DDP) and the TPU environment orchestrator*

+++

(guides)=

### 🛠️ How-to Guides
Focused recipes for solving specific problems.

*   **[Numerical Parity Workflow](numerical_parity.ipynb)**
    *Validating TPU outputs match CPU within tolerance*
*   **Scaling to Large Models (FSDP)** *(coming soon)*
    *Sharding models across TPU chips with FSDP*
*   **[Advanced torch.compile on TPU](advanced_compile.ipynb)**
    *Graph breaks, backends, and compilation debugging*
*   **[Precision Management (BFloat16 & AMP)](precision_management.ipynb)**
    *Casting, autocasting, and precision diagnostics*
*   **[Custom Kernels with Pallas](custom_pallas_kernel.ipynb)**
    *Writing custom TPU kernels with JAX Pallas*
*   **[Performance Profiling](performance_profiling.ipynb)**
    *Capturing and analyzing TPU performance traces*

+++

(reference)=

### 📖 Reference
Lookup tables and quick-reference materials.

*   **[API Mapping Table](api_mapping.ipynb)**
    *PyTorch ↔ TorchTPU API equivalents*
*   **[Environment Variables](environment_variables.ipynb)**
    *All env vars that control TorchTPU behavior*
*   **[Error Code Glossary](error_glossary.ipynb)**
    *Common errors, what they mean, and how to fix them*
*   **Hardware Alignment Cheat Sheet** *(coming soon)*
    *MXU tile sizes, padding rules, and performance cliffs*

+++

(explanation)=

### 🧠 Explanation
Deep dives into how TorchTPU works under the hood.

*   **The XLA & StableHLO Pipeline** *(coming soon)*
    *How PyTorch ops become TPU machine code*
*   **[The Compilation Cache](compilation_cache.ipynb)**
    *How compiled graphs are cached and reused*
*   **[Eager Mode](eager_mode.ipynb)**
    *Guide to Fused Eager execution and debugging modes.*
*   **[Precision Management](precision_management.ipynb)**
    *Guide to controlling floating-point precision on TPU.*
*   **Strict SPMD Synchronization** *(coming soon)*
    *Why all ranks must execute the same graph*

+++

(custom_ops)=

### 🚀 Advanced: Custom Ops and Kernels
Extend PyTorch on TPU from basic composition to low-level Pallas kernels. This progressive guide walks through different mechanisms to extend PyTorch functionality, starting with the easiest composition approach and gradually building up towards the most powerful and complex TPU hardware-level vector kernels.

*   **[Extending Modules via Composition](extending_modules_and_functions_via_composition.ipynb)**
    *Quantization Aware Training via module composition*
*   **[Customizing Autograd via torch.autograd.Function](customizing_autograd_via_torch_autograd_function.ipynb)**
    *Defining custom backward passes with torch.autograd.Function*
*   **[Background on ATen Ops](background_on_aten_ops.ipynb)**
    *Enabling aot_graphs logging to view intermediate ATen IR*
*   **[Background on torch.compile](background_on_torch_compile.ipynb)**
    *Concept of operator fusion and memory bottlenecks*
*   **[Background on torch.compile on TPUs via XLA](background_on_torch_compile_on_tpus_via_xla.ipynb)**
    *Dumping and inspecting TPU HLO & LLO vector instructions*
*   **[Custom Ops via JAX Kernels](quantized_sum.ipynb)**
    *General custom ops concepts and wrapping JAX popcnt*
*   **[Custom Ops via Pallas: Python Baseline](quant.ipynb)**
    *Quantization and packing unoptimized baseline in Python*
*   **[Custom Ops via Pallas: TPU Kernel](custom_ops_via_pallas.ipynb)**
    *Writing high-performance binarization and packing vector kernels in Pallas*
