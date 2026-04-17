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

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **Performance Profiling with xProf**

    Understanding where time is spent is critical for training large models on TPU. This guide covers how to capture execution traces with the TPU Profiler and analyze them in TensorBoard.

    > [!WARNING]
    > **Prerequisite:** TensorBoard requires `setuptools<81`. The latest version removes `pkg_resources`, which TensorBoard depends on. Install it first:
    > ```shell
    > pip install "setuptools<81"
    > pip install xprof tensorboard
    > ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **1. The TPU Profiler**

    The `torch_tpu.profiler` provides a **context manager** interface for tracing both CPU and TPU activities. You wrap the code you want to profile in a `profiler.profile()` block, and the collected traces are saved to disk for viewing in TensorBoard.

    **Key components:**

    | Component | Role |
    |-----------|------|
    | `profiler.profile()` | Context manager that starts/stops trace collection |
    | `ProfilerActivity.CPU` | Traces CPU-side operations (dispatch, data loading) |
    | `ProfilerActivity.TPU` | Traces TPU hardware execution (compute, memory, interconnect) |
    | `xprof_trace_handler` | Callback that saves traces to a directory for TensorBoard |
    """)
  return


@app.cell
def _():
  import torch
  from torch_tpu._internal import profiler
  from torch_tpu._internal import sync
  import tpu_utils

  device = tpu_utils.safe_init()
  return device, profiler, sync, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **2. Capturing a Profile**

    The cell below runs a small training loop inside the profiler context. The key points:

    1. **Wrap your training loop** inside `profiler.profile()`
    2. **Specify activities** — `CPU` for host-side ops, `TPU` for hardware execution
    3. **Set `on_trace_ready`** to save traces to a directory

    > **⚠️ Important:** The profiler adds overhead. Profile only a representative subset of steps (e.g., 100–1000), not your entire training run.
    """)
  return


@app.cell
def _(device, profiler, sync, torch):
  # Build a small model and optimizer
  model = (
      torch.nn.Sequential(
          torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
      )
      .to(device)
      .to(torch.bfloat16)
  )

  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

  # Where to save the profiler traces
  log_dir = "/tmp/profiler_output"

  # Profile 100 training steps
  with profiler.profile(
      activities=[
          profiler.ProfilerActivity.CPU,
          profiler.ProfilerActivity.TPU,
      ],
      on_trace_ready=profiler.xprof_trace_handler(dir_name=log_dir),
  ):
    for step in range(100):
      optimizer.zero_grad()
      loss = model(data).sum()
      loss.backward()
      optimizer.step()

  print(f"✅ Profile captured: {100} steps saved to {log_dir}")
  return (log_dir,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **3. Viewing Traces in TensorBoard**

    Run the following commands:

    **Step 1: Install the profiler plugin** (required for the dashboard to appear):
    ```shell
    pip install tensorboard-plugin-profile "setuptools<81"
    ```

    **Step 2: Launch TensorBoard:**
    ```shell
    tensorboard --logdir=/tmp/profiler_output --port=6006
    ```

    xprof can also be run as a standalone server. 
    ```shell
    xprof --logdir=profiler/demo --port=6006
    ```

    **Step 3:** Open `http://localhost:6006` in your browser and select the **"Profile"** tab.

    > [!IMPORTANT]
    > You must run TensorBoard from the VS Code integrated terminal so that port forwarding works automatically.

    | View | What it shows |
    |------|--------------|
    | **Overview** | High-level summary: step time, device utilization, top ops |
    | **Trace Viewer** | Timeline of every op on CPU and TPU — look for gaps (idle hardware) |
    | **Op Profile** | Breakdown of time per operation — find the most expensive ops |
    | **Memory Profile** | HBM usage over time — detect memory spikes |
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **4. What to Look For**

    ### **Red Flags in Traces**

    | Pattern | Meaning | Fix |
    |---------|---------|-----|
    | **Large gaps** between TPU ops | Host-side bottleneck (data loading, Python overhead) | Prefetch data, reduce Python logic in the loop |
    | **Frequent short TPU bursts** | Too many graph breaks → frequent recompilation | Reduce `.item()` / `.cpu()` calls, use `torch.compile` |
    | **Low MXU utilization** | Tensor dimensions not aligned to tile size (128) | Pad dimensions to multiples of 128 |

    ### **The Ideal Profile**

    A well-optimized TPU workload shows:
    - **Long, continuous TPU compute blocks** — the compiler fused many ops into one execution
    - **Minimal CPU→TPU gaps** — data loading is overlapped with compute
    - **High MXU utilization (>60%)** — tensor shapes are hardware-aligned
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **5. Verifying the Traces Were Captured**

    The cell below checks that the profiler output directory contains trace files.
    """)
  return


@app.cell
def _(log_dir):
  import os

  trace_files = os.listdir(log_dir)
  print(f"Trace directory: {log_dir}")
  print(f"Files found: {len(trace_files)}")
  for f in sorted(trace_files)[:10]:
    size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
    print(f"  {f} ({size_kb:.1f} KB)")

  if trace_files:
    print(f"\n✅ Traces ready. View with:")
    print(f"   tensorboard --logdir={log_dir}")
  else:
    print("\n❌ No trace files found. Check profiler configuration.")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **6. Quick Reference**

    | Step | Command / Code |
    |------|---------------|
    | **Import** | `from torch_tpu._internal import profiler, sync` |
    | **Profile** | `with profiler.profile(activities=[...], on_trace_ready=...):` |
    | **Save** | `profiler.xprof_trace_handler(dir_name="/tmp/profiler_output")` |
    | **View** | `tensorboard --logdir=/tmp/profiler_output` |

    > [!TIP]
    > Profile only 100–1000 steps to keep trace files manageable. The profiler adds ~5–10% overhead per step.
    """)
  return


if __name__ == "__main__":
  app.run()
