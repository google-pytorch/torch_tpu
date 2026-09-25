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

    Standard `torch.profiler` (integrated with `torch_tpu`) provides a **context manager** interface for tracing both CPU and TPU activities. You wrap the code you want to profile in a `torch.profiler.profile()` block, and the collected traces are saved to disk for viewing in TensorBoard.

    **Key components:**

    | Component | Role |
    |-----------|------|
    | `torch.profiler.profile()` | Standard context manager that starts/stops trace collection |
    | `torch.profiler.ProfilerActivity.CPU` | Traces CPU-side operations (dispatch, data loading) |
    | `torch.profiler.ProfilerActivity.TPU` | Traces TPU hardware execution (compute, memory, interconnect) |
    | `torch.profiler.tensorboard_trace_handler` | Callback that saves traces to a directory for TensorBoard |
    | `TpuProfilerConfig` | Helper for custom TPU profiling configuration |
    """)
  return


@app.cell
def _():
  import torch
  from torch.tpu.profiler import TpuProfilerConfig

  device = torch.device("tpu")
  return TpuProfilerConfig, device, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **2. Custom Profiling Configuration (`TpuProfilerConfig`)**

    Options passed via `TpuProfilerConfig` allow fine-grained control over host and device tracer levels as well as output directory structures.

    ### **Sanctioned Configuration Options**

    | Option Key | Type | Default | Description |
    | :--- | :--- | :--- | :--- |
    | `host_tracer_level` | `int` | `2` | Controls CPU host tracer verbosity (0=off, 1=critical, 2=all). |
    | `device_tracer_level` | `int` | `1` | Controls TPU hardware device tracer verbosity (0=off, 1=on). |
    | `python_tracer_level` | `int` | `0` | Controls Python function tracing level (0=off, 1=on). |
    | `run_dir` | `pathlib.Path` | `None` | Output directory where TPU hardware trace files (`.xplane.pb`) are saved. |


    ```python
    import pathlib
    import torch
    from torch.tpu.profiler import TpuProfilerConfig

    config = TpuProfilerConfig(
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=0,
        run_dir=pathlib.Path("./profiler_output"),
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.TPU,
        ],
        experimental_config=config,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
    ) as prof:
        model(inputs)
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **3. Capturing a Profile**

    The cell below runs a small training loop inside the profiler context. The key points:

    1. **Wrap your training loop** inside `torch.profiler.profile()`
    2. **Specify activities** — `CPU` for host-side ops, `TPU` for hardware execution
    3. **Set `on_trace_ready`** to save traces to a directory using `torch.profiler.tensorboard_trace_handler`
    4. **(Optional) Pass `experimental_config`** with `TpuProfilerConfig` to control tracer levels

    > **⚠️ Important:** The profiler adds overhead. Profile only a representative subset of steps (e.g., 100–1000), not your entire training run.
    """)
  return


@app.cell
def _(TpuProfilerConfig, device, torch):
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

  # Configure TPU-specific tracer levels
  config = TpuProfilerConfig(
      host_tracer_level=2,
      device_tracer_level=1,
  )

  # Profile 100 training steps with native PyTorch profiler
  with torch.profiler.profile(
      activities=[
          torch.profiler.ProfilerActivity.CPU,
          torch.profiler.ProfilerActivity.TPU,
      ],
      experimental_config=config,
      on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
  ) as prof:
    for step in range(100):
      optimizer.zero_grad()
      loss = model(data).sum()
      loss.backward()
      optimizer.step()
      torch.accelerator.synchronize()
      prof.step()

  print(f"✅ Profile captured: {100} steps saved to {log_dir}")
  return (log_dir,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **4. Viewing Traces in TensorBoard**

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
    ## **5. What to Look For**

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
    ## **6. Verifying the Traces Were Captured**

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
    ## **7. Kernel Profiling (Fine-Grained Performance Counters)**

    Kernel Profiling enables the collection of fine-grained hardware performance counters (e.g., for Pallas kernels) to diagnose performance bottlenecks that are otherwise invisible in standard event traces.

    > [!NOTE]
    > Fine-grained performance counter sampling is an advanced hardware feature available for **Ironwood (TPU v7/TPU7x)** and subsequent TPU generations.

    ### **Enabling Performance Counters**

    You can enable performance counters by passing the counter configuration via `experimental_options` in `TpuProfilerConfig`:

    ```python
    from torch.tpu.profiler import TpuProfilerConfig

    config = TpuProfilerConfig(
        experimental_options={
            "tpu_enable_periodic_counter_sampling": True,
            "tpu_tc_perf_counter_sampling_options": (
                "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
                " indices:11 indices:56 indices:57 indices:58"
            ),
        }
    )
    ```

    For the detailed list of configuration parameters (such as sampling intervals, scaling, and trigger modes) and instructions on how to determine counter indices, refer to the official [OpenXLA Kernel Profiling Guide](https://openxla.org/xprof/kernel-profiling).
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **8. Quick Reference**

    | Step | Command / Code |
    |------|---------------|
    | **Import** | `from torch.tpu.profiler import TpuProfilerConfig` |
    | **Profile** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):` |
    | **Save** | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")` |
    | **View** | `tensorboard --logdir=/tmp/profiler_output` |

    > [!TIP]
    > Profile only 100–1000 steps to keep trace files manageable. The profiler adds ~5–10% overhead per step.
    """)
  return


if __name__ == "__main__":
  app.run()
