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
    # **The Compilation Cache**

    Understanding "Cold Starts" vs. "Warm Hits" is essential for achieving peak TPU performance. This guide explains how graph fingerprints are generated, why compilation storms happen, and how to ensure your training loop stays on the fast path.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **How the Cache Works**

    The first time a specific sequence of operations is executed, the backend must compile it into TPU machine code. This is a **"Cold Start"** and for large models, can add up to several seconds or even minutes.

    Once compiled, the resulting binary is stored in the **Compilation Cache**. The second time you run same sequence of ops (e.g., iteration 2 of a training loop), it will be a **"Warm Hit"** and execution starts instantly.

    ```
    Iteration 1:  Record Graph → Fingerprint → Cache MISS → Compile → Execute → Store in Cache
    Iteration 2:  Record Graph → Fingerprint → Cache HIT  → Execute (instant)
    Iteration 3:  Record Graph → Fingerprint → Cache HIT  → Execute (instant)
    ...
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Graph Fingerprinting**

    The backend generates a **fingerprint** for each computation graph based on:

    1. **Input shapes** — e.g., `[32, 128]` vs. `[16, 128]` Note that **input scalar** values also affect the fingerprint.
    2. **Input dtypes** — e.g., `float32` vs. `bfloat16`
    3. **Sequence of operations** — the exact ATen ops in order

    If **any** of these change, the fingerprint most likely changes, and the cache produces a **miss** — triggering a full compilation.
    """)
  return


@app.cell
def cache_stats_demo():
  import torch

  device = torch.device("tpu")

  # Run some operations to populate the cache
  x = torch.randn(32, 128, device="tpu")
  y = x @ x.T
  _ = y.cpu()  # Trigger materialization → compilation

  # Run the same graph again (should be a cache hit)
  x2 = torch.randn(32, 128, device="tpu")
  y2 = x2 @ x2.T
  _ = y2.cpu()
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Avoiding Unnecessary Recompilations**
    If your input shapes change every iteration (e.g., variable sequence lengths without padding), or if you have data-dependent operations (e.g. nonzero()), the backend may generate a new fingerprint and trigger a re-compilation of the changed ops.

    | Scenario | Fingerprints | Performance |
    | :--- | :--- | :--- |
    | Fixed batch size (32), fixed sequence length (128) | 1 unique graph | ✅ Fast — always cache hits |
    | Fixed batch size (32), 3 sequence buckets (128, 512, 1024) | 3 unique graphs | ✅ Good — 3 cold starts, then all hits |
    | Variable sequence lengths (every batch different) | N unique graphs | ❌ Re-compilation |
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    A common cause of recompilation in production is an inconsistent batch size.

    * **Scenario:** Batch size = 32, dataset has 100 samples. Batches 1–3 are size 32, but batch 4 is size 4.
    * **Result:** The shape change (32 → 4) triggers re-compilation on the last step of **every epoch**.
    * **Solution:** Use `drop_last=True` in your DataLoader.

    ```python
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,  # CRITICAL: prevents recompilations from odd batch shapes
        num_workers=0,  # CRITICAL: avoids os.fork deadlock post-PJRT initialization
    )
    ```

    > [!WARNING]
    > **Data Loading on TPU (`num_workers=0` Recommended)**:
    > PyTorch's `DataLoader` on Linux defaults to `multiprocessing.get_context("fork")`. Because the TPU runtime (PJRT) initializes multithreaded drivers and acquires hardware device locks (`/tmp/libtpu_lockfile`), forking worker subprocesses causes silent deadlocks before step 0.
    > 
    > Always set **`num_workers=0`** to load data in the main process. For high-throughput asynchronous input pipelines, use Google's [`Grain`](https://github.com/google/grain) library (`grain.python.DataLoader`).
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Warmup Runs**

    When measuring step time, make sure to exclude the first few steps as that can include the warmup (compilation) phase.

    **Recommendation:** Run 2 to 3 warmup iterations before starting your timer.

    ```python
    # Warmup iterations to populate the cache
    for _ in range(2):
        output = model(warmup_data)
        output.cpu()  # Force materialization → compilation

    # Now start measuring real performance
    start_time = time.time()
    # ... actual training loop ...
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    > [!TIP]
    > Use the TPU profiler (`torch.profiler`) to check HBM usage.
    """)
  return


if __name__ == "__main__":
  app.run()
