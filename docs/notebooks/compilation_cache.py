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

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo
  import os

  # Environment Configuration
  os.environ["ACCELERATOR_TYPE"] = "v6e-4"
  return mo, os


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

    The first time a specific sequence of operations (a "graph") is materialized, the backend must compile it into TPU machine code. This is a **"Cold Start"** and can take several seconds or even minutes for complex models.

    Once compiled, the resulting binary is stored in the **Compilation Cache**. The second time you run that same graph (e.g., iteration 2 of a training loop), it will be a **"Warm Hit"** and execution starts instantly.

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

    1. **Input shapes** — e.g., `[32, 128]` vs. `[16, 128]` 
    2. **Input dtypes** — e.g., `float32` vs. `bfloat16`
    3. **Input scalars**
    3. **Sequence of operations** — the exact ATen ops in order

    If **any** of these change, the fingerprint most likely changes, and the cache produces a **miss** — triggering a full compilation.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Monitoring Cache Performance**

    You can audit your cache performance directly from Python to ensure your training loop is hitting the "Warm" path.
    """)
  return


@app.cell
def cache_stats_demo():
  import torch
  import tpu_utils

  device = tpu_utils.safe_init()

  # Run some operations to populate the cache
  x = torch.randn(32, 128, device=device)
  y = x @ x.T
  _ = y.cpu()  # Trigger materialization → compilation

  # Run the same graph again (should be a cache hit)
  x2 = torch.randn(32, 128, device=device)
  y2 = x2 @ x2.T
  _ = y2.cpu()

  # Fetch cache statistics
  cache_stats = torch.tpu._get_cache_stats()
  print(f"Total Requests: {cache_stats.num_cache_reqs}")
  print(f"Total Hits:     {cache_stats.num_cache_hits}")
  print(
      "Hit Rate:      "
      f" {cache_stats.num_cache_hits / max(cache_stats.num_cache_reqs, 1) * 100:.1f}%"
  )

  # Inspect individual cache entries
  for i, entry in enumerate(cache_stats.per_entry_stats):
    print(f"\nEntry {i}:")
    print(f"  Read Count: {entry.read_count}")
    print(f"  Compilation Duration (us): {entry.compilation_duration}")
    print(f"  Last Read: {entry.last_read}")
  return device, torch


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **The "Compilation Storm"**
    If your input shapes change every iteration (e.g., variable sequence lengths without padding), or if you have data-dependent operations (e.g. nonzero()), the backend generates a new fingerprint and triggers a full re-compilation every time. This is called a "Compilation Storm" and makes training crawl.

    | Scenario | Fingerprints | Performance |
    | :--- | :--- | :--- |
    | Fixed batch size (32), fixed sequence length (128) | 1 unique graph | ✅ Fast — always cache hits |
    | Fixed batch size (32), 3 sequence buckets (128, 512, 1024) | 3 unique graphs | ✅ Good — 3 cold starts, then all hits |
    | Variable sequence lengths (every batch different) | N unique graphs | ❌ **Storm** — constant re-compilation |
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **The "Final Batch" Trap**

    A common cause of compilation storms in production is an inconsistent batch size.

    * **Scenario:** Batch size = 32, dataset has 100 samples. Batches 1–3 are size 32, but batch 4 is size 4.
    * **Result:** The shape change (32 → 4) triggers re-compilation on the last step of **every epoch**.
    * **Solution:** Use `drop_last=True` in your DataLoader.

    ```python
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True,
        drop_last=True  # CRITICAL for TPU performance
    )
    ```
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Warmup Runs**

    Because the first iteration is always a Cold Start, **never measure performance on the first pass**.

    **Recommendation:** Run 1–2 warmup iterations before starting your timer.

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
    ## **Key Metrics to Monitor**

    | Metric | Healthy Value | Problem Indicator |
    | :--- | :--- | :--- |
    | **Cache Hit Rate** | > 95% after warmup | < 80% suggests dynamic shapes |
    | **Unique Graphs** | 1–5 per training loop | > 10 suggests a compilation storm |
    | **Cold Start Duration** | Seconds (simple), minutes (complex) | If every step is a cold start, shapes are changing |

    > [!TIP]
    > Use `torch.tpu._hbm_usage_summary()` to check how much HBM is consumed by cached compiled binaries.
    """)
  return


if __name__ == "__main__":
  app.run()
