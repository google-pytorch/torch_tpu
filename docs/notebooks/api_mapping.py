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
"""Marimo notebook for API mapping."""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return mo


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # **API Mapping Table: `torch.cuda` → `torch.tpu`**

    Once the backend is initialized via `torch.accelerator.current_accelerator()`, TorchTPU registers a dedicated Python module: **`torch.tpu`**. This module is designed as a drop-in replacement for the standard `torch.cuda` API.

    Use this reference to quickly translate CUDA-based code to TorchTPU.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Core Device Queries**

    | CUDA API | TPU API | Description |
    | :--- | :--- | :--- |
    | `torch.cuda.is_available()` | `torch.tpu.is_available()` | Checks if the TPU backend is ready. |
    | `torch.cuda.device_count()` | `torch.tpu.device_count()` | Number of devices assigned to the process. |
    | `torch.cuda.current_device()` | `torch.tpu.current_device()` | Returns the active core index. |
    | `torch.cuda.get_device_properties()` | *(Implicit in `torch.accelerator.current_accelerator()`)* | Hardware metadata via PjRt logs. |
    | `torch.cuda.set_device(rank)` | Handled automatically by `torch.accelerator.current_accelerator()` | Backend init is automatic. |
    """)
  return


@app.cell
def device_queries():
  import torch

  import tpu_utils  # Only used in Marimo notebooks

  # Initialize hardware
  device = tpu_utils.safe_init()  # Only used in Marimo notebooks

  print(f"torch.tpu.is_available():    {torch.tpu.is_available()}")
  print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
  print(f"torch.tpu.device_count():    {torch.tpu.device_count()}")
  print(f"torch.tpu.current_device():  {torch.tpu.current_device()}")
  return (torch,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Random Number Generation**

    | CUDA API | TPU API | Description |
    | :--- | :--- | :--- |
    | `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Seeds the current core's RNG. |
    | `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Seeds all TPU cores (for distributed). |
    """)
  return


@app.cell
def seeding_example(torch):
  # Set the seed for the TPU-resident Philox generator across all cores
  torch.tpu.manual_seed_all(42)
  print("Seeded all TPU cores with value 42.")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Automatic Mixed Precision (AMP)**

    TorchTPU integrates with the standard `torch.amp.autocast` framework. The device string is `"tpu"`.

    | CUDA API | TPU API | Description |
    | :--- | :--- | :--- |
    | `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Enables automatic mixed precision. |
    | `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Lists hardware-accelerated dtypes. |
    """)
  return


@app.cell
def amp_example(torch):
  supported_dtypes = torch.tpu.get_amp_supported_dtype()
  print(f"AMP supported dtypes: {supported_dtypes}")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Streams and Events**

    The `torch.tpu` module provides `Stream` and `Event` objects to maintain API parity with CUDA.

    | CUDA API | TPU API | Status |
    | :--- | :--- | :--- |
    | `torch.cuda.current_stream()` | `torch.tpu.current_stream()` | **Placeholder (dummy)** |
    | `torch.cuda.default_stream()` | `torch.tpu.default_stream()` | **Placeholder (dummy)** |
    | `torch.cuda.set_stream()` | `torch.tpu.set_stream()` | **Placeholder (dummy)** |

    > [!NOTE]
    > In TorchTPU's deferred execution model, the XLA compiler and PjRt runtime manage execution ordering and hardware concurrency automatically. Streams are not needed.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Cache and HBM Telemetry**

    TorchTPU exposes specialized methods for auditing the Compilation Cache and hardware memory (HBM). These have **no CUDA equivalent**.

    | TPU API | Description |
    | :--- | :--- |
    | `torch.tpu._hbm_usage_summary()` | HBM usage by compiled binaries. |
    | `torch.tpu._get_cache_stats()` | Detailed cache hit/miss ratios and compilation latencies. |
    """)
  return


@app.cell
def cache_telemetry(torch):
  # HBM usage summary (this is not a stable API, use with caution)
  print(torch.tpu._hbm_usage_summary())

  # Cache statistics (this is not a stable API, use with caution)
  stats = torch.tpu._get_cache_stats()
  print(f"\nCache Requests: {stats.num_cache_reqs}")
  print(f"Cache Hits:     {stats.num_cache_hits}")
  return


if __name__ == "__main__":
  app.run()
