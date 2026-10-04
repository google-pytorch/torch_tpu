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

    Once the backend is initialized via `device = torch.device("tpu")`, TorchTPU registers a dedicated Python module: **`torch.tpu`**. This module is designed as a drop-in replacement for the standard `torch.cuda` API.

    Use this reference to quickly translate CUDA-based code to TorchTPU.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Core Device Queries**

    | CUDA API | TPU API | Description |
    | :--- | :--- | :--- |
    | `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Checks if the TPU backend is ready. |
    | `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Checks if PyTorch's TPU state has been initialized. |
    | `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Number of devices assigned to the process. |
    | `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Returns the active core index. |
    | `torch.cuda.get_device_name()` | `torch.tpu.get_device_name()` | Returns human-readable model name of the attached TPU. |
    | `torch.cuda.get_device_properties()` | `torch.tpu.get_device_properties()` | Returns static device properties (name, memory, cores). |
    | `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Validates the active device index for the current process. |
    """)
  return


@app.cell
def device_queries():
  import torch

  # Initialize hardware
  device = torch.device("tpu")

  print(
      f"torch.accelerator.is_available():    {torch.accelerator.is_available()}"
  )
  print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
  print(
      f"torch.accelerator.device_count():    {torch.accelerator.device_count()}"
  )
  print(
      "torch.accelerator.current_device_index(): "
      f" {torch.accelerator.current_device_index()}"
  )
  return (torch,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **Random Number Generation**

    | CUDA API | TPU API | Description |
    | :--- | :--- | :--- |
    | `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Seeds the current core's RNG. |
    | `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Seeds all TPU cores (for distributed). |
    | `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Returns initial random seed of current TPU device. Requires PyTorch >= 2.13; use `torch.tpu.initial_seed()` on older versions. |
    | `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Returns RNG state tensor for the current device. Requires PyTorch >= 2.13; use `torch.tpu.get_rng_state()` on older versions. |
    | `torch.cuda.set_rng_state(state)` | `torch.tpu.set_rng_state(state)` | Sets RNG state tensor for the current device. |
    """)
  return


@app.cell
def seeding_example(torch):
  # Set the seed across all cores
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

    For streams and events, prefer standard `torch.accelerator` APIs. Use `torch.tpu` only for TPU-specific features.

    | CUDA API | TPU / Accelerator API | Description |
    | :--- | :--- | :--- |
    | `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Returns the active stream. |
    | `torch.cuda.set_stream()` | `torch.accelerator.set_stream()` | Sets the active stream. |
    | `torch.cuda.default_stream()` | `torch.tpu.default_stream()` | Returns the default stream. |

    > [!NOTE]
    > In TorchTPU's deferred execution model, the XLA compiler and PjRt runtime manage execution ordering and hardware concurrency automatically. Streams are not needed.
    """)
  return


if __name__ == "__main__":
  app.run()
