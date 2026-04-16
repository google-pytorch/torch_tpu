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

"""Marimo notebook for Error Code Glossary."""

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
    # **Error Code Glossary**

    When a model fails on the TPU, the error output can be complex — spanning Python tracebacks, C++ backtraces, and XLA compiler messages. This guide lets you **trigger and diagnose** each error type interactively.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **1. Enabling Rich Error Context**

    By default, TorchTPU shows only the Python traceback when a Python exception is raised. Setting `TORCH_SHOW_CPP_STACKTRACES=1` **before** importing `torch` enriches errors with:

    - **C++ Error Trace** — Where the error was first detected in TorchTPU's C++ code and how it was propagated through C++ code.
    - **Captured Traceback** — The full dispatcher path from Python → ATen → TorchTPU

    The cell below triggers a real error so you can see both layers.
    """)
  return


@app.cell
def _(os):
  import traceback

  # Enable C++ stack traces BEFORE importing torch
  os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

  import torch
  import tpu_utils

  device = tpu_utils.safe_init()

  # Trigger an error: negative dimension is invalid
  try:
    torch.ones(-1, device=device)
  except RuntimeError as e:
    error_text = str(e)
    print("═" * 60)
    print("CAUGHT RuntimeError (with C++ context)")
    print("═" * 60)
    print(error_text[:2000])  # Truncate if very long
  return device, error_text, torch, traceback


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **2. Anatomy of a TorchTPU Error**

    The error above has **three sections**:

    | Section | What it contains | What to look for |
    |---------|-----------------|-----------------|
    | **Python Traceback** | Standard `.py` file + line number | *Your* code that triggered the crash |
    | **C++ Error Trace** | Innermost C++ safety check | Which validation failed (e.g. `ValidateTensorByteSize`) |
    | **Captured Traceback** | Full dispatcher journey | Path from `THPVariable_ones` → `at::native::ones` → `AtenEmptyMemoryFormat` |

    > **Tip:** The C++ error trace's topmost entry is the *deepest* point — that's the specific check that failed.
    """)
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **3. Common Error Patterns (Live Demos)**

    Each cell below triggers a real error, catches it, and displays the message. This lets you recognize these patterns when they appear in your own code.

    ---

    ### **3a. Shape Mismatch**
    When tensor shapes are incompatible for an operation, the StableHLO lowering fails.
    """)
  return


@app.cell
def _(device, torch):
  # Shape Mismatch: matmul with incompatible dimensions
  a = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
  b = torch.randn(5, 3, device=device, dtype=torch.bfloat16)  # 5 ≠ 8

  try:
    c = a @ b  # Records deferred op
    _ = c.cpu()  # Materialization triggers the error
  except RuntimeError as shape_err:
    print("═" * 60)
    print("SHAPE MISMATCH ERROR")
    print("═" * 60)
    print(str(shape_err)[:1500])
    print("\n💡 Fix: Ensure inner dimensions match (a.shape[1] == b.shape[0])")
  return a, b, shape_err


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ---

    ### **3b. dtype Mismatch**
    Mixing `float32` and `int64` in operations that require matching types.
    """)
  return


@app.cell
def _(device, torch):
  # dtype mismatch: float + int operations
  float_t = torch.randn(4, 4, device=device, dtype=torch.float32)
  int_t = torch.randint(0, 10, (4, 4), device=device, dtype=torch.int64)

  try:
    result = torch.mm(float_t, int_t)  # mm requires matching dtypes
    _ = result.cpu()
  except RuntimeError as dtype_err:
    print("═" * 60)
    print("DTYPE MISMATCH ERROR")
    print("═" * 60)
    print(str(dtype_err)[:1500])
    print("\n💡 Fix: Cast tensors to the same dtype before the operation")
    print("   e.g. int_t = int_t.to(torch.float32)")
  return dtype_err, float_t, int_t


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ---

    ### **3c. The `inference_mode` Crash**
    `torch.compile` cannot capture graphs under `@torch.inference_mode`. Use `@torch.no_grad()` instead.
    """)
  return


@app.cell
def _():
  import subprocess, sys, textwrap

  # Run in a subprocess to isolate the dispatcher corruption
  # that inference_mode + torch.compile causes.
  script = textwrap.dedent("""
        import os
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        os.environ["ACCELERATOR_TYPE"] = "v6e-4"
        import torch
        device = torch.device("tpu")

        model = torch.nn.Linear(16, 16).to(device).to(torch.bfloat16)

        @torch.inference_mode()
        def bad_infer(x):
            return model(x)

        compiled = torch.compile(bad_infer, backend="tpu")
        test = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
        _ = compiled(test).cpu()
    """)

  proc = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      timeout=120,
  )

  print("═" * 60)
  print("INFERENCE_MODE + TORCH.COMPILE ERROR")
  print("═" * 60)
  # Show the last part of stderr which contains the actual error
  err_lines = proc.stderr.strip().split("\n")
  # Filter to the RuntimeError and a few lines before it
  error_start = next(
      (
          i
          for i, l in enumerate(err_lines)
          if "RuntimeError" in l or "INTERNAL ASSERT" in l
      ),
      max(0, len(err_lines) - 20),
  )
  print("\n".join(err_lines[error_start : error_start + 15]))
  print("\n💡 Fix: Replace @torch.inference_mode() with @torch.no_grad()")
  return


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **4. Diagnostic Tools (Live)**

    ### **4a. OpTracer — Find CPU Fallbacks**

    The `OpTracer` records every ATen op dispatched during execution. High counts of `aten.copy_` or `aten._to_copy` indicate CPU roundtrips ("Performance Cliffs").
    """)
  return


@app.cell
def _(device, torch):
  from torch_tpu._internal.utils import utils
  from torch_tpu._internal import sync

  # Build a small model and materialize its weights
  diag_model = (
      torch.nn.Sequential(
          torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)
      )
      .to(device)
      .to(torch.bfloat16)
  )
  sync.synchronize(list(diag_model.parameters()), wait=True)

  tracer = utils.OpTracer()
  x_diag = torch.randn(4, 32, device=device, dtype=torch.bfloat16)

  with tracer:
    out = diag_model(x_diag)
    _ = out.cpu()

  print("OpTracer Report:")
  print(tracer._pformat())
  print(
      "💡 Look for aten.copy_ or aten._to_copy — those indicate CPU fallbacks."
  )
  return diag_model, out, sync, tracer, utils, x_diag


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ---

    ### **4b. `EagerMode.DEFER_NEVER` — Pinpoint the Exact Failing Line**

    In normal deferred mode, errors surface at the *materialization point* (`.cpu()`), not the line that caused them. `EagerMode.DEFER_NEVER` forces immediate execution so the error appears on the exact culprit line.
    """)
  return


@app.cell
def _(device, torch):
  from torch_tpu._internal import execution_mode as em

  print("Running with EagerMode.DEFER_NEVER (eager mode)...")
  print("Each operation materializes immediately.\n")

  with em.eager_mode(em.EagerMode.DEFER_NEVER):
    p = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    q = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    r = p + q  # Executes immediately in NEVER mode
    print(f"p + q completed eagerly: shape={r.shape}")

    # In this mode, an error would appear on THIS line, not later at .cpu()
    s = r @ q  # Also immediate
    print(f"r @ q completed eagerly: shape={s.shape}")

  print(
      "\n✅ EagerMode.DEFER_NEVER forces immediate execution — useful for"
      " NaN/Inf debugging."
  )
  print("⚠️  Much slower than deferred mode. Use only for debugging.")
  return em, p, q, r, s


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    ## **5. Diagnostic Quick Reference**

    | Goal | Tool / Flag | Output |
    | :--- | :--- | :--- |
    | Rich C++ traces | `TORCH_SHOW_CPP_STACKTRACES=1` | Full C++ backtrace in `RuntimeError` |
    | Find graph breaks | `torch._dynamo.explain(model)(input)` | Human-readable explanation |
    | Graph break debugging | `TORCH_LOGS="+dynamo"` | Detailed trace (run from terminal) |
    | OOM line attribution | `EagerMode.DEFER_NEVER` context | OOM on exact culprit line |
    | Fallback detection | `OpTracer` | High `aten.copy` counts = CPU fallbacks |
    | HLO inspection | `XLA_FLAGS="--xla_dump_hlo_as_text ..."` | Post-optimization graph as text |

    ---

    ## **Debugging Levels**

    ```
    Level 1: Python Traceback         → Always available (default)
    Level 2: + C++ Stack Traces       → Set TORCH_SHOW_CPP_STACKTRACES=1
    Level 3: + ATen Op Audit          → Use OpTracer or utils.format_model
    Level 4: + HLO/LLO Dump          → Set XLA_FLAGS and LIBTPU_INIT_ARGS
    Level 5: + Eager Mode Pinpoint    → Use EagerMode.DEFER_NEVER context
    ```

    > [!NOTE]
    > For **bug reports** to the TorchTPU team, always include Level 1 + Level 2 output.
    """)
  return


if __name__ == "__main__":
  app.run()
