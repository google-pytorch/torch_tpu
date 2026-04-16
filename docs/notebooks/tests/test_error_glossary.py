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

"""Tests for error_glossary.py — verifies each executable cell runs without error."""

import os
import torch


def test_cpp_stacktraces(device):
  """Cell: trigger negative dimension error, catch RuntimeError with C++ context."""
  os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

  try:
    torch.ones(-1, device=device)
    assert False, "Should have raised RuntimeError"
  except RuntimeError as e:
    error_text = str(e)
    assert len(error_text) > 0


def test_shape_mismatch(device):
  """Cell: shape mismatch in matmul, catch RuntimeError."""
  a = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
  b = torch.randn(5, 3, device=device, dtype=torch.bfloat16)

  try:
    c = a @ b
    _ = c.cpu()
    assert False, "Should have raised RuntimeError for shape mismatch"
  except RuntimeError:
    pass  # Expected


def test_dtype_mismatch(device):
  """Cell: dtype mismatch in mm, catch RuntimeError."""
  float_t = torch.randn(4, 4, device=device, dtype=torch.float32)
  int_t = torch.randint(0, 10, (4, 4), device=device, dtype=torch.int64)

  try:
    result = torch.mm(float_t, int_t)
    _ = result.cpu()
    assert False, "Should have raised RuntimeError for dtype mismatch"
  except RuntimeError:
    pass  # Expected


def test_inference_mode_crash():
  """Cell: run subprocess test for inference_mode + torch.compile incompatibility."""
  import subprocess
  import sys
  import textwrap

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
  # We expect this to fail — the test is that the subprocess completes (doesn't hang)
  # and produces an error message
  assert proc.returncode != 0 or "error" in proc.stderr.lower()


def test_optracer_diagnostic(device):
  """Cell: run OpTracer on a model for diagnostic output."""
  from torch_tpu._internal.utils import utils
  from torch_tpu._internal import sync

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

  output = tracer._pformat()
  assert output is not None and output != ""


def test_eager_mode_never(device):
  """Cell: run operations in EagerMode.DEFER_NEVER (eager mode)."""
  from torch_tpu._internal import execution_mode as em

  with em.eager_mode(em.EagerMode.DEFER_NEVER):
    p = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    q = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    r = p + q
    assert r.shape == (4, 4)

    s = r @ q
    assert s.shape == (4, 4)
