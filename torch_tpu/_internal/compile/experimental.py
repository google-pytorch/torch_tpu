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

"""Experimental compilation utilities for TorchTPU."""

from collections.abc import Callable, Sequence
import contextlib
import time
from typing import Any

from absl import logging
import torch


# pylint: disable=g-bad-exception-name
class AsyncCompilationSubmitted(Exception):
  """Control-flow signal to force AOTAutograd to skip execution after compile.

  This prevents blocking immediately following submission, which would negate
  any benefits of asynchronous compilation.

  Attributes:
    artifact: The CompiledArtifact (or AsyncCompiledArtifact) produced by the
      compilation that was just submitted.
  """

  def __init__(self, artifact: Any = None):
    super().__init__()
    self.artifact = artifact

  def resolve(self) -> None:
    """Waits for the pending background compilation to complete."""
    if self.artifact is not None:
      resolve_fn = getattr(self.artifact, "_resolve", None)
      if callable(resolve_fn):
        resolve_fn()


def resolve_compilations(artifacts: Sequence[Any]) -> int:
  """Waits for a sequence of pending compilation artifacts to finish compiling.

  Args:
    artifacts: A sequence of compilation artifacts or AsyncCompilationSubmitted
      exceptions.

  Returns:
    The number of compilation artifacts that were resolved.
  """
  resolved_count = 0
  for item in artifacts:
    if isinstance(item, AsyncCompilationSubmitted):
      item.resolve()
      resolved_count += 1
    else:
      resolve_fn = getattr(item, "_resolve", None)
      if callable(resolve_fn):
        resolve_fn()
        resolved_count += 1
  return resolved_count


def _unpack_warmup_args(
    args: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Unpacks warmup argument item into positional tuple and keyword dictionary."""
  if isinstance(args, tuple):
    if (
        len(args) == 2
        and isinstance(args[0], tuple)
        and isinstance(args[1], dict)
    ):
      return args[0], args[1]
    return args, {}
  elif isinstance(args, dict):
    return (), args
  elif isinstance(args, list):
    return (args,), {}
  else:
    return (args,), {}


def async_compile(
    fn: Callable[..., Any],
    warmup_inputs: Sequence[Any],
    *,
    no_grad: bool = True,
    **backend_kwargs: Any,
) -> Callable[..., Any]:
  """Compiles `fn` across multiple static shapes asynchronously, overlapping Dynamo tracing with XLA compilation.

  Args:
    fn: The PyTorch callable or nn.Module to compile.
    warmup_inputs: A sequence of argument tuples or keyword dictionaries
      representing the different static shapes/buckets to compile. Note:
      multiple positional arguments must be passed as tuples (e.g., `[(x, y),
      (z, w)]`); passing a list (e.g., `[[t1, t2]]`) is treated as a single
      positional list argument `fn([t1, t2])`.
    no_grad: Whether to wrap execution with `torch.no_grad()`. Defaults to True
      for inference workloads. Set to False when gradients are required.
    **backend_kwargs: Additional keyword arguments passed to `TpuBackend`.

  Returns:
    The compiled callable ready for execution.
  """
  target_fn = torch.no_grad()(fn) if no_grad else fn
  compiled = torch.compile(
      target_fn,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={
          "experimental_async_compile": True,
          **backend_kwargs,
      },
  )

  warmup_calls = 0
  warmup_start = time.perf_counter()
  artifacts = []
  grad_ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
  with grad_ctx:
    for args in warmup_inputs:
      pos_args, kw_args = _unpack_warmup_args(args)
      try:
        compiled(*pos_args, **kw_args)
      except AsyncCompilationSubmitted as e:
        if e.artifact is not None:
          artifacts.append(e.artifact)
        warmup_calls += 1

  lowering_seconds = time.perf_counter() - warmup_start
  logging.info(
      "Submitted %d warmup calls in %.3fs",
      warmup_calls,
      lowering_seconds,
  )

  wait_start = time.perf_counter()
  resolved_count = resolve_compilations(artifacts)
  wait_seconds = time.perf_counter() - wait_start
  logging.info(
      "Warmup barrier resolved %d compilation artifacts in %.3fs",
      resolved_count,
      wait_seconds,
  )

  return compiled
  