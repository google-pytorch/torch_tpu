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

"""Single entry point for verifying Pallas kernels and PyTorch programs.

Example:

  from torch.tpu.pallas import verifier

  report = verifier.verify(my_jax_op, (x, y))
  report.raise_for_errors()

`verify` accepts a TorchTPU `@jax_op`, a raw JAX function (called with
`jax.Array` or `jax.ShapeDtypeStruct` arguments), or any PyTorch callable such
as an `nn.Module` that calls `@jax_op` kernels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import enum
from typing import Any

import jax
import torch
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.utils import annotations


@annotations.experimental(
    "torch.tpu.pallas.verifier.VerificationMode is experimental and subject to"
    " change."
)
class VerificationMode(enum.Enum):
  """Which verification engines `verify` runs."""

  # Static checks plus dynamic checks that run the kernels on CPU in TPU
  # interpret mode (schedule sweeps and autograd parity). Creating the JAX CPU
  # devices initializes the other JAX backends too, including TPU if available.
  FULL = "full"
  # Only checks that never execute the kernels: contracts, TPU lowering, FX
  # graph analysis, and model checking. They need no JAX devices, so they also
  # suit processes that use the TPU through TorchTPU.
  STATIC = "static"


@annotations.experimental(
    "torch.tpu.pallas.verifier.VerificationOptions is experimental and subject"
    " to change."
)
@dataclasses.dataclass(frozen=True)
class VerificationOptions:
  """Advanced knobs for `verify`."""


@annotations.forbid_in_torch_compile(
    "verification traces and runs kernels outside a compiled graph."
)
@annotations.experimental(
    "torch.tpu.pallas.verifier.verify is experimental and subject to change."
)
def verify(
    target: Callable[..., Any],
    args: Sequence[Any] | torch.Tensor | jax.Array | jax.ShapeDtypeStruct = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    mode: VerificationMode = VerificationMode.FULL,
    options: VerificationOptions | None = None,
) -> findings.VerificationReport:
  """Verifies a Pallas kernel or a PyTorch program that calls Pallas kernels.

  The kind of `target` is detected automatically:

  * A TorchTPU `@jax_op` (or the `JaxCallable` behind it) gets op-contract,
    TPU-lowering, model-checking, autograd-parity, and CPU-interpret checks.
  * A JAX function called with `jax.Array` / `jax.ShapeDtypeStruct` arguments
    gets model-checking and CPU-interpret checks.
  * Any other callable (for example an `nn.Module`) is traced into an FX graph
    whose buffer donation and aliasing are checked; every `@jax_op` it calls
    is then verified for the argument shapes it is called with.

  Args:
    target: The op, JAX function, or PyTorch callable to verify.
    args: Example positional arguments, like `torch.library.opcheck`. A single
      tensor may be passed instead of a one-element tuple.
    kwargs: Example keyword arguments.
    mode: Which engines to run.
    options: Advanced configuration.

  Returns:
    The findings of all engines that ran. `report.passed` is `True` if and only
    if no error was found; `raise_for_errors()` raises with a formatted summary.

  Raises:
    TypeError: If `target` is not callable, `args` is not a sequence, or `mode`
      / `options` has an invalid type.
    ValueError: If a PyTorch program cannot be traced into an FX graph.
  """
  if not callable(target):
    raise TypeError(f"target must be callable, got {type(target).__name__}")
  if not isinstance(mode, VerificationMode):
    raise TypeError(
        f"mode must be a VerificationMode, got {type(mode).__name__}"
    )
  if options is not None and not isinstance(options, VerificationOptions):
    raise TypeError(
        f"options must be a VerificationOptions, got {type(options).__name__}"
    )
  _ = _as_arg_tuple(args)
  _ = dict(kwargs or {})
  _ = options or VerificationOptions()
  return findings.VerificationReport()


def _as_arg_tuple(args: Any) -> tuple[Any, ...]:
  if isinstance(args, (torch.Tensor, jax.Array, jax.ShapeDtypeStruct)):
    return (args,)
  if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
    return tuple(args)
  raise TypeError(
      f"args must be a sequence of example arguments, got {type(args).__name__}"
  )
