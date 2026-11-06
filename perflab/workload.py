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

"""The workload of a benchmark: the batch it runs, for throughput metrics.

Step time alone only compares runs that do the same work per step. Throughput
(samples or tokens per second) also compares runs at different batch sizes, e.g.
a TPU and a GPU run of a benchmark that each run at their best batch size.

The workload is a parameter of a benchmark, like its stepper: each benchmark
declares the `Workload` it runs at by default (`BenchmarkSpec.workload`), the
harness resolves the workload of each case (the default, or another batch size
in a batch sweep) and passes it to the factory as `Context.workload`, and the
factory builds its inputs from it. The harness then knows the work of each step
without inspecting the inputs: `step_work` scales the workload by the number of
forward passes of a step of the benchmark's stepper.

This module deliberately does not import torch, so that metrics can depend on it
without depending on torch.
"""

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from typing import Any

from perflab import step_lib

# The input keyword arguments that hold the token ids or embeddings of a token
# based model. Their first two dimensions are the batch and sequence dimensions.
_TOKEN_INPUT_KEYS = ("input_ids", "inputs_embeds")


@dataclasses.dataclass(frozen=True)
class Workload:
  """The inputs of one forward pass of a benchmark, per process.

  Attributes:
    batch_size: The number of samples (sequences, images, audio clips, ...) that
      one forward pass processes.
    seq_len: The number of tokens per sample of a token based model, or None if
      the model does not process tokens (e.g. an image classifier) or the
      benchmark does not declare them.
  """

  batch_size: int
  seq_len: int | None = None

  def __post_init__(self):
    if self.batch_size < 1:
      raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
    if self.seq_len is not None and self.seq_len < 1:
      raise ValueError(f"seq_len must be >= 1 or None, got {self.seq_len}")

  def with_batch_size(self, batch_size: int) -> "Workload":
    """Returns this workload at another batch size."""
    return dataclasses.replace(self, batch_size=batch_size)

  def require_seq_len(self) -> int:
    """Returns the tokens per sample of a token based workload.

    Raises:
      ValueError: If the workload declares no tokens.
    """
    if self.seq_len is None:
      raise ValueError(f"workload {self} declares no seq_len")
    return self.seq_len


@dataclasses.dataclass(frozen=True)
class StepWork:
  """The work that one benchmark step does, per process.

  Attributes:
    workload: The inputs of one forward pass of the step.
    samples_per_step: The samples that one step processes, or None if a sample
      count is not meaningful for the step (decode).
    tokens_per_step: The tokens that one step processes (prefill and training)
      or generates (decode), or None if the workload declares no tokens.
  """

  workload: Workload
  samples_per_step: int | None
  tokens_per_step: int | None


def step_work(
    workload: Workload,
    stepper: step_lib.StepperType,
    stepper_kwargs: Mapping[str, Any],
) -> StepWork:
  """Returns the work that one step of a benchmark does.

  Args:
    workload: The inputs of one forward pass of the benchmark.
    stepper: The benchmark's stepper type.
    stepper_kwargs: The benchmark's stepper keyword arguments.
  """
  if stepper is step_lib.StepperType.DECODER_ONLY_DECODE:
    # A decode step generates `output_tokens` tokens for every sequence of the
    # batch; `seq_len` is the length of the unmeasured prefill.
    output_tokens = stepper_kwargs.get(
        "output_tokens", step_lib.DEFAULT_DECODE_OUTPUT_TOKENS
    )
    return StepWork(
        workload=workload,
        samples_per_step=None,
        tokens_per_step=workload.batch_size * output_tokens,
    )
  passes = 1
  if stepper in (
      step_lib.StepperType.TRAINING,
      step_lib.StepperType.SINGLE_TRACE_TRAINING,
  ):
    passes = stepper_kwargs.get("accum_steps", step_lib.DEFAULT_ACCUM_STEPS)
  samples = workload.batch_size * passes
  return StepWork(
      workload=workload,
      samples_per_step=samples,
      tokens_per_step=(
          None if workload.seq_len is None else samples * workload.seq_len
      ),
  )


def _iter_leaves(tree: Any) -> Iterator[Any]:
  """Yields the leaves of nested tuples, lists and mappings, in order."""
  if isinstance(tree, Mapping):
    for value in tree.values():
      yield from _iter_leaves(value)
  elif isinstance(tree, (tuple, list)):
    for value in tree:
      yield from _iter_leaves(value)
  else:
    yield tree


def _shape(value: Any) -> tuple[int, ...] | None:
  """Returns the shape of a tensor-like value, or None for other values."""
  shape = getattr(value, "shape", None)
  if shape is None:
    return None
  try:
    return tuple(int(d) for d in shape)
  except (TypeError, ValueError):
    return None


def check_built_inputs(
    workload: Workload, benchmark_args: Sequence[Any]
) -> str | None:
  """Checks that a benchmark built its inputs at its workload.

  The declared workload, not the inputs, is what the harness exports. This only
  checks that a factory honors it, e.g. in the FakeTensorMode CPU smoke test,
  which builds every benchmark: a token based workload needs token ids (or
  embeddings) of shape `(batch_size, seq_len, ...)`, or packed token ids of
  shape `(batch_size * seq_len,)`; any other workload needs an input whose
  leading dimension is `batch_size`.

  Args:
    workload: The workload that the benchmark was built at.
    benchmark_args: The output of the benchmark factory, `(model, input_args,
      input_kwargs, ...)`.

  Returns:
    Why the inputs do not match the workload, or None if they do.
  """
  input_args = benchmark_args[1] if len(benchmark_args) > 1 else ()
  input_kwargs = benchmark_args[2] if len(benchmark_args) > 2 else {}
  shapes = [
      shape
      for shape in map(_shape, _iter_leaves((input_args, input_kwargs)))
      if shape
  ]
  if workload.seq_len is not None:
    b, s = workload.batch_size, workload.seq_len
    if isinstance(input_kwargs, Mapping):
      token_shapes = [
          _shape(input_kwargs[key])
          for key in _TOKEN_INPUT_KEYS
          if key in input_kwargs
      ]
      if any(shape and shape[:2] == (b, s) for shape in token_shapes):
        return None
    if any(shape[:2] == (b, s) or shape == (b * s,) for shape in shapes):
      return None
    return (
        f"no input has shape ({b}, {s}, ...) or ({b * s},) of workload"
        f" {workload}; input shapes: {shapes}"
    )
  if any(shape[0] == workload.batch_size for shape in shapes):
    return None
  return (
      f"no input has leading dimension {workload.batch_size} of workload"
      f" {workload}; input shapes: {shapes}"
  )
