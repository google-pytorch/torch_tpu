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

"""Tests FP8 collectives on multiple TPUs.

FP8 training all-gathers FP8 weights under FSDP and all-to-alls FP8 tokens for
expert-parallel MoE dispatch. These collectives only move data, so every output
byte must match the expected input byte, including the Inf, negative zero and
subnormal encodings. NaNs only need to stay NaNs.
"""

from collections.abc import Callable
import math
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils

_FP8_DTYPES = (
    ("e4m3fn", torch.float8_e4m3fn),
    ("e5m2", torch.float8_e5m2),
)

# Every FP8 dtype, both eagerly and under torch.compile.
_FP8_DTYPES_AND_MODES = tuple(
    dict(
        testcase_name=f"{name}_{mode}",
        dtype=dtype,
        compiled=mode == "compiled",
    )
    for name, dtype in _FP8_DTYPES
    for mode in ("eager", "compiled")
)

# Every per-rank (_ROWS, _COLS) input has 512 elements, so it holds every FP8
# encoding.
_ROWS = 16
_COLS = 32


def _run(
    fn: Callable[..., Any],
    x: torch.Tensor,
    compiled: bool,
) -> Any:
  """Runs `fn(x)` eagerly or compiled on TPU.

  When `compiled=True`, passes options to `torch.compile` to inspect the
  StableHLO IR and verifies that the collective stays in the FP8 element type.

  Args:
    fn: The function to execute.
    x: The input tensor (must already be on TPU).
    compiled: Whether to compile `fn`.

  Returns:
    The output of `fn(x)`.
  """
  if not compiled:
    return fn(x)

  debugs = []
  compiled_fn = torch.compile(
      fn,
      backend="tpu",
      fullgraph=True,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  out = compiled_fn(x)

  shlo_text = "\n".join(
      mlir for debug in debugs for mlir in debug.stablehlo_forward_text
  )
  # Verify that a StableHLO collective op is present in the graph.
  collective_ops = (
      "stablehlo.all_gather",
      "stablehlo.all_to_all",
      "stablehlo.reduce_scatter",
  )
  if not any(op in shlo_text for op in collective_ops):
    raise AssertionError(
        f"Expected one of {collective_ops} in StableHLO text, got:\n{shlo_text}"
    )

  # Check that the collective stays in the FP8 element type.
  fp8_name = "f8E4M3FN" if x.dtype == torch.float8_e4m3fn else "f8E5M2"
  if fp8_name not in shlo_text:
    raise AssertionError(
        f"Expected FP8 element type {fp8_name} in StableHLO text, got:\n"
        f"{shlo_text}"
    )

  return out


def _fp8_codes(
    dtype: torch.dtype, shape: tuple[int, ...], seed: int
) -> torch.Tensor:
  """Returns an FP8 CPU tensor whose bytes cycle through all 256 encodings.

  The cycle starts at `seed`, so tensors built with different seeds hold
  different data.

  Args:
    dtype: The FP8 dtype of the result.
    shape: The shape of the result.
    seed: The encoding of the first element.
  """
  codes = (torch.arange(math.prod(shape)) + seed) % 256
  return codes.to(torch.uint8).view(dtype).reshape(shape)


def _assert_bytes_equal_except_nans(
    actual: torch.Tensor, expected: torch.Tensor, what: str
) -> None:
  """Asserts that `actual` has the dtype, shape and bytes of `expected`.

  Args:
    actual: The tensor to check.
    expected: The expected CPU tensor.
    what: Describes `actual` in the error message.
  """
  if actual.dtype != expected.dtype or actual.shape != expected.shape:
    raise AssertionError(
        f"{what}: expected {expected.dtype} {list(expected.shape)}, got"
        f" {actual.dtype} {list(actual.shape)}"
    )
  actual = actual.cpu()
  actual_bytes = actual.view(torch.uint8)
  expected_bytes = expected.view(torch.uint8)
  # TODO(amainz): Compare NaN bytes too once XLA:TPU FP8 NaN canonicalization in
  # VPU copy/concat/slice/bitcast-convert is fixed.
  # NaNs only need to stay NaNs: XLA:TPU canonicalizes FP8 NaNs in on-chip
  # copies, concatenates, slices and bitcast-converts. e4m3fn 0xff becomes 0x7f,
  # and e5m2 0x7d, 0x7f and 0xfd-0xff become 0x7e.
  both_nan = actual.float().isnan() & expected.float().isnan()
  mismatches = ((actual_bytes != expected_bytes) & ~both_nan).nonzero()
  if mismatches.numel():
    first = tuple(mismatches[0].tolist())
    raise AssertionError(
        f"{what}: {mismatches.shape[0]} of {expected.numel()} bytes differ;"
        f" first at index {first}: got {actual_bytes[first].item():#04x},"
        f" expected {expected_bytes[first].item():#04x}"
    )


def _split_matrix(world_size: int) -> list[list[int]]:
  """Returns the rows rank `src` sends to rank `dst` as `[src][dst]`.

  The splits are uneven and include empty slices, as in MoE token dispatch.
  TPU requires every rank to have the same input and output buffer sizes, so
  for a world size that is a multiple of 4, every rank sends and receives the
  same total number of rows.

  Args:
    world_size: The number of ranks.
  """
  return [
      [(src + 3 * dst) % 4 for dst in range(world_size)]
      for src in range(world_size)
  ]


def _per_rank_inputs(dtype: torch.dtype, world_size: int) -> list[torch.Tensor]:
  """Returns the `(_ROWS, _COLS)` tensor that each rank contributes."""
  return [
      _fp8_codes(dtype, (_ROWS, _COLS), seed=37 * src)
      for src in range(world_size)
  ]


def run_all_gather_into_tensor(
    dtype: torch.dtype, compiled: bool, stacked: bool = False
) -> None:
  """Tests all_gather_into_tensor, concatenating or stacking the inputs."""
  rank = dist.get_rank()
  inputs = _per_rank_inputs(dtype, dist.get_world_size())
  expected = torch.stack(inputs) if stacked else torch.cat(inputs)

  def all_gather(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty(expected.shape, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(output, x)
    return output

  output = _run(all_gather, inputs[rank].to("tpu"), compiled)

  _assert_bytes_equal_except_nans(output, expected, f"rank {rank}")


def run_all_gather(dtype: torch.dtype, compiled: bool) -> None:
  """Tests all_gather into a list of tensors."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  inputs = _per_rank_inputs(dtype, world_size)

  def all_gather(x: torch.Tensor) -> list[torch.Tensor]:
    outputs = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(outputs, x)
    return outputs

  outputs = _run(all_gather, inputs[rank].to("tpu"), compiled)

  for src in range(world_size):
    _assert_bytes_equal_except_nans(
        outputs[src], inputs[src], f"rank {rank}, output {src}"
    )


def run_all_to_all_single(
    dtype: torch.dtype, compiled: bool, uneven: bool = False
) -> None:
  """Tests all_to_all_single with equal or uneven splits."""
  # TODO(amainz): Eager only; uneven splits fail under torch.compile for every
  # dtype ("cannot Materialize() a placeholder tensor" at
  # process_group_tpu.cc:1792).
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  if uneven:
    splits = _split_matrix(world_size)
    inputs = [
        _fp8_codes(dtype, (sum(splits[src]), _COLS), seed=37 * src)
        for src in range(world_size)
    ]
    expected = torch.cat(
        [inputs[src].split(splits[src])[rank] for src in range(world_size)]
    )
    output_split_sizes = [splits[src][rank] for src in range(world_size)]
    input_split_sizes = splits[rank]

    def all_to_all_uneven(x: torch.Tensor) -> torch.Tensor:
      output = torch.empty(expected.shape, dtype=x.dtype, device=x.device)
      dist.all_to_all_single(
          output,
          x,
          output_split_sizes=output_split_sizes,
          input_split_sizes=input_split_sizes,
      )
      return output

    output = _run(all_to_all_uneven, inputs[rank].to("tpu"), compiled)
  else:
    inputs = [
        _fp8_codes(dtype, (world_size * _ROWS, _COLS), seed=37 * src)
        for src in range(world_size)
    ]
    expected = torch.cat(
        [inputs[src].chunk(world_size)[rank] for src in range(world_size)]
    )

    def all_to_all_even(x: torch.Tensor) -> torch.Tensor:
      output = torch.empty(expected.shape, dtype=x.dtype, device=x.device)
      dist.all_to_all_single(output, x)
      return output

    output = _run(all_to_all_even, inputs[rank].to("tpu"), compiled)

  _assert_bytes_equal_except_nans(output, expected, f"rank {rank}")


def run_all_to_all(dtype: torch.dtype, uniform: bool) -> None:
  """Tests all_to_all on lists of tensors with uniform or uneven shapes.

  Only runs eagerly, as torch.compile doesn't capture all_to_all.

  Args:
    dtype: The FP8 dtype to exchange.
    uniform: Whether all tensors have the same shape.
  """
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  splits = _split_matrix(world_size)

  def rows(src: int, dst: int) -> int:
    return _ROWS if uniform else splits[src][dst]

  # inputs[src][dst] is the tensor that rank `src` sends to rank `dst`.
  inputs = [
      [
          _fp8_codes(dtype, (rows(src, dst), _COLS), seed=37 * src + 11 * dst)
          for dst in range(world_size)
      ]
      for src in range(world_size)
  ]
  outputs = [
      torch.empty((rows(src, rank), _COLS), dtype=dtype, device="tpu")
      for src in range(world_size)
  ]

  dist.all_to_all(outputs, [t.to("tpu") for t in inputs[rank]])

  for src in range(world_size):
    _assert_bytes_equal_except_nans(
        outputs[src], inputs[src][rank], f"rank {rank}, output {src}"
    )


def run_reduce_scatter_tensor(dtype: torch.dtype, compiled: bool) -> None:
  """Tests reduce_scatter_tensor with a sum that is exact in FP8."""
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  # Every partial sum of ones is an integer no larger than the world size.
  # Exact sum 1..W holds for world size <= 8 (since 9 is not representable in
  # e5m2). Both FP8 formats represent integers 1..8 exactly, so the result does
  # not depend on the reduction order.
  x = torch.ones((world_size * _ROWS, _COLS), dtype=dtype)
  expected = torch.full((_ROWS, _COLS), float(world_size), dtype=dtype)

  def reduce_scatter(t: torch.Tensor) -> torch.Tensor:
    output = torch.empty(expected.shape, dtype=t.dtype, device=t.device)
    dist.reduce_scatter_tensor(output, t)
    return output

  output = _run(reduce_scatter, x.to("tpu"), compiled)

  _assert_bytes_equal_except_nans(output, expected, f"rank {rank}")


class Fp8CollectivesTest(continuous_test_utils.TpuMultiProcContinuousTest):

  world_size = 8

  @parameterized.named_parameters(*_FP8_DTYPES_AND_MODES)
  def test_all_gather_into_tensor(
      self, dtype: torch.dtype, compiled: bool
  ) -> None:
    run_all_gather_into_tensor(dtype=dtype, compiled=compiled)

  @parameterized.named_parameters(*_FP8_DTYPES)
  def test_all_gather_into_tensor_stacked(self, dtype: torch.dtype) -> None:
    run_all_gather_into_tensor(dtype=dtype, compiled=False, stacked=True)

  @parameterized.named_parameters(*_FP8_DTYPES_AND_MODES)
  def test_all_gather(self, dtype: torch.dtype, compiled: bool) -> None:
    run_all_gather(dtype=dtype, compiled=compiled)

  @parameterized.named_parameters(*_FP8_DTYPES_AND_MODES)
  def test_all_to_all_single(self, dtype: torch.dtype, compiled: bool) -> None:
    run_all_to_all_single(dtype=dtype, compiled=compiled)

  @parameterized.named_parameters(*_FP8_DTYPES)
  def test_all_to_all_single_uneven(self, dtype: torch.dtype) -> None:
    run_all_to_all_single(dtype=dtype, compiled=False, uneven=True)

  @parameterized.named_parameters(*_FP8_DTYPES)
  def test_all_to_all_uniform(self, dtype: torch.dtype) -> None:
    run_all_to_all(dtype=dtype, uniform=True)

  @parameterized.named_parameters(*_FP8_DTYPES)
  def test_all_to_all_uneven(self, dtype: torch.dtype) -> None:
    run_all_to_all(dtype=dtype, uniform=False)

  @parameterized.named_parameters(*_FP8_DTYPES_AND_MODES)
  def test_reduce_scatter_tensor(
      self, dtype: torch.dtype, compiled: bool
  ) -> None:
    run_reduce_scatter_tensor(dtype=dtype, compiled=compiled)


if __name__ == "__main__":  # We are in the parent process.
  multiprocessing.handle_test_main(absltest.main)
