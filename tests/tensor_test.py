# Copyright 2025 Google LLC
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

import time
from typing import Any, Callable, Tuple
from absl.testing import absltest
import numpy as np
import torch
from torch.testing._internal import common_device_type
from torch.testing._internal import common_utils as torch_test_utils
from torch_tpu._internal import sync
from torch_tpu._internal.utils import utils


CheckValueMode = utils.CheckValueMode


def _copy_(rng, dtype):
  input_dims = [3, 2, 4]
  source_dims = input_dims[rng.integers(0, 3) :]
  sample_input = torch.randn(input_dims).to(dtype)
  source_tensor = torch.randn(source_dims).to(dtype)
  return {
      "test_name": "torch.Tensor.copy_",
      "func": torch.Tensor.copy_,
      "sample_input": sample_input,
      "args": {
          "other": source_tensor,
      },
  }


def _expand(rng, dtype):  # pylint: disable=unused-argument
  input_dims = [3, 1]
  expand_args = [3, 4]
  sample_input = torch.randn(input_dims).to(dtype)
  return {
      "test_name": "torch.Tensor.expand",
      "func": torch.Tensor.expand,
      "sample_input": sample_input,
      "args": {
          "size": expand_args,
      },
  }


def _floor(rng, dtype):  # pylint: disable=unused-argument
  input_dims = [3, 1]
  sample_input = torch.randn(input_dims).to(dtype)
  return {
      "test_name": "torch.Tensor.floor",
      "func": torch.Tensor.floor,
      "sample_input": sample_input,
      "args": {},
  }


def _floor_(rng, dtype):  # pylint: disable=unused-argument
  input_dims = [3, 1]
  sample_input = torch.randn(input_dims).to(dtype)
  return {
      "test_name": "torch.Tensor.floor_",
      "func": torch.Tensor.floor_,
      "sample_input": sample_input,
      "args": {},
  }


def _isfinite(rng, dtype):  # pylint: disable=unused-argument
  input_dims = [3, 1]
  p_inf = 0.3  # Total probability for any infinity
  tensor = torch.randn(input_dims, requires_grad=False)

  pos_inf_mask = torch.rand_like(tensor) < (p_inf / 2)
  neg_inf_mask = torch.rand_like(tensor) < (p_inf / 2)

  tensor[pos_inf_mask] = torch.inf
  tensor[neg_inf_mask & ~pos_inf_mask] = (
      -torch.inf
  )  # Use ~ to invert pos_inf_mask

  return {
      "test_name": "torch.Tensor.isfinite",
      "func": torch.Tensor.isfinite,
      "sample_input": tensor.to(dtype),
      "args": {},
  }


def _random_(rng, dtype):

  def _run_random(*args, **kwargs):
    tensor = torch.Tensor.random_(*args, **kwargs)
    device = tensor.device
    mean = tensor.to("cpu").mean()  # Not implemented on TPU
    std = torch.std(tensor.to("cpu"))  # Not implemented on TPU
    return (mean + std).to(device)

  range_size = 50
  max_val = 100
  from_val = rng.integers(0, max_val - range_size)
  to_val = rng.integers(from_val + 1, from_val + range_size)

  args = {}
  if rng.integers(2):
    if rng.integers(2):
      args = {"to": to_val}
  else:
    args = {"from": from_val, "to": to_val}

  sample_input = torch.empty(1000).to(dtype)
  atol = max(1.0, args.get("from", float(1 << 24)) * 1e-2)
  return {
      "test_name": "torch.Tensor.random",
      "func": _run_random,
      "sample_input": sample_input,
      "args": args,
      "tolerance": {
          torch.float32: {"rtol": 2e-1, "atol": atol},
          torch.float16: {"rtol": 2e-1, "atol": atol},
          torch.bfloat16: {"rtol": 2e-1, "atol": atol},
      },
  }


ops = [
    # go/keep-sorted start
    _copy_,
    _expand,
    _floor,
    _floor_,
    _isfinite,
    _random_,
    # go/keep-sorted end
]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
  """Reshapes a tensor for broadcast. Used for rotary embeddings.

  Args:
    freqs_cis: Frequency-based complex exponential values for rotary positional
      embeddings.
    x: Input tensor to apply rotary positional embeddings to.

  Returns:
    Reshaped freqs_cis that can be broadcasted with x.
  """

  ndim = x.ndim
  assert 1 <= ndim
  assert freqs_cis.shape == (x.shape[1], x.shape[-1])
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies rotary positional embeddings to the input tensors.

  This is the original implementation in Llama3 8B.

  Args:
    xq: Input query tensor.
    xk: Input key tensor.
    freqs_cis: Frequency-based complex exponential values for rotary positional
      embeddings.

  Returns:
    Rotated query and key tensors.
  """

  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_xq(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
  """Applies rotary positional embeddings to the input query tensor.

  This is the query-only half of apply_rotary_emb.

  Args:
    xq: Input query tensor.
    freqs_cis: Frequency-based complex exponential values for rotary positional
      embeddings.

  Returns:
    Rotated query tensor.
  """

  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq)


def apply_rotary_float_half(x: torch.Tensor, freqs_cis: torch.Tensor):
  """Applies rotary embeddings without the use of complex numbers.

  This is yho@google.com's implementation.

  Args:
    x: Input tensor.
    freqs_cis: Frequency-based complex exponential values for rotary positional
      embeddings.

  Returns:
    Rotated tensor.
  """

  # Reshape for pairing dimensions.
  x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
  x_r, x_i = x_reshaped.unbind(dim=-1)

  # Get freqs as real and imag, and reshape for broadcast.
  freqs_cos = reshape_for_broadcast(freqs_cis.real, x_r)
  freqs_sin = reshape_for_broadcast(freqs_cis.imag, x_r)

  # Apply rotation.
  x_out_r = x_r * freqs_cos - x_i * freqs_sin
  x_out_i = x_r * freqs_sin + x_i * freqs_cos

  # Recombine and flatten.
  x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(3)

  return x_out.type_as(x)


def apply_rotary_emb_float(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies rotary embeddings without the use of complex numbers.

  Args:
    xq: Input query tensor.
    xk: Input key tensor.
    freqs_cis: Frequency-based complex exponential values for rotary positional
      embeddings.

  Returns:
    A tuple containing the rotated query and key tensors.
  """

  xq_out = apply_rotary_float_half(xq, freqs_cis)
  xk_out = apply_rotary_float_half(xk, freqs_cis)
  return xq_out, xk_out


def rand_complex(
    *shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
  """Returns a random complex tensor.

  Args:
    *shape: Shape of the tensor.
    device: Device to allocate the tensor on.

  Returns:
    A complex tensor with random real and imaginary parts.
  """

  # XLA doesn't support rand() with complex dtypes, so we piece together
  # the real and imaginary parts.
  real = torch.rand(*shape, dtype=torch.float32, device=device)
  imag = torch.rand(*shape, dtype=torch.float32, device=device)
  return torch.complex(real, imag)


def materialize(*tensors: Tuple[torch.Tensor, ...]) -> None:
  """Materializes the given tensors synchronously.

  This does not transfer the tensors to the host.

  Args:
    *tensors: Tensors to materialize.
  """

  for t in tensors:
    sync.synchronize(t, wait=True)


def make_materialized_rope_inputs() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
  """Returns materialized random inputs for rotary positional embeddings.

  The tensor dtypes and shapes are chosen to match the Llama3 8B model.
  """

  tpu = torch.accelerator.current_accelerator()
  xq = torch.rand(1, 2024, 4, 128, dtype=torch.bfloat16, device=tpu)
  xk = torch.rand(1, 2024, 1, 128, dtype=torch.bfloat16, device=tpu)
  freqs_cis = rand_complex(2024, 64, device=tpu)
  materialize(xq, xk, freqs_cis)
  return xq, xk, freqs_cis


def print_avg_duration(
    label: str,
    func: Callable[..., Any],
    num_preheat_runs: int = 10,
    num_runs: int = 100,
) -> float:
  """Prints the average duration of running the function.

  Args:
    label: Label to print with the duration.
    func: Function to run.
    num_preheat_runs: Number of times to run the function before timing.
    num_runs: Number of times to run the function after timing starts.

  Returns:
    The average duration of running the function (after preheat) in
    milliseconds.
  """

  for _ in range(num_preheat_runs):
    func()

  start_time = time.time()
  for _ in range(num_runs):
    func()
  duration_ms = (time.time() - start_time) * 1000
  avg_duration_ms = duration_ms / num_runs
  print(f"Time per run for {label}: {avg_duration_ms:.3f} ms")
  return avg_duration_ms


class TensorTest(torch_test_utils.TestCase):
  """Test methods in torch.Tensor."""

  num_runs_per_test = 10

  def test_rope_shlo(self):
    xq, xk, freqs_cis = make_materialized_rope_inputs()
    print("SHLO for apply_rotary_emb:")
    print(utils.format_model(apply_rotary_emb, xq, xk, freqs_cis, shlo=True))
    print_avg_duration(
        "apply_rotary_emb",
        lambda: materialize(*apply_rotary_emb(xq, xk, freqs_cis)),
    )

  def test_rope_q_shlo(self):
    xq, _, freqs_cis = make_materialized_rope_inputs()
    print("SHLO for apply_rotary_emb_xq:")
    print(utils.format_model(apply_rotary_emb_xq, xq, freqs_cis, shlo=True))
    print_avg_duration(
        "apply_rotary_emb_xq",
        lambda: materialize(apply_rotary_emb_xq(xq, freqs_cis)),
    )

  def test_rope_float_shlo(self):
    xq, xk, freqs_cis = make_materialized_rope_inputs()
    print("SHLO for apply_rotary_emb_float:")
    print(
        utils.format_model(apply_rotary_emb_float, xq, xk, freqs_cis, shlo=True)
    )
    print_avg_duration(
        "apply_rotary_emb_float",
        lambda: materialize(*apply_rotary_emb_float(xq, xk, freqs_cis)),
    )

  def test_rope_float_q_shlo(self):
    xq, _, freqs_cis = make_materialized_rope_inputs()
    print("SHLO for apply_rotary_float_xq:")
    print(utils.format_model(apply_rotary_float_half, xq, freqs_cis, shlo=True))
    print_avg_duration(
        "apply_rotary_emb_float_xq",
        lambda: materialize(apply_rotary_float_half(xq, freqs_cis)),
    )

  @common_device_type.dtypes(torch.float32, torch.float16, torch.bfloat16)
  def test(self, dtype):
    seed = torch.initial_seed()
    rng = np.random.default_rng(seed=seed)
    for op in ops:
      for _ in range(self.num_runs_per_test):
        kwargs = op(rng, dtype)
        self._test(dtype, kwargs)

  def _test(self, dtype, kwargs):
    tpu_d = torch.accelerator.current_accelerator()

    test_name = kwargs["test_name"]
    sample_input = kwargs.get("sample_input", None)
    func = kwargs["func"]
    args = kwargs["args"]
    tolerance_dict = kwargs.get("tolerance", {})
    tolerance = tolerance_dict.get(dtype, {})

    print(
        f">>> Testing {test_name}, dtype: {dtype}, args={args},"
        f" tolerance={tolerance}",
        flush=True,
    )
    cpu_result = None
    try:
      if sample_input is None:
        cpu_result = func(**args)
      else:
        cpu_input = torch.clone(sample_input)
        cpu_result = func(cpu_input, **args)
    except Exception as e:  # pylint: disable=broad-except
      print(f"Test {test_name}, dtype: {dtype} FAILED with exception: {e}")
      print(f"sample_input: {sample_input}")
      if cpu_result is not None:
        print(f"cpu_result: {cpu_result}")
      raise e

    tuple_i = None
    tpu_result_cpu = None
    cpu_result_i = None
    tpu_result_cpu_i = None

    # Move tensors in args to TPU.
    for name, value in args.items():
      if torch.is_tensor(value):
        args[name] = torch.clone(value).to(tpu_d)

    try:
      if sample_input is None:
        tpu_result = func(**args)
      else:
        tpu_input = torch.clone(sample_input).to(tpu_d)
        assert tpu_input.device.type == "tpu"
        tpu_result = func(tpu_input, **args)

      assert tpu_result.device.type == "tpu"

      assert isinstance(tpu_result, type(cpu_result))
      if isinstance(tpu_result, torch.Tensor):
        tpu_result_cpu = tpu_result.cpu()
        utils.assert_close(
            actual=tpu_result_cpu,
            expected=cpu_result,
            check_value=CheckValueMode.LOOSE,
            **tolerance,
        )
      elif isinstance(tpu_result, tuple):
        assert len(cpu_result) == len(tpu_result)
        for i in range(len(cpu_result)):
          tuple_i = i
          cpu_result_i = cpu_result[i]
          tpu_result_i = tpu_result[i]
          tpu_result_cpu_i = tpu_result_i.cpu()
          utils.assert_close(
              actual=tpu_result_cpu_i,
              expected=cpu_result_i,
              **tolerance,
          )
      print("Test PASSED")

    except Exception as e:  # pylint: disable=broad-except
      print(f"Test {test_name}, dtype: {dtype} FAILED with exception: {e}")
      print(f"sample_input: {sample_input}")
      if tuple_i is not None:
        # Failure happened during tuple unpacking.
        print(f"tuple_i={tuple_i}")
        if cpu_result_i is not None:
          print(f"cpu_result_i={cpu_result_i}")
        if tpu_result_cpu_i is not None:
          print(f"tpu_result_cpu_i={tpu_result_cpu_i}")
      elif tpu_result_cpu is not None:
        print(f"cpu_result={cpu_result}")
        print(f"tpu_result_cpu={tpu_result_cpu}")
      raise e


common_device_type.instantiate_device_type_tests(
    TensorTest, globals(), only_for={"cpu"}
)

if __name__ == "__main__":
  # Call absltest.main even we do not use absl for testing.
  # It ensures the log prints correctly.
  absltest.main()

  # Initialize seed with the current system time
  torch.manual_seed(time.time())
  # Uncomment to set a specific seed value.
  #  torch.manual_seed(1234)
  print(f"Torch initial seed: {torch.initial_seed()}")

  torch_test_utils.TestCase._default_dtype_check_enabled = True
  torch_test_utils.run_tests()
