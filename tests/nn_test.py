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

from absl.testing import absltest
import numpy as np
import torch
import torch.accelerator
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tpu._internal.utils import utils


CheckValueMode = utils.CheckValueMode


def conv1d_gen(rng, dtype):
  in_channels = rng.integers(1, 5)
  out_channels = rng.integers(1, 5)
  kernel_size = rng.integers(1, 5)
  dilation = rng.integers(1, 4)
  min_dim = kernel_size * dilation
  b = rng.integers(1, 4)
  l = rng.integers(min_dim, 4 * min_dim)
  bias = rng.integers(0, 2) == 1
  stride = rng.integers(1, max(2, l))
  padding = rng.integers(0, 4)
  groups = rng.integers(1, 4)
  if groups == 2:
    in_channels = groups * in_channels
    out_channels = groups * out_channels
  elif groups == 3:
    groups = in_channels
    out_channels = groups * out_channels
  sample_input = torch.randn([b, in_channels, l]).to(dtype)
  return {
      "_model_name": "torch.nn.Conv1d",
      "_model_builder": torch.nn.Conv1d,
      "_sample_input": sample_input,
      "args": {
          "in_channels": in_channels,
          "out_channels": out_channels,
          "kernel_size": kernel_size,
          "stride": stride,
          "padding": padding,
          "dilation": dilation,
          "bias": bias,
          "groups": groups,
          "dtype": dtype,
      },
      "tolerance": {
          torch.float32: {"rtol": 1e-5, "atol": 1e-02},
          torch.float16: {"rtol": 1e-3, "atol": 9.8e-03},
          torch.bfloat16: {"rtol": 1e-2, "atol": 9.8e-03},
      },
  }


def conv2d_gen(rng, dtype):
  in_channels = rng.integers(1, 5)
  out_channels = rng.integers(1, 5)
  kernel_size = rng.integers(1, 5)
  dilation = rng.integers(1, 4)
  min_dim = kernel_size * dilation
  b = rng.integers(1, 4)
  h = rng.integers(min_dim, 4 * min_dim)
  w = rng.integers(min_dim, 4 * min_dim)
  bias = rng.integers(0, 2) == 1
  stride = rng.integers(1, max(2, min(h, w)))
  padding = rng.integers(0, 4)
  groups = rng.integers(1, 4)
  if groups == 2:
    in_channels = groups * in_channels
    out_channels = groups * out_channels
  elif groups == 3:
    groups = in_channels
    out_channels = groups * out_channels
  sample_input = torch.randn([b, in_channels, h, w]).to(dtype)
  return {
      "_model_name": "torch.nn.Conv2d",
      "_model_builder": torch.nn.Conv2d,
      "_sample_input": sample_input,
      "args": {
          "in_channels": in_channels,
          "out_channels": out_channels,
          "kernel_size": kernel_size,
          "stride": stride,
          "padding": padding,
          "dilation": dilation,
          "bias": bias,
          "groups": groups,
          "dtype": dtype,
      },
      "tolerance": {
          torch.float32: {"rtol": 1e-5, "atol": 5.3e-03},
          torch.float16: {"rtol": 1e-3, "atol": 1e-2},
          torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
      },
  }


def conv3d_gen(rng, dtype):
  in_channels = rng.integers(1, 5)
  out_channels = rng.integers(1, 5)
  kernel_size = rng.integers(1, 5)
  dilation = rng.integers(1, 4)
  min_dim = kernel_size * dilation
  b = rng.integers(1, 4)
  d = rng.integers(min_dim, 4 * min_dim)
  h = rng.integers(min_dim, 4 * min_dim)
  w = rng.integers(min_dim, 4 * min_dim)
  bias = rng.integers(0, 2) == 1
  stride = rng.integers(1, max(2, min(d, h, w)))
  padding = rng.integers(0, 4)
  groups = rng.integers(1, 4)
  if groups == 2:
    in_channels = groups * in_channels
    out_channels = groups * out_channels
  elif groups == 3:
    groups = in_channels
    out_channels = groups * out_channels
  sample_input = torch.randn([b, in_channels, d, h, w]).to(dtype)
  return {
      "_model_name": "torch.nn.Conv3d",
      "_model_builder": torch.nn.Conv3d,
      "_sample_input": sample_input,
      "args": {
          "in_channels": in_channels,
          "out_channels": out_channels,
          "kernel_size": kernel_size,
          "stride": stride,
          "padding": padding,
          "dilation": dilation,
          "bias": bias,
          "groups": groups,
          "dtype": dtype,
      },
      "tolerance": {
          torch.float32: {"rtol": 1e-5, "atol": 1e-02},
          torch.float16: {"rtol": 1e-3, "atol": 1.2e-02},
          torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
      },
  }


def relu_gen(rng, dtype):
  dim = rng.integers(1, 10)
  inplace = rng.integers(0, 2) == 1
  sample_input = torch.randn([dim]).to(dtype)
  return {
      "_model_name": "torch.nn.ReLU",
      "_model_builder": torch.nn.ReLU,
      "_sample_input": sample_input,
      "args": {
          "inplace": inplace,
      },
  }


def batchnorm2d_gen(rng, dtype):
  b = rng.integers(1, 4)
  h = rng.integers(1, 10)
  w = rng.integers(1, 10)
  num_features = rng.integers(1, 10)
  eps = rng.uniform(1e-3, 1e-2)
  momentum = rng.uniform(1e-2, 1e-1)
  affine = rng.integers(0, 2) == 1
  track_running_stats = rng.integers(0, 2) == 1
  sample_input = torch.randn([b, num_features, h, w]).to(dtype)
  return {
      "_model_name": "torch.nn.BatchNorm2d",
      "_model_builder": torch.nn.BatchNorm2d,
      "_sample_input": sample_input,
      "args": {
          "num_features": num_features,
          "eps": eps,
          "momentum": momentum,
          "affine": affine,
          "track_running_stats": track_running_stats,
          "dtype": dtype,
      },
      "tolerance": {
          torch.float32: {"rtol": 1e-5, "atol": 1e-5},
          torch.float16: {"rtol": 1e-1, "atol": 1e-1},
          torch.bfloat16: {"rtol": 5e-1, "atol": 5e-1},
      },
  }


ops_db = [
    conv1d_gen,
    conv2d_gen,
    conv3d_gen,
    relu_gen,
    batchnorm2d_gen,
]


class TestNn(TestCase):
  """Test methods in torch.nn."""

  num_runs_per_test = 10

  @dtypes(torch.float32, torch.float16, torch.bfloat16)
  def test(self, device, dtype):
    seed = torch.initial_seed()
    rng = np.random.default_rng(seed=seed)
    for op_config_generator in ops_db:
      for _ in range(self.num_runs_per_test):
        config = op_config_generator(rng, dtype)
        self._run_eval(config, dtype)

  def _run_eval(self, kwargs, dtype):
    model_name = kwargs["_model_name"]
    model_builder = kwargs["_model_builder"]
    sample_input = kwargs["_sample_input"]
    args = kwargs["args"]
    tolerance_dict = kwargs.get("tolerance", {})
    tolerance = tolerance_dict.get(dtype, {})
    tpu_result_to_cpu = None
    tpu_result_to_cpu_i = None

    print(
        f">>> Testing {model_name}, dtype: {dtype}, args={args},"
        f" tolerance={tolerance}",
        flush=True,
    )
    cpu_model = model_builder(**args)
    cpu_model.eval()

    cpu_result = cpu_model(sample_input)

    tpu_d = torch.accelerator.current_accelerator()
    tpu_model = model_builder(**args)
    tpu_model.load_state_dict(cpu_model.state_dict())
    tpu_model.to(tpu_d)
    tpu_model.eval()

    try:
      tpu_sample_input = torch.clone(sample_input).to(tpu_d)
      assert tpu_sample_input.device.type == "tpu"
      tpu_result = tpu_model(tpu_sample_input)

      assert tpu_result.device.type == "tpu"
      tpu_result_to_cpu = tpu_result.cpu()

      assert isinstance(cpu_result, type(tpu_result))
      if isinstance(tpu_result, torch.Tensor):
        tpu_result_to_cpu = tpu_result.to("cpu")
        utils.assert_close(
            cpu_result,
            tpu_result_to_cpu,
            check_value=CheckValueMode.LOOSE,
            **tolerance,
        )
      elif isinstance(tpu_result, tuple):
        assert len(cpu_result) == len(tpu_result)
        for i in range(len(cpu_result)):
          cpu_result_i = cpu_result[i]
          tpu_result_i = tpu_result[i]
          tpu_result_to_cpu_i = tpu_result_i.to("cpu")
          utils.assert_close(
              cpu_result_i,
              tpu_result_to_cpu_i,
              check_value=CheckValueMode.LOOSE,
              **tolerance,
          )

    except Exception as e:
      print(f"Test {model_name}, dtype: {dtype} FAILED with exception: {e}")
      print(f"sample_input: {sample_input}")
      print(f"cpu_result={cpu_result}")
      if tpu_result_to_cpu is not None:
        print(f"tpu_result_to_cpu={tpu_result_to_cpu}")
      elif tpu_result_to_cpu_i is not None:
        print(f"cpu_result_i={cpu_result_i}")
        print(f"tpu_result_to_cpu_i={tpu_result_to_cpu_i}")
      raise e


instantiate_device_type_tests(TestNn, globals(), only_for={"cpu"})

if __name__ == "__main__":
  # Call absltest.main even we do not use absl for testing.
  # It ensures the log prints correctly.
  absltest.main()

  # Initialize seed with the current system time
  torch.manual_seed(time.time())
  # Uncomment to set a specific seed value.
  #  torch.manual_seed(1234)
  print(f"Torch initial seed: {torch.initial_seed()}")

  TestCase._default_dtype_check_enabled = True
  run_tests()
