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

"""Integration tests for TPU Autograd tracing and numerical parity.

This suite validates that compiled models on TPU produce loss and gradients
consistent with a CPU reference. It specifically tests the interaction between
`torch.compile`, the `TpuBackend`, and Dynamo's `trace_autograd_ops`
configuration
to ensure backward passes are correctly handled whether they are internal or
external to the compiled graph.
"""

import copy
import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal import sync
from torch_tpu._internal.utils import utils


@dataclasses.dataclass
class RunConfig:
  model: torch.nn.Module
  inputs: torch.Tensor
  device: str
  include_bwd: bool
  should_compile: bool


def create_model_and_input(
    seed: int = 1234,
) -> tuple[torch.nn.Module, torch.Tensor]:
  torch.manual_seed(seed)
  batch_size = 1
  seq_len = 128
  features = 128
  dtype = torch.bfloat16

  model = torch.nn.Sequential(
      torch.nn.Linear(features, features, dtype=dtype),
      torch.nn.LayerNorm(features, dtype=dtype),
      torch.nn.ReLU(),
  )

  inputs = torch.randn(
      (batch_size, seq_len, features),
      dtype=dtype,
  )

  return (model, inputs)


def sync_device(
    device: str | torch.device, tensor_to_sync: torch.Tensor, wait: bool = True
) -> None:
  if device == 'cpu':
    torch.cpu.synchronize()
  else:
    sync.synchronize(tensor_to_sync, wait=wait)


def sync_loss_and_grads(
    device: str | torch.device, loss_val: torch.Tensor, model: torch.nn.Module
) -> None:
  sync_device(device, loss_val)
  for _, p in model.named_parameters():
    if p.grad is not None:
      sync_device(device, p.grad)


def all_tensors_are_close_to_reference(
    tensor: torch.Tensor, reference_tensor: torch.Tensor
) -> None:
  utils.assert_close(
      tensor,
      reference_tensor,
      check_value=utils.CheckValueMode.LOOSE,
      rtol=3e-2,
      atol=5e-5,
  )


def execute_training_step(config: RunConfig) -> torch.Tensor:
  model = config.model
  inputs = config.inputs
  device = config.device
  include_bwd = config.include_bwd
  should_compile = config.should_compile

  inputs_on_device = inputs.to(device)
  model_on_device = model.to(device)
  model_on_device.train()
  model_on_device.zero_grad(set_to_none=True)

  # Definitions allow toggling whether .backward() is part of the graph or
  # handled by standard Autograd outside the backend.
  def train_step(x: torch.Tensor) -> torch.Tensor:
    y_pred = model_on_device(x)
    loss = torch.mean(y_pred)
    loss.backward()
    return loss.detach()

  def train_step_no_bwd(x: torch.Tensor) -> torch.Tensor:
    y_pred = model_on_device(x)
    loss = torch.mean(y_pred)
    return loss

  selected_fn = train_step if include_bwd else train_step_no_bwd

  if should_compile:
    runnable_model = torch.compile(
        selected_fn,
        dynamic=False,
        backend=torch_tpu_compile.TpuBackend(),
    )
  else:
    runnable_model = selected_fn

  loss_val = runnable_model(inputs_on_device)

  # If backward wasn't in the compiled graph, run it now
  if not include_bwd:
    loss_val.backward()

  sync_device(device, loss_val)
  for _, p in model.named_parameters():
    if p.grad is not None:
      sync_device(device, p.grad)

  return loss_val


class TraceAutogradOpsTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tpu_device = torch.device('tpu')
    cls.base_model, cls.inputs = create_model_and_input()
    # Establish a 'Ground Truth' on CPU to compare TPU results against
    cls.cpu_config = RunConfig(
        cls.base_model,
        cls.inputs,
        'cpu',
        include_bwd=False,
        should_compile=False,
    )
    cls.cpu_config.loss = execute_training_step(cls.cpu_config)  # pyrefly: ignore[missing-attribute]

  @parameterized.named_parameters(
      dict(
          testcase_name='test_trace_autograd_ops_true_include_bwd',
          trace_autograd_ops=True,
          include_bwd=True,
      ),
      dict(
          testcase_name='test_trace_autograd_ops_true_exclude_bwd',
          trace_autograd_ops=True,
          include_bwd=False,
      ),
      dict(
          testcase_name='test_trace_autograd_ops_false_include_bwd',
          trace_autograd_ops=False,
          include_bwd=True,
      ),
      dict(
          testcase_name='test_trace_autograd_ops_false_exclude_bwd',
          trace_autograd_ops=False,
          include_bwd=False,
      ),
  )
  @absltest.skipIf(
      not hasattr(torch._dynamo.config, 'trace_autograd_ops'),
      'Test requires the trace_autograd_ops flag in Dynamo.',
  )
  def test_trace_autograd_ops(
      self, trace_autograd_ops: bool, include_bwd: bool
  ):
    tpu_config = RunConfig(
        copy.deepcopy(self.base_model),
        self.inputs.clone(),
        self.tpu_device,
        include_bwd=include_bwd,
        should_compile=True,
    )

    with torch._dynamo.config.patch(trace_autograd_ops=trace_autograd_ops):
      tpu_config.loss = execute_training_step(tpu_config)  # pyrefly: ignore[missing-attribute]

    all_tensors_are_close_to_reference(
        tpu_config.loss,  # pyrefly: ignore[missing-attribute]
        self.cpu_config.loss,  # pyrefly: ignore[missing-attribute]
    )

    for name, cpu_param in self.cpu_config.model.named_parameters():
      cpu_grad = cpu_param.grad
      self.assertIsNotNone(cpu_grad)

      all_tensors_are_close_to_reference(
          tpu_config.model.get_parameter(name).grad,
          cpu_grad,
      )


if __name__ == '__main__':
  absltest.main()
