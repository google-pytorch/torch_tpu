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

"""Unit tests for DeepSeek layers.

These tests focus on single TPU mode.
"""

import copy
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import log_utils
from torch_tpu._internal.utils import utils
from examples.deepseek import model

log_utils.log_to_stderr()


CheckValueMode = utils.CheckValueMode

torch.set_printoptions(profile="full")

_ACC = flags.DEFINE_enum(
    "acc", None, ["tpu", "gpu"], required=True, help="Accelerator to test."
)


def _init_mini_config():
  """Initialize a mini config for testing."""
  config = model.ModelArgs()

  config.max_batch_size = 4
  config.max_seq_len = 64 * 4
  config.dtype = "bf16"
  config.vocab_size = 1024
  config.dim = 64
  config.inter_dim = 64
  config.moe_inter_dim = 148
  # These two lines create a model with one dense layer, one MoE layer.
  config.n_layers = 2
  config.n_dense_layers = 1
  config.n_heads = 8
  # moe
  config.n_routed_experts = 8
  config.n_shared_experts = 2
  config.n_activated_experts = 2
  config.n_expert_groups = 1
  config.n_limited_groups = 1
  config.score_func = "softmax"
  config.route_scale = 1.0
  # mla
  config.q_lora_rank = 0
  config.kv_lora_rank = 4
  config.qk_nope_head_dim = 4  # dim (32) / heads (8)
  config.qk_rope_head_dim = 2  # Half of nope head
  config.v_head_dim = 16
  # No change to rope settings
  return config


# torch.device is a uncapitalized class.
def _init_mini_embeds(
    config: model.ModelArgs, dtype: torch.dtype, device: str | torch.device
):
  """Initialize a mini embedding sequence.

  Args:
    config: The model config dataclass.
    dtype: The dtype of returned tensor.
    device: The device of returned tensor.

  Returns:
    A tensor of shape (batch_size, seq_len, dim).
  """
  retval = torch.empty(
      config.max_batch_size,
      config.max_seq_len,
      config.dim,
      dtype=dtype,
      device=device,
  )
  torch.nn.init.trunc_normal_(retval)
  return retval


def _init_tensor(size: tuple[int, ...]):
  """Helper to initialize a tensor with trunc_normal.

  Args:
    size: The shape of the tensor to initialize.

  Returns:
    A tensor of the given shape with truncated normal values.
    The dtype will be torch.get_default_dtype(), and
    the device will be the default device.
  """
  retval = torch.empty(*size)
  torch.nn.init.trunc_normal_(retval)
  return retval


def _init_weights(module: torch.nn.Module | None):
  """Helper to initialize module weights.

  Kaiming uniform for matrices to avoid exploding activations.

  Args:
    module: The module to initialize. The weights of the module and all its
      children will be initialized in place.
  """
  if module is None:
    return
  for parameter in module.parameters(recurse=False):
    if parameter.dim() >= 2:
      torch.nn.init.kaiming_uniform_(parameter)
    else:
      torch.nn.init.trunc_normal_(parameter)

  for child in module.children():
    _init_weights(child)


class GateSpy:
  """A context manager to spy on a gate layer.

  This class is necessary because in bf16, the routing scores in the gate layer
  can be close but not in the same order, triggering flakes.

  Args:
    cpu_gate: The CPU gate layer to spy on.
    acc_gate: The accelerator gate layer to return the spied result on.
    accelerator_device: The device of the accelerator gate layer.
  """

  def __init__(
      self,
      cpu_gate: model.Gate,
      acc_gate: model.Gate,
      accelerator_device: torch.device,
  ):
    self._cpu_gate = cpu_gate
    self._acc_gate = acc_gate
    self._accelerator_device = accelerator_device

    self._cpu_patcher = None
    self._acc_patcher = None

    self._original_cpu_forward = None
    self._saved_result = None

  def _spy(self, *args, **kwargs):
    """A spy that saves the result of the original forward method."""
    result = self._original_cpu_forward(*args, **kwargs)
    self._saved_result = result
    return result

  def _acc_gate_returner(self, *_, **__):
    """A mock that returns the spied result on the accelerator device."""
    cpu_weights, cpu_indices = self._saved_result
    acc_weights = cpu_weights.to(self._accelerator_device)
    acc_indices = cpu_indices.to(self._accelerator_device)
    return acc_weights, acc_indices

  def __enter__(self):
    self._original_cpu_forward = self._cpu_gate.forward
    self._cpu_patcher = unittest.mock.patch.object(
        self._cpu_gate, "forward", new=self._spy
    )
    self._acc_patcher = unittest.mock.patch.object(
        self._acc_gate, "forward", side_effect=self._acc_gate_returner
    )
    self._cpu_patcher.start()
    self._acc_patcher.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._cpu_patcher.stop()
    self._acc_patcher.stop()


class SingleAcceleratorTest(parameterized.TestCase):
  """Tests for DeepSeek layers in single TPU mode."""

  def setUp(self):
    super().setUp()
    # This is consistent with model.py's main function.
    torch.set_default_dtype(torch.bfloat16)

    # This flag is defined as an int and has a default value,
    # per the source, but that behavior is not documented.
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)
    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    if _ACC.value == "tpu":

      self.accelerator_device = torch.device("tpu")
    elif _ACC.value == "gpu":
      self.accelerator_device = torch.device("cuda:0")
    else:
      raise RuntimeError(f"Unexpected flag value: {_ACC.value}")

  def test_computation(self):
    """Sanity test of accelerator computation."""
    # Arrange
    cpu_x = torch.tensor(2.0, device="cpu")
    acc_x = cpu_x.to(self.accelerator_device)

    # Act
    cpu_output = cpu_x**0.5
    acc_output = acc_x**0.5

    logging.info("cpu_output: %s", cpu_output)
    logging.info("acc_output: %s", acc_output)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  # Reference sizes: vocab_size=128_280, dim=7_168. But those will OOM smaller
  # test machines.
  # https://huggingface.co/deepseek-ai/DeepSeek-V3?show_file_info=model.safetensors.index.json
  @parameterized.named_parameters(
      ("indices_0_vocab_size_8_dim_7168", 0, 8, 7_168),
      ("indices_1_vocab_size_8_dim_7168", 1, 8, 7_168),
      ("indices_2_vocab_size_8_dim_7168", 2, 8, 7_168),
      ("indices_128_vocab_size_8_dim_7168", 128, 8, 7_168),
      ("indices_2048_vocab_size_8_dim_7168", 2048, 8, 7_168),
      ("indices_2049_vocab_size_8_dim_7168", 2049, 8, 7_168),
      ("indices_0_vocab_size_128280_dim_8", 0, 128_280, 8),
      ("indices_1_vocab_size_128280_dim_8", 1, 128_280, 8),
      ("indices_2_vocab_size_128280_dim_8", 2, 128_280, 8),
      ("indices_128_vocab_size_128280_dim_8", 128, 128_280, 8),
      ("indices_2048_vocab_size_128280_dim_8", 2048, 128_280, 8),
      ("indices_2049_vocab_size_128280_dim_8", 2049, 128_280, 8),
  )
  def test_embedding(self, num_indices: int, vocab_size: int, dim: int):
    """Test embedding on single accelerator."""

    # Arrange
    cpu_layer = model.ParallelEmbedding(vocab_size=vocab_size, dim=dim).to(
        "cpu"
    )
    _init_weights(cpu_layer)

    # nn.Module.to() is not sufficient to copy a module, unlike a Tensor.
    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    cpu_indices = torch.randint(0, vocab_size, (1, num_indices), device="cpu")
    acc_indices = cpu_indices.to(self.accelerator_device)

    # Act
    cpu_output = cpu_layer(cpu_indices)
    acc_output = acc_layer(acc_indices)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  @parameterized.named_parameters(
      ("B=1, in=1, out = 1", 1, 1, 1),
      ("B=1, in=7168, out = 7168", 1, 7_168, 7_168),
      ("B=2, in=1, out = 1", 2, 1, 1),
      ("B=2, in=1, out = 7168", 2, 1, 7_168),
      ("B=2, in=7168, out = 1", 2, 7_168, 1),
      ("B=2, in=7168, out = 7168", 2, 7_168, 7_168),
  )
  def test_linear_bf16(self, bsz: int, in_features: int, out_features: int):
    """Test linear layer accuracy between single tpu and cpu in bf16."""
    # Arrange
    cpu_layer = model.Linear(in_features, out_features).to("cpu")
    _init_weights(cpu_layer)

    # nn.Module.to() is not sufficient to copy a module, unlike a Tensor.
    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    cpu_x = _init_tensor((bsz, in_features)).cpu()
    acc_x = cpu_x.to(self.accelerator_device)

    # Check test setup.
    assert cpu_layer.weight.dtype == torch.bfloat16
    assert acc_layer.weight.dtype == torch.bfloat16
    assert cpu_layer.weight.element_size() > 1  # Triggers the non-fp8 case.
    assert acc_layer.weight.element_size() > 1  # Triggers the non-fp8 case.

    # Act
    cpu_output = cpu_layer(cpu_x)
    acc_output = acc_layer(acc_x)

    # Assert
    utils.assert_close(
        acc_output.to("cpu"), cpu_output, check_value=CheckValueMode.LOOSE
    )

  # TODO: b/432774613 - Design FP8 tests once Ironwood is available.

  # NOTE: Skipping row/col parallel. They are just linear on single TPU.

  # Dim = {512, 2048} values are from args.dim and args.kv_lora_rank.
  @parameterized.parameters(
      (1, 1, 512),
      (1, 1, 2048),
      (1, 2, 512),
      (1, 2, 2048),
      (2, 1, 512),
      (2, 1, 2048),
      (2, 2, 512),
      (2, 2, 2048),
  )
  def test_rms_norm(self, bsz: int, seqlen: int, dim: int):
    """Test RMSNorm layer accuracy between single tpu and cpu."""
    # Arrange
    cpu_layer = model.RMSNorm(dim).to("cpu")
    _init_weights(cpu_layer)

    # nn.Module.to() is not sufficient to copy a module, unlike a Tensor.
    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    cpu_x = _init_tensor((bsz, seqlen, dim)).cpu()
    acc_x = cpu_x.to(self.accelerator_device)

    # Act
    cpu_output = cpu_layer(cpu_x)
    acc_output = acc_layer(acc_x)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  def test_set_default_device(self):
    """Test set_default_device support."""
    # Arrange
    previous_default_device = torch.get_default_device()
    try:
      # Clear prior default device.
      torch.set_default_device("meta")

      # Act
      torch.set_default_device("cpu")
      result = str(torch.tensor(1.0).device)

      # Assert
      assert "cpu" == result, f"{result}"

      # Act
      torch.set_default_device(self.accelerator_device)
      result = str(torch.tensor(1.0).device)

      # Assert
      assert str(self.accelerator_device) == str(result), f"{result}"
    finally:
      # Restore prior default device.
      torch.set_default_device(previous_default_device)

  def test_precompute_freqs_cis(self):
    """Test precompute_freqs_cis.

    The function tested is NOT a trainable layer. It is used
    only once to calculate positional embeddings
    using RoPE. https://arxiv.org/abs/2104.09864.

    Its use of imaginary numbers (the i in cis is the imaginary part)
    may depened on unexpected ops.
    """
    # Arrange
    args = model.ModelArgs()

    previous_default_device = torch.get_default_device()
    try:
      torch.set_default_device("cpu")
      cpu_result = model.precompute_freqs_cis(args)

      # Act
      torch.set_default_device(self.accelerator_device)
      acc_result = model.precompute_freqs_cis(args)

      # Assert
      utils.assert_close(
          actual=acc_result.cpu(),
          expected=cpu_result,
          check_value=CheckValueMode.LOOSE,
          atol=0.0,
          rtol=torch.finfo(torch.bfloat16).eps,
      )
    finally:
      torch.set_default_device(previous_default_device)

  # TODO: b/432776862 - Fails on TPU, precompute_freqs_cis not implemented yet.
  @unittest.skip("precompute_freqs_cis not yet supported.")
  def test_apply_rotary_emb(self):
    """Test apply_rotary_emb."""
    raise NotImplementedError("precompute_freqs_cis not yet supported.")

  @parameterized.parameters(
      ("bshd,bthd->bsht", (2, 3, 5, 7), (2, 13, 5, 7)),
      ("bshd,hdc->bshc", (2, 3, 5, 7), (5, 7, 11)),
      ("bshc,btc->bsht", (2, 3, 5, 11), (2, 13, 11)),
      ("bshr,btr->bsht", (2, 3, 5, 17), (2, 13, 17)),
      ("bsht,bthd->bshd", (2, 3, 5, 13), (2, 13, 5, 7)),
      ("bsht,btc->bshc", (2, 3, 5, 13), (2, 13, 11)),
      ("bshc,hdc->bshd", (2, 3, 5, 11), (5, 7, 11)),
  )
  def test_deepseek_einsum(self, equation: str, left_shape, right_shape):
    """Test DeepSeek's specific usage patterns of einsum."""
    # Arrange
    # dtype is defaulted to bf16 in setUp().
    cpu_left = _init_tensor(left_shape).cpu()
    cpu_right = _init_tensor(right_shape).cpu()

    acc_left = cpu_left.to(self.accelerator_device)
    acc_right = cpu_right.to(self.accelerator_device)

    # Act
    cpu_output = torch.einsum(equation, cpu_left, cpu_right)
    acc_output = torch.einsum(equation, acc_left, acc_right)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  def test_print_acc_tensor(self):
    # Arrange
    x = torch.tensor(1.0, device=self.accelerator_device)

    # Act
    s = str(x)

    logging.info("x: %s", s)

  def test_print_acc_tensor_workaround(self):
    # Arrange
    x = torch.tensor(1.0, device=self.accelerator_device)

    # Act
    x = x.cpu()
    s = str(x)

    logging.info("x: %s", s)

  @parameterized.parameters(
      (1, 1, 1),
      (1, 1, 2),
      (1, 2, 1),
      (1, 2, 2),
      (2, 1, 1),
      (2, 1, 2),
      (2, 2, 1),
      (2, 2, 2),
      (4, 32, 64),
  )
  def test_MLP(self, bsz: int, dim: int, inter_dim: int):
    """Test DeepSeek's SwiGLU MLP."""
    # Arrange
    cpu_layer = model.MLP(dim=dim, inter_dim=inter_dim).to("cpu")
    _init_weights(cpu_layer)
    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    cpu_x = _init_tensor((bsz, dim)).cpu()
    acc_x = cpu_x.to(self.accelerator_device)

    # Act
    cpu_output = cpu_layer(cpu_x)
    acc_output = acc_layer(acc_x)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  def test_gate(self):
    """Test DeepSeek's gating mechanism."""
    # Arrange
    dtype = torch.get_default_dtype()
    # In bf16, raw routing scores are too close and result in flakey tests.
    torch.set_default_dtype(torch.double)
    try:
      mini_config = _init_mini_config()

      cpu_layer = model.Gate(mini_config).to("cpu")
      _init_weights(cpu_layer)
      acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

      cpu_input = _init_mini_embeds(
          mini_config, torch.get_default_dtype(), "cpu"
      )
      acc_input = cpu_input.to(self.accelerator_device)

      # Act
      cpu_output_weights, cpu_output_indices = cpu_layer(cpu_input)
      acc_output_weights, acc_output_indices = acc_layer(acc_input)

      # Assert
      utils.assert_close(
          acc_output_weights.to("cpu"),
          cpu_output_weights,
          check_value=CheckValueMode.LOOSE,
      )

    finally:
      torch.set_default_dtype(dtype)

  @parameterized.parameters(
      (1, 1, 1),
      (1, 1, 2),
      (1, 2, 1),
      (1, 2, 2),
      (2, 1, 1),
      (2, 1, 2),
      (2, 2, 1),
      (2, 2, 2),
      (4, 32, 64),
  )
  def test_expert(self, bsz: int, dim: int, inter_dim: int):
    """Test DeepSeek's SwiGLU Expert, which is unsharded unlike MLP."""
    # Arrange

    cpu_layer = model.Expert(dim=dim, inter_dim=inter_dim).to("cpu")
    _init_weights(cpu_layer)

    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    cpu_x = _init_tensor((bsz, dim)).cpu()
    acc_x = cpu_x.to(self.accelerator_device)

    # Act
    cpu_output = cpu_layer(cpu_x)
    acc_output = acc_layer(acc_x)

    # Assert
    utils.assert_close(acc_output.to("cpu"), cpu_output)

  def test_moe(self):
    """Test DeepSeek's MoE layer."""
    # Arrange
    mini_config = _init_mini_config()

    cpu_mini_input = _init_mini_embeds(
        mini_config, torch.get_default_dtype(), "cpu"
    )
    acc_mini_input = cpu_mini_input.to(self.accelerator_device)

    with unittest.mock.patch.object(
        model.Linear, "dtype", new=torch.get_default_dtype()
    ):
      cpu_layer = model.MoE(mini_config).to("cpu")
      _init_weights(cpu_layer)
    acc_layer = copy.deepcopy(cpu_layer).to(self.accelerator_device)

    # Check test setup: at least two experts.
    self.assertGreater(
        cpu_layer.experts_end_idx, cpu_layer.experts_start_idx + 2
    )

    # Act
    with GateSpy(cpu_layer.gate, acc_layer.gate, self.accelerator_device):
      cpu_output = cpu_layer(cpu_mini_input)
      acc_output = acc_layer(acc_mini_input)

    # Assert
    # The looped summation of the MoE layer appears to be ordered
    # differently on the accelerator and CPU, requiring wider tolerances.
    # Over 10 runs on GPU, most failed by exactly 0.0078125 or 0.00390625,
    # or exactly bf16's epsilon or half of that.
    utils.assert_close(
        actual=acc_output.to("cpu"),
        expected=cpu_output,
        check_value=CheckValueMode.LOOSE,
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=torch.finfo(torch.bfloat16).eps,
    )

  def test_mini_transformer(self):
    """Test mini sized config of the full DeepSeek transformer model."""
    # Arrange
    # This code based on model.py::main
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(self.accelerator_device)
    args = _init_mini_config()

    acc_x = torch.randint(
        0, args.vocab_size, (args.max_batch_size, args.max_seq_len)
    )
    cpu_x = acc_x.to("cpu")

    acc_model = model.Transformer(args)
    _init_weights(acc_model)
    cpu_model = copy.deepcopy(acc_model).to("cpu")

    # Check test setup: two layers, one dense and one MoE.
    self.assertIsInstance(cpu_model.layers[0].ffn, model.MLP)
    self.assertIsInstance(cpu_model.layers[1].ffn, model.MoE)

    # Act
    with GateSpy(
        cpu_model.layers[1].ffn.gate,
        acc_model.layers[1].ffn.gate,
        self.accelerator_device,
    ):
      # With multiple layers and experts, numerical differences of
      # the logits will accumulate beyond typical atol/rtol values.
      # Use the argmax to compare the logits.
      cpu_output = cpu_model(cpu_x).argmax(dim=-1)
      acc_output = acc_model(acc_x).argmax(dim=-1)

    logging.info("cpu_output: %s", cpu_output)
    logging.info("acc_output: %s", acc_output)

    # Assert
    utils.assert_close(
        actual=acc_output.cpu(),
        expected=cpu_output,
        atol=0.0,
        rtol=0.0,
    )


if __name__ == "__main__":
  absltest.main()
