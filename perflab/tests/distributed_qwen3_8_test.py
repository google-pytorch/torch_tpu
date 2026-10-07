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

"""Unit tests for the distributed Qwen3.8 SimpleFSDP training benchmark."""

from absl.testing import absltest
import torch
from torch import nn
import torch.multiprocessing as mp
from torch_tpu._internal.utils import test_utils
from torch_tpu._internal.distributed import multiprocessing
from perflab import context as context_lib
from perflab import target as target_lib
from perflab.distributed_models import qwen3_8
from perflab.model_utils import torchtitan_qwen3_5
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD_SIZE = 2


def _worker_cpu_training_step() -> None:
  """One SimpleFSDP training step of the debug model on gloo / CPU."""
  torch.distributed.init_process_group(backend="gloo", init_method="env://")
  try:
    seq_len = 64
    config = torchtitan_qwen3_5.text_config(
        "debugmodel", num_layers=4, seq_len=seq_len
    )
    model = qwen3_8.build_model(config, "cpu", _WORLD_SIZE, torch.float32)
    tokens, labels = qwen3_8.make_inputs(
        config.vocab_size, seq_len, torch.distributed.get_rank(), "cpu"
    )
    # A plain optimizer: `capturable=True` (used by the benchmark) is TPU / GPU
    # only.
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    before = [p.detach().to_local().clone() for p in model.parameters()]

    loss = qwen3_8.cross_entropy_loss(model, (tokens,), {"labels": labels})
    loss.backward()
    opt.step()

    assert torch.isfinite(loss), f"Non-finite loss {loss}"
    after = [p.detach().to_local() for p in model.parameters()]
    assert any(
        not torch.equal(b, a) for b, a in zip(before, after)
    ), "No parameter changed after the optimizer step"
  finally:
    torch.distributed.destroy_process_group()


class _LogitsModel(nn.Module):
  """Returns fixed logits, ignoring its input."""

  def __init__(self, logits: torch.Tensor):
    super().__init__()
    self.logits = logits

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    del tokens  # Unused.
    return self.logits


class DistributedQwen38BenchmarkTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def test_rejects_unsupported_platforms(self):
    for platform in (
        target_lib.Platform.V7_1X1X1,
        target_lib.Platform.V7_2X2X2,
        target_lib.Platform.B200_1,
    ):
      ctx = context_lib.Context(
          target=target_lib.make_target(platform),
          run_scope=context_lib.RunScope.PRESUBMIT,
      )
      with self.assertRaisesRegex(
          target_lib.UnsupportedBenchmark, "not supported on platform"
      ):
        qwen3_8.qwen3_8_27b_4_layers_simple_fsdp_training(ctx)

  def test_requires_initialized_process_group(self):
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.V7_1X1X1_2TC),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    with self.assertRaisesRegex(
        target_lib.UnsupportedBenchmark, "initialized torch.distributed"
    ):
      qwen3_8.qwen3_8_27b_4_layers_simple_fsdp_training(ctx)

  def test_cross_entropy_loss(self):
    logits = torch.randn(5, 7, dtype=torch.bfloat16)
    labels = torch.randint(0, 7, (5,))
    loss = qwen3_8.cross_entropy_loss(
        _LogitsModel(logits), (torch.zeros(5),), {"labels": labels}
    )
    self.assertEqual(loss.dtype, torch.float32)
    test_utils.assert_close(
        loss, nn.functional.cross_entropy(logits.float(), labels)
    )

  def test_text_config_is_qwen3_8_27b_hybrid_layout(self):
    config = torchtitan_qwen3_5.text_config("27B", num_layers=4, seq_len=2048)
    self.assertEqual(config.dim, 5120)
    self.assertEqual(config.vocab_size, 248320)
    self.assertIsNone(config.vision_encoder)
    # 3 GatedDeltaNet layers followed by 1 gated softmax attention layer.
    self.assertEqual(
        [layer.delta_net is not None for layer in config.layers],
        [True, True, True, False],
    )
    self.assertEqual(
        [layer.attention is not None for layer in config.layers],
        [False, False, False, True],
    )

  def test_simple_fsdp_training_step_ws2_cpu(self):
    distributed_utils.dist_run(_WORLD_SIZE, _worker_cpu_training_step)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
