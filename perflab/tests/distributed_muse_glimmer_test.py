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

"""Unit tests for the distributed Muse Glimmer SimpleFSDP training benchmark."""

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.distributed_models import muse_glimmer
from perflab.model_utils import muse_glimmer as muse_glimmer_utils
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD_SIZE = 2


def _worker_cpu_training_step() -> None:
  """One SimpleFSDP training step of the debug model on gloo / CPU."""
  torch.distributed.init_process_group(backend="gloo", init_method="env://")
  try:
    # The prototype builds its device mesh on the accelerator it detects.
    parallel_dims = muse_glimmer_utils._submodule("distributed.parallel_dims")  # pylint: disable=protected-access
    parallel_dims.device_type = "cpu"
    # Eager flex attention has no backward on CPU.
    muse_glimmer_utils.use_sdpa_flex_attention()

    num_docs, doc_len = 2, 32
    rank = torch.distributed.get_rank()
    bundle = muse_glimmer_utils.build_model(
        "debugmodel",
        4,
        doc_len,
        "cpu",
        torch.float32,
        world_size=_WORLD_SIZE,
    )
    assert bundle.parallel_dims is not None
    tokens, kwargs = muse_glimmer_utils.make_batch(
        bundle, num_docs, doc_len, rank, _WORLD_SIZE, "cpu"
    )
    assert tokens.shape == (num_docs * doc_len,), tokens.shape
    opt = muse_glimmer_utils.make_optimizer(bundle.model)
    before = [p.detach().to_local().clone() for p in bundle.model.parameters()]

    loss = muse_glimmer_utils.cross_entropy_loss(
        bundle.model, (tokens,), kwargs
    )
    loss.backward()
    opt.step()

    assert torch.isfinite(loss), f"Non-finite loss {loss}"
    after = [p.detach().to_local() for p in bundle.model.parameters()]
    assert any(
        not torch.equal(b, a) for b, a in zip(before, after)
    ), "No parameter changed after the optimizer step"
  finally:
    torch.distributed.destroy_process_group()


class DistributedMuseGlimmerBenchmarkTest(
    seed_test_utils.MultiProcessRepeatableTest
):

  def test_registers_batch_size_variants(self):
    for suffix in ("", "_bs2", "_bs4"):
      spec = registry_lib.REGISTRY[
          f"muse_glimmer_30b_4_layers_simple_fsdp_training{suffix}"
      ]
      self.assertEqual(
          spec.skipped_run_modes,
          {"eager_default", "eager_defer_never_and_launch_blocking"},
      )

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
        muse_glimmer.muse_glimmer_30b_4_layers_simple_fsdp_training(ctx)

  def test_requires_initialized_process_group(self):
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.V7_1X1X1_2TC),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    with self.assertRaisesRegex(
        target_lib.UnsupportedBenchmark, "initialized torch.distributed"
    ):
      muse_glimmer.muse_glimmer_30b_4_layers_simple_fsdp_training(ctx)

  def test_simple_fsdp_training_step_ws2_cpu(self):
    distributed_utils.dist_run(_WORLD_SIZE, _worker_cpu_training_step)


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
