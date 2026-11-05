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

"""Dynamic registration of benchmarks against static model list."""

from collections.abc import Mapping
import functools
import itertools

import torch
from perflab import common
from perflab import context as context_lib
from perflab import flags as flags_lib
from perflab import mode as mode_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab import workload as workload_lib
from perflab.gen_utils import gen_utils

# The run modes that only TPUs and GPUs run. The FakeTensorMode CPU smoke test
# runs `eager_default`, the only run mode of CPU targets (`mode.modes_for`).
_ACCELERATOR_ONLY_RUN_MODES = frozenset({
    common.RunMode.COMPILED.value,
    common.RunMode.EAGER_OPTIMIZED.value,
})

# Generated benchmarks that the FakeTensorMode CPU smoke test
# (`--fake_tensor_mode`) cannot run, with the reason. Every other generated
# benchmark whose model config is in-tree runs in the smoke test, so that a
# change to the shared model loading (e.g. `tests/module_registry.py`) that
# breaks a generated benchmark fails in presubmit instead of in the nightly
# benchmark runs.
_FAKE_TENSOR_MODE_UNSUPPORTED: Mapping[str, str] = {}

# Generated benchmarks whose model forward branches or indexes on values
# computed from floating-point model weights (non-constant FakeTensors):
# * `lahja_sa_ahmad_v1_forward_gen` (`VitsModel`): sizes its output time axis
#   with `torch.arange(predicted_lengths.max(), ...)` from `duration_predictor`.
# * `lightglue_superpoint_forward_gen` (`LightGlueForKeypointMatching`):
#   extracts keypoint coordinates with `torch.nonzero(scores[0] > threshold)`,
#   prunes keypoints with boolean masks, and checks `torch.any(early_stopped)`.
# * `zoedepth_nyu_kitti_forward_gen` (`ZoeDepthForDepthEstimation`): selects a
#   metric depth head via `names[torch.argmax(domain_vote, dim=-1).item()]`.
# The FakeTensorMode CPU smoke test enables placeholder outputs for non-constant
# scalar reads (`_local_scalar_dense`), `nonzero`, and boolean mask indexing
# only for these benchmarks, so other benchmarks still fail loudly on unexpected
# data-dependent control flow.
_FAKE_TENSOR_MODE_DATA_DEPENDENT: frozenset[str] = frozenset({
    # go/keep-sorted start
    "lahja_sa_ahmad_v1_forward_gen",
    "lightglue_superpoint_forward_gen",
    "zoedepth_nyu_kitti_forward_gen",
    # go/keep-sorted end
})

_PRESUBMIT_REASON = (
    "Generated benchmark; too many to run on accelerators in presubmit scope."
)


@functools.cache
def has_in_tree_config(model_id: str) -> bool:
  """Returns whether `module_registry` reads the config of `model_id` in-tree.

  The FakeTensorMode CPU smoke test has no network access, so it builds only
  the models whose configs `module_registry` reads from the in-tree resources
  instead of GCS.

  Args:
    model_id: The HuggingFace model ID of a `gen_utils/models.csv` entry.
  """
  # Imported here rather than at module scope: importing module_registry is
  # likely to probe the CUDA driver, which poisons fork() for CUDA, and only
  # the FakeTensorMode CPU smoke test needs it here.
  from tests import module_registry  # pylint: disable=g-import-not-at-top

  return (
      module_registry.TransformersProvider.in_tree_config(model_id) is not None
  )


def fake_tensor_smoke_skip_reason(model_id: str, name: str) -> str | None:
  """Returns why the FakeTensorMode smoke test skips a generated benchmark.

  Args:
    model_id: The HuggingFace model ID of the benchmark's model.
    name: The name of the generated benchmark.

  Returns:
    The reason, or None if the smoke test runs the benchmark.
  """
  if not has_in_tree_config(model_id):
    return f"the config of {model_id} is not in-tree"
  return _FAKE_TENSOR_MODE_UNSUPPORTED.get(name)


def _make_factory(
    model_id: str,
    is_training: bool,
    model_load_fn: gen_utils.ModelLoadFn,
    allow_data_dependent_outputs: bool = False,
):
  """Creates a benchmark factory for a given model."""

  def factory(ctx: context_lib.Context):
    if allow_data_dependent_outputs:
      mode_lib.allow_data_dependent_fake_outputs()
    model, inputs = model_load_fn(ctx, model_id, is_training)
    if is_training:
      opt = torch.optim.AdamW(
          model.parameters(),
          lr=1e-4,
          capturable=True,
          fused=True,
      )
      return model, (), inputs, opt
    return model, (), inputs

  return factory


def _iter_gen_benchmark_specs():
  """Yields all combinations of models and execution steps for benchmarking."""
  for model_gen_spec, (is_training, step) in itertools.product(
      gen_utils.iter_models_csv(),
      [
          (False, step_lib.StepperType.FORWARD),
          (True, step_lib.StepperType.TRAINING),
      ],
  ):
    yield model_gen_spec, is_training, step


def _register_gen_benchmarks():
  """Registers generated benchmarks in the global registry.

  Every generated benchmark is registered regardless of the run scope, with
  `PresubmitPolicy.CPU_SMOKE_ONLY`. Under `--fake_tensor_mode`, in any run
  scope, only the benchmarks that the FakeTensorMode CPU smoke can build are
  registered (see `fake_tensor_smoke_skip_reason`).
  """
  _, fake_tensor_mode = flags_lib.collection_flag_values()
  for model_gen_spec, is_training, step in _iter_gen_benchmark_specs():
    name = model_gen_spec.benchmark_name(is_training=is_training, suffix="gen")
    if fake_tensor_mode and fake_tensor_smoke_skip_reason(
        model_gen_spec.model_id, name
    ):
      continue
    extra_config = model_gen_spec.get_extra_config(is_training, suffix="gen")
    skipped_run_modes = extra_config.skipped_run_modes
    if fake_tensor_mode and _ACCELERATOR_ONLY_RUN_MODES - skipped_run_modes:
      # The smoke test only builds the benchmark and takes a step, so it also
      # checks the benchmarks that skip `eager_default` but run on accelerators
      # in another mode.
      skipped_run_modes = skipped_run_modes - {
          common.RunMode.EAGER_DEFAULT.value
      }
    step_kwargs = extra_config.step_kwargs
    model_load_fn = extra_config.model_load_fn

    registry_lib.add_benchmark(
        factory=_make_factory(
            model_gen_spec.model_id,
            is_training,
            model_load_fn,
            allow_data_dependent_outputs=(
                name in _FAKE_TENSOR_MODE_DATA_DEPENDENT
            ),
        ),
        stepper=step,
        stepper_kwargs=step_kwargs,
        name=name,
        skipped_run_modes=skipped_run_modes,
        presubmit=registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY,
        presubmit_reason=_PRESUBMIT_REASON,
        # Batch only: the input generator decides the sequence length of each
        # model (see `load_model.default_load_model`).
        workload=registry_lib.scoped_workload(
            full=workload_lib.Workload(extra_config.full_scope_batch_size),
            presubmit=workload_lib.Workload(1),
        ),
    )


_register_gen_benchmarks()
