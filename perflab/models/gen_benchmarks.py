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

import itertools
import sys

from absl import flags
import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import step_lib
from perflab.gen_utils import gen_utils


def _make_factory(
    model_id: str, is_training: bool, model_load_fn: gen_utils.ModelLoadFn
):
  """Creates a benchmark factory for a given model."""

  def factory(ctx: context_lib.Context):
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
  """Registers generated benchmarks in the global registry."""
  is_presubmit = False
  if flags.FLAGS.is_parsed():
    if context_lib.RUN_SCOPE.value == context_lib.RunScope.PRESUBMIT:
      is_presubmit = True
  else:
    # Fallback for import time before flags are parsed
    if "--run_scope=presubmit" in sys.argv:
      is_presubmit = True
    else:
      for i, arg in enumerate(sys.argv):
        if arg == "--run_scope" and i + 1 < len(sys.argv):
          if sys.argv[i + 1] == "presubmit":
            is_presubmit = True
            break

  if is_presubmit:
    # We will need to split the target up to prevent timeouts.
    return

  for model_gen_spec, is_training, step in _iter_gen_benchmark_specs():
    extra_config = model_gen_spec.get_extra_config(is_training, suffix="gen")
    skipped_run_modes = extra_config.skipped_run_modes
    step_kwargs = extra_config.step_kwargs
    model_load_fn = extra_config.model_load_fn

    registry_lib.add_benchmark(
        factory=_make_factory(
            model_gen_spec.model_id, is_training, model_load_fn
        ),
        stepper=step,
        stepper_kwargs=step_kwargs,
        name=model_gen_spec.benchmark_name(
            is_training=is_training, suffix="gen"
        ),
        skipped_run_modes=skipped_run_modes,
    )


_register_gen_benchmarks()
