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

r"""XManager launch script for qwen3_kv_cache_bd.py on tpu-v7x.

/google/bin/releases/xmanager/cli/xmanager.par \
launch third_party/py/torch_tpu/examples/qwen3/xm_launch_bd.py \
    -- \
--xm_resource_alloc="msca-dynamic/pytorch-tpu-dynamic-xm"
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_abc

_MODEL_NAME = flags.DEFINE_string(
    'model_name',
    'Qwen/Qwen3-0.6B',
    'Name of the model to run. Can be "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", '
    ' or "Qwen/Qwen3-4B".',
)
_RUN_MODE = flags.DEFINE_enum(
    'run_mode',
    'static',
    ['static', 'bd', 'both'],
    'The run mode for inference.',
)
_DECODE_STEPS = flags.DEFINE_integer(
    'decode_steps', 4, 'Number of decode steps to run.'
)
_PADDING_LENGTH = flags.DEFINE_integer(
    'padding_length', 10, 'Padding length for the dynamic tensors.'
)
_SEQ_LEN = flags.DEFINE_integer(
    'seq_len', 2048, 'Sequence length for the input.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with xm_abc.create_experiment(
      experiment_title=(
          'KV Cache BD model: %s run_mode: %s, seq_len: %d, decode_steps: %d,'
          ' padding_length: %d'
      )
      % (
          _MODEL_NAME.value,
          _RUN_MODE.value,
          _SEQ_LEN.value,
          _DECODE_STEPS.value,
          _PADDING_LENGTH.value,
      ),
      attribution_urls=['rh/efforts/1910'],
  ) as experiment:
    requirements = xm.JobRequirements(gf='1x1x1')
    [executable] = experiment.package([
        xm.bazel_binary(
            label='//examples/qwen3:qwen3_kv_cache_bd',
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=xm_abc.bazel_args.for_requirements(requirements),
            args={
                'alsologtostderr': 'true',
                'model_name': _MODEL_NAME.value,
                'run_mode': _RUN_MODE.value,
                'decode_steps': _DECODE_STEPS.value,
                'padding_length': _PADDING_LENGTH.value,
                'seq_len': _SEQ_LEN.value,
                'xprof_end_2_end_upload': 'true',
                'xprof_host_trace_level': '3',
                'xprof_e2e_enable_python_tracer': 'true',
            },
        ),
    ])

    experiment.add(
        xm.Job(
            executable,
            executor=xm_abc.Borg(
                requirements=requirements,
                logs_read_access_roles=['all-users'],
            ),
        )
    )


if __name__ == '__main__':
  app.run(main)
