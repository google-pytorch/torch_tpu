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

"""Registered end-to-end model performance benchmarks."""

from examples.benchmarks.e2e import benchmark_function_db
from examples.benchmarks.e2e import benchmark_registry
from examples.benchmarks.e2e import benchmark_utils
from examples.benchmarks.e2e import model_utils
from examples.benchmarks.e2e import performance_utils
from examples.benchmarks.e2e import test_utils

_HF_LLAMA_3_2_1B_BENCHMARK_NAME = "hf_llama_3_2_1b"


@benchmark_registry.register_benchmark(
    benchmark_name=_HF_LLAMA_3_2_1B_BENCHMARK_NAME,
    run_modes=[
        benchmark_utils.RunMode.EAGER_DEFAULT,
        benchmark_utils.RunMode.EAGER_OPTIMIZED,
        benchmark_utils.RunMode.EAGER_DEFER_NEVER_AND_LAUNCH_BLOCKING,
        benchmark_utils.RunMode.COMPILED,
    ],
)
def test_llama_3_2_1b_forward(
    run_mode,
) -> performance_utils.PerformanceBenchmarkConfig:
  """Returns the config for the forward pass of Llama-3.2-1B."""
  return performance_utils.PerformanceBenchmarkConfig(
      supported_platforms=[
          benchmark_utils.Platform.GFC_1X1X1,
          benchmark_utils.Platform.B200_1,
      ],
      benchmark_category=benchmark_utils.BenchmarkCategory.HUGGINGFACE_LLM,
      run_mode=run_mode,
      is_training=False,
      model_and_input_args=performance_utils.ModelAndInputArgs(
          model_name="meta-llama/Llama-3.2-1B",
          sequence_length=4096,
          batch_size=1,
      ),
      model_and_input_factory=model_utils.huggingface_llm_model_builder,
      eval_factory=benchmark_function_db.huggingface_eval_factory,
  )


class RegisteredModelPerformanceTest(test_utils.BenchmarkTest):
  """Standalone test class for registered model benchmarks."""

  pass


if __name__ == "__main__":

  # pylint: disable=g-import-not-at-top,g-bad-import-order
  import multiprocessing as mp
  from absl.testing import absltest
  from torch_tpu._internal.shims.pyglib.contrib.g3_multiprocessing import g3_multiprocessing
  # pylint: enable=g-import-not-at-top,g-bad-import-order

  benchmark_registry.populate_standalone_test_class(
      RegisteredModelPerformanceTest
  )

  mp.set_start_method("spawn")
  g3_multiprocessing.handle_test_main(absltest.main)
