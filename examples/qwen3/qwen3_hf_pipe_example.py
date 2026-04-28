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

"""Example use of Qwen3 model."""

import time

from absl import app
import torch
from torch_tpu._internal import compiler_options as compiler
from examples import paths
import transformers


_MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/Qwen/Qwen3-0.6B"
_MAX_DECODE_STEPS = 2


def main(argv):
  del argv
  torch.manual_seed(123)

  # Run the following command to test:
  # blaze --blazerc=/dev/null test //examples/qwen3:qwen3_hf_pipe_example --compilation_mode=opt
  pipe_tpu = transformers.pipeline(
      "text-generation",
      model=_MODEL_PATH,
      device=torch.accelerator.current_accelerator(),
  )

  start_time = time.time()
  output_tpu = pipe_tpu(
      [
          {"role": "user", "content": "Who are you?"},
      ],
      max_new_tokens=_MAX_DECODE_STEPS,
  )

  elapsed_time = time.time() - start_time
  output_tpu = output_tpu[0]["generated_text"][1]["content"]

  print(f"Generation time: {elapsed_time:.5f} s")
  print(f"Output: {output_tpu}")


if __name__ == "__main__":
  app.run(main)
