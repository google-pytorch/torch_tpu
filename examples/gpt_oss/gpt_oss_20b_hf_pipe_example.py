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

"""Example gpt-oss-20b model inference."""

import time

from absl import app
import torch
from examples import paths
import transformers


XM_HOME = paths.XM_HOME
MAX_DECODE_STEPS = 2


def main(argv):
  del argv
  torch.manual_seed(123)

  pipe_tpu = transformers.pipeline(
      "text-generation",
      model=f"{XM_HOME}weights/huggingface/openai/gpt-oss-20b",
      device="tpu",
  )

  start_time = time.perf_counter()
  output_tpu = pipe_tpu(
      [
          {"role": "user", "content": "Who are you?"},
      ],
      max_new_tokens=MAX_DECODE_STEPS,
  )

  elapsed_time = time.perf_counter() - start_time
  output_tpu = output_tpu[0]["generated_text"][1]["content"]

  print(f"Generation time: {elapsed_time:.5f} s")
  print(f"Output: {output_tpu}")


if __name__ == "__main__":
  app.run(main)
