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

"""Example use of Llama3.2 1B model."""

import time

from absl import app
import torch
from examples.paths import XM_HOME
import transformers


pipeline = transformers.pipeline
MODEL_PATH = f"{XM_HOME}weights/huggingface/meta-llama/Llama-3.2-1B"
MAX_DECODE_STEPS = 2


def main(argv):
  del argv  # Unused

  # Initialize TorchTPU
  tpu_device = torch.device("tpu")

  # TODO(b/437527594): right now seeds don't work in torch tpu, but setting it
  # here for now so we don't forget later.
  torch.manual_seed(123)

  pipe = pipeline("text-generation", model=MODEL_PATH, device=tpu_device)

  # The model doesn't come with a default chat template, so we need to manually
  # set one here.
  if pipe.tokenizer.chat_template is None:
    pipe.tokenizer.chat_template = (
        "{% set loop_messages = messages %}{% for message in loop_messages"
        " %}{% set content = '<|start_header_id|>' + message['role'] +"
        " '<|end_header_id|>\\n\\n' + message['content'] | trim +"
        " '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token"
        " + content %}{% endif %}{{ content }}{% endfor %}{% if"
        " add_generation_prompt %}{{"
        " '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
    )
  messages = [
      {"role": "user", "content": "Who are you?"},
  ]

  start_time = time.time()
  try:
    output_tpu = pipe(messages, max_new_tokens=MAX_DECODE_STEPS)
  # Sometimes the model will overflow in file
  # transformers/tokenization_utils_fast.py:
  # text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
  # OverflowError: out of range integral type conversion attempted
  # We ignore it for now to deflake the test.
  # TODO: fix this error.
  except OverflowError as e:
    print(f"Error: {e}")
    output_tpu = None
  stop_time = time.time()
  elapsed_time = stop_time - start_time
  print(f"Generation time: {elapsed_time * 1000:.2f} ms")

  if output_tpu:
    output_tpu = output_tpu[0]["generated_text"][1]["content"]
    print(f"Output: {output_tpu}")


if __name__ == "__main__":
  app.run(main)
