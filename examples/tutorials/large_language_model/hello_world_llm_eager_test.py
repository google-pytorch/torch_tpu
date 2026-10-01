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

from absl import logging
from absl.testing import absltest
import torch
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers


log_utils.log_to_stderr()


MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/Qwen/Qwen3-0.6B"
INPUT_TEXT = "Hello, I am a"


class HelloWorldLlmEagerTest(absltest.TestCase):

  def test_hello_world_llm_eager(self):
    device = torch.accelerator.current_accelerator()
    torch.manual_seed(42)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(
        device
    )
    inputs = tokenizer(INPUT_TEXT, return_tensors="pt").to(device)

    # Generate tokens
    with torch.no_grad():
      outputs = model.generate(
          **inputs,
          max_length=30,
          pad_token_id=tokenizer.pad_token_id,
      )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("Input: '%s'", INPUT_TEXT)
    logging.info("Output: '%s'", result)
    self.assertStartsWith(result, INPUT_TEXT)


if __name__ == "__main__":
  absltest.main()
