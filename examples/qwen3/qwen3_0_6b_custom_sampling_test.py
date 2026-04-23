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

"""Tests of Qwen3 as implemented by HuggingFace Transformers.

   !!!
   This code is WIP and meant to drive developments, such as compile time
   improvements. Do NOT use for production or export to OSS.
   !!!

   To run on GB200 locally in a cloudtop:
     blaze test -c opt --config cuda --test_strategy=local \
        qwen3_0_6B_custom_sampling_test_cuda
"""

import os
import time

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.utils import log_utils
from examples import paths

log_utils.log_to_stderr()


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["tpu", "cuda"],
    required=True,
    help="Accelerator to test.",
)
_WEIGHTS = flags.DEFINE_string(
    name="weights",
    default="/tmp/Qwen3-0.6B",
    help="Local weights for GB200 experiments.",
)
_MAX_STEPS = 10
_WEIGHTS_DIR = f"{paths.XM_HOME}weights/huggingface/Qwen/Qwen3-0.6B"


class SingleAcceleratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Standard absltest flag handling.
    seed = absltest.FLAGS.test_random_seed
    if seed is None or not isinstance(seed, int):
      raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

    torch.manual_seed(seed)
    logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)

    # Identify accelerator device (tpu, or cuda)
    if not hasattr(self, "acc_device"):
      self.acc_device = torch.device(_DEVICE.value)
      # TODO(pganssle): Evaluate whether this assertion is still necessary.
      assert str(self.acc_device).split(":", 1)[0] == _DEVICE.value

    # Load model directly from transformers.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    weights = _WEIGHTS_DIR

    # For GB200, this test has to be run local and in order to make that
    # work, weights need to be copied to a local directory.
    if _DEVICE.value == "cuda":
      weights = _WEIGHTS.value
      if not os.path.isdir(weights):
        raise ValueError(f"Cannot find weights in local dir: {weights}")
    print("Getting model weights from:", weights)

    # Load model(s) for CPU and accelerator.
    self.tokenizer = AutoTokenizer.from_pretrained(weights)
    self.model_cpu = AutoModelForCausalLM.from_pretrained(
        weights,
        torch_dtype=torch.bfloat16,
    )
    self.model_acc = AutoModelForCausalLM.from_pretrained(
        weights,
        torch_dtype=torch.bfloat16,
    )
    self.model_acc = self.model_acc.to(self.acc_device)

    # Assert that things are Ok so far. Asserts are fine here.
    assert str(self.model_cpu.device) == "cpu", "model_cpu device should be cpu"
    assert (
        self.model_cpu.dtype == torch.bfloat16
    ), "model_cpu dype should be bfloat16"

    assert str(self.model_acc.device) == f"{self.acc_device}:0", (
        f"model_acc.device should be '{self.acc_device}:0, got"
        f" {self.model_acc.device}"
    )
    assert (
        self.model_acc.dtype == torch.bfloat16
    ), "model_acc dtype should be bfloat16"

    # Compute prefill.
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    self.inputs_cpu = inputs.input_ids
    self.inputs_acc = torch.clone(self.inputs_cpu).to(self.acc_device)
    self.decode_steps = 40

  def model_generate(
      self, model, tokenizer, initial_inputs: torch.Tensor, decode_steps
  ):
    """Runs and times prefill and 'decode_steps' calls to the model."""
    native_device = model.device
    with torch.no_grad():
      # Prefill.
      start_time = time.time()
      output = model(input_ids=initial_inputs)
      end_time = time.time()
      logits = output.logits
      print(f"Prefill time: {(end_time - start_time) * 1000:.2f} ms")

      # Decode steps.
      input_ids = initial_inputs
      start_time = time.time()
      for i in range(decode_steps):
        # greedy sampling
        logits = logits.to("cpu")
        next_token = (
            torch.argmax(logits[:, -1, :], dim=-1)
            .unsqueeze(-1)
            .to(native_device)
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        logits = model(input_ids=input_ids)
        logits = logits.logits
      end_time = time.time()
      decode_time = end_time - start_time
      print(
          "Avg. decode time per token:"
          f" {decode_time * 1000 / decode_steps:.2f} ms"
      )

    # We keep appending all the generated tokens to inputs_ids. The final output
    # is the collected input_ids tensor
    output_ids = input_ids
    output_list = output_ids.tolist()
    output_text = tokenizer.decode(output_list[0], skip_special_tokens=True)
    return output_text

  def test_cpu(self):
    print("-" * 10, f" CPU ({self.decode_steps} decode steps)", "-" * 10)
    output = self.model_generate(
        self.model_cpu, self.tokenizer, self.inputs_cpu, _MAX_STEPS
    )
    print("-" * 10, "Model Output", "-" * 10)
    print(output, "\n\n")

  def test_acc(self):
    print(
        "-" * 10, f" ACC cold start({self.decode_steps} decode steps)", "-" * 10
    )
    output_acc_cold = self.model_generate(
        self.model_acc, self.tokenizer, self.inputs_acc, _MAX_STEPS
    )
    print("-" * 10, "Model Output - run 1", "-" * 10)
    print(output_acc_cold, "\n\n")

    print(
        "-" * 10, f" ACC warm start({self.decode_steps} decode steps)", "-" * 10
    )
    output_acc_warm = self.model_generate(
        self.model_acc, self.tokenizer, self.inputs_acc, _MAX_STEPS
    )
    print("-" * 10, "Model Output - run 2", "-" * 10)
    print(output_acc_warm, "\n\n")
    self.assertEqual(output_acc_cold, output_acc_warm)


if __name__ == "__main__":
  absltest.main()
