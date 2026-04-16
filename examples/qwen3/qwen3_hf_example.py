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
from absl import logging
import torch
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers

log_utils.log_to_stderr()


#  for backend registration

TOKENIZER_PATH = f"{paths.XM_HOME}weights/huggingface/Qwen/Qwen3-0.6B"
MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/Qwen/Qwen3-0.6B"


def model_generate(
    model, initial_inputs: torch.Tensor, tokenizer, max_decode_steps
):
  native_device = model.device
  with torch.no_grad():
    start_time = time.time()
    output = model(input_ids=initial_inputs)
    end_time = time.time()
    logits = output.logits

    prefill_time = end_time - start_time
    print(f"Prefill time: {prefill_time * 1000:.2f} ms")
    input_ids = initial_inputs
    next_token = None
    start_time = time.time()
    for i in range(max_decode_steps):
      print(f"Decode step {i}, device={logits.device}")
      if next_token == 0:
        break
      # greedy sampling
      logits = logits.to("cpu")
      next_token = (
          torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1).to(native_device)
      )
      input_ids = torch.cat([input_ids, next_token], dim=1)
      logits = model(input_ids=input_ids)
      logits = logits.logits
    end_time = time.time()
    decode_time = end_time - start_time
    print(f"Decode time per token: {decode_time * 1000 / i:.2f} ms")
  output = input_ids
  output_list = output.tolist()
  output_text = tokenizer.decode(output_list[0], skip_special_tokens=True)
  return output_text


# pylint: disable=unused-argument
def main(argv):
  # Initialize TorchTPU
  tpu_device = torch.device("tpu")
  torch.manual_seed(123)

  # Load tokenizer from HuggingFace
  tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

  # Load CPU model from HuggingFace
  model_cpu = transformers.AutoModelForCausalLM.from_pretrained(
      MODEL_PATH, torch_dtype=torch.bfloat16
  )
  assert str(model_cpu.device) == "cpu", "model_cpu device should be cpu"
  assert model_cpu.dtype == torch.bfloat16, "model_cpu dtype should be bfloat16"

  # Load CPU model from HuggingFace and move it to the TPU
  model_tpu = transformers.AutoModelForCausalLM.from_pretrained(
      MODEL_PATH, torch_dtype=torch.bfloat16
  )
  model_tpu = model_tpu.to("tpu")
  assert (
      str(model_tpu.device) == f"{tpu_device}:0"
  ), f"model_tpu device should be {tpu_device}:0, got {model_tpu.device}"
  assert model_tpu.dtype == torch.bfloat16, "model_tpu dtype should be bfloat16"

  text = "Who are you?"
  messages = [{"role": "user", "content": text}]
  inputs = tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
  ).input_ids

  # Note: This is a smoke test to ensure torch.compile runs without error.
  # 1. Speed: Input increases as autoregressive continues, dynamo will recompile
  # fx_graph. It will be slow, as besides TPU compilation, dynamo generating
  # fx_graph also takes time.
  # 2. dynamic: dynamic to false as we cannot handle symint now.
  # TODO: Experiment with pad and slice on symint
  # TODO: Experiment vllm style warming up (precompile)
  max_decode_steps = 2

  output_cpu = model_generate(model_cpu, inputs, tokenizer, max_decode_steps)
  logging.info("output_cpu=%s", output_cpu)

  output_tpu = model_generate(
      model_tpu, inputs.to("tpu"), tokenizer, max_decode_steps
  )
  logging.info(f"output_tpu=%s", output_tpu)
  assert output_cpu == output_tpu

  model_tpu_compiled = torch.compile(model_tpu, dynamic=False, backend="tpu")
  output_tpu_compiled = model_generate(
      model_tpu_compiled, inputs.to("tpu"), tokenizer, max_decode_steps
  )
  logging.info(f"output_tpu_compiled=%s", output_tpu_compiled)
  assert output_cpu == output_tpu_compiled


if __name__ == "__main__":
  app.run(main)
