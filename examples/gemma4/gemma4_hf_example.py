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

"""Example use of HuggingFace Gemma4 E2B model baseline on TorchTPU."""

import os

from absl import app
from absl import flags
from absl import logging
import torch
import torch_tpu
from torch_tpu._internal.utils import log_utils
from examples import paths
import transformers

log_utils.log_to_stderr()

MODEL_ID = "google/gemma-4-e2b"
MODEL_PATH = f"{paths.XM_HOME}weights/huggingface/{MODEL_ID}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_LOAD_PRETRAINED_WEIGHTS = flags.DEFINE_bool(
    "load_pretrained_weights",
    False,
    "Whether to load pre-trained weights from disk rather than initializing"
    " from full config.",
)


def create_gemma4_e2b_hf_model():
  """Creates a HuggingFace Gemma-4-E2B baseline model from the full model config."""
  config_path = CONFIG_PATH if os.path.exists(CONFIG_PATH) else MODEL_ID
  config = transformers.AutoConfig.from_pretrained(config_path)
  model = transformers.AutoModelForCausalLM.from_config(config)
  return config, model


def main(argv):
  del argv
  torch.manual_seed(123)

  tpu_available = True
  tpu_device = None
  try:
    torch.tensor([1.0]).to("tpu")
    tpu_device = torch.device("tpu")
  except RuntimeError:
    tpu_available = False
    logging.info(
        "TPU device not available in environment. Running CPU-only baseline..."
    )

  if _LOAD_PRETRAINED_WEIGHTS.value:
    logging.info(
        "Loading HuggingFace Gemma-4-E2B pre-trained model from %s...",
        MODEL_PATH,
    )
    config = transformers.AutoConfig.from_pretrained(MODEL_PATH)
    model_cpu = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16
    )
  else:
    logging.info(
        "Initializing HuggingFace Gemma-4-E2B baseline model from full"
        " config..."
    )
    config, model_cpu = create_gemma4_e2b_hf_model()

  vocab_size = (
      config.text_config.vocab_size
      if hasattr(config, "text_config")
      else config.vocab_size
  )

  # Prepare sample input
  input_ids = torch.randint(0, vocab_size, (2, 10), device="cpu")

  # CPU execution baseline
  logging.info("Running HuggingFace Gemma-4-E2B forward pass on CPU...")
  model_cpu.eval()
  with torch.no_grad():
    output_cpu = model_cpu(input_ids)

  hf_hidden_cpu = (
      output_cpu.last_hidden_state
      if hasattr(output_cpu, "last_hidden_state")
      else output_cpu[0]
  )
  logging.info("CPU output shape: %s", hf_hidden_cpu.shape)

  if not tpu_available:
    logging.info("Skipping TPU execution as TPU device is not available.")
    return

  # TPU execution baseline
  logging.info("Moving HuggingFace Gemma-4-E2B baseline model to TPU...")
  model_tpu = model_cpu.to(tpu_device)
  model_tpu.eval()

  input_ids_tpu = input_ids.to(tpu_device)
  with torch.no_grad():
    output_tpu = model_tpu(input_ids_tpu)

  hf_hidden_tpu = (
      output_tpu.last_hidden_state
      if hasattr(output_tpu, "last_hidden_state")
      else output_tpu[0]
  )

  logging.info("TPU output shape: %s", hf_hidden_tpu.shape)
  assert (
      hf_hidden_tpu.shape == hf_hidden_cpu.shape
  ), f"Shape mismatch: {hf_hidden_tpu.shape} vs {hf_hidden_cpu.shape}"
  assert not torch.isnan(hf_hidden_tpu).any(), "TPU output contains NaN"
  assert not torch.isinf(hf_hidden_tpu).any(), "TPU output contains Inf"
  torch.testing.assert_close(
      hf_hidden_tpu.cpu(), hf_hidden_cpu, rtol=5e-2, atol=3.0
  )

  logging.info(
      "HuggingFace Gemma-4-E2B full config baseline model test passed"
      " successfully!"
  )


if __name__ == "__main__":
  app.run(main)
