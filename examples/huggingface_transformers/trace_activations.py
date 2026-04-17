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

"""CLI tool to trace a HuggingFace model with ActivationTracer."""

from absl import app
from absl import flags
from etils import epath
import torch
from torch_tpu._internal.utils import tracer_utils
from torch_tpu._internal.utils import utils
from examples import paths
import transformers


_DEVICE = flags.DEFINE_enum(
    "device",
    None,
    ["cpu", "cuda", "tpu"],
    "The device to run the model on.",
)

_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-0.6B",
    "The model id to trace, e.g. 'openai-community/gpt2'.",
)

_WEIGHTS_DIR = flags.DEFINE_string(
    "weights_dir",
    f"{paths.XM_HOME}weights/huggingface",
    "The directory containing HuggingFace model weights.",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    paths.TRACES_HOME,
    "The directory to save the activation trace file.",
)


def main(_) -> None:
  if _DEVICE.value == "tpu":

    device = torch.device("tpu")
  elif _DEVICE.value == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA requested but not available.")
    device = torch.device("cuda")
  elif _DEVICE.value == "cpu":
    device = torch.device("cpu")
  else:
    raise ValueError(f"Unsupported device: {_DEVICE.value}")

  model_id = _MODEL_ID.value
  weights_dir = _WEIGHTS_DIR.value

  print(f"Loading model {model_id} on device: {device}")
  model = transformers.AutoModelForCausalLM.from_pretrained(
      epath.Path(weights_dir) / model_id, torch_dtype="auto"
  ).to(device)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      epath.Path(weights_dir) / model_id
  )

  # Create a dummy input. A real run should add the chat template.
  prompt = "To be or not..."
  inputs = tokenizer(prompt, return_tensors="pt").to(device)

  with utils.ActivationTracer(model) as tracer:
    _ = model(**inputs)

  print(tracer_utils.pformat_activation_tracer(tracer))

  output_dir = epath.Path(_OUTPUT_DIR.value)
  output_path = output_dir / f"activation_trace_{model_id}_{device}.txt"
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w") as f:
    f.write(tracer_utils.pformat_activation_tracer(tracer))
  print(f"Activation trace saved to {output_path}")


if __name__ == "__main__":
  app.run(main)
