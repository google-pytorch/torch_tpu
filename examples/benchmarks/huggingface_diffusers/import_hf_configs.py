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

"""Downloads HuggingFace model configurations.

This script downloads all files from a HuggingFace model repository except for
weights files (e.g., .safetensors, .bin, .pth). The files are stored in a
directory named after the model in `model_configs`.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import huggingface_hub


_MODEL_ID = flags.DEFINE_string(
    "model_id",
    None,
    "HF model ID, e.g., 'stabilityai/stable-diffusion-xl-base-1.0'",
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "third_party/py/torch_tpu/examples/benchmarks/huggingface_diffusers/model_configs",
    "The directory to save the model configurations to.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _MODEL_ID.value is None:
    raise app.UsageError("Please provide a model_id.")

  model_id = _MODEL_ID.value
  output_path = os.path.join(_OUTPUT_DIR.value, model_id)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  huggingface_hub.snapshot_download(
      repo_id=model_id,
      local_dir=output_path,
      # Only include .json files for model configs and indices
      allow_patterns=["*.json"],
  )
  print(f"Model configs for {model_id} downloaded to {output_path}")


if __name__ == "__main__":
  flags.mark_flag_as_required("model_id")
  app.run(main)
