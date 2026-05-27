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

"""Prints PyTorch version."""

import torch
from torch_tpu._internal import env


def main() -> None:
  print(f"PyTorch version: {torch.__version__}")
  print(f"TorchTPU built against PyTorch version: {env.PYTORCH_BUILD_VERSION}")


if __name__ == "__main__":
  main()
