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

"""Prints the installed PyTorch and the PyTorch the loaded torch_tpu targets.

Run in an environment with the torch_tpu wheel installed. The multi-ABI wheel
bundles one extension glue per supported PyTorch release and picks one at
import time, so besides `torch.__version__` this reports which glue was
loaded and the PyTorch release its headers came from, making it obvious in CI
logs which combination a test run actually exercised.
"""

import torch
from torch_tpu._internal import env


def main() -> None:
  print(f"PyTorch version: {torch.__version__}")
  print(f"TorchTPU extension loaded from: {env.__file__}")
  print(f"TorchTPU built against PyTorch version: {env.PYTORCH_BUILD_VERSION}")


if __name__ == "__main__":
  main()
