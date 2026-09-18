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

"""Example of a simple call to torch.nn.Linear"""

from absl import app
import torch

torch._logging.set_logs(aot_graphs=True)  # pylint: disable=protected-access


@torch.compile(backend="aot_eager")
def fwd_bwd(layer_one, layer_two, data):
  output = layer_two(layer_one(data))
  output.sum().backward()


def main(argv):
  del argv
  fwd_bwd(
      layer_one=torch.nn.Linear(64, 64),
      layer_two=torch.nn.functional.relu,
      data=torch.randn(32, 64),
  )
  print("Success.")


if __name__ == "__main__":
  app.run(main)
