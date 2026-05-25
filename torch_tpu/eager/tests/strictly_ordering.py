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

"""A sample graph to ensure that we're strictly ordering execution.

We want to ensure that we're strictly ordering execution, if we call multiple
graphs in a row, we want to greedily materialize possible outputs in the first
iteration of the graph prior to execution one node of the next graph, as not
following that mechanism can lead to large memory explosions.
"""

from absl import app
import torch
from torch_tpu._internal import sync


def main(argv):
  del argv
  device = torch.device("tpu")

  print(f"Running on device: {device}")

  inputs = torch.ones((128, 128), dtype=torch.float32, device=device)

  # Create a common deferred operation to ensure both outputs are in the same
  # subgraph.
  common = inputs + 0

  output_a = common + 1
  output_b = common + 2

  print(f"Output: {output_b}")

  # Ensure that the output_a has already been materialized because it is a leaf
  # in the same subgraph as output_b.
  assert sync.is_materializing(output_a)


if __name__ == "__main__":
  app.run(main)
