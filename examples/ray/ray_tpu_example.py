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

"""Example case using Ray on Torch TPU allocating individual TPU cores to actors."""

import os
from absl import app
import ray


@ray.remote(
    resources={"TPU": 0.5}
)  # Request half a TPU chip (1 core on a 2-core v7 chip)
class TPUActor:
  """Ray Actor that allocates and binds to a specific physical TPU core."""

  def __init__(self, chip_index: int, core_index: int):
    # Set standard PJRT environment variables to bind this actor process
    # to the specific TPU chip and core before PyTorch/XLA initializes!
    os.environ["TPU_VISIBLE_DEVICES"] = str(chip_index)
    # PJRT_LOCAL_DEVICE_IDS partitions the local host devices.
    # On v7 TPU VMs, Core 0 is index 0 and Core 1 is index 1.
    os.environ["PJRT_LOCAL_DEVICE_IDS"] = str(core_index)

    # Import torch inside the actor after setting env vars!
    # pylint: disable=g-import-not-at-top
    import torch
    # pylint: enable=g-import-not-at-top

    self.device = torch.device("tpu")
    print(
        f"TPUActor initialized on Chip {chip_index} Core {core_index} ->"
        f" device: {self.device}",
        flush=True,
    )

  def run_operation(self):
    import torch  # pylint: disable=g-import-not-at-top
    # Execute operation on the designated TPU core
    x = torch.ones(3, 3, device=self.device)
    y = x * 3
    return y.cpu().tolist()


def main(argv):
  del argv  # Unused
  print("Initializing Ray...", flush=True)
  if not ray.is_initialized():
    ray.init()

  print("\nCluster resources:", ray.cluster_resources())

  print("\nInstantiating TPU Actors...")
  # Allocate 2 independent actors sharing Chip 0 (Core 0 and Core 1)
  actor_0 = TPUActor.remote(chip_index=0, core_index=0)
  actor_1 = TPUActor.remote(chip_index=0, core_index=1)

  print("\nSubmitting parallel TPU operations...")
  fut_0 = actor_0.run_operation.remote()
  fut_1 = actor_1.run_operation.remote()

  print("\nWaiting for results...")
  res_0, res_1 = ray.get([fut_0, fut_1])

  print(f"Actor 0 result: {res_0}")
  print(f"Actor 1 result: {res_1}")

  assert res_0 == [[3.0] * 3] * 3, "Actor 0 verification failed"
  assert res_1 == [[3.0] * 3] * 3, "Actor 1 verification failed"
  print("\nRay TPU core example completed successfully!", flush=True)


if __name__ == "__main__":
  app.run(main)
