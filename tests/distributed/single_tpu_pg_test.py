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

"""Basic single TPU tests for TPU process group backend."""

import datetime

from absl import logging
from absl.testing import absltest
import portpicker
import torch
from torch_tpu._internal.distributed import tpu_distributed


class SingleTpuPGTest(absltest.TestCase):

  def test_init_pg(self):
    """Test registration and initialization call from PyTorch."""

    device = torch.device("tpu")
    logging.info("TPU device: %s", device)
    logging.info("global_device_id: %s", tpu_distributed.global_device_id())
    logging.info(
        "device_debug_string: %s", tpu_distributed.device_debug_string()
    )

    # `tpu_dist` is auto-registered by torch_tpu's loader when the TPU device
    # is initialized; no manual register_backend() call is needed here.
    self.assertTrue(torch.distributed.is_backend_available("tpu_dist"))

    # Standard initialization of a PyTorch ProcessGroup.
    torch.distributed.init_process_group(
        backend="tpu_dist",
        init_method=f"tcp://localhost:{portpicker.pick_unused_port()}",
        rank=0,
        world_size=1,
    )
    self.assertTrue(torch.distributed.is_initialized())

  def test_manual_pg(self):
    """Test manual/explicit ProcessGroup creation."""

    device = torch.device("tpu")
    logging.info("TPU device: %s", device)

    store = torch.distributed.TCPStore(
        host_name="localhost",
        port=portpicker.pick_unused_port(),
        world_size=1,
        is_master=True,
    )
    pg = tpu_distributed.create_process_group(
        store, 0, 1, datetime.timedelta(seconds=5)
    )
    logging.info("ProcessGroup: %s", pg)
    self.assertEqual(pg._get_backend_name(), "tpu_dist")


if __name__ == "__main__":
  absltest.main()
