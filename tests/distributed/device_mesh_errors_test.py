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

"""Tests for topology-aware device mesh error handling."""

import dataclasses
from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu import _loader
from torch_tpu._internal.distributed import device_mesh
from tests import seed_test_utils
from tests.distributed import device_mesh_test_utils


class DeviceMeshErrorsTest(seed_test_utils.MultiProcessRepeatableTest):

  def setUp(self):
    super().setUp()
    _loader._init_device("tpu")  # INIT_DEVICE_OK=CPU runner

  def test_pre_initialization_errors(self):
    # If the process group is not initialized, topology_aware_mesh should
    # raise a RuntimeError.
    with mock.patch("torch.distributed.is_initialized", return_value=False):
      with self.assertRaisesRegex(
          RuntimeError, "process group must be initialized"
      ):
        torch.tpu.topology_aware_mesh(mesh_shape=(8, 8))

  def test_multi_slice_rejects_per_slice_mesh_shape(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "must divide mesh_shape"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(1, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_multi_slice_rejects_mesh_shape_smaller_than_world(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "describes the full mesh"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(2, 4),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_multi_slice_rejects_all_ones_dcn_mesh_shape(self):
    # (1, 1) asks for no DCN split, which two slices cannot satisfy. It has to
    # be rejected rather than falling through to the single-slice path.
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "covers 1 slices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(1, 1),
          ),
      )

  def test_multi_slice_rejects_all_ones_dcn_mesh_shape_3d(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)

    with self.assertRaisesRegex(ValueError, "covers 1 slices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(2, 2, 4),
              dcn_mesh_shape=(1, 1, 1),
          ),
      )

  def test_multi_slice_requires_megascale(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(
        num_slices=2, has_slice_index=False
    )

    with self.assertRaisesRegex(RuntimeError, "megascale_port"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_rejects_partial_slice_index_reporting(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)
    # One rank came up without the MegaScale client, so it cannot say which
    # slice it is on and the job's layout is only partly observable.
    infos[3] = dataclasses.replace(infos[3], has_slice_index=False)

    with self.assertRaisesRegex(RuntimeError, "15 of 16 torch ranks"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(mesh_shape=(2, 8)),
      )

  def test_multi_slice_rejects_ragged_slices(self):
    infos = device_mesh_test_utils.v7x_multislice_infos(num_slices=2)
    # Drop one device from slice 1 and pad slice 0 to keep the world size, so
    # only the per-slice balance check can catch it.
    infos = infos[:9] + infos[10:] + [infos[9]]
    infos[-1] = dataclasses.replace(infos[-1], slice_index=0, id=100099)

    with self.assertRaisesRegex(ValueError, "same number of devices"):
      device_mesh_test_utils.patched_call(
          infos,
          lambda: torch.tpu.topology_aware_mesh(
              mesh_shape=(2, 8),
              dcn_mesh_shape=(2, 1),
          ),
      )

  def test_infer_dcn_mesh_shape_refuses_to_split_shard_axis(self):
    # dp_replicate=2 cannot absorb 4 slices. Falling through to dp_shard would
    # split shard groups across DCN, so this has to be refused instead.
    with self.assertRaisesRegex(ValueError, "Refusing to fall back"):
      device_mesh._infer_dcn_mesh_shape((1, 2, 64, 1, 1), 4)

  def test_infer_dcn_mesh_shape_rejects_all_trivial_mesh(self):
    with self.assertRaisesRegex(ValueError, "every axis has extent 1"):
      device_mesh._infer_dcn_mesh_shape((1, 1, 1), 4)

  def test_infer_dcn_mesh_shape_rejects_indivisible_mesh(self):
    with self.assertRaisesRegex(ValueError, "Cannot spread mesh_shape"):
      device_mesh._infer_dcn_mesh_shape((1, 3, 5, 1, 1), 4)


if __name__ == "__main__":
  absltest.main()
