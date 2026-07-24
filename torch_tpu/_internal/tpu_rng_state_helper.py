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

# pylint: disable=protected-access
"""Runtime override of PyTorch RNG functionalization for TPU."""

import contextlib
import torch
import torch._prims_common
from torch._subclasses import fake_tensor


class TpuRngStateHelper:
  """Helper class to get and set TPU RNG state during functionalization."""

  @staticmethod
  def get_torch_state_as_tuple(fake_mode=contextlib.nullcontext()):
    """Gets the TPU RNG state as a (seed, offset) tuple."""
    with fake_tensor.unset_fake_temporarily():
      seed_val, offset_val = (
          torch.tpu.get_rng_state().view(torch.int64).tolist()
      )
    with fake_mode:
      seed = torch.tensor(seed_val, device="tpu")
      offset = torch.tensor(offset_val, device="tpu")
      return seed, offset

  @staticmethod
  def set_torch_state_tensor(seed, offset):
    """Sets the TPU RNG state from seed and offset tensors."""
    if isinstance(seed, fake_tensor.FakeTensor) or isinstance(
        offset, fake_tensor.FakeTensor
    ):
      return
    seed_portion = seed.reshape([1]).view(torch.uint8)
    offset_portion = offset.reshape([1]).view(torch.uint8)
    new_state = torch.cat([seed_portion, offset_portion])
    torch.tpu.set_rng_state(new_state)

  @staticmethod
  def set_new_offset(relative_offset):
    """Sets a new offset in the TPU RNG state."""
    if isinstance(relative_offset, fake_tensor.FakeTensor):
      return
    with fake_tensor.unset_fake_temporarily():
      state = torch.tpu.get_rng_state().view(torch.int64)
      state[1] = relative_offset.item()
      torch.tpu.set_rng_state(state.view(torch.uint8))


def enable_tpu_rng_functionalization():
  """Enables PyTorch RNG functionalization support for TPU."""
  original_helper = torch._prims_common.CUDARngStateHelper

  def override_method(method_name, tpu_implementation):
    original_method = getattr(original_helper, method_name)

    @staticmethod
    def overridden(*args, **kwargs):
      if (
          not torch.cuda.is_available()
          and hasattr(torch, "tpu")
          and torch.tpu.is_available()
      ):
        return tpu_implementation(*args, **kwargs)
      return original_method(*args, **kwargs)

    setattr(original_helper, method_name, overridden)

  for method in (
      "get_torch_state_as_tuple",
      "set_torch_state_tensor",
      "set_new_offset",
  ):
    override_method(method, getattr(TpuRngStateHelper, method))
