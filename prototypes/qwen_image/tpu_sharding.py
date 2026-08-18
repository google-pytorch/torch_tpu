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

#!/usr/bin/env python3
"""TPU Sharding Engine for Qwen 20B Image Model."""

import logging  # PYTHON_LOGGING_OK=Standalone prototype script
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
from prototypes.qwen_image.config import TPUConfig

logger = logging.getLogger("Qwen20B_TPU_Sharding")


def setup_tpu_environment(
    tpu_config: TPUConfig, enable_pjrt_env: bool = True
) -> None:
  """Configures environment variables required for TPU VM execution."""
  if tpu_config.skip_mds_query:
    os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
    os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
    os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")

  if tpu_config.tpu_accelerator_type:
    os.environ.setdefault(
        "TPU_ACCELERATOR_TYPE", tpu_config.tpu_accelerator_type
    )

  if tpu_config.worker_hostnames:
    os.environ.setdefault("TPU_WORKER_HOSTNAMES", tpu_config.worker_hostnames)

  os.environ.setdefault("TPU_WORKER_ID", str(tpu_config.worker_id))
  if enable_pjrt_env:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
  os.environ.setdefault("TPU_NUM_DEVICES", str(tpu_config.num_devices))

  logger.info(
      f"[TPU Env] ACCELERATOR_TYPE={tpu_config.tpu_accelerator_type},"
      f" WORKER_HOSTNAMES={tpu_config.worker_hostnames},"
      f" NUM_DEVICES={tpu_config.num_devices}"
  )


def mark_step_tpu() -> None:
  """Synchronizes pending operations on TPU hardware."""
  if hasattr(torch, "tpu") and torch.tpu.is_available():
    torch.tpu.synchronize()
  elif hasattr(torch, "accelerator") and torch.accelerator.is_available():
    torch.accelerator.synchronize()


def get_tpu_devices() -> List[torch.device]:
  """Resolves and returns all available TPU accelerator devices."""
  try:
    import torch_tpu

    _ = torch_tpu
  except ImportError:
    pass

  if hasattr(torch, "tpu") and torch.tpu.is_available():
    num_devices = torch.tpu.device_count()
    devices = [torch.device(f"tpu:{i}") for i in range(num_devices)]
    logger.info(f"[TorchTPU] Detected {len(devices)} TPU Devices: {devices}")
    return devices

  if hasattr(torch, "accelerator") and torch.accelerator.is_available():
    dev = torch.accelerator.current_accelerator()
    logger.info(f"[PyTorch Accelerator] Device: {dev}")
    return [dev]

  logger.info("[Fallback] TPU hardware not active, using CPU device.")
  return [torch.device("cpu")]


def _move_tensors_to_device(obj, dev: torch.device):
  """Recursively moves tensors within nested tuples, lists, or dicts to `dev`."""
  if isinstance(obj, torch.Tensor):
    return obj.to(dev)
  if isinstance(obj, tuple):
    return tuple(_move_tensors_to_device(x, dev) for x in obj)
  if isinstance(obj, list):
    return [_move_tensors_to_device(x, dev) for x in obj]
  if isinstance(obj, dict):
    return {k: _move_tensors_to_device(v, dev) for k, v in obj.items()}
  return obj


def _find_first_tensor_device(args, kwargs):
  """Finds the device of the first tensor argument in args or kwargs."""
  for a in args:
    if isinstance(a, torch.Tensor):
      return a.device
  for v in kwargs.values():
    if isinstance(v, torch.Tensor):
      return v.device
  return None


class TPUShardingManager:
  """Manages multi-device sharding for the Qwen 20B Transformer model on TPUs."""

  def __init__(self, config: TPUConfig):
    self.config = config
    self.devices = get_tpu_devices()
    pjrt_dev = os.environ.get("PJRT_DEVICE", "")
    is_pjrt_active = pjrt_dev.upper() == "TPU"
    setup_tpu_environment(config, enable_pjrt_env=is_pjrt_active)
    self.num_devices = len(self.devices)
    self.primary_device = self.devices[0]

  def _attach_io_device_hooks(
      self, target_model: nn.Module, input_dev: torch.device
  ) -> None:
    """Attaches hooks to move pipeline inputs to TPU and outputs back to caller device."""
    if getattr(target_model, "_tpu_io_hook_attached", False):
      return

    def pre_hook(module, args, kwargs):
      module._last_input_device = _find_first_tensor_device(args, kwargs)
      new_args = _move_tensors_to_device(args, input_dev)
      new_kwargs = _move_tensors_to_device(kwargs, input_dev)
      return new_args, new_kwargs

    def post_hook(module, args, output):
      _ = args
      mark_step_tpu()
      orig_dev = getattr(module, "_last_input_device", None)
      if orig_dev is not None:
        return _move_tensors_to_device(output, orig_dev)
      return output

    target_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    target_model.register_forward_hook(post_hook)
    target_model._tpu_io_hook_attached = True

  def shard_model(self, model: nn.Module) -> nn.Module:
    """Shards the Qwen 20B Transformer model across TPU devices according to configured strategy."""
    strategy = self.config.sharding_strategy.lower()
    if strategy == "auto":
      strategy = "offload" if self.num_devices == 1 else "spmd"

    logger.info(
        f"[TPUShardingManager] Applying sharding strategy: '{strategy}' "
        f"across {self.num_devices} TPU device(s)."
    )

    pjrt_dev = os.environ.get("PJRT_DEVICE", "")
    if strategy == "single":
      return self._place_single_device(model)
    elif strategy == "offload":
      return self._shard_offload(model)
    elif strategy == "spmd" and (
        pjrt_dev.upper() == "TPU" or self.primary_device.type == "tpu"
    ):
      return self._shard_spmd(model)
    elif strategy == "layer":
      return self._shard_layerwise(model)
    else:
      return self._shard_offload(model)

  def _place_single_device(self, model: nn.Module) -> nn.Module:
    """Places transformer model onto primary TPU device."""
    logger.info(f"Moving model to primary TPU device: {self.primary_device}...")
    target_model = getattr(model, "transformer", model)
    target_model.to(self.primary_device)
    self._attach_io_device_hooks(target_model, self.primary_device)
    mark_step_tpu()
    return model

  def _shard_spmd(self, model: nn.Module) -> nn.Module:
    """Shards model parameters across TPU devices using PyTorch DTensor SPMD mesh sharding."""
    logger.info(
        "[SPMD / Multi-TPU] Applying SPMD Mesh Sharding across"
        f" {self.num_devices} TPU devices..."
    )
    try:
      from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

      target_model = getattr(model, "transformer", model)
      mesh = DeviceMesh("tpu", torch.arange(self.num_devices))

      for _, module in target_model.named_modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
          if param.ndim >= 1 and param.shape[0] % self.num_devices == 0:
            sharded = distribute_tensor(param.data, mesh, [Shard(0)])
            setattr(
                module,
                param_name,
                nn.Parameter(sharded, requires_grad=param.requires_grad),
            )

      self._attach_io_device_hooks(target_model, self.devices[0])
      mark_step_tpu()
      logger.info(
          "[SPMD / Multi-TPU] SPMD Mesh Sharding successfully applied to all"
          " transformer parameters!"
      )
      return model
    except Exception as e:
      logger.warning(
          f"[SPMD / Multi-TPU] Failed to apply SPMD mesh sharding ({e})."
          " Falling back to layerwise/offload sharding."
      )
      if self.num_devices > 1:
        return self._shard_layerwise(model)
      return self._shard_offload(model)

  def _shard_layerwise(self, model: nn.Module) -> nn.Module:
    """Distributes Transformer blocks sequentially across available TPU devices."""
    logger.info(
        f"[Layerwise Sharding] Distributing blocks across {self.num_devices}"
        " TPU devices..."
    )

    target_model = getattr(model, "transformer", model)

    blocks_attr = None
    for attr in [
        "transformer_blocks",
        "blocks",
        "layers",
        "double_blocks",
        "single_blocks",
    ]:
      if hasattr(target_model, attr):
        blocks_attr = attr
        break

    if blocks_attr is None:
      logger.warning(
          "[Layerwise Sharding] Could not locate transformer blocks attribute."
          " Placing on primary device."
      )
      return self._place_single_device(model)

    def make_pre_hook(dev):
      def hook(module, args, kwargs):
        _ = module
        new_args = _move_tensors_to_device(args, dev)
        new_kwargs = _move_tensors_to_device(kwargs, dev)
        mark_step_tpu()
        return new_args, new_kwargs

      return hook

    for attr in [
        "img_in",
        "txt_norm",
        "txt_in",
        "time_text_embed",
        "pos_embed",
        "pos_embedder",
        "patch_embed",
    ]:
      if (
          hasattr(target_model, attr)
          and getattr(target_model, attr) is not None
      ):
        mod = getattr(target_model, attr)
        mod.to(self.devices[0])
        mod.register_forward_pre_hook(
            make_pre_hook(self.devices[0]), with_kwargs=True
        )

    blocks = getattr(target_model, blocks_attr)
    num_blocks = len(blocks)
    blocks_per_device = int(np.ceil(num_blocks / self.num_devices))

    for idx, block in enumerate(blocks):
      target_device_idx = min(idx // blocks_per_device, self.num_devices - 1)
      target_device = self.devices[target_device_idx]
      block.to(target_device)
      block.register_forward_pre_hook(
          make_pre_hook(target_device), with_kwargs=True
      )
      import gc

      gc.collect()
      mark_step_tpu()

    for attr in ["norm_out", "proj_out", "final_layer", "out_proj"]:
      if (
          hasattr(target_model, attr)
          and getattr(target_model, attr) is not None
      ):
        mod = getattr(target_model, attr)
        mod.to(self.devices[-1])
        mod.register_forward_pre_hook(
            make_pre_hook(self.devices[-1]), with_kwargs=True
        )

    self._attach_io_device_hooks(target_model, self.devices[0])
    mark_step_tpu()
    return model

  def _shard_offload(self, model: nn.Module) -> nn.Module:
    """Streams transformer blocks between CPU RAM and TPU device on-demand during forward pass."""
    logger.info(
        "[Offload Sharding] Registering CPU<->TPU block streaming hooks..."
    )

    target_model = getattr(model, "transformer", model)
    all_blocks = []
    if (
        hasattr(target_model, "double_blocks")
        and getattr(target_model, "double_blocks") is not None
    ):
      all_blocks.extend(list(target_model.double_blocks))  # pytype: disable=bad-argument-type
    if (
        hasattr(target_model, "single_blocks")
        and getattr(target_model, "single_blocks") is not None
    ):
      all_blocks.extend(list(target_model.single_blocks))  # pytype: disable=bad-argument-type

    if not all_blocks:
      for attr in ["transformer_blocks", "blocks", "layers"]:
        if (
            hasattr(target_model, attr)
            and getattr(target_model, attr) is not None
        ):
          all_blocks.extend(list(getattr(target_model, attr)))
          break

    if hasattr(model, "enable_sequential_cpu_offload"):
      try:
        model.enable_sequential_cpu_offload(device=self.primary_device)
        return model
      except Exception as e:
        logger.warning(f"Failed to enable sequential CPU offload: {e}")

    if hasattr(model, "enable_model_cpu_offload"):
      try:
        model.enable_model_cpu_offload(device=self.primary_device)
        return model
      except Exception as e:
        logger.warning(f"Failed to enable model CPU offload: {e}")

    from accelerate.hooks import attach_align_device_hook

    for block in all_blocks:
      attach_align_device_hook(
          block, execution_device=self.primary_device, offload=True
      )

    return model
