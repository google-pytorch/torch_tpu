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

"""TorchComms backend and communicator implementation for TorchTPU.

Provides the `TorchCommsTPU` class and registration routines for integrating
TorchTPU with PyTorch's `torchcomms` subsystem and `torch.distributed`.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

from absl import logging
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

try:
  from torch_tpu._internal.distributed import tpu_distributed
except ImportError:
  tpu_distributed = None  # pyrefly: ignore[assignment]


class TorchCommsTPU:
  """TorchComms communicator implementation for TPU devices.

  Wraps the underlying `ProcessGroupTpu` backend and satisfies the TorchComms /
  `_TorchComm` protocol, supporting communicator lifecycle management, rank/size
  queries, device mapping, and collective communication delegation.
  """

  def __init__(
      self,
      backend_name: str = "tpu",
      device: torch.device | str | None = None,
      name: str | None = None,
      store: dist.Store | None = None,
      hints: dict[str, Any] | None = None,
      group_rank: int | None = None,
      group_size: int | None = None,
      pg: dist.ProcessGroup | None = None,
      timeout: datetime.timedelta | None = None,
  ):
    self._backend_name = str(backend_name).lower()

    # Resolve device
    if device is None:
      try:
        self._device = torch.device("tpu:0")
      except RuntimeError:
        self._device = torch.device("privateuseone:0")
    elif isinstance(device, str):
      try:
        self._device = torch.device(device)
      except RuntimeError:
        self._device = torch.device(
            device.replace("tpu", "privateuseone")
            if "tpu" in device
            else device
        )
    else:
      self._device = device

    # Resolve rank and size
    if group_rank is not None:
      self._rank = int(group_rank)
    else:
      self._rank = int(
          os.environ.get(
              "TORCHCOMM_RANK",
              os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")),
          )
      )

    if group_size is not None:
      self._size = int(group_size)
    else:
      self._size = int(
          os.environ.get(
              "TORCHCOMM_SIZE",
              os.environ.get("WORLD_SIZE", "1"),
          )
      )

    self._name = str(name or f"torchcomm_tpu_{self._rank}_{self._size}")
    self._store = store
    self._hints = dict(hints or {})
    self._timeout = timeout or datetime.timedelta(seconds=30)
    self._finalized = False

    # Initialize or assign underlying ProcessGroup backend
    if pg is not None:
      self._pg = pg
    elif tpu_distributed is not None and self._store is not None:
      try:
        self._pg = tpu_distributed.create_process_group(
            self._store, self._rank, self._size, self._timeout
        )
      except Exception as e:
        logging.warning("Could not instantiate ProcessGroupTpu: %s", e)
        self._pg = None
    elif dist.is_initialized():
      try:
        self._pg = (
            c10d._get_default_group()
            if hasattr(c10d, "_get_default_group")
            else dist.group.WORLD
        )
      except Exception:
        self._pg = None
    else:
      self._pg = None

  @property
  def rank(self) -> int:
    """The rank of this process within the communicator."""
    return self._rank

  @property
  def size(self) -> int:
    """The total number of processes in the communicator."""
    return self._size

  @property
  def device(self) -> torch.device:
    """The TPU device bound to this communicator."""
    return self._device

  @property
  def name(self) -> str:
    """The unique name of the communicator."""
    return self._name

  @property
  def backend_name(self) -> str:
    """The backend name string ('tpu')."""
    return self._backend_name

  @property
  def device_type(self) -> str:
    """The device type string ('tpu')."""
    return self._device.type

  @property
  def hints(self) -> dict[str, Any]:
    """Communicator hints dictionary."""
    return dict(self._hints)

  @property
  def store(self) -> dist.Store | None:
    """The underlying rendezvous store if provided."""
    return self._store

  @property
  def timeout(self) -> datetime.timedelta:
    """Communicator timeout duration."""
    return self._timeout

  def get_rank(self) -> int:
    """Return the communicator rank."""
    return self._rank

  def get_size(self) -> int:
    """Return the communicator world size."""
    return self._size

  def get_device(self) -> torch.device:
    """Return the bound device."""
    return self._device

  def get_name(self) -> str:
    """Return the communicator name."""
    return self._name

  def get_backend_name(self) -> str:
    """Return the backend name string."""
    return self._backend_name

  def get_backend(self) -> dist.ProcessGroup | None:
    """Return the underlying backend object."""
    return self._pg

  def unsafe_get_backend(self) -> dist.ProcessGroup | None:
    """Directly access the underlying backend object."""
    return self._pg

  def is_finalized(self) -> bool:
    """Check if the communicator has been finalized/closed."""
    return self._finalized

  def finalize(self) -> None:
    """Release communicator resources and mark as finalized."""
    if self._finalized:
      return
    self._finalized = True
    self._pg = None

  def __enter__(self) -> TorchCommsTPU:
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.finalize()

  def split(self, color: int, key: int) -> TorchCommsTPU:
    """Create a new subgroup communicator based on color and key."""
    if self._finalized:
      raise RuntimeError("Cannot split a finalized communicator.")
    subgroup_name = f"{self._name}_split_{color}_{key}"
    return TorchCommsTPU(
        backend_name=self._backend_name,
        device=self._device,
        name=subgroup_name,
        store=self._store,
        hints=self._hints,
        group_rank=key,
        group_size=self._size,
    )

  def _check_not_finalized(self) -> None:
    if self._finalized:
      raise RuntimeError(
          f"TorchCommsTPU '{self._name}' has already been finalized."
      )

  # ---------------- Collective Operations ---------------- #

  def all_reduce(
      self,
      tensor: torch.Tensor,
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.AllreduceOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    return self._pg.allreduce([tensor], opts)

  def all_gather(
      self,
      tensor_list: list[torch.Tensor],
      tensor: torch.Tensor,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.AllgatherOptions()
    opts.asyncOp = async_op
    return self._pg.allgather([tensor_list], [tensor], opts)

  def all_gather_single(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.AllgatherOptions()
    opts.asyncOp = async_op
    return self._pg._allgather_base(output, input, opts)

  def all_gather_into_tensor(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      async_op: bool = False,
  ) -> Any:
    """Alias for all_gather_single."""
    return self.all_gather_single(output, input, async_op=async_op)

  def reduce_scatter(
      self,
      output: torch.Tensor,
      tensor_list: list[torch.Tensor],
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    return self._pg.reduce_scatter([output], [tensor_list], opts)

  def reduce_scatter_single(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.ReduceScatterOptions()
    opts.reduceOp = op
    opts.asyncOp = async_op
    return self._pg._reduce_scatter_base(output, input, opts)

  def reduce_scatter_tensor(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    """Alias for reduce_scatter_single."""
    return self.reduce_scatter_single(output, input, op=op, async_op=async_op)

  def broadcast(
      self,
      tensor: torch.Tensor,
      src: int = 0,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.BroadcastOptions()
    opts.rootRank = src
    opts.asyncOp = async_op
    return self._pg.broadcast([tensor], opts)

  def barrier(self, async_op: bool = False) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.BarrierOptions()
    try:
      return self._pg.barrier(opts)
    except Exception:
      if dist.is_initialized():
        return dist.barrier()
      raise

  def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    if hasattr(self._pg, "experimental_send"):
      return self._pg.experimental_send([tensor], dst, tag)
    raise NotImplementedError("P2P send is not supported on this TPU backend.")

  def recv(self, tensor: torch.Tensor, src: int, tag: int = 0) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    if hasattr(self._pg, "experimental_recv"):
      return self._pg.experimental_recv([tensor], src, tag)
    raise NotImplementedError("P2P recv is not supported on this TPU backend.")

  def all_to_all(
      self,
      output_list: list[torch.Tensor],
      input_list: list[torch.Tensor],
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.AllToAllOptions()
    opts.asyncOp = async_op
    return self._pg.alltoall(output_list, input_list, opts)

  def all_to_all_single(
      self,
      output: torch.Tensor,
      input: torch.Tensor,
      output_split_sizes: list[int] | None = None,
      input_split_sizes: list[int] | None = None,
      async_op: bool = False,
  ) -> Any:
    self._check_not_finalized()
    if self._pg is None:
      raise RuntimeError(
          "Underlying ProcessGroupTpu is not initialized on TorchCommTPU."
      )
    opts = c10d.AllToAllOptions()
    opts.asyncOp = async_op
    return self._pg.alltoall_base(
        output, input, output_split_sizes or [], input_split_sizes or [], opts
    )

  # ---------------- Stubs for Unsupported / Future APIs ---------------- #

  def create_pair_comm(
      self, peer_rank: int, hints: dict[str, Any] | None = None
  ) -> Any:
    """Creates a dedicated 1-to-1 PairComm with a peer rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "PairComm / dedicated P2P communicator is not yet implemented for"
        " TorchCommTPU (scheduled for Step 5)."
    )

  def split_group(
      self, ranks: list[int], hints: dict[str, Any] | None = None
  ) -> TorchCommTPU:
    """Creates a subgroup communicator for an explicit list of ranks."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Explicit rank subset subgroup splitting is not yet implemented for"
        " TorchCommTPU (scheduled for Step 4)."
    )

  def reconfigure(
      self,
      ranks: list[int] | None = None,
      size: int | None = None,
      timeout: Any = None,
  ) -> None:
    """Dynamically reconfigures communicator topology or rank membership."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Dynamic communicator reconfiguration is not yet implemented for"
        " TorchCommTPU (scheduled for Step 7)."
    )

  def create_window(
      self, tensor: torch.Tensor, hints: dict[str, Any] | None = None
  ) -> Any:
    """Creates a one-sided RMA Window allocation over a tensor."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Window / SymmetricMemory primitives are not yet supported for"
        " TorchCommTPU (scheduled for Step 6)."
    )

  def map_remote_tensor(
      self, tensor: torch.Tensor, peer_rank: int
  ) -> torch.Tensor:
    """Maps a remote rank's tensor memory into local address space."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Remote tensor memory mapping is not yet supported for TorchCommTPU"
        " (scheduled for Step 6)."
    )

  def create_batch(self) -> Any:
    """Creates a batch handle to coalesce collective operations."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Batched collective execution is not yet implemented for TorchCommTPU."
    )

  def all_reduce_coalesced(
      self,
      tensors: list[torch.Tensor],
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    """Coalesced all-reduce across a list of tensors."""
    self._check_not_finalized()
    raise NotImplementedError(
        "all_reduce_coalesced is not yet implemented for TorchCommTPU."
    )

  def all_gather_coalesced(
      self,
      output_lists: list[list[torch.Tensor]],
      input_list: list[torch.Tensor],
      async_op: bool = False,
  ) -> Any:
    """Coalesced all-gather across lists of tensors."""
    self._check_not_finalized()
    raise NotImplementedError(
        "all_gather_coalesced is not yet implemented for TorchCommTPU."
    )

  def reduce_scatter_coalesced(
      self,
      output_list: list[torch.Tensor],
      input_lists: list[list[torch.Tensor]],
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    """Coalesced reduce-scatter across lists of tensors."""
    self._check_not_finalized()
    raise NotImplementedError(
        "reduce_scatter_coalesced is not yet implemented for TorchCommTPU."
    )

  def all_to_all_coalesced(
      self,
      output_lists: list[list[torch.Tensor]],
      input_lists: list[list[torch.Tensor]],
      async_op: bool = False,
  ) -> Any:
    """Coalesced all-to-all across lists of tensors."""
    self._check_not_finalized()
    raise NotImplementedError(
        "all_to_all_coalesced is not yet implemented for TorchCommTPU."
    )

  def reduce(
      self,
      tensor: torch.Tensor,
      dst: int = 0,
      op: dist.ReduceOp = dist.ReduceOp.SUM,
      async_op: bool = False,
  ) -> Any:
    """Reduces a tensor to a single destination root rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "Point-to-point reduce to root is not yet implemented for TorchCommTPU."
    )

  def gather(
      self,
      output_list: list[torch.Tensor] | None = None,
      input_tensor: torch.Tensor | None = None,
      dst: int = 0,
      async_op: bool = False,
  ) -> Any:
    """Gathers tensors from all ranks into output_list on dst rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "gather to root rank is not yet implemented for TorchCommTPU."
    )

  def scatter(
      self,
      output_tensor: torch.Tensor,
      scatter_list: list[torch.Tensor] | None = None,
      src: int = 0,
      async_op: bool = False,
  ) -> Any:
    """Scatters a list of tensors from src rank to all ranks."""
    self._check_not_finalized()
    raise NotImplementedError(
        "scatter from root rank is not yet implemented for TorchCommTPU."
    )

  def monitored_barrier(
      self, timeout: Any = None, wait_all_ranks: bool = True
  ) -> Any:
    """Monitored health-check barrier across ranks."""
    self._check_not_finalized()
    raise NotImplementedError(
        "monitored_barrier is not yet implemented for TorchCommTPU."
    )

  def register_flight_recorder_hook(self, hook: Any) -> None:
    """Registers a FlightRecorder tracing hook."""
    self._check_not_finalized()
    raise NotImplementedError(
        "FlightRecorder tracing hook is not yet implemented for TorchCommTPU"
        " (scheduled for Step 8)."
    )

  def get_flight_recorder_traces(self) -> list[dict[str, Any]]:
    """Returns recent circular buffer collective traces."""
    self._check_not_finalized()
    raise NotImplementedError(
        "FlightRecorder tracing is not yet implemented for TorchCommTPU"
        " (scheduled for Step 8)."
    )

  def broadcast_object_list(
      self,
      object_list: list[Any],
      src: int = 0,
      device: torch.device | None = None,
  ) -> None:
    """Broadcasts a list of picklable Python objects from src rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "broadcast_object_list is not yet implemented for TorchCommTPU."
    )

  def all_gather_object(
      self,
      object_list: list[Any],
      obj: Any,
  ) -> None:
    """All-gathers picklable Python objects across all ranks."""
    self._check_not_finalized()
    raise NotImplementedError(
        "all_gather_object is not yet implemented for TorchCommTPU."
    )

  def gather_object(
      self,
      obj: Any,
      object_gather_list: list[Any] | None = None,
      dst: int = 0,
  ) -> None:
    """Gathers picklable Python objects to dst rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "gather_object is not yet implemented for TorchCommsTPU."
    )

  def scatter_object_list(
      self,
      scatter_object_output_list: list[Any],
      scatter_object_input_list: list[Any] | None = None,
      src: int = 0,
  ) -> None:
    """Scatters a list of picklable Python objects from src rank."""
    self._check_not_finalized()
    raise NotImplementedError(
        "scatter_object_list is not yet implemented for TorchCommsTPU."
    )


def create_torchcomms_tpu(
    backend_str: str,
    device: torch.device | str | None,
    name: str | None = None,
    store: dist.Store | None = None,
    hints: dict[str, Any] | None = None,
) -> TorchCommsTPU:
  """Factory function to instantiate a TorchCommsTPU communicator."""
  return TorchCommsTPU(
      backend_name=backend_str,
      device=device,
      name=name,
      store=store,
      hints=hints,
  )


# Aliases for compatibility
TorchCommTPU = TorchCommsTPU
create_torchcomm_tpu = create_torchcomms_tpu


def register_torchcomms_tpu() -> bool:
  """Registers the TPU backend with torchcomms and torch.distributed.Backend."""
  registered = False

  # 1. Register with torchcomms if available
  try:
    import torchcomms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error # pyrefly: ignore[missing-import]

    if hasattr(torchcomms, "register_backend"):
      torchcomms.register_backend("tpu", create_torchcomm_tpu, devices=["tpu"])
      torchcomms.register_backend(
          "tpu_dist", create_torchcomm_tpu, devices=["tpu"]
      )
      registered = True
  # TODO: b/537290986 - Fix API mismatch/version skew between torch_tpu and
  # internal torchcomms. Handled TypeError silently to restore legacy behavior
  # unblocked by src/layout migration.
  except (ImportError, AttributeError, TypeError):
    pass

  # 2. Register with PyTorch c10d Backend
  try:
    if tpu_distributed is not None:
      for name in ("tpu", "tpu_dist"):
        try:
          torch.distributed.Backend.register_backend(
              name, tpu_distributed.create_process_group, devices=["tpu"]
          )
        except Exception:
          pass
      if hasattr(torch.distributed.Backend, "default_device_backend_map"):
        torch.distributed.Backend.default_device_backend_map["tpu"] = "tpu"
  except Exception:
    pass

  return registered


def is_torchcomms_registered(backend_name: str = "tpu") -> bool:
  """Check whether the given backend is registered in torchcomms."""
  try:
    import torchcomms  # pylint: disable=g-import-not-at-top # pytype: disable=import-error # pyrefly: ignore[missing-import]

    if hasattr(torchcomms, "_is_backend_registered"):
      return torchcomms._is_backend_registered(backend_name)
    if hasattr(torchcomms, "is_backend_registered"):
      return torchcomms.is_backend_registered(backend_name)
    if hasattr(torchcomms, "is_backend_built"):
      return torchcomms.is_backend_built(backend_name)
  except (ImportError, AttributeError):
    pass
  return False
