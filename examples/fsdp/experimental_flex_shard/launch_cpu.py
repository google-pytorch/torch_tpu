"""Local file-store launcher, avoiding container-host rendezvous addresses."""

import os
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp

import probe


def worker(rank, path, world_size):
  os.environ["GLOO_SOCKET_IFNAME"] = "lo"
  dist.init_process_group(
    "gloo", rank=rank, world_size=world_size, init_method=f"file://{path}"
  )
  probe.main()


if __name__ == "__main__":
  with tempfile.TemporaryDirectory() as directory:
    mp.start_processes(
      worker, args=(f"{directory}/store", 2), nprocs=2, start_method="fork"
    )
