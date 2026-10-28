"""Launch the unvalidated FlexShard smoke test on one TPU host.

Match the manual FSDP example's launch setup: initialize the environment before
starting workers, and run one worker per detected device. No TPU run has been
performed for this experimental example.
"""

import torch
from torch.distributed import launcher
from torch.tpu import distributed

import tpu_smoke


def main():
  distributed.set_tpu_launch_env()
  config = launcher.LaunchConfig(
    min_nodes=1,
    max_nodes=1,
    nproc_per_node=torch.accelerator.device_count(),
    rdzv_backend="c10d",
    rdzv_endpoint="localhost:0",
    max_restarts=0,
  )
  launcher.elastic_launch(config, tpu_smoke.main)()


if __name__ == "__main__":
  main()
