# Distributed Launchers examples

TorchTPU distributed requires the following variables to be set prior to
initializing a TorchTPU process. Most of these variables are set by torchrun
when launching local processes:

```bash
export TORCH_TPU_SLICEBUILDER_ADDRESSES=...
export TORCH_TPU_TOPOLOGY=...

torchrun --nproc_per_node=8 script.py
```

*   `WORLD_SIZE`: The number of participating processes
*   `RANK`: The individual process' id in the world
*   `LOCAL_RANK`: For a given node (host), which rank is this process
*   `GROUP_RANK`: Which node (host) are we running on
*   `MASTER_ADDR`: The coordination service's address
*   `MASTER_PORT`: The coordination service's port

The variables that must be set for a TorchTPU environment are:

*   `TORCH_TPU_SLICEBUILDER_ADDRESSES`: The addresses and ports to be used for
    creating the multi-TPU ICI network (one per WORLD_SIZE)
*   `TORCH_TPU_TOPOLOGY`: The global TPU topology to be used when creating this
    mesh
