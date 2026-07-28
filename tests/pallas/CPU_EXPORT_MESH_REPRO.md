# Pallas CPU export mesh repro

## Problem

`jax.shard_map` needs a global JAX mesh to export a Pallas kernel that uses
mesh-relative device IDs. The existing TorchTPU communication example builds
that mesh with `jax.make_mesh`, which enumerates physical JAX devices and
initializes the JAX TPU backend.

TorchTPU applications do not otherwise need a JAX TPU client. They use JAX to
trace and export StableHLO, then register the serialized module with TorchTPU.
In applications such as SGLang, TorchTPU has already established the physical
workers, global ranks, and TP/DP/EP process groups.

The proposed experiment uses distinct JAX CPU devices as export-only mesh
placeholders. The exported module still targets TPU:

1. TorchTPU initializes the physical runtime and assigns each worker its rank
   and physical partition ID.
2. Every Torch worker creates the same ordered logical CPU mesh.
3. `shard_map` uses that mesh to assign global partitions and lower
   mesh-relative ICI device IDs.
4. `jax.export(..., platforms=["tpu"])` produces a global TPU StableHLO module.
5. `pallas.jax_op` registers that module with TorchTPU.
6. TorchTPU partitions and executes it on the physical TPU runtime.

There is no runtime association between a CPU `Mesh` object and a Torch process
group, and axis names do not create one. TorchTPU partitions the global module
using physical partition IDs. Those IDs are not necessarily equal to Torch
global ranks; on the TPU v7 host used for this repro, for example, Torch rank 2
owned physical partition 4 and rank 6 owned physical partition 2.

The current single-axis implementation therefore relies on this contract:

```text
logical export partition i <=> TorchTPU physical partition ID i
```

That implicit contract is the main issue this repro is intended to expose. A
long-term API should construct the export mesh from an explicit mapping between
logical axes, Torch process-group ranks, and physical partition IDs, without
initializing the JAX TPU backend. It should validate that mapping when the
exported module is registered. This is particularly important for
multidimensional TP/DP/EP meshes: canonically sorting Torch ranks is incorrect
when rank order and physical partition order differ.

## Repros

With TorchTPU initialized first in both cases, compare physical JAX device
discovery with CPU logical-mesh construction:

```bash
python3 tests/pallas/jax_mesh_runtime_init_repro.py \
  --mode=physical-jax-devices --world-size=8
python3 tests/pallas/jax_mesh_runtime_init_repro.py \
  --mode=cpu-export-mesh --world-size=8
```

Export the same global program in every Torch worker and print both the logical
mesh ordinal and TorchTPU runtime identity:

```bash
python3 tests/pallas/jax_cpu_mesh_multihost_repro.py \
  --launch-singlehost=8 --check-torchtpu
```

The export-only repro proves that the serialized global module can be produced
without a JAX TPU backend. The SGLang fused-MoE integration additionally proves
execution of a CPU-mesh-exported Pallas program containing ICI communication.
This branch does not yet provide the proposed rank-order validation API.

CPU-only export must be configured before JAX initializes a backend because
JAX backend selection is process-global and cached. The implementation rejects
late configuration rather than silently querying or initializing the TPU
backend. This is a constraint documented and tested by the PR, not an alternate
runtime ordering proposed by this repro.
