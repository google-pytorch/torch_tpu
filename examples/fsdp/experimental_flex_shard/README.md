# Experimental traceable FlexShard executor

This example explores using FlexShard storage with parameter-access collectives
that remain visible to graph capture and activation checkpointing. **TPU
execution is unvalidated.** Keep `simple_fsdp.py` and the existing examples as the
supported paths; this is not a TPU port of the public `flex_shard()` runtime.

## What the example uses

`graph_flex_shard.py` reuses public FlexShard's `BucketSpec`,
`ShardedBucketStorage`, parameter metadata, tied-alias checks, and precision and
reduction policies. It supplies a separate `GraphShard` placement and an
experimental executor with functional all-gather, reduce-scatter, all-reduce,
and `wait_tensor` operations. Parameter properties materialize weights without
execution-time parameter swapping, including inside checkpointed forwards.

Each communication bucket contains one canonical parameter. Tied aliases share
its local storage, but repeated parameter reads can issue repeated collectives.
This does not preserve FlexShard's multi-parameter coalescing, prefetch, or CUDA
overlap. The example imports private core APIs and is pinned to public FlexShard
commit `233e90691a5523023f6cf98c20b4f4529bee9ea4`.

## Run the CPU probes

The core-only probes use a public source checkout, not a vendored snapshot. They
do not exercise FlexShard's CUDA, TorchAO, or Triton integrations. In an isolated
Python 3.12 environment, install the CPU wheel and tools:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/nightly/cpu \
  'torch==2.16.0.dev20261027+cpu'
python -m pip install numpy ruff
FLEX_SHARD_SRC="$(mktemp -d)"
git clone https://github.com/meta-pytorch/flex_shard.git "$FLEX_SHARD_SRC"
git -C "$FLEX_SHARD_SRC" checkout 233e90691a5523023f6cf98c20b4f4529bee9ea4
export PYTHONPATH="$FLEX_SHARD_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
```

From this example directory, run two real CPU/Gloo ranks:

```bash
OMP_NUM_THREADS=1 python launch_cpu.py
OMP_NUM_THREADS=1 python edge_probe.py
python -m ruff check --no-force-exclude *.py
python -m ruff format --check --no-force-exclude *.py
```

The probes compare gradients and updates with an unsharded global-batch model.
They cover dim-0 and nonzero-dimension sharding, replication, BF16 compute with
FP32 reduction, checkpoint wrappers, and forward checkpointing. They capture
full forward and backward graphs, execute an exported joint graph with explicit
parameter inputs, and check that checkpointed backward rematerializes gathers.
The edge probe covers empty local shards, negative dimensions, custom AVG
scaling, tied weights, unused-parameter optimizer behavior, local state-dict
roundtrips, and setup rejection checks. Graph dumps and a summary are written to
`results/`.

The CPU probe has a test-only compatibility flag for older wheels that lack
`_has_grad_dtype_override`; `summary.json` records whether it was needed. The
TPU probe has no such bridge. The joint exporter rejects unused trainable
parameters, so the CPU probe freezes its unused parameter only for joint export;
ordinary compiled training and AdamW separately check its unused behavior.

## Try the TPU smoke test

Use mutually compatible TorchTPU, PyTorch, and public FlexShard builds.
FlexShard's package requires PyTorch >= 2.16.0.dev0. Install the pinned public
FlexShard dependency in that environment, then run `python launch_tpu.py` on one
configured TPU host. The launcher follows the [manual FSDP example](../manual/)
by setting the TPU launch environment before starting workers and detecting the
device count rather than assuming eight workers.

`tpu_smoke.py` uses `tpu_dist`, a 1D TPU `DeviceMesh`, `spmd_safe` annotation,
forward activation checkpointing, and `torch.compile(backend="tpu",
fullgraph=True)`. It is an **unrun smoke test**, not a numerical or performance
benchmark. Before replacing SimpleFSDP, validate byte-backed storage and dtype
views, TPU lowering and execution of collective/autograd graphs, distributed
numerical accuracy, and checkpoint peak memory and performance.

## Use the experimental entry point

For an initialized model on the target device and a 1D data-parallel mesh:

```python
import graph_flex_shard as simple_fsdp

simple_fsdp.data_parallel(model, mesh, mode="fully_shard")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

`data_parallel` defaults to replication and uses SUM gradient reduction, like
SimpleFSDP. Divide the loss by the world size exactly once for average gradients.
The direct `graph_flex_shard(model, buckets)` entry point honors each bucket's
SUM/AVG policy and AVG divide factor.

Initialize identical global parameters on every rank, use parameters read-only
in forward, and create the optimizer after wrapping. Do not move or replace
parameters afterward. Parameter access and backward participation must be
rank-uniform; divergent collective participation can hang.

Not supported: HSDP, outer TP/EP or partial-gradient layouts, `full_dtensor`,
`ignored_params`, offload, deferred synchronization, sparse or `max_norm`
embeddings, coalesced buckets, native FSDP2 parity, or distributed/optimizer
checkpoint integration. Local state-dict checks do not establish distributed
checkpoint compatibility. Explicit nondefault `grad_dtype` behavior is also
unvalidated.
