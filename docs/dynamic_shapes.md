# Dynamic shapes

<!-- Link to colab and to generated notebook -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

When you compile a model with `torch.compile(model)`, TorchTPU lowers the
computation to StableHLO and compiles it with the XLA compiler (see
[How TorchTPU works](./how_torchtpu_works.md)). The XLA compiler is
traditionally designed around **statically known tensor shapes** so that High
Bandwidth Memory (HBM) buffers can be laid out ahead of time, operator pipelines
can be fused, and matrix multiplications can be tiled for the TPU's systolic
arrays (MXUs). When input tensor dimensions vary across calls in a
`torch.compile` workload, compiling for each exact shape would invalidate
Dynamo's shape guards and trigger a costly full XLA recompilation on every new
shape.

To support **dynamic shapes** without constant recompilations, TorchTPU uses
**Bounded Dynamism**, where the compile-time upper bounds of dynamic dimensions
are fixed while the actual runtime dimension sizes are tracked dynamically by
XLA.

## How dynamic shapes are handled by TorchTPU

When you compile a model or function with default dynamism, that is
`torch.compile(model, dynamic=None)` (as input shapes vary across calls) or
explicitly mark dynamic dimensions with `torch._dynamo.mark_dynamic`, Dynamo
traces the computation and produces an FX graph where changing dimensions are
represented as symbolic integer variables (`torch.SymInt`) instead of fixed
constants. TorchTPU handles these symbolic dimensions as follows:

### Compilation

*   TorchTPU intercepts the traced FX graph, determines an overall
    `[lower_bound, upper_bound]` range for each symbolic dimension (from
    `mark_dynamic(..., min=..., max=...)` or inferred by Dynamo from the initial
    runtime size `S`), and divides that range into contiguous buckets aligned to
    multiples of `bucket_size` (default `128`, matching the TPU's 128-lane tile
    dimension), up to `num_dynamic_buckets` (default `1`, maximum `8`):
    *   The first bucket covers from `lower_bound` up to the next multiple of
        `128` at or after `S` (`U_1`), the second bucket covers from `U_1 + 1`
        up to the next multiple of `128` (`U_2`), and so on (for example,
        `[lower_bound, 128]`, `[129, 256]`, ...).
    *   When a dimension has a finite `upper_bound`: if the first bucket already
        reaches `upper_bound` (or `num_dynamic_buckets == 1`), that bucket is
        capped at `upper_bound`; if `upper_bound` is reached in a later bucket,
        the cap is rounded up to the `128k - c` bucket grid (where `c` is the
        constant offset of output shapes such as `S + 1`) so output layouts stay
        consistent across bucket switches.
    *   TorchTPU registers a single range guard across the full span of all
        buckets with Dynamo, so Dynamo only triggers recompilation if the
        dimension grows beyond the overall `upper_bound`.
*   For each bucket, TorchTPU replaces the symbolic dimension with the bucket's
    static upper bound `U`, passes the actual runtime dimension size as an
    input, and lowers the graph to StableHLO with bounded tensor types that
    track both the static upper bound `U` and the active runtime size across all
    operations.
*   The resulting module is compiled by XLA for each configured bucket upper
    bound `U`.

### Execution

*   At runtime, based on the actual size of the dynamic dimension, TorchTPU
    selects the corresponding compiled bucket executable.

*   When executing the compiled module, XLA pads the dynamic input tensor up to
    the bucket's static upper bound `U` and resets values in the padded region
    beyond the active runtime size to algebraic identity values, ensuring that
    padded elements do not affect the output result.

*   For dynamic output tensors, padding remains at the end of the buffer,
    allowing TorchTPU to return the buffer to PyTorch with a lightweight shape
    metadata update rather than physically slicing the tensor.

## Motivating use case: Autoregressive decoding

A primary workload that motivates bounded dynamism is **autoregressive LLM
decoding** with a dynamically growing KV cache. In a standard decode loop, each
step appends the new token's key and value states to the cached history along
the sequence dimension via `torch.cat`:

*   **Prefill:** Input prompt has shape `[B, S]` (batch size `B`, prompt
    sequence length `S`), producing a KV cache of shape `[B, H_kv, S, D]`
    (`H_kv` key/value heads, head dimension `D`).
*   **Decode step 1:** Input token has shape `[B, 1]`, concatenating with the
    cache to produce shape `[B, H_kv, S + 1, D]`.
*   **Decode step `t`:** The KV cache sequence length grows to `S + t`.

With static-shape compilation (`dynamic=False`), Dynamo guards on the exact
sequence length of the KV cache. Every decode step changes the tensor shape,
invalidates the guard, and triggers a **full XLA recompilation on every
generated token**—turning millisecond decode steps into multi-second
compilations unless you rewrite the model to preallocate static buffers and
perform in-place mutations.

**Bounded dynamism** solves this without requiring model rewrites: as the KV
cache grows across decode steps (`S + 1`, `S + 2`, ...), all shapes within the
compiled bucket (`<= 128`, `<= 256`, ...) reuse the same XLA executable.

For a hands-on walkthrough applying bounded dynamism to a full LLM decode loop
(and comparing against static KV cache preallocation), see
[LLM inference end-to-end](./case_studies_muse_glimmer_30b_inference.md).

## Example: Autoregressive loop with a growing tensor

To see how static compilation (`dynamic=False`) and bounded dynamism using
default dynamism (`dynamic=None`) behave in practice, consider a minimal
autoregressive loop where each step appends a new token along the sequence
dimension (`dim=2`) using `torch.cat`:

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. Running with `dynamic=False`

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Sample output:

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**What happened:** With `dynamic=False`, Dynamo guards on the exact sequence
length of `kv_cache` (`16`, `17`, `18`, ...). Because `torch.cat` increases
`dim=2` by `1` on every iteration, the shape guard fails on every call and
triggers a full XLA recompilation at every step.

### 2. Running with `dynamic=None` (Bounded Dynamism)

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Sample output:

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**What happened:** With `dynamic=None`, only the first 3 iterations incur
compilation overhead, and from step 4 onward execution drops to about 0.4 ms per
step (~100x faster) with zero recompilations:

*   **Step 1 (`seq_len=16 -> 17`):** Dynamo sees the inputs for the first time
    and compiles a static executable for `seq_len=16`.
*   **Step 2 (`seq_len=17 -> 18`):** Because `dim=2` changed from `16` to `17`,
    Dynamo automatically marks `dim=2` as dynamic (`SymInt`) and TorchTPU
    compiles a bounded-dynamic executable for the `[17, 128]` bucket using the
    initial input tensor layout.
*   **Step 3 (`seq_len=18 -> 19`):** The bounded-dynamic output buffer produced
    by step 2 has XLA's padded output layout, so TorchTPU compiles the
    `[17, 128]` bucket once more for that steady-state XLA layout.
*   **Steps 4–10 (`seq_len=19 -> 26`):** Both the sequence length (within the
    `<= 128` bucket) and the XLA buffer layout match the cached executable, so
    every subsequent step dispatches the pre-compiled XLA executable directly
    without recompiling.

> **Tip:** If you already know which dimension will vary ahead of time, you can
> call `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` before the
> first step. This tells Dynamo to trace `dim=2` as dynamic immediately on Step
> 1—skipping the initial static compilation on Step 1 and letting you specify
> explicit `[min, max]` bounds.

## Current limitations

Bounded dynamism in TorchTPU is actively evolving and is currently optimized
around the **autoregressive KV cache** use case. Keep the following limitations
in mind when applying it to your models:

*   **Bucket limit and unbounded dynamism:** At most **8 buckets**
    (`num_dynamic_buckets <= 8`) are compiled per trace. Unbounded dynamism is
    not supported yet—if a dynamic dimension grows past the highest compiled
    bucket's upper bound, a recompilation is triggered.
*   **Dynamo shape constraints:** Bucket upper bounds must satisfy active Dynamo
    shape constraints (such as divisibility requirements from `reshape` or
    `view`). Otherwise, TorchTPU falls back to a single bucket with upper bound
    `2 * S` (twice the initial runtime size `S`).
*   **Compute and HBM overhead up to the bucket upper bound:** Because execution
    runs on buffers padded to the bucket's static upper bound `U`, HBM footprint
    and FLOPs scale with `U` rather than the actual runtime size. Setting a very
    large `bucket_size` or `mark_dynamic(..., max=U)` when the actual size is
    much smaller than `U` pays the compute and memory cost of `U`.
*   **XLA layout changes:** Even when a tensor's shape falls within an
    already-compiled bucket, a change in the underlying XLA physical layout of
    an input buffer will trigger a bucket recompilation.
*   **No persistent compilation cache (`AOTAutogradCache`):** Bounded-dynamic
    executables are not serialized to `AOTAutogradCache` yet, so they are
    recompiled on each new process start.
*   **Operator coverage:** Not all PyTorch operators support bounded dynamic
    shapes yet.
*   **Views as model inputs:** Support for passing tensor views directly as
    inputs to the compiled model is limited.
*   **Zero-element dynamic inputs:** A dynamic input tensor with a zero-sized
    dimension (such as an empty `(0, S)` cache with `S` dynamic, or a slice that
    becomes empty at runtime) raises `NotImplementedError`; compile such calls
    with `dynamic=False`.
*   **Data-dependent dynamism:** Operations whose output shapes depend on
    runtime tensor values (such as `torch.nonzero`) are not supported.
*   **Custom kernels:** Bounded dynamic shapes are not supported with custom TPU
    kernels (such as Pallas kernels registered via `torch.tpu.pallas.jax_op`).
*   **dynamic=True:** torch.compile() *dynamic=True* mode is not supported.
