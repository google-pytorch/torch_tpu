# FLUX optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/case_studies_flux.ipynb)

This case study walks through how we took a reference PyTorch implementation of
the [FLUX](https://github.com/black-forest-labs/flux) diffusion transformer
`DoubleStreamBlock` and made a training step **1.48x faster** on Cloud TPU with
TorchTPU at a sequence length of 32,768.

This page is a condensed companion to the full tutorial. It highlights the key
concept behind each optimization with short, self-contained snippets. For the
complete model code, benchmarks, XProf traces, and tests, see:

*   [FLUX Optimization Tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md):
    full walkthrough of every milestone, including XProf traces and instructions
    for reproducing every benchmark.
*   [Authoring Custom Pallas Kernels for PyTorch](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md):
    step-by-step walkthrough of the Pallas SplashAttention kernel used in
    Milestone 4.

The implementation is based on the
[TorchTitan implementation of FLUX](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
and the
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/)
repository.

If you are new to examples, you may want to familiarize yourself with running
examples on GKE [through our other documents](./get_started_on_gke.md).

## Results at a glance

Mean training step time (forward + loss + backward + optimizer step) for a
single `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configuration | Mean Step Time | Speedup vs. Baseline |
| :--- | ---: | ---: |
| Baseline reference (eager) | 178.61 ms | 1.00x |
| Pallas SplashAttention only (eager) | 149.81 ms | 1.19x |
| Zero-copy views only (eager) | 178.30 ms | 1.00x |
| `torch.compile` only | 159.33 ms | 1.12x |
| Zero-copy views + fused RoPE + Pallas SplashAttention (eager) | 156.73 ms | 1.14x |
| Zero-copy views + `torch.compile` | 159.08 ms | 1.12x |
| Zero-copy views + fused RoPE + `torch.compile` | 153.60 ms | 1.16x |
| **Zero-copy views + fused RoPE + Pallas SplashAttention + `torch.compile`** | **122.51 ms** | **1.46x** |
<!-- mdformat on -->

**Measurement setup:**

-   **Hardware:** a single TPU v6e chip (32 GiB HBM).
-   **Software:** Python 3.14, PyTorch 2.15 (pre-release), JAX 0.12, einops
    0.8.2, and TorchTPU built from source.
-   **Model shape:** `batch_size=2`, `hidden_size=1024`, `num_heads=8`
    (`head_dim=128`), `img_seq_len=32256`, `txt_seq_len=512` (total sequence
    length 32,768).
-   **Precision:** bfloat16 block parameters and inputs.
-   **SplashAttention block sizes:** 2048 for `block_q`, `block_kv`, and the
    dK/dV backward tiles.
-   **Timing:** 3 warmup steps followed by 10 timed steps. The timed steps are
    dispatched back to back and `torch.accelerator.synchronize()` runs once
    after the last one, so the mean step time (total time divided by the number
    of steps) includes device execution. The profiler is disabled during the
    timed steps.

At this sequence length, attention dominates the step, so SplashAttention and
compilation give the largest gains. At shorter sequence lengths, compilation
matters more (about 3x at a sequence length of 640) and the attention kernel
matters less.

The lessons generalize to most transformer models on TPU:

1.  **Avoid hidden layout copies**: Prefer zero-copy views (`.view()`,
    `torch.unbind()`) over reshuffles that materialize new HBM buffers.
2.  **Store less, compute more**: Express positional embeddings as compact
    elementwise math instead of materialized matrices.
3.  **Compile the whole block**: `torch.compile()` lets XLA fuse elementwise
    chains and remove intermediate HBM round-trips.
4.  **Reach for custom kernels when needed**: Pallas kernels give explicit
    control over on-chip tiling for memory-bound operations like attention at
    long sequence lengths.

### How to read this page

Each milestone is presented as a pair of sections:

-   **Before (reference)**: the original implementation and its bottleneck.
-   **After (optimized)**: the optimized implementation, followed by a check
    that it produces the same result as the reference.

Code cells are labeled with a matching `# BEFORE (reference)` or `# AFTER
(optimized)` comment on their first line.

**Prerequisites:** The Milestone 1 reference cell uses
[einops](https://github.com/arogozhnikov/einops) (`pip install einops`).

```python
import torch

device = torch.device("tpu")
print(f"Running on {device}.")
```

## Milestone 1: Zero-copy tensor views

In `DoubleStreamBlock`, the image and text streams project hidden states of
shape `(B, L, 3 * H * D)` into queries, keys, and values. How that projection is
split into Q, K, and V determines whether TPU has to copy data in HBM during
training.

### Before (reference): `einops.rearrange`

The reference implementation moves the `3` (Q/K/V) axis to the front:

```py
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)
```

In the forward pass, `einops.rearrange` returns strided views of the projection,
so no data moves yet. The cost shows up in the **backward pass**: to route
gradients back through the axis reordering, autograd materializes the gradient
in a new buffer, which runs on TPU as an extra HBM-to-HBM copy in every
attention block (visible as a `copy` op in XProf).

```python
# BEFORE (reference): einops.rearrange moves the Q/K/V axis to the front.
import einops

num_heads, head_dim = 4, 32
qkv = torch.randn(2, 16, 3 * num_heads * head_dim, device=device)

q_before, _k, _v = einops.rearrange(
    qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads
)
print(f"q shape: {tuple(q_before.shape)}")
```

### After (optimized): `.view()` + `torch.unbind()`

The optimized implementation keeps the `3` axis adjacent to `(H, D)` and splits
it with `torch.unbind`. The backward pass then stacks the Q, K, and V gradients
directly into the original layout, so no extra copy is needed. In the results
table, the effect on step time is small (**178.61 ms to 178.30 ms** in eager
mode, **159.33 ms to 159.08 ms** compiled) because the copy is a small fraction
of the step, but it is removed at no cost.

> **Note:** The cells in this milestone only show that the two approaches
> produce the same Q, K, and V. In the forward pass, both return the same
> strided views, so their outputs look identical. The extra HBM copy from
> `einops.rearrange` only occurs in the backward pass, and is visible in an
> XProf trace of a full training step rather than in the cell output. See
> [Milestone 1 of the FLUX Optimization Tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-1)
> for the traces.

```python
# AFTER (optimized): keep Q/K/V adjacent to (H, D) and unbind.
_b, _l, _ = qkv.shape
q_after, _k, _v = torch.unbind(qkv.view(_b, _l, 3, num_heads, head_dim), dim=2)
print(f"q shape: {tuple(q_after.shape)}")
print(f"Matches reference: {torch.equal(q_after.cpu(), q_before.cpu())}")
```

## Milestone 2: Fused Algebraic RoPE

FLUX applies multi-axis Rotary Position Embeddings (RoPE) to queries and keys.
The cells below implement both versions for a single RoPE axis.

### Before (reference): 2x2 rotation matrices

The reference implementation materializes a full `2x2` rotation matrix `[[cos,
-sin], [sin, cos]]` for every frequency pair, and applies it as a matrix-vector
product on reshaped 6D query/key tensors. This stores `cos` and `sin` twice, and
the extra stacking and reshaping show up as large copy blocks in XProf.

```python
# BEFORE (reference): materialize 2x2 rotation matrices, then contract.
def rope_ref(pos, dim, theta=10000):
  """Returns rotation matrices of shape (B, S, D/2, 2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)],
      dim=-1,
  )
  return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope_ref(xq, xk, freqs_cis):
  """6D matrix-vector contraction against the rotation matrices."""
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return (
      xq_out.reshape(*xq.shape).type_as(xq),
      xk_out.reshape(*xk.shape).type_as(xk),
  )


_b, _s, _h, rope_head_dim = 2, 64, 4, 32
positions = (
    torch.arange(_s, dtype=torch.float32, device=device)
    .unsqueeze(0)
    .expand(_b, _s)
)
rope_xq = torch.randn(_b, _s, _h, rope_head_dim, device=device)
rope_xk = torch.randn(_b, _s, _h, rope_head_dim, device=device)

# Add a broadcast head dimension: (B, S, 1, D/2, 2, 2).
_freqs_ref = rope_ref(positions, rope_head_dim).unsqueeze(2)
q_rope_ref, k_rope_ref = apply_rope_ref(rope_xq, rope_xk, _freqs_ref)

print(f"Rotation matrix freqs shape: {tuple(_freqs_ref.shape)}")
```

### After (optimized): compact `[cos, sin]` pairs + elementwise rotation

The optimized implementation stores only compact `[cos, sin]` pairs and applies
the rotation directly with elementwise math:

```py
# (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
xq_out_0 = xq_0 * cos - xq_1 * sin
xq_out_1 = xq_0 * sin + xq_1 * cos
```

This halves the positional embedding footprint, maps to fused multiply-add
instructions, and fuses with the surrounding projection and attention ops under
compilation. On top of zero-copy views and `torch.compile`, it reduces the step
time from **159.08 ms to 153.60 ms (3.4%)**.

Because both versions perform the same multiplications and additions on the same
operands, they produce identical results. The cell below runs the optimized
version on the same inputs and compares it against the reference:

> **Note:** These cells show the smaller frequency tensor and that both versions
> produce identical results. The speedup only appears when RoPE runs inside the
> compiled `DoubleStreamBlock`, where the elementwise rotation fuses with the
> surrounding ops. See
> [Milestone 2 of the FLUX Optimization Tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-2)
> for the XProf traces.

```python
# AFTER (optimized): store compact [cos, sin] pairs, rotate elementwise.
def rope(pos, dim, theta=10000):
  """Returns compact [cos, sin] pairs of shape (B, S, D/2, 2)."""
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)


def apply_rope(xq, xk, freqs_cis):
  """Direct 2D algebraic rotation with elementwise math."""
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]
  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)
  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]
  xq_out = torch.stack(
      [xq_0 * cos - xq_1 * sin, xq_0 * sin + xq_1 * cos], dim=-1
  ).view(b, s, h, d)
  xk_out = torch.stack(
      [xk_0 * cos - xk_1 * sin, xk_0 * sin + xk_1 * cos], dim=-1
  ).view(b, s, h, d)
  return xq_out.type_as(xq), xk_out.type_as(xk)


# Add a broadcast head dimension: (B, S, 1, D/2, 2).
freqs = rope(positions, rope_head_dim).unsqueeze(2)
_q_new, _k_new = apply_rope(rope_xq, rope_xk, freqs)

print(f"Compact freqs shape:         {tuple(freqs.shape)}")
print(
    "Max |q_ref - q_new|:        "
    f" {(q_rope_ref.cpu() - _q_new.cpu()).abs().max().item():.3e}"
)
print(
    "Max |k_ref - k_new|:        "
    f" {(k_rope_ref.cpu() - _k_new.cpu()).abs().max().item():.3e}"
)
torch.testing.assert_close(
    _q_new.cpu(), q_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_new.cpu(), k_rope_ref.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
```

## Milestone 3: Whole-block XLA compilation

### Before (reference): eager execution

In eager mode, every PyTorch operation launches an independent TPU program, so
intermediate activations round-trip through HBM and the host frequently stalls
on dispatch. In the results table, the baseline `DoubleStreamBlock` takes
**178.61 ms** per step in eager mode.

The cell below runs the fused RoPE from Milestone 2 eagerly, op by op:

```python
# BEFORE (reference): eager execution, one TPU program per op.
q_eager, k_eager = apply_rope(rope_xq, rope_xk, freqs)
print("Ran fused RoPE eagerly.")
```

### After (optimized): `torch.compile()`

Compiling the entire `DoubleStreamBlock` with `torch.compile()` lowers it to a
single XLA graph, letting the compiler fuse elementwise ops, normalizations, and
residual additions:

```py
block = DoubleStreamBlock(
    hidden_size=1024,
    num_heads=8,
    mlp_ratio=4.0,
    config=OptimizationConfig(
        use_zero_copy_view=True,
        use_fused_rope=True,
        compile=True,
    ),
)
compiled_block = block.maybe_compile()  # Wraps the block in torch.compile().
```

Compilation brings the step time from **178.61 ms to 159.33 ms (1.12x)**. The
gain is modest at a sequence length of 32,768 because attention dominates the
step. At a sequence length of 640, compilation is about 3x faster.

The same pattern applies to any function. The cell below compiles the fused RoPE
and checks that it matches eager execution:

> **Note:** These cells only check that compiling doesn't change the result. The
> fusion and the speedup show up in XProf and step-time measurements of the full
> `DoubleStreamBlock`, not in the cell output. See
> [Milestone 3 of the FLUX Optimization Tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/README.md#milestone-3)
> for eager vs. compiled traces.

```python
# AFTER (optimized): compile into a single fused XLA graph.
compiled_apply_rope = torch.compile(apply_rope, backend="tpu")
_q_compiled, _k_compiled = compiled_apply_rope(rope_xq, rope_xk, freqs)

torch.testing.assert_close(
    _q_compiled.cpu(), q_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
torch.testing.assert_close(
    _k_compiled.cpu(), k_eager.cpu()
)  # TORCH_ASSERT_CLOSE_OK=<Using standard PyTorch assert for user facing documentation only.>
print("Compiled fused RoPE matches eager execution.")
```

## Milestone 4: Custom Pallas SplashAttention kernel

At long sequence lengths (for example, `S = 2048` or `S = 4096`), full joint
attention becomes memory-bound. This milestone is shown as snippets only; see
the
[Pallas kernel walkthrough](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
for the full, runnable implementation.

### Before (reference): standard PyTorch SDPA

```py
attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

Un-tiled attention materializes an `O(S^2)` score matrix in HBM.

### After (optimized): Pallas SplashAttention

```py
attn = splash_sdpa(q, k, v, is_causal=False)  # Drop-in replacement.
```

`splash_sdpa` is a custom
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) SplashAttention
kernel that computes attention in on-chip tiles, based on the
[TorchTitan implementation of SplashAttention](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

The kernel is exposed to PyTorch with `torch.tpu.pallas.jax_op`. Its forward and
backward functions are annotated with `jax.Array` types, from which `jax_op`
infers the PyTorch custom operator schema. The forward pass saves the softmax
`logsumexp` residual so the backward kernel does not need to re-run the forward
computation:

```py
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)


def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)


def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  return torch_bwd_fn(q, k, v, out, lse, grad_out)


torch_fwd_fn.register_autograd(backward, setup_context=setup_context)
```

> **Important:** **JAX is not executed at runtime.** Pallas is distributed
> inside the `jax` package, so the kernel is written with `jax` syntax. However,
> `torch.tpu.pallas.jax_op` only uses JAX once, at trace time, with abstract
> shape/dtype placeholders to export the kernel as an MLIR module. During
> training and inference, both in eager mode and under `torch.compile`, the
> kernel runs directly on `torch.Tensor` TPU buffers.

**Results**: In the results table, SplashAttention alone is **1.19x** faster
than the eager baseline (178.61 ms to 149.81 ms). With zero-copy views and fused
RoPE enabled, it is faster than standard SDPA in both eager (**182.61 ms to
156.73 ms, 1.17x**) and compiled (**153.60 ms to 122.51 ms, 1.25x**) modes.

These results depend on configuring the kernel for the workload. With the
default 512-token blocks, SplashAttention was about 10% slower than standard
SDPA under compilation at the same sequence length, and float32 inputs were
slower still. Benchmark a few block sizes for your sequence length, run the
kernel in bfloat16, and profile before reaching for a custom kernel.

**Learn more:**

*   **Pallas SplashAttention example (used in this case study)**:
    [`splash_attention.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention.py)
    contains the full kernel integration, and
    [`splash_attention_test.py`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/splash_attention_test.py)
    verifies forward and gradient parity against PyTorch SDPA. The
    [kernel walkthrough](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/FLUX_optimization/pallas_kernels.md)
    explains block sizes, masks, autograd registration, and testing step by
    step.
*   **Pallas basics**: [Custom Kernels with Pallas](./custom_kernels.md)
    introduces writing Pallas kernels with TorchTPU.
