# Authoring Custom Pallas Kernels for PyTorch on Google Cloud TPUs

This tutorial provides a step-by-step guide on how to design, implement, and
integrate custom **Pallas** kernels for PyTorch models running on **Google Cloud
TPUs**.

We use the [SplashAttention kernel](splash_attention.py) developed for the
[FLUX Optimization Tutorial](README.md) as a real-world reference
implementation. This kernel and its accompanying test suite are directly based
on the
[TorchTitan implementation of SplashAttention](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/experiments/tpu/kernels/splash_attention.py).

--------------------------------------------------------------------------------

## Overview: Why Pallas for PyTorch on TPU?

Modern deep learning architectures frequently encounter operations that
bottleneck standard compiler pipelines:

-   **Custom Sparsity & Masking**: Causal masks, local sliding windows, and
    block-sparse attention patterns often require complex indexing that standard
    ops cannot express efficiently.
-   **Fusion Barriers**: While `torch.compile(backend="tpu")` fuses many
    elementwise chains, certain specialized tensor contractions benefit from
    explicit hardware tiling.

### What is Pallas?

**Pallas** is a kernel programming model for TPUs and GPUs developed as part of
JAX. For comprehensive guides and reference documentation on Pallas, see:

-   [Pallas: A JAX Kernel Language](https://docs.jax.dev/en/latest/pallas/index.html):
    Overview of the Pallas programming model, core concepts (`Refs`,
    `BlockSpec`), and kernel design.
-   [Pallas on TPUs](https://docs.jax.dev/en/latest/pallas/tpu.html): Guides on
    targeting Google Cloud TPU architectures, managing on-chip memory spaces
    (VMEM / SMEM), and pipelining.
-   [JAX Pallas Repository](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas):
    Official Pallas source repository and experimental TPU kernel
    implementations.

### Why Does `jax` Appear Here? (Kernel Authoring vs. Runtime Execution)

> [!IMPORTANT] **We are NOT running JAX during PyTorch model execution.** `jax`
> is used strictly as a **kernel authoring DSL and ahead-of-time MLIR compiler**
> (analogous to how `@triton.jit` uses Python DSL syntax to emit PTX/TTGIR for
> PyTorch on GPUs).

Because Pallas is distributed inside the `jax` Python package, defining or
configuring a Pallas kernel uses `jax` / `jax.numpy` primitives and `jax.Array`
type annotations. However, when you register a kernel with
`torch.tpu.pallas.jax_op`:

1.  **Trace-Time MLIR Export Only**: On the first call for a given input
    shape/dtype signature, `torch.tpu.pallas.jax_op` traces the Python function
    **once** using abstract shape/dtype descriptors (`jax.ShapeDtypeStruct` —
    never real JAX arrays or device buffers) and lowers the Pallas kernel into a
    serialized **StableHLO / Mosaic MLIR module**.
2.  **Registered in the `torch_tpu` C++ Runtime**: The serialized MLIR module is
    registered directly with `torch_tpu`'s C++ custom-call registry as a native
    `torch.library.custom_op`.
3.  **100% Native PyTorch / XLA Runtime**: During training and inference (both
    in eager mode and inside `torch.compile(backend="tpu")`), **no JAX runtime
    is ever invoked**. PyTorch `torch.Tensor` buffers are never converted to
    `jax.Array`; the lowered MLIR kernel is compiled and executed natively by
    the PyTorch XLA compiler directly on your `torch.Tensor` TPU buffers.

--------------------------------------------------------------------------------

## Step-by-Step Walkthrough

Developing a Pallas kernel for PyTorch in
[`splash_attention.py`](splash_attention.py) involves five key steps:

1.  **Configuring Block Sizes, Masks, and Layouts
    (`_make_splash_attention_fn`)**
2.  **Authoring the JAX Forward & Backward Functions
    (`_make_splash_attention_fn`)**
3.  **Registering Custom Ops & Autograd via `torch.tpu.pallas.jax_op`
    (`_get_splash_op`)**
4.  **Wrapping the PyTorch User Interface (`splash_sdpa`)**
5.  **Writing Numerical & Gradient Parity Tests (`SplashAttentionTest`)**

--------------------------------------------------------------------------------

### Step 1: Configuring Block Sizes, Masks, and Layouts

On Cloud TPUs, High-Bandwidth Memory (HBM) is much slower than on-chip Vector
Memory (VMEM). A Pallas kernel tiles large tensors into smaller blocks that fit
inside VMEM.

Inside `_make_splash_attention_fn` in
[`splash_attention.py`](splash_attention.py), we configure `BlockSizes` for both
forward (`block_q`, `block_kv`) and backward (`block_q_dkv`, `block_kv_dkv`)
passes, along with the tensor layout (`HEAD_DIM_MINOR` or `SEQ_MINOR`) and
attention masks (`LocalMask`, `CausalMask`, or `FullMask`):

```python
# Inside _make_splash_attention_fn(...) in splash_attention.py:
block_sizes = splash_attention_kernel.BlockSizes(
    block_q=min(block_q, seq_len),
    block_kv=min(block_kv, seq_len),
    block_kv_compute=min(block_kv_compute, seq_len),
    block_q_dkv=min(block_q_dkv, seq_len),
    block_kv_dkv=min(block_kv_dkv, seq_len),
    block_kv_dkv_compute=min(block_kv_dkv_compute, seq_len),
    block_q_dq=min(block_q_dq, seq_len)
    if not use_fused_bwd_kernel and block_q_dq is not None
    else None,
    block_kv_dq=min(block_kv_dq, seq_len)
    if not use_fused_bwd_kernel and block_kv_dq is not None
    else None,
    use_fused_bwd_kernel=use_fused_bwd_kernel,
    q_layout=_get_layout(q_layout),
    k_layout=_get_layout(k_layout),
    v_layout=_get_layout(v_layout),
)

mask_shape = (seq_len, seq_len)
if local_window_size is not None:
  single_mask = splash_attention_mask.LocalMask(
      shape=mask_shape,
      window_size=(local_window_size, 0),
      offset=0,
  )
elif is_causal:
  single_mask = splash_attention_mask.CausalMask(shape=mask_shape)
else:
  single_mask = splash_attention_mask.FullMask(mask_shape)

multi_head_mask = splash_attention_mask.MultiHeadMask(
    masks=(single_mask,) * n_heads
)
```

--------------------------------------------------------------------------------

### Step 2: Authoring the JAX Forward & Backward Functions

A complete custom kernel requires both a **forward pass** and an efficient
**backward pass**. Rather than relying on automatic differentiation to re-run
the forward computation during backpropagation, the forward pass saves the
Softmax `logsumexp` residual (`save_residuals=True`) and passes it directly into
the backward kernel.

Continuing inside `_make_splash_attention_fn` in
[`splash_attention.py`](splash_attention.py), we compute the static host NumPy
`MaskInfo` metadata and define the batch-level forward (`splash_fn`) and
backward (`splash_bwd_fn`) functions. Note that:

-   The `jax.Array` type annotations are used strictly by
    `torch.tpu.pallas.jax_op` to inspect the function signature and infer the
    PyTorch custom operator schema (`(Tensor q, Tensor k, Tensor v) -> ...`).
-   Operations like `jnp.array(...)`, `jax.vmap(...)`, and `jax.lax.scan(...)`
    are **not** executed at runtime; they are traced **once** with abstract
    shape/dtype placeholders to emit the StableHLO/Mosaic MLIR module for XLA:

```python
# Inside _make_splash_attention_fn(...) in splash_attention.py:
fwd_mask_info, _mask_function = splash_attention_mask_info.process_mask(
    multi_head_mask,
    (block_sizes.block_q, block_sizes.block_kv),
    head_shards=1,
    q_seq_shards=1,
)
dq_mask_info = None
dkv_mask_info = None
if block_sizes.has_backward_blocks:
  if not use_fused_bwd_kernel:
    dq_mask_info, _ = splash_attention_mask_info.process_mask(
        multi_head_mask,
        (block_sizes.block_q_dq, block_sizes.block_kv_dq),
        head_shards=1,
        q_seq_shards=1,
    )
  dkv_mask_info, _ = splash_attention_mask_info.process_mask_dkv(
      multi_head_mask,
      (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
      head_shards=1,
      q_seq_shards=1,
      shrink_grid=not use_fused_bwd_kernel,
  )

def _single_fwd_with_lse(q_b, k_b, v_b):
  """Forward for one batch element; returns (out_b, logsumexp_b)."""
  out_b, (logsumexp_b,) = splash_attention_kernel._splash_attention_forward(
      fwd_mask_info,
      q_b,
      k_b,
      v_b,
      segment_ids=None,
      sinks=None,
      mask_value=_mask_value,
      is_mqa=is_mqa,
      block_sizes=block_sizes,
      residual_checkpoint_name=_residual_checkpoint_name,
      save_residuals=True,
      mask_function=_mask_function,
      attn_logits_soft_cap=_attn_logits_soft_cap,
      interpret=_interpret,
  )
  return out_b, logsumexp_b

def splash_fn(
    q: jax.Array, k: jax.Array, v: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Forward: returns (out, logsumexp) for the whole batch."""
  head_dim = q.shape[-1]
  scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
  q_scaled = q * scale
  out, logsumexp = jax.vmap(_single_fwd_with_lse)(q_scaled, k, v)
  return out, logsumexp

def _single_bwd(q_scaled_b, k_b, v_b, out_b, logsumexp_b, g_b):
  """Backward for one batch element using precomputed residuals."""
  res = (
      q_scaled_b, k_b, v_b, None, None,
      out_b, logsumexp_b, dq_mask_info, dkv_mask_info,
  )
  all_grads = splash_attention_kernel._splash_attention_bwd(
      False, _mask_value, is_mqa, block_sizes,
      _residual_checkpoint_name, _mask_function,
      _attn_logits_soft_cap, _interpret, res, g_b,
  )
  return all_grads[3], all_grads[4], all_grads[5]  # dq_scaled, dk, dv

def splash_bwd_fn(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    out: jax.Array,
    logsumexp: jax.Array,
    g: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Backward: uses saved (out, logsumexp) — no extra forward pass."""
  head_dim = q.shape[-1]
  scale = jnp.array(1.0 / math.sqrt(head_dim), dtype=q.dtype)
  q_scaled = q * scale

  if use_vmap_bwd:
    dq_scaled, dk, dv = jax.vmap(_single_bwd)(
        q_scaled, k, v, out, logsumexp, g
    )
  else:
    _, (dq_scaled, dk, dv) = jax.lax.scan(
        _scan_bwd_body, None, (q_scaled, k, v, out, logsumexp, g),
    )
  dq = dq_scaled * scale
  return dq, dk, dv
```

--------------------------------------------------------------------------------

### Step 3: Lowering to MLIR & Registering Custom Ops via `torch.tpu.pallas.jax_op`

`torch.tpu.pallas.jax_op` lowers a type-annotated Pallas function into a
serialized MLIR module (using abstract shape/dtype placeholders) and registers
it as a native PyTorch `torch.library.custom_op` along with a default
`register_fake` implementation for **TorchDynamo**, **AOTAutograd**, and
**`torch.compile`**. Once lowered, calling `torch_fwd_fn(q, k, v)` dispatches
the compiled XLA kernel directly on PyTorch `torch.Tensor` device buffers
without invoking JAX at runtime.

Because Pallas kernels are specialized on static parameters (such as `seq_len`,
`n_heads`, and `block_sizes`), `_get_splash_op` in
[`splash_attention.py`](splash_attention.py) hashes the configuration parameters
to register a unique operator per configuration, connects the backward pass via
`register_autograd`, and caches the registered operator **name**. Adding
`@torch._dynamo.assume_constant_result` allows TorchDynamo to treat the operator
lookup as a compile-time constant. Returning a string rather than the
`CustomOpDef` object matters: TorchDynamo cannot install guards on the
`CustomOpDef` object, so returning it directly breaks `torch.compile`.

```python
# Inside _get_splash_op(**splash_kwargs) in splash_attention.py:
splash_fn, splash_bwd_fn = _make_splash_attention_fn(**splash_kwargs)

identifier = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:16]
op_fwd_name = f"splash_attn_fwd_{identifier}"
op_bwd_name = f"splash_attn_bwd_{identifier}"
torch_fwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_fwd_name}", splash_fn)
torch_bwd_fn = torch.tpu.pallas.jax_op(f"pallas::{op_bwd_name}", splash_bwd_fn)

def setup_context(ctx, inputs, output):
  q, k, v = inputs
  out, lse = output
  ctx.save_for_backward(q, k, v, out, lse)

def backward(ctx, grad_out, grad_lse):
  del grad_lse
  q, k, v, out, lse = ctx.saved_tensors
  grad_q, grad_k, grad_v = torch_bwd_fn(q, k, v, out, lse, grad_out)
  return grad_q, grad_k, grad_v

torch_fwd_fn.register_autograd(backward, setup_context=setup_context)

_splash_op_cache[cache_key] = op_fwd_name
return op_fwd_name
```

--------------------------------------------------------------------------------

### Step 4: Wrapping the PyTorch User Interface (`splash_sdpa`)

The public entry point `splash_sdpa` in
[`splash_attention.py`](splash_attention.py) provides a drop-in replacement for
`F.scaled_dot_product_attention` accepting `(B, H, S, D)` tensors:

1.  Adjusts tile sizes via `_adjust_block` so `seq_len` is evenly divisible by
    the chosen block sizes.
2.  Looks up the cached custom op by name under `torch.ops.pallas` and invokes
    it directly.

```python
# Inside splash_sdpa(...) in splash_attention.py:
batch, n_heads, seq_len, head_dim = q.shape
n_kv_heads = k.shape[1]

def _adjust_block(b_size: int, s_len: int) -> int:
  if s_len % b_size == 0:
    return b_size
  for candidate in [512, 256, 128, 64, 32]:
    if s_len % candidate == 0 and candidate <= b_size:
      return candidate
  return math.gcd(s_len, b_size)

block_q = _adjust_block(block_q, seq_len)
block_kv = _adjust_block(block_kv, seq_len)
# ... (adjust remaining block sizes) ...

torch_fwd_fn = getattr(torch.ops.pallas, _get_splash_op(**splash_args))

result, _ = torch_fwd_fn(q, k, v)
```

--------------------------------------------------------------------------------

### Step 5: Numerical Parity & Gradient Verification

Every custom kernel should be verified against PyTorch's reference math
implementation on Cloud TPU hardware. In
[`splash_attention_test.py`](splash_attention_test.py), `SplashAttentionTest`
verifies:

1.  **Forward Parity (`check_forward_parity`)**: Compares `splash_sdpa` against
    `F.scaled_dot_product_attention` under `SDPBackend.MATH` for both
    bidirectional (`test_flux_bidirectional_mha_parity`) and causal
    (`test_causal_mha_parity`) attention.
2.  **Backward Gradient Parity (`test_mha_backward_parity`)**: Compares
    `q.grad`, `k.grad`, and `v.grad` produced by the custom Pallas backward
    kernel against reference autograd gradients.

```python
# Inside SplashAttentionTest in splash_attention_test.py:
def check_forward_parity(
    self, splash_fn, b, n_heads, n_kv_heads, seq_len, head_dim, is_causal=False
):
  torch.manual_seed(42)
  q = torch.randn(b, n_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
  k = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)
  v = torch.randn(b, n_kv_heads, seq_len, head_dim, dtype=torch.float32, device=self.device)

  with attention.sdpa_kernel([attention.SDPBackend.MATH]):
    expected_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

  actual_out = splash_fn(q, k, v)
  torch.testing.assert_close(
      actual_out.cpu(), expected_out.cpu(), rtol=1e-2, atol=1e-2
  )
```

--------------------------------------------------------------------------------

## Running & Testing on TPU

### 1. From the Command Line (TPU VM or Container Shell)

From this directory, run the custom Pallas kernel parity test suite:

```bash
python3 ./splash_attention_test.py
```

### 2. Running on GKE

To run this test suite on a single-device TPU v7x pod on GKE, update the
container `command:` in
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` to run:

```bash
cd /workspace/examples/FLUX_optimization/
python3 ./splash_attention_test.py
```

Then:

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

1.  If you haven't set up your cluster yet, complete the **One-time setup**
    section in
    [Get started with TorchTPU on GKE](../../docs/markdown/1_basics_3_get_started_on_gke.md).
2.  Submit the job by following the **Single-Host Workload Execution (1-chip /
    1-device slice)** section in the same guide.
