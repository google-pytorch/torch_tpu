# FLUX Performance Optimization Tutorial on Google Cloud TPUs

This tutorial walks users through an example of the kinds of optimizations we
can perform on models like FLUX for TPU.

The lessons we will cover are broadly generalizable to other models and can be
applied to achieve better performance on TPU.

## Overview

Modern diffusion transformers like FLUX combine dual-stream attention,
multi-axis RoPE, and continuous timestep/prompt conditioning. The reference
implementation suffers from multiple performance bottlenecks on TPUs. In this
tutorial we will focus on:

1.  The advantages of using torch.compile on TPUs.
2.  Elimination of resizing and inefficient layout transformations through views
3.  Removing caching of full matrix and performing contractions directly and
    using single element wise-ops.
4.  Optimization through the use of custom kernels

--------------------------------------------------------------------------------

## Code & Architecture References

This tutorial is directly based on the
[torchtitan implementation of FLUX](https://github.com/google-pytorch/torchtitan/blob/main/torchtitan/models/flux/model/)
and
[black-forest-labs/flux](https://github.com/black-forest-labs/flux/blob/main/src/flux/)
repository.

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

NOTE: We are not tackling distributed in this example. However, it is worth
mentioning that from our torchtitan experiments, we know that SimpleFSDP
performs significantly better than FSDPv2. For more details, see the
[FSDPv2 to SimpleFSDP Example](../fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).

--------------------------------------------------------------------------------

## Benchmark Matrix on Cloud TPU Hardware

Mean training step time (forward + Flow Matching MSE loss + backward + AdamW
step) for a single `DoubleStreamBlock`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Configuration | `use_zero_copy_view` | `use_fused_rope` | `use_splash_attention` | `compile` | Mean Step Time | Speedup vs Baseline | Optimization Details |
| :--- | :---: | :---: | :---: | :---: | ---: | ---: | :--- |
| **1. Baseline Reference (Eager)** | ❌ False | ❌ False | ❌ False | ❌ False | **176.05 ms** | 1.00x (ref) | Reference baseline |
| [**2. Standalone SplashAttention (Eager)**](#milestone-4) | ❌ False | ❌ False | ✅ True | ❌ False | **147.50 ms** | **1.19x** | [Milestone 4: SplashAttention](#milestone-4) |
| [**3. Zero-Copy Views Only (Eager)**](#milestone-1) | ✅ True | ❌ False | ❌ False | ❌ False | **175.39 ms** | **1.00x** | [Milestone 1: Zero-Copy Views](#milestone-1) |
| [**4. Compile Only (Unoptimized Graph)**](#milestone-3) | ❌ False | ❌ False | ❌ False | ✅ True | **157.33 ms** | **1.12x** | [Milestone 3: XLA Compilation](#milestone-3) |
| [**5. Zero-Copy + Fused RoPE + SplashAttention (Eager)**](#milestone-4) | ✅ True | ✅ True | ✅ True | ❌ False | **153.60 ms** | **1.15x** | [Milestones 1](#milestone-1), [2](#milestone-2) & [4](#milestone-4) |
| [**6. Zero-Copy Views + Compile**](#milestone-1) | ✅ True | ❌ False | ❌ False | ✅ True | **156.56 ms** | **1.12x** | [Milestone 1](#milestone-1) + [Milestone 3](#milestone-3) |
| [**7. Fully Optimized (Zero-Copy + Fused RoPE + SplashAttention + Compile)**](#milestone-4) | ✅ True | ✅ True | ✅ True | ✅ True | **119.35 ms** | **1.48x** | [Milestones 1](#milestone-1), [2](#milestone-2), [3](#milestone-3) & [4](#milestone-4) |
| [**8. Zero-Copy + Fused RoPE + Compile (Standard SDPA)**](#milestone-2) | ✅ True | ✅ True | ❌ False | ✅ True | **151.27 ms** | **1.16x** | [Milestones 1](#milestone-1), [2](#milestone-2) & [3](#milestone-3) |
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

> [!NOTE]
>
> -   **Sequence length matters**: At this sequence length, attention dominates
>     the step. At shorter sequence lengths, compilation gives a larger speedup
>     (for example, about 3x at a sequence length of 640), while the attention
>     and layout optimizations matter less.
> -   **SplashAttention needs to be configured for the workload**: Running in
>     bfloat16 and using larger block sizes is what makes SplashAttention faster
>     than standard SDPA here. With float32 inputs and the default 512-token
>     blocks, SplashAttention was slower than standard SDPA at the same sequence
>     length.

--------------------------------------------------------------------------------

## Step-by-Step Code Comparison: Before vs. After

Below are the specific code transformations applied in `flux.py` across our four
optimization milestones.

For XPROF traces, you will want to pay attention to the XLA Modules for tracking
closer how things are being executed. Highlighed sections will be focused in
these.

<a id="milestone-1"></a>

### Milestone 1: Zero-Copy Tensor Views (`use_zero_copy_view`)

For the next session, the next are used when referring to dimensions:

-   `B`: batch size
-   `L`: sequence length
-   `H`: number of attention heads
-   `D`: head dimension

In `DoubleStreamBlock` (the double-stream attention block implemented by our
FLUX implementation), both the image stream (`img_qkv`) and text stream
(`txt_qkv`) project hidden states of shape `(B, L, H * D)` into fused QKV
tensors of shape `(B, L, 3 * H * D)`, unpack them into query, key, and value
tensors of shape `(B, L, H, D)`, and later merge attention heads back from
`(B, H, L, D)` to `(B, L, H * D)`. This interaction causes inefficient copies to
be created which significantly slow down our model.

#### Identifying issue

By looking at the code from the original application, find this
`einops.rearrange` call. A rearrange call like this might introduce inefficient
copies depending on how it gets lowered. Therefore, it is a good candidate for
further exploration. If we trace the code associated with the construction and
application of the 5D rotation matrix we can see the inefficiencies and extra
copies. See:

![Reference einops rearrange copy](images/milestone1_rearrange_copy.png)

Meanwhile, in our implemented version we see the copies eliminated:

![Zero-copy views eliminated copy](images/milestone1_zero_copy.png)

This copy might appear small, but in a transformer block that performs this
operation multiple times, it adds up to a significant amount of time.

Therefore, we create an alternative implementation that avoids these
inefficiencies. The new implementation uses `torch.view` and `torch.unbind` to
create the desired tensor views without the need for a copy.

We could also have gone the opposite direction, and used the XPROF traces to
find the inefficiencies, and then trace them back to the source.

#### Before: Reference `einops.rearrange` (Physical HBM Copy)

```python
import einops

# 1. Unpack fused QKV projection: moves K=3 to outermost dimension (dim 0)
#    Forces physical HBM-to-HBM layout transposition (tt_jit_copy__as_strided_inverse)
img_q, img_k, img_v = einops.rearrange(
    img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads
)

# ... (Joint Attention) ...

# 2. Reshape attention output heads back to hidden dimension
attn = einops.rearrange(attn, "B H L D -> B L (H D)")
```

#### After: Zero-Copy `.view()` + `torch.unbind()` (Metadata View Only)

```python
# 1. Reshape keeping K=3 adjacent to (H, D) and unbind along dim=2
#    Zero-copy slice views referencing the existing projection buffer in HBM!
b, l, _ = img_qkv.shape
img_qkv = img_qkv.view(b, l, 3, self.num_heads, self.head_dim)
img_q, img_k, img_v = torch.unbind(img_qkv, dim=2)

# ... (Joint Attention) ...

# 2. Native transpose + contiguous reshape
b, _, l, _ = attn.shape
attn = attn.transpose(1, 2).reshape(b, l, -1)
```

> **Why this matters on TPU**: When unpacking `(B, L, 3 * H * D)`, keeping the
> `3` dimension at index `2` (`(B, L, 3, H, D)`) preserves contiguous memory
> blocks of size `H * D` for each head group. Calling `torch.unbind(img_qkv,
> dim=2)` creates three tensor headers (`img_q`, `img_k`, `img_v`) with adjusted
> storage offsets and strides without moving a single byte in TPU HBM. The
> layout copy that `einops.rearrange` introduces in the backward pass is
> removed. In the benchmark matrix, this is a small fraction of the step:
> **176.05 ms to 175.39 ms** in eager mode and **157.33 ms to 156.56 ms** under
> compilation. The benefit grows with the amount of QKV data moved per step,
> such as in full models with many blocks.

--------------------------------------------------------------------------------

<a id="milestone-2"></a>

### Milestone 2: Fused Algebraic N-Dimensional RoPE (`use_fused_rope`)

FLUX uses 3-axis Rotary Position Embeddings (`axes_dim=(16, 56, 56)` summing to
`head_dim=128`) to encode temporal/frame, vertical, and horizontal patch
coordinates.

#### Identifying issue

Similar to the above, we can see from the original implementation the way in
which the 5D rotation matrix is constructed and applied. We see that the
construction of the 5D rotation matrix requires 4 tensor stacking operations and
a reshape. The application of the matrix requires a further reshape to get the
desired 4D tensor.

Looking at the code trace for the code we see large copy blocks slowing us down:

![Reference 5D RoPE copies](images/milestone2_rope_reference_copies.png)

Therefore we set out to create an alternative implementation that avoids these
issues. We set out to create a 2D rotation matrix that can be applied directly
to the 4D tensor, and can be fused by XLA. The new trace for the section is much
cleaner:

![Fused 2D RoPE trace](images/milestone2_fused_rope.png)

#### Before: 5D Rotation Matrix Construction & Application (`rope_ref` / `apply_rope_ref`)

```python
def rope_ref(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  # Explicitly stacks 4 values per frequency pair: [cos, -sin, sin, cos]
  out = torch.stack(
      [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
  )
  # Reshapes into a 5D tensor: (B, S, D/2, 2, 2)
  return einops.rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()


def apply_rope_ref(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  # Reshapes queries and keys to 6D: (B, S, H, D/2, 1, 2) for 2x2 matrix-vector product
  xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
  xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
  xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
  xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
  return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
```

#### After: Compact 2D Algebraic Rotation (`rope` / `apply_rope`)

```python
def rope(pos: Tensor, dim: int, theta: int = 10000) -> Tensor:
  scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
  omega = 1.0 / (theta**scale)
  out = torch.einsum("...n,d->...nd", pos, omega)
  # Stores only compact [cos, sin] pairs: (B, S, D/2, 2) — 50% less HBM footprint!
  return torch.stack([torch.cos(out).float(), torch.sin(out).float()], dim=-1)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
  cos = freqs_cis[..., 0]
  sin = freqs_cis[..., 1]

  b, s, h, d = xq.shape
  xq_ = xq.float().view(b, s, h, d // 2, 2)
  xk_ = xk.float().view(b, s, h, d // 2, 2)

  xq_0, xq_1 = xq_[..., 0], xq_[..., 1]
  xk_0, xk_1 = xk_[..., 0], xk_[..., 1]

  # Direct 2D algebraic rotation: (x0*cos - x1*sin, x0*sin + x1*cos)
  # Maps directly to fused multiply-add (FMA) instructions on TPU VPU
  xq_out_0 = xq_0 * cos - xq_1 * sin
  xq_out_1 = xq_0 * sin + xq_1 * cos
  xk_out_0 = xk_0 * cos - xk_1 * sin
  xk_out_1 = xk_0 * sin + xk_1 * cos

  xq_out = torch.stack([xq_out_0, xq_out_1], dim=-1).view(b, s, h, d).type_as(xq)
  xk_out = torch.stack([xk_out_0, xk_out_1], dim=-1).view(b, s, h, d).type_as(xk)
  return xq_out, xk_out
```

> **Why this matters on TPU**: In the reference implementation, constructing
> full `2x2` rotation matrices in HBM stores redundant entries (`cos`, `-sin`,
> `sin`, `cos`), doubling memory footprint and bandwidth consumption for
> positional embeddings. Applying them as matrix multiplications on reshaped 6D
> tensors creates high-dimensional indexing overhead and memory fragmentation.
> By contrast, storing only compact `[cos, sin]` pairs cuts HBM bandwidth by
> 50%, while the algebraic rotation `(x0*cos - x1*sin, x0*sin + x1*cos)` maps
> directly to native Fused Multiply-Add (FMA) instructions. When compiled, this
> enables seamless fusion into upstream projection and downstream attention
> kernels, reducing the compiled step time from **156.56 ms to 151.27 ms (a 3.4%
> improvement)** in the benchmark matrix.

--------------------------------------------------------------------------------

<a id="milestone-3"></a>

### Milestone 3: Whole-Block XLA Compilation (`compile`)

TODO(pgmoka): Add trace details.

With layout copies eliminated and RoPE expressed algebraically, we compile the
entire `DoubleStreamBlock` using `torch.compile` targeting the TPU XLA backend:

```python
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
compiled_block = block.maybe_compile()  # Remove this line to run in eager mode.
```

#### Identifying issue

In eager mode, you will see when tracing the model a high number of non-fused
operations. By compiling your model, you will enable XLA to fuse operations
together, eliminating intermediate buffers and reducing HBM footprint.

Rather than looking at a specific module, we can take a high level look at the
eager and compiled traces:

-   **Eager**:

    ![Eager trace](images/milestone3_eager_trace.png)

-   **Compiled**:

    ![Compiled trace](images/milestone3_compiled_trace.png)

In the compiled case, we see many more fused operations, resulting in a much
faster computation.

> **Why this matters on TPU**: In eager mode, every PyTorch operation launches
> an independent device kernel, requiring intermediate activations to round-trip
> through TPU High-Bandwidth Memory (HBM) and causing frequent host-device
> dispatch stalls. Invoking `torch.compile(backend="tpu")` lowers the entire
> `DoubleStreamBlock` larger XLA computation graph. The XLA compiler fuses
> sequences of elementwise operations, normalizations, and residual additions.
> Compilation also enables automatic buffer donation and asynchronous
> instruction pipelining, reducing the step time from **176.05 ms to 157.33 ms
> (1.12x)** in the benchmark matrix. Because attention dominates the step at a
> sequence length of 32,768, the compiler has less to fuse. At a sequence length
> of 640, the same change is about 3x faster.

--------------------------------------------------------------------------------

<a id="milestone-4"></a>

### Milestone 4: Custom Pallas SplashAttention Kernel (`use_splash_attention`)

At large sequence lengths (e.g., S = 2048 or S = 4096), full joint attention
becomes memory-bound and incurs quadratic O(S^2) memory traffic when computing
un-tiled attention matrices. We integrate a custom Pallas SplashAttention kernel
using the `torch_tpu` Pallas bridge to compute attention in SRAM tiles:

#### Before: Standard PyTorch SDPA

```python
# Standard PyTorch SDPA
attn = F.scaled_dot_product_attention(
    q_sdpa, k_sdpa, v_sdpa, is_causal=False
)
```

#### After: Custom Pallas SplashAttention (`splash_sdpa`)

```python
# Custom Pallas SplashAttention kernel via torch_tpu Pallas bridge
from kernels.splash_attention import splash_sdpa

attn = splash_sdpa(q_sdpa, k_sdpa, v_sdpa, is_causal=False)
```

> **Why this matters on TPU**: While standard SDPA performs well for small
> sequence lengths under compilation, scaling to high resolutions stresses TPU
> HBM bandwidth. SplashAttention uses Pallas to execute custom on-chip tiling
> directly on TPU Vector and Tensor processing hardware. This prevents
> materialization of intermediate attention scores in HBM, reducing memory
> scaling from O(S^2) to O(S). At a sequence length of 32,768 in bfloat16 with
> 2048-token blocks, SplashAttention is faster than standard SDPA in both eager
> (**181.67 ms to 153.60 ms, 1.18x**) and compiled (**151.27 ms to 119.35 ms,
> 1.27x**) modes, with zero-copy views and fused RoPE enabled in both cases.
>
> **Tuning SplashAttention**: Block sizes and precision have a large effect. At
> the same sequence length, the default 512-token blocks were about 10% slower
> than standard SDPA under compilation, and float32 inputs were slower still.
> Benchmark a few block sizes (for example 512, 1024, and 2048) for your
> sequence length, and run the kernel in bfloat16.
>
> **Note on Pallas & JAX usage**: Although Pallas kernels are defined using the
> `jax` package DSL (see [`pallas_kernels.md`](pallas_kernels.md) for a full
> walkthrough), **we are not running JAX at runtime**. `torch.tpu.pallas.jax_op`
> only uses JAX once at trace time with abstract shape/dtype placeholders to
> export the kernel into a standalone StableHLO/Mosaic MLIR module. During model
> execution, the lowered MLIR kernel runs natively inside the PyTorch/XLA
> runtime directly on `torch.Tensor` buffers with zero JAX runtime overhead.

--------------------------------------------------------------------------------

## Running Verification & Performance Profiling on TPU

### Understanding the Profiling Methodology in `flux_test.py`

For profiling we use
[Xprof](https://github.com/openxla/xprof?tab=readme-ov-file#quick-start).

When benchmarking compiled models on TPUs, proper measurement hygiene is
essential:

1.  **Separating Warmup from Timed Steps (`--num_warmup_steps=2`)**:
    `torch.compile(backend="tpu")` performs Just-In-Time (JIT) tracing and XLA
    HLO compilation on the first forward/backward invocation. `flux_test.py`
    runs dedicated warmup steps *before* starting the benchmark timer so that
    `Mean Step Time` reflects true steady-state TPU hardware execution.
2.  **Synchronizing Once After the Timed Loop
    (`torch.accelerator.synchronize()`)**: TPU execution is asynchronous, so the
    host can return from a step before the device has finished it. As in a real
    training loop, `flux_test.py` dispatches the timed steps back to back and
    synchronizes once after the last step, so that `Mean Step Time` (total time
    divided by the number of steps) includes all device execution rather than
    just host dispatch.
3.  **Profiling Outside the Timed Loop (`--enable_profiler`)**: The profiler
    adds host overhead, so `flux_test.py` runs it on a few extra, untimed steps
    after the benchmark, wrapped in
    `torch.profiler.profile(activities=[ProfilerActivity.CPU,
    ProfilerActivity.TPU])`. `format_tpu_operator_table(prof)` then aggregates
    device-side `self_device_time_total` microseconds to display the top TPU
    operators.

### 1. From the Command Line (TPU VM or Container Shell)

From this directory, run the verification test suites. Run the RoPE mathematical
and module-level parity tests:

<!-- doc-test: flux_rope_parity_doc_test -->

```bash
python3 ./flux_rope_parity_test.py
```

Run the custom Pallas SplashAttention kernel parity tests:

<!-- doc-test: splash_attention_doc_test -->

```bash
python3 ./splash_attention_test.py
```

Run the layout parity, compilation parity, and baseline speed test:

<!-- doc-test: flux_doc_test -->

```bash
python3 ./flux_test.py
```

### 2. Run Interactive Benchmarks Across Optimization Milestones

You can run `flux_test.py` directly on a TPU host and toggle individual
optimization flags via CLI arguments. The commands below use a sequence length
of 640 for faster iteration. To match the sequence length used in the benchmark
matrix, pass `--img_seq_len=32256 --txt_seq_len=512 --dtype=bfloat16` instead:

```bash
# Configuration 1: Unoptimized Baseline (Eager)
python3 ./flux_test.py \
  --batch_size=2 --img_seq_len=512 --txt_seq_len=128 --hidden_size=1024 \
  --use_zero_copy_view=false --use_fused_rope=false --compile=false \
  --num_warmup_steps=2 --num_benchmark_steps=10

# Configuration 2: Zero-Copy Views Only (Eager)
python3 ./flux_test.py \
  --batch_size=2 --img_seq_len=512 --txt_seq_len=128 --hidden_size=1024 \
  --use_zero_copy_view=true --use_fused_rope=false --compile=false \
  --num_warmup_steps=2 --num_benchmark_steps=10

# Configuration 5: Full Optimization (Zero-Copy Views + Fused 2D RoPE + XLA Compile)
python3 ./flux_test.py \
  --batch_size=2 --img_seq_len=512 --txt_seq_len=128 --hidden_size=1024 \
  --use_zero_copy_view=true --use_fused_rope=true --compile=true \
  --num_warmup_steps=2 --num_benchmark_steps=10

# Configuration 6: Pallas SplashAttention (Eager)
python3 ./flux_test.py \
  --use_splash_attention=true --compile=false \
  --num_warmup_steps=2 --num_benchmark_steps=10

# Configuration 7: Zero-Copy Views + Fused 2D RoPE + Pallas SplashAttention + Compile
python3 ./flux_test.py \
  --use_zero_copy_view=true --use_fused_rope=true --use_splash_attention=true --compile=true \
  --num_warmup_steps=2 --num_benchmark_steps=10
```

### 3. Running on GKE

To run these tests or benchmarks on a single-device TPU v7x pod on GKE, update
the container `command:` in
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` to run:

```bash
cd /workspace/examples/FLUX_optimization/
python3 ./flux_rope_parity_test.py
python3 ./splash_attention_test.py
python3 ./flux_test.py
```

Then:

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

1.  If you haven't set up your cluster yet, complete the **One-time setup**
    section in
    [Get started with TorchTPU on GKE](../../docs/markdown/1_basics_3_get_started_on_gke.md).
2.  Submit the job by following the **Single-Host Workload Execution (1-chip /
    1-device slice)** section in the same guide.
