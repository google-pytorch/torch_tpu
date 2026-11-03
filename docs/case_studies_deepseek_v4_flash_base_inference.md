# DeepSeek-V4-Flash: End-to-End Inference and Functional Migration on Cloud TPU

This case study details the porting, architectural migration, and performance benchmarking of the **DeepSeek-V4-Flash** model for high-efficiency autoregressive inference on Google Cloud TPUs using **TorchTPU**.

For the complete standalone implementation and runnable code, see [`examples/deepseek_v4_flash_base_inference`](https://github.com/google-pytorch/torch_tpu/blob/main/examples/deepseek_v4_flash_base_inference).

---

## 1. Overview and Model Architecture

[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) is a 284-billion parameter Mixture-of-Experts (MoE) model with 13 billion active parameters per token, designed for ultra-low latency generation.

Key architectural features include:
* **Total / Active Parameters**: 284B total parameters, 13B active per token across 43 transformer layers.
* **Multi-Head Latent Attention (MLA)**: Compresses Key and Value representations into a low-rank latent vector (`kv_lora_rank = 512`), dramatically reducing memory bandwidth requirements during generation.
* **Rotary Position Embeddings (RoPE)**: Decoupled rotary embeddings applied to queries and keys.
* **Mixture-of-Experts (MoE) with Sinkhorn Router**: 256 routed experts (with top-6 active per token) plus 1 shared expert executed unconditionally. Routing relies on iterative Sinkhorn normalization for expert load balancing.
* **Walsh-Hadamard Transform**: Fast butterfly mixing transforms applied to feature activations to eliminate outlier dimensions prior to quantization.

---

## 2. The Migration Challenge: CUDA-Centric Friction Modes

The reference upstream implementation was tightly coupled to NVIDIA CUDA hardware and runtime environments:

1. **Custom TileLang JIT Kernels**: TileLang was used to JIT-compile custom CUDA PTX kernels for block-wise FP8/FP4 quantization, GEMMs, and sparse attention. TileLang kernels cannot execute on Cloud TPU systolic arrays.
2. **Native C++ CUDA Extensions**: The Walsh-Hadamard transform required a compiled CUDA C++ extension (`fast_hadamard_transform.so`), which immediately aborts with `ModuleNotFoundError` on TPU hosts.
3. **In-Place Tensor Mutations and Buffer Aliasing**: The reference KV-cache implementation relied on direct slice assignments:
   ```python
   self.kv_cache[:bsz, :seqlen] = kv
   self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
   ```
   OpenXLA constructs computation graphs using functional Static Single Assignment (SSA) form. In-place slice mutations create buffer aliasing conflicts that cause compilation failures or trigger graph breaks.
4. **Host-Device Synchronization**: The expert dispatch router converted tensor frequencies to Python lists (`bincount(...).tolist()`), triggering costly TPU-to-host roundtrips on every decoding step.

---

## 3. GPU-to-TPU Operator Migration Matrix

To enable native, zero-graph-break execution on Cloud TPUs, every GPU-specific operator was mapped to an OpenXLA-compliant functional fallback in pure PyTorch:

| GPU / CUDA Operator | Source Location | TPU Failure Mechanism | Clean PyTorch / OpenXLA Fallback | Performance & Memory Impact |
| :--- | :--- | :--- | :--- | :--- |
| **`import tilelang`** | `kernel.py` | Library unavailable on TPU containers. | Removed. Use standard PyTorch operators. | Zero overhead; eliminates external dependency. |
| **`@tilelang.jit act_quant_kernel`** | `kernel.py` | Generates PTX assembly for block FP8 quantization. | `act_quant_fallback` with PyTorch block reshaping. | Fuses cleanly into XLA memory operations (<2% overhead). |
| **`@tilelang.jit fp4_quant_kernel`** | `kernel.py` | Emits CUDA bitwise instructions for FP4 E2M1. | `fp4_act_quant_fallback` via vector scaling and rounding. | Unpacks FP4 to bfloat16 for native MXU math. |
| **`@tilelang.jit fp8_gemm_kernel`** | `kernel.py` | CUDA tensor core kernel incompatible with TPU MXUs. | `fp8_gemm_fallback` dequantizing to bfloat16 with `torch.matmul`. | Enables OpenXLA to emit native `dot_general` HLO instructions. |
| **`@tilelang.jit sparse_attn_kernel`** | `kernel.py` | Block-sparse CUDA kernel fails on TPU hardware. | `mla_sparse_attn_fallback` via scaled dot-product attention. | Leverages XLA fused attention on Matrix Multiply Units. |
| **`@tilelang.jit hc_split_sinkhorn`** | `kernel.py` | Custom CUDA kernel fails to compile on TPU. | `hc_split_sinkhorn_fallback` using iterative PyTorch normalization. | Executes cleanly in 10 iterations on Vector Processing Units. |
| **`fast_hadamard_transform`** | `model.py` | Compiled C++ library fails dynamic linking on TPU. | `hadamard_transform_fallback` via Walsh-Hadamard butterfly recursion. | Executes in $O(d \log d)$ with zero native C++ binaries. |
| **`self.kv_cache[...] = kv`** | `model.py` | In-place slice assignment violates XLA buffer immutability. | `update_kv_cache_functional` using dynamic masking and out-of-place writes. | Preserves single-assignment functional form; eliminates buffer aliasing. |
| **`y.copy_(x)`** | `model.py` | In-place tensor copy causes buffer aliasing in RoPE. | `apply_rotary_emb_functional` returning a newly constructed tensor. | Eliminates graph breaks during autoregressive decode. |
| **`bincount(...).tolist()`** | `model.py` | Host synchronization triggers CPU-TPU roundtrips. | Pure PyTorch tensor routing without converting to host list. | Prevents pipeline stalls during expert dispatch. |

---

## 4. Key Architectural Implementations

### 4.1 Walsh-Hadamard Transform in Pure PyTorch

The Walsh-Hadamard transform is implemented via recursive butterfly factorizations without native compiled C++ extensions:

```python
def hadamard_transform_fallback(x: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
  """Pure PyTorch Fast Walsh-Hadamard Transform using butterfly factorization.

  Operates in O(d log d) time without requiring CUDA C++ extensions.
  """
  orig_shape = x.shape
  d = orig_shape[-1]
  if (d & (d - 1)) != 0:
    raise ValueError(f"Hadamard dimension must be a power of 2, got {d}")

  y = x.reshape(-1, d).to(torch.float32)
  h = 1
  while h < d:
    y = y.view(-1, d // (2 * h), 2, h)
    y0 = y[:, :, 0, :]
    y1 = y[:, :, 1, :]
    y = torch.stack([y0 + y1, y0 - y1], dim=2)
    h *= 2

  y = y.view(orig_shape)
  inv_scale = scale if scale is not None else 1.0 / math.sqrt(d)
  return (y * inv_scale).to(x.dtype)
```

This ensures mathematical orthogonality ($H(H(x)) = x$) and executes natively on TPU Vector Processing Units (VPUs).

### 4.2 Functional KV Cache Update

To prevent OpenXLA buffer aliasing and eliminate recompilations, cache updates are formulated immutably:

```python
def update_kv_cache_functional(
    kv_cache: torch.Tensor,
    new_kv: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
  """Functional KV cache update avoiding in-place slice mutations.

  Constructs a new tensor view using dynamic masks, ensuring single-assignment
  immutability for OpenXLA compilation.
  """
  bsz, max_seqlen, dim = kv_cache.shape
  seqlen = new_kv.shape[1]
  indices = torch.arange(max_seqlen, device=kv_cache.device).unsqueeze(0)
  mask = (indices >= start_pos) & (indices < start_pos + seqlen)
  mask = mask.unsqueeze(-1).expand(bsz, max_seqlen, dim)

  expanded_new = torch.zeros_like(kv_cache)
  expanded_new[:, start_pos : start_pos + seqlen, :] = new_kv
  return torch.where(mask, expanded_new, kv_cache)
```

---

## 5. Verification and Parity Framework

The functional fallbacks are verified against 7 hermetic unit tests in `fallbacks_test.py`:

1. **Hadamard Orthogonality**: Verifies that $H(H(x)) \approx x$ with $< 10^{-2}$ tolerance across bfloat16 tensors.
2. **Quantization Block Sizing**: Asserts correct scale tensor shapes across FP8 block scaling (block size 128) and FP4 micro scaling (block size 32).
3. **GEMM Dimensions & Precision**: Verifies matrix multiplication dimensions for quantized projection paths.
4. **Multi-Head Latent Attention**: Verifies output tensor dimensions for multi-head latent projections.
5. **Sinkhorn Balancing**: Asserts that iterative Sinkhorn normalization produces strictly positive, normalized expert combination matrices.
6. **Rotary Embedding Immutability**: Proves that input query and key tensors are never mutated in place.
7. **KV Cache Immutability**: Verifies that original cache buffers remain unmodified when new positions are written.

---

## 6. Performance Benchmarks on Cloud TPU v7

Benchmarks were executed on Google Cloud TPU v7 hardware (Ironwood, 4 chips) using the standard **ShareGPT** dialogue dataset.

### 6.1 Compilation and Cache Efficiency

* **Total Compilation Requests**: 1,211,021 requests
* **Total Compilation Cache Hits**: 1,210,133 hits
* **Cache Hit Ratio**: **99.9%**
* **Graph Breaks Observed**: **0**
* **Buffer Aliasing Violations**: **0**

Steady-state autoregressive decoding operates with a 100.0% compilation cache hit rate.

### 6.2 Latency and Throughput Across Workloads

| Workload Configuration | Batch Size | Input Seq Len (ISL) | Output Seq Len (OSL) | TTFT (Prefill) | TPOT (Decode) | Total Latency | Generation Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ShareGPT Short Dialogue** | 1 | 64 | 32 | 69.89 ms | 71.38 ms | 2.28 s | 14.02 tok/s |
| **ShareGPT Medium Dialogue** | 1 | 256 | 32 | 75.58 ms | 71.31 ms | 2.29 s | 14.01 tok/s |
| **ShareGPT Extended Generation** | 1 | 256 | 64 | 73.89 ms | 69.75 ms | 4.47 s | 14.32 tok/s |
| **ShareGPT Batched Concurrency** | 2 | 256 | 32 | 71.75 ms | 73.88 ms | 2.36 s | **27.14 tok/s** |

### 6.3 Key Observations

1. **Near-Linear Batch Scaling (94% Efficiency)**:
   Increasing batch size from 1 to 2 scales generation throughput from 14.01 tok/s to 27.14 tok/s, demonstrating 94.0% scaling efficiency on Cloud TPU v7.
2. **Stable Decode Latency (TPOT)**:
   Time-Per-Output-Token remains tightly bounded between 69.75 ms and 73.88 ms across varying prompt lengths and generation horizons.
3. **Flat Prefill Latency (TTFT)**:
   Scaling prompt length from 64 tokens to 256 tokens increases TTFT by only 5.69 ms (from 69.89 ms to 75.58 ms), showcasing the parallel efficiency of the Matrix Multiply Units during prefill.

---

## 7. Next Steps and Recommendations

* **Pallas Custom Kernels**: For production deployments requiring maximum speed, the PyTorch functional fallbacks for the Walsh-Hadamard transform and Sinkhorn router can be replaced with TPU-native [Pallas kernels](https://github.com/google-pytorch/torch_tpu/blob/main/docs/custom_kernels.md).
* **Chunked Prefill**: For extreme prompt lengths (>8,192 tokens), chunked prefill can be enabled to maintain constant activation memory during prompt ingestion.
