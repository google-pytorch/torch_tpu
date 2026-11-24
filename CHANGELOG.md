# Changelog

This changelog tracks user-facing changes, new APIs, deprecations, and runtime
updates across TorchTPU releases.

## 0.3.0 (2026-11-10)

### Python, PyTorch, and packaging support

*   Added wheel builds and dependency lockfiles for Python 3.15 (`cp315`) and
    free-threaded Python 3.15t (`cp315t`), alongside existing support for Python
    3.11, 3.12, 3.13, 3.14, and 3.14t.
*   Promoted PyTorch 2.15.0 into the standard multi-ABI wheel build
    (`torch>=2.11.0` on Python `<3.15`, `torch>=2.15.0` on Python `>=3.15`).
*   Updated the minimum required `libtpu` version to `libtpu>=0.0.50`.
*   Upgraded the open-source build toolchain to Bazel 9.3.1 and `rules_python`
    2.4.6.

### Public and experimental Python APIs

*   Added on-device tensor layout inspection and specification APIs:
    `torch.tpu.TpuLayout` (with `.element_size_in_bits`, `.minor_to_major`, and
    `.tiles`), `torch.tpu.LayoutContext`, and
    `torch.tpu.get_device_layout_if_materialized`.
*   Added `torch.tpu.TpuVersion` (`V5E`, `V5P`, `V6E`, `V7`, `V8I`, and `V8T`)
    and expanded `torch.tpu.get_device_properties()` with `.version`, `.name`,
    `.topology`, `.devices_per_chip`, `.hbm_bytes_per_device`,
    `.hbm_bytes_per_chip`, `.bf16_flops_per_device`, `.bf16_flops_per_chip`,
    `.sparse_cores_per_device`, and `.sparse_core_simd_width`.
*   Added `torch.tpu.pallas.verifier` (`verify`, `VerificationMode`,
    `VerificationOptions`, `VerificationReport`, `VerificationFinding`,
    `VerificationError`, `Phase`, `Severity`, and `FindingCategory`) to check FX
    input donation, `@jax_op` contracts, TPU lowering, and autograd parity for
    custom Pallas kernels.
*   Extended `torch.tpu.pallas.jax_op` to accept `AbstractMesh`, multi-axis
    `PartitionSpec`, and negative indices in `donate_argnums`.
*   Exposed `ParamGroup.{params, opt_steps, opt_state_m, opt_state_v}` on
    `torch.tpu.optimizers.AdamW`, `torch.tpu.optimizers.Muon`, and
    `torch.tpu.optimizers.Sgd`.
*   Added SimpleFSDP (`DTensor`) model support to
    `torch.tpu.optimizers.SingleTraceTrainer`.
*   Exposed `torch.tpu.Stream` and `torch.tpu.Event` directly as classes rather
    than factory functions.

### Deprecations

*   Deprecated `torch.tpu.get_device_name` in favor of
    `torch.tpu.get_device_properties().name`.
*   Deprecated `torch.tpu.get_device_properties().total_memory` in favor of
    `torch.tpu.get_device_properties().hbm_bytes_per_device`.
*   Deprecated `torch.tpu.get_device_properties().multi_processor_count` in
    favor of `torch.accelerator.device_count()`.

### Attention, custom kernels, and SparseCore

*   Migrated FlashAttention kernels to Kiln and added support for unaligned
    sequence lengths without manual padding, sliding-window causal masks, and
    broadcasted `attn_mask` inputs over query and key dimensions.
*   Added `score_mod` and `mask_mod` support and automatic folding of
    equal-length packed documents in `flex_attention`.
*   Updated FlashAttention and SplashAttention VMEM tiling heuristics and routed
    short single-tile sequences through XLA scaled dot-product attention.
*   Added a rowwise scaled grouped matmul Pallas kernel for TPU v7x, lowered 2D
    by 2D `_grouped_mm` (ragged `K`) to the grouped matmul `grad_y` form, fused
    transposed 3D expert weights into `RaggedDot`, and preserved `NestedTensor`
    metadata in `align_jagged_to_multiple` and `strip_jagged_padding`.
*   Added fixed-window minibatching for SparseCore embedding lookups in forward
    and backward passes, configured SparseCore gradient output layouts before
    dispatch, and wired `sc_simd_width` and `hbm_word_size_in_4b` through
    SparseCore input preprocessing.

### Operator lowerings, dtypes, and numerics

*   Added `torch.float8_e5m3fnu` (`f8E5M3FNU`) dtype support and guarded
    unsupported `float8` and `float4` inputs across operators to match PyTorch
    CUDA error behavior.
*   Added native StableHLO lowerings for `aten::einsum` (forward and backward),
    `aten::nonzero_static`, `aten::_upsample_bilinear2d_aa.out`,
    `aten::_upsample_bicubic2d_aa.out`,
    `aten::upsample_linear1d_backward.grad_input`,
    `aten::_weight_norm_interface_backward`, `native_batch_norm_backward`,
    separable matmul bilinear upsampling, non-divisible `adaptive_avg_pool2d`
    and `adaptive_avg_pool3d`, and host-sync-free `torch.linalg.lu` under
    `torch.compile`.
*   Added bounded dynamic shape support in `avg_pool` backward and `tril`/`triu`
    lowerings, and rewrote dynamic `nonzero` in Kiln with automatic unbounded
    kernel loading.
*   Fixed numerical accuracy and edge cases in `int64` `torch.lcm`,
    `aten::_padded_dense_to_jagged_forward`, `torch.exp2`, `torch.xlogy`,
    `torch.logit`, `bfloat16`/`float16` inverse trig and hyperbolic ops, `gelu`
    CDF calculation via `erfc`, `leaky_relu` backward at `0`, `nan_to_num`
    `-0.0` compilation cache keys, floating-point `remainder` with infinite
    divisors, and `grid_sampler` backward 1D scatter.

### Distributed runtime, streams, and documentation

*   Added subslice compilation and pipeline parallelism support so process
    subgroups (such as tensor-parallel pipeline stages) compile and launch
    stage-local collectives independently of other subgroups.
*   Executed point-to-point (`P2P`) sends asynchronously with receive readiness
    tracking, tracked per-stream tensor sequence numbers for transfer keys, and
    deleted consumed `P2P` descriptors from the `c10d` store.
*   Honored the launcher MegaScale base port in multi-slice jobs, skipped
    creating a MegaScale PJRT client when the world fits in a single slice,
    avoided single-slice subslice lookups on multi-slice device assignments,
    built FSDP2 `split_with_sizes_copy` outputs without per-rank views, and
    expanded `${RANK}` in `XLA_FLAGS` and `LIBTPU_INIT_ARGS` before caching.
*   Fixed `TpuStream.wait_event` to block the stream it is called on, separated
    user-facing `Event` objects from internal background materialization events,
    recorded async assertion failures only on the active stream, made the
    visible TPU device count consistent before and after runtime initialization,
    reported eager cache hits in `TORCH_TRACE`, and excluded `momentum` from the
    `BatchNorm` compilation cache key.
*   Reorganized the documentation site and added an FSDP and SimpleFSDP best
    practices case study.

## 0.2.0 (2026-10-19)

*   Initial public release of TorchTPU (`torch-tpu`) on PyPI, providing an eager
    and `torch.compile` backend for Google Cloud TPUs across Python 3.11–3.14
    (including free-threaded 3.14t) and PyTorch 2.11–2.14.
