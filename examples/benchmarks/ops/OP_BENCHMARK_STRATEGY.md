# TorchTPU Op Benchmark Strategy

This document outlines the two-phase strategy for op-level benchmarking and data
deduplication.

---

## 1. Prototyping vs. Production Op Capture Strategy

### Phase 1: Prototyping & UI Iteration (Fast)

*   **Op Capture on CPU**: Perform fast offline op trace capture on CPU using
    `OpCaptureMode` to extract op signatures and execution frequencies across
    models (e.g. HuggingFace model suite) without needing TPU VM hardware
    allocation.
*   **De-duplication**: Group captured ops into unique
    `(op_name, args_shapes, kwargs)` signatures.

### Phase 2: Production Deployment (High Precision)

*   **Op Capture on TPU Hardware**: Execute official op trace capture on TPU
    (`device='tpu'`) to guarantee capturing:
    *   **TPU Native Overrides**: Native C++ kernels (e.g.,
        `torch_tpu::scaled_mm_v2`, FlashAttention, TPU RMSNorm).
    *   **Device Memory & Transfers**: Explicit `aten._to_copy` to `tpu:0` and
        layout alignment (`aten.contiguous`).
    *   **TPU Dynamo Tracing**: `PrivateUse1` backend compiler lowerings under
        `torch.compile(backend="torch_tpu")`.
*   **Single-Execution Benchmark**: On TPU hardware, benchmark each unique op
    signature **once**, preventing $N \times M$ redundant benchmarks across
    models.
