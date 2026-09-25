---
torch_tpu_version: 0.1.1.dev20260922
---

```python
import os

# Enable Fused Eager mode globally before TPU device initialization
os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"

import torch
```

# **TorchTPU Eager Mode and Execution Control**

This guide provides an architectural overview and practical walkthrough of
TorchTPU's execution modes. TorchTPU supports multiple dispatch and execution
paradigms to balance interactive debugging flexibility with high-throughput
hardware utilization on Google Cloud TPUs.

## **1. The Three Eager Execution Modes**

TorchTPU provides three distinct execution modes controlled via environment
variables:

| Mode     | Environment Variable            | Dispatch         | Typical Use     |
:          :                                 : Behavior         : Case            :
| :------- | :------------------------------ | :--------------- | :-------------- |
| **Strict | *(Default when unset or `"0"`)* | Ops are          | Interactive     |
: Eager**  :                                 : dispatched       : exploration,    :
:          :                                 : immediately and  : CPU/GPU drop-in :
:          :                                 : executed         : parity.         :
:          :                                 : asynchronously   :                 :
:          :                                 : on the device    :                 :
:          :                                 : stream.          :                 :
| **Fused  | `TORCH_TPU_DEFER_AND_FUSE="1"`  | Operations are   | High-throughput |
: Eager**  :                                 : deferred and     : eager training  :
:          :                                 : compiled as      : & inference.    :
:          :                                 : fused multi-op   :                 :
:          :                                 : XLA clusters     :                 :
:          :                                 : prior to         :                 :
:          :                                 : materialization. :                 :
| **Debug  | `TORCH_TPU_LAUNCH_BLOCKING="1"` | Ops are          | Step-by-step    |
: Eager**  :                                 : dispatched one   : debugging,      :
:          :                                 : at a time and    : pinpointing     :
:          :                                 : the host CPU     : NaNs or runtime :
:          :                                 : synchronizes     : crashes.        :
:          :                                 : synchronously    :                 :
:          :                                 : after each op.   :                 :

--------------------------------------------------------------------------------

## **2. Configuring Execution Modes via Environment Variables**

Because TorchTPU initializes its runtime and driver configurations upon first
device touch, execution modes must be configured via environment variables
**before** initializing TPU operations (either in your shell or before the first
TPU tensor is created):

### **A. Shell Configuration (Recommended)**

```bash
# 1. High-Performance Fused Eager Mode (Recommended for eager workloads)
export TORCH_TPU_DEFER_AND_FUSE="1"

# 2. Synchronous Debug Eager Mode (Pinpointing crashes, equivalent to CUDA_LAUNCH_BLOCKING=1)
export TORCH_TPU_LAUNCH_BLOCKING="1"

# 3. Strict Eager Mode (Standard PyTorch eager behavior)
unset TORCH_TPU_DEFER_AND_FUSE
unset TORCH_TPU_LAUNCH_BLOCKING
```

### **B. Programmatic Configuration (Before Device Initialization)**

```python
import os

# Enable Fused Eager mode before importing torch or creating TPU tensors:
os.environ["TORCH_TPU_DEFER_AND_FUSE"] = "1"

import torch
device = torch.device("tpu")
```

## **3. Under the Hood: Graph Fusion & Materialization**

How does **Fused Eager** (`TORCH_TPU_DEFER_AND_FUSE="1"`) work on TPU silicon?

1.  **Deferred Queueing**: When element-wise or arithmetic operations are
    called, TorchTPU records them in an internal deferred execution buffer
    rather than launching individual device kernels immediately.
2.  **Multi-Op Fusion**: When execution is forced by a **materialization
    trigger** (such as transferring data to CPU via `.cpu()`, inspecting values
    via `.item()`, or calling `torch.tpu.synchronize()`), XLA compiles the
    accumulated operations into a single fused hardware cluster.
3.  **Hardware Efficiency**: Intermediate tensors stay resident in fast on-chip
    Vector Memory (VMEM) or VPU registers instead of suffering multiple
    High-Bandwidth Memory (HBM) round-trips.

We can observe this deferred queueing and materialization behavior directly:

```python
import time as _time

# Verify active environment setting
print(
    "TORCH_TPU_DEFER_AND_FUSE:"
    f" {os.environ.get('TORCH_TPU_DEFER_AND_FUSE', '0')}"
)

# Initialize TPU tensor
_x = torch.randn(2048, 2048, device="tpu", dtype=torch.bfloat16)

# Queue operations into the deferred execution buffer
torch.tpu.synchronize()
_t0 = _time.perf_counter()
_y = _x + 1.0
_z = _y * 2.0 - 0.5
_defer_ms = (_time.perf_counter() - _t0) * 1000

# Materialization via host copy (.cpu()) forces fused compilation and execution
_t_mat = _time.perf_counter()
_z_cpu = _z.cpu()
_materialize_ms = (_time.perf_counter() - _t_mat) * 1000

# Explicit device synchronization
_t_sync = _time.perf_counter()
torch.tpu.synchronize()
_sync_ms = (_time.perf_counter() - _t_sync) * 1000

print(
    f"Deferred Queue Time:              {_defer_ms:.3f} ms (asynchronous op"
    " recording)"
)
print(
    f"Materialization (.cpu()) Time:    {_materialize_ms:.3f} ms (fused"
    " compile + execute)"
)
print(f"Explicit Sync (synchronize()):    {_sync_ms:.3f} ms")
print(f"Materialized Sample Value:        {_z_cpu[0, 0].item():.4f}")
```

## **4. Comparing Fused Eager vs. `torch.compile(backend="tpu")`**

While **Fused Eager** mode fuses adjacent element-wise operations on the fly
without changing your eager development workflow, **`torch.compile(...,
backend="tpu")`** provides full Ahead-of-Time (AOT) whole-graph optimization
across entire models.

The benchmark below compares step times for a mathematical chain under **Fused
Eager** versus **`torch.compile(backend="tpu")`**:

```python
import time as _time

# Benchmark on TPU Silicon
_size = 4096
_x = torch.randn(_size, _size, device="tpu", dtype=torch.bfloat16)

def _run_chain(t):
  _y = t
  for _ in range(10):
    _y = (_y + 1.0) * 2.0 - 0.5
  return _y

_compiled_chain = torch.compile(_run_chain, backend="tpu")

# --- Warmup ---
for _ in range(3):
  _ = _run_chain(_x)
  torch.tpu.synchronize()
  _ = _compiled_chain(_x)
  torch.tpu.synchronize()

# --- Benchmark ---
_num_trials = 20

# Fused Eager benchmark
torch.tpu.synchronize()
_t0 = _time.perf_counter()
for _ in range(_num_trials):
  _ = _run_chain(_x)
  torch.tpu.synchronize()
_time_fused = (_time.perf_counter() - _t0) / _num_trials

# torch.compile benchmark
torch.tpu.synchronize()
_t0 = _time.perf_counter()
for _ in range(_num_trials):
  _ = _compiled_chain(_x)
  torch.tpu.synchronize()
_time_compiled = (_time.perf_counter() - _t0) / _num_trials

print(
    "Fused Eager (TORCH_TPU_DEFER_AND_FUSE=1) Step Time:"
    f" {_time_fused * 1000:.3f} ms"
)
print(
    "Compiled (torch.compile(backend='tpu')) Step Time: "
    f" {_time_compiled * 1000:.3f} ms"
)
if _time_compiled > 0:
  print(
      "Relative Performance:                              "
      f" {_time_fused / _time_compiled:.2f}x of compiled peak"
  )
```

## **5. Materialization Triggers in Fused Eager**

In **Fused Eager** mode, operations remain queued in the deferred buffer until a
materialization trigger forces execution:

*   **Host Transfers**: `.cpu()`, `.item()`, `.tolist()`, `.numpy()`.
*   **Explicit Synchronization**: `torch.tpu.synchronize()`.
*   **Value Inspection / Printing**: `print(tensor)` or `repr(tensor)` forces
    materializing values to display text.
*   **Data-Dependent Control Flow**: Python conditional checks (e.g., `if
    (tensor > 0).all():`) require host evaluation.
*   **Buffer Thresholds**: TorchTPU automatically flushes and compiles when
    operation queues reach internal memory or node limits.

--------------------------------------------------------------------------------

## **6. Summary & Recommendations**

| Workload / Use Case  | Recommended Configuration      | Rationale            |
| :------------------- | :----------------------------- | :------------------- |
| **Interactive        | Default (Strict Eager,         | Standard PyTorch     |
: Prototyping**        : variables unset)               : mental model;        :
:                      :                                : immediate op         :
:                      :                                : execution.           :
| **Debugging Crashes  | `export                        | Synchronous host     |
: / NaNs**             : TORCH_TPU_LAUNCH_BLOCKING="1"` : blocking after every :
:                      :                                : op; exact stack      :
:                      :                                : traces.              :
| **High-Performance   | `export                        | Multi-op fusion in   |
: Eager Training**     : TORCH_TPU_DEFER_AND_FUSE="1"`  : VMEM without         :
:                      :                                : whole-graph          :
:                      :                                : compilation.         :
| **Production Serving | `torch.compile(model,          | End-to-end           |
: & Peak Throughput**  : backend="tpu")`                : whole-graph          :
:                      :                                : optimization across  :
:                      :                                : forward and backward :
:                      :                                : passes.              :
