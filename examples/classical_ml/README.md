# PyTorch on Cloud TPU: Classical ML Training & Telemetry Guide

A comprehensive recipe and reference guide for training PyTorch models on
**Google Cloud TPU v6e** using the native **TorchTPU (`torch_tpu`)** backend,
instrumented with **TensorBoard** and **XProf TPU hardware profiling**.

--------------------------------------------------------------------------------

## 📖 Workload Overview

This recipe demonstrates:

1.  **Dispatch Modes**: Native TPU Eager execution (`device =
    torch.accelerator.current_accelerator()`) vs. Ahead-of-Time compilation
    (`torch.compile(model, backend='tpu', dynamic=False)`).
2.  **Asynchronous Micro-Batching**: Enqueues tight micro-batch windows
    (`--micro_batches=8`) continuously onto the TPU using `non_blocking=True`
    transfers, appending detached step losses to a Python list
    (`losses.append(loss.detach())`) and performing a single on-device reduction
    (`torch.stack(losses).sum()`) after the inner loop to avoid per-step eager
    kernel dispatches and host-device synchronization stalls.
3.  **Step-Scheduled XProf & Focused Telemetry**: Uses `torch.profiler.profile`
    with `torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=1)` and
    `profiler.step()` to capture 5-plane `.xplane.pb` hardware traces complete
    with `ProfilerStep` markers for steady-state steps, alongside clean
    TensorBoard scalar logging (`Loss/train`, `Perf/step_time_ms`,
    `Accuracy/{Eager,Compiled}`).
4.  **TPU Best Practices**: `DataLoader(num_workers=0)` fork-safety, static
    batch shapes with `drop_last=True`, and systolic Matrix Execution Unit (MXU)
    tile alignment.

--------------------------------------------------------------------------------

## 🛠️ Software Stack

This recipe is designed for Cloud TPU VMs running:

*   **TPU Hardware**: Google Cloud TPU v6e (`ct6e-standard-4t`) or compatible
    Cloud TPU architectures.
*   **PyTorch**: `torch>=2.13.0` CPU build (aligned with the `torch_tpu` glue
    layer).
*   **TorchTPU**: `torch_tpu` native C++ backend and `libtpu`.
*   **Telemetry**: `tensorboard`, `tensorboard-plugin-profile` / `xprof`.

--------------------------------------------------------------------------------

## 🚀 Step-by-Step Usage Guide

### Step 1: Environment Setup

Inside your Cloud TPU VM:

```bash
# 1. Create and activate a clean Python virtual environment
python3 -m venv ~/venv312
source ~/venv312/bin/activate

# 2. Install PyTorch CPU and TorchTPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-tpu libtpu

# 3. Install Telemetry & Profiler Packages
pip install tensorboard tensorboard-plugin-profile
```

Verify your hardware recognition:

```bash
python3 -c "import torch, torch_tpu; print(f'PyTorch: {torch.__version__} | Accelerator: {torch.accelerator.current_accelerator()}')"
```

--------------------------------------------------------------------------------

### Step 2: Run Training & Benchmark

Execute the training benchmark script:

```bash
python3 train_classical_ml_torchtpu.py \
    --log_dir=./runs/tpu_classical_ml \
    --batch_size=256 \
    --micro_batches=8 \
    --epochs=3
```

#### Expected Output (Verified on Cloud TPU v6e)

```text
================================================================
 TorchTPU Classical ML Training & Profiling Benchmark on TPU v6e
 Python: 3.12.13 | PyTorch: 2.13.0+cpu | TorchTPU: 0.1.1.dev
================================================================

[TensorBoard] Logging metrics to ./runs/tpu_classical_ml

--- Phase 1: Eager Mode Execution on TPU ---
[Eager] Epoch  1/ 3 | Step   8/234 | Loss: 1.8149 | Latency: 68.79ms | Throughput:  3721.6 samples/s
...
[Eager] Epoch  3 Finished in 4.962s | Avg Loss: 0.0768 | Epoch Throughput: 12071.8 samples/s
Test Accuracy (Eager): 97.44%

--- Phase 2: Compiled Mode Execution on TPU (torch.compile) ---
[Compiled (torch.compile)] Epoch  1/ 3 | Step   8/234 | Loss: 1.8149 | Latency: 170.13ms | Throughput:  1504.7 samples/s
...
[Compiled (torch.compile)] Epoch  3 Finished in 4.838s | Avg Loss: 0.0768 | Epoch Throughput: 12381.5 samples/s
Test Accuracy (Compiled): 97.44%

=================================================================
                FINAL BENCHMARK COMPARISON
=================================================================
 Metric                     | Eager Mode       | Compiled Mode
 ---------------------------+------------------+-----------------
 Warm Step Latency (ms)     |        21.21     |        20.66
 Warm Throughput (samples/s)|      12068.2     |      12393.9
 Cold Window Step Time (ms) |        68.79     |       170.13
 Total Train Duration (s)   |       15.327     |       15.738
 Test Accuracy (%)          |        97.44%    |        97.44%
 ---------------------------+------------------+-----------------
 Steady-State Speedup Factor (Warm Compiled vs Eager): 1.03x
=================================================================
```

--------------------------------------------------------------------------------

### Step 3: Capture XProf Hardware Traces & Launch TensorBoard

To capture step-scheduled TPU XProf hardware traces (`*.xplane.pb`), pass
`--enable_profiler`:

```bash
python3 train_classical_ml_torchtpu.py \
    --log_dir=./runs/tpu_classical_ml_profile \
    --enable_profiler \
    --profile_wait=2 \
    --profile_warmup=1 \
    --profile_active=3
```

Start the TensorBoard dashboard (use `--logdir=./runs/tpu_classical_ml` for the
pure benchmark run or `--logdir=./runs/tpu_classical_ml_profile` to inspect the
XProf hardware trace):

```bash
tensorboard --logdir=./runs/tpu_classical_ml_profile --port=6006
```

Open **`http://localhost:6006`** in your browser:

*   **Scalars Dashboard**:
    -   `Loss/train`: Step-level training convergence across micro-batch windows
        for Eager and Compiled modes.
    -   `Perf/step_time_ms`: Steady-state step latency in milliseconds across
        execution windows.
    -   `Accuracy/Eager` and `Accuracy/Compiled`: Final evaluation accuracy on
        the held-out test split.
    -   *(Note: Host CPU/RAM and TPU HBM memory telemetry are automatically
        captured within XProf hardware traces rather than duplicated via host
        scalar polling).*
*   **Profile Dashboard (XProf)**:
    -   **Overview Page**: Step-time breakdown (compute vs. input/host overhead)
        populated via `ProfilerStep` markers generated by
        `torch.profiler.schedule` + `profiler.step()`.
    -   **Trace Viewer**: Interactive 5-plane XLA execution timeline displaying
        `ProfilerStep#<step>` boundaries, host thread activity, and TPU device
        execution queues.
    -   **Memory Viewer & Op Profile**: TPU High Bandwidth Memory (HBM)
        allocation curve and XLA fusion arithmetic breakdown.

--------------------------------------------------------------------------------

## 💡 Key Architectural Best Practices for TPU

1.  **`DataLoader(num_workers=0)` Fork Safety**:
    *   PyTorch's default `DataLoader(num_workers > 0)` uses `os.fork()`.
        Forking worker processes after PJRT initializes libtpu duplicates
        multithreaded runtime state and hardware device locks
        (`/tmp/libtpu_lockfile`), causing workers to deadlock silently. Always
        use `num_workers=0` (or Google `Grain`).
2.  **Static Tensor Shapes (`drop_last=True`)**:
    *   XLA compiles dedicated machine code for each unique tensor dimension.
        Setting `drop_last=True` ensures consistent batch dimensions and avoids
        triggering unexpected AOT graph recompilations on the trailing batch.
3.  **MXU Systolic Tile Alignment**:
    *   TPU Matrix Execution Units (MXUs) operate on systolic tiles (e.g.
        128x128). Aligning batch sizes and hidden feature dimensions to
        multiples of 128 maximizes hardware compute utilization.
4.  **Asynchronous Execution & Deferred Loss Reduction**:
    *   Calling `torch.accelerator.synchronize()` or `loss.item()` on every
        single step is an anti-pattern that drains the asynchronous host-device
        execution queue and stalls pipelined TPU execution.
    *   Instead, group training steps into tight micro-batch windows (e.g. 8
        iterations per window), transfer tensors with `non_blocking=True`, store
        detached loss references in a Python list
        (`losses.append(loss.detach())`), and perform a single reduction
        (`torch.stack(losses).sum().item()`) after the loop. This avoids
        dispatching per-step eager addition kernels between compiled graph
        executions and synchronizes only once per window.
5.  **Step-Scheduled XProf Profiling (`torch.profiler.schedule`)**:
    *   Always profile steady-state execution steps using
        `torch.profiler.schedule(wait=2, warmup=1, active=3, repeat=1)` and call
        `profiler.step()` at the end of each macro-batch window. This skips
        initial cold XLA graph compilation, bounds trace file size, and injects
        `ProfilerStep` markers so TensorBoard's Overview Page can compute
        accurate step-time breakdowns.
6.  **Warm-Up & Compilation Amortization**:
    *   The first execution window under `torch.compile` incurs an initial AOT
        graph trace and XLA compilation cost, and Epoch 1 includes trace flush
        overhead when profiling is active. Evaluating steady-state latency from
        Epoch 2+ isolates cached XLA executable performance.
