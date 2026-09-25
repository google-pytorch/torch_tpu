---
torch_tpu_version: 0.1.1.dev20260922
---

```python
import os
import shutil
import time

# Pre-set environment variables globally for HTML export so that all examples have output
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"

dump_dir = "/tmp/hlo_dump"
if os.path.exists(dump_dir):
  shutil.rmtree(dump_dir)
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

tier2_name = "tpu_tier2_cache"
tier3_path = "/tmp/tpu_tier3_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
for p in [tier2_path, tier3_path]:
  if os.path.exists(p):
    shutil.rmtree(p)
  os.makedirs(p, exist_ok=True)

os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path
```

# **Environment Variables Reference**

TorchTPU's behavior is controlled through several environment variables that
configure the XLA compiler, the TPU driver (`libtpu`), and PyTorch's own debug
subsystem.

### ⚠️ **CRITICAL: Process-Global Initialization**

TorchTPU environment variables are **locked** the moment the backend initializes
(usually during `import torch` or `torch.device("tpu")`).

**To try a different environment variable snippet:**

1.  **Restart the Marimo Kernel** (Click the restart icon in the bottom menu or
    use the keyboard shortcut).
2.  **Run ONLY the specific cell** for the variable you want to test.

## **PyTorch Debug & Safety**

These variables control debugging context and internal safety checks for
TorchTPU.

| Variable                        | Purpose                                  |
| :------------------------------ | :--------------------------------------- |
| `TORCH_SHOW_CPP_STACKTRACES`    | Enriches Python `RuntimeError` with      |
:                                 : massive C++ backtraces from ATen and the :
:                                 : TorchTPU backend.                        :
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Enables expensive internal debug checks  |
:                                 : and invariant validation (e.g., catching :
:                                 : out-of-bounds indices).                  :

### **Snippet: C++ Stacktraces**

Setting this variable makes it much easier to diagnose errors that occur deep
within the C++ backend or XLA. Instead of a simple Python `RuntimeError`, you'll
receive a detailed `C++ CapturedTraceback` showing exact execution frames.

```python
def _run():
  # RESTART KERNEL BEFORE RUNNING
  import torch

  os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
  device_debug = torch.device("tpu")

  # Trigger a shape mismatch error
  a = torch.randn(10).to(device_debug)
  b = torch.randn(20).to(device_debug)

  print("Attempting invalid addition (10 + 20)...")
  try:
    c = a + b
  except RuntimeError as e:
    print(
        "\nCaught Expected Error with C++"
        f" Trace:\n{str(e)[:500]}...\n(Traceback truncated for brevity)"
    )

_run()
```

### **Snippet: Internal Debug Checks & Boundary Validation**

Enabling these checks helps catch subtle bugs related to index out-of-bounds
(e.g., in `torch.take`), though it significantly slows down execution due to
extra CPU/TPU synchronization. When enabled, operations that would normally fail
silently or cause data corruption instead raise an explicit exception like
`take(): expected indices to be in range`.

```python
def _run():
  # RESTART KERNEL BEFORE RUNNING
  import torch

  os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
  device_safe = torch.device("tpu")

  # DEMONSTRATE: Index Validation
  # Without this variable, TPU skips bounds checks for speed.
  try:
    x = torch.tensor([1.0, 2.0, 3.0], device=device_safe)
    indices = torch.tensor([5], device=device_safe)  # Clearly out of bounds
    print("Attempting torch.take() with out-of-bounds index...")
    y = torch.take(x, indices)
    y.cpu()
  except Exception as e:
    print(f"\nCaught Expected Debug Exception: {e}")

_run()
```

## **Hardware & Infrastructure**

These variables specify the TPU hardware topology and process networking. In
most cases, these are set automatically by the TorchTPU launcher or Google Cloud
environment.

| Variable                           | Purpose                                |
| :--------------------------------- | :------------------------------------- |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Addresses for the TPU slice builder    |
:                                    : service.                               :
| `TORCH_TPU_TOPOLOGY`               | Describes the physical topology of the |
:                                    : TPU slice.                             :

## **Execution & Compilation**

These variables control how PyTorch graphs are lowered and optimized by XLA.

| Variable    | Values                   | Purpose                            |
| :---------- | :----------------------- | :--------------------------------- |
| `XLA_FLAGS` | `--xla_dump_hlo_as_text` | Direct XLA compiler flags. Dumping |
:             : etc.                     : HLO prints hundreds of lines of    :
:             :                          : intermediate representation for    :
:             :                          : deep performance debugging.        :

### **Snippet: Dumping XLA IR (HLO)**

By setting `XLA_FLAGS`, you can force the compiler to dump the High-Level
Optimizer (HLO) text to a directory. This is invaluable for performance
debugging.

The dump directory (e.g., `/tmp/hlo_dump`) will contain:

-   **`*.after_optimizations.txt`**: The final HLO graph showing hardware-level
    operations.
-   **`*.flagfile`**: A full list of active XLA compiler flags (verifies
    `xla_optimization_level`).
-   **`*tpu_comp_env.txt`**: Internal TPU environment settings used during
    compilation.

```python
def _run():
  # RESTART KERNEL BEFORE RUNNING
  import torch
  import shutil
  import glob

  dump_dir = "/tmp/hlo_dump"
  os.makedirs(dump_dir, exist_ok=True)

  # Force dumping of HLO as text to a specific directory
  os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"
  device_hlo = torch.device("tpu")

  # Run a simple op to generate the HLO
  x = torch.randn(10).to(device_hlo)
  y = x + 1
  _ = y.to("cpu")

  # Read the dumped HLO files from the directory and print them inline
  hlo_files = glob.glob(os.path.join(dump_dir, "*after_optimizations.txt"))
  flag_files = glob.glob(os.path.join(dump_dir, "*.flagfile"))
  env_files = glob.glob(os.path.join(dump_dir, "*tpu_comp_env.txt"))

  if flag_files:
    print(f"--- Compiler Flags ({os.path.basename(flag_files[0])}) ---")
    with open(flag_files[0], "r") as f:
      # Search for specific interesting flags
      flags = f.readlines()
      for flag in flags:
        if "xla_optimization_level" in flag or "xla_dump_hlo_as_text" in flag:
          print(flag.strip())
    print()

  if env_files:
    print(f"--- TPU Env Settings ({os.path.basename(env_files[0])}) ---")
    with open(env_files[0], "r") as f:
      # Show first few lines as example of internal env state
      print("".join(f.readlines()[:5]).strip())
    print("...\n")

  if not hlo_files:
    print("No optimized HLO files generated. Check dump directory.")
  else:
    # Display the post-optimization HLO graph
    target_file = hlo_files[0]
    print(f"--- HLO IR ({os.path.basename(target_file)}) ---")
    with open(target_file, "r") as f:
      # Truncate to first 40 lines to prevent massive output spam
      lines = f.readlines()
      print("".join(lines[:40]))
      if len(lines) > 40:
        print(f"\n... (truncated {len(lines) - 40} lines) ...")

_run()
```

## **Compilation Caching**

TorchTPU uses multi-tiered caching to avoid redundant compilation across runs
and ranks. Tier 2 is stored locally within the host and Tier 3 is a compilation
cache shared between hosts. While Tier 3 usually uses shared storage (like a
`gs://` bucket), we use local folders here to demonstrate how they work.

| Variable                                 | Purpose                         |
| :--------------------------------------- | :------------------------------ |
| `TORCH_TPU_TIER2_COMPILATION_CACHE`      | Path to the tier-2 cache (local |
:                                          : within the host).               :
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Path to the tier-3              |
:                                          : (shared/persistent) cache.      :

### **Snippet: Multi-tiered Caching (Tier-2 and Tier-3)**

By pointing to specific directories, compiled binary artifacts are saved to disk
and reused in future sessions, dramatically reducing JIT latency on subsequent
runs.

```python
def _run():
  # RESTART KERNEL BEFORE RUNNING
  import torch
  import shutil
  import time

  tier2_name = "tpu_tier2_cache"
  # The backend automatically places named tier-2 caches inside /dev/shm/torch_tpu_cache/
  tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
  tier3_path = "/tmp/tpu_tier3_cache"

  # Ensure directories exist
  for p in [tier2_path, tier3_path]:
    os.makedirs(p, exist_ok=True)

  # TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME
  # that will be created inside /dev/shm/
  os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
  os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

  device_cache = torch.device("tpu")

  # Run multiple operations with different shapes/parameters to generate multiple cache entries
  def run_ops(d):
    # Op 1: 1024x1024 MatMul
    a = torch.randn(1024, 1024, device=d)
    _ = (a @ a).to("cpu")

    # Op 2: 512x512 Convolution-like op
    b = torch.randn(512, 512, device=d)
    _ = torch.sin(b).cos().to("cpu")

    # Op 3: Small vector ops
    c = torch.randn(1000, device=d)
    _ = torch.exp(c).to("cpu")

  print("Executing operations to populate cache...")
  run_ops(device_cache)

  print("Waiting for async cache writes...")
  time.sleep(4)

  def count_files(cache_path):
    count = 0
    for root, _, f in os.walk(cache_path):
      # Count only .bin files (executables)
      count += len([name for name in f if name.endswith(".bin")])
    return count

  t2_count = count_files(tier2_path)
  t3_count = count_files(tier3_path)

  print(f"\nTier-2 Cache ({tier2_path}): {t2_count} files found.")
  print(f"Tier-3 Cache ({tier3_path}): {t3_count} files found.")

  if t2_count > 0:
    print("✅ Success: Tier-2 cache is working locally in shared memory!")

_run()
```

## **Quick Reference Summary**

Variable                                 | Scope    | Must Set Before       | Possible Values
:--------------------------------------- | :------- | :-------------------- | :--------------
`CLOUD_TPU_TASK_ID`                      | Cloud    | `N/A` (Set by GCE)    | `0`, `1`, `2`, ... (integer format)
`LOCAL_RANK`                             | Launcher | `import torch`        | `0`, `1`, `2`, `3` ... (integer)
`MASTER_ADDR`                            | Launcher | `import torch`        | `localhost`, `127.0.0.1`, `10.0.0.1`
`MASTER_PORT`                            | Launcher | `import torch`        | `12355`, `8471`, `<port_number>`
`RANK`                                   | Launcher | `import torch`        | `0`, `1`, `2`, ... (integer)
`TORCH_SHOW_CPP_STACKTRACES`             | Debug    | `import torch`        | `0` (False), `1` (True)
`TORCH_TPU_ENABLE_DEBUG_CHECKS`          | Safety   | `import torch`        | `0` (False), `1` (True)
`TORCH_TPU_TIER2_COMPILATION_CACHE`      | Cache    | `torch.device("tpu")` | `/dev/shm/torch_tpu_cache/`, `<local_path>`
`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Cache    | `torch.device("tpu")` | `gs://<bucket_name>/<path_to_cache>`
`TORCH_TPU_SLICEBUILDER_ADDRESSES`       | Cloud    | `import torch`        | `10.0.0.1:8471,10.0.0.2:8471`
`TORCH_TPU_TOPOLOGY`                     | Cloud    | `import torch`        | `2x2x1`, `2x2x4`, `4x4x4`, etc.
`TPU_CHIPS_PER_HOST_BOUNDS`              | Cloud    | `N/A` (Set by GCE)    | `1,1,1`, `2,2,1`, `2,2,2`
`TPU_HOST_BOUNDS`                        | Cloud    | `N/A` (Set by GCE)    | `1,1,1`, `2,2,1`, `2,2,2`
`TPU_PROCESS_ADDRESSES`                  | Cloud    | `N/A` (Set by GCE)    | `10.0.0.1:8471,10.0.0.2:8471`
`TPU_PROCESS_PORT`                       | Cloud    | `N/A` (Set by GCE)    | `8471`, `<port_number>`
`TPU_VISIBLE_CHIPS`                      | Cloud    | `N/A` (Set by GCE)    | `0`, `0,1`, `0,1,2,3`
`WORLD_SIZE`                             | Launcher | `import torch`        | `1`, `2`, `4`, `8`, `16`, ... (integer)
