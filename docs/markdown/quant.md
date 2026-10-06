---
torch_tpu_version: 0.1.1.dev20260922
---

# Python Custom Ops: Quantize & Pack
Baseline single operation for quantizing and packing.

```python
import os
import sys
from typing import Final, Tuple

LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"
sys.argv.extend(
    [f"--xla_jf_dump_to={LLO_DUMP_TO}", "--xla_jf_dump_llo_text=true"]
)
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_TO} --xla_jf_dump_llo_text=true"
)

XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
)

from absl import app as absl_app  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top
import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

# pylint: disable=protected-access
torch._logging.set_logs(aot_graphs=True)
```

```python
# Copied from qat_linear.py
def quantize(values):
  """Returns signs and a scale as a quantized representation of the values."""
  scale = abs(values).mean()
  signs = torch.sign(values)
  signs[signs == 0.0] = 1.0
  return signs, scale
```

```python
# Copied from qat_linear.py
def pack32(values):
  """Packs a vector of 32 quantized values (signs) into a uint32"""
  # Omit size and dtype asserts to avoid compilation errors on TPU

  bits = (values == 1.0).to(torch.uint32)
  bit_patterns = 2 ** torch.arange(
      32, dtype=torch.uint32, device=values.device
  )
  return torch.sum(bits * bit_patterns, dim=0, dtype=torch.uint32)
```

```python
# Copied from qat_linear.py
def pack(values):
  return torch.vmap(pack32)(values)
```

```python
def quant_and_pack(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  signs, scale = quantize(values)
  packed = pack(signs)
  return packed, scale
```

```python
@torch.compile(backend="tpu")
def compiled_quant_and_pack(values):
  _dummy = None  # Force PyTorch Dynamo recompile
  return quant_and_pack(values)
```

```python
def dump_dir(directory):
  import os

  entries = os.listdir(directory) if os.path.exists(directory) else []
  print(f"\n=== Files in {directory} (Total: {len(entries)}) ===")

  # Print all files first
  for f in sorted(entries):
    print(f)

  # Print contents of all files
  for f in sorted(entries):
    full_path = os.path.join(directory, f)
    if os.path.isfile(full_path):
      print(f"\n=== File: {f} ===")
      with open(full_path, "r") as file:
        print(file.read())
```

```python
def main():
  tpu_available = False
  try:
    x = torch.randn(1024, 32, device="tpu")
    tpu_available = True
  except Exception:
    pass

  if not tpu_available:
    print("TPU not available. Run this notebook on a TPU VM.")
    return 0

  packed, scale = compiled_quant_and_pack(x)

  # Force execution
  _ = packed.cpu()
  _ = scale.cpu()

  dump_dir(XLA_DUMP_TO)
  dump_dir(LLO_DUMP_TO)

  print("Success.")
  return 0

main()
```
