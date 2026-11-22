"""Compares aten::nansum on TPU against CPU on representative inputs."""

import torch
import torch_tpu  # pylint: disable=unused-import

NAN = float("nan")


def check(name, fn):
  cpu = fn("cpu")
  tpu = fn("tpu").cpu()
  ok = cpu.dtype == tpu.dtype and torch.allclose(cpu, tpu, equal_nan=True)
  print(f"{name}: allclose: {ok} (cpu={cpu.tolist()}, tpu={tpu.tolist()})")


def matrix(device, dtype=torch.float32):
  data = [[1.0, NAN, 3.0], [NAN, NAN, NAN], [-2.0, 0.5, NAN]]
  return torch.tensor(data, device=device, dtype=dtype)


def ints(device):
  return torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.int32)


def with_out(device):
  out = torch.empty(3, device=device)
  torch.nansum(matrix(device), dim=1, out=out)
  return out


check("full reduction", lambda d: torch.nansum(matrix(d)))
check("dim=()", lambda d: torch.nansum(matrix(d), dim=()))
check("dim=1 (all-NaN row is 0)", lambda d: torch.nansum(matrix(d), dim=1))
check("dim=0 keepdim", lambda d: torch.nansum(matrix(d), dim=0, keepdim=True))
check(
    "dim=(0, 1) keepdim",
    lambda d: torch.nansum(matrix(d), dim=(0, 1), keepdim=True),
)
check("out=", with_out)
check("dtype=bfloat16", lambda d: torch.nansum(matrix(d), dtype=torch.bfloat16))
check("bfloat16 input", lambda d: torch.nansum(matrix(d, torch.bfloat16), dim=1))
check("int32 input -> int64", lambda d: torch.nansum(ints(d)))
check(
    "int32 input dtype=float32",
    lambda d: torch.nansum(ints(d), dim=1, dtype=torch.float32),
)
check(
    "bool input -> int64",
    lambda d: torch.nansum(torch.tensor([True, False, True], device=d)),
)
check("0-d NaN", lambda d: torch.nansum(torch.tensor(NAN, device=d)))
check("empty", lambda d: torch.nansum(torch.empty(0, 3, device=d), dim=0))

print("--- error cases")
try:
  out = torch.empty(3, device="tpu")
  torch.nansum(matrix("tpu"), dim=1, dtype=torch.bfloat16, out=out)
except RuntimeError as e:
  print(f"dtype/out mismatch: {type(e).__name__}: {e}")
try:
  torch.nansum(torch.tensor([1 + 1j], device="tpu"))
except NotImplementedError as e:
  print(f"complex input: {type(e).__name__}: {e}")
try:
  torch.nansum(matrix("tpu"), dim=2)
except IndexError as e:
  print(f"dim out of range: {type(e).__name__}: {e}")
