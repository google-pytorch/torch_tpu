"""CPU view arguments to compiled TPU partitions (PR 6106). Run on the TPU VM.

Covers the original repro (CPU view with a storage offset) and the four
stride-dependent cases apullin reported on the PR, which pass on main and
broke with the contiguous_format version of the fix.
"""
import os

os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", "0")

import torch  # noqa: E402
import torch_tpu  # noqa: E402,F401

results = []


def check(name, f, *args):
  expected = f(*[a.cpu() if isinstance(a, torch.Tensor) else a for a in args])
  try:
    actual = torch.compile(f, backend="tpu", fullgraph=True)(*args).cpu()
    ok = torch.allclose(actual, expected.to(actual.dtype), equal_nan=True)
    detail = "" if ok else f" expected={expected.tolist()} got={actual.tolist()}"
  except Exception as e:  # noqa: BLE001
    ok, detail = False, f" raised {type(e).__name__}: {str(e)[:160]}"
  results.append(ok)
  print(f"{name}: {'OK' if ok else 'FAIL'}{detail}")


# Original repro: view with a storage offset computed in a CPU partition.
cum = torch.tensor([0, 2])
cumf = torch.tensor([0.0, 2.0])
x6 = torch.arange(6.0, device="tpu")
check("x * cum[1]", lambda x: x * cum[1].to(x.device), x6)
check("x * cumf[1]", lambda x: x * cumf[1].to(x.device), x6)
check("x + cum[1]", lambda x: x + cum[1].to(x.device), x6)
check("x * cum[1::2]", lambda x: x * torch.tensor([0, 2, 0, 3, 0, 4])[1::2].to(x.device),
      torch.arange(3.0, device="tpu"))

# apullin's stride-dependent cases.
c = torch.arange(9.0).view(3, 3)


def f1(x, c):
  y = c.t().to(x.device)
  s = (x * y).sum().cpu()
  return y * s


def f2(x, c):
  return torch.as_strided(c.t().to(x.device), (3,), (1,), 0) + x


def f3(x, c):
  return c.t().to(x.device).t().reshape(9) + x


img = torch.arange(24.0).view(1, 2, 3, 4).contiguous(
    memory_format=torch.channels_last)


def f4(x, img):
  return img.to(x.device).permute(0, 2, 3, 1).reshape(1, 3, 8) + x


check("f1 transpose scaled by its own sum", f1, torch.ones(3, 3, device="tpu"), c)
check("f2 as_strided of transpose", f2, torch.zeros(3, device="tpu"), c)
check("f3 transpose.t().reshape", f3, torch.zeros(9, device="tpu"), c)
check("f4 channels_last permute reshape", f4, torch.zeros(1, 3, 8, device="tpu"), img)

print(f"SUMMARY: {sum(results)}/{len(results)} OK")
