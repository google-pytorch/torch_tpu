"""Compares aten::std_mean.correction on TPU with CPU.

Usage: python check_std_mean.py
"""

import warnings

import torch
import torch_tpu  # pylint: disable=unused-import


def check(name, fn, **kwargs):
  """Runs fn on CPU and TPU and prints whether the outputs are close."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    cpu_outs = fn("cpu")
    tpu_outs = fn("tpu")
  ok = all(
      torch.allclose(t.cpu(), c, equal_nan=True, **kwargs)
      for t, c in zip(tpu_outs, cpu_outs)
  )
  print(f"{name}: allclose: {ok}")
  if not ok:
    for t, c in zip(tpu_outs, cpu_outs):
      print("  tpu:", t.cpu())
      print("  cpu:", c)


def main():
  torch.manual_seed(0)
  x32 = torch.randn(3, 4, 5)
  x16 = x32.to(torch.bfloat16)

  def std_mean(x, **kwargs):
    return lambda device: torch.std_mean(x.to(device), **kwargs)

  check("dim=None", std_mean(x32))
  check("dim=1", std_mean(x32, dim=1))
  check("dim=(0, 2)", std_mean(x32, dim=(0, 2)))
  check("dim=1 keepdim", std_mean(x32, dim=1, keepdim=True))
  check("dim=(0, 2) keepdim", std_mean(x32, dim=(0, 2), keepdim=True))
  check("correction=0", std_mean(x32, dim=1, correction=0))
  check("correction=1", std_mean(x32, dim=1, correction=1))
  check("correction=2", std_mean(x32, dim=1, correction=2))
  check("correction == N", std_mean(x32, dim=1, correction=4))
  check("correction > N", std_mean(x32, dim=1, correction=5))
  check("bfloat16 dim=None", std_mean(x16), rtol=1e-2, atol=1e-2)
  check("bfloat16 dim=2", std_mean(x16, dim=2), rtol=1e-2, atol=1e-2)

  def std_and_mean(device):
    x = x32.to(device)
    std, mean = torch.std_mean(x, dim=1)
    return std, mean, torch.std(x, dim=1), torch.mean(x, dim=1)

  check("matches torch.std and torch.mean", std_and_mean)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", UserWarning)
    std, _ = torch.std_mean(torch.ones(1, device="tpu"), correction=1)
  print("correction >= N warning:", [str(w.message) for w in caught])
  print("correction >= N std is nan:", bool(torch.isnan(std).all()))

  try:
    torch.std_mean(torch.arange(6, device="tpu").reshape(2, 3))
    print("integer input: no exception")
  except RuntimeError as e:
    print("integer input:", type(e).__name__, e)


if __name__ == "__main__":
  main()
