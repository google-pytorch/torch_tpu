"""Compares torch.addbmm on TPU against CPU for the cases the unit tests cover.

Run on a TPU VM with torch_tpu installed:
  python3 scripts/checks/check_addbmm.py
"""

import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401

DEVICE = "tpu"


def _operands(dtype=torch.float32, batches=2):
  m = torch.arange(15.0).reshape(3, 5).to(dtype)
  b1 = (torch.arange(batches * 12.0).reshape(batches, 3, 4) / 8).to(dtype)
  b2 = (torch.arange(batches * 20.0).reshape(batches, 4, 5) / 8).to(dtype)
  return m, b1, b2


def _compare(name, fn, **tolerances):
  expected = fn("cpu")
  actual = fn(DEVICE).cpu()
  ok = torch.allclose(actual, expected, equal_nan=True, **tolerances)
  ok = ok and actual.dtype == expected.dtype and actual.shape == expected.shape
  print(f"{name}: allclose: {ok}")
  if not ok:
    print("  expected:", expected)
    print("  actual:  ", actual)


def main():
  m, b1, b2 = _operands()

  def default(device):
    return torch.addbmm(m.to(device), b1.to(device), b2.to(device))

  _compare("default beta/alpha", default)

  def beta_zero_nan_input(device):
    nan_input = torch.full((3, 5), float("nan"), device=device)
    return torch.addbmm(nan_input, b1.to(device), b2.to(device), beta=0)

  _compare("beta=0 with NaN input", beta_zero_nan_input)

  def scaled(device):
    return torch.addbmm(
        m.to(device), b1.to(device), b2.to(device), beta=0.5, alpha=2
    )

  _compare("alpha=2 beta=0.5", scaled)

  for shape in ([5], [1], []):
    bias = torch.arange(1.0, 1.0 + torch.Size(shape).numel()).reshape(shape)

    def broadcast(device, bias=bias):
      return torch.addbmm(bias.to(device), b1.to(device), b2.to(device))

    _compare(f"input broadcast from {shape}", broadcast)

  m1, b1_one, b2_one = _operands(batches=1)

  def batch_one(device):
    return torch.addbmm(m1.to(device), b1_one.to(device), b2_one.to(device))

  _compare("batch size 1", batch_one)

  def batch_zero(device):
    b1_empty = torch.empty(0, 3, 4, device=device)
    b2_empty = torch.empty(0, 4, 5, device=device)
    return torch.addbmm(m.to(device), b1_empty, b2_empty, beta=0.5, alpha=2)

  _compare("batch size 0 (beta * input)", batch_zero)

  mh, b1h, b2h = _operands(dtype=torch.bfloat16)

  def bfloat16(device):
    return torch.addbmm(mh.to(device), b1h.to(device), b2h.to(device))

  _compare("bfloat16", bfloat16, rtol=1e-2, atol=1e-2)

  def out_variant(device):
    out = torch.empty(3, 5, device=device)
    torch.addbmm(m.to(device), b1.to(device), b2.to(device), out=out)
    return out

  _compare("out=", out_variant)

  def inplace(device):
    return m.to(device).clone().addbmm_(b1.to(device), b2.to(device))

  _compare("in-place", inplace)

  try:
    torch.addbmm(
        m.to(DEVICE),
        torch.ones(1, 3, 4, device=DEVICE),
        torch.ones(2, 4, 5, device=DEVICE),
    )
    print("mismatched batch dimensions: no exception raised")
  except RuntimeError as e:
    print(f"mismatched batch dimensions: {type(e).__name__}: {e}")


if __name__ == "__main__":
  main()
