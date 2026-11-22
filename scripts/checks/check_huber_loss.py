"""Compares aten::huber_loss on TPU to CPU, forward and backward.

The broadcasting case calls the aten op directly because
torch.nn.functional.huber_loss broadcasts the inputs itself.
"""

import torch
import torch_tpu  # pylint: disable=unused-import

NONE, MEAN, SUM = 0, 1, 2
X = [[-3.0, -2.0, -1.0, -0.5], [0.0, 0.5, 1.0, 2.0]]


def run(device, reduction, delta, dtype, x_data=X, y_data=None):
  x = torch.tensor(x_data, dtype=dtype, device=device, requires_grad=True)
  if y_data is None:
    y = torch.zeros_like(x)
  else:
    y = torch.tensor(y_data, dtype=dtype, device=device)
  loss = torch.ops.aten.huber_loss(x, y, reduction, delta)
  loss.sum().backward()
  return loss.detach().cpu(), x.grad.cpu()


def check(name, **kwargs):
  cpu_loss, cpu_grad = run("cpu", **kwargs)
  tpu_loss, tpu_grad = run("tpu", **kwargs)
  ok = torch.allclose(cpu_loss, tpu_loss, rtol=1e-2, atol=1e-2) and (
      torch.allclose(cpu_grad, tpu_grad, rtol=1e-2, atol=1e-2)
  )
  print(f"{name}: allclose: {ok}")
  if not ok:
    print("  cpu loss", cpu_loss, "\n  tpu loss", tpu_loss)
    print("  cpu grad", cpu_grad, "\n  tpu grad", tpu_grad)


def main():
  for reduction, label in ((NONE, "none"), (MEAN, "mean"), (SUM, "sum")):
    check(f"f32 {label} delta=1", reduction=reduction, delta=1.0,
          dtype=torch.float32)
  check("bf16 mean delta=0.5", reduction=MEAN, delta=0.5, dtype=torch.bfloat16)
  check("f32 none |d| == delta boundary", reduction=NONE, delta=0.5,
        dtype=torch.float32, x_data=[0.5, -0.5, 0.499, 0.501])
  check("f32 mean broadcast target", reduction=MEAN, delta=1.0,
        dtype=torch.float32, x_data=[-2.0, 0.0, 2.0],
        y_data=[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

  x = torch.zeros(2, device="tpu")
  for delta in (0.0, -1.0):
    try:
      torch.nn.functional.huber_loss(x, x, delta=delta)
      print(f"delta={delta}: no exception")
    except RuntimeError as e:
      print(f"delta={delta}: {type(e).__name__}: {e}")
  xi = torch.tensor([1, 2], device="tpu")
  try:
    torch.nn.functional.huber_loss(xi, xi)
    print("int64 inputs: no exception")
  except NotImplementedError as e:
    print(f"int64 inputs: {type(e).__name__}: {e}")


if __name__ == "__main__":
  main()
