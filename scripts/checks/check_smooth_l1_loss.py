"""Compares aten::smooth_l1_loss on TPU to CPU, forward and backward.

Calls the aten op directly: torch.nn.functional.smooth_l1_loss broadcasts the
inputs itself and delegates beta == 0 to l1_loss.
"""

import torch
import torch_tpu  # pylint: disable=unused-import

NONE, MEAN, SUM = 0, 1, 2
X = [[-2.0, -1.0, -0.5, 0.0], [0.25, 0.5, 1.0, 3.0]]


def run(device, reduction, beta, dtype, x_data=X, y_data=None):
  x = torch.tensor(x_data, dtype=dtype, device=device, requires_grad=True)
  if y_data is None:
    y = torch.zeros_like(x)
  else:
    y = torch.tensor(y_data, dtype=dtype, device=device)
  loss = torch.ops.aten.smooth_l1_loss(x, y, reduction, beta)
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
    check(f"f32 {label} beta=1", reduction=reduction, beta=1.0,
          dtype=torch.float32)
  check("bf16 mean beta=0.5", reduction=MEAN, beta=0.5, dtype=torch.bfloat16)
  check("f32 none beta=0 (l1, grad -1 at d == 0)", reduction=NONE, beta=0.0,
        dtype=torch.float32)
  check("f32 none |d| == beta boundary", reduction=NONE, beta=0.5,
        dtype=torch.float32, x_data=[0.5, -0.5, 0.499, 0.501])
  check("f32 mean broadcast target", reduction=MEAN, beta=1.0,
        dtype=torch.float32, x_data=[-2.0, 0.0, 2.0],
        y_data=[[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])

  x = torch.zeros(2, device="tpu")
  try:
    torch.nn.functional.smooth_l1_loss(x, x, beta=-1.0)
    print("negative beta: no exception")
  except RuntimeError as e:
    print(f"negative beta: {type(e).__name__}: {e}")
  xi = torch.tensor([1, 2], device="tpu")
  try:
    torch.nn.functional.smooth_l1_loss(xi, xi)
    print("int64 inputs: no exception")
  except NotImplementedError as e:
    print(f"int64 inputs: {type(e).__name__}: {e}")


if __name__ == "__main__":
  main()
