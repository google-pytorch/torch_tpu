"""TPU vs CPU semantics check for torch.logaddexp2. Run on the TPU VM."""
import torch
import torch_tpu  # noqa: F401

inf = float("inf")
a = torch.tensor([0.0, 1.0, -1.0, 100.0, -100.0, inf, -inf, inf, float("nan")])
b = torch.tensor([0.0, -1.0, 1.0, -100.0, 100.0, inf, -inf, -inf, 1.0])
cpu = torch.logaddexp2(a, b)
tpu = torch.logaddexp2(a.to("tpu"), b.to("tpu")).cpu()
print("cpu", cpu.tolist())
print("tpu", tpu.tolist())
print("vectors allclose:", torch.allclose(cpu, tpu, equal_nan=True))

x = torch.arange(6.0).reshape(2, 3)
y = torch.tensor([0.5, -2.0, 3.0])
print("broadcast allclose:",
      torch.allclose(torch.logaddexp2(x, y),
                     torch.logaddexp2(x.to("tpu"), y.to("tpu")).cpu()))
xb, yb = x.to(torch.bfloat16), y.to(torch.bfloat16)
print("bfloat16 allclose:",
      torch.allclose(torch.logaddexp2(xb, yb).float(),
                     torch.logaddexp2(xb.to("tpu"), yb.to("tpu")).cpu().float(),
                     rtol=1e-2, atol=1e-2))
try:
  torch.logaddexp2(torch.tensor([1, 2], device="tpu"),
                   torch.tensor([1, 2], device="tpu"))
  print("int: no error (BAD)")
except NotImplementedError as e:
  print("int ->", e)
