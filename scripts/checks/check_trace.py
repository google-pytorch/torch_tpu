"""Compares aten::trace on TPU against CPU on representative inputs."""

import torch
import torch_tpu  # pylint: disable=unused-import


def check(name, fn):
  cpu = fn("cpu")
  tpu = fn("tpu").cpu()
  ok = cpu.dtype == tpu.dtype and torch.allclose(cpu, tpu)
  print(f"{name}: allclose: {ok} (cpu={cpu.item()}, tpu={tpu.item()})")


check("3x3 float32", lambda d: torch.trace(torch.arange(9.0, device=d).reshape(3, 3)))
check("2x3 wide", lambda d: torch.trace(torch.arange(6.0, device=d).reshape(2, 3)))
check("3x2 tall", lambda d: torch.trace(torch.arange(6.0, device=d).reshape(3, 2)))
check("0x0", lambda d: torch.trace(torch.empty(0, 0, device=d)))
check("1x1", lambda d: torch.trace(torch.tensor([[2.5]], device=d)))
# The pinned CPU torch has no bfloat16 trace kernel; reference via float32.
xb = torch.arange(9.0).reshape(3, 3).to(torch.bfloat16)
cpu_bf16 = torch.trace(xb.float()).to(torch.bfloat16)
tpu_bf16 = torch.trace(xb.to("tpu")).cpu()
print(f"bfloat16: allclose: {torch.equal(cpu_bf16, tpu_bf16)} (cpu={cpu_bf16.item()}, tpu={tpu_bf16.item()}, dtype={tpu_bf16.dtype})")
check("int64", lambda d: torch.trace(torch.arange(1, 10, device=d).reshape(3, 3)))
check(
    "int32 -> int64",
    lambda d: torch.trace(torch.arange(1, 10, device=d, dtype=torch.int32).reshape(3, 3)),
)

print("--- error cases")
for shape in [(3,), (2, 2, 2), ()]:
  try:
    torch.trace(torch.ones(shape, device="tpu"))
  except RuntimeError as e:
    print(f"shape {shape}: {type(e).__name__}: {e}")
