"""Minimal repro: a CPU int64 tensor captured by the compiled region and used as an index."""
import sys, torch
from absl import logging
logging.set_verbosity(logging.INFO)
tpu = torch.device("tpu")
cum = torch.tensor([0, 1])  # CPU, like Captum's num_outputs_cumsum
cases = {
  "select_with_cpu_scalar_tensor": lambda x: x[cum[0]] * 2,
  "mul_by_cpu_scalar_tensor":      lambda x: x * cum[1],
  "cpu_only_then_python_int":      lambda x: x[int(cum[0])] * 2,
}
x = torch.arange(6.0).reshape(2, 3).to(tpu)
for name, fn in cases.items():
    if len(sys.argv) > 1 and sys.argv[1] != name: continue
    print(f"===== {name}")
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend="tpu")(x)
        print("OK", out.cpu().tolist(), "eager:", fn(x.cpu()).tolist())
    except Exception as e:
        print("FAIL", str(e).splitlines()[-1][:200])
