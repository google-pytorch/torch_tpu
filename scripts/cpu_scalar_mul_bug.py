"""On main: compiled `x * cpu_int_tensor[1]` returns zeros. Narrow it down."""
import torch
tpu = torch.device("tpu")
x = torch.arange(6.0).reshape(2, 3).to(tpu)
cum = torch.tensor([0, 2]); cumf = torch.tensor([0.0, 2.0]); two = torch.tensor(2)
cases = {
  "x * cum[1]            (cpu i64 getitem)": lambda x: x * cum[1],
  "x * cumf[1]           (cpu f32 getitem)": lambda x: x * cumf[1],
  "x * two               (cpu 0-d i64 input)": lambda x: x * two,
  "x + cum[1]": lambda x: x + cum[1],
  "x * cum[1].to(x.device)": lambda x: x * cum[1].to(x.device),
  "x * cum[1].item()": lambda x: x * cum[1].item(),
  "x[cum[1] - 1]        (cpu index)": lambda x: x[cum[1] - 1],
}
for name, fn in cases.items():
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend="tpu")(x).cpu()
        exp = fn(x.cpu())
        print(("OK  " if torch.equal(out, exp) else "BAD ") + name, "->", out.flatten().tolist()[:6], "expected", exp.flatten().tolist()[:6])
    except Exception as e:
        print("FAIL", name, str(e).splitlines()[-1][:150])
print("eager cum[1].to(tpu):", cum[1].to(tpu).item(), " eager x*cum[1]:", (x * cum[1]).cpu().flatten().tolist()[:3])
