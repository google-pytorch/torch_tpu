import torch
tpu = torch.device("tpu")
x = torch.arange(6.0).reshape(2, 3).to(tpu)
cum = torch.tensor([0, 2])
y = cum[1]; print("eager view: offset", y.storage_offset(), "to(tpu).item", y.to(tpu).item(), "to(tpu).storage_offset", y.to(tpu).storage_offset())
cases = {
  "x * cum[1].clone()": lambda x: x * cum[1].clone(),
  "x * cum[1]": lambda x: x * cum[1],
  "x * cum[0:1]  (offset 0, shape [1])": lambda x: x * cum[0:1],
  "x * cum[1:2]  (offset 1, shape [1])": lambda x: x * cum[1:2],
  "x * (cum[1] * 1)  (fresh cpu tensor)": lambda x: x * (cum[1] * 1),
}
for name, fn in cases.items():
    torch._dynamo.reset()
    out = torch.compile(fn, backend="tpu")(x).cpu(); exp = fn(x.cpu())
    print(("OK  " if torch.equal(out, exp) else "BAD ") + name, out.flatten().tolist()[:3], "expected", exp.flatten().tolist()[:3])
