import torch
tpu = torch.device("tpu")
cum = torch.tensor([0, 2]); x = torch.arange(6.0).reshape(2, 3).to(tpu)
y = cum[1].to(tpu); yc = cum[1].clone().to(tpu)
print("storage nbytes: view.to ->", y.untyped_storage().nbytes(), "| clone.to ->", yc.untyped_storage().nbytes(), "| offsets", y.storage_offset(), yc.storage_offset())
xt = torch.tensor([0, 2], device=tpu)
def t(name, fn):
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, backend="tpu")(x).cpu(); exp = fn(x.cpu()) if "xt" not in name else (x.cpu() * 2)
        print(("OK  " if torch.equal(out, exp) else "BAD ") + name, out.flatten().tolist()[:3])
    except Exception as e: print("FAIL", name, str(e).splitlines()[-1][:160])
t("x * xt[1]   (TPU view input)", lambda x: x * xt[1])
t("x * cum[1:2]", lambda x: x * cum[1:2])
t("x * cum[0:1]", lambda x: x * cum[0:1])
