import torch
tpu = torch.device("tpu")
cum = torch.tensor([0, 2]); x = torch.arange(6.0).reshape(2, 3).to(tpu)
y = cum[1].to(tpu)
print("eager y.item", y.item(), "| (y*1).item", (y * 1).item(), "| (x*y)[0]", (x * y).cpu().flatten().tolist()[:3], "| (x+y)[0]", (x + y).cpu().flatten().tolist()[:3])
z = cum[1:2].to(tpu); print("slice [1:2] -> ", z.cpu().tolist(), "(x*z)[0]", (x * z).cpu().flatten().tolist()[:3])
w = torch.arange(12.0).reshape(3, 4)[1:3, 2:4].to(tpu); print("2d view ->", w.cpu().tolist(), "expected [[6,7],[10,11]]", "| w*1 ->", (w * 1).cpu().tolist())
with torch.no_grad():
    y2 = cum[1].to(tpu); print("no_grad y2*1:", (y2 * 1).item())
import torch._dynamo
@torch._dynamo.disable
def move(t): return t.to(tpu)
y3 = move(cum[1]); print("dynamo.disable move y3*1:", (y3 * 1).item(), "(x*y3)[0]", (x * y3).cpu().flatten().tolist()[:3])
