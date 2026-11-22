"""Minimal LIG-style repro: forward hook replaces an embedding output with a larger-batch tensor; backward under compile."""
import torch, torch.nn as nn
tpu = torch.device("tpu"); torch.manual_seed(0)
class M(nn.Module):
    def __init__(s): super().__init__(); s.emb = nn.Embedding(50, 16); s.lin = nn.Linear(16, 50)
    def forward(s, ids): return s.lin(s.emb(ids)).mean(1)   # [B, 50]
m = M().to(tpu); ids = torch.randint(0, 50, (1, 5), device=tpu)
def run(model, steps):
    scaled = m.emb(ids).detach().repeat(steps, 1, 1).requires_grad_(True)  # [steps, 5, 16]
    h = m.emb.register_forward_hook(lambda mod, a, out: scaled)
    try:
        out = model(ids.repeat(steps, 1))   # hook swaps output -> [steps, 50]
        grads = torch.autograd.grad(out[:, 3].sum(), scaled)[0]
    finally: h.remove()
    return grads.detach().cpu()
eager = run(m, 4)
comp = torch.compile(m, backend="tpu")
try:
    c = run(comp, 4); print("compiled hook backward OK, max diff vs eager:", (c - eager).abs().max().item())
except Exception as e: print("compiled hook backward FAILED:", str(e).splitlines()[-1][:300])
try:
    c = run(comp, 8); print("steps=8 OK")
except Exception as e: print("steps=8 FAILED:", str(e).splitlines()[-1][:300])
