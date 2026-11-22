"""torch_tpu #6036: eager device buffers for distinct Python scalars are never evicted. Reproduce on the 0.2.0
wheel, then check whether Captum IG (which scales inputs by per-step Python floats) triggers it."""
import gc, os, torch, torch.nn as nn
from captum.attr import IntegratedGradients
tpu = torch.device("tpu")

# Under torchrun (one worker per chip) join the tpu_dist group so each rank owns its own chip,
# and tag every line with the rank.
if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    import builtins, torch.distributed as dist
    dist.init_process_group(backend="tpu_dist")
    _print = builtins.print
    print = lambda *a, **k: _print(f"[rank {dist.get_rank()}]", *a, **k, flush=True)

def live():
    torch.accelerator.synchronize(); gc.collect()
    return torch.accelerator.memory_stats()["allocation.all.current"]

x = torch.zeros(8, 128, device=tpu)
b = live()
for i in range(5000): x.add_(0.5)
print(f"same scalar x5000     : +{live()-b} live allocations")
b = live()
for i in range(5000): x.add_(i / 1024)
print(f"new scalar each x5000 : +{live()-b} live allocations   (issue #6036 expects ~+5000)")
torch.accelerator.empty_cache(); print(f"after empty_cache     : {live()-b:+d} vs before loop")

mlp = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 4)).to(tpu).eval()
inp = torch.randn(2, 16, device=tpu)
ig = IntegratedGradients(mlp)
ig.attribute(inp, target=1, n_steps=50); b = live()
for _ in range(20): ig.attribute(inp, target=1, n_steps=50)
print(f"captum IG same n_steps x20   : +{live()-b} live allocations")
b = live()
for s in range(10, 110):
    ig.attribute(inp, target=1, n_steps=s)
print(f"captum IG n_steps 10..109    : +{live()-b} live allocations (distinct step sizes -> distinct scalars?)")
