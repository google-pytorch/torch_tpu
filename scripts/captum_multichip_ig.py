"""Integrated Gradients sharded across TPU chips with torch.distributed (tpu_dist).

One worker per chip (torchrun). Each rank evaluates a disjoint slice of the Riemann
grid for LayerIntegratedGradients on Qwen3-0.6B, then all_reduce(SUM) assembles the
full attribution. Rank 0 also computes the same attribution with Captum's own
LayerIntegratedGradients on one chip, so the sharded result is checked for
agreement and the two wall-clock times are compared.

Launch on the VM:
  eval $(python -m torch.tpu.distributed | sed 's/^/export /')
  torchrun --nproc_per_node=$WORLD_SIZE captum_multichip_ig.py [--steps 64] [--compile]
"""
import argparse, os, time
import torch, torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=64)
p.add_argument("--model", default="Qwen/Qwen3-0.6B")
p.add_argument("--compile", action="store_true")
p.add_argument("--method", default="riemann_trapezoid")
p.add_argument("--batch", type=int, default=0, help="grid points per forward on each rank (0 = all of this rank's)")
p.add_argument("--ref-batch", type=int, default=0, help="internal_batch_size for the single-chip Captum reference")
args = p.parse_args()

torch.set_float32_matmul_precision("highest")
dist.init_process_group(backend="tpu_dist")
rank, world = dist.get_rank(), dist.get_world_size()
tpu = torch.device("tpu")
log = lambda *a: print(f"[rank {rank}/{world}]", *a, flush=True)
log("device", torch.tpu.get_device_properties().name, "visible", torch.accelerator.device_count())

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, device_map="tpu").eval()
call = torch.compile(model, backend="tpu") if args.compile else model
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
text = ("Integrated Gradients attributes a prediction to its inputs by integrating gradients along a path from a "
        "baseline to the input. On a TPU the question is whether one fixed-shape program beats a loop. The answer is")
ids = tok(text, return_tensors="pt").input_ids.to(tpu)
target = tok(" yes").input_ids[0]
emb = model.get_input_embeddings()
fwd_ids = lambda x: call(input_ids=x, logits_to_keep=1).logits[:, -1, target]
fwd_emb = lambda e: call(inputs_embeds=e, logits_to_keep=1).logits[:, -1, target]

with torch.no_grad():
    x_emb = emb(ids); b_emb = emb(torch.full_like(ids, pad))

# --- sharded IG: this rank's slice of the grid -------------------------------------------------
step_sizes_fn, alphas_fn = approximation_parameters(args.method)
alphas = alphas_fn(args.steps); steps = step_sizes_fn(args.steps)
mine = list(range(rank, args.steps, world))
log(f"grid points {len(mine)} of {args.steps}")

a_mine = torch.tensor([alphas[i] for i in mine], device=tpu).view(-1, 1, 1)
w_mine = torch.tensor([steps[i] for i in mine], device=tpu).view(-1, 1, 1)
B = args.batch or len(mine)   # grid points per forward pass on this rank (Captum: all n_steps at once)

def sharded_ig():
    acc = torch.zeros_like(x_emb)
    for s in range(0, len(mine), B):
        a, w = a_mine[s:s + B], w_mine[s:s + B]
        e = (b_emb + a * (x_emb - b_emb)).detach().requires_grad_(True)      # [B, T, D], like Captum's batch
        (g,) = torch.autograd.grad(fwd_emb(e).sum(), e)
        acc = acc + (w * g).sum(0, keepdim=True)
    dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    return ((x_emb - b_emb) * acc).sum(-1).cpu()   # per-token attribution

t0 = time.perf_counter(); attr_s = sharded_ig(); log(f"sharded warmup {time.perf_counter()-t0:.1f}s")
ts = []
for _ in range(3):
    dist.barrier(); t0 = time.perf_counter(); attr_s = sharded_ig(); ts.append(time.perf_counter() - t0)
log(f"sharded IG ({world} chips, {args.steps} steps) warm: {[f'{t:.3f}' for t in ts]}")

# --- single-chip reference on rank 0 -----------------------------------------------------------
if rank == 0:
    lig = LayerIntegratedGradients(fwd_ids, emb)
    ref_run = lambda: lig.attribute(ids, baselines=torch.full_like(ids, pad), n_steps=args.steps, method=args.method,
                                    internal_batch_size=args.ref_batch or None, return_convergence_delta=True)
    log(f"captum reference: 1 chip, internal_batch_size={args.ref_batch or 'all'}")
    t0 = time.perf_counter(); ref, delta = ref_run(); log(f"captum warmup {time.perf_counter()-t0:.1f}s")
    tr = []
    for _ in range(3):
        t0 = time.perf_counter(); ref, delta = ref_run(); tr.append(time.perf_counter() - t0)
    ref = ref.sum(-1).cpu()
    rel = ((attr_s - ref).norm() / ref.norm()).item()
    log(f"captum single-chip LIG warm: {[f'{t:.3f}' for t in tr]}  convergence delta {delta.item():.3e}")
    log(f"sharded vs captum: max abs {(attr_s - ref).abs().max().item():.3e}  rel {rel:.2e}")
    log(f"speedup (median warm): {sorted(tr)[1] / sorted(ts)[1]:.2f}x on {world} chips")
    top = attr_s[0].topk(5).indices.tolist()
    log("top tokens:", [tok.decode(ids[0, i]) for i in top])
dist.barrier(); dist.destroy_process_group()
