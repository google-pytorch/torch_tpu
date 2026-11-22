"""Captum on a real HF model (Qwen3-0.6B): CPU-vs-TPU agreement + IG timing.
Run on the TPU VM: ~/env/bin/python hf_captum_bench.py"""
import copy, statistics, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
from torchtpu_xai import integrated_gradients

torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu")
MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
cpu_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tpu_m = copy.deepcopy(cpu_m).to(tpu).eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids
target = tok(" Paris").input_ids[0]
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
base = torch.full_like(ids, pad)
print("tokens:", [tok.decode(t) for t in ids[0]], "target:", repr(tok.decode(target)))


def captum_ig(model, dev, steps):
    fwd = lambda x: model(input_ids=x).logits[:, -1, target]
    lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
    a = lig.attribute(ids.to(dev), baselines=base.to(dev), n_steps=steps, internal_batch_size=steps)
    return a.sum(-1)[0].detach().cpu()  # per-token


def xai_ig(model, dev, steps):
    return integrated_gradients(model, ids[0].to(dev), target, steps=steps, baseline_id=pad).scores.cpu()


def timeit(fn, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, ts[0], statistics.median(ts[1:])


def rel(a, b): return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


print("\n== correctness (steps=32)")
c = captum_ig(cpu_m, "cpu", 32)
t = captum_ig(tpu_m, tpu, 32)
x = xai_ig(tpu_m, tpu, 32)
print(f"captum cpu vs captum tpu : max_rel_err={rel(t, c):.2e}")
print(f"captum cpu vs xai tpu    : max_rel_err={rel(x, c):.2e}")
print("top tokens (captum cpu):", [tok.decode(ids[0][i]) for i in c.abs().topk(3).indices])
print("top tokens (captum tpu):", [tok.decode(ids[0][i]) for i in t.abs().topk(3).indices])

print("\n== timing seconds: first run (incl. compile) / warm median")
print(f"{'steps':>5} {'captum cpu':>16} {'captum tpu':>16} {'xai tpu':>16}")
for s in (8, 32, 64):
    r = []
    for fn, m, d in ((captum_ig, cpu_m, "cpu"), (captum_ig, tpu_m, tpu), (xai_ig, tpu_m, tpu)):
        try:
            _, first, warm = timeit(lambda: fn(m, d, s)); r.append(f"{first:6.2f}/{warm:6.3f}")
        except Exception as e:  # noqa: BLE001
            r.append(f"FAIL {type(e).__name__}")
    print(f"{s:>5} {r[0]:>16} {r[1]:>16} {r[2]:>16}")
