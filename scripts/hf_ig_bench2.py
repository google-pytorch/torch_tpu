"""IG scaling on Qwen3-0.6B with logits_to_keep=1 (only last-token logits exist -> no [steps,T,V] OOM).
  PREC=highest ~/env/bin/python hf_ig_bench2.py
  PREC=high    ~/env/bin/python hf_ig_bench2.py
  TORCH_TPU_DEFER_AND_FUSE=1 PREC=high ~/env/bin/python hf_ig_bench2.py"""
import copy, gc, os, statistics, time
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients

PREC = os.environ.get("PREC", "highest"); torch.set_float32_matmul_precision(PREC)
tpu = torch.device("tpu"); MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
cpu_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tpu_m = copy.deepcopy(cpu_m).to(tpu).eval()
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
print(f"prec={PREC} defer_and_fuse={os.environ.get('TORCH_TPU_DEFER_AND_FUSE','0')}")

prompt = ("Integrated Gradients attributes a model's prediction to its input features by integrating gradients along "
          "a straight path from a baseline to the input. On a TPU, the question is whether batching every interpolation "
          "step into one fixed-shape program beats a Python loop. The answer is")
ids = tok(prompt, return_tensors="pt").input_ids; target = tok(" yes").input_ids[0]
print(f"prompt tokens={ids.shape[1]}")


def captum_ig(model, dev, steps, compiled=None):
    m = compiled or model
    fwd = lambda x: m(input_ids=x, logits_to_keep=1).logits[:, -1, target]
    lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
    return lig.attribute(ids.to(dev), baselines=torch.full_like(ids, pad).to(dev), n_steps=steps).sum(-1).cpu()


def timeit(fn, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts[0], statistics.median(ts[1:]) if runs > 1 else ts[0]


def mem():
    try: return f"{torch.accelerator.max_memory_allocated() / 2**30:.1f}G"
    except Exception: return "n/a"


comp_m = torch.compile(tpu_m, backend="tpu")
cols = [("captum cpu", lambda s: captum_ig(cpu_m, "cpu", s)),
        ("captum tpu eager", lambda s: captum_ig(tpu_m, tpu, s)),
        ("captum tpu compile", lambda s: captum_ig(tpu_m, tpu, s, compiled=comp_m))]
print("seconds: first(incl. compile)/warm-median   [peak HBM]")
print(f"{'steps':>5} " + " ".join(f"{n:>26}" for n, _ in cols))
for s in (32, 128, 256, 512):
    row = []
    for n, fn in cols:
        if "cpu" in n and s > 128: row.append(f"{'skipped':>26}"); continue
        try:
            first, warm = timeit(lambda: fn(s), runs=2 if "cpu" in n else 3)
            row.append(f"{first:8.2f}/{warm:8.3f} [{mem():>5}]")
        except Exception as e:  # noqa: BLE001
            row.append(f"{('FAIL ' + type(e).__name__)[:26]:>26}")
        gc.collect()
    print(f"{s:>5} " + " ".join(row), flush=True)
