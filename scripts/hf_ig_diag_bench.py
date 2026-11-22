"""Qwen3-0.6B IG on TPU: (diag) why torchtpu_xai IG differs from Captum; (bench) scaling.
Run on the TPU VM:
  ~/env/bin/python hf_ig_diag_bench.py diag
  PREC=highest ~/env/bin/python hf_ig_diag_bench.py bench
  PREC=high    ~/env/bin/python hf_ig_diag_bench.py bench
  TORCH_TPU_DEFER_AND_FUSE=1 PREC=high ~/env/bin/python hf_ig_diag_bench.py bench
"""
import copy, os, statistics, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
from torchtpu_xai import integrated_gradients

MODE = sys.argv[1] if len(sys.argv) > 1 else "diag"
PREC = os.environ.get("PREC", "highest")
torch.set_float32_matmul_precision(PREC)
tpu = torch.device("tpu")
MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
cpu_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tpu_m = copy.deepcopy(cpu_m).to(tpu).eval()
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
print(f"mode={MODE} prec={PREC} defer_and_fuse={os.environ.get('TORCH_TPU_DEFER_AND_FUSE', '0')}")


def captum_ig(model, dev, ids, target, steps, method="gausslegendre"):
    fwd = lambda x: model(input_ids=x).logits[:, -1, target]
    lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
    a, delta = lig.attribute(ids.to(dev), baselines=torch.full_like(ids, pad).to(dev), n_steps=steps,
                             internal_batch_size=steps, method=method, return_convergence_delta=True)
    return a.sum(-1)[0].detach().cpu(), abs(delta.item())


def xai_ig(model, dev, ids, target, steps):
    a = integrated_gradients(model, ids[0].to(dev), target, steps=steps, baseline_id=pad)
    return a.scores.cpu(), a.convergence_delta


def rel(a, b): return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def timeit(fn, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return ts[0], statistics.median(ts[1:]) if runs > 1 else ts[0]


if MODE == "diag":
    ids = tok("The capital of France is", return_tensors="pt").input_ids
    target = tok(" Paris").input_ids[0]
    ref, d_ref = captum_ig(cpu_m, "cpu", ids, target, 256)          # high-step reference
    print(f"reference: captum cpu gausslegendre 256 steps, delta={d_ref:.3e}, sum={ref.sum():.4f}")
    for steps in (32, 128):
        for name, fn in (("captum gausslegendre", lambda: captum_ig(tpu_m, tpu, ids, target, steps)),
                         ("captum riemann_right", lambda: captum_ig(tpu_m, tpu, ids, target, steps, "riemann_right")),
                         ("xai riemann_right  ", lambda: xai_ig(tpu_m, tpu, ids, target, steps))):
            a, d = fn()
            print(f"steps={steps:3} {name} tpu: rel_err_vs_ref={rel(a, ref):.2e} conv_delta={d:.3e} sum={a.sum():.4f}")
    # same rule, same steps, cpu vs tpu for *my* implementation
    c, _ = xai_ig(cpu_m, "cpu", ids, target, 32); t, _ = xai_ig(tpu_m, tpu, ids, target, 32)
    print(f"xai cpu vs xai tpu (32 steps): rel_err={rel(t, c):.2e}")

else:
    prompt = ("Integrated Gradients attributes a model's prediction to its input features by integrating "
              "gradients along a straight path from a baseline to the input. On a TPU, the question is whether "
              "batching every interpolation step into one fixed-shape program beats a Python loop. The answer is")
    ids = tok(prompt, return_tensors="pt").input_ids
    target = tok(" yes").input_ids[0]
    print(f"prompt tokens={ids.shape[1]}")
    comp_m = torch.compile(tpu_m, backend="tpu")
    cols = [("captum cpu", lambda s: captum_ig(cpu_m, "cpu", ids, target, s)),
            ("captum tpu", lambda s: captum_ig(tpu_m, tpu, ids, target, s)),
            ("xai tpu eager", lambda s: xai_ig(tpu_m, tpu, ids, target, s)),
            ("xai tpu compile", lambda s: xai_ig(comp_m, tpu, ids, target, s))]
    print("seconds: first(incl. compile)/warm-median")
    print(f"{'steps':>5} " + " ".join(f"{n:>18}" for n, _ in cols))
    for s in (32, 128, 256):
        row = []
        for n, fn in cols:
            if n == "captum cpu" and s > 128:
                row.append(f"{'skipped':>18}"); continue
            try:
                first, warm = timeit(lambda: fn(s), runs=2 if "cpu" in n else 3)
                row.append(f"{first:8.2f}/{warm:8.3f}")
            except Exception as e:  # noqa: BLE001
                row.append(f"{('FAIL ' + type(e).__name__)[:18]:>18}")
        print(f"{s:>5} " + " ".join(row), flush=True)
