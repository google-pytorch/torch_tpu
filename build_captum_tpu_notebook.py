"""Generate captum_on_torchtpu.ipynb (the sprint tutorial). Run: python3 build_captum_tpu_notebook.py
Result cells are left as TODO slots to be filled from FINDINGS.md / results_*.txt."""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
md("""
# Captum on TorchTPU: explain a PyTorch model on a Cloud TPU
Captum is PyTorch's explainability library. TorchTPU makes a TPU a normal PyTorch device. This notebook shows
that Captum runs unmodified on `torch.device("tpu")`, how to make its numbers match CPU, and what it costs.

Tested on a Cloud TPU v6e-1 (torch 2.15, torch_tpu 0.2.0, Captum 0.9.0). On Colab, choose a TPU runtime
(TorchTPU on free Colab is not yet confirmed; a TPU VM is the reference setup).
"""),
code("!pip install -q torch_tpu captum \"transformers>=5\""),
md("## 1. TPU is a PyTorch device"),
code("""
import torch
device = torch.device("tpu")
x = torch.randn(4, 4, device=device)
print(x.device, (x @ x).cpu().sum())
"""),
md("""
## 2. The one setting attribution needs
TPUs run fp32 matmuls at reduced precision by default. Attributions then differ from CPU by ~1e-3
(LayerGradCam up to 1e-2). Set the precision to "highest" and every deterministic Captum method matches CPU to ~1e-7.
"""),
code('torch.set_float32_matmul_precision("highest")'),
md("## 3. Captum on a small model: CPU vs TPU"),
code("""
import copy, torch.nn as nn
from captum.attr import IntegratedGradients, Saliency, LayerGradCam

torch.manual_seed(0)
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).eval()
img = torch.randn(2, 3, 16, 16, requires_grad=True)
tpu_cnn = copy.deepcopy(cnn).to(device)

def rel(a, b): return ((a.cpu() - b.cpu()).abs().max() / b.cpu().abs().max()).item()
for name, make in [("IntegratedGradients", lambda m: IntegratedGradients(m)),
                   ("Saliency", lambda m: Saliency(m)),
                   ("LayerGradCam", lambda m: LayerGradCam(m, m[2]))]:
    c = make(cnn).attribute(img, target=1)
    t = make(tpu_cnn).attribute(img.to(device), target=1)
    print(f"{name:20} max rel err CPU vs TPU: {rel(t, c):.1e}")
"""),
md("## 4. A real model, fully on TPU: Qwen3-0.6B with `device_map=\"tpu\"`"),
code("""
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="tpu").eval()

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
target = tok(" Paris").input_ids[0]
pad = tok.pad_token_id or 0

fwd = lambda x: model(input_ids=x).logits[:, -1, target]
lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
attr, delta = lig.attribute(ids, baselines=torch.full_like(ids, pad), n_steps=64, return_convergence_delta=True)
scores = attr.sum(-1)[0].cpu()
tokens = [tok.decode(t) for t in ids[0].cpu()]
for t, s in zip(tokens, scores.tolist()): print(f"{s:+.4f}  {t!r}")
print("convergence delta:", abs(delta.item()))
"""),
md("### Heatmap"),
code("""
import html
from IPython.display import HTML
def heatmap(tokens, scores):
    m = max(abs(s) for s in scores) or 1.0
    spans = [f'<span title="{s:+.4f}" style="background:rgba({"220,50,47" if s>=0 else "38,139,210"},{abs(s)/m:.2f});padding:2px;margin:1px;border-radius:3px">{html.escape(t)}</span>'
             for t, s in zip(tokens, scores)]
    return HTML("<div style='font-family:monospace;line-height:2'>" + "".join(spans) + "</div>")
heatmap(tokens, scores.tolist())
"""),
md("""
## 5. How fast? (TODO: table from results_ig.txt)
Integrated Gradients cost scales with `n_steps`. Captum runs all steps in one batched forward/backward when
`internal_batch_size` is None, which is exactly the shape a TPU likes. Compare CPU vs TPU eager vs
`torch.compile(backend="tpu")` vs `TORCH_TPU_DEFER_AND_FUSE=1`, at matmul precision "highest" and "high".
"""),
code("""
import time
def bench(steps, runs=3):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        lig.attribute(ids, baselines=torch.full_like(ids, pad), n_steps=steps).cpu()
        ts.append(time.perf_counter() - t0)
    return ts[0], sorted(ts[1:])[len(ts[1:])//2]
for s in (32, 128, 256):
    first, warm = bench(s); print(f"steps={s:3}  first(incl. compile)={first:.2f}s  warm={warm:.3f}s")
"""),
md("""
## 6. Compatibility matrix and tests (TODO: fill from results_matrix.txt / results_accel_tests.txt)
- Every `captum.attr` method on TorchTPU: OK / RUNS (stochastic) / DIFF / FAIL.
- Captum's own 78 attribution configs, CPU vs TPU, via the new device-generic `tests/attr/test_accelerator.py`
  (`CAPTUM_TEST_DEVICE=tpu pytest tests/attr/test_accelerator.py`).

## Takeaways
1. Captum runs unmodified on TorchTPU.
2. Set `torch.set_float32_matmul_precision("highest")` for CPU-level numerics.
3. TODO: speed and what to avoid.
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "TPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("captum_on_torchtpu.ipynb", "w"), indent=1)
print("wrote captum_on_torchtpu.ipynb")
