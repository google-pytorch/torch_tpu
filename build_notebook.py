"""Generate hf_on_torchtpu.ipynb. Run: python3 build_notebook.py"""
import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)}

cells = [
md("""
# Hugging Face on TorchTPU: run Qwen3 on a TPU with `device_map="tpu"`
Runtime: **Runtime > Change runtime type > TPU**. Covers: TPU as a normal PyTorch device, loading an HF model, and comparing eager modes with `torch.compile`.
"""),
code("!pip install -q torch_tpu \"transformers>=5\""),
md("## 1. TPU is just a PyTorch device"),
code("""
import torch
device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
print(x.device, torch.matmul(x, torch.ones((4, 4), device=device)).cpu().sum())
"""),
md("## 2. Load and run a Hugging Face model"),
code("""
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="tpu")
inputs = tok("Explain what a TPU is in two sentences.", return_tensors="pt").to("tpu")

def bench(m, runs=3):
    for i in range(runs):
        t0 = time.perf_counter()
        out = m.generate(**inputs, max_new_tokens=64, do_sample=False).cpu()
        dt = time.perf_counter() - t0
        n = out.shape[-1] - inputs["input_ids"].shape[-1]
        print(f"run {i}: {n / dt:.1f} tok/s ({dt:.2f}s)")
    return out

out = bench(model)
print(tok.decode(out[0], skip_special_tokens=True))
"""),
md("""
## 3. Execution modes
Strict eager is the default. The other eager modes are set with environment variables **before the process starts**, so run them via the script `hf_tpu_bench.py`:
```
TORCH_TPU_DEFER_AND_FUSE=1 python hf_tpu_bench.py
TORCH_TPU_LAUNCH_BLOCKING=1 python hf_tpu_bench.py   # debug only
```
"""),
code("!wget -q https://raw.githubusercontent.com/<your-user>/<your-repo>/main/hf_tpu_bench.py\n!python hf_tpu_bench.py\n!TORCH_TPU_DEFER_AND_FUSE=1 python hf_tpu_bench.py\n!python hf_tpu_bench.py --compile"),
md("## 4. Results\nPaste your tok/s table here after running (first run includes compilation; compare steady-state runs)."),
]
nb = {"cells": cells, "metadata": {"accelerator": "TPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("hf_on_torchtpu.ipynb", "w"), indent=1)
