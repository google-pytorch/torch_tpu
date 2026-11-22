"""Find which graph input is a CPU tensor when Captum LIG wraps a torch.compile(backend="tpu") Qwen3 model."""
import copy, traceback, torch
import torch._dynamo as dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu"); MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(tpu).eval()
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(tpu)
target = tok(" Paris").input_ids[0]
tpu_backend = dynamo.lookup_backend("tpu")
def diag_backend(gm, example_inputs):
    print(f"--- graph with {len(example_inputs)} inputs")
    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for n, ex in zip(phs, example_inputs):
        if isinstance(ex, torch.Tensor) and ex.device.type != "tpu":
            print(f"CPU INPUT: {n.name} shape={tuple(ex.shape)} dtype={ex.dtype} value={ex.tolist() if ex.numel() < 8 else '...'}")
            print("  users:", [u.target for u in n.users][:5])
            src = n.meta.get("grapharg"); print("  grapharg:", getattr(src, "source", None))
    return tpu_backend(gm, example_inputs)
comp = torch.compile(model, backend=diag_backend)
def fwd(input_ids):
    return comp(input_ids=input_ids, logits_to_keep=1).logits[:, -1, target]
try:
    print("direct compiled call ok:", fwd(ids).shape)
except Exception as e: print("direct call failed:", str(e)[:200])
lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
try:
    attr = lig.attribute(ids, baselines=torch.full_like(ids, tok.pad_token_id), n_steps=4)
    print("LIG ok:", attr.shape)
except Exception as e:
    print("LIG failed:", str(e).splitlines()[-1][:300])
