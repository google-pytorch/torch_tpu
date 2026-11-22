"""Guard vs torch_tpu #4408 (compiled SDPA backward -> NaN grads): Captum LIG on Qwen3-0.6B, eager vs
torch.compile(backend="tpu") vs CPU, with logits_to_keep=1. Run on the TPU VM."""
import copy, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu"); MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
cpu_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tpu_m = copy.deepcopy(cpu_m).to(tpu).eval(); comp_m = torch.compile(tpu_m, backend="tpu")
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
ids = tok("The quick brown fox jumps over the lazy dog because it", return_tensors="pt").input_ids
target = tok(" was").input_ids[0]
print("attn impl:", getattr(tpu_m.config, "_attn_implementation", "?"))

def lig(model, call, dev, steps=32):
    fwd = lambda x: call(input_ids=x, logits_to_keep=1).logits[:, -1, target]
    a, d = LayerIntegratedGradients(fwd, model.get_input_embeddings()).attribute(
        ids.to(dev), baselines=torch.full_like(ids, pad).to(dev), n_steps=steps, return_convergence_delta=True)
    return a.sum(-1)[0].detach().cpu(), abs(d.item())

c, dc = lig(cpu_m, cpu_m, "cpu"); e, de = lig(tpu_m, tpu_m, tpu); k, dk = lig(tpu_m, comp_m, tpu)
rel = lambda a, b: ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
print(f"nan: cpu={c.isnan().any().item()} eager={e.isnan().any().item()} compiled={k.isnan().any().item()}")
print(f"rel err eager vs cpu    : {rel(e, c):.2e}  (deltas cpu={dc:.3f} eager={de:.3f})")
print(f"rel err compiled vs cpu : {rel(k, c):.2e}  (delta compiled={dk:.3f})")
print(f"rel err compiled vs eager: {rel(k, e):.2e}")
# plain gradient check too: d(logit)/d(embeds) eager vs compiled
emb = tpu_m.get_input_embeddings()(ids.to(tpu)).detach().requires_grad_(True)
g_e = torch.autograd.grad(tpu_m(inputs_embeds=emb, logits_to_keep=1).logits[0, -1, target], emb)[0]
emb2 = emb.detach().clone().requires_grad_(True)
g_k = torch.autograd.grad(comp_m(inputs_embeds=emb2, logits_to_keep=1).logits[0, -1, target], emb2)[0]
print(f"raw grad: nan eager={g_e.isnan().any().item()} compiled={g_k.isnan().any().item()} rel err={rel(g_k.cpu(), g_e.cpu()):.2e}")
