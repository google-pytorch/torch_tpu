"""#4408 guard via the inputs_embeds path: d(logit)/d(embeds) eager vs torch.compile(backend="tpu") vs CPU."""
import copy, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu"); MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
cpu_m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tpu_m = copy.deepcopy(cpu_m).to(tpu).eval(); comp_m = torch.compile(tpu_m, backend="tpu")
ids = tok("The quick brown fox jumps over the lazy dog because it", return_tensors="pt").input_ids
target = tok(" was").input_ids[0]
print("attn impl:", getattr(tpu_m.config, "_attn_implementation", "?"))
def grad(model, call, dev):
    emb = model.get_input_embeddings()(ids.to(dev)).detach().requires_grad_(True)
    out = call(inputs_embeds=emb, logits_to_keep=1).logits[0, -1, target]
    return torch.autograd.grad(out, emb)[0].detach().cpu()
rel = lambda a, b: ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
c = grad(cpu_m, cpu_m, "cpu"); e = grad(tpu_m, tpu_m, tpu); k = grad(tpu_m, comp_m, tpu)
print(f"nan: cpu={c.isnan().any().item()} eager={e.isnan().any().item()} compiled={k.isnan().any().item()}")
print(f"rel err eager vs cpu     : {rel(e, c):.2e}")
print(f"rel err compiled vs cpu  : {rel(k, c):.2e}")
print(f"rel err compiled vs eager: {rel(k, e):.2e}")
