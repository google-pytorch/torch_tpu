"""Other PyTorch explainability libraries on TorchTPU (none mention TPU upstream). PASS = ran without exception.
Run on the TPU VM in the secondary env: ~/env2/bin/python other_xai_libs_tpu.py"""
import traceback, torch, torch.nn as nn
torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu"); res = []

def check(name):
    def d(fn):
        try: fn(); res.append((name, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            res.append((name, "FAIL", f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")); traceback.print_exc()
    return d

torch.manual_seed(0)
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).to(tpu).eval()
img = torch.randn(2, 3, 16, 16, device=tpu)
mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(tpu).eval()
bg = torch.randn(10, 16, device=tpu); x = torch.randn(3, 16, device=tpu)

@check("pytorch-grad-cam GradCAM")
def _():
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    print(GradCAM(model=cnn, target_layers=[cnn[2]])(input_tensor=img, targets=[ClassifierOutputTarget(1)] * 2).shape)

@check("pytorch-grad-cam ScoreCAM")
def _():
    from pytorch_grad_cam import ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    print(ScoreCAM(model=cnn, target_layers=[cnn[2]])(input_tensor=img, targets=[ClassifierOutputTarget(1)] * 2).shape)

@check("shap DeepExplainer")
def _():
    import shap; print(type(shap.DeepExplainer(mlp, bg).shap_values(x)))

@check("shap GradientExplainer")
def _():
    import shap; print(type(shap.GradientExplainer(mlp, bg).shap_values(x)))

@check("inseq saliency (tiny-gpt2, device=tpu)")
def _():
    import inseq
    m = inseq.load_model("sshleifer/tiny-gpt2", "saliency", device="tpu")
    print(m.attribute("Hello world", generation_args={"max_new_tokens": 3}).sequence_attributions[0].target_attributions.shape)

@check("transformer_lens run_with_cache (gpt2, device=tpu)")
def _():
    from transformer_lens import HookedTransformer
    m = HookedTransformer.from_pretrained("gpt2", device="tpu")
    logits, cache = m.run_with_cache("Hello world"); print(logits.shape, len(cache))

@check("nnsight trace (tiny-gpt2, device_map=tpu)")
def _():
    from nnsight import LanguageModel
    m = LanguageModel("sshleifer/tiny-gpt2", device_map="tpu")
    with m.trace("Hello world"):
        h = m.transformer.h[0].output[0].save()
    print(h.shape)

print("\n=== SUMMARY ===")
for n, s, e in res: print(f"{s:5} {n} {e}")
