"""CPU vs TPU attribution agreement. Same weights, same inputs, fp32. Run on the TPU VM."""
import copy, os
import torch
import torch.nn as nn
from captum.attr import (IntegratedGradients, Saliency, InputXGradient,
                         LayerIntegratedGradients, LayerGradCam, GradientShap)

tpu = torch.device("tpu")
if os.environ.get("PREC"):
    torch.set_float32_matmul_precision(os.environ["PREC"]); print("matmul precision:", os.environ["PREC"])
torch.manual_seed(0)


def rel(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def run(name, make_attr, model, x, tol, **kw):
    try:
        cpu = make_attr(model.cpu().eval()).attribute(x.cpu(), **kw)
        m_t = copy.deepcopy(model).to(tpu).eval()
        kw_t = {k: (v.to(tpu) if torch.is_tensor(v) else v) for k, v in kw.items()}
        out = make_attr(m_t).attribute(x.to(tpu), **kw_t)
        e = rel(out, cpu)
        print(f"{'OK  ' if e < tol else 'DIFF'} {name:28} max_rel_err={e:.2e}")
    except Exception as ex:  # noqa: BLE001
        print(f"FAIL {name:28} {type(ex).__name__}: {str(ex)[:140]}")


mlp = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 5))
x = torch.randn(4, 16, requires_grad=True)
run("Saliency", Saliency, mlp, x, 1e-2, target=2)
run("InputXGradient", InputXGradient, mlp, x, 1e-2, target=2)
run("IntegratedGradients n=50", IntegratedGradients, mlp, x, 1e-2, target=2, n_steps=50)
run("LayerIntegratedGradients", lambda m: LayerIntegratedGradients(m, m[0]), mlp, x, 1e-2, target=2, n_steps=20)
run("GradientShap (seeded)", GradientShap, mlp, x, 1.0, baselines=torch.zeros(8, 16), target=2)

cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                    nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
img = torch.randn(2, 3, 16, 16, requires_grad=True)
run("LayerGradCam", lambda m: LayerGradCam(m, m[2]), cnn, img, 1e-2, target=1)

# our batched IG vs captum IG on CPU, evaluated on TPU
from torchtpu_xai import integrated_gradients
from tests.test_attribution import TinyLM
lm = TinyLM().eval(); ids = torch.tensor([3, 5, 7, 2])
c = integrated_gradients(lm, ids, 4, steps=128).scores
t = integrated_gradients(copy.deepcopy(lm).to(tpu), ids.to(tpu), 4, steps=128).scores
e = rel(t, c); print(f"{'OK  ' if e < 1e-2 else 'DIFF'} {'torchtpu_xai IG (cpu vs tpu)':28} max_rel_err={e:.2e}")
