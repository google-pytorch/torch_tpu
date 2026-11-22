"""Does TorchTPU support what attribution methods need? Run: ~/env/bin/python tpu_smoke.py
Each check prints PASS/FAIL with the error, so one failure doesn't hide the rest."""
import traceback
import torch
import torch.nn as nn

dev = torch.device("tpu")
results = []


def check(name):
    def deco(fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception as e:  # noqa: BLE001
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:160]}"))
            traceback.print_exc()
    return deco


def net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4)).to(dev)


@check("backward")
def _():
    n = net(); n(torch.randn(2, 8, device=dev)).sum().backward()
    assert n[0].weight.grad is not None


@check("autograd.grad wrt input")
def _():
    x = torch.randn(2, 8, device=dev, requires_grad=True)
    (g,) = torch.autograd.grad(net()(x).sum(), x); g.cpu()


@check("forward hook (capture activations)")
def _():
    n = net(); got = []
    n[0].register_forward_hook(lambda m, i, o: got.append(o))
    n(torch.randn(2, 8, device=dev)); assert got and got[0].shape == (2, 8)


@check("tensor hook / full backward hook")
def _():
    n = net(); got = []
    n[2].register_full_backward_hook(lambda m, gi, go: got.append(go[0]))
    n(torch.randn(2, 8, device=dev, requires_grad=True)).sum().backward()
    assert got


@check("double backward (create_graph)")
def _():
    x = torch.randn(2, 8, device=dev, requires_grad=True)
    (g,) = torch.autograd.grad(net()(x).pow(2).sum(), x, create_graph=True)
    g.sum().backward()


@check("custom autograd.Function")
def _():
    class Sq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x): ctx.save_for_backward(x); return x * x
        @staticmethod
        def backward(ctx, g): (x,) = ctx.saved_tensors; return 2 * x * g
    x = torch.randn(4, device=dev, requires_grad=True); Sq.apply(x).sum().backward()


@check("torch.func.grad")
def _():
    from torch.func import grad
    grad(lambda x: (x ** 2).sum())(torch.randn(4, device=dev)).cpu()


@check("compile + backward on view input (issue #4409)")
def _():
    n = net(); f = torch.compile(n, backend="tpu")
    base = torch.randn(4, 8, device=dev, requires_grad=True)
    f(base[:2]).sum().backward()


@check("compile + backward on cloned view (workaround)")
def _():
    n = net(); f = torch.compile(n, backend="tpu")
    base = torch.randn(4, 8, device=dev, requires_grad=True)
    f(base[:2].clone()).sum().backward()


@check("torchtpu_xai integrated_gradients (eager)")
def _():
    from tests.test_attribution import TinyLM
    from torchtpu_xai import integrated_gradients
    m = TinyLM().to(dev).eval()
    a = integrated_gradients(m, torch.tensor([3, 5, 7, 2], device=dev), target_id=4, steps=64)
    print("IG delta", a.convergence_delta); assert a.convergence_delta < 5e-2


@check("captum IntegratedGradients (eager)")
def _():
    from captum.attr import IntegratedGradients
    n = net().eval(); x = torch.randn(2, 8, device=dev, requires_grad=True)
    IntegratedGradients(n).attribute(x, target=1, n_steps=16).cpu()


@check("captum LayerGradCam-style layer attribution (hooks)")
def _():
    from captum.attr import LayerIntegratedGradients
    n = net().eval(); x = torch.randn(2, 8, device=dev)
    LayerIntegratedGradients(n, n[0]).attribute(x, target=1, n_steps=8).cpu()


print("\n=== SUMMARY ===")
for name, s, err in results:
    print(f"{s:5} {name} {err}")
