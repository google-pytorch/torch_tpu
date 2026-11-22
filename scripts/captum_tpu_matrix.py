"""Compatibility matrix: every captum.attr method, CPU vs TorchTPU. Run on the TPU VM.
Columns: status (OK = runs and matches CPU; RUNS = stochastic, only checked for running;
DIFF = runs but numerics off; FAIL = exception), max rel err, cpu s, tpu s (warm)."""
import copy, time, traceback, sys
import torch, torch.nn as nn
import captum.attr as A

torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu")
torch.manual_seed(0)
VERBOSE = "-v" in sys.argv

mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 5))
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4))
x_mlp = torch.randn(2, 16); x_cnn = torch.randn(2, 3, 8, 8)
mask16 = torch.arange(16).view(1, 16) // 4          # 4 feature groups (keeps Shapley/Lime cheap)
STOCHASTIC = {"GradientShap", "LayerGradientShap", "NeuronGradientShap", "DeepLiftShap", "LayerDeepLiftShap",
              "NeuronDeepLiftShap", "FeaturePermutation", "ShapleyValueSampling", "Lime", "KernelShap", "NoiseTunnel"}

# name -> (model, input, constructor(model), attribute kwargs)
L = lambda m: m[0]          # first layer (Linear / Conv)
C = lambda m: m[2]          # conv layer for GradCam
cases = {
 "IntegratedGradients":      (mlp, x_mlp, lambda m: A.IntegratedGradients(m), dict(target=1, n_steps=16)),
 "Saliency":                 (mlp, x_mlp, lambda m: A.Saliency(m), dict(target=1)),
 "InputXGradient":           (mlp, x_mlp, lambda m: A.InputXGradient(m), dict(target=1)),
 "DeepLift":                 (mlp, x_mlp, lambda m: A.DeepLift(m), dict(target=1)),
 "DeepLiftShap":             (mlp, x_mlp, lambda m: A.DeepLiftShap(m), dict(target=1, baselines=torch.randn(6, 16))),
 "GradientShap":             (mlp, x_mlp, lambda m: A.GradientShap(m), dict(target=1, baselines=torch.zeros(6, 16))),
 "GuidedBackprop":           (mlp, x_mlp, lambda m: A.GuidedBackprop(m), dict(target=1)),
 "Deconvolution":            (mlp, x_mlp, lambda m: A.Deconvolution(m), dict(target=1)),
 "GuidedGradCam":            (cnn, x_cnn, lambda m: A.GuidedGradCam(m, C(m)), dict(target=1)),
 "FeatureAblation":          (mlp, x_mlp, lambda m: A.FeatureAblation(m), dict(target=1)),
 "FeaturePermutation":       (mlp, x_mlp, lambda m: A.FeaturePermutation(m), dict(target=1)),
 "Occlusion":                (mlp, x_mlp, lambda m: A.Occlusion(m), dict(target=1, sliding_window_shapes=(4,))),
 "ShapleyValueSampling":     (mlp, x_mlp, lambda m: A.ShapleyValueSampling(m), dict(target=1, feature_mask=mask16, n_samples=5)),
 "ShapleyValues":            (mlp, x_mlp, lambda m: A.ShapleyValues(m), dict(target=1, feature_mask=mask16)),
 "Lime":                     (mlp, x_mlp, lambda m: A.Lime(m), dict(target=1, feature_mask=mask16, n_samples=20)),
 "KernelShap":               (mlp, x_mlp, lambda m: A.KernelShap(m), dict(target=1, feature_mask=mask16, n_samples=20)),
 "LRP":                      (mlp, x_mlp, lambda m: A.LRP(m), dict(target=1)),
 "NoiseTunnel":              (mlp, x_mlp, lambda m: A.NoiseTunnel(A.Saliency(m)), dict(target=1, nt_samples=4)),
 "LayerConductance":         (mlp, x_mlp, lambda m: A.LayerConductance(m, L(m)), dict(target=1, n_steps=8)),
 "LayerActivation":          (mlp, x_mlp, lambda m: A.LayerActivation(m, L(m)), dict()),
 "LayerGradientXActivation": (mlp, x_mlp, lambda m: A.LayerGradientXActivation(m, L(m)), dict(target=1)),
 "LayerGradCam":             (cnn, x_cnn, lambda m: A.LayerGradCam(m, C(m)), dict(target=1)),
 "InternalInfluence":        (mlp, x_mlp, lambda m: A.InternalInfluence(m, L(m)), dict(target=1, n_steps=8)),
 "LayerDeepLift":            (mlp, x_mlp, lambda m: A.LayerDeepLift(m, L(m)), dict(target=1)),
 "LayerDeepLiftShap":        (mlp, x_mlp, lambda m: A.LayerDeepLiftShap(m, L(m)), dict(target=1, baselines=torch.randn(6, 16))),
 "LayerGradientShap":        (mlp, x_mlp, lambda m: A.LayerGradientShap(m, L(m)), dict(target=1, baselines=torch.zeros(6, 16))),
 "LayerIntegratedGradients": (mlp, x_mlp, lambda m: A.LayerIntegratedGradients(m, L(m)), dict(target=1, n_steps=8)),
 "LayerFeatureAblation":     (mlp, x_mlp, lambda m: A.LayerFeatureAblation(m, L(m)), dict(target=1)),
 "LayerLRP":                 (mlp, x_mlp, lambda m: A.LayerLRP(m, L(m)), dict(target=1)),
 "NeuronConductance":        (mlp, x_mlp, lambda m: A.NeuronConductance(m, L(m)), dict(target=1, neuron_selector=0, n_steps=8)),
 "NeuronGradient":           (mlp, x_mlp, lambda m: A.NeuronGradient(m, L(m)), dict(neuron_selector=0)),
 "NeuronIntegratedGradients":(mlp, x_mlp, lambda m: A.NeuronIntegratedGradients(m, L(m)), dict(neuron_selector=0, n_steps=8)),
 "NeuronDeepLift":           (mlp, x_mlp, lambda m: A.NeuronDeepLift(m, L(m)), dict(neuron_selector=0)),
 "NeuronDeepLiftShap":       (mlp, x_mlp, lambda m: A.NeuronDeepLiftShap(m, L(m)), dict(neuron_selector=0, baselines=torch.randn(6, 16))),
 "NeuronGradientShap":       (mlp, x_mlp, lambda m: A.NeuronGradientShap(m, L(m)), dict(neuron_selector=0, baselines=torch.zeros(6, 16))),
 "NeuronGuidedBackprop":     (mlp, x_mlp, lambda m: A.NeuronGuidedBackprop(m, L(m)), dict(neuron_selector=0)),
 "NeuronDeconvolution":      (mlp, x_mlp, lambda m: A.NeuronDeconvolution(m, L(m)), dict(neuron_selector=0)),
 "NeuronFeatureAblation":    (mlp, x_mlp, lambda m: A.NeuronFeatureAblation(m, L(m)), dict(neuron_selector=0)),
}


def to(dev, v): return v.to(dev) if torch.is_tensor(v) else v


def run_on(dev, model, x, make, kw):
    m = copy.deepcopy(model).to(dev).eval()
    xi = x.to(dev).requires_grad_(True)
    kw = {k: to(dev, v) for k, v in kw.items()}
    attr = make(m)
    out = attr.attribute(xi, **kw); torch.as_tensor(out).cpu()      # warm (compile)
    t0 = time.perf_counter(); out = attr.attribute(xi, **kw); out = out.detach().cpu(); dt = time.perf_counter() - t0
    return out, dt


print(f"{'method':27} {'status':6} {'max_rel_err':>11} {'cpu_s':>7} {'tpu_s':>7}  note")
summary = {"OK": 0, "RUNS": 0, "DIFF": 0, "FAIL": 0}
for name, (model, x, make, kw) in cases.items():
    if not hasattr(A, name.split("(")[0]):
        print(f"{name:27} SKIP   (not in this captum)"); continue
    try:
        c, tc = run_on("cpu", model, x, make, kw)
        t, tt = run_on(tpu, model, x, make, kw)
        err = ((t.float() - c.float()).abs().max() / (c.float().abs().max() + 1e-12)).item()
        if name in STOCHASTIC: st = "RUNS"
        else: st = "OK" if err < 1e-3 else "DIFF"
        print(f"{name:27} {st:6} {err:11.2e} {tc:7.3f} {tt:7.3f}")
    except Exception as e:  # noqa: BLE001
        st = "FAIL"; msg = f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
        print(f"{name:27} {st:6} {'-':>11} {'-':>7} {'-':>7}  {msg}")
        if VERBOSE: traceback.print_exc()
    summary[st] += 1
print("\nSUMMARY", summary)
