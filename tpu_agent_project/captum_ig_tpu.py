# Copyright 2026 Joinal Ahmed. Apache-2.0.
"""Captum Integrated Gradients on a Cloud TPU, cross-checked against the CPU.

Dispatched onto the TPU VM by the TPU agent suite's compute layer
(`runtime.py compute run --code captum_ig_tpu.py --output results.json ...`).
It provisions nothing: torch_tpu and captum must already be in the target's
`python_interpreter`.

What it measures, for a small MLP and a small CNN with fixed seeds:
  * CPU-vs-TPU max relative error of Saliency, IntegratedGradients and
    LayerIntegratedGradients, at the default fp32 matmul precision and at
    `torch.set_float32_matmul_precision("highest")` (the one setting every
    Captum-on-TPU user needs; see ../FINDINGS.md section 2).
  * IG convergence delta (sum of attributions vs F(x) - F(baseline)).
  * Warm wall-clock of one IG call on TPU and on CPU.
  * Optionally (--profile) an XProf trace of one warm IG call, copied to
    `xprof.xplane.pb` beside the script so `--output` can carry it home,
    following the pattern of the suite's own hello_tpu_with_profile.py.

Prints a markdown table (copied verbatim into reports) and writes
`results.json`. Exits non-zero when no TPU runtime is present or an
attribution disagrees with the CPU beyond tolerance at "highest" precision.
"""
import argparse
import copy
import json
import os
import pathlib
import platform
import shutil
import sys
import time

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients, Saliency

TOL_HIGHEST = 1e-3   # "highest" precision should match CPU to ~1e-7 (FINDINGS §2)
N_STEPS = 50
TIMING_REPS = 5


def acquire_device():
    """Return (device, label). Only "native-tpu" means torch_tpu drove hardware."""
    if os.environ.get("MOCK_TPU") == "1":
        return torch.device("cpu"), "cpu-simulated"
    try:
        import torch_tpu  # noqa: F401  registers the "tpu" PrivateUse1 backend
        dev = torch.device("tpu")
        torch.zeros(1, device=dev)
        return dev, "native-tpu"
    except Exception as exc:  # noqa: BLE001
        print(f"[CAPTUM-TPU] no TPU runtime: {type(exc).__name__}: {exc}")
        return torch.device("cpu"), "cpu-fallback"


def rel_err(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def models_and_inputs():
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 5)).eval()
    x = torch.randn(4, 16)
    cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).eval()
    img = torch.randn(2, 3, 16, 16)
    return mlp, x, cnn, img


def cases(mlp, x, cnn, img):
    """(name, model, input, make_attr, kwargs) for each attribution we compare."""
    return [
        ("Saliency/MLP", mlp, x, Saliency, dict(target=2)),
        ("IntegratedGradients/MLP", mlp, x, IntegratedGradients,
         dict(target=2, n_steps=N_STEPS)),
        ("LayerIntegratedGradients/MLP[0]", mlp, x,
         lambda m: LayerIntegratedGradients(m, m[0]), dict(target=2, n_steps=20)),
        ("IntegratedGradients/CNN", cnn, img, IntegratedGradients,
         dict(target=1, n_steps=N_STEPS)),
    ]


def attribute(make_attr, model, inp, device, **kw):
    m = copy.deepcopy(model).to(device)
    return make_attr(m).attribute(inp.clone().to(device).requires_grad_(True), **kw)


def compare_all(device, precision):
    # "default" = leave the runtime's own setting alone (TPU: reduced-precision
    # fp32 matmuls); torch_tpu accepts only highest/high/medium as values.
    if precision != "default":
        torch.set_float32_matmul_precision(precision)
    mlp, x, cnn, img = models_and_inputs()
    rows = []
    for name, model, inp, make_attr, kw in cases(mlp, x, cnn, img):
        ref = attribute(make_attr, model, inp, torch.device("cpu"), **kw)
        out = attribute(make_attr, model, inp, device, **kw)
        rows.append({"method": name, "precision": precision, "max_rel_err": rel_err(out, ref)})
    return rows


def convergence(device):
    torch.set_float32_matmul_precision("highest")
    mlp, x, _, _ = models_and_inputs()
    m = copy.deepcopy(mlp).to(device)
    ig = IntegratedGradients(m)
    attr, delta = ig.attribute(x.to(device).requires_grad_(True), target=2,
                               n_steps=N_STEPS, return_convergence_delta=True)
    return {"attribution_sum": attr.sum().item(),
            "max_abs_convergence_delta": delta.abs().max().item()}


def time_ig(device):
    torch.set_float32_matmul_precision("highest")
    mlp, x, _, _ = models_and_inputs()
    m = copy.deepcopy(mlp).to(device)
    ig = IntegratedGradients(m)
    xi = x.to(device).requires_grad_(True)
    for _ in range(2):  # warm (first call on TPU compiles)
        ig.attribute(xi, target=2, n_steps=N_STEPS).cpu()
    ts = []
    for _ in range(TIMING_REPS):
        t0 = time.perf_counter()
        ig.attribute(xi, target=2, n_steps=N_STEPS).cpu()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def profile_ig(device, trace_dir, trace_file):
    """One warm IG call under torch_tpu's XProf profiler; returns a status string."""
    try:
        from torch_tpu._internal import profiler
    except ImportError as exc:
        return f"unavailable ({exc})"
    torch.set_float32_matmul_precision("highest")
    mlp, x, _, _ = models_and_inputs()
    m = copy.deepcopy(mlp).to(device)
    ig = IntegratedGradients(m)
    xi = x.to(device).requires_grad_(True)
    for _ in range(3):
        ig.attribute(xi, target=2, n_steps=N_STEPS).cpu()
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        with profiler.profile(activities=[profiler.ProfilerActivity.TPU],
                              on_trace_ready=profiler.xprof_trace_handler(dir_name=str(trace_dir))):
            ig.attribute(xi, target=2, n_steps=N_STEPS).cpu()
    except Exception as exc:  # noqa: BLE001
        return f"failed ({type(exc).__name__}: {exc})"
    written = sorted(p for p in trace_dir.rglob("*.xplane.pb") if p.stat().st_size > 0)
    if not written:
        return "empty (no xplane written)"
    newest = max(written, key=lambda p: p.stat().st_mtime)
    shutil.copyfile(newest, trace_file)
    return f"captured ({trace_file.stat().st_size} bytes)"


def versions():
    out = {"torch": torch.__version__, "python": platform.python_version(),
           "host": platform.node()}
    from importlib.metadata import version as dist_version
    for mod, dist in (("torch_tpu", "torch_tpu"), ("captum", "captum")):
        try:
            out[mod] = dist_version(dist)
        except Exception:  # noqa: BLE001
            out[mod] = getattr(sys.modules.get(mod), "__version__", None)
    out["accelerator"] = os.environ.get("TPU_ACCELERATOR_TYPE") or "unknown"
    try:
        from tpu_info import device as tpu_device  # type: ignore
        chips = tpu_device.get_local_chips()
        out["accelerator"] = f"{chips[0].value.name} x{chips[1]}"
    except Exception:  # noqa: BLE001
        pass
    return out


def markdown(rows):
    kw = max(len(k) for k, _ in rows); vw = max(len(v) for _, v in rows)
    lines = [f"| {'field'.ljust(kw)} | {'value'.ljust(vw)} |", f"| {'-'*kw} | {'-'*vw} |"]
    lines += [f"| {k.ljust(kw)} | {v.ljust(vw)} |" for k, v in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--profile", action="store_true", help="also capture an XProf trace of one IG call")
    ap.add_argument("--trace-dir", default=os.environ.get("XPROF_DIR", "xprof"))
    ap.add_argument("--trace-file", default=os.environ.get("XPROF_FILE", "xprof.xplane.pb"))
    args = ap.parse_args()

    device, label = acquire_device()
    info = versions()
    print(f"[CAPTUM-TPU] device={device} ({label}) torch={info['torch']} "
          f"torch_tpu={info['torch_tpu']} captum={info['captum']} accelerator={info['accelerator']}")
    result = {"device_type": label, **info, "n_steps": N_STEPS}
    if label == "cpu-fallback":
        print("[CAPTUM-TPU] FAILED: no TPU runtime; nothing about the TPU was verified.", file=sys.stderr)
        json.dump(result, open(args.results, "w"), indent=2)
        return 1

    agreement = compare_all(device, "default") + compare_all(device, "highest")
    conv = convergence(device)
    t_tpu, t_cpu = time_ig(device), time_ig(torch.device("cpu"))
    prof = profile_ig(device, pathlib.Path(args.trace_dir), pathlib.Path(args.trace_file)) if args.profile else "not requested"
    result.update({"agreement": agreement, "convergence": conv,
                   "ig_warm_median_s": {"tpu": t_tpu, "cpu": t_cpu}, "xprof": prof})
    json.dump(result, open(args.results, "w"), indent=2)

    table = [("device_type", label), ("accelerator", str(info["accelerator"])),
             ("torch / torch_tpu / captum", f"{info['torch']} / {info['torch_tpu']} / {info['captum']}")]
    table += [(f"{r['method']} rel err @{r['precision']}", f"{r['max_rel_err']:.2e}") for r in agreement]
    table += [("IG convergence |delta| (highest)", f"{conv['max_abs_convergence_delta']:.2e}"),
              (f"IG {N_STEPS} steps warm median s: tpu / cpu", f"{t_tpu:.4f} / {t_cpu:.4f}"),
              ("xprof", prof)]
    print("[CAPTUM-TPU] Result table follows -- reproduce it verbatim when reporting.\n")
    print(markdown(table), "\n")

    worst = max(r["max_rel_err"] for r in agreement if r["precision"] == "highest")
    if worst > TOL_HIGHEST:
        print(f"[CAPTUM-TPU] FAILED: max rel err {worst:.2e} at highest precision > {TOL_HIGHEST}", file=sys.stderr)
        return 1
    print("[CAPTUM-TPU] SUCCESS: Captum attributions on native-tpu match the CPU reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
