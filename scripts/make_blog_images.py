"""Render the figures for blog_captum_torchtpu.md into images/.

Every number comes from FINDINGS.md. Run from the project root:
    .venv/bin/python scripts/make_blog_images.py
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

matplotlib.use("Agg")

OUT = pathlib.Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(exist_ok=True)

# Validated default palette (light mode), fixed slot order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9.5, "font.size": 10.5,
})


def finish(fig, name, note=None, top=1.0):
  if fig.axes:
    fig.tight_layout(rect=[0, 0.09 if note else 0, 1, top])
  if note:
    fig.text(0.01, 0.012, note, fontsize=8.5, color=MUTED, ha="left", va="bottom")
  fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
  fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.25)
  plt.close(fig)
  print("wrote", name)


def hbar(ax, labels, values, color, fmt, gap=0.25, log=False):
  """Thin horizontal bars with direct labels, baseline anchored."""
  y = range(len(labels))
  ax.barh(y, values, height=1 - gap, color=color, zorder=3)
  ax.set_yticks(list(y), labels)
  ax.invert_yaxis()
  ax.grid(axis="y", visible=False)
  ax.spines["left"].set_visible(False)
  ax.tick_params(axis="y", length=0)
  if log:
    ax.set_xscale("log")
  xmax = max(v for v in values if v is not None)
  for yi, v in zip(y, values):
    ax.text(v * (1.06 if log else 1) + (0 if log else xmax * 0.012), yi, fmt(v), va="center",
            ha="left", fontsize=9.5, color=INK2, zorder=4)


# ----------------------------------------------------------------------------- 1. cover
def cover():
  fig = plt.figure(figsize=(12, 6.3))
  fig.text(0.06, 0.78, "Explainability as a TorchTPU performance lens", fontsize=30, weight="bold", color=INK)
  fig.text(0.06, 0.66, "Captum on Cloud TPUs: does it run, is it right, is it fast, and what it found in TorchTPU",
           fontsize=15, color=INK2)
  stats = [("38", "captum.attr methods run on TPU,\n0 failures"),
           ("1e-7", "relative error vs CPU with\nmatmul precision \"highest\""),
           ("38x", "faster than CPU at 128 IG steps\nwith one environment variable"),
           ("6", "upstream PRs open, 5 more\nbranches ready")]
  for i, (big, small) in enumerate(stats):
    x = 0.06 + i * 0.235
    fig.text(x, 0.40, big, fontsize=40, weight="bold", color=[BLUE, AQUA, ORANGE, BLUE][i])
    fig.text(x, 0.26, small, fontsize=11, color=INK2, va="top", linespacing=1.4)
  fig.text(0.06, 0.08, "Cloud TPU v6e-1  ·  torch 2.15  ·  torch_tpu 0.2.0  ·  Captum 0.9.0",
           fontsize=10.5, color=MUTED)
  finish(fig, "00_cover")


# ----------------------------------------------------------------------------- 2. precision
def precision():
  methods = ["Saliency", "InputXGradient", "IntegratedGradients (50 steps)", "LayerIntegratedGradients",
             "LayerGradCam"]
  default = [2.2e-3, 1.7e-3, 2.1e-3, 3.1e-3, 1.0e-2]
  highest = [1.7e-7, 7.0e-8, 4.4e-8, 1.0e-7, 2.2e-7]
  fig, ax = plt.subplots(figsize=(10, 4.6))
  y = list(range(len(methods)))
  h = 0.36
  ax.barh([yi - h / 2 - 0.02 for yi in y], default, height=h, color=BLUE, label="default matmul precision", zorder=3)
  ax.barh([yi + h / 2 + 0.02 for yi in y], highest, height=h, color=ORANGE,
          label='torch.set_float32_matmul_precision("highest")', zorder=3)
  for yi, d, hgt in zip(y, default, highest):
    ax.text(d * 1.12, yi - h / 2 - 0.02, f"{d:.1e}", va="center", fontsize=9, color=INK2)
    ax.text(hgt * 1.12, yi + h / 2 + 0.02, f"{hgt:.1e}", va="center", fontsize=9, color=INK2)
  ax.set_xscale("log")
  ax.set_xlim(2e-8, 6e-2)
  ax.set_yticks(y, methods)
  ax.invert_yaxis()
  ax.grid(axis="y", visible=False)
  ax.spines["left"].set_visible(False)
  ax.tick_params(axis="y", length=0)
  ax.set_xlabel("max relative error of TPU attribution vs CPU (log scale, lower is better)")
  ax.set_title("One setting turns 1e-3 disagreement into 1e-7 agreement")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
  finish(fig, "01_precision_default_vs_highest", "Small MLP and CNN, fp32, same weights and inputs. v6e-1, torch_tpu 0.2.0.")


# ----------------------------------------------------------------------------- 3. execution modes (53 tokens)
def execution_modes_53():
  modes = ["Captum, CPU", "Captum, TPU strict eager", "Captum, TPU DeferAndFuse", "batched IG, TPU torch.compile"]
  s32 = [4.48, 0.880, 0.238, 0.176]
  s128 = [17.04, 1.078, 0.450, None]
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
  for ax, vals, steps in zip(axes, [s32, s128], [32, 128]):
    present = [(m, v) for m, v in zip(modes, vals) if v is not None]
    hbar(ax, [m for m, _ in present], [v for _, v in present], BLUE, lambda v: f"{v:.2f} s")
    if len(present) < len(modes):
      ax.set_yticks(list(range(len(modes))), modes)
      ax.text(0.02, len(modes) - 1, "out of memory at 128 steps\n(see memory section)", va="center",
              fontsize=9, color=MUTED, transform=ax.get_yaxis_transform())
      ax.set_ylim(len(modes) - 0.4, -0.6)
    ax.set_title(f"{steps} integration steps")
    ax.set_xlabel("warm wall time per attribution (s)")
    ax.set_xlim(0, max(v for v in vals if v) * 1.22)
  axes[1].tick_params(axis="y", labelleft=False)
  fig.suptitle("LayerIntegratedGradients on Qwen3-0.6B: 38x over CPU with one environment variable",
               x=0.01, ha="left", fontsize=13, weight="bold")
  finish(fig, "02_execution_modes_53tok", top=0.92, note=
         "53-token prompt, fp32, matmul precision \"high\". DeferAndFuse = TORCH_TPU_DEFER_AND_FUSE=1. "
         "First-call compile: 26-29 s DeferAndFuse, 68-112 s torch.compile.")


# ----------------------------------------------------------------------------- 4. shipped example
def shipped_example():
  modes = ["strict eager", "DeferAndFuse", "torch.compile(backend=\"tpu\")"]
  warm = [0.80, 0.36, 0.16]
  first = [5, 43, 130]
  hbm = [10.6, 11.1, 15.3]
  err = [9.2e-5, 9.3e-5, 7.0e-5]
  fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
  panels = [("warm median (s)", warm, lambda v: f"{v:.2f} s"), ("first call incl. compile (s)", first, lambda v: f"{v:.0f} s"),
            ("peak HBM (GiB)", hbm, lambda v: f"{v:.1f} GiB"), ("max rel err vs CPU", err, lambda v: f"{v:.1e}")]
  for ax, (title, vals, fmt) in zip(axes, panels):
    hbar(ax, modes, vals, BLUE, fmt)
    ax.set_title(title, fontsize=11.5)
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(visible=False)
  for ax in axes[1:]:
    ax.tick_params(axis="y", labelleft=False)
  fig.suptitle("The shipped example: same attribution, three execution modes", x=0.01, ha="left", fontsize=13,
               weight="bold")
  finish(fig, "03_shipped_example_modes", top=0.90, note=
         "examples/explainability/captum_attribution.py: Qwen3-0.6B, 42-token prompt, 64 steps, fp32 \"highest\", v6e-1. "
         "Compile mode needs Captum #1952 and torch_tpu #6106.")


# ----------------------------------------------------------------------------- 5. XProf
def xprof():
  modes = ["strict eager", "DeferAndFuse"]
  to_copy = [1.35, 0.29]
  other = [1.93 - 1.35, 0.78 - 0.29]
  fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
  ax = axes[0]
  y = [0, 1]
  ax.barh(y, to_copy, height=0.55, color=ORANGE, label="aten::_to_copy (host<->device copies)", zorder=3)
  ax.barh(y, other, left=[t + 0.012 for t in to_copy], height=0.55, color=BLUE, label="everything else", zorder=3)
  for yi, t, o in zip(y, to_copy, other):
    share = f"{t:.2f} s  ({t / (t + o):.0%})"
    if t > 0.6:
      ax.text(t / 2, yi, share, va="center", ha="center", fontsize=9.5, color="white", zorder=4)
      ax.text(t + o + 0.04, yi, f"total {t + o:.2f} s", va="center", fontsize=9.5, color=INK2)
    else:
      ax.text(t + o + 0.04, yi, f"total {t + o:.2f} s   (_to_copy {share})", va="center", fontsize=9.5, color=INK2)
  ax.set_yticks(y, modes)
  ax.invert_yaxis()
  ax.grid(axis="y", visible=False)
  ax.spines["left"].set_visible(False)
  ax.tick_params(axis="y", length=0)
  ax.set_xlim(0, 2.45)
  ax.set_xlabel("Self CPU time, 3 profiled attributions (s)")
  ax.set_title("Host time: 70% of strict eager is blocking copies")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
  ax2 = axes[1]
  hbar(ax2, modes, [2.6, 0.57], BLUE, lambda v: f"{v:.2f} ms")
  ax2.set_xlim(0, 3.4)
  ax2.set_xlabel("avg per _to_copy call (ms), 522 calls each")
  ax2.set_title("Same copies, a fifth of the cost")
  finish(fig, "04_xprof_to_copy",
         "torch.profiler with ProfilerActivity.TPU + TpuProfilerConfig, LayerIntegratedGradients 32 steps, Qwen3-0.6B. "
         "Self TPU time: 5.0 s strict vs 1.23 s DeferAndFuse.")


# ----------------------------------------------------------------------------- 6. memory
def memory():
  cfgs = ["32 steps, one batch", "128 steps, one batch", "128 steps, internal_batch_size=64", "256 steps, one batch"]
  gb = [8.0, 23.9, 10.6, None]
  fig, ax = plt.subplots(figsize=(10, 3.9))
  present = [(c, v) for c, v in zip(cfgs, gb) if v is not None]
  hbar(ax, [c for c, _ in present], [v for _, v in present], BLUE, lambda v: f"{v:.1f} GB")
  ax.set_ylim(len(cfgs) - 0.4, -0.6)
  ax.text(0.3, 3, "out of memory: the [256, 53, 151936] fp32 logits alone are 8 GB", va="center", fontsize=9.5,
          color=ORANGE)
  ax.axvline(32, color=ORANGE, linestyle="--", linewidth=1.4, zorder=2)
  ax.text(31.6, -0.45, "v6e-1 HBM: 32 GB", ha="right", va="center", fontsize=9, color=ORANGE)
  ax.set_xlim(0, 36)
  ax.set_xlabel("peak HBM (GB), Qwen3-0.6B, 53-token prompt, logits_to_keep=1")
  ax.set_title("Peak memory is set by the logits tensor; internal_batch_size bounds it")
  finish(fig, "05_peak_hbm_steps", "128 steps in one batch: 0.857 s warm on TPU vs 11.2 s on CPU. With internal_batch_size=64: 1.42 s.")


# ----------------------------------------------------------------------------- 7. scalar leak
def scalar_leak():
  cases = ["x.add_(0.5) x5000\n(same scalar)", "x.add_(i/1024) x5000\n(new scalar each call)",
           "Captum IG, n_steps 10..109\n(100 attributions)"]
  before = [2, 4999, 6361]
  after = [2, 903, 280]
  fig, ax = plt.subplots(figsize=(10, 4.2))
  y = list(range(len(cases)))
  h = 0.36
  ax.barh([yi - h / 2 - 0.02 for yi in y], before, height=h, color=BLUE, label="torch_tpu 0.2.0", zorder=3)
  ax.barh([yi + h / 2 + 0.02 for yi in y], after, height=h, color=ORANGE, label="with PR #6102 (bounded cache)", zorder=3)
  for yi, b, a in zip(y, before, after):
    ax.text(b + 60, yi - h / 2 - 0.02, f"+{b:,}", va="center", fontsize=9.5, color=INK2)
    ax.text(a + 60, yi + h / 2 + 0.02, f"+{a:,}", va="center", fontsize=9.5, color=INK2)
  ax.set_yticks(y, cases)
  ax.invert_yaxis()
  ax.grid(axis="y", visible=False)
  ax.spines["left"].set_visible(False)
  ax.tick_params(axis="y", length=0)
  ax.set_xlim(0, 7400)
  ax.set_xlabel("live device allocations added (torch.accelerator.memory_stats, after gc and synchronize)")
  ax.set_title("Issue #6036: every distinct Python scalar left a device buffer behind")
  ax.legend(loc="lower right")
  finish(fig, "06_scalar_buffer_leak", "scripts/scalar_leak_check.py on v6e-1. empty_cache() released none of the leaked buffers.")


# ----------------------------------------------------------------------------- 8. IG convergence
def convergence():
  steps = [32, 128, 256]
  total = [-26.6, -8.2, 44.4]
  fig, ax = plt.subplots(figsize=(10, 4.2))
  ax.axhline(0, color=AXIS, linewidth=1)
  ax.plot(steps, total, color=BLUE, linewidth=2, marker="o", markersize=7, zorder=3)
  for s, t in zip(steps, total):
    ax.annotate(f"{t:+.1f}", (s, t), textcoords="offset points", xytext=(0, -18 if t > 40 else (10 if t > -20 else -16)), ha="center",
                fontsize=10, color=INK2)
  ax.annotate("convergence delta at 256 steps: 27.3\n(against a total of 44.4)", (256, 44.4), xytext=(140, 28),
              fontsize=9.5, color=ORANGE, arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 1})
  ax.set_xticks(steps)
  ax.set_xlabel("Integrated Gradients steps (Gauss-Legendre, Captum on CPU)")
  ax.set_ylabel("total attribution (sum over tokens)")
  ax.set_title("The reference had not converged: raw logit + pad baseline is a poor IG setup for an LLM")
  ax.set_xlim(10, 290)
  ax.set_ylim(-34, 52)
  finish(fig, "07_ig_convergence_qwen3",
         "Qwen3-0.6B fp32, prompt \"The capital of France is\", target \" Paris\". Not TPU-specific; always request return_convergence_delta.")


# ----------------------------------------------------------------------------- 9. contributions
def contributions():
  items = [
      ("torch_tpu #6102", "bounded scalar buffer cache (fixes #6036)", "open PR"),
      ("torch_tpu #6103", "issue: aten::max_unpool2d missing", "open issue"),
      ("torch_tpu #6105", "max_unpool2d / max_unpool3d lowerings", "open PR"),
      ("torch_tpu #6106", "compile against the layout CPU args are copied with", "open PR"),
      ("captum #1951", "device-generic accelerator comparison tests", "open PR"),
      ("captum #1952", "Python ints for LIG output offsets", "open PR"),
      ("missing-ops-batch  (#6111)", "8 ops: smooth_l1/huber loss, logaddexp(2), nansum, std_mean, trace, addbmm", "branch, issue filed"),
      ("explainability-example  (#6114)", "example script + 7-section docs case study", "branch, issue filed"),
      ("dynamism-denylist-triage  (#6112)", "11 stale deny-list entries removed, clamp.Tensor diagnosed", "branch, issue filed"),
      ("docs-memory-apis  (#4413)", "torch.accelerator memory APIs in the API reference", "branch, issue filed"),
      ("loader-import-error-context  (#6113)", "extension loader names glue + installed PyTorch", "branch, issue filed"),
  ]
  colors = {"open PR": BLUE, "open issue": ORANGE, "branch, issue filed": AQUA}
  fig, ax = plt.subplots(figsize=(12, 5.6))
  ax.set_axis_off()
  ax.set_xlim(0, 1)
  ax.set_ylim(0, len(items) + 1.2)
  ax.text(0, len(items) + 0.9, "What explainability found in TorchTPU, and where each fix lives", fontsize=13, weight="bold")
  for i, (ref, desc, kind) in enumerate(items):
    y = len(items) - i
    ax.add_patch(plt.Rectangle((0, y - 0.32), 0.012, 0.64, color=colors[kind]))
    ax.text(0.025, y, ref, fontsize=10.5, weight="bold", va="center", family="monospace")
    ax.text(0.36, y, desc, fontsize=10.5, va="center", color=INK2)
    ax.text(0.99, y, kind, fontsize=9.5, va="center", ha="right", color=colors[kind])
  finish(fig, "08_contributions_map", "Branches are pushed to joinalahmed/torch_tpu; PRs wait on the repository's 3-open-PR limit for users without write access.")


if __name__ == "__main__":
  for fn in (cover, precision, execution_modes_53, shipped_example, xprof, memory, scalar_leak, convergence,
             contributions):
    fn()
