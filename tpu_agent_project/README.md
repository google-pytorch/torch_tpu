# Captum on TPU through the TPU agent suite

A first project with the **TPU agent suite** (`google-pytorch/tpu-agent-suite`,
v0.7.0, experimental). It drives the Captum explainability-on-TorchTPU work of
this repository through the suite's two layers: the **unified compute layer**
dispatches a Captum Integrated Gradients script onto the Cloud TPU VM, and the
**`tpu-agent` skill** (Claude Code) answers a sizing question with cited TPU
facts and reads the XProf trace the run brought back.

## What the TPU agent suite is

The torch_tpu FAQ ("Is there an agent to make my onboarding easier?") points
to it. It is an open-source *package of agents*, not a model: Markdown agent
prompts, skills, JSON request/result schemas, Python scripts and a knowledge
base, compiled by its own framework into native plugins for several agent
harnesses (Claude Code, Antigravity, Codex and Cursor). Installed into a
workspace it adds:

| Piece | Where it lands (Claude Code, project scope) |
|---|---|
| `tpu-agent` router skill + `tpu-agent-export-trajectory` | `.claude/skills/` |
| Specialist sub-agents: `tpu-model-onboarding`, `tpu-diagnostics` (XProf), `tpu-knowledge-base` (cited facts), `tpu-kernel-development` + `kernel-worker` (Pallas), `tpu-unwiring`, `analyze-cuda-reference` | `.claude/agents/` |
| Lifecycle hooks (run id per session, schema validation of every delegation, tool allow-list enforcement on the router, job reaping at session end) | `.claude/settings.json` |
| Managed memory blocks (never answer public TPU facts from memory; all TPU work goes through `compute`) | `CLAUDE.md` |
| Compute profile(s) + ownership ledger | `.tpu-agent-suite/` |
| Runtime, scripts, schemas, knowledge-base pages | `~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/` |
| Per-session artifacts and an experiment journal | `~/.tpu-agent-suite/runs/<run_id>/` |

Design choices worth knowing before using it:

- **You talk to one skill.** `tpu-agent` has `read`/`write`/`ask` tools and
  deliberately **no shell**; a PreToolUse hook blocks anything else once the
  skill is active, so the only path to TPU work is delegating to a specialist.
- **Every TPU job goes through `runtime.py compute`.** Backends: `local`,
  `gce_tpu_vm` (gcloud ssh/scp with a persistent control socket), `gke_pod`,
  `gke_job` (Kueue-aware), `gke_xpk`. `compute run --code X --input A --output B`
  copies the script and inputs to the target, runs it under the profile's
  `python_interpreter`, and copies declared outputs back. Jobs a session leaves
  running are cancelled by a SessionEnd hook.
- **Facts are cited.** `tpu-knowledge-base` resolves public TPU facts against
  an index of public pages and returns URLs; the memory block forbids
  answering such facts from memory.
- **Scope today is inference onboarding, profiling and kernels** for
  Hugging Face models on v6e/v7x. There is no "run my arbitrary script"
  specialist, which is why this project uses the compute layer directly for
  the Captum workload and the agents for what they own (facts, profiles).

## Layout

| File | Purpose |
|---|---|
| `setup.sh` | clone + build the suite, install it here for Claude Code, create the Python 3.12 host venv, save the `gce_tpu_vm` profile (`xai-v6e4`, `us-east5-b`) |
| `captum_ig_tpu.py` | the workload: Saliency / IntegratedGradients / LayerIntegratedGradients on an MLP and a CNN, CPU-vs-TPU agreement at default and `"highest"` fp32 matmul precision, IG convergence delta, warm timing, optional XProf capture |
| `run_captum_ig.sh` | `compute preflight` -> `compute inspect` -> `compute run ... --output results.json --output xprof.xplane.pb` |
| `prompts/knowledge_base.md`, `prompts/profile.md` | the two `tpu-agent` requests |
| `run_agent.sh` | headless Claude Code (`claude -p`) in this workspace with a tool allow-list |
| `results/` | what the runs produced (trace files are git-ignored) |
| `FEEDBACK.md` | detailed first-user feedback for the suite's authors |
| `.claude/`, `CLAUDE.md` | written by the suite's installer; committed so the workspace is inspectable. `.tpu-agent-suite/` (compute profile, ownership ledger, current run id) is also written by it, and the installer adds it to `.git/info/exclude`, so it stays local |

## How to run

Prerequisites: `uv`, `gcloud` authenticated (`source ../scripts/env.sh`),
Claude Code >= 2.1.219, a TPU VM whose `~/env/bin/python3` has `torch_tpu`
and `captum` (this repo's `scripts/tpu_vm.sh setup` does that).

```bash
./setup.sh                              # one-time; no VM contact
./run_captum_ig.sh                      # Captum IG on the TPU VM via the compute layer
./run_agent.sh prompts/knowledge_base.md   # cited TPU facts (host only)
./run_agent.sh prompts/profile.md          # tpu-diagnostics reads results/ig.xplane.pb
```

Interactive use works too: `cd tpu_agent_project && source .venv/bin/activate && claude`,
then ask in plain language; the venv matters because the hooks call `python3`
and the runtime needs 3.12.

## What it produced so far (status 2026-11-01)

| Step | Status | Output |
|---|---|---|
| Suite built from source (all four harnesses) and installed here for Claude Code | done | `.claude/`, `CLAUDE.md`, `.tpu-agent-suite/` |
| `gce_tpu_vm` compute profile for `xai-v6e4` saved via `setup plan` / `setup apply` | done | `.tpu-agent-suite/environment.json` (local, see Layout) |
| `tpu-agent` -> `tpu-knowledge-base`, headless, IG memory-sizing question | done: 3 turns, 7.7 min, $2.85 | [`results/knowledge_base_answer.md`](results/knowledge_base_answer.md) |
| `captum_ig_tpu.py` logic check with `MOCK_TPU=1` (CPU) | done, exit 0 | not kept |
| `compute preflight` against the VM | ok: gcloud, account, VM READY (v6e-4) | [`results/preflight.txt`](results/preflight.txt) |
| `compute setup verify` (software on the target) | **fail** on the suite's own pins: `torch<2.14` vs torch 2.15 shipped by torch_tpu 0.2.0; jax, diffusers, pandas, tpu-info, ml_dtypes, google-cloud-storage not installed. `compute run` still works | [`results/setup_verify.json`](results/setup_verify.json) |
| `compute inspect` (live chips over ssh) | `status: unreachable`: the probe needs `tpu_info` in the target interpreter | [`results/inspect.json`](results/inspect.json) |
| `compute run captum_ig_tpu.py --profile` on the TPU | done: native-tpu, 51 s round trip incl. ssh/scp, SUCCESS | [`results/run.log`](results/run.log), [`results/results.json`](results/results.json), `results/ig.xplane.pb` (5.6 MB, not committed) |
| first attempt of the same run | failed: my bug (`"default"` is not a torch_tpu precision value); the compute layer then retried the scp of each missing output 10x, 3.5 min total | [`results/run_attempt1_failed.log`](results/run_attempt1_failed.log) |
| `tpu-agent` -> `tpu-diagnostics` on the trace, headless | done: 11 turns, 14.3 min, $7.74; verdict overhead-bound, TensorCore idle 98% | [`results/xprof_report.md`](results/xprof_report.md) (757 lines, every number from an xprof tool) |

The TPU steps ran on a replacement `xai-v6e4` (the first one failed at TPU
initialisation with a host-side `tpunetd START_SESSION` error and was deleted).
The result table the workload printed, verbatim:

| field                                            | value                        |
| ------------------------------------------------ | ---------------------------- |
| device_type                                      | native-tpu                   |
| accelerator                                      | unknown                      |
| torch / torch_tpu / captum                       | 2.15.0+cu132 / 0.2.0 / 0.9.0 |
| Saliency/MLP rel err @default                    | 2.22e-03                     |
| IntegratedGradients/MLP rel err @default         | 2.07e-03                     |
| LayerIntegratedGradients/MLP[0] rel err @default | 3.13e-03                     |
| IntegratedGradients/CNN rel err @default         | 1.02e-02                     |
| Saliency/MLP rel err @highest                    | 1.65e-07                     |
| IntegratedGradients/MLP rel err @highest         | 4.36e-08                     |
| LayerIntegratedGradients/MLP[0] rel err @highest | 1.01e-07                     |
| IntegratedGradients/CNN rel err @highest         | 3.07e-07                     |
| IG convergence |delta| (highest)                 | 7.45e-08                     |
| IG 50 steps warm median s: tpu / cpu             | 0.0253 / 0.0037              |
| xprof                                            | captured (5634477 bytes)     |

This reproduces the central finding of [`../FINDINGS.md`](../FINDINGS.md) on a
v6e-4 through the suite: at the TPU's default fp32 matmul precision Captum
attributions differ from CPU by 2e-3 (1e-2 for the CNN); with
`torch.set_float32_matmul_precision("highest")` they agree to 1e-7, and IG's
convergence delta is at float noise. At this tiny size the TPU is slower than
the CPU (25 ms vs 4 ms per 50-step IG call): the work is overhead-bound, which
is what the XProf trace is for.

What `tpu-diagnostics` found in that trace, numbers copied from its report:

| Metric | Value | XProf tool |
|---|---|---|
| Profiled window (TPU:0, 1 of 4 v6e chips) | 23.254 ms | get_roofline_model / query_events_db |
| Total device op time | 476.277 us | get_kernel_stats |
| Time inside XLA programs (363 launches) | 980.735 us | query_events_db |
| Gaps between programs (362 gaps, median 56.961 us) | 22,273 us = 95.8% | list_xplane_events |
| Duty cycle / idle | 2.0% / 98.0% | get_kpi_metrics, get_overview |
| MXU utilization | 0.0% | get_kpi_metrics |
| `X64SplitLow`/`X64SplitHigh` custom calls (302 occurrences) | 231.367 us = 48.58% of op time | get_kernel_stats |
| The MLP's four matmul fusions | 5.23 us = 0.02% of the window | get_kernel_stats |

One Captum IG call in eager mode is 363 XLA programs, one per PyTorch op, and
the gaps between them are the time. The `X64Split*` custom calls split Python
float/int scalars that arrive on device as f64/s64: the same scalar-dispatch
path as the buffer leak fixed in google-pytorch/torch_tpu#6102, now visible as
half of all device op time. The agent also flagged that the roofline tool's
`bound_by: HBM` label is an argmax over utilizations that are all 0.00% and
refused to treat it as evidence. This is the `torch.compile` / DeferAndFuse
argument from the first blog, made from a trace rather than a stopwatch.

What the knowledge-base run showed, in short: 32 GB HBM per v6e chip at 1638
GBps, a v6e-4 is four separate 32 GB HBMs (so an unsharded IG batch must fit in
one chip), and `torch.accelerator.memory_*` is the stable memory API for the
`tpu` device. Each claim carries its public URL, and what no source states is
said so. See [FEEDBACK.md](FEEDBACK.md) for what worked and what did not.
