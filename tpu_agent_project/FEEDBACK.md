# First-user feedback on the TPU agent suite (v0.7.0, Claude Code)

From: Joinal Ahmed (Google Developer Expert, Cloud & AI; TPU Builders Program).
Context: used the suite for one small project, Captum Integrated Gradients on a
Cloud TPU v6e-4 VM via `gce_tpu_vm`, from Claude Code 2.1.306 on macOS, in a
Python 3.12 venv. Everything below was observed, with the command or file
that shows it; "suggestion" lines are opinions. Full project:
https://github.com/joinalahmed/captum-on-torchtpu/tree/main/tpu_agent_project

## What worked well

1. **Build and install from source were clean.** `make build` compiled all
   four harnesses; `installer.py install --destination <ws> --non-interactive`
   wrote exactly what `docs/install.md` section 5 says, and `ownership.json`
   makes the uninstall story credible.
2. **`setup plan` probes are genuinely useful.** It read the gcloud project,
   listed TPU VMs and suggested `xai-v6e4` / `us-east5-b` by itself; the
   question list and `suggested_from` fields are a good interface for an agent
   to ask a user with.
3. **`compute` as the single path to the target** is the right abstraction for
   a researcher: `run --code X --output A:B` is "run this file over there and
   bring these back", identical across VM and GKE, with job tracking and
   SessionEnd reaping. Confirmed on the VM: `run --code captum_ig_tpu.py
   --output results.json:... --output xprof.xplane.pb:...` took 51 s end to
   end for a ~30 s workload, with the ssh control socket making the scp legs
   cheap. The "declared output ... was not produced on the target" error is
   clear when it happens.
4. **Citation-only knowledge base.** For three memory-sizing questions the
   agent returned one URL per claim and said explicitly what no source states
   (free HBM after runtime allocations; PJRT "reserved" semantics). That is
   the behaviour a capacity plan needs. See `results/knowledge_base_answer.md`.
5. **The journal.** `experiment.md` under the run directory made it possible
   to see what the specialist did without reading the transcript.
6. **The design notes in frontmatter** (why the router has no shell, why `ask`
   lives only at the router) are unusually honest and made the suite easier to
   trust and to extend.

## Problems and rough edges

### P1. `python3` in hooks points at the wrong interpreter on a stock Mac
The compiled hooks and the `CLAUDE.md` block call bare `python3`
(`framework/compiler.py` line ~1774 and ~3238). On macOS that resolves to
Xcode's Python 3.9. The runtime itself still runs (stdlib only), but the
SessionStart hook then reports the host check against that interpreter and
tells the agent:

> ask the user to install these packages, which
> /Applications/Xcode.app/Contents/Developer/usr/bin/python3 is missing ...
> The command is `/Applications/Xcode.app/.../python3 -m pip install ...`.
> Don't start other work ... until they are installed.

That would have an agent install XProf and the GCP clients into the system
Python. The docs do say "activate the virtualenv", but nothing records which
interpreter the install was done with.
*Suggestion:* record `sys.executable` at install time and compile it into the
hooks (or an `interpreter` field in `environment.json` the hooks read), and
have the host-check message say "activate the environment the suite was
installed with" before suggesting a pip command.

### P2. No specialist for "run my workload"
The router's `query-understanding.md` is onboarding-shaped: it pins a Hub
model id and plans inference/benchmark/serve. A request to run an existing
script (an attribution, an eval, a training step) on the configured target has
no owner, and the router cannot do it itself (no shell). I ended up calling
`runtime.py compute run` from a shell script, outside the agents.
*Suggestion:* a small `tpu-workload-runner` specialist (accepts a script path,
inputs, outputs, timeout; produces exit code, log path, outputs) would make
the compute layer reachable from natural language and give onboarding and
diagnostics a shared primitive. `tpu-hello-world` is almost that already.

### P3. `tpu-hello-world` is not installed or reachable
`package/agents/tpu-hello-world.md` exists and its prompt is a model of how to
report a smoke check, but it is not in the release manifest's reachable set:
nothing under `.claude/agents/` or `.claude/skills/` references it after
install, and its prompt hard-codes `backend` must be `gke_job`. A first user
on a TPU VM has no agent-driven "can PyTorch see the TPU" check.
*Suggestion:* wire it into `tpu-agent` for all backends; it is the natural
first prompt.

### P4. Cost and latency of a knowledge-base lookup
Three facts, headless `claude -p`: 3 turns, 7.7 min, $2.85 (Claude Code JSON
result). Most time was the KB agent fetching and summarising four pages on
first use. The cache will amortise this, but a first session feels slow and a
user has no progress signal.
*Suggestion:* ship summaries for the curated index pages (they are already
"installed read-only"), and have the KB agent print which page it is fetching.

### P5. Minor
- The installer printed "Added a managed block to CLAUDE.md." twice (two
  blocks, one message each; reads like a duplicate).
- `compute setup plan --backend gce_tpu_vm` rejects `--project <id>` even
  though the driver documents `--project` as a CLI arg; the docs could say
  that `plan` takes only `--profile` and `--backend`.
- `requirements-client.txt` is heavy for a knowledge-base-only user (XProf,
  Vertex AI, BigQuery, DuckDB, pandas). A split (`-kb`, `-diagnostics`,
  `-onboarding`) or extras would help.
- `docs/install.md` says Python 3.12+, but nothing checks the version at
  install or build time (no `sys.version_info` check in `installer.py`,
  `compiler.py` or `runtime.py`).
- The installer appends `.tpu-agent-suite/` to the repository's
  `.git/info/exclude` without saying so. Sensible default, but it should be in
  the install output and in `docs/install.md` section 5, since a user who wants
  to share a compute profile with a team will look for it in `.gitignore`.
- Hooks are compiled with absolute `~/.tpu-agent-suite/...` paths into
  `.claude/settings.json`; committing a workspace to git therefore commits
  a per-user path. A `$HOME`-relative form or a tiny launcher script would
  make workspaces shareable.

## Observations from the TPU run

### P6. `setup verify` fails on the suite's own pins against a stock torch_tpu VM
`requirements-tpu.txt` pins `torch>=2.13.0,<2.14` (the comment says shims
exist for 2.11/2.12/2.13/2.15 but not 2.14, as of a 2026-09 dev build).
Released torch_tpu 0.2.0 installs torch 2.15.0, so verify reports
`torch 2.15.0 installed; needs >=2.13.0,<2.14` with the fix "reinstall
torch_tpu", which would reinstall the same thing. It also fails on jax,
diffusers, pandas, tpu-info, ml_dtypes and google-cloud-storage missing,
although a plain `compute run` needs none of them and worked. The result is
that a correct, released-torch_tpu VM can never reach VERIFIED.
*Suggestion:* pin torch via torch_tpu's own requirement (or `!=2.14.*`), and
split target requirements into "needed by compute" (nothing beyond the
interpreter) and per-specialist sets checked only when that specialist runs.
Output: `results/setup_verify.json`.

### P7. `compute inspect` needs `tpu_info` in the target interpreter
With `~/env` lacking `tpu_info`, inspect returned `status: unreachable,
chips: 0` for a VM that preflight had just reported READY (v6e-4) and that
ran the workload a minute later. "unreachable" is the wrong word for "probe
dependency missing"; an agent reading it would conclude the VM is down.
*Suggestion:* fall back to `/dev/accel*` or `gcloud describe` for the chip
count, and report `probe_missing: tpu_info` instead of unreachable.
Output: `results/inspect.json`.

### P8. Output scp retries after a failed script
When the script exits non-zero before writing its declared outputs, the gce
driver still tries to fetch each output and retries the scp 10x per file
(about 1.5 min each). The failed first attempt took 3.5 min for a 10-second
crash. *Suggestion:* skip output collection when the script's exit status is
non-zero, or retry only on transport errors, not on "No such file".
Output: `results/run_attempt1_failed.log`.

### Smaller
- `accelerator` stays `unknown` on a VM: `TPU_ACCELERATOR_TYPE` is not set
  there and `tpu_info` was absent. The hello-world script has the same gap.
- Stdout and stderr of the remote script come back interleaved with libtpu's
  INFO logging and the gcloud ssh banner; a `--quiet` or a "script output
  only" mode would make the result table easier to find.

## Observations from tpu-diagnostics

This was the best part of the suite. Headless, from a prompt naming a local
`.xplane.pb`: the router delegated once, the specialist located the XProf skill
of the installed `xprof` 2.24.2, copied the trace into the run's artifacts so
cache files stayed out of the workspace, recorded its sha256, ran the tools
(about 21 s of tool time) and wrote a 757-line report with every number
labelled measured / derived (formula given) / inferred, and the raw JSON kept
under the run's artifacts. It caught that `get_roofline_model`'s
`bound_by: HBM` is an argmax over all-zero utilizations and said so rather
than reporting a bandwidth limit. 11 turns, 14.3 min, $7.74.

Small things:
- The report is long for the question asked. A `summary` field in
  `profiling-result.json` with the ten headline numbers, and the long form as
  an artifact, would suit a caller better.
- The agent "skipped the compute profile setup prompt because no job was
  needed", which was the right call, but the SessionStart message still tells
  every session to run `setup verify` first. For host-only work (reading a
  trace, a KB question) that nudge is noise.
- Both headless runs ended with the router's `ask` tool unused and no
  clarification needed; the eval-style prompts in `docs/using.md` carry over
  well to `claude -p`.

## What I would like to contribute
- A knowledge-base source: Captum attributions on TPU match CPU only with
  `torch.set_float32_matmul_precision("highest")` (measured 1e-3 vs 1e-7 rel.
  error; https://github.com/joinalahmed/captum-on-torchtpu/blob/main/FINDINGS.md).
- A `gce_tpu_vm`-capable hello-world / workload-runner agent, if P2/P3 are
  wanted.
