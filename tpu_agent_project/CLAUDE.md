<!-- tpu-agent-suite:tpu-agent-suite:tpu-knowledge-base:begin -->
For a public fact about TPU hardware, runtimes or software -- a spec, a
default, a supported version, what a doc says -- do NOT answer from memory.
Route the question to the `tpu-knowledge-base` agent, general and multi-part
questions included, as the user asked it rather than broken into
sub-questions of your own. It splits the question itself, resolves each fact
against the source index installed at `~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/data/tpu-knowledge-base/index`, and
returns the entry plus the public URL it came from. Send it a JSON object,
`{"query": "<the question>"}`; a plain-text request is rejected before the
agent starts. Add `entry_id` only with an id or keyword provenance returned
to you; a guessed one sends it to the nearest match, which may be a page it
has to build.
If it returns suggested keywords instead of sources, nothing was looked up:
those are not an answer.
Footnote every such claim with the URL it returns. If provenance reports that a
source does not state something, say so explicitly rather than filling the gap
yourself. Provenance reads only the sources its index names; it does not search
the web.
The playbooks and the Pallas kernel pages are pages at
`~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/data/tpu-knowledge-base/pages/`; `kernel-optimization-index.md` there
routes the kernel pages. Cite the pages each claim relies on; if none covers a
topic, say so explicitly. The installed pages are read-only: never edit them.
<!-- tpu-agent-suite:tpu-agent-suite:tpu-knowledge-base:end -->

<!-- tpu-agent-suite:tpu-agent-suite:compute:begin -->
This applies only to the TPU agent suite: inside the `tpu-agent` skill or one
of its agents, or on a task that skill covers -- onboarding, optimizing,
profiling or benchmarking a model on Cloud TPUs -- even before you load it.
For anything else -- including kubectl, manifests or other clusters the user
names -- just do the task; don't run `python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/framework/runtime.py compute --package tpu-agent-suite@0.7.0` to check whether
this applies.

For that work, every job, server, log and port-forward goes through
`python3 ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/framework/runtime.py compute --package tpu-agent-suite@0.7.0`. Don't write Kubernetes manifests or run `kubectl`,
`xpk`, `gcloud compute` or `ssh` against that cluster yourself: it is shared,
and compute is what keeps this run to its own jobs. Its section in each agent's
instructions, and `compute --help`, say how. A job name or image a task asks
for still goes through compute: give the name as `--name` (on `gke_job` the
compute profile's `job_name_prefix` goes in front and a taken name gets a short
suffix, so the name stays whole inside the job's; a task that requires names to
start a certain way can pass `--job-name-prefix` for that call), and pass
`--image <ref>` to `run` or `submit` to use an image other than the profile's.
If a rule still can't be met through compute, say so in your report rather
than reaching the cluster another way.
<!-- tpu-agent-suite:tpu-agent-suite:compute:end -->
