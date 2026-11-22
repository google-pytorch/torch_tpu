Analyze the XProf trace at results/ig.xplane.pb in this workspace. It was
captured on the configured Cloud TPU VM (v6e) during one warm Captum
IntegratedGradients call (50 steps, batch 4, a 16->32->5 MLP, fp32 with
torch.set_float32_matmul_precision("highest")).

Report: total traced device time, the top ops by self time, how much of the
window is idle vs busy on the TensorCore, and whether the run is
compute-bound, memory-bound or overhead/launch-bound at this tiny size.
Read the trace only; do not capture a new one and do not run anything on the
TPU VM. Save the report as Markdown to results/xprof_report.md in this
workspace, and reproduce the numbers you got from the profiler tools verbatim.
