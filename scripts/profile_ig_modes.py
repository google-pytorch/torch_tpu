"""XProf profile of Captum LayerIntegratedGradients on Qwen3-0.6B in one execution mode per process.
  ~/env/bin/python profile_ig_modes.py strict
  TORCH_TPU_DEFER_AND_FUSE=1 ~/env/bin/python profile_ig_modes.py daf
  ~/env/bin/python profile_ig_modes.py compile
Traces -> ~/prof/<mode>/ (open with: xprof --logdir ~/prof --port 6006)"""
import os, pathlib, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import LayerIntegratedGradients
from torch.tpu.profiler import TpuProfilerConfig

mode = sys.argv[1]; torch.set_float32_matmul_precision("highest")
tpu = torch.device("tpu"); MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="tpu").eval()
call = torch.compile(model, backend="tpu") if mode == "compile" else model
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
ids = tok("Integrated Gradients attributes a prediction to its inputs by integrating gradients along a path from a "
          "baseline to the input. On a TPU the question is whether one fixed-shape program beats a loop. The answer is",
          return_tensors="pt").input_ids.to(tpu)
target = tok(" yes").input_ids[0]
fwd = lambda x: call(input_ids=x, logits_to_keep=1).logits[:, -1, target]
lig = LayerIntegratedGradients(fwd, model.get_input_embeddings())
run = lambda: lig.attribute(ids, baselines=torch.full_like(ids, pad), n_steps=64).cpu()

t0 = time.perf_counter(); run(); print(f"[{mode}] warmup/compile {time.perf_counter()-t0:.1f}s")
ts = []
for _ in range(3):
    t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
print(f"[{mode}] warm runs: {[f'{t:.3f}' for t in ts]}")

log_dir = os.path.expanduser(f"~/prof/{mode}"); pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
cfg = TpuProfilerConfig(host_tracer_level=2, device_tracer_level=1, run_dir=pathlib.Path(log_dir))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.TPU],
                            experimental_config=cfg,
                            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)) as prof:
    for _ in range(3):
        run(); torch.accelerator.synchronize(); prof.step()
print(f"[{mode}] trace files:", sorted(os.listdir(log_dir))[:10])
try:
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
except Exception as e:  # noqa: BLE001
    print("key_averages failed:", type(e).__name__, str(e)[:120])
