# Muse Glimmer 30B (standalone)

A self-contained implementation of the Muse Glimmer 30B decoder stack and a
forward-pass benchmark for it. Everything the example needs (RoPE, masks, device
handling, timing, metric reporting) is implemented in this directory; the only
runtime dependency is PyTorch (plus `absl` and, optionally, `tensorboardX` for
metric export).

File                | Contents
------------------- | ----------------------------------------------------
`model.py`          | Config, RoPE helpers, attention, FFN, decoder stack.
`benchmark.py`      | Device selection, timing loop, metric reporting.
`benchmark_test.py` | `absltest` entry point; also runnable as a script.

## Architecture

*   Gated grouped-query attention (32 query heads, 2 KV heads, head dim 128)
    with a sigmoid gate on the attention output.
*   QK RMSNorm without a learned scale, and a constant query scale factor.
*   Sliding-window attention (2048 tokens) in every layer except every 4th one,
    which uses full causal attention.
*   RoPE applied in sliding-window layers only.
*   SwiGLU feed-forward blocks, with pre- and post-norms around both the
    attention and the feed-forward sub-blocks.

The RoPE tables and the sliding-window mask are owned by `MuseGlimmer30B` and
passed to every layer, so each is materialized once rather than once per layer.

The LM head is intentionally omitted: during training it is usually fused with
the loss computation, so the benchmark model stops at the final norm.

## Running

As a plain Python script (auto-detects TPU, CUDA, or CPU):

```shell
python3 examples/muse_glimmer/benchmark_test.py
```

Without an accelerator the example defaults to a small smoke configuration. Pass
`--smoke_test=false` to run the full 30B configuration, and override individual
dimensions as needed:

```shell
python3 examples/muse_glimmer/benchmark_test.py \
  --smoke_test=false --device=cuda --batch_size=8 --seq_len=5120
```

Useful flags: `--device`, `--dtype`, `--compile`, `--num_layers`,
`--batch_size`, `--seq_len`, `--vocab_size`, `--warmup_steps`,
`--measure_steps`, `--benchmark_name`.

`torch.compile` is enabled by default on accelerators (backend `tpu` on TPU,
`inductor` elsewhere) and disabled on CPU, where inductor needs a host C++
compiler that is not always available; pass `--compile` explicitly to override.

As a test target (runs the smoke configuration):

```shell
blaze test //examples/muse_glimmer:benchmark_test
```

## Metrics

Timings are printed to stdout. When `TENSORBOARD_OUTPUT_DIR` is set, the same
values are also written as TensorBoard scalars named
`torch_tpu/<BENCHMARK_PLATFORM>/<benchmark_name>/<run_mode>/torch/<metric>`,
matching the naming used by the e2e benchmark harness. The metrics emitted are
`first_step_time_{seconds,ms}`, `warmup_overhead_{seconds,ms}`,
`post_warmup_step_time_{seconds,ms}`, `num_warmup_steps`,
`peak_device_memory_mb`, `wall_time`, and `succeeded`.

## Running on B200 through the benchmark registry

Add an entry like the following to
`examples/benchmarks/e2e/harness/benchmark_registry.pbtxt`:

```proto
benchmarks {
  name: "muse_glimmer_30b_forward"
  description: "Muse Glimmer 30B forward pass"
  owner: "TorchTPU"

  workload {
    action: "./user_repo/.github/actions/run-benchmark"
    action_inputs { key: "script" value: "examples/muse_glimmer/benchmark_test.py" }
    action_inputs { key: "runtime_flags" value: "--smoke_test=false" }
  }

  environment_configs {
    id: "gpu-b200"
    runner_label: "linux-x86-a4-224-b200-8gpu"
    container_image: "us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build-cuda13.2-cudnn9.15:latest"
    workload_action_inputs { key: "torch_spec" value: "torch==2.13.0" }
    workload_action_inputs { key: "env_vars_hw" value: "BENCHMARK_PLATFORM=b200_1,TORCH_DEVICE_BACKEND_AUTOLOAD=0" }
  }

  metrics { pattern: ".*/post_warmup_step_time_ms$" unit: "ms" stats { stat: LAST_VALUE } }
  metrics { pattern: ".*/first_step_time_ms$" unit: "ms" stats { stat: LAST_VALUE } }
  metrics { pattern: ".*/warmup_overhead_ms$" unit: "ms" stats { stat: LAST_VALUE } }
  metrics { pattern: ".*/peak_device_memory_mb$" unit: "MB" stats { stat: LAST_VALUE } }
  metrics { pattern: ".*/succeeded$" unit: "bool" stats { stat: LAST_VALUE } }
}
```

The full configuration holds ~26.5B parameters (~53 GB in bfloat16), so it needs
a single B200 with the model resident in HBM; reduce `--num_layers`,
`--batch_size` or `--seq_len` to fit smaller devices.
