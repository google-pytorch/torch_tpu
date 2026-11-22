# XProf report: warm Captum IntegratedGradients call on TPU v6e (read-only analysis of an existing trace)

- Trace analysed: `/Users/joinal/pytorvhtpu/tpu_agent_project/results/ig.xplane.pb` (5,634,477 bytes, sha256 `18a85084ade01a230e32e82a21c51db14a465229e790ce11e3e67b2115f5686e`). A byte-identical copy was read from `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/trace/ig.xplane.pb` so XProf's per-directory cache files land in the run's artifacts, not in `results/`.
- Workload (as described by the caller, not re-run): one warm `captum.attr.IntegratedGradients` call, 50 steps, batch 4, 16->32->5 MLP, fp32 with `torch.set_float32_matmul_precision('highest')`, torch 2.15.0 / torch_tpu 0.2.0 / captum 0.9.0, Cloud TPU VM v6e, one device in use out of four addressable.
- Constraints honoured: nothing was captured or run on the TPU VM, no compute job was submitted; only `xprof` subcommands were run on this host against the local `.xplane.pb`.
- Profiler: **xprof 2.24.2** (`/Users/joinal/pytorvhtpu/tpu_agent_project/.venv/bin/xprof`). Every tool answered in about one second; total tool wall time about 21 s.
- Raw tool outputs (every number below is copied from them): `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/*.json`, HLO text under `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/hlo_text/`, derived arithmetic in `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/derived_module_timeline.json`, `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/derived_time_budget.json`, `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/aggregate_xplane_events_ops_summary.json`.

Throughout, **Measured** means a value the XProf CLI returned; **Derived** means arithmetic on measured values (formula given); **Inferred** means an interpretation.

## 1. What the trace covers (measured)

`check_kernel_profiling` plane/line inventory:

```json
{
  "planes": [
    {
      "name": "/device:TPU:0",
      "lines": [
        {
          "name": "XLA Modules",
          "event_count": 363
        },
        {
          "name": "XLA Ops",
          "event_count": 472
        },
        {
          "name": "Async XLA Ops",
          "event_count": 0
        },
        {
          "name": "TC Overlay",
          "event_count": 0
        },
        {
          "name": "Pallas Primitives",
          "event_count": 0
        },
        {
          "name": "_counters_",
          "event_count": 0
        },
        {
          "name": "counters_0",
          "event_count": 43641
        }
      ],
      "total_events": 44476
    },
    {
      "name": "/host:metadata",
      "lines": [],
      "total_events": 0
    },
    {
      "name": "/device:CUSTOM:Megascale Trace",
      "lines": [],
      "total_events": 0
    },
    {
      "name": "Task Environment",
      "lines": [],
      "total_events": 0
    },
    {
      "name": "/host:CPU",
      "lines": [],
      "total_events": 0
    }
  ],
  "capture_valid_for_device_analysis": true,
  "capture_valid_for_llo_analysis": false,
  "capture_valid_for_counter_sampling": false
}
```

`get_hosts`: one host, `ig`. `get_overview.run_environment`:

```json
{
  "device_core_count": "1",
  "device_type": "TPU v6 Lite",
  "host_count": "1",
  "is_training": "false",
  "task_count": "1 (num tasks per host = 1)"
}
```

- Devices traced: `/device:TPU:0` only (1 of the 4 addressable v6e chips; the caller said one device was in use). It carries XLA op events: **472 op occurrences over 73 distinct HLO ops** (`aggregate_xplane_events --group_by=line,event`, counter rows excluded; `check_kernel_profiling` line `XLA Ops` event_count 472), and 363 `XLA Modules` executions of 16 distinct programs (`list_hlo_modules`: count 16).
- `/host:CPU` holds **0 events**, so nothing host-side (Python, torch_tpu dispatch, transfers) is in this trace. Host-side time can only be inferred from device gaps.
- No `Steps` line: `get_step_trace` -> `"status": "NO_DATA"` ("the trace may lack step markers"); `check_host_boundness` -> `"status": "INSUFFICIENT_DATA"`. Every per-step average XProf prints is therefore `0.00` (zero steps), not a measurement of zero idle.
- `counters_0` holds 43,641 zero-duration hardware-counter rows (one sample, `sample: 9`). XProf flags `capture_valid_for_counter_sampling: false`; `get_kernel_utilization` failed (`INTERNAL_ERROR`, "no data returned"). The raw counter values are quoted in section 6 as a cross-check only.
- `get_llo_analysis`: `"status": "UNAVAILABLE", "reason": "LLO_DATA_ABSENT"` (the trace was captured without `--xla_xprof_register_llo_debug_info=true`). No LLO/bundle-level data.

## 2. Headline numbers (measured)

`get_kpi_metrics`:
```json
{
  "step_time_ms": "0.00",
  "duty_cycle_percent": "2.0%",
  "mxu_utilization_percent": "0.0%",
  "roofline_utilization": "0.00%",
  "flops_provenance": "opaque_custom_call",
  "derived_flops_fraction": 0.0,
  "peak_hbm_gib": -1.0,
  "accelerator_info": {
    "device_type": "TPU v6 Lite",
    "device_core_count": "1"
  }
}
```

`get_overview.performance_summary`:
```json
{
  "device_duty_cycle_percent": "2.0%",
  "device_idle_time_percent": "98.0%",
  "device_op_time_eager_percent": "0.0%",
  "device_tf_op_percent": "100.0%",
  "flop_rate_utilization_relative_to_roofline": "0.00%",
  "host_idle_time_percent": "100.0%",
  "host_op_time_eager_percent": "0.0%",
  "host_tf_op_percent": "0.0%",
  "memory_bw_utilization_relative_to_hw_limit": "0.00%",
  "mxu_utilization_percent": "0.0%",
  "active_steptime_ms_average": "0.00",
  "host_transfer_ms_average": "0.00",
  "steptime_ms_average": "0.00",
  "steptime_ms_standard_deviation": "0.00",
  "tc_active_ms_average": "0.00",
  "tc_idle_ms_average": "0.00",
  "tc_infeed_ms_average": "0.00",
  "tc_outfeed_ms_average": "0.00",
  "roofline_efficiency_percent": "0.00%",
  "compute_efficiency_percent": "0.00%",
  "max_mem_bw_utilization_percent": "0.00%",
  "hbm_bw_utilization_percent": "0.00%",
  "bound_by": "HBM",
  "operational_intensity_flop_per_byte": 6.5841
}
```

`get_kernel_stats --include_summary=True --limit=20` (summary fields; the 20 kernel rows are in the saved file):
```json
{
  "total_device_duration_ns": 476277,
  "total_device_duration_us": 476.277,
  "total_device_duration_ms": 0.476277,
  "by_line_duration_ns": {
    "XLA Modules": 980735,
    "XLA Ops": 476277,
    "counters_0": 0
  },
  "custom_call_duration_us": 231.367,
  "custom_call_share_pct": 48.58,
  "total_records": 73,
  "top_level_records": 73,
  "returned_records": 20,
  "total_duration_us_all_records": 476.4732,
  "stats": {
    "mean_us": 2.7022,
    "std_us": 6.6094
  }
}
```

`query_events_db` (window span of the device planes):
```json
[
  {
    "device": "TPU:0",
    "category": "XLA Modules",
    "n": 363,
    "sum_dur_us": 980.735,
    "min_start_ns": 134946542,
    "max_end_ns": 158200471,
    "span_ms": 23.253929
  },
  {
    "device": "TPU:0",
    "category": "XLA Ops",
    "n": 472,
    "sum_dur_us": 476.277,
    "min_start_ns": 134946559,
    "max_end_ns": 158200470,
    "span_ms": 23.253911
  },
  {
    "device": "TPU:0",
    "category": "counters_0",
    "n": 43641,
    "sum_dur_us": 0.0,
    "min_start_ns": 0,
    "max_end_ns": 0,
    "span_ms": 0.0
  }
]
```

`aggregate_xplane_events --plane_regex=TPU --group_by=line`:
```json
[
  {
    "line": "XLA Modules",
    "count": 363,
    "total_duration_ps": 980895702,
    "avg_duration_ps": 2702192.0165289254,
    "min_duration_ps": 396406,
    "p50_duration_ps": 455078.0,
    "p90_duration_ps": 5942546.8,
    "p99_duration_ps": 6167443.72,
    "max_duration_ps": 107281172,
    "std_dev_ps": 6609429.654949361
  },
  {
    "line": "XLA Ops",
    "count": 472,
    "total_duration_ps": 476473204,
    "avg_duration_ps": 1009477.1271186441,
    "min_duration_ps": 78,
    "p50_duration_ps": 929297.0,
    "p90_duration_ps": 1439906.0,
    "p99_duration_ps": 1635417.8800000001,
    "max_duration_ps": 1745000,
    "std_dev_ps": 208922.7560399737
  },
  {
    "line": "counters_0",
    "count": 43641,
    "total_duration_ps": 0,
    "avg_duration_ps": 0.0,
    "min_duration_ps": 0,
    "p50_duration_ps": 0.0,
    "p90_duration_ps": 0.0,
    "p99_duration_ps": 0.0,
    "max_duration_ps": 0,
    "std_dev_ps": 0.0
  }
]
```

## 3. The four questions

### 3.1 Total traced device time

- **Measured:** `get_kernel_stats.total_device_duration_us = 476.277` (wall-clock device *op* time, disjoint-interval union of the `XLA Ops` line); `total_duration_us_all_records = 476.4732` (sum of outermost kernel durations). `get_hlo_stats.total_self_time_ms_all_records = 0.4765` ms agrees.
- **Measured:** `XLA Modules` line total = 980.896 us over 363 program executions (device time inside XLA programs, including per-program entry/exit that is not attributed to any op).
- **Measured:** profiled window = 23.253929 ms (first module start to last module end; `get_hlo_op_profile.total_profile_time_ms = 23.2539`, `get_roofline_model.program.total_time_ms = 23.254`).
- **Derived:** of the 23.254 ms window, 476.3 us (2.05%) is HLO op execution, 504.6 us (2.17%) is time inside XLA programs but outside any traced op (`modules - ops`), and 22273.0 us (95.78%) is gaps between consecutive program executions (sum over the 362 gaps from `list_xplane_events` on the `XLA Modules` line).

### 3.2 Top ops by self time

`get_hlo_stats --limit=0 --sort_by=self_time` returned 77 records (`truncated: false`); the first 20 by `total_self_time_us` are below. `self_time_percent` is XProf's share of the *whole profiled window* (IDLE included), which is why the largest op is 0.32%. `source_file`/`source_line` are `""`/`0` on every record and `get_top_hlo_ops` rows carry no `stack_frame`, so provenance is **empty in trace** for all ops (the capture had no Python source metadata).

| # | op (trace name) | HLO module | category | occurrences | total self time (us) | per call (us, derived = total/occurrences) | self_time_percent of window | bound_by (tool) |
|---|---|---|---|---|---|---|---|---|
| 1 | `broadcast_multiply_fusion` | `tt_jit_mul(350907899631732211)` | loop fusion | 50 | 75.33875 | 1.5068 | 0.3240% | HBM |
| 2 | `broadcast_add_fusion` | `tt_jit_add(2232073991294053191)` | loop fusion | 50 | 49.4 | 0.9880 | 0.2124% | HBM |
| 3 | `broadcast_subtract_fusion` | `tt_jit_sub(6798965666565663463)` | loop fusion | 51 | 49.12 | 0.9631 | 0.2112% | HBM |
| 4 | `custom-call` | `tt_jit_sub(6798965666565663463)` | custom-call | 51 | 47.27875 | 0.9270 | 0.2033% | Unknown |
| 5 | `custom-call` | `tt_jit_add(2232073991294053191)` | custom-call | 50 | 46.4 | 0.9280 | 0.1995% | Unknown |
| 6 | `custom-call.1` | `tt_jit_sub(6798965666565663463)` | custom-call | 51 | 46.3925 | 0.9097 | 0.1995% | Unknown |
| 7 | `custom-call` | `tt_jit_mul(350907899631732211)` | custom-call | 50 | 45.825 | 0.9165 | 0.1971% | Unknown |
| 8 | `custom-call.1` | `tt_jit_add(2232073991294053191)` | custom-call | 50 | 45.58 | 0.9116 | 0.1960% | Unknown |
| 9 | `cat.1` | `tt_jit_cat(8834897215032019949)` | concatenate | 1 | 1.7375 | 1.7375 | 0.0075% | HBM |
| 10 | `fusion` | `tt_jit_linear_backward(12749327565423576208)` | convolution fusion | 1 | 1.39125 | 1.3913 | 0.0060% | HBM |
| 11 | `convolution_add_fusion` | `tt_jit_linear(2232135972950488025)` | convolution fusion | 1 | 1.36625 | 1.3662 | 0.0059% | HBM |
| 12 | `broadcast_multiply_fusion` | `tt_jit_mul(9558768019819878584)` | loop fusion | 1 | 1.28 | 1.2800 | 0.0055% | HBM |
| 13 | `copy.18` | `tt_jit_cat(8834897215032019949)` | data formatting | 1 | 1.25875 | 1.2588 | 0.0054% | HBM |
| 14 | `convolution_add_fusion` | `tt_jit_linear(2454675186444914655)` | convolution fusion | 1 | 1.25625 | 1.2563 | 0.0054% | HBM |
| 15 | `copy.1` | `tt_jit_mul(9558768019819878584)` | data formatting | 1 | 1.215 | 1.2150 | 0.0052% | HBM |
| 16 | `fusion` | `tt_jit_linear_backward(14565005824055855247)` | convolution fusion | 1 | 1.2125 | 1.2125 | 0.0052% | HBM |
| 17 | `as_strided.1` | `tt_jit_mul(9558768019819878584)` | data formatting | 1 | 1.2025 | 1.2025 | 0.0052% | HBM |
| 18 | `copy.3` | `tt_jit_cat(8834897215032019949)` | data formatting | 1 | 1.135 | 1.1350 | 0.0049% | HBM |
| 19 | `copy.21` | `tt_jit_cat(8834897215032019949)` | data formatting | 1 | 1.12875 | 1.1287 | 0.0049% | HBM |
| 20 | `subtract_multiply_fusion` | `tt_jit_tanh_backward(7455261703457177013)` | loop fusion | 1 | 1.11625 | 1.1162 | 0.0048% | HBM |

Verbatim `get_hlo_stats` records for the top 8 (the rows that carry 97% of op time):
```json
[
  {
    "rank": 1,
    "program_id": 350907899631732211,
    "category": "loop fusion",
    "op_name": "broadcast_multiply_fusion",
    "tf_op_name": "mul/mul:",
    "occurrences": 50,
    "total_time_us": 75.33875,
    "total_self_time_us": 75.33875,
    "self_time_percent": 0.32398310878993497,
    "measured_flop_rate": 0.042474822053724014,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.3189344450907315,
    "bound_by": "HBM",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 2,
    "program_id": 2232073991294053191,
    "category": "loop fusion",
    "op_name": "broadcast_add_fusion",
    "tf_op_name": "add/add:",
    "occurrences": 50,
    "total_time_us": 49.4,
    "total_self_time_us": 49.4,
    "self_time_percent": 0.21243736555521278,
    "measured_flop_rate": 0.06477732793522267,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.4863992393740758,
    "bound_by": "HBM",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 3,
    "program_id": 6798965666565663463,
    "category": "loop fusion",
    "op_name": "broadcast_subtract_fusion",
    "tf_op_name": "sub/sub:",
    "occurrences": 51,
    "total_time_us": 49.12,
    "total_self_time_us": 49.12,
    "self_time_percent": 0.2112332671269646,
    "measured_flop_rate": 0.06644951140065146,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.4989553109442372,
    "bound_by": "HBM",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 4,
    "program_id": 6798965666565663463,
    "category": "custom-call",
    "op_name": "custom-call",
    "tf_op_name": "",
    "occurrences": 51,
    "total_time_us": 47.27875,
    "total_self_time_us": 47.27875,
    "self_time_percent": 0.20331524487335056,
    "measured_flop_rate": 0.0,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.0,
    "bound_by": "Unknown",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 5,
    "program_id": 2232073991294053191,
    "category": "custom-call",
    "op_name": "custom-call",
    "tf_op_name": "",
    "occurrences": 50,
    "total_time_us": 46.4,
    "total_self_time_us": 46.4,
    "self_time_percent": 0.19953631096683955,
    "measured_flop_rate": 0.0,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.0,
    "bound_by": "Unknown",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 6,
    "program_id": 6798965666565663463,
    "category": "custom-call",
    "op_name": "custom-call.1",
    "tf_op_name": "",
    "occurrences": 51,
    "total_time_us": 46.3925,
    "total_self_time_us": 46.3925,
    "self_time_percent": 0.19950405833036858,
    "measured_flop_rate": 0.0,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.0,
    "bound_by": "Unknown",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 7,
    "program_id": 350907899631732211,
    "category": "custom-call",
    "op_name": "custom-call",
    "tf_op_name": "",
    "occurrences": 50,
    "total_time_us": 45.825,
    "total_self_time_us": 45.825,
    "self_time_percent": 0.19706360883740134,
    "measured_flop_rate": 0.0,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.0,
    "bound_by": "Unknown",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  },
  {
    "rank": 8,
    "program_id": 2232073991294053191,
    "category": "custom-call",
    "op_name": "custom-call.1",
    "tf_op_name": "",
    "occurrences": 50,
    "total_time_us": 45.58,
    "total_self_time_us": 45.58,
    "self_time_percent": 0.1960100227126842,
    "measured_flop_rate": 0.0,
    "flops": 0.0,
    "measured_memory_bw_gbs": 0.0,
    "bound_by": "Unknown",
    "source_file": "",
    "source_line": 0,
    "core_type": "TensorCore",
    "parent_op_name": ""
  }
]
```

The five `custom-call` rows are XLA's `X64SplitLow` / `X64SplitHigh` custom calls (`get_top_hlo_ops` names them `custom-call [X64SplitLow]`, `custom-call.1 [X64SplitHigh]`). `get_kernel_stats` merges identical HLO text across modules:
```json
[
  {
    "kernel_name": "%custom-call = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target=\"X64SplitLow\"",
    "canonical_name": "%custom-call",
    "short_name": "%custom-call",
    "hlo_op_name": "%custom-call",
    "raw_name": "%custom-call = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target=\"X64SplitLow\"",
    "total_duration_us": 93.677,
    "execution_count": 101,
    "avg_duration_us": 0.9275,
    "is_intra_kernel_region": false
  },
  {
    "kernel_name": "%custom-call.1 = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target=\"X64SplitHigh\"",
    "canonical_name": "%custom-call.1",
    "short_name": "%custom-call.1",
    "hlo_op_name": "%custom-call.1",
    "raw_name": "%custom-call.1 = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target=\"X64SplitHigh\"",
    "total_duration_us": 91.9711,
    "execution_count": 101,
    "avg_duration_us": 0.9106,
    "is_intra_kernel_region": false
  },
  {
    "kernel_name": "%broadcast_multiply_fusion = f32[4,16]{1,0:T(4,128)} fusion(f32[4,16]{1,0:T(4,128)} %Arg_0.0, f32[]{:T(128)} %custom-call), kind=kLoop, calls=%fused_computation",
    "canonical_name": "%broadcast_multiply_fusion",
    "short_name": "%broadcast_multiply_fusion",
    "hlo_op_name": "%broadcast_multiply_fusion",
    "raw_name": "%broadcast_multiply_fusion = f32[4,16]{1,0:T(4,128)} fusion(f32[4,16]{1,0:T(4,128)} %Arg_0.0, f32[]{:T(128)} %custom-call), kind=kLoop, calls=%fused_computation",
    "total_duration_us": 75.3407,
    "execution_count": 50,
    "avg_duration_us": 1.5068,
    "is_intra_kernel_region": false
  }
]
```

`get_hlo_op_profile --view=category` (where op time goes by category; fractions are of the whole window):
```json
[
  {
    "category": "IDLE",
    "total_self_time_ms": 22.7774,
    "fraction_of_total_time": 0.9795,
    "total_flops": 0.0,
    "bytes_accessed": 0.0,
    "op_count": 1,
    "top_op": "IDLE"
  },
  {
    "category": "custom-call",
    "total_self_time_ms": 0.2315,
    "fraction_of_total_time": 0.01,
    "total_flops": 0.0,
    "bytes_accessed": 0.0,
    "op_count": 5,
    "top_op": "custom-call"
  },
  {
    "category": "loop fusion",
    "total_self_time_ms": 0.1762,
    "fraction_of_total_time": 0.0076,
    "total_flops": 28864.0,
    "bytes_accessed": 154312.0,
    "op_count": 5,
    "top_op": "broadcast_multiply_fusion"
  },
  {
    "category": "data formatting",
    "total_self_time_ms": 0.0578,
    "fraction_of_total_time": 0.0025,
    "total_flops": 0.0,
    "bytes_accessed": 77312.0,
    "op_count": 53,
    "top_op": "copy.18"
  },
  {
    "category": "convolution fusion",
    "total_self_time_ms": 0.0053,
    "fraction_of_total_time": 0.0002,
    "total_flops": 545000.0,
    "bytes_accessed": 141524.0,
    "op_count": 4,
    "top_op": "fusion"
  },
  {
    "category": "concatenate",
    "total_self_time_ms": 0.0024,
    "fraction_of_total_time": 0.0001,
    "total_flops": 0.0,
    "bytes_accessed": 27200.0,
    "op_count": 2,
    "top_op": "cat.1"
  },
  {
    "category": "non-fusion elementwise",
    "total_self_time_ms": 0.0021,
    "fraction_of_total_time": 0.0001,
    "total_flops": 64.0,
    "bytes_accessed": 51968.00000000001,
    "op_count": 2,
    "top_op": "tanh.1"
  },
  {
    "category": "reduce",
    "total_self_time_ms": 0.001,
    "fraction_of_total_time": 0.0,
    "total_flops": 3136.0,
    "bytes_accessed": 13059.999999999998,
    "op_count": 1,
    "top_op": "reduce"
  },
  {
    "category": "pad",
    "total_self_time_ms": 0.001,
    "fraction_of_total_time": 0.0,
    "total_flops": 0.0,
    "bytes_accessed": 4803.999999999999,
    "op_count": 1,
    "top_op": "select_backward.3"
  }
]
```

### 3.3 Idle vs busy on the TensorCore

- **Measured:** `device_duty_cycle_percent = "2.0%"`, `device_idle_time_percent = "98.0%"` (`get_overview`); `duty_cycle_percent = "2.0%"` (`get_kpi_metrics`); IDLE category `total_self_time_ms = 22.7774` of `total_profile_time_ms = 23.2539` (`fraction_of_total_time = 0.9795`).
- **Derived breakdown of the 23.254 ms window:** busy in HLO ops 476.3 us = 2.05%; inside programs but not in an op 504.6 us = 2.17%; between programs 22273.0 us = 95.78%. Gap statistics over the 362 inter-program gaps: min 18.289 us, p50 56.961 us, p90 68.865 us, max 465.58 us (after `tt_jit_stack`), mean 61.528 us.
- **Not measured:** `tc_idle_ms_average`, `tc_infeed_ms_average`, `steptime_ms_average` are all `"0.00"` because the trace has no step markers (zero steps to average over). They do not say the TensorCore was never idle.

Per-program time budget (**derived** from `list_xplane_events` module events and `get_hlo_stats` per-program op time; `module - ops` is device time inside the program not attributed to any traced op):

| XLA program | launches | module total (us) | module avg (us) | traced op total (us) | module - ops (us) | op occurrences |
|---|---|---|---|---|---|---|
| `tt_jit_sub(6798965666565663463)` | 51 | 300.524 | 5.893 | 142.791 | 157.733 | 153 |
| `tt_jit_add(2232073991294053191)` | 50 | 295.865 | 5.917 | 141.38 | 154.485 | 150 |
| `tt_jit_mul(350907899631732211)` | 50 | 122.066 | 2.441 | 121.164 | 0.902 | 100 |
| `tt_jit_stack(7200413339117915842)` | 1 | 107.281 | 107.281 | 0.72 | 106.561 | 1 |
| `tt_jit_ones_like(11419731517694175880)` | 200 | 83.945 | 0.42 | 0.0 | 83.945 | 0 |
| `tt_jit_cat(8834897215032019949)` | 1 | 55.389 | 55.389 | 55.339 | 0.05 | 51 |
| `tt_jit_mul(9558768019819878584)` | 1 | 3.714 | 3.714 | 3.697 | 0.017 | 3 |
| `tt_jit_sum(9674474330115236837)` | 1 | 2.031 | 2.031 | 2.014 | 0.017 | 2 |
| `tt_jit_linear_backward(12749327565423576208)` | 1 | 1.7 | 1.7 | 1.391 | 0.309 | 3 |
| `tt_jit_linear_backward(14565005824055855247)` | 1 | 1.531 | 1.531 | 1.212 | 0.319 | 3 |
| `tt_jit_linear(2232135972950488025)` | 1 | 1.38 | 1.38 | 1.366 | 0.014 | 1 |
| `tt_jit_linear(2454675186444914655)` | 1 | 1.271 | 1.271 | 1.256 | 0.015 | 1 |
| `tt_jit_tanh_backward(7455261703457177013)` | 1 | 1.131 | 1.131 | 1.116 | 0.015 | 1 |
| `tt_jit_tanh(1028774433826226245)` | 1 | 1.083 | 1.083 | 1.069 | 0.014 | 1 |
| `tt_jit_mul(11390128859972454306)` | 1 | 1.004 | 1.004 | 0.99 | 0.014 | 1 |
| `tt_jit_select_backward(10733800078036239645)` | 1 | 0.98 | 0.98 | 0.968 | 0.012 | 1 |

The launch sequence (`list_xplane_events`, 363 events) is: 50 x (`tt_jit_sub`, `tt_jit_mul`, `tt_jit_add`) -- the per-alpha `baseline + alpha * (input - baseline)` interpolation -- then `tt_jit_cat`, forward (`tt_jit_linear`, `tt_jit_tanh`, `tt_jit_linear`), 200 x `tt_jit_ones_like`, `tt_jit_stack`, then the backward programs (`select_backward`, `linear_backward`, `tanh_backward`, `linear_backward`), `tt_jit_mul`, `tt_jit_sum`, `tt_jit_sub`, `tt_jit_mul`. Each PyTorch op is its own XLA program (`tt_jit_<op>`): the device sees 363 separate program launches for one IG call.

### 3.4 Compute-bound, memory-bound, or overhead/launch-bound?

`get_roofline_model` program block and device constants (**measured**):
```json
{
  "program": {
    "bound_by": "HBM",
    "bound_by_ridge_point_flop_per_byte": 577.963,
    "operational_intensity_flop_per_byte": 6.5841,
    "bottleneck_operational_intensity_flop_per_byte": 6.5895,
    "roofline_efficiency_percent": "0.00%",
    "compute_efficiency_percent": "0.00%",
    "max_mem_bw_utilization_percent": "0.00%",
    "optimal_flop_rate_gflops": 10793.57,
    "dma_stall_percent": "0.00%",
    "measured_flop_rate_gflops": 0.14,
    "model_flop_rate_gflops": 0.02,
    "measured_memory_bw_gibs": 0.02,
    "hbm_bw_gibs": 0.02,
    "hbm_bw_utilization_percent": "0.00%",
    "cmem_read_bw_utilization_percent": "N/A",
    "cmem_write_bw_utilization_percent": "N/A",
    "vmem_read_bw_utilization_percent": "0.00%",
    "vmem_write_bw_utilization_percent": "0.00%",
    "total_time_ms": 23.254,
    "flops_provenance": "opaque_custom_call",
    "metric_semantics": {
      "measured_memory_bw_gibs": "Total bytes across ALL memory spaces (HBM + CMEM + VMEM + bytes with no memory space attributed, such as SparseCore) divided by total op time. It is NOT an HBM figure and may legitimately exceed peak_hbm_bw_gibs. Compare hbm_bw_gibs against peak_hbm_bw_gibs instead.",
      "max_mem_bw_utilization_percent": "Maximum over the per-memory-space utilizations (hbm_bw/peak_hbm_bw, cmem_read, cmem_write, vmem_read, vmem_write). It is NOT measured_memory_bw_gibs divided by peak_hbm_bw_gibs.",
      "operational_intensity_flop_per_byte": "FLOPs (bf16-normalized) divided by bytes across ALL memory spaces. Because the denominator includes non-HBM bytes it is always <= bottleneck_operational_intensity_flop_per_byte, and comparing it against a single-space ridge point overstates memory boundness. Recomputed values convert GiB/s to GB/s first, so they are FLOPs per byte like the backend value.",
      "bound_by": "Argmax over resource utilizations (Compute wins ties). Verify it with bottleneck_operational_intensity_flop_per_byte against bound_by_ridge_point_flop_per_byte, not with operational_intensity_flop_per_byte.",
      "flops_provenance": "Source of the program FLOPs: 'xla_cost_model' (XLA static cost model), 'derived_from_shapes' (no cost-model FLOPs remain; all FLOPs come from shape-derived custom calls), 'mixed' (shape-derived custom call FLOPs plus cost-model FLOPs for the other ops, or plus opaque custom calls that count as 0; see derived_flops_fraction and has_opaque_custom_calls), or 'opaque_custom_call'.",
      "derived_flops_fraction": "Share of program FLOPs that is shape-derived (0.0 to 1.0). 0.0 when no custom call FLOPs were derived from shapes.",
      "has_opaque_custom_calls": "True if any analyzed op (not only top_operations) is a custom call whose FLOPs could not be derived."
    },
    "has_opaque_custom_calls": true,
    "derived_flops_fraction": 0.0
  },
  "device_info": {
    "cmem_read_ridge_point": 0.0,
    "cmem_write_ridge_point": 0.0,
    "device_type": "TPU v6 Lite",
    "has_cmem": 0.0,
    "has_merged_vmem": 1.0,
    "hbm_ridge_point": 577.963,
    "megacore": 0.0,
    "peak_cmem_read_bw_gibs": 0.0,
    "peak_cmem_write_bw_gibs": 0.0,
    "peak_flop_rate": 946700.0,
    "peak_hbm_bw_gibs": 1525.5,
    "peak_vmem_read_bw_gibs": 21696.1,
    "peak_vmem_write_bw_gibs": 15020.4,
    "time_scale_multiplier": 1.0,
    "vmem_read_ridge_point": 40.6379,
    "vmem_write_ridge_point": 58.699
  },
  "total_operations_analyzed": 73,
  "custom_call_share_pct": 1.0,
  "custom_call_warning": "CustomCalls account for 1.00% of device self-time. XLA assigns 0 FLOPs to opaque custom calls (FLOPs are invisible to the XLA roofline model, not zero). Do NOT interpret 0 FLOP/s as idle compute; use `get_kernel_utilization` or `get_llo_static_analysis` to inspect Pallas/CustomCall hardware utilization and static schedule efficiency."
}
```

- **Tool verdict (measured):** `bound_by: "HBM"`, `operational_intensity_flop_per_byte: 6.5841` vs `hbm_ridge_point: 577.963` FLOP/byte. The intensity sits far below the ridge, so *within the 0.48 ms the ops ran*, the roofline model classes them as memory-side. But every utilization the model reports is zero to two decimals: `compute_efficiency_percent: "0.00%"` (measured 0.14 GFLOP/s of a 946700 GFLOP/s peak), `hbm_bw_utilization_percent: "0.00%"` (`hbm_bw_gibs: 0.02` of `peak_hbm_bw_gibs: 1525.5`), `mxu_utilization_percent: "0.0%"`. `bound_by` is an argmax over utilizations that are all ~0, so it is not evidence that bandwidth is limiting anything.
- **Inferred: the run is overhead/launch-bound, not compute- or memory-bound.** The evidence, all measured above: (1) 95.8% of the window is gaps between the 363 program launches (median gap 57 us, versus a median program duration of 0.46 us -- `aggregate_xplane_events` p50 for `XLA Modules` is 455,078 ps); (2) a further 2.2% is per-program time outside any op; (3) only 2.0% is op execution, and the ops that dominate it are ~1 us launches on 512-byte tensors (`f32[4,16]`) and ~0.9 us scalar `X64Split*` custom calls, i.e. fixed per-op cost, not data movement (the top fusion moves 25.6 KB at 0.32 GB/s against a 1.6 TB/s HBM); (4) the actual MLP matmuls -- four `convolution` fusions with `operand_precision={highest,highest}` -- total 5.2 us, 0.02% of the window, with `compute_efficiency_percent` 0.03-0.11% each.
- Because `/host:CPU` has no events, the trace cannot show *what* the host was doing during the gaps (Python, torch_tpu op lowering/compilation-cache lookup, argument transfer). That the gaps are host-side dispatch is an inference from their regularity (one per PyTorch op, ~57 us each) and from the absence of any device work in them.

## 4. Fields requested by the caller (copied as the tools printed them)

| field | value | source tool | note |
|---|---|---|---|
| duty_cycle_percent | `2.0%` | get_kpi_metrics | TensorCore op time / window. |
| mxu_utilization_percent | `0.0%` | get_kpi_metrics | `flops_provenance: opaque_custom_call`; five `X64Split*` custom calls (48.58% of op time) carry no cost model, so this is a lower bound. They are scalar splits, so the bound is not loose in practice. |
| roofline_utilization | `0.00%` | get_kpi_metrics | same provenance caveat; `get_roofline_model.roofline_efficiency_percent` = `0.00%`. |
| peak_hbm_gib | `-1.0` | get_kpi_metrics | `-1.0` is the tool's unavailable sentinel. `get_memory_profile` likewise returns `-1.0` for every field (stderr: "Can not load hlo proto from options"). `get_peak_allocations` sums buffer sizes per module (largest `tt_jit_stack`: 3.208 MiB), which is not a peak-HBM measurement. |
| tc_idle_ms_average | `0.00` | get_overview | per-step average over zero steps (no step markers); not a measurement of idle. Use `device_idle_time_percent = 98.0%`. |
| sc_idle_ms_average | N/A | none | no tool reports a SparseCore idle figure for this trace; no SparseCore plane is present. |
| tc_infeed_ms_average | `0.00` | get_overview | per-step average over zero steps; no infeed ops appear in the op table. |

## 5. Hot ops: provenance, fusion and call shapes

One row per op in the top-20 list; every custom call with device time is in it. `source` is `empty in trace` everywhere because the HLO metadata carries no file:line and `get_top_hlo_ops` returns no `stack_frame`. Fusion membership and the fused ops come from `get_hlo_text` of each module (saved under `hlo_text/`). `call_shapes` are copied from the HLO instruction text, layout suffix included.

| # | trace name | module | HLO instruction (as printed) | source | per call (us) | fusion | fused with | call shapes |
|---|---|---|---|---|---|---|---|---|
| 1 | `broadcast_multiply_fusion` | `tt_jit_mul(350907899631732211)` | `broadcast_multiply_fusion` | empty in trace | 1.5068 | broadcast_multiply_fusion (kind=kLoop, ROOT of tt_jit_mul ENTRY) | mul.6 = broadcast(param_1.1); mul.2 = multiply(param_0, mul.6) |  |
| 2 | `broadcast_add_fusion` | `tt_jit_add(2232073991294053191)` | `broadcast_add_fusion` | empty in trace | 0.988 | broadcast_add_fusion (kind=kLoop, ROOT of tt_jit_add ENTRY) | add.12 = broadcast(param_1.1); add.11 = add(param_0, add.12) |  |
| 3 | `broadcast_subtract_fusion` | `tt_jit_sub(6798965666565663463)` | `broadcast_subtract_fusion` | empty in trace | 0.9631 | broadcast_subtract_fusion (kind=kLoop, ROOT of tt_jit_sub ENTRY) | sub.6 = broadcast(param_1.1); sub.2 = subtract(param_0, sub.6) |  |
| 4 | `custom-call` | `tt_jit_sub(6798965666565663463)` | `%custom-call = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target="X64SplitLow"` | empty in trace | 0.927 | null (not fused) | [] | operands ['s64[]{:T(128)}'] -> outputs ['u32[]{:T(128)}'] |
| 5 | `custom-call` | `tt_jit_add(2232073991294053191)` | `%custom-call = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target="X64SplitLow"` | empty in trace | 0.928 | null (not fused) | [] | operands ['s64[]{:T(128)}'] -> outputs ['u32[]{:T(128)}'] |
| 6 | `custom-call.1` | `tt_jit_sub(6798965666565663463)` | `%custom-call.1 = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target="X64SplitHigh"` | empty in trace | 0.9097 | null (not fused) | [] | operands ['s64[]{:T(128)}'] -> outputs ['u32[]{:T(128)}'] |
| 7 | `custom-call` | `tt_jit_mul(350907899631732211)` | `%custom-call = f32[]{:T(128)} custom-call(f64[]{:T(128)} %Arg_1.0), custom_call_target="X64SplitLow"` | empty in trace | 0.9165 | null (not fused) | [] | operands ['f64[]{:T(128)}'] -> outputs ['f32[]{:T(128)}'] |
| 8 | `custom-call.1` | `tt_jit_add(2232073991294053191)` | `%custom-call.1 = u32[]{:T(128)} custom-call(s64[]{:T(128)} %Arg_1.0), custom_call_target="X64SplitHigh"` | empty in trace | 0.9116 | null (not fused) | [] | operands ['s64[]{:T(128)}'] -> outputs ['u32[]{:T(128)}'] |
| 9 | `cat.1` | `tt_jit_cat(8834897215032019949)` | `cat.1` | empty in trace | 1.7375 | null (not fused) | [] |  |
| 10 | `fusion` | `tt_jit_linear_backward(12749327565423576208)` | `fusion` | empty in trace | 1.3913 | fusion (kind=kOutput, tt_jit_linear_backward) | fusion.1 = fusion(param_0) calls=bitcast_fusion; fusion.2 = fusion(param_1) calls=bitcast_fusion.1; convolution.1 = convolution(fusion.1, fusion.2), dim_labels=bf_io->bf, operand_precision={highest,highest} |  |
| 11 | `convolution_add_fusion` | `tt_jit_linear(2232135972950488025)` | `convolution_add_fusion` | empty in trace | 1.3662 | convolution_add_fusion (kind=kOutput, tt_jit_linear) | fusion = fusion(param_0.2) calls=bitcast_fusion; fusion.1 = fusion(param_1.2) calls=bitcast_fusion.1; convolution.1 = convolution(fusion, fusion.1), dim_labels=bf_oi->bf, operand_precision={highest,highest}; broadcast.0 = broadcast(param_2.1), dimensions={1}; linear.0 = add(convolution.1, broadcast.0) |  |
| 12 | `broadcast_multiply_fusion` | `tt_jit_mul(9558768019819878584)` | `broadcast_multiply_fusion` | empty in trace | 1.28 | broadcast_multiply_fusion (kind=kLoop, ROOT of tt_jit_mul(9558768019819878584) ENTRY) | broadcast.0 = broadcast(param_1.1), dimensions={0}; mul.0 = multiply(param_0, broadcast.0) |  |
| 13 | `copy.18` | `tt_jit_cat(8834897215032019949)` | `copy.18` | empty in trace | 1.2588 | null (not fused) | [] |  |
| 14 | `convolution_add_fusion` | `tt_jit_linear(2454675186444914655)` | `convolution_add_fusion` | empty in trace | 1.2563 | convolution_add_fusion (kind=kOutput, tt_jit_linear) | fusion = fusion(param_0.2) calls=bitcast_fusion; fusion.1 = fusion(param_1.2) calls=bitcast_fusion.1; convolution.1 = convolution(fusion, fusion.1), dim_labels=bf_oi->bf, operand_precision={highest,highest}; broadcast.0 = broadcast(param_2.1), dimensions={1}; linear.0 = add(convolution.1, broadcast.0) |  |
| 15 | `copy.1` | `tt_jit_mul(9558768019819878584)` | `copy.1` | empty in trace | 1.215 | null (not fused) | [] |  |
| 16 | `fusion` | `tt_jit_linear_backward(14565005824055855247)` | `fusion` | empty in trace | 1.2125 | fusion (kind=kOutput, tt_jit_linear_backward) | fusion.1 = fusion(param_0) calls=bitcast_fusion; fusion.2 = fusion(param_1) calls=bitcast_fusion.1; convolution.1 = convolution(fusion.1, fusion.2), dim_labels=bf_io->bf, operand_precision={highest,highest} |  |
| 17 | `as_strided.1` | `tt_jit_mul(9558768019819878584)` | `as_strided.1` | empty in trace | 1.2025 | null (not fused) | [] |  |
| 18 | `copy.3` | `tt_jit_cat(8834897215032019949)` | `copy.3` | empty in trace | 1.135 | null (not fused) | [] |  |
| 19 | `copy.21` | `tt_jit_cat(8834897215032019949)` | `copy.21` | empty in trace | 1.1287 | null (not fused) | [] |  |
| 20 | `subtract_multiply_fusion` | `tt_jit_tanh_backward(7455261703457177013)` | `subtract_multiply_fusion` | empty in trace | 1.1162 | subtract_multiply_fusion (kind=kLoop, ROOT of tt_jit_tanh_backward ENTRY) | tanh_backward.3 = broadcast(constant.0 = 1); tanh_backward.2 = multiply(param_1.2, param_1.2); tanh_backward.1 = subtract(tanh_backward.3, tanh_backward.2); tanh_backward.0 = multiply(param_0, tanh_backward.1) |  |

Notes on the rows: the `X64Split*` custom calls exist because the Python `float`/`int` scalars that Captum multiplies and adds (`alpha`, step counts) arrive on the device as `f64[]`/`s64[]` and torch_tpu/XLA split them into 32-bit halves; in `tt_jit_add` and `tt_jit_sub` the split is followed by a ~40-instruction scalar chain (count-leading-zeros, shifts, selects in memory space `S(6)`) that rebuilds an `f32` from the `s64` -- those scalar instructions are not traced as ops, which is why those two programs show 3.1 us per launch of `module - ops` while `tt_jit_mul` (f64 input, one split) shows 0.02 us. The 50 `copy.N` ops in `tt_jit_cat` are layout changes from `{1,0:T(4,128)}` to `{0,1:T(8,128)}` XLA inserts so the 50 `[4,16]` interpolants can be concatenated.

## 6. What the tools could not measure

- Host activity: `/host:CPU` plane has 0 events. The 22.3 ms of inter-program gaps cannot be attributed to a specific host function from this trace.
- Steps: no step markers, so `get_step_trace` (NO_DATA), `check_host_boundness` (INSUFFICIENT_DATA) and all `*_ms_average` fields are empty/zero.
- Source lines: every `source_file`/`source_line` is empty; no stack frames.
- Peak HBM: `peak_hbm_gib = -1.0`, `get_memory_profile` all `-1.0`.
- LLO / bundle-level data: `LLO_DATA_ABSENT` (`required_flags: ["--xla_xprof_register_llo_debug_info=true"]`).
- Hardware-counter utilization: `get_kernel_utilization` -> `INTERNAL_ERROR` "no data returned"; `capture_valid_for_counter_sampling: false`. The single counter snapshot does exist, and is consistent with the op-level picture, but XProf does not treat it as a valid utilization measurement: `vf_chip_tc_tcs_tc_misc_tcs_stats_tcs_stats_counters_unprivileged_count_cycles` = 325797574 cycles, `vf_chip_tc_tcs_tc_misc_tcs_stats_tcs_stats_counters_unprivileged_count_mxu_busy_0` (cycles with no MXU executing a matmul) = 325797391 -- i.e. the MXUs were busy in 183 of 325797574 counted cycles.
- FLOPs/bytes of the `X64Split*` custom calls: `opaque_custom_call`; XProf assigns them 0 FLOPs and 0 bytes (invisible, not zero). Their time is measured (231.367 us).
- Anything about the other three chips of the v6e-4: not traced.

## 7. Levers, ranked by share of the profiled window

Phase is `mixed` (this is not a serving workload). Shares are **derived** from measured durations as described; the tool that produced each duration is named. Recommendations are **inferred** and are suggestions for the caller to weigh, not changes made here.

| rank | kind | share of window | what the cost is (measured) | source tool(s) | next step (inferred) |
|---|---|---|---|---|---|
| 1 | host-critical-path | 95.8% | 22273.0 us in 362 gaps between 363 program launches; median gap 56.961 us, max 465.58 us. Device idle, waiting for the next program. | list_xplane_events (gaps), aggregate_xplane_events (module count) | Cut the number of device programs per IG call: build the 50 interpolants with one batched tensor expression (alphas as a `[50,1]` device tensor) instead of 150 scalar ops; avoid the 200 `ones_like` by passing a single `grad_outputs` tensor; or trace/compile the whole attribution so it becomes a few programs. Then re-profile with the Python tracer on so the host side is visible. |
| 2 | other | 2.2% | 504.6 us inside programs but outside any traced op: ~3.1 us per `tt_jit_add`/`tt_jit_sub` launch (scalar s64->f32 reconstruction chain after `X64Split*`), 106.6 us in `tt_jit_stack`, 84 us in 200 `tt_jit_ones_like` launches with no traced op, ~53 us in `tt_jit_cat`. | aggregate_xplane_events (XLA Modules), get_kernel_stats (XLA Ops), get_hlo_text (scalar chain) | Keep the interpolation scalars as device `f32` tensors rather than Python `int`/`float` so no 64-bit split or reconstruction is emitted; fewer launches removes the per-program entry/exit. |
| 3 | other | 0.99% | `X64SplitLow`/`X64SplitHigh` custom calls: 231.367 us, 48.58% of op time, 302 occurrences at ~0.92 us each; 0 FLOPs/0 bytes in the cost model (opaque). | get_kernel_stats, get_hlo_stats | Same as 2: pass scalars as 32-bit device tensors; the custom calls disappear with the 64-bit operands. |
| 4 | other | 0.75% | 151 `[4,16]` broadcast loop fusions (mul/add/sub) totalling 173.86 us, ~1.0-1.5 us each, moving 25.6 KB each at 0.3-0.5 GB/s. Fixed launch cost, not bandwidth. | get_hlo_stats, get_roofline_model | Vectorise over the 50 steps so one fusion of `[200,16]` replaces 150 of `[4,16]`. |
| 5 | other | 0.25% | data-formatting copies: 53 ops, 57.8 us; 50 of them are `{1,0:T(4,128)} -> {0,1:T(8,128)}` layout changes feeding `cat.1`. | get_hlo_op_profile, get_hlo_text | Disappears if the interpolants are produced directly as one `[200,16]` tensor. |
| 6 | kernel-block | 0.022% | the MLP itself: four convolution fusions (`convolution_add_fusion` x2 forward, `fusion` x2 backward), 5.23 us total, `compute_efficiency_percent` 0.03-0.11%, `operand_precision={highest,highest}`. | get_hlo_stats, get_roofline_model | Nothing to gain here at this size; the matmuls are 0.02% of the window. |

`exposed-collective`, `sharding-overhead`, `load-imbalance`, `recompilation-padding`, `scheduler-batching`, `memory-capacity`: no evidence in this single-device trace (no collectives, one chip, no recompiles visible, no memory data).

## 8. Tool inventory

| tool | artifact | outcome |
|---|---|---|
| `get_hosts` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_hosts.json` | ok |
| `check_kernel_profiling` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/check_kernel_profiling.json` | ok |
| `get_overview` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_overview.json` | ok |
| `get_kpi_metrics` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_kpi_metrics.json` | ok |
| `get_hlo_stats --limit=0` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_hlo_stats.json` | ok, 77 records |
| `get_top_hlo_ops --limit=20` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_top_hlo_ops.json` | ok |
| `get_roofline_model --top_n=20` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_roofline_model.json` | ok |
| `get_kernel_stats --include_summary=True --limit=20` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_kernel_stats.json` | ok |
| `get_device_time_attribution` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_device_time_attribution.json` | ok |
| `check_host_boundness` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/check_host_boundness.json` | INSUFFICIENT_DATA |
| `get_hlo_op_profile --view=category` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_hlo_op_profile_category.json` | ok |
| `get_profile_summary` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_profile_summary.json` | ok |
| `get_step_trace` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_step_trace.json` | NO_DATA |
| `list_hlo_modules` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/list_hlo_modules.json` | ok, 16 modules |
| `get_device_information` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_device_information.json` | ok |
| `get_kernel_utilization` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_kernel_utilization.json` | INTERNAL_ERROR (no counter data) |
| `get_memory_profile` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_memory_profile.json` | all -1.0 |
| `get_peak_allocations` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_peak_allocations.json` | ok (buffer sizes only) |
| `get_llo_analysis` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_llo_analysis.json` | UNAVAILABLE / LLO_DATA_ABSENT |
| `aggregate_xplane_events --group_by=line` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/aggregate_xplane_events_by_line.json` | ok |
| `aggregate_xplane_events --group_by=line,event` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/aggregate_xplane_events_by_line_event.spill.json (+ ops summary)` | ok, spilled (16.5 MB) |
| `get_perf_counters` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/get_perf_counters.json, get_perf_counters_mxu.json` | ok, one snapshot |
| `list_xplane_events (XLA Modules)` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/list_xplane_events_modules.json` | ok, 363 events |
| `get_hlo_text x10` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/hlo_text/*.hlo.txt` | ok |
| `query_events_db` | `/Users/joinal/.tpu-agent-suite/runs/20261101T150739Z-0ceef7d42aee/artifacts/tpu-diagnostics/query_events_db_span.json, query_events_db_span.result.json` | ok |

Measured with xprof 2.24.2. Report generated from the saved tool outputs; nothing was re-run on the TPU.
