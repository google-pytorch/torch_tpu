## Summary

Adds `tests/attr/test_accelerator.py`, a device-generic counterpart of the
`cpu_cuda` mode in `tests/attr/test_data_parallel.py`.

Every config in `captum/testing/attr/helpers/test_config.py` (78 configs) is
run on CPU and on an accelerator, and the attributions (and convergence deltas,
where available) are compared with the config's `dp_delta`. The accelerator is
chosen from, in order:

1. `CAPTUM_TEST_DEVICE` (e.g. `tpu`, `cuda`, `mps`, `xpu`)
2. `torch.accelerator.current_accelerator()`
3. CUDA, if available

Tests are generated only when a non-CPU device is found, so CPU-only CI is
unchanged. `test_data_parallel.py` is left as is; its DataParallel modes are
CUDA-specific.

The class sets `torch.set_float32_matmul_precision("highest")` for its
duration and restores the previous value afterwards. Accelerators that run fp32
matmuls at reduced precision by default (TPU, TF32 on Ampere+) otherwise differ
from CPU by ~1e-3, above `dp_delta`.

Motivation: TorchTPU (`torch_tpu`) exposes Cloud TPUs as
`torch.device("tpu")`. This gives Captum a way to verify all attribution
methods there, and on MPS/XPU, with the existing configs rather than a parallel
test set.

## Test plan

- `ufmt format` / `flake8` clean with the repo configs.
- CPU-only machine: the file imports and generates no tests (class is not
  defined), `pytest tests/attr/test_accelerator.py` collects 0 items.
- Cloud TPU v6e-1, torch 2.15.0, torch_tpu 0.2.0, Captum 0.9.0:
  `CAPTUM_TEST_DEVICE=tpu pytest tests/attr/test_accelerator.py`
  -> 266 tests generated, **262 passed, 4 failed** in 5m25s.
  - `basic_multi_input_dl_shap_DeepLiftShap`: values +6.1e-5 vs -6.1e-5 at a
    near-zero attribution, |diff| 1.2e-4 > dp_delta 1e-4. Precision noise at
    the tolerance boundary, not a method difference.
  - `conv_layer_single_target_LayerDeepLift`, `conv_neuron_NeuronDeepLift`,
    `conv_neuron_slice_selector_NeuronDeepLift`: TorchTPU raises
    `operator 'aten::max_unpool2d' is not implemented for TPU` from DeepLift's
    MaxPool rule. Backend gap, reported as google-pytorch/torch_tpu#6103; the suite
    surfacing it is the point of the change.
- CUDA: not run by me; the generated tests are the same comparisons
  `test_data_parallel.py::cpu_cuda` already performs.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
