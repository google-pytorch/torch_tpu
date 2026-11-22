## Description

Implements `aten::max_unpool2d`, `aten::max_unpool2d.out`, `aten::max_unpool3d`
and `aten::max_unpool3d.out` for TPU (`csrc/ops/pooling/max_unpool_aten_kernels.{h,cc}`).

Lowering: `self` and `indices` (shape `[batch..., in_spatial...]`) are reshaped
to `[B, S_in]` and scattered into a `[B, S_out]` zero constant with a StableHLO
`ScatterOp` that uses `B` as a batching dimension (`input_batching_dims =
scatter_indices_batching_dims = {0}`, `inserted_window_dims =
scatter_dims_to_operand_dims = {1}`); the update region returns the update
value. The result is reshaped to `[batch..., output_size...]`. The 2-D and 3-D
variants share the helper; `stride` and `padding` of `max_unpool3d` only take
part in validation, as in ATen. `F.max_unpool1d` is implemented in
Python on top of `aten::max_unpool2d` (unsqueeze/squeeze), so all three
functional variants work after this change.

Validation mirrors `max_unpooling{2,3}d_forward_out_cpu`: Long indices,
`output_size` length, 3-D/4-D (4-D/5-D) input, `indices` shaped like `self`,
non-empty non-batch dimensions. An empty batch or an empty output volume
returns an empty tensor. Autograd needs no new kernel: PyTorch's derivative
is a gather at `indices`, which TPU already supports.

Differences from CPU, both documented in the tests:
- Integer dtypes are accepted (CPU raises `"max_unpool2d" not implemented
  for 'Long'`); the scatter is dtype-agnostic.
- Out-of-range indices are not reported. CPU raises `Found an invalid max
  index`; StableHLO scatter drops out-of-bounds updates, so the position is
  left at zero. Checking would need a device-to-host sync.

Motivation: Captum's DeepLift family (`DeepLift`, `DeepLiftShap`,
`LayerDeepLift`, `NeuronDeepLift`) implements its MaxPool rule with
`F.max_unpool{1,2,3}d`, so CNNs with max pooling could not be explained on TPU
(meta-pytorch/captum#1951 surfaced it: 3 of its 266 configs failed on this op).

Fixes #6103.

## Test Plan

Environment: Cloud TPU v6e-1 (us-east5-b), Python 3.14, torch 2.11.0+cpu, wheel
built from this branch with
`bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel --config=no_rbe`.

- `tests/ops_unit_test.py`: new `test_max_unpool2d` (3 shapes x 2 kernel
  sizes x float32/bfloat16/float16, functional and `out=`),
  `test_max_unpool2d_default_output_size`, `test_max_unpool2d_backward`,
  `test_max_unpool2d_empty_batch`, `test_max_unpool3d` (batched and
  unbatched) -> 25 tests, all pass (run as
  `PYTHONPATH=. python tests/ops_unit_test.py OpsUnitTest.test_max_unpool*`
  against the pip-installed wheel).
- `tests/errors_test.py`: four new `max_unpool2d` error-message tests (indices
  dtype, rank, shape mismatch, empty spatial dim) -> 4 pass
  (`--test_mode=tpu`). GPU messages are taken from ATen's source and were not
  run on a GPU.
- `tests/native_functions_test.py` after removing the four entries from
  `UNREGISTERED_OPS` -> no finding for `max_unpool*`. (Outside bazel, against
  torch 2.11's `RegistrationDeclarations.h`, it reports two unrelated
  mismatches, `cholesky`/`cholesky.out` and `_scaled_grouped_mm_v2.out`, that
  come from the torch version, not this change.)
- Repro from #6103 and Captum's three DeepLift conv configs
  (`CAPTUM_TEST_DEVICE=tpu pytest tests/attr/test_accelerator.py -k DeepLift`)
  -> repro returns `[1, 1, 4, 4]` on TPU; Captum: 38 passed, 1 failed. The
  three conv configs that raised `aten::max_unpool2d is not implemented`
  (`conv_layer_single_target_LayerDeepLift`, `conv_neuron_NeuronDeepLift`,
  `conv_neuron_slice_selector_NeuronDeepLift`) now pass and match CPU; the one
  failure (`basic_multi_input_dl_shap_DeepLiftShap`) is a pre-existing 1e-4
  tolerance miss unrelated to this op.
- `ops_test.py` / GPU goldens: not updated (no GPU access);
  `nn.functional.max_unpool2d` OpInfo coverage can be added by a maintainer
  with golden regeneration.

Assisted-by: Claude Fable 5.1
