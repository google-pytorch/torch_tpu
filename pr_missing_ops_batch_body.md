## Description

Implements eight ATen ops that were unregistered for TPU, each as a direct StableHLO lowering:

- `smooth_l1_loss.out`, `smooth_l1_loss_backward.grad_input`, `huber_loss`, `huber_loss.out`, `huber_loss_backward.out`: one piecewise-loss builder (`csrc/ops/smooth_l1_loss/`) handles both losses. Forward reduces with `none`/`sum`/`mean` in the accumulate dtype; backward uses the clamped derivative and scales `mean` by `1 / numel` in at least f32, like `mse_loss_backward`. `beta == 0` is L1 loss, and the ATen error messages for negative `beta` / non-positive `delta` are preserved.
- `logaddexp.out`, `logaddexp2.out` (`csrc/ops/logaddexp/`): `max(a, b) + log1p(exp(-|a - b|))` (base 2: `log1p(exp2(-|a - b|)) / ln 2`), with the same-infinity case selected explicitly so `logaddexp(inf, inf)` is `inf` instead of NaN. Integral inputs promote to float through `BinaryOpOut`.
- `nansum`, `nansum.out` (`csrc/ops/nansum/`): `select(isnan(x), 0, x)` followed by the shared sum reduction; integral inputs reduce directly and promote to int64 like `sum`.
- `std_mean.correction` (`csrc/ops/reductions/std_aten_kernels.cc`): a two-output dispatch mirroring `var_mean`, with `sqrt` applied in-graph; BF16/F16 compute in F32.
- `trace` (`csrc/ops/trace/`): masks the input with `iota(0) == iota(1)` and reduces over both dimensions; integral inputs accumulate in int64 and the `expected a matrix` error matches ATen.
- `addbmm`, `addbmm.out`, `addbmm_` (`csrc/ops/addbmm/`): `beta * self + alpha * sum_b(batch1[b] @ batch2[b])` built from the existing `bmm` lowering plus a batch-dim sum. `beta == 0` leaves `self` out of the graph (ATen ignores a NaN `self` in that case) and an empty batch dimension reduces to `beta * self`; both host-side decisions are added to the cache keys.

Also adds the `OpName` entries, the `ImplStable` registrations, and removes the ops from `UNREGISTERED_OPS` in `tests/native_functions_data.py`.

The probe that found these ran every op in `UNREGISTERED_OPS` with a plausible sample input on 0.2.0; this batch covers the ones that are common in loss functions and in model introspection code (Captum's attribution methods hit several of them).

## Test Plan

New unit tests in `tests/ops_unit_test.py`, run on a Cloud TPU v6e-1 against a wheel built from this branch (`bazel build --config=wheel_common //ci/wheel:torch_tpu_wheel`):

```
PYTHONPATH=. python -m pytest tests/ops_unit_test.py -k "piecewise or smooth_l1 or logaddexp or nansum or std_mean or trace or addbmm"
53 passed
```

- `test_piecewise_l1_l2_loss` (loss × reduction × threshold × dtype, forward and backward), `test_smooth_l1_loss_beta_zero_is_l1`, `test_piecewise_loss_broadcast_and_out`, `test_piecewise_loss_invalid_threshold`
- `test_logaddexp` (logaddexp, logaddexp2: infinities, `out=`, broadcasting)
- `test_nansum`, `test_std_mean`, `test_trace` (incl. backward and the dim-check error), `test_addbmm` (incl. backward, `out=`, in-place, empty batch, the 3D-check error)

A probe over `UNREGISTERED_OPS` confirms each op now dispatches to TPU instead of raising `not implemented for TPU`.

`tests/native_functions_test.py` passes with the updated `UNREGISTERED_OPS`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
