Approach on the branch, for review before the PR slot frees:

**Direct StableHLO lowerings only.** The first version of `nansum`, `trace`, `std_mean` and `addbmm` called other ATen ops from inside the kernel. That builds, but `OpNameStack` aborts at runtime with `CompositeOpCheck: Op trace is decomposed into as_strided`, and `known_composite_ops` is capped with a "do not add" note. So every op is built from `mlir::stablehlo` and the existing shared builders (`BuildSumShlo`, `BuildMeanShlo`, `BuildBmmShlo`, `BuildIsNanShlo`, `BuildSqrtShlo`, `UnaryOp`/`UnaryOpOut`, `DispatchOp`).

Per op:

- **`smooth_l1_loss` / `huber_loss` (+ backward)**: one piecewise builder in `csrc/ops/smooth_l1_loss/`, parametrised by threshold (`beta` or `delta`) and by which branch is quadratic, since the two losses are the same function up to a scale. Forward reduces `none`/`sum`/`mean` in the accumulate dtype; backward applies the clamped derivative and divides by `numel` for `mean` in at least f32, the way `mse_loss_backward` already does here. `beta == 0` degenerates to L1; the ATen error strings for negative `beta` and non-positive `delta` are kept verbatim.
- **`logaddexp` / `logaddexp2`**: `max(a, b) + log1p(exp(-|a - b|))`, base 2 via `exp2` and `/ ln 2`. The same-sign-infinity case is `Select`ed explicitly so `logaddexp(inf, inf)` is `inf` rather than `inf - inf = NaN`. Integral inputs promote through `BinaryOpOut` like the CPU kernels.
- **`nansum`**: `Select(isnan(x), 0, x)` then the shared sum reduction. Integral and bool inputs skip the mask and reduce to int64 like `sum`; complex is rejected with the ATen message. `dtype=` overrides the output type.
- **`std_mean.correction`**: mirrors `var_mean`'s two-output `DispatchOp<2, 2>`; `sqrt` is applied in-graph on the variance. BF16/F16 compute in F32 and cast back; complex inputs return `(real std, complex mean)` as on CPU.
- **`trace`**: `Select(iota(0) == iota(1), x, 0)` then a sum over both dims with `kDropDims`. Integral inputs accumulate in int64; Bool/Half/BFloat16 raise `NotImplementedError` like CPU; the `expected a matrix, but got tensor with dim N` message matches ATen.
- **`addbmm` / `addbmm.out` / `addbmm_`**: `beta * self + alpha * sum_b(bmm(batch1, batch2))` on top of the existing `bmm` lowering. Two host-side decisions go into the cache key via `param_keys.SetParam`: `beta_is_zero` (ATen ignores a NaN `self` then, so `self` is left out of the graph) and `has_batches` (an empty batch dim reduces to `beta * self`). Matmul precision comes from `GetAndAddPrecisionTo`, so `set_float32_matmul_precision` is honoured.

Two small things reviewers may want to know: `IgnoreInCacheKey` rejects `PromotedScalar` because scalars are already excluded from keys, so `beta`/`alpha` are passed as tensors; and the matmul-based tests set `set_float32_matmul_precision("highest")` around the CPU comparison, otherwise `addbmm` differs from CPU at 1e-2.

Tests: 53 new cases in `tests/ops_unit_test.py`, all passing on a v6e-1 against a wheel built from the branch. The ops are removed from `UNREGISTERED_OPS`.

I would like to own this one. Non-collaborators cannot self-assign, so if a maintainer can assign me, I will keep the branch rebased and open the PR the moment a slot frees.
