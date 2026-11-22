Built and tested branch [`logaddexp`](https://github.com/joinalahmed/torch_tpu/tree/logaddexp) (head ad18185ce) on a v6e-1.

Two fixes relative to the first push:
- The stablehlo builder functions take `MlirOp&`, so nested temporaries did not compile. Every intermediate is now a named lvalue.
- The kernel macro prefixes error messages with `logaddexp(): `; the errors test expects that, matching the neighbouring tests.

Results:
- `torch.logaddexp` on TPU matches CPU for 0, ±1, ±100, ±inf, mixed inf and nan (`torch.allclose(..., equal_nan=True)` is True).
- Integer inputs raise `NotImplementedError: logaddexp(): "logaddexp" not implemented for 'int64'`, as ATen does.
- `bazel test //tests:ops_unit_test_tpu-v6e //tests:errors_test_tpu //tests:ops_test_tpu-v6e --test_filter=*logaddexp*`: all three PASSED.

I will open the PR as soon as one of my three open PRs closes.
