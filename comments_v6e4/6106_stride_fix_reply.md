Thanks, confirmed. Forcing `contiguous_format` was wrong for dense non-contiguous arguments; your `torch.empty_like(arg)` with the default `preserve_format` plus the unchanged `.to(device)` copy is what de1abb50c does now, and your four cases are in `test_cpu_argument_strides_are_preserved`.

On a v6e-1 (libtpu 0.0.51, wheel from this branch, compile caches off): the original repro cases and your four cases all match CPU (8/8), `tests/compile/split_compile_test.py` 18 passed, `bazel test //tests/compile:split_compile_test` PASSED. PR description updated.
