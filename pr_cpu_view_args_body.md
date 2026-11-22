## Description

`torch.compile(backend="tpu")` returned wrong values when a TPU partition
took a view of a CPU tensor as an argument. With `cum = torch.tensor([0, 2])`
on CPU and `x` on TPU, compiled `x * cum[1]` returned zeros and `x + cum[1]`
returned `x`; eager mode and `x * cum[1].item()` were correct.

Cause: `cum[1]` is computed in a CPU partition, so the TPU partition's
example input is a 0-d CPU view with storage offset 1. `placeholder_like`
preserves view layouts, so the executable was compiled with a 2-element
parameter followed by `stablehlo.slice [1:2]`. At run time
`_WrapperModule.forward` copies the argument with `.to(device)`, which
produces a fresh 1-element buffer with storage offset 0, so the executable
read past the buffer and got 0.

Fix: `_SubmodCompiler.compile_submod` compiles TPU partitions against
contiguous, offset-0 metadata for their CPU arguments
(`torch.empty_like(arg, memory_format=torch.contiguous_format)`), and
`_WrapperModule.forward` copies them with `memory_format=torch.contiguous_format`
so the two agree for non-dense views as well.

Found while running Captum `LayerIntegratedGradients` under
`torch.compile(backend="tpu")`: Captum indexes a CPU `int64` tensor of
cumulative output counts inside a hook, which lands exactly in this pattern.

## Test Plan

Environment: Cloud TPU v6e-1, Python 3.14, torch 2.11.0+cpu, wheel built from
`main` with the Python change copied over the installed package. All runs with
`TORCHINDUCTOR_AUTOGRAD_CACHE=0 TORCHINDUCTOR_FX_GRAPH_CACHE=0
TORCH_TPU_TIER2_COMPILATION_CACHE=0`: a cached executable hides the fix (and
the bug) because the AOTAutograd cache key does not depend on the argument's
storage offset.

- New `tests/compile/split_compile_test.py::test_cpu_view_argument_to_tpu_partition`
  (`select`, `float_select`, `strided_slice`, `transpose`): 3 of 4 fail
  before the change (`transpose` passes either way), 4 of 4 pass after.
- Whole `tests/compile/split_compile_test.py`: 17 passed.
- Standalone repro (`x * cum[1]`, `x * cumf[1]`, `x + cum[1]`,
  `x * cum[1].to(x.device)`, 0-d CPU input, `.item()`, CPU index): 4 wrong
  before, 7 of 7 match CPU after.

Assisted-by: Claude Fable 5.1
