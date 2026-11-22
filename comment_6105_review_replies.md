# Draft replies for basioli-k's review on #6105 (not posted)

Head after rework: d4c4bd843 (local only, untested on TPU as of 2026-11-05).

## L109 "Can we try to use BuildScatterShlo from scatter.h"
Done. The lowering now reshapes `self` and `indices` to `[B, S_in]` and calls
`BuildScatterShlo(zeros[B, S_out], /*dim=*/1, index, src, ScatterOp::kReplace, dtype)`;
the hand-built `ScatterDimensionNumbersAttr` and region are gone.

## L116 "Function name makes it unclear that this returns output dimensions as well"
Split into `absl::Status ValidateMaxUnpoolInputs(...)` and
`absl::StatusOr<Dimensions> ComputeOutputDimensions(self, output_size)`.

## L119 "Can we validate input dtypes?"
Added `ValidateNotLowPrecisionFloat(self)` followed by an `IsFloatingPoint(self)`
check that raises NotImplementedError. ATen's max_unpool kernels are only
implemented for floating-point inputs: on CPU (torch 2.11) bool, int32, int64,
uint8, complex64 and float8_e4m3fn all fail with `"max_unpool2d" not implemented
for '...'`, so TPU now matches. New test
`test_max_unpool2d_non_floating_point_input` covers bool, int32 and float8_e4m3fn.

## L143 "use ConcreteSizeToDimensions on output_size"
Done, in `ComputeOutputDimensions`.

## L152 "This can just return absl::Status and we can populate the out tensor"
Done. `MaxUnpoolOut(...) -> absl::Status` writes into `out`; the non-out
kernels allocate `MakeEmptyTensor({0}, dtype, device)` and call it, the same
shape as `binary_cross_entropy`.

## L165 "Is this resizing needed?"
Not on the dispatch path: `DispatchOpOut` resizes `out` to `out_dims` before
assigning the buffer, so that call is removed. It is still needed on the
zero-size path (empty batch or empty output volume), where the kernel assigns
a zero-size buffer directly and `out` may arrive as `torch.empty(0)`; that
path keeps `ResizeTensorIfShapeDiffers`, like `searchsorted` and
`adaptive_avg_pool` do.

## L169 "use NumElements from utils.h"
Done (`NumElements` lives in `csrc/common/error_utils.h`); also used for the
batch and spatial sizes in the lowering.

## L212 / L238 "can this use the out variant?"
Done; both non-out kernels go through `MaxUnpoolOut`.

## Test plan to fill in after the VM run
- `bazel test //tests:ops_test_tpu-v6e --test_filter=TestOps.test_max_unpool*`
- `pytest tests/ops_unit_test.py -k unpool` (expect 25 passed + unchanged)
- `pytest tests/errors_test.py -k max_unpool2d` (expect 5 passed)
