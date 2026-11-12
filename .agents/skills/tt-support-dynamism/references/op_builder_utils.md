# Bounded-Dynamism `op_builder_utils` Utilities

Reference for the `tt-support-dynamism` skill. Check
`csrc/ops/op_builder_utils.h` before writing any lowering code, and read each
utility's header comment for its full contract: the header is the source of
truth, and this table is not exhaustive.

Utility                                                     | When to Use
----------------------------------------------------------- | -----------
`MakeScalarConstant` / `MakeConstantLike`                   | Scalar constants or constants matching a static/bounded input shape (never pass dynamic shapes to `MakeConstant`).
`ApplyBroadcastIfNeeded` / `BroadcastIfNeeded`              | NumPy or explicit-dimension broadcasting across static and multi-dim bounded operands.
`Iota`                                                      | Sequential index tensors matching a reference op, propagating dynamic bounds and runtime sizes.
`Unsqueeze` / `Squeeze` / `Flatten`                         | Inserting/removing size-1 dims or collapsing all dims with bounded support.
`ReshapeToStaticDimensions` / `ReshapeFromStaticDimensions` | Collapse/expand reshapes (plus moving size-1 dims) that propagate bounds from upper-bound static shapes. Not for permutations; a bounded dim cannot expand into multiple non-1 dims.
`GetNumElements`                                            | Element counts over all or selected dims (a constant when static, `GetDimensionSize` products when bounded).
`GetDimensions` / `GetStaticShape`                          | Inspecting `stablehlo::Dimensions` (with `boundOp`) or extracting upper-bound static shapes.
`RngBitGeneratorLike`                                       | Random bits at upper-bound shapes with runtime sizes restored on bounded dims (needs `shape_reference` for dynamic shapes).
`ConvolutionLike`                                           | Convolutions when XLA rejects bounded spatial dims (zeroes elements past runtime sizes and re-bounds outputs); some bounded configurations return `NotImplementedError`.

Not every helper in `op_builder_utils.h` is bounded-aware: some still query
static dim sizes (e.g., `SliceBatchDimensions`). Verify before reusing one on a
bounded operand.

## Adding a New Utility

*   **Static-result StableHLO ops** (like `Iota` or `RngBitGeneratorLike`): emit
    the op at `GetStaticShape(...)` and attach runtime sizes with
    `stablehlo::GetDimensionSize` + `stablehlo::SetDimensionSize` (see the
    file-local helpers in `op_builder_utils.cc`). XLA's `DynamicPadder` handles
    out-of-bounds padding once dynamic dimensions are preserved, so do **not**
    mask out-of-bounds elements manually.
*   **XLA rejects bounded inputs** for an op that reads or accumulates across
    those dimensions (like `ConvolutionLike`): strip the inputs to static upper
    bounds, zero the elements past runtime sizes, then re-bound the result.
*   Helpers in the anonymous namespace of `op_builder_utils.cc` are only for
    building new utilities in that file; never call them from op files.
*   Cover every new utility in `csrc/ops/op_builder_utils_test.cc`.
