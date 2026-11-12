---
name: tt-fix-composite-ops
description: >-
  Guides the migration of composite ATen operators in TorchTPU to native operators
  that lower directly to StableHLO. Use this skill when you need to optimize
  an operator by removing its reliance on other PyTorch operators, thereby
  bypassing the dispatcher overhead and eliminating it from the known composite ops list.
---

# TorchTPU Fixing Composite Ops

This skill guides you through the process of migrating a "composite" ATen
operator in TorchTPU into a "native" operator.

## 0. Context: Composite vs. Native Ops

*   **Native Ops**: Operators that lower directly to StableHLO (SHLO) or CHLO
    computations and dispatch to TPU via `DispatchOp` without calling other ATen
    operators.
*   **Composite Ops**: Operators implemented in TorchTPU by delegating to other
    ATen operators (e.g., calling `to()`, `view()`, `slice()`, or other PyTorch
    operators inside the TPU kernel).
*   **How Composite Ops are Tracked**: TorchTPU uses `OpNameStack` to track the
    call stack of operators. If an op is called while the stack is not empty, it
    indicates decomposition. If the calling op is not in the
    `known_composite_ops` allowlist in `csrc/common/op_name_stack.cc`, the
    system crashes with a `CompositeOpCheck` error. Our goal is to implement all
    operators natively and empty this list.

## 1. Identify Candidate and Trace Composition

1.  The user will specify which operator to fix. Do not choose one arbitrarily
    unless asked.
2.  Locate the operator's implementation in `csrc/ops/` (e.g.,
    `csrc/ops/all_any/all_any_aten_kernels.cc`).
3.  Locate its unit tests. Do not assume they are only in `ops_unit_test.py`:
    search for the operator name (e.g., `_add_relu`) under `tests/` with `rg`.
    See [tt-write-tests](../tt-write-tests/SKILL.md) for the standard test
    files, and watch for specialized ones (e.g. `tests/view_test.py`,
    `tests/distributed/...`).
4.  Trace the composition:
    *   Temporarily remove the candidate operator from the `known_composite_ops`
        list in `csrc/common/op_name_stack.cc` and decrement the expected size
        check.
    *   Run all the located unit tests. For example: `bazel test
        //tests:ops_unit_test --test_filter=test_your_op
        --test_sharding_strategy=disabled`
    *   Observe the crash log. The error message will identify which operator it
        is decomposing into (e.g., decomposing into `kToCopy` because of a
        `.to()` call).

## 2. Design Phase

Before modifying the code, write a design document in Markdown and share it with
the user for review. The design doc must include:

1.  **Explanation of Composition**: Explain exactly how the operator is
    currently composite (e.g., "It calls `result.to(at::kByte)` at the end of
    the computation, which dispatches to `kToCopy`").
2.  **Proposed Native Implementation**:
    *   Explain how you will eliminate the dispatch to sub-operators.
    *   If a cast is needed, propose doing it inside the StableHLO graph using
        `mlir::stablehlo::ConvertElementType` instead of calling
        `at::Tensor::to()`.
    *   If allocation is needed, propose using internal native helpers (see
        section 4).
3.  **Code Snippets**: Include a draft code snippet showing the proposed C++
    changes.
4.  **Wholistic Design**:
    *   Consider the family of operators in the same directory (e.g.,
        `csrc/ops/all_any/`). If one operator in the family is composite, others
        might be too. Design a solution that works for all of them.
    *   Extract common helper code to avoid duplication.
5.  **Human Approval (HARD GATE)**: Do NOT start coding or making any
    implementation changes in the codebase until the user has explicitly
    confirmed (e.g., by saying "approved" or "looks good") that they are happy
    with the design.

## 3. Implementation Phase

1.  Implement the approved design in the operator's C++ file.
2.  **Avoid PyTorch Dispatcher**: Ensure you do not call any ATen operators that
    route through the dispatcher inside the `TT_KERNEL` block. Use internal
    helpers instead (see section 4).
3.  **Remove from Allowlist**:
    *   Remove the operator from the `known_composite_ops` list in
        `csrc/common/op_name_stack.cc`.
    *   Decrement the expected size in the `ABSL_CHECK_EQ` check at the bottom
        of the list.
4.  **Test**:
    *   Run all the located unit tests again. They must pass without triggering
        the `CompositeOpCheck` crash.
    *   If they fail or crash on a different sub-operator, update your design
        and implementation.
5.  **Review**: Share the diff with the user and iterate on feedback. Update the
    design doc if any new implementation details emerge.

## 4. Native Helpers to Use Instead of ATen Ops

To implement operators natively without going through the dispatcher, use these
internal helpers instead of their ATen/PyTorch equivalents:

<!-- mdformat off(reason: standard GFM table rendering on GitHub) -->
PyTorch/ATen Op                  | Native C++ Helper                     | Header File                                                                                                                | Note
:------------------------------- | :------------------------------------ | :------------------------------------------------------------------------------------------------------------------------- | :---
`at::empty`, `at::empty_strided` | `MakeEmptyTensor`                     | `csrc/eager/tensor_to_buffer.h`                                                                                            | Allocates a new TPU tensor.
`at::native::resize_output`      | `ResizeTensorIfShapeDiffers`          | `csrc/ops/resize/resize_aten_kernels.h`                                                                                    | Resizes output tensor if needed.
`tensor.to(dtype)`               | `mlir::stablehlo::ConvertElementType` | [StablehloBuilder.h](https://github.com/openxla/stablehlo/blob/main/stablehlo/integrations/cpp/builder/StablehloBuilder.h) | Do the cast inside the StableHLO graph before compilation.
Copying data to tensor           | `AssignBufferToAtTensor`              | `csrc/eager/tensor_to_buffer.h`                                                                                            | Assigns raw device buffer to ATen tensor.
`dest.copy_(src)`                | `CopyTensor`                          | `csrc/ops/copy_from/copy_from_aten_kernels.h`                                                                              | Copy data natively (CPU<->TPU, TPU<->TPU) bypassing dispatcher.
<!-- mdformat on -->
