---
name: tt-implement-ops
description: >-
  Guides the implementation of new ATen operators in TorchTPU, including StableHLO lowering, C++ registration, and testing.
  Use when implementing a new operator, adding support for an existing ATen operator, or writing StableHLO lowerings.
---

# TorchTPU Op Implementation

This skill guides you through the process of implementing a new ATen operator in
TorchTPU.

## Workflow

### 1. Implement Op Lowering

*   Implement the StableHLO lowering in `csrc/ops/` using the `TT_KERNEL` macro.
*   See [Typical File Locations](references/file_locations.md) for files you may
    need to modify.
*   Create a `BUILD` file in the specific operator's directory (e.g.,
    `csrc/ops/{op_name}/BUILD`). See
    [BUILD Template](references/build_template.md) for the required structure.
*   Keep the ATen API minimal. Separate complex StableHLO lowerings in an
    anonymous namespace.
*   Put StableHLO lowering helpers in the anonymous namespace of the kernel
    `.cc` by default. Only factor them into a separate builder library
    (`{op_name}/{op_name}.{h,cc}` with its own `torch_tpu_cc_library`) when a
    second user needs them, such as another op's kernel or builder, a test, or a
    non-eager path.
*   To find available StableHLO and CHLO builder APIs, see
    [StableHLO APIs](references/stablehlo_apis.md).
*   **Crucial Rules**:
    *   **Match CUDA behavior**: Always read the CUDA implementation of the op
        in the [PyTorch source](https://github.com/pytorch/pytorch) (e.g.,
        `aten/src/ATen/native/cuda/`) first and make the TPU implementation
        match its behavior. In particular, check which dtype the CUDA
        implementation uses for computing intermediate results and do the same
        for TPU. Also, ensure that the TPU implementation supports the same
        input dtypes and produces the same result dtype as the CUDA
        implementation.
    *   **Validate inputs before dispatching**: Invalid inputs should be caught
        and reported as informative and readable user errors. Such validation
        should be done on the op input parameters *before* dispatching the op so
        that the errors will reference Python code written by the user. Ensure
        that the op input validation matches the CUDA implementation.
    *   **Format error messages**: Follow
        [tt-write-error-tests](../tt-write-error-tests/SKILL.md#error-message-format)
        when writing user-facing or internal (`ABSL_CHECK`) error checks.
    *   **No `at::Tensors` in MLIR Builder methods**: Builder methods are
        deferred via lambda capture. Capturing an ATen tensor can cause
        deadlocks. Capture only needed fields (e.g., `dtype`, `num_dimensions`).
    *   **Avoid passing shapes**: Rely on StableHLO type inference. Only pass
        shapes if they depend on scalars (e.g., `torch.ones`, reshapes).
    *   **Watch for dtype mismatches** in StableHLO op chains.
    *   Follow [dynamism guidelines](../tt-support-dynamism/SKILL.md) for
        bounded dynamism.
    *   **Do NOT add `(void)param_keys;`**: The `TT_KERNEL` macro handles it;
        adding it to suppress unused variable warnings is incorrect and
        unnecessary.
    *   **Always use `DispatchOpOut` for out-variant and in-place ops**: out-
        variant (`Aten*Out`) and in-place (`Aten*Inplace`) ops **MUST** use
        `DispatchOpOut<kArity, kNumOutputs>` (defined in
        `csrc/eager/op_dispatcher.h`), never `DispatchOp`, and never call
        internal buffer donation functions manually. `DispatchOpOut`
        automatically verifies and configures buffer donation in eager mode to
        reuse output memory on TPU and avoid allocating unnecessary buffers. It
        supports single-output ops (`kNumOutputs = 1`), fixed multi-output ops
        (`kNumOutputs = N`), and dynamic variadic/foreach ops
        (`kNumOutputs = kDynamicSize`). See the out-variant kernel in
        [Minimal Example](references/example_aten_kernels.cc).
    *   **Use dimension types from `csrc/common/dimension_types.h`**: Prefer
        `Dimensions`, `Strides`, and `Indices` over `std::vector<int64_t>` for
        better efficiency (avoids heap allocation for <= 6 elements). Use
        `CopyIntVector(foo)` from `csrc/common/utils.h` rather than
        `(foo.begin(), foo.end())` when copying an integer range into
        `Dimensions`, `Strides`, `Indices`, or `SmallInt64Vector`. This avoids
        repetitive iterator boilerplate and provides uniform conversion across
        integer range types (`at::IntArrayRef`, `c10::DimVector`, `absl::Span`,
        `llvm::ArrayRef`, `std::vector`).
    *   **Use `InferComputationDtype` from `csrc/common/aten_utils.h`**: Always
        use `InferComputationDtype` instead of upstream `at::toOpMathType` to
        determine the computation (opmath) dtype for an `mlir::ElementType` or
        `at::ScalarType` (including when materializing scalar operands via
        `PromotedScalar::GetTensor`). Upstream `at::toOpMathType` crashes with
        an internal assert on unsigned/sub-byte integer types (`UInt16`,
        `UInt32`, `UInt64`, `Int2`, `Int4`, `UInt2`, `UInt4`) and does not
        promote FP4 to F32 for XLA HLO.
    *   **Keep kernel headers minimal**: The kernel header
        (`<op>_aten_kernels.h`) should declare only functions that are
        registered in `csrc/eager/tpu_aten_kernels.cc` or called from another
        translation unit. Put all other helpers in an anonymous namespace in the
        `.cc`, and don't declare functions you don't define.
    *   **Proactive Dependency Analysis**: Analyze header files and proactively
        include all necessary dependencies in `BUILD` files before running
        tests. Avoid iterative trial-and-error failures.
    *   **License Headers**: Use the current year in all new file license
        headers. You can find the current year from your environment metadata or
        by running the command `date +%Y`. In copied/moved/edited files,
        preserve the original copyright year.
    *   **Formatting**: Strictly check for extraneous spaces and use
        Python-style comments (`#`) in `BUILD` files (avoid C-style comments).
*   **TPU Efficiency & Precision**:
    *   Verify choice of StableHLO/CHLO ops for efficient TPU performance (avoid
        unnecessary reshapes).
    *   Avoid intermediate materialization that leads to precision loss.
    *   Ensure careful handling of accumulation types and element type
        conversions.

See [Minimal Example](references/example_aten_kernels.cc) for a reference
implementation.

### 2. Register Operator

*   Register the op in `csrc/eager/tpu_aten_kernels.cc`.
*   Update `csrc/ops/op_names.h` and `csrc/ops/op_names.cc` alphabetically.
*   Update `tests/native_functions_data.py`.
*   Run `bazel test //tests:native_functions_test` to verify registration.

### 3. Testing

*   Follow guidelines in [testing skill](../tt-write-tests/SKILL.md) for TDD and
    test locations.

### 4. Code Health and Formatting

*   Follow [tt-create-change](../tt-create-change/SKILL.md) for formatting
    (`nox -s format`), opening the pull request early, and post-push validation.
