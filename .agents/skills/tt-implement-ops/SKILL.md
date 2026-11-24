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
        input dtypes, produces the same result dtype, and matches edge-case
        semantics (`±inf`, `NaN`, `-0.0`, `numel == 0`) as the CUDA
        implementation. When following TDD, check CUDA edge cases first and add
        tests in `tests/ops_unit_test.py` for any gaps not covered by
        `tests/ops_test.py`.
    *   **Validate inputs before dispatching and before shortcuts**: Invalid
        inputs should be caught and reported as informative and readable user
        errors. Validate devices and dtypes (e.g., via `GetBaseBuffer`,
        `ConvertTo<mlir::ElementType>`, `ValidateNotLowPrecisionFloat`,
        `ValidateOutDtype`, or explicit `TT_CHECK_THROW`/`TT_RET_CHECK` checks)
        *before* shape-mismatch checks or `numel() == 0` early-return shortcuts,
        and *before* dispatching the op so that errors fail uniformly regardless
        of shape and reference Python code written by the user. Ensure that the
        op input validation matches the CUDA implementation, and add test cases
        in `tests/errors_test.py` (see
        [tt-write-error-tests](../tt-write-error-tests/SKILL.md)) covering each
        CUDA error case (written first when following TDD).
    *   **Format error messages**: Follow
        [tt-write-error-tests](../tt-write-error-tests/SKILL.md#error-message-format)
        when writing user-facing or internal (`ABSL_CHECK`) error checks.
    *   **No `at::Tensors` in MLIR Builder methods**: Builder methods are
        deferred via lambda capture. Capturing an ATen tensor can cause
        deadlocks. Capture only needed fields (e.g., `dtype`, `num_dimensions`).
    *   **Avoid passing shapes**: Rely on StableHLO type inference. Only pass
        shapes if they depend on scalars (e.g., `torch.ones`, reshapes).
    *   **Watch for dtype mismatches** in StableHLO op chains.
    *   **Spell out concrete MLIR types instead of `auto`**: Use explicit
        `mlir::MlirOp` and `mlir::SmallVector<mlir::MlirOp>` declarations rather
        than `auto` when the type is not visible on the right-hand side. Builder
        helpers return different types (`mlir::MlirOp`,
        `absl::StatusOr<mlir::MlirOp>`, `mlir::Value`,
        `mlir::SmallVector<mlir::MlirOp>`), so explicit types make each
        intermediate value immediately clear.
    *   Follow [dynamism guidelines](../tt-support-dynamism/SKILL.md) for
        bounded dynamism.
    *   **Scalar promotion and compilation cache keys (`PromoteScalar`,
        `param_keys`, `IgnoreInCacheKey`)**:
        *   Promote `at::Scalar`, `std::optional<at::Scalar>`, or
            `at::ArrayRef<at::Scalar>` via `PromoteScalar(...)` *before*
            `TT_KERNEL` so scalar value changes do not trigger recompilation
            (promote only on branches that actually dispatch tensor computation,
            as debug builds assert `.GetTensor()` was called on every
            `PromotedScalar` unless `.AvoidPromoting(...)` applies). Use
            `.AvoidPromoting(...)` sparingly: skipping arithmetic for a special
            scalar value (such as `ScalarValue::kZero`) breaks `NaN` and
            `0 * ±inf -> NaN` propagation unless the PyTorch/CUDA spec
            explicitly ignores that operand (e.g., `beta == 0` in `addmm`).
        *   Pass the exact kernel parameter list to `TT_KERNEL` and forward
            `std::move(param_keys)` into `.op_param_cache_keys`. Pass `_` in
            place of `param_keys` only when no non-Tensor parameter affects
            StableHLO; **do NOT add `(void)param_keys;`** (`TT_KERNEL` handles
            it).
        *   Wrap a non-Tensor parameter in `IgnoreInCacheKey(arg, "<reason>")`
            *only* when it cannot change the compiled StableHLO graph
            (`"Unused"`, `"Doesn't affect SHLO"` for host/memory-only options,
            or `"Delegates to <Op>"` when returning another `Aten*` kernel that
            hashes it). Never wrap `at::Tensor`/`TensorList` (already hashed by
            the dispatcher) or any parameter that alters StableHLO ops,
            attributes, dtypes, or control flow.
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
    *   **Backward kernels**: Check `output_mask[i]` before computing each
        gradient and check `.defined()` on optional `Tensor` inputs (such as
        `weight` or `bias`).
    *   **C++ sink moves, `const` definitions, and switches**: Keep `std::move`
        on last-use `DeviceBufferRef` sink arguments to `MakeTensor` /
        `MakeStorage` / `MakeDataPtr`, pass read-only `absl::Span<const T>` by
        value, keep `const` on by-value parameters in `.cc` *definitions* while
        omitting top-level `const` on `.h` *declarations*, and omit `default:`
        on exhaustive enum/`ScalarType` switches so new enumerators are caught
        at compile time.
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
        conversions: never upcast float host scalars to `float64`; for
        `Dot`/`DotGeneral`, wire `GetAndAddPrecisionTo(param_keys)` and account
        for TPU generation precision toggles and StableHLO dot attributes
        (typically `float32` accumulation for `fp16`/`bf16`/`fp4`, whereas
        quantized and low precision integer/FP8 matmuls follow separate rules
        such as `int32` accumulation for `int8` or native FP8 operands with
        explicit scale and dequantization handling).
    *   Preserve IEEE-754 edge semantics: preserve `-0.0` signbits via
        `TOTALORDER` comparison or integer bitcast sign extraction (plain
        `compare(x, 0.0)` treats `-0.0 == +0.0`; check strict `GT` vs `GE` at
        `0.0` in activation backward ops; `EncodeParamCacheKey(double)` in
        `csrc/common/cache_key.h` already distinguishes `-0.0` from `+0.0` in
        `OpParamCacheKeys`, so avoid `.AvoidPromoting(ScalarValue::kZero)` or
        collapsing `0.0` in custom keys when `-0.0` and `+0.0` differ), use
        `Expm1` / `Log1p` near zero instead of `Exp(x) - 1`, and guard `±inf`
        (`0 * inf -> NaN`) and size-0 reduction/batch divisions
        (`0 / 0 -> NaN`).

See [Minimal Example](references/example_aten_kernels.cc) for a reference
implementation.

### 2. Register Operator

*   Register the op in `csrc/eager/tpu_aten_kernels.cc`.
*   Update `csrc/ops/op_names.h` and `csrc/ops/op_names.cc` alphabetically.
*   Update `tests/native_functions_data.py`.
*   Run `bazel test //tests:native_functions_test` to verify registration.

### 3. Testing

*   Remove `@op_testing.unimplemented_on_tpu` from the existing `test_<op>` stub
    in `tests/ops_test.py` once the op is implemented, and follow
    [testing skill](../tt-write-tests/SKILL.md) for TDD and test locations.

### 4. Code Health and Formatting

*   Follow [tt-create-change](../tt-create-change/SKILL.md) for formatting
    (`nox -s format`), opening the pull request early, and post-push validation.
