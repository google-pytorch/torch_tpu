# torch_tpu de-facto coding conventions (derived from origin/main @ 214c68c83, 2026-11-05)

All `path:line` citations are relative to `/Users/joinal/pytorvhtpu/upstream/torch_tpu`.
Kernels studied (all touched 2026-10/11 by Googlers): `dot`, `digamma`, `distance` (cdist/pdist),
`logcumsumexp`, `cumsum`, `glu`, `nonzero`/`nonzero_static`, `fmax`, plus the shared `binary`/`unary`
kernels. Cleanup commits by Gunhyun Park / Claudio Basile (2026-10/11) are used as the "what reviewers
remove" signal.

Enforced-by-tooling facts (not just taste):
- `.clang-format` is `BasedOnStyle: Google` (`.clang-format:22-23`); 80 columns.
- `tests/cpp_style_test.py` + `tests/cpp_style_lib.py` lint header guards and namespaces
  (`cpp_style_lib.py:210-230`, `:303-420`): every top-level namespace must be `torch_tpu`; the only
  approved nested names are `error`, `internal`, `kiln`; only the innermost may be anonymous.
- `tests/error_coverage_test.py` requires every `TT_ERROR` / `TT_RET_CHECK` / `TT_CHECK_THROW` to be hit
  by an error test, or carry `// ERROR_COV_INFEASIBLE=<why>` on the same line
  (`tests/error_coverage_lib.py:14-24`, `:82-86`).
- Debug builds crash on malformed error messages (`csrc/common/error_utils.cc:174-282`): see §4.
- Python: ruff, `line-length = 80`, `indent-width = 2`, `target-version = "py311"`, google pydocstyle
  (`pyproject.toml:128-148`, `:214-215`).
- `tests/native_functions_test.py` checks every op that PyTorch requires a backend to register is
  registered; unregistered ones are listed in `tests/native_functions_data.py` `UNREGISTERED_OPS`
  (remove the entry when you add the kernel).

---

## 1. C++ file layout

### 1.1 License header, guard, includes

New files use the block-comment Apache header with the current year (2026); BUILD and `.py` use `#`:

```cpp
/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 ...
 * limitations under the License.
 */
```
(`csrc/ops/dot/dot_checks.cc:1-15`; `nonzero_aten_kernels.cc:1-13` uses `//` style, the older variant;
do not copy it.)

Header guard is the path in caps: `TORCH_TPU_CSRC_OPS_DOT_DOT_CHECKS_H_`
(`csrc/ops/dot/dot_checks.h:17-18`, closing `#endif  // TORCH_TPU_CSRC_OPS_DOT_DOT_CHECKS_H_` at `:37`).

Include order (clang-format Google): own header; blank; `<std>` headers; blank; one ASCII-sorted block
of quoted includes (`"ATen/..."`, `"absl/..."`, `"c10/..."`, `"csrc/..."`, `"llvm/..."`, `"mlir/..."`,
`"stablehlo/..."`, `"torch/..."`). Uppercase sorts before lowercase:

```cpp
#include "csrc/ops/dot/dot_aten_kernels.h"

#include <utility>

#include "ATen/core/ATen_fwd.h"
#include "absl/status/statusor.h"
#include "csrc/common/cache_key.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/fixed_size_span.h"
#include "csrc/eager/device_buffer.h"
#include "csrc/eager/op_dispatcher.h"
#include "csrc/eager/tensor_to_buffer.h"
#include "csrc/ops/dot/dot.h"
#include "csrc/ops/dot/dot_checks.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
```
(`csrc/ops/dot/dot_aten_kernels.cc:17-34`). Includes are IWYU-exact; keep-only comments look like
`#include "c10/util/ArrayRef.h"  // IWYU pragma: keep for at::IntArrayRef` (`csrc/ops/op_builder_utils.h:43`).

### 1.2 Files per op

Directory `csrc/ops/<op>/` with:
- `<op>_aten_kernels.h` / `.cc`: the `Aten*` entry points. The header includes only `"ATen/core/ATen_fwd.h"`
  and declares the kernels (`csrc/ops/dot/dot_aten_kernels.h:20-26`).
- Optional `<op>.h` / `.cc` holding `Build<Op>Shlo` builders when other ops reuse them (`csrc/ops/cumsum/cumsum.h`,
  `csrc/ops/nonzero/nonzero.h`, `csrc/ops/dot/dot.h`). Otherwise the builder sits in the anonymous namespace
  of the `_aten_kernels.cc` (`csrc/ops/glu/glu_aten_kernels.cc:58-125`, `csrc/ops/fmax/fmax_aten_kernels.cc:33-48`).
  `digamma_aten_kernels.h:25-26` exports `BuildDigammaShlo` because polygamma reuses it.
- Optional `<op>_checks.h/.cc` for validation shared by several kernels (`csrc/ops/dot/dot_checks.cc`).
- `BUILD` (§6).

### 1.3 Namespaces

```cpp
namespace torch_tpu {

namespace {
// helpers: Build*Shlo, Validate*, <Op>Out helpers returning absl::Status
}  // namespace

at::Tensor AtenDot(const at::Tensor& lhs, const at::Tensor& rhs) { ... }

}  // namespace torch_tpu
```
(`csrc/ops/dot/dot_aten_kernels.cc:36-75`). Inside `namespace torch_tpu` never write `torch_tpu::` or
`::torch_tpu::` (removed in 4fdb01eb0); the qualifiers stay only inside `#define`s.

### 1.4 Naming

- Kernels: `Aten<OpName><Variant>` matching the ATen schema: `AtenCumsumOut`, `AtenGluBackwardGradInput`,
  `AtenAddRelu_Tensor`, `AtenNonzeroStaticOut` (`csrc/ops/binary_aten_kernels.h:158-277`).
- StableHLO builders: `Build<Op>Shlo(non-tensor params..., mlir::MlirOp input)` with the MlirOp **last**
  so `absl::bind_front` works (`csrc/ops/cumsum/cumsum.h:29-31`, `csrc/ops/nonzero/nonzero.h:26-33`).
- Validators: `Validate<What>` returning `absl::Status` / `absl::StatusOr<T>` (`ValidateIsVector`,
  `ValidateCdistForwardInputs`, `ValidateAndGetDotOutputType`).
- Predicates: `Is<Thing>` (`IsBool`, `IsPrivateUse1Device`).
- Locals: `snake_case`; MLIR values carry `_op` suffix (`lhs_op`, `x1_op`); constants `kCamel`;
  `OpName::k<Camel>`; the TT_KERNEL cache-key variable is `param_keys` (glu's `params_key` is an outlier).
- `const` on by-value params in **definitions** (`const int64_t normalized_dim`), not in declarations
  (`csrc/ops/cumsum/cumsum.cc:40-42` vs `cumsum.h:29-31`).

### 1.5 Comments

Full sentences, capitalized, period-terminated, `//` only. Explain *why* and cite PyTorch behavior:

```cpp
    // Dimension must be validated before resizing or early returns so that
    // invalid dims (e.g. dim > 0 on a 0-D scalar or empty tensor) throw
    // IndexError without mutating out.
    TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                       SafeWrapDim(dim, self.dim()));
```
(`csrc/ops/cumsum/cumsum_aten_kernels.cc:56-60`). Math derivations go above the builder
(`csrc/ops/glu/glu_aten_kernels.cc:136-146`). Argument comments: `/*dim=*/0`, `/*num_carries=*/1`
(`csrc/ops/distance/dist_aten_kernels.cc`, `csrc/ops/cumsum/cumsum.cc:78`). TODO forms:
`// TODO: b/477401982 - ...` (`src/torch_tpu/__init__.py:44`), `// TODO(wan): ...` (`error_utils.cc:1002`).
Marker comments with a reason are a house pattern: `// CRASH_OK`, `// CRASH_OK=indicates invalid kernel`,
`// M_IMPL_OK=implementing ImplStable().`, `// REF_CAPTURE_OK=called synchronously`,
`// ERROR_COV_INFEASIBLE=...`, `// NON_TORCH_TPU_OK=<reason>`, `// STATUS_CODE_OK=for alias`.

---

## 2. Kernel body idioms

### 2.1 `TT_KERNEL`

```cpp
at::Tensor AtenDot(const at::Tensor& lhs, const at::Tensor& rhs) {
  TT_KERNEL(OpName::kDot, param_keys, (lhs, rhs), {
    TT_ASSIGN_OR_THROW(auto result, Dot(lhs, rhs, std::move(param_keys)));
    return MakeTensor(std::move(result));
  });
}
```
(`csrc/ops/dot/dot_aten_kernels.cc:68-73`). Rules from `csrc/ops/macros/kernel.h:35-73`:
- Pass *all* kernel params, exactly, in the parenthesized list; it logs them and computes cache keys
  from every non-Tensor argument.
- Use `_` instead of `param_keys` when there are no non-tensor params (`AtenFmaxOut`, `AtenNonzero`);
  if a non-tensor param is present you get a `static_assert` telling you to wrap it in
  `IgnoreInCacheKey()` or use `param_keys` (`kernel.h:260-272`).
- Scalars are promoted *before* the macro and the `PromotedScalar` is what goes in the list:
  `auto promoted_alpha = PromoteScalar(alpha).AvoidPromoting(ScalarValue::kOne);`
  then `TT_KERNEL(OpName::kAddOut, param_keys, (self, other, promoted_alpha, out), {...})`
  (`csrc/ops/binary_aten_kernels.cc:1146-1176`). Debug builds crash if `GetTensor()` isn't called.
- Only `TT_*THROW*` macros are used inside the TT_KERNEL body; helpers return `absl::Status[Or]` and
  use `TT_RET_CHECK` / `TT_RETURN_IF_ERROR` / `TT_ASSIGN_OR_RETURN`. Mixing fails to compile
  (`error_utils.h:554-570`).

### 2.2 Dispatch

- Unary: `UnaryOp(self, builder, UnaryOpOptions)` -> `StatusOr<at::Tensor>`; `UnaryOpOut(self, out, builder, options)`
  -> `Status`; `UnaryOpInPlace` (`csrc/ops/unary_aten_kernels.h:60-84`). Options (`:33-58`):
  `op_name`, `op_param_cache_keys`, `out_dtype`, `out_dims`, `computation_dtype`,
  `allow_out_dtype_cast` (default true), `allow_low_precision_float` (default false).
- Binary: `BinaryOp` / `BinaryOpOut(self, other, out, builder, BinaryOpOptions)` / `TernaryOpOut`
  (`csrc/ops/binary_aten_kernels.h:127-150`). Options `:44-73`: `reverse_operands`, `force_float_inputs`,
  `result_dtype` (nullopt = `at::result_type` promotion), `allow_out_dtype_cast`, `allow_low_precision_float`.
- N-ary: `DispatchOp<kArity>(op_builder, {t1, t2}, {.out_dtype=, .out_dims=, .op_param_cache_keys=})`
  returns `StatusOr<DeviceBufferRef>`; wrap with `MakeTensor(std::move(buf))`. `DispatchOpOut<kArity>(..., out, options)`
  resizes/assigns `out` and auto-donates (`csrc/eager/op_dispatcher.h:394`, `:479`; usage doc `:52-93`).
  Designated initializers, in struct field order: `out_dtype`, `out_dims`, `out_layout`,
  `computation_dtype`, `op_param_cache_keys` (`op_dispatcher.h:161-180`).
- Multi-output: `DispatchOp<kArity, kNumOutputs>` with `FixedSizeSpan`/`std::array` out dtypes/dims.

Op builder lambda shape:

```cpp
  auto op_builder = [current_precision](FixedSizeSpan<mlir::MlirOp, 2> inputs)
      -> absl::StatusOr<mlir::MlirOp> {
    auto& [lhs_op, rhs_op] = inputs;
    return BuildDotShlo(lhs_op, rhs_op, current_precision);
  };
```
(`csrc/ops/dot/dot_aten_kernels.cc:47-51`). Structured bindings by reference (`auto&`), fixed in c7c02b5d9.
When the builder's last param is the MlirOp, prefer
`absl::bind_front(BuildCumsumShlo, normalized_dim, out_mlir_type)` (`cumsum_aten_kernels.cc:85-90`).

### 2.3 Dtypes

- `TT_ASSIGN_OR_THROW(const mlir::ElementType out_mlir_type, ConvertTo<mlir::ElementType>(out.scalar_type()));`
- `InferOutputDtype(self)` (float/complex keep, else default dtype) `csrc/common/aten_utils.h:59`.
- `InferComputationDtype(dtype)` instead of `at::toOpMathType` (`aten_utils.h:61-78`; a58f6fc04).
- `ToAccumulateType(scalar_type)` for `.computation_dtype` (`csrc/common/dtype.h:152`, glu).
- Predicates: `IsBool / IsComplex / IsIntegral / IsLowPrecisionFloat` templates accept Tensor, Scalar
  or ScalarType (`aten_utils.h:134-205`); `IsFloatingPoint(tensor)`, `IsPrivateUse1Device(tensor)`.
- `ValidateNotLowPrecisionFloat(x)` -> NotImplementedError "not implemented for float8_e5m2" (`aten_utils.h:183-188`).
- `ValidateOutDtype(out, result_dtype, allow_out_dtype_cast)` (`aten_utils.h:109-124`); messages
  "expected the output dtype to be int64, got float32" / "result type float32 can't be cast to the desired output type int32".
- In builders: `GetElementTypeOrDie(op)`, `GetTensorTypeOrDie(op)`, `CastIfNeeded(op, type)`,
  `ConvertIfInteger(op, type)` (`csrc/ops/op_builder_utils.h:161,229,631,804`).

### 2.4 Shapes

- Types: `Dimensions`, `Strides`, `Indices` = `absl::InlinedVector<int64_t, 6>` (`csrc/common/dimension_types.h:33-39`).
- Build them with `CopyIntVector(any_int_range)` (`csrc/common/utils.h:51-66`), never `Dimensions(b, e)` (e73ce6bdd).
- `ConcreteSizeToDimensions(sym_sizes)` for SymInt sizes (`aten_utils.h:92-100`); `NumElements(dims)`,
  `SafeWrapDim(dim, rank)` -> IndexError (`csrc/common/error_utils.h:148-159`); `InferSize(a, b)` broadcast
  (`op_builder_utils.h:99-118`).
- In builders: `GetAllDimensions(op)`, `GetBatchDimensions(rank, dim)`, `GetStaticShape(op)`,
  `BroadcastIfNeeded`, `ApplyBroadcastIfNeeded(a, b)` -> `(auto [a_, b_])`, `MakeConstantLike(op, 0.0)`,
  `MakeScalarConstant(builder, v, type)`, `Iota`, `Unsqueeze/Squeeze/Flatten` (`op_builder_utils.h:329-545, 829-832`).
- Pass `.out_dims = tensor.sizes()` directly; don't copy into a `Dimensions` first (c7c02b5d9).

### 2.5 `out=` variants and zero-size tensors (the canonical order)

```cpp
at::Tensor& AtenCumsumOut(const at::Tensor& self, int64_t dim,
                          std::optional<at::ScalarType> dtype,
                          at::Tensor& out) {
  TT_KERNEL(OpName::kCumsumOut, param_keys, (self, dim, dtype, out), {
    TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
        << "expected the output tensor to be on "
        << GetPrivateUse1DeviceDebugName() << ", got " << out.device();
    ...dtype checks...
    TT_ASSIGN_OR_THROW(const int64_t normalized_dim,
                       SafeWrapDim(dim, self.dim()));
    TT_THROW_IF_ERROR(ResizeTensorIfShapeDiffers(out, self.sizes()));
    // Fast-path: empty tensors require no computation.
    if (out.numel() == 0) {
      return out;
    }
```
(`csrc/ops/cumsum/cumsum_aten_kernels.cc:40-69`). Order: device -> dtype -> dim validation (so IndexError
precedes mutation) -> `ResizeTensorIfShapeDiffers` (`csrc/ops/resize/resize_aten_kernels.h:55-61`) ->
`numel()==0` early return -> 0-D special case -> NotImplemented checks -> dispatch. For functional
variants: `MakeEmptyTensor(sizes, dtype, device)` (`csrc/eager/tensor_to_buffer.h:219-221`) and return it
when any output dim is 0, *before* the bf16/f16 rejection (`csrc/ops/distance/dist_aten_kernels.cc:651-672`).
`out=` tensors from `nonzero` also run `at::assert_no_internal_overlap(out); at::assert_no_overlap(out, self);`
(`csrc/ops/nonzero/nonzero_aten_kernels.cc:110-118`). Dtype-preserving ops pass
`.allow_out_dtype_cast = false` (nonzero `:84`, `:162`; abs `unary_aten_kernels.cc:253`).
Builders also guard rank-0/0-element inside SHLO when needed (`logcumsumexp_aten_kernels.cc:109-118`).

### 2.6 Bounded dynamism

Builders check `input_type.hasStaticShape()` and fall back to `DynamicSlice`/`GetDimensionSize`/
`SetDimensionSize`; unsupported dynamic cases return `TT_RET_CHECK(..., error::kPythonNotImplementedError)
<< "expected dimension " << dim << " of input to have a static size, got dynamic size"`
(`csrc/ops/glu/glu_aten_kernels.cc:58-105`, check at `:81`).

---

## 3. `TT_*` macro table (`csrc/common/error_utils.h`)

| Macro | Line | Context | Effect |
|---|---|---|---|
| `TT_ERROR(code) << msg` | 381 | Status-returning | builds a `StatusBuilder`; `return TT_ERROR(error::kInvalidArgument) << "..."` |
| `TT_RET_CHECK(cond, code) << msg` | 421 | Status-returning | returns error when `cond` false; `<<` evaluated lazily |
| `TT_RETURN_IF_ERROR(status)` | 674 | Status-returning | propagate; can append `<< " in X"` or `.SetPrepend()` |
| `TT_ASSIGN_OR_RETURN(lhs, rexpr[, _ << ...])` | 485 | Status-returning | parenthesize lhs with commas: `((auto [a, b]), f())` |
| `TT_ASSIGN_OR_CRASH(lhs, rexpr)` | 549 | anywhere, programmer error | `ABSL_CHECK_OK`-style crash; needs `// CRASH_OK` |
| `TT_ASSIGN_OR_THROW(lhs, rexpr)` | 631 | only Torch-bound kernels (inside `TT_KERNEL`) | throws `TtError`; compile error in Status-returning fn |
| `TT_THROW_IF_ERROR(status)` | 706 | kernels | preferred way to surface a helper's Status (keeps C++ trace) |
| `TT_CHECK_THROW(cond, code) << msg` | 728 | kernels | replaces `TORCH_CHECK`; message is mandatory |
| `TT_SOURCE_LOCATION` | 391 | | `c10::SourceLocation` of caller |
| `TT_KERNEL(op, keys, (args), {...})` | `csrc/ops/macros/kernel.h:73` | kernel entry | logging, Python ctx capture, cache keys, `TtError`->c10 translation |
| `TT_MAKE_OP_PARAM_CACHE_KEYS(a, b)` | `kernel.h:143` | | manual cache keys when `_` is used |

Error codes (`error_utils.h:60-101`) and their Python exception (`error_utils.cc:998-1021`):

| `error::` | absl code | Python |
|---|---|---|
| `kInvalidArgument` (default choice for user errors) | InvalidArgument | `RuntimeError` |
| `kPythonRuntimeError` | Unknown | `RuntimeError` |
| `kPythonNotImplementedError` | Unimplemented | `NotImplementedError` |
| `kPythonIndexError` / `kIndexError` (what `SafeWrapDim` returns, `error_utils.cc:684-690`) | OutOfRange | `IndexError` |
| `kResourceExhausted` | ResourceExhausted | `torch.OutOfMemoryError` |
| `kInternal`, `kFailedPrecondition`, `kNotFound`, `kUnavailable`, ... | same-named | `RuntimeError` (fallback) |

Pick `kPythonNotImplementedError` whenever CUDA/CPU would say `"op" not implemented for 'X'`
(dtype/feature unsupported), `kInvalidArgument` for shape/device/value errors, `kIndexError` via `SafeWrapDim`.

## 4. Error message wording (crash-checked in debug builds, `csrc/common/error_utils.cc`)

- Start lowercase (`:174-207`); only `TorchTPU`, `PjRt`, `XLA` may start capitalized (`:71-80`).
- Do not end with `.`; last char must be alnum or one of `])}>'` (`:216-240`).
- If the message mentions "expected" or ", got", it must be exactly `expected ..., got ...` (`:264-282`).
- Use PyTorch dtype names (`float32`, not `f32`): always `ToString(scalar_type)` / `ToString(mlir::ElementType)`
  (`:284-310`). Shapes/dims via `ToString(tensor.sizes())`.
- The `op(): ` prefix is added automatically from the root op (`ToBaseName`, `csrc/ops/op_names.cc:25-50`);
  never write it yourself.
- Canonical phrasings seen in maintainers' code:
  - `"expected the " << name << " argument to be a 1D tensor, got " << dim << "D of shape " << ToString(sizes)` (`dot_checks.cc:36-38`)
  - `"expected the input dtype to be floating point, got " << ToString(dtype)` (`logcumsumexp_aten_kernels.cc:162-164`)
  - `"expected the output tensor to be on " << GetPrivateUse1DeviceDebugName() << ", got " << out.device()` (`cumsum_aten_kernels.cc:43-45`)
  - `"expected inputs to have the same shape, got " << ToString(a) << " vs " << ToString(b)` (`dot_checks.cc:45-47`)
  - `"expected the size of dimension " << dim << " to be even, got " << n` (`glu_aten_kernels.cc:238-240`)
  - `"expected 'size' to be non-negative, got " << size` (`nonzero_aten_kernels.cc:120-123`)
  - `"not implemented for " << ToString(dtype)` with `kPythonNotImplementedError` (`dot_checks.cc:63-66`)
  - `"the input dtypes cannot be bool"` (`dot_checks.cc:61-62`)

---

## 5. Registration

1. Add `k<OpCamel>` to `enum class OpName` (sorted block, `csrc/ops/op_names.h:27-30`) and a `case` in
   `constexpr ToString()` returning the ATen overload name: `case OpName::kDot: return "dot";`,
   `"cumsum.out"`, `"div.out_mode"` (`op_names.h:893-915`, `:1229`). Names feed the compile cache; keep stable.
   `op_names_test.cc` asserts uniqueness.
2. Include the kernel header in `csrc/eager/tpu_aten_kernels.cc` and register inside
   `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`: `ImplStable<OpName::kDot>(m, AtenDot);` (`:371`, `:508`).
   `ImplInternal<>` is for `_`-prefixed TPU-internal ops (`:261-271`); custom ops are `m.def`'d in
   `TORCH_LIBRARY(tpu, m)` (`:1286`) and Meta kernels in `TORCH_LIBRARY_IMPL(aten, Meta, m)` (`:1513`).
3. Add `"//csrc/ops/<op>:<op>_aten_kernels"` to `tpu_aten_kernels` deps in `csrc/eager/BUILD` (`:719`, `:776-785`).
4. Delete the op from `UNREGISTERED_OPS` in `tests/native_functions_data.py` if present.
5. Mention "Register kernels in tpu_aten_kernels.cc via ImplStable." in the commit body (650d3084f does).

---

## 6. BUILD target

```python
# Domain experts for this package:
# adivinpatel04
# cbasile-g

load("//build_files:build_defs.bzl", "torch_tpu_cc_library", "torch_tpu_package_end")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//:__subpackages__"],
)

torch_tpu_cc_library(
    name = "digamma_aten_kernels",
    srcs = ["digamma_aten_kernels.cc"],
    hdrs = ["digamma_aten_kernels.h"],
    deps = [ ... ],
)

torch_tpu_package_end()
```
(`csrc/ops/digamma/BUILD:15-52`). Deps are buildifier-sorted: `":local"`, `"//csrc/..."`, `"//shims/..."`,
`"@com_google_absl//..."`, `"@llvm-project//..."`, `"@stablehlo//..."`. One dep per include, nothing unused
(a4a266103 pruned dozens). Observed include -> target mapping:

| include prefix | dep |
|---|---|
| `csrc/common/x.h` | `//csrc/common:x` |
| `csrc/eager/op_dispatcher.h`, `device_buffer.h`, `tensor_to_buffer.h` | `//csrc/eager:op_dispatcher`, `:device_buffer`, `:tensor_to_buffer` |
| `csrc/ops/op_builder_utils.h`, `op_names.h`, `unary_aten_kernels.h`, `binary_aten_kernels.h`, `scan_builder.h`, `precision_context.h` | `//csrc/ops:<name>` |
| `csrc/ops/macros/kernel.h` | `//csrc/ops/macros:kernel` |
| `csrc/ops/resize/resize_aten_kernels.h` | `//csrc/ops/resize:resize_aten_kernels` |
| `ATen/...` | `//shims/torch:aten_headers` |
| `torch/library.h`, `ATen/MemoryOverlap.h`-style torch headers | `//shims/torch:torch_headers` |
| `c10/core/...` | `//shims/torch/c10/core:base_headers` |
| `torch/headeronly/core/ScalarType.h` | `//shims/torch/torch/headeronly:torch_headeronly` |
| `absl/status/statusor.h`, `absl/status/status.h`, `absl/functional/bind_front.h`, `absl/log/absl_log.h` | `@com_google_absl//absl/status:statusor`, `//absl/status`, `//absl/functional:bind_front`, `//absl/log:absl_log` |
| `mlir/IR/*` / `llvm/ADT/*` | `@llvm-project//mlir:IR` / `@llvm-project//llvm:Support` |
| `stablehlo/dialect/StablehloOps.h` | `@stablehlo//:stablehlo_ops` |
| `stablehlo/integrations/cpp/builder/{AttrTypeBuilderUtil,MlirBuilder,StablehloBuilder,ChloBuilder}.h` | `@stablehlo//stablehlo/integrations/cpp/builder:{attr_type_builder_util,mlir_builder,stablehlo_builder,chlo_builder}` |

Wrap BUILD at 80 columns (`AGENTS.md`). Separate `torch_tpu_cc_library` for the SHLO builder when it is
shared (`csrc/ops/cumsum/BUILD:27-72`).

---

## 7. Tests

### 7.1 Unit test (`tests/ops_unit_test.py`, class `OpsUnitTest(TorchTpuVsCpuTestBase)` at `:144`;
grads in `OpsGradUnitTest` `:17463`)

```python
  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("float16", torch.float16),
      ("float64", torch.float64),
  )
  def test_digamma_negative_zero(self, dtype: torch.dtype):
    # Test that digamma(-0.0) returns +inf and digamma(+0.0) returns -inf.
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.digamma(
            to(torch.tensor([-0.0, 0.0], dtype=dtype), device=device)
        )
    )
```
(`tests/ops_unit_test.py:15228-15240`). `assert_close_tpu_vs_cpu(fn, *, check_value=, check_dtype=, rtol=, atol=)`
(`tests/op_testing.py:6644-6670`); direct comparisons use `utils.assert_close(actual, expected)`
(`src/torch_tpu/_internal/utils/test_utils.py:81-112`; `check_value` omitted = LOOSE). Variable naming
convention `golden_*` / `tpu_*` (`ops_unit_test.py:57-61`). Cover `out=` resizing, 0-D, empty tensors in one
test (`test_logcumsumexp_out_dtype`, `:1646-1680`; `test_nonzero_static`, `:8108-8150`). OpInfo coverage is
`tests/ops_test.py`: `def test_digamma(self): self.do_test_op("digamma")` (`:1910`) plus tolerance entries
in `ACCURACY_OVERRIDES_VS_GPU` (`:219`, `:409-415`).

### 7.2 Error test (`tests/errors_test.py`, runs on TPU and GPU)

```python
  def test_digamma_out_unsupported_int_dtype(self):
    t = torch.tensor([1.5, 2.5], device=et.device(), dtype=torch.float32)
    out = torch.zeros(2, device=et.device(), dtype=torch.int32)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""digamma(): result type float32 can't be cast to the desired output type int32""",
        gpu="""result type Float can't be cast to the desired output type Int""",
        message_reviewed_by="gunhyun",
    ):
      torch.digamma(t, out=out)
```
(`tests/errors_test.py:3756-3765`). Conventions: `et.device()`; triple-quoted exact messages
(regex only for unstable text, `tests/error_testing.py:274-333`); `message_reviewed_by="<ldap>"`;
`@parameterized.named_parameters({"testcase_name": "dot", "op": torch.dot, "op_name": "dot"}, ...)` with
f-string `tpu=f"""{op_name}(): ..."""` (`:8450-8475`); IndexError for dim errors (`:3980-3989`);
`if et.is_on_gpu(): self.skipTest("GPU behavior difference")` when GPU diverges (`:3941-3956`).
TPU-only errors (dynamism, NotImplemented paths) go to `tests/tpu_errors_test.py` with no `gpu=` and
`@et.why_tpu_only("...")` (`tpu_errors_test.py:1439-1464`); errors raised inside the builder need `.cpu()`.
Every new `TT_RET_CHECK`/`TT_CHECK_THROW`/`TT_ERROR` needs one of these or `// ERROR_COV_INFEASIBLE=`.

---

## 8. Python (`src/torch_tpu/_internal/...`)

- 2-space indent, 80 cols, double quotes, Google docstrings (Args:/Returns:/Yields:), module docstring,
  `__all__` (`src/torch_tpu/_internal/precision/precision.py:14-100`).
- Google import style: `import dataclasses`, `from collections.abc import Mapping, Sequence`,
  `from typing import Any, Final, TypeAlias`; py3.11 syntax `X | None`, builtin generics, `enum.StrEnum`
  (2c17e0797). Inline suppressions: `# pylint: disable=protected-access`, `# pyrefly: ignore[...]`, `# noqa: F401`.
- Public API: implementation lives in `torch_tpu/_internal/<pkg>/_impl.py`; one canonical public path under
  `torch.tpu.*` declared in `_PUBLIC_MODULES` (`src/torch_tpu/_internal/public_api.py:14-120`); symbols are
  decorated with `annotations.experimental(reason)` / `stable()` / `deprecated(version, reason)` /
  `internal_or_experimental` / `forbid_in_torch_compile` (`src/torch_tpu/_internal/utils/annotations.py:391-561`).
  pybind classes: `Precision = experimental("...")(precision_impl.Precision)`.
- Deprecation: mark `@deprecated`, keep >= 2 minor releases (`docs/developers/api_policy.md:54-58`).
  Internal knobs: `TORCH_TPU_INTERNAL_*` env vars, `torch_tpu_internal_*` C++ flags, no Python `flags.DEFINE_*`
  in library code (`docs/developers/api_design_guide.md:27-80`). Custom C++ ops register as `torch.ops.tpu._foo`
  and are re-exported as `torch.tpu.ops.foo` (`api_design_guide.md:170-190`).
- BUILD: `torch_tpu_py_library` with `py_library_providing_imports_info` + `*.modspec.json` for packages
  (`src/torch_tpu/_internal/precision/BUILD`); new deps go in `pyproject.toml` then `./requirements/lock_environments.sh`.
- Tests: `absltest` + `parameterized`, `seed_test_utils.RepeatableTest` bases, `setUpModule()` calling `et.set_up_module()`.

---

## 9. Commit / PR message style (`git log -300 origin/main`)

- Subject: imperative, no type prefix required; optional `[Ops][<op>]` (24/300), `[torch_tpu]` (11),
  `[TorchTPU MXU Improvement]`. Length min 18 / median 66 / p90 91 chars (not 50/72-bound); 31% end with `.`,
  most don't. Examples: `[Ops][dot] Validate input vector length equality in torch.dot`,
  `Return early from split_with_sizes_copy when the input has no rows`,
  `Standardize on IsBool across operator lowerings`.
- Body: prose paragraphs (or bullets) saying what changed and why, naming files/functions in backticks;
  "Behavior and generated programs are unchanged." when refactoring. External PRs follow
  `.github/pull_request_template.md`: `## Description` ... `## Test Plan` (exact bazel targets, device,
  torch/libtpu versions, "`nox -s lint`: clean") and `Fixes #N` at top; copybara appends `Closes #N`,
  `Generated from cl/...`, `COPYBARA_INTEGRATE_REVIEW=...`, `PiperOrigin-RevId:` (e14e4d1a1, 1e71c2b51).
- Bug links: `b/123456` in TODOs and bodies (43 hits), `Fixes #N` (18), `Closes #N` (19).
- Trailers: none authored by humans survive; the only trailer in 300 commits is bot `PiperOrigin-RevId`.
  An `Assisted-by:` trailer mentioned in e14e4d1a1's body was stripped on import; `Co-Authored-By` never
  appears (and the CLA bot rejects `noreply@anthropic.com`).

---

## 10. Template: new unary op `foo` with `out=` variant (assembled from real snippets)

`csrc/ops/foo/foo_aten_kernels.h`
```cpp
/* Copyright 2026 Google LLC ... Apache 2.0 block (dot_checks.h:1-15) */
#ifndef TORCH_TPU_CSRC_OPS_FOO_FOO_ATEN_KERNELS_H_
#define TORCH_TPU_CSRC_OPS_FOO_FOO_ATEN_KERNELS_H_

#include "ATen/core/ATen_fwd.h"

namespace torch_tpu {

at::Tensor& AtenFooOut(const at::Tensor& self, at::Tensor& out);

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_OPS_FOO_FOO_ATEN_KERNELS_H_
```

`csrc/ops/foo/foo_aten_kernels.cc` (pattern of `digamma_aten_kernels.cc:39-78` + `cumsum_aten_kernels.cc:44-46`)
```cpp
#include "csrc/ops/foo/foo_aten_kernels.h"

#include <utility>

#include "ATen/core/TensorBody.h"
#include "absl/status/statusor.h"
#include "csrc/common/aten_utils.h"
#include "csrc/common/device_type.h"
#include "csrc/common/error_utils.h"
#include "csrc/common/to_string.h"
#include "csrc/ops/macros/kernel.h"
#include "csrc/ops/op_builder_utils.h"
#include "csrc/ops/op_names.h"
#include "csrc/ops/unary_aten_kernels.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {
namespace {

// Lowers foo(x) = <math>. Integer inputs are converted to `out_mlir_type`
// first, matching PyTorch's float promotion for foo.
absl::StatusOr<mlir::MlirOp> BuildFooShlo(mlir::ElementType out_mlir_type,
                                          mlir::MlirOp input) {
  TT_ASSIGN_OR_RETURN(mlir::MlirOp op, ConvertIfInteger(input, out_mlir_type));
  return mlir::stablehlo::Tanh(op);  // placeholder body
}

}  // namespace

at::Tensor& AtenFooOut(const at::Tensor& self, at::Tensor& out) {
  TT_KERNEL(OpName::kFooOut, _, (self, out), {
    TT_CHECK_THROW(IsPrivateUse1Device(out), error::kInvalidArgument)
        << "expected the output tensor to be on "
        << GetPrivateUse1DeviceDebugName() << ", got " << out.device();
    TT_CHECK_THROW(!IsComplex(self), error::kInvalidArgument)
        << "expected the input dtype not to be complex, got "
        << ToString(self.scalar_type());
    TT_ASSIGN_OR_THROW(const mlir::ElementType out_dtype,
                       ConvertTo<mlir::ElementType>(InferOutputDtype(self)));
    TT_THROW_IF_ERROR(UnaryOpOut(self, out,
                                 absl::bind_front(BuildFooShlo, out_dtype),
                                 {.op_param_cache_keys = OpParamCacheKeys::Empty(),
                                  .out_dtype = out_dtype}));
    return out;
  });
}

}  // namespace torch_tpu
```
(If `foo` takes a non-tensor parameter, name the keys `param_keys`, pass it as
`.op_param_cache_keys = std::move(param_keys)` and add `"//csrc/common:cache_key"` only if you spell
`OpParamCacheKeys` yourself; `UnaryOpOut` validates `out` dtype via `ValidateOutDtype` and resizes it.)

Binary variant (pattern of `fmax_aten_kernels.cc:33-57`): `BuildFooShlo(mlir::MlirOp self, mlir::MlirOp other)`
starting with `TT_ASSIGN_OR_RETURN((auto [self_, other_]), ApplyBroadcastIfNeeded(self, other));`, kernel body
`TT_THROW_IF_ERROR(BinaryOpOut(self, other, out, BuildFooShlo, {.op_param_cache_keys = OpParamCacheKeys::Empty()}));`.
For a scalar `alpha`-style parameter copy `AtenAddOut` (`binary_aten_kernels.cc:1146-1176`).

`csrc/ops/foo/BUILD` (pattern of `digamma/BUILD`; add `"@com_google_absl//absl/functional:bind_front"` for bind_front,
`"//shims/torch:torch_headers"` only if a `torch/` or non-ATen header is included)
```python
torch_tpu_cc_library(
    name = "foo_aten_kernels",
    srcs = ["foo_aten_kernels.cc"],
    hdrs = ["foo_aten_kernels.h"],
    deps = [
        "//csrc/common:aten_utils",
        "//csrc/common:device_type",
        "//csrc/common:error_utils",
        "//csrc/common:to_string",
        "//csrc/ops:op_builder_utils",
        "//csrc/ops:op_names",
        "//csrc/ops:unary_aten_kernels",
        "//csrc/ops/macros:kernel",
        "//shims/torch:aten_headers",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/status:statusor",
        "@stablehlo//stablehlo/integrations/cpp/builder:attr_type_builder_util",
        "@stablehlo//stablehlo/integrations/cpp/builder:mlir_builder",
        "@stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder",
    ],
)
```

Registration: `kFooOut,` in the enum + `case OpName::kFooOut: return "foo.out";` (`op_names.h`);
`ImplStable<OpName::kFooOut>(m, AtenFooOut);` and `#include "csrc/ops/foo/foo_aten_kernels.h"` in
`csrc/eager/tpu_aten_kernels.cc`; `"//csrc/ops/foo:foo_aten_kernels",` in `csrc/eager/BUILD`; drop
`'foo.out'` from `tests/native_functions_data.py`.

Unit test (`tests/ops_unit_test.py`, inside `OpsUnitTest`):
```python
  @parameterized.named_parameters(
      ("float32", torch.float32),
      ("bfloat16", torch.bfloat16),
      ("int32", torch.int32),
  )
  def test_foo(self, dtype: torch.dtype):
    self.assert_close_tpu_vs_cpu(
        lambda device: torch.foo(
            to(torch.tensor([-1.5, 0.0, 2.0], dtype=dtype), device=device)
        )
    )

  def test_foo_out_resizes_empty(self):
    device = torch.device("tpu")
    x = torch.empty(0, 4, dtype=torch.float32, device=device)
    out = torch.empty(2, 3, dtype=torch.float32, device=device)
    torch.foo(x, out=out)
    self.assertEqual(out.shape, (0, 4))
```

Error tests (`tests/errors_test.py`):
```python
  def test_foo_cpu_out_tensor(self):
    t = torch.ones(2, device=et.device())
    out = torch.ones(2, device="cpu")
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foo(): expected the output tensor to be on tpu, got cpu""",
        gpu="""Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!""",
        message_reviewed_by="<reviewer-ldap>",
    ):
      torch.foo(t, out=out)

  def test_foo_unsupported_complex(self):
    t = torch.tensor([1 + 1j], device=et.device(), dtype=torch.complex64)
    with et.assert_raises_message(
        RuntimeError,
        tpu="""foo(): expected the input dtype not to be complex, got complex64""",
        gpu=""""foo_cuda" not implemented for 'ComplexFloat'""",
    ):
      torch.foo(t)
```

---

## 11. Anti-patterns reviewers removed (from `git log` cleanups, 2026-10/11)

| Removed | Replacement | Commit |
|---|---|---|
| `torch_tpu::ToString(...)`, `::torch_tpu::X` inside `namespace torch_tpu` | unqualified | 4fdb01eb0 |
| `Dimensions(v.begin(), v.end())`, `Strides{...begin(), ...end()}` | `CopyIntVector(v)` | e73ce6bdd |
| `std::iota` / loops producing `0..rank-1` or "all dims but `dim`" | `GetAllDimensions(op)`, `GetBatchDimensions(rank, dim)` | c54e3ddac |
| `t.has_value() && t->defined()` | `SanitizeOptionalTensor(t)` | c9af1d9b0, e7a41427f |
| `x.scalar_type() == at::kBool`, `c10::isComplexType(...)`, `isIntegralType(..., true)` | `IsBool / IsComplex / IsIntegral` | d86cca284, 744746391, b609aa34d |
| `at::toOpMathType(...)` | `InferComputationDtype(...)` | a58f6fc04, b6bb16066 |
| unused BUILD deps / `#include`s; `//shims/torch:torch_headers` when only ATen is used | prune | a4a266103, c7c02b5d9 |
| `const Dimensions d = {t.size(0), ...}; .out_dims = absl::MakeConstSpan(d)` | `.out_dims = t.sizes()` | c7c02b5d9 |
| `auto [a, b] = inputs;` in op-builder lambdas | `auto& [a, b] = inputs;` | c7c02b5d9 |
| always promoting `value`/`alpha` scalars to device tensors | `PromoteScalar(v).AvoidPromoting(ScalarValue::kOne)` + `IsOne()` fast path | 042190e86, 836600e28 |
| near-duplicate kernels differing in one SHLO op | one helper parameterized by an enum | ea9135046 |
| removing `std::move` from `MakeTensor(std::move(buf))` | keep `std::move` (sink parameter) | 23726269c reverts a171cd284 |
| `MaterializationBlockMode`, dead flags | delete | 4308a3581, b455b3a63 |
| `TORCH_CHECK(...)` | `TT_CHECK_THROW(cond, error::k...) << msg` | `error_utils.h:728-737` |
| `.op_name = OpName::kX` equal to the TT_KERNEL op | omit (ABSL_CHECK crash) | `op_dispatcher.h:405-414` |
| `MakeTensor(scalar)` | `PromoteScalar(scalar)` (`[[deprecated]]`) | `op_dispatcher.h:542-548` |
| error text "Expected X but got Y." / `f32` | "expected x, got y" / `float32` | `error_utils.cc:174-310` |
| Python `Optional[X]`, `List[int]`, `(str, Enum)` | `X | None`, `list[int]`, `enum.StrEnum` | 2c17e0797 |
| `EXPECT_*` inside death-test children | `ABSL_CHECK*` | 2cd7a38ed |
