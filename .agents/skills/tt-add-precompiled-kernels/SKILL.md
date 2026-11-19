---
name: tt-add-precompiled-kernels
description: >-
  Adds precompiled unbounded dynamic kernels to TorchTPU: Mosaic kernels written
  in Kiln that the compilation cache loads on an eager cache miss instead of
  compiling a static kernel per shape. Covers the kernel and its buffer
  contract, adding missing Pallas features to Kiln, the extentless cache key,
  kernel ids and the compatibility predicate, registration, end-to-end tests,
  C++ unit tests only for what end-to-end tests can't cover, and retiring the
  dynamic_ops_test.py cases once the op is fully implemented. Use when
  adding, extending or testing a kernel in
  csrc/experimental/precompiled_kernels/, like the nonzero kernels. Don't use
  for bounded dynamism in op lowerings (use tt-support-dynamism) or for an op's
  ATen kernel and StableHLO lowering (use tt-implement-ops).
---

# Adding Precompiled Kernels

Precompiled unbounded dynamic kernels are Mosaic kernels, written in Kiln, whose
executables accept unbounded dimension sizes: on an eager compilation cache
miss, the cache loads a compatible registered kernel instead of compiling the
traced graph for the new shape. This skill adds one end to end, with the nonzero
kernels as the worked example. All paths are relative to the TorchTPU root, the
directory that holds `csrc/` and `tests/`.

## How a Kernel Is Found

*   An op's ATen kernel records a graph whose `CompilationCacheKey` combines an
    extentless key, a dimensions key and a compile options key. The extentless
    key fingerprints everything within an op lowering graph but the dimension
    sizes: op names, op param cache keys, operand ranks and dtypes, edges,
    donated inputs, graph outputs and the core pinning mode
    (`GraphSignature::GetKey` in `csrc/common/cache_key.cc`).
*   On an eager cache miss, `UnboundedDynamicKernelRegistry`
    (`csrc/common/unbounded_dynamic_kernel_registry.h`) returns the first
    kernel, in registration order, that is registered under the graph's
    extentless key and whose `is_compatible` accepts the logical input and
    output shapes. A kernel can be registered under several extentless keys, so
    that graphs with any of them share its executable. torch.compile never uses
    these kernels. `is_compatible` is a callback function that is used to filter
    out kernels within the registry using fields that are not captured in the
    extentless key.
*   So a kernel only serves graphs that consist of exactly the op as recorded.
    If eager execution fuses the op with neighboring ops, the graph's key
    differs and a static kernel is compiled.
*   Kernels in the registry are marked as either `enabled_by_default=true` or
    `enabled_by_default=false`.
    `TORCH_TPU_INTERNAL_USE_UNBOUNDED_DYNAMIC_KERNELS`, read once per process,
    selects the kernels: `default` (or unset) loads those with
    `enabled_by_default=true`, `all` also loads the `enabled_by_default=false`
    ones, and `none` loads none.

## Reference Implementation: nonzero

*   Kernels, keys and metadata:
    `csrc/experimental/precompiled_kernels/nonzero_indices.{h,cc}`.
*   Registration: `kernel_registration.cc` and `BUILD` in the same directory.
*   Compatibility unit test: `nonzero_indices_test.cc`, also there.
*   The ATen kernel whose graph the key mirrors:
    `csrc/ops/nonzero/nonzero_aten_kernels.cc`.
*   End-to-end tests: `tests/dynamic_ops_test.py`, and `test_nonzero` in
    `tests/ops_test.py`.
*   Kiln: `csrc/internal/mosaic/kiln/kiln.h` maps its concepts to Pallas,
    `lib/pipeline.h` declares `EmitPipeline`, and `kiln_test.cc` and
    `kiln_e2e_test.cc` hold small kernels.
*   A kernel registered under several extentless keys:
    `csrc/experimental/precompiled_kernels/sort/sort.{h,cc}`, whose one stable
    kernel serves both stable and unstable sorts.

nonzero is not fully implemented (step 7): its kernels serve F32 inputs of rank
1 to 8 in their default layouts, and other inputs fall back to static kernels.

## Workflow

Kiln features that the kernel needs go in a parent commit (step 2), and steps
1–7 make the kernel commit.

### 1. Scope the Kernel

Decide, and document in the header, before writing code:

*   **The graphs it serves**: for each ATen overload, the ranks, dtypes and op
    param values. Each combination has its own extentless key, and each overload
    records its own `OpName` (e.g. `nonzero.out` records `kNonzeroOut`, not
    `kNonzero`).
*   **The buffer contract**: the on-device layout of every input and output, any
    physical view of the outputs, and the arithmetic bounds the caller checks
    (see `nonzero_indices.h`). XLA's default layouts can depend on sizes: the
    tile of a 1-D F32 array is N rounded up to a power of two in [128, 1024],
    and S64[K,D] switches to a row-major layout when K <= D rounded up to a
    power of two.
*   **What stays static**: inputs outside the contract (other layouts, sizes or
    dtypes) fall back to static kernels. They become the not-served cases of
    step 5.

### 2. Write the Kernel in Kiln

Add `csrc/experimental/precompiled_kernels/{kernel}.{h,cc}` (e.g.
`nonzero_indices`), following `nonzero_indices.cc`:

*   **Kernel**: `kiln::CreateKernel` with `kiln::InSpec::Hbm` and
    `kiln::OutSpec::Hbm` specs of `kiln::kDynamic` dimensions
    (`BuildNonzeroIndices`). The C++ runs once, at build time: `kiln::Scalar`
    and `kiln::Tile` are device values, `kiln::For` and `kiln::If` emit device
    control flow, and C++ loops unroll. Stream HBM operands through VMEM with
    `kiln::EmitPipeline`.
*   **Parameters**: the payloads as unbounded-dynamic shapes with the contract's
    tiled layouts (`MakeUnbounded`), then, for each dynamic payload, an
    S32[rank] shape with layout `{0:T(128)}` that receives its runtime
    dimensions (`NonzeroParameters`).
*   **Outputs**: `Execute` allocates the outputs whose size the host knows (e.g.
    nonzero's K) and donates them: list them in `donated_output_params` and
    alias the kernel's result to them. Mosaic has no S64 memrefs, so write int64
    outputs through an S32 view: `physical_shape_fn` maps the logical shape to
    the view, and `output_transform` bitcasts the result back
    (`DestinationPhysicalShape` and `BitcastToLogicalIndices`).
*   **Metadata**: `Make{Op}DynamicMetadata()` returns the kernel's
    `UnboundedDynamicExecutionMetadata` (`csrc/common/compilation.h`): the
    parameter of each logical input, the donated outputs, the metadata parameter
    of each dynamic payload (`metadata_params`), the parameter shapes, the
    output transform and `is_compatible` (step 3). Its hooks are plain function
    pointers cached with the executable, so pass per-kernel constants as
    template parameters, not captures.
*   **Build function**: build the `kiln::Kernel` and compile it with
    `CompileMosaicKernel` (`kernel_utils.h`), as `BuildNonzeroIndicesExecutable`
    does. It serializes the kernel, wraps it with `BuildMosaicCustomCall`
    (`mosaic_custom_call.h`; `operand_to_metadata` maps each operand to its
    metadata parameter, -1 for none, and `aliased_parameter` names the donated
    output's parameter), sets `set_run_backend_only(true)` so that no HLO pass
    rewrites the custom call, sets the result layout and `argument_layouts`, and
    compiles with `client.CompileAndLoad`, wrapped in `AdaptXlaError`.
*   **Header**: exposes only `Get{Op}UnboundedDynamicKernels()`, returning a
    `std::vector<UnboundedDynamicKernelRegistration>` (or
    `Get{Op}UnboundedDynamicKernel()`, returning one
    `UnboundedDynamicKernelRegistration`, for a single kernel, like sort), and
    documents each kernel's buffer contract. Each registration pairs an
    `UnboundedDynamicKernel` with `additional_search_keys`, the extentless keys
    of any other graphs it serves (step 3).

**Missing Kiln features**: Kiln mirrors Pallas TPU (see the concept map in
`kiln.h`). If the kernel needs something that Pallas TPU has but Kiln lacks
(e.g. a primitive, a DMA or control-flow form, or a memory space), don't work
around the gap in the kernel, e.g. by emulating it with other ops or emitting
MLIR directly. Add it to Kiln in a parent commit of the kernel commit, so that
it is reviewed and lands on its own, then write the kernel with it:

1.  Mirror the Pallas API and its semantics, and lower it to the Mosaic ops that
    Pallas emits (JAX's
    [`jax/_src/pallas/mosaic/lowering.py`](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/lowering.py)
    and `primitives.py`).
2.  Put core concepts in `core/` and higher-level helpers, like `EmitPipeline`,
    in `lib/`. Keep MLIR out of Kiln's public headers, which `kiln_test.cc`
    checks. Export new headers from `kiln.h` and its `:kiln` target, and add the
    Pallas mapping to the concept map.
3.  Test the emitted MLIR with FileCheck in `kiln_test.cc`, and the results on a
    TPU in `kiln_e2e_test.cc`.

### 3. Split Keys, Ids and Compatibility

Each `UnboundedDynamicKernel` is identified by its `key`,
`MakeUnboundedDynamicKernelCacheKey(extentless_key, id)` (`kernel_utils.h`),
which also names its executable in the compilation cache. Graphs with that
`extentless_key` find the kernel, as do graphs with any of the extentless keys
in the `additional_search_keys` of its `UnboundedDynamicKernelRegistration`.
Each piece of information belongs to exactly one of these three:

*   **Extentless key** (`{Op}CacheKey`): what the graph records apart from
    dimension sizes. Build a `GraphSignature` that mirrors the op's ATen kernel
    exactly: the same core pinning mode, inputs (ranks and dtypes; the
    dimensions are placeholders), `AddOp` with the same `OpName`,
    `OpParamCacheKeys` and donated inputs, outputs, and graph outputs. Return
    `GetKey().extentless_key()`. A mismatch fails silently: the kernel is never
    found. Kernels for different ranks, dtypes or op param values have different
    keys, so nothing else needs to tell them apart. The extentless key is
    constructed by creating a dummy graph upon kernel registration to reuse the
    key construction logic.
*   **`is_compatible`**: checks only what the key leaves out, e.g. on-device
    layouts, compared with the buffer contract by `HasDeviceLayouts`
    (`kernel_utils.h`, which resolves them with `ResolveXlaLayouts`). A kernel
    is only compatible with the op if the extentless key matches and
    is_compatible() == true. Never re-check operand counts, ranks, dtypes or op
    params: the registry only calls it for graphs with one of the kernel's
    extentless keys.
*   **Kernel id** (the `id_key`): tells apart kernels that share an extentless
    key, so name each id after the property outside the key that its kernel
    requires or selects, never after rank, dtype or op params. For example,
    `kNonzeroKGreaterThan2` names the 2-D kernel's K > 2 requirement, and
    `kNonzeroInputMinorTile1024` to `kNonzeroInputMinorTile128` name the
    kernels' input tiles. Declare the ids in an `enum class {Op}KernelId : int`
    with explicit non-negative values. They are persisted in tier-2 cache entry
    names, so never reuse or renumber them; -1 is reserved for keys that don't
    identify a kernel.

Set the kernel's `op_name` to the `OpName` passed to `AddOp`: the
`ops_test_kiln` targets only run the ops whose ATen name, without the overload
(e.g. `sort` for `kSortValuesStable`), is a registered `op_name`.

When kernels share an extentless key, registration order is lookup order: make
their predicates disjoint (like the 1-D tiles) or register the most specific
kernel first.

When one kernel serves graphs with different extentless keys, e.g. a stable sort
kernel that serves both `torch.sort(x)` and `torch.sort(x, stable=True)`,
register it once, rather than one kernel per key, so that it compiles once and
they share its executable: build its `key` from one of the extentless keys and
list the others in `additional_search_keys`. When merging existing kernels this
way, keep the `key` of one of them, so that its persisted cache entries stay
valid, and retire the other ids with a comment in the enum.

### 4. Register

*   **BUILD**: add a `torch_tpu_cc_library` for `{kernel}` to
    `csrc/experimental/precompiled_kernels/BUILD` (or to a subpackage's `BUILD`,
    like `sort/`), and make it a dep of `:kernel_registration`, which
    `csrc/eager:tpu_aten_kernels` already links in (`alwayslink`).
*   **Registration**: in `kernel_registration.cc`, add
    `TT_REGISTER_DYNAMIC_KERNELS(Get{Op}UnboundedDynamicKernels());`, or
    `TT_REGISTER_DYNAMIC_KERNEL(Get{Op}UnboundedDynamicKernel());` for a single
    kernel. The argument runs during static initialization, so it must not read
    flags, the environment or other registries. Two kernels with one `key` crash
    at startup.
*   **`enabled_by_default`**: `true` only if the op's static path compiles once
    per data-dependent output shape, as nonzero's does per K. Otherwise, leave
    the kernel opt-in (`false`, loaded with `all`) until benchmarks show that it
    beats the static kernels.
*   **C++ unit tests, only for what end-to-end tests can't cover**: steps 5 and
    6 test the kernel through the path that users hit (key, registration,
    compatibility and execution), so don't repeat that in C++. Add a C++ unit
    test, like nonzero's parameterized `IsCompatible` test in
    `nonzero_indices_test.cc`, only for behavior that no end-to-end test can
    reach, e.g. custom layouts set with `Shape::set_layout`, and limit it to
    that behavior. Make it a `torch_tpu_cc_test` named `{kernel}_test`, with the
    same `BUILD` attributes as `nonzero_indices_test` (e.g. its TPU tag).

### 5. Test End to End in `tests/dynamic_ops_test.py`

Add, next to existing helpers:

*   **Cases**: a helper (e.g. `_{op}_case`) that builds deterministic inputs,
    and the `_{OP}_SERVED_CASES` and `_{OP}_NOT_SERVED_CASES` tuples of
    `(name, input, ...)` cases. Comment each group of cases with the kernel edge
    it covers: each kernel id, tile and scratch-window boundaries, masked tails,
    the smallest and largest sizes, and for not-served cases, each reason
    (layout, dtype, rank, sizes outside the contract).
*   **`test_{op}_served`**: run the op once on an input of the same shape that
    the same kernel serves, to load the kernel and any static graph the op also
    needs (nonzero's count graph). Then call
    `self.run_and_check_cache(op, *inputs, served=True)`, which compares the
    result with the CPU's, checks that the inputs are unmodified, and expects 1
    dynamic hit and 0 compilations. Pin the buffer contract by asserting
    `torch.tpu.get_device_layout_if_materialized` of the inputs and the result.
*   **`test_{op}_not_served`**: call `torch.tpu._clear_cache()`, warm up only
    the other graphs, then call `run_and_check_cache` with `served=False` and
    the expected number of static compilations.
*   **Warm-up inputs**: create them on the CPU and copy them to the TPU, since a
    TPU factory op could fuse into the op's graph and change its key.
*   **Mode**: the `dynamic_ops_test` target in `tests/BUILD` runs in the `all`
    mode so all dynamic kernels can be run.

### 6. Test the Op's OpInfo Samples in `tests/ops_test.py`

*   `ops_test_kiln` and `ops_test_kiln_grad` in `tests/BUILD` set the mode to
    `all` and run `test_{op}` for every op with a registered kernel `op_name`
    (step 3). They skip each sample that no Kiln kernel served, and fail if none
    was served.
*   If no OpInfo sample can reach the kernel (e.g. sort's, which only serves
    1024 elements), pass `kiln_kernel_unserved_reason` to `do_test_op` and keep
    the `dynamic_ops_test.py` cases.
*   The other `ops_test` targets set the mode to `none`. They cover the static
    lowering and fail if a Kiln kernel serves a sample anyway.
*   If the op has no `test_{op}` yet, add one first, following the
    `tt-write-tests` skill.
*   To skip a known issue of one kind of kernel, e.g. in a `skip_if`, use
    `op_testing.is_kiln_kernel_mode()`. Prefer narrowing `is_compatible` for a
    Kiln kernel bug, though: users hit the bug in the default mode.
*   Raise the `shard_count` of `ops_test_kiln` and `ops_test_kiln_grad` in
    `tests/BUILD` if they near their timeouts.

### 7. Check Whether the Op Is Fully Implemented

The op is fully implemented when its kernels serve every input that `test_{op}`
runs in eager mode, so that `_{OP}_NOT_SERVED_CASES` would only hold inputs that
compile nothing (e.g. nonzero with K = 0, which skips the index graph). Check:

*   every dtype that `test_{op}` doesn't exclude, and every rank;
*   every variant it checks: functional, `out=` (`check_out_variant`) and
    in-place, each with its own `OpName`;
*   the default layouts at every size, including the sizes where they change;
*   every output size, including the edge cases.

If the op is fully implemented, delete its case helper, case tuples and tests
from `tests/dynamic_ops_test.py`: from then on, `test_{op}` in
`tests/ops_test.py` covers the kernels on all its samples in `ops_test_kiln`,
and the static path in the other `ops_test` targets. If no op is left in
`dynamic_ops_test.py`, delete it and its target. Otherwise, keep the cases.

## Verification

Build and test as `AGENTS.md` describes, one command at a time.
`torch_tpu_py_test` targets with `run_on_accelerators` only exist as per-TPU
variants (e.g. `dynamic_ops_test_tpu-v5lite` on GitHub).

*   Kiln parent commit: `kiln_test` and `kiln_e2e_test` in
    `csrc/internal/mosaic/kiln/BUILD`.
*   Steps 2–4: `{kernel}` and `:kernel_registration` in
    `csrc/experimental/precompiled_kernels/BUILD`, and `{kernel}_test` if you
    added one.
*   Steps 5 and 7: `dynamic_ops_test` in `tests/BUILD`.
*   Steps 6 and 7, with `--test_filter=test_{op}`: `ops_test_kiln`,
    `ops_test_kiln_grad`, and the `tests/BUILD` target that runs `test_{op}` in
    CI (`ops_test`, or the op's category target, e.g. `nonzero_ops_test`).
