# Adding a New `torch.ops.tpu._dynamic_*` Custom Op

Reference for the `tt-support-dynamism` skill, §3 Case D. Last resort: only add
a new op when no composition of existing TPU custom ops (the `_dynamic_*`
family, `_set_dimension_logical_size`) plus ATen ops on runtime size tensors
expresses the semantics correctly or efficiently.

Follow the existing `_dynamic_*` ops (e.g., `_dynamic_slice`,
`_dynamic_arange`):

1.  Add an `OpName` in `csrc/ops/op_names.h`.
2.  In `csrc/eager/tpu_aten_kernels.cc`, add the schema (`m.def`) in
    `TORCH_LIBRARY(tpu, m)` and an `ImplInternal` registration in
    `TORCH_LIBRARY_IMPL(tpu, PrivateUse1, m)`. Take runtime sizes as 0-D tensors
    and static upper bounds as `int[]`.
3.  Implement the C++ lowering under `csrc/ops/dynamic/<op>/`, built on
    `op_builder_utils` utilities (see
    [op_builder_utils.md](op_builder_utils.md)) and covered in
    `csrc/ops/op_builder_utils_test.cc` where a new utility is needed.
4.  No Meta/fake kernel is needed, since these ops are inserted by FX passes
    after tracing. The pass must set a correct `node.meta["val"]` (upper-bound
    shape and dtype) on every inserted node, because later passes read it.
