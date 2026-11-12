# Typical File Locations for Op Implementation

When implementing a new operator, you will typically modify or create files in
the following locations:

*   `src/torch_tpu/BUILD`
*   `csrc/ops/{op_name}/BUILD`
*   `csrc/ops/{op_name}/{op_name}_aten_kernels.cc`
*   `csrc/eager/tpu_aten_kernels.cc`
*   `csrc/ops/{op_name}/{op_name}_aten_kernels.h`
*   `tests/native_functions_data.py`

Only when the StableHLO lowering helpers have a second user (see the skill):

*   `csrc/ops/{op_name}/{op_name}.cc`
*   `csrc/ops/{op_name}/{op_name}.h`

Plus the relevant test files — see
[tt-write-tests](../../tt-write-tests/SKILL.md).
