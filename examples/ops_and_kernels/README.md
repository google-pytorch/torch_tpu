<!--
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Extending PyTorch functionality, from the basics to Pallas

This guide walks through different mechanisms to extend PyTorch functionality,
with an emphasis on the TorchTPU backend.

It starts with the easiest approach and gradually builds up towards the most
powerful and complex. It also provides background information to help you decide
on the right approach for your use case.

If you want to jump to a specific approach, here's a table of contents:

-   [Builtin functionality: modules, functions](#builtin-functionality-modules-functions)
-   [Extending modules and functions via composition](#extending-modules-and-functions-via-composition)
-   [Customizing autograd via torch.autograd.Function](#torchautogradfunction)
-   [Background on ATen ops](#background-on-aten-ops)
-   [Background on `torch.compile`](#background-on-torchcompile)
-   [Background on `torch.compile` on TPUs via XLA](#background-on-torchcompile-on-tpus-via-xla)
-   [Custom ops via Python](#custom-ops-via-python)
-   [Custom ops via HLO kernels](#custom-ops-via-hlo-kernels)
-   [Custom ops via JAX kernels (without Pallas)](#custom-ops-via-jax-kernels-without-pallas)
-   [Custom ops via existing Pallas kernels from Tokamax](#custom-ops-via-existing-pallas-kernels-from-tokamax)
-   [Custom ops via Pallas](#custom-ops-via-pallas)
-   [Conclusion](#conclusion)

This guide is structured as a series.

### Builtin functionality: modules, functions

There are two types of builtin callables in PyTorch: modules and functions.

*   Functions, such as `torch.nn.functional.linear`, are designed to be
    stateless. Stateful values, such as weights, are provided as explicit
    arguments to these functions.
*   Modules, such as `nn.Linear`, hold state such as trainable weights. The
    computation of a module is a function of both the input and the weights.
    Modules are always subclasses of `torch.nn.Module`.

A common pattern in the spirit of separation of concerns is for code in modules
to focus on state management and hooks. The actual computation is delegated to a
function.

For example, `torch.nn.Linear`'s `forward()` method delegates the core logic to
`torch.nn.functional.linear`:

```python
def forward(self, input: Tensor) -> Tensor:
  return F.linear(input, self.weight, self.bias)
```

[PyTorch source](https://github.com/pytorch/pytorch/blob/446033e622b25fea5fb72c3b1c5d9f9da2a11ef3/torch/nn/modules/linear.py#L130-L134)

The rest of the torch.nn.Linear class handles creation, saving, and loading
weights via the built-in functionality of `torch.nn.Module` and
`torch.nn.Parameter`.

```python
def __init__(...) -> None:
  ...
  self.weight = Parameter(
      torch.empty((out_features, in_features), **factory_kwargs)
  )
```

[PyTorch source](https://github.com/pytorch/pytorch/blob/446033e622b25fea5fb72c3b1c5d9f9da2a11ef3/torch/nn/modules/linear.py#L96)

### Extending modules and functions via composition

The most basic way to use PyTorch to build your own custom functionality is to
combine PyTorch-provided functions into larger functions, aka composition. When
you need to hold state, such as trainable weights, you create a subclass of
`torch.nn.Module`.

As a toy problem, suppose you want to create a custom module that handles
quantization aware training (QAT). In QAT, the master weights are held in a high
precision format, but the effect of quantization is simulated during training.
For simplicity, the quantization scheme will quantize to one bit, encoding only
+1 and -1, along with a block scaling factor.

> This guide's focus is on extending PyTorch functionality, so its discussion of
> quantization should not be considered realistic.

Look at the sample implementation in [qat_linear.py](qat_linear.py). Notice
that:

*   `QATLinear` is a subclass of `torch.nn.Module`.
*   `QATLinear` holds state in the form of trainable parameters, e.g. weights.
*   The core computation in `forward()` is delegated to `qat_linear()`.
*   `qat_linear` itself calls `quantize` and `dequantize` to implement QAT.

### torch.autograd.Function

`torch.autograd.Function` is the next approach to extending the functionality of
PyTorch. It enables developers to define a custom backward pass.

As a toy problem, suppose you decide that you want to modify the vanishing
gradients problem of sigmoid and instead clamp the gradient to a minimum of
0.01.

Look at the sample implementation in
[sigmoid_no_vanishing_grad.py](sigmoid_no_vanishing_grad.py). Notice that:

*   `torch.autograd.Function` is subclassed
*   `forward` and `backward` are overridden
*   The `backward` implements the key clamping logic using the PyTorch functions
    `torch.where` and the Python operators `<=` and `>`.
*   The output of the forward pass is saved for the backward pass.

As a manual test, the main function prints a table of the outputs and grads for
both the custom sigmoid and the regular `torch.sigmoid`.

Input  | Custom Output | Regular Output | Custom Grad | Regular Grad
-----: | ------------: | -------------: | ----------: | -----------:
-100.0 | 0.0000        | 0.0000         | 0.0100      | 0.0000
-10.0  | 0.0000        | 0.0000         | 0.0100      | 0.0000
-1.0   | 0.2689        | 0.2689         | 0.2500      | 0.1966
0.0    | 0.5000        | 0.5000         | 0.2500      | 0.2500
1.0    | 0.7311        | 0.7311         | 0.2500      | 0.1966
10.0   | 1.0000        | 1.0000         | 0.0100      | 0.0000
100.0  | 1.0000        | 1.0000         | 0.0100      | 0.0000

> Warning: In PyTorch 1.0, custom ops (a topic discussed below) were implemented
> using this same mechanism. In PyTorch 2.0, custom ops have their own first
> class support and should not use `torch.autograd.Function`.

### Background on ATen ops

There is a third class of callables in PyTorch beyond functions and modules:
ops.

Behind the scenes of a call to a function like `torch.nn.functional.linear(a,
b)` or even one with a custom autograd function like `_SigmoidNoVanishing`,
PyTorch converts these calls into ATen and c10d ops. ATen stands for "A Tensor
Library", and c10d is an abbreviation for caffe distributed. c10d ops are for
collectives to enable communication between devices. Everything else is an ATen
op. ATen contains thousands of ops, and these ops form an Intermediate
Representation (IR).

As you saw in the implementation of `torch.nn.Linear`, the forward method simply
calls `torch.nn.functional.linear`. Ultimately, that call will get translated by
this block of C++ into the aten ops of `addmm` and `.t` (transpose).

```cpp
return at::addmm(*bias, input, weight.t());
```

[PyTorch source](https://github.com/pytorch/pytorch/blob/5b4d57d8bb7dbe3ddb5dced469df4f5dd1e82a11/aten/src/ATen/native/Linear.cpp#L85)

You will now explore how to view the IR that the PyTorch IR engine creates
internally. Open [call_linear.py](call_linear.py). Notice that logs for aot are
enabled:

```python
torch._logging.set_logs(aot_graphs=True)
```

AOT stands for Ahead-Of-Time compilation. In this context, it basically means
the series of aten ops that your Python code was lowered into.

Inspecting the logs, you'll see the key lines: a transpose, then an addmm.

```log
t: "f32[64, 64][1, 64]cpu" = torch.ops.aten.t.default(primals_1);  primals_1 = None
addmm: "f32[32, 64][64, 1]cpu" = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = t = None
```

You now know how to view the ATen ops that PyTorch creates internally.

Exercise for the reader: Apply the same logging as in
[call_linear.py](call_linear.py) to [qat_linear.py](qat_linear.py) and figure
out which ATen ops correspond to the bit operations.

```python
  # Shift each of the 32 bits by zero to 31 positions.
  shifted = packed >> torch.arange(32, dtype=torch.uint32)
  # AKA shifted = packed.bitwise_right_shift(torch.arange(32, dtype=torch.uint32))

  # Mask out the upper bits.
  values = shifted & 1
  # AKA values = shifted.bitwise_and(1)
```

Exercise for the reader: Apply the same logging technique to
[sigmoid_no_vanishing_grad.py](sigmoid_no_vanishing_grad.py) and figure out
which ATen ops correspond to `torch.where` and the `>` and `<=` operator.

```python
local_grad = torch.where(
    torch.logical_and(result > 0.1, result <= 0.9),
    torch.tensor(0.25, device=result.device),
    torch.tensor(0.01, device=result.device),
)
```

Answer:

```
gt: "b8[7, 1][1, 1]cpu" = torch.ops.aten.gt.Scalar(sigmoid, 0.1)
le: "b8[7, 1][1, 1]cpu" = torch.ops.aten.le.Scalar(sigmoid, 0.9)
```

### Background on `torch.compile`

When you use
[`torch.compile`](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html)
with the default `inductor` backend on CUDA, one of the optimizations is fusing
ops. Without fusion, a sequence of operations like addition followed by ReLU
would require multiple reads and writes to High Bandwidth Memory (HBM) in a
synchronous, serial fashion. The addition would load data, compute, and write
back to HBM. After the addition op completes, the ReLU op would then read that
data back, compute, and write again. Fusion allows data to be loaded once,
processed for both operations, and written back once.

Take a look at [pointwise_fusion_cuda.py](pointwise_fusion_cuda.py). This script
demonstrates a full fusion within a single Triton kernel for a simple operation:

```python
@torch.compile(backend="inductor")
def fwd(x, y):
  return torch.nn.functional.relu(torch.add(x, y))
```

Inspect the logs for this test (enabled via `TORCH_LOGS="output_code"`). You
find a single Triton kernel named `triton_poi_fused_add_relu_0`. Even without
becoming a Triton expert, you can get a basic idea of what this kernel is doing.
Moving data from HBM to registers is explicit, and the computation happens in a
fused way: there is no intermediate write to HBM.

```python
@triton.jit
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # ...
    tmp0 = tl.load(in_ptr0 + (x0), xmask) # Load x from HBM to registers
    tmp1 = tl.load(in_ptr1 + (x0), xmask) # Load y from HBM to registers
    tmp2 = tmp0 + tmp1 # Pointwise addition
    tmp3 = tl.full([1], 0, tl.int32) # Create tensor of zeros in registers
    tmp4 = triton_helpers.maximum(tmp3, tmp2) # Apply ReLU in registers
    tl.store(out_ptr0 + (x0), tmp4, xmask) # Store result to HBM
```

> [!NOTE] The reason this example uses a pointwise op rather than a matmul:
> Inductor will often avoid attempting to fuse matmuls because the native
> library, cuBLAS, is already highly optimized.

### Background on `torch.compile` on TPUs via XLA

TorchTPU is the backend for PyTorch to run on Google TPUs. TorchTPU uses XLA to
compile PyTorch rather than Inductor. XLA stands for "Accelerated Linear
Algebra" and is an [open-source ML compiler](https://openxla.org/xla). The path
from your Python code to assembly on a TPU:

1.  Your Python code will require minor changes (e.g. setting "TPU" as the
    device")
1.  PyTorch's dispatcher lowers your Python code to ATen and c10d (no change)
1.  TorchTPU lowers the ATen and c10d ops to StableHLO, a dialect of
    [MLIR](https://mlir.llvm.org/) that is device agnostic.
1.  XLA lowers the StableHLO to HLO ops
1.  XLA optimizes the HLO over several passes.
1.  XLA lowers the HLO ops to device-specific LLO operators.
1.  XLA optimizes the LLO over several passes, targeting the specific TPU
    hardware.

Look at [pointwise_fusion_xla.py](pointwise_fusion_xla.py). To inspect how XLA
compiles this operation, the script sets environment variables to dump the HLO
ops and LLO ops.

HLO ops:

```python
XLA_DUMP_TO: Final[str] = "/tmp/xla_dump"
os.environ["XLA_FLAGS"] = f"--xla_dump_to={XLA_DUMP_TO} --xla_dump_hlo_as_text"
```

LLO ops: This variant sets both flags and env vars for maximum compatibility in
different environments.

```python
LLO_DUMP_TO: Final[str] = "/tmp/llo_dump"

flags = [ f"--xla_jf_dump_to={LLO_DUMP_TO}", "--xla_jf_dump_llo_text=true", ]
sys.argv.extend(flags) os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)
```

This is a large amount of logging but it follows a regular pattern. First, look
in the stdout for the section that lists files in the xla_dump directory, with
the header starting with `=== Files in /tmp/xla_dump (Total:`. Notice that there
are several boilerplate ops, including copy and seed. The final module looks
promising: it has relu in the name.
`module_0031.tt_jit_pointwise_fusion_xla_L41C0_fwd_relu`. Also, the `L41C0` is a
hint that this module is coming from the function defined near line 41. Note
that the filenames can be different depending on your environment.

The suffix `before_optimizations` is a good place to start. Search for the
contents of the file
`module_0031.tt_jit_pointwise_fusion_xla_L41C0_fwd_relu.before_optimizations.txt`.

First, there are multiple lines of metadata to help you track this HLO back to
your Python source code. Then, the actual HLO. Without becoming an expert on
HLO, you see an add and a relu. You have found the right place.

```
ENTRY %main.1 (Arg_0.1: f32[1024], Arg_1.1: f32[1024]) -> f32[1024] {
  %Arg_0.1 = f32[1024]{0} parameter(0)
  %Arg_1.1 = f32[1024]{0} parameter(1)
  %add.1 = f32[1024]{0} add(%Arg_0.1, %Arg_1.1), metadata={op_name="add/add" stack_frame_id=40}
  %constant.1 = f32[] constant(0)
  %relu.2 = f32[1024]{0} broadcast(%constant.1), dimensions={}, metadata={op_name="relu/relu" stack_frame_id=40}
  ROOT %relu.3 = f32[1024]{0} maximum(%add.1, %relu.2), metadata={op_name="relu/relu" stack_frame_id=40}
}
```

The XLA compiler will optimize this HLO, initially to more optimized HLO. The
final output has the suffix `codegen` so look for the file
`module_0031.tt_jit_pointwise_fusion_xla_L43C0_fwd_relu.after_codegen.txt`. This
HLO looks nearly identical, but does give a clue: the HLO has been un-inlined
and the core add and relu ops are in a fused_computation assigned to a variable
`add_maximum_fusion`.

This is an important keyword for you to connect the HLO to LLO.

```
%fused_computation (param_0.2: f32[1024], param_1.2: f32[1024]) -> f32[1024] {
  %param_0.2 = f32[1024]{0:T(1024)} parameter(0)
  %param_1.2 = f32[1024]{0:T(1024)} parameter(1)
  %add.0 = f32[1024]{0:T(1024)} add(%param_0.2, %param_1.2), metadata={op_name="add/add" stack_frame_id=40}
  %constant.0 = f32[]{:T(128)} constant(0)
  %relu.1 = f32[1024]{0:T(1024)} broadcast(%constant.0), dimensions={}, metadata={op_name="relu/relu" stack_frame_id=40}
  ROOT %relu.0 = f32[1024]{0:T(1024)} maximum(%add.0, %relu.1), metadata={op_name="relu/relu" stack_frame_id=40}
}

ENTRY %main.1 (Arg_0.1: f32[1024], Arg_1.1: f32[1024]) -> f32[1024] {
  %Arg_1.1 = f32[1024]{0:T(1024)} parameter(1), backend_config={"flag_configs":[],"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[]}
  %Arg_0.1 = f32[1024]{0:T(1024)} parameter(0), backend_config={"flag_configs":[],"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[]}
  ROOT %add_maximum_fusion = f32[1024]{0:T(1024)} fusion(%Arg_0.1, %Arg_1.1), kind=kLoop, calls=%fused_computation, metadata={op_name="relu/relu" stack_frame_id=40}, backend_config={"flag_configs":[],"window_config":{"kernel_window_bounds":[],"output_window_bounds":["1"],"input_window_bounds":[],"estimated_cycles":"1840","iteration_bounds":["1"],"cost_model_type":"COST_MODEL_TYPE_INVALID","ml_estimated_microseconds":0,"is_mask":false,"pad_output_on_minor_dim":"0","pad_input_on_minor_dim":"0","estimated_vmem_bytes":"0","estimated_bundle_count":"0","estimated_scoped_vmem_bytes":"0"},"scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"16777216"}],"used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"12288"}],"retry_config":{"retry_count":"0"},"aliasing_operands":{"lists":[]}}
}
```

Look at the LLO dump files by searching for `=== Files in /tmp/llo_dump (`.
There are many files with the `add_maximum_fusion` token. The first one is the
easiest view the LLO; the files after that represent many layers of hardware
specific optimization, resulting in difficult to read VLIW. Search for one that
starts with numbers and ends with `add_maximum_fusion-01-original.txt`. Without
attempting to become an expert at all the LLO instructions, you can pick out the
addition and max functions: vadd and vmax.

```
=== File: 1777396851540540302-add_maximum_fusion-01-original.txt ===
// [Enable stack traces via -xla_jf_collect_llo_stack_trace or -xla_jf_debug_level=2]
$region0: #{add_maximum_fusion}
  #allocation6 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
  %s0 = inlined_call_operand.hbm [shape: f32[1024], index: 0, kind: input, shape index: {}] /* operand 0 */
  %s1 = inlined_call_operand.hbm [shape: f32[1024], index: 1, kind: input, shape index: {}] /* operand 1 */
  %s2 = inlined_call_operand.hbm [shape: f32[1024], index: 2, kind: output, shape index: {}] /* operand 2 */
  $region1: #{add_maximum_fusion} parent=0
    #allocation0 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 0']
    #allocation1 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
    #allocation2 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
    #allocation3 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 1']
    #allocation4 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped, tag = 'scoped memory for add_maximum_fusion']
    #allocation5 [shape = 'u8[4096]{0}', space=vmem, size = 0x1000, scoped, tag = 'operand span for operand 2']
    %3 = vsyncpa [#allocation1], 0
    %4 = vsyncpa [#allocation4], 0
    %5 = vsyncpa [#allocation2], 0
    %7 = vsyncadd [#allocation1], 0
    %s9 = sshll.u32 %s0, 4
    %s10 = int_to_ptr.hbm [resolvable:$true] %s9
    %s11 = sshll.u32 [#allocation0], 4
    %s12 = int_to_ptr.vmem [resolvable:$true] %s11
    %14 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s10, /*size_in_granules=*/128, /*vmem=*/%s12, /*dst_syncflagno=*/[#allocation1] /*
base_bounds: (1)
dynamic_base_bounds: (1)
window_bounds: (1)
iteration_bounds: (1)
strides: (1)
pad_low: (0)
pad_high: (0)
element_size_in_bytes: 4096
second_minor_factor: 1 */
    %16 = vsyncadd [#allocation4], 0
    %s18 = sshll.u32 %s1, 4
    %s19 = int_to_ptr.hbm [resolvable:$true] %s18
    %s20 = sshll.u32 [#allocation3], 4
    %s21 = int_to_ptr.vmem [resolvable:$true] %s20
    %23 = dma.hbm_to_vmem [thread:$0]  /*hbm=*/%s19, /*size_in_granules=*/128, /*vmem=*/%s21, /*dst_syncflagno=*/[#allocation4] /*
base_bounds: (1)
dynamic_base_bounds: (1)
window_bounds: (1)
iteration_bounds: (1)
strides: (1)
pad_low: (0)
pad_high: (0)
element_size_in_bytes: 4096
second_minor_factor: 1 */
    %25 = dma.done [#allocation1], 128 /* pipeline-emitter-dma-wait */
    %27 = dma.done [#allocation4], 128 /* pipeline-emitter-dma-wait */
    %v28 = vld [vmem:[#allocation0] sm:$0xff]
    %v29 = vld [vmem:[#allocation3] sm:$0xff]
    %30 = xla_tuple %v28, %v29
    %31 = xla_tuple %30
    %v32 = vadd.f32 %v28, %v29
    %33 = xla_tuple %v32
    %34 = xla_tuple %v32, 0.0
    %35 = xla_tuple %34
    %v36 = vmax.f32 %v32, 0.0
    %37 = xla_tuple %v36
    %38 = vst [vmem:[#allocation5] sm:$0xff] /*vst_source=*/%v36
    %40 = vsyncadd [#allocation2], 0
    %s42 = sshll.u32 [#allocation5], 4
    %s43 = int_to_ptr.vmem [resolvable:$true] %s42
    %s44 = sshll.u32 %s2, 4
    %s45 = int_to_ptr.hbm [resolvable:$true] %s44
    %47 = dma.vmem_to_hbm [thread:$0]  /*vmem=*/%s43, /*size_in_granules=*/128, /*hbm=*/%s45, /*dst_syncflagno=*/[#allocation2] /*
base_bounds: (1)
dynamic_base_bounds: (1)
window_bounds: (1)
iteration_bounds: (1)
strides: (1)
pad_low: (0)
pad_high: (0)
element_size_in_bytes: 4096
second_minor_factor: 1 */
    %49 = dma.done [#allocation2], 128 /* pipeline-emitter-dma-wait */
    %50 = vsyncpa [#allocation1], 1
    %51 = vsyncpa [#allocation4], 1
    %52 = vsyncpa [#allocation2], 1
```

The heart of creating kernels is control of the final outputted LLO to better
control the TPU, whether the higher order language is HLO, JAX without Pallas,
or JAX with Pallas.

But before you investigate writing kernels, you need to understand how to wrap
those kernels as custom ops using the PyTorch custom op API. This will create
the necessary wrapper so that PyTorch understands how to compose it's
functionality like compile and autograd with a kernel.

### Introduction to custom ops

You started this journey learning to extend PyTorch functionality by composing
existing functionality.

Then, with `torch.autograd.Function`, you learned the first technique to gain
increased control of PyTorch's core engines.

You learned some necessary background information on ops and compile, and now
you are ready to start creating your own custom ops that live alongside the
existing ATen ops.

### Custom ops via Python

Custom ops in Python are a well covered topic in the
[PyTorch docs](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html).

These custom ops have all the advantages of PyTorch's ATen ops: they compose
well with `torch.compile` to avoid a graph break, ultimately leading to better
performance.

### Custom ops via C++ HLO kernels

Stay tuned for a forthcoming guide to implementing custom ops via HLO kernels in
C++. That said, since `jax.lax` is a nearly 1:1 mapping to StableHLO ops, the
next section on kernels in JAX may give you the ability to generate the HLO ops
you want, in pure Python, without diving into C++ build details.

### Custom ops via JAX kernels (without Pallas)

JAX is a different framework from TorchTPU but it ultimately compiles down to
the same HLO as TorchTPU. In some cases, JAX may give you some additional
control over the HLO. For example, JAX provides the
[population_count op](https://docs.jax.dev/en/latest/_autosummary/jax.lax.population_count.html),
which lowers directly to the
[StableHLO popcnt op](https://openxla.org/stablehlo/spec#popcnt). This makes JAX
a solution to implement some custom ops.

As a toy problem, suppose you want to implement via JAX a custom op that
performs a quantized sum for the one-bit format developed in
[qat_linear](qat_linear.py). You decide to take advantage of popcount as a "sum"
function for the one bit format. The sum of a tensor in this quantization format
can be inferred from the number of bits set *without conversion to a native
PyTorch floating point format*.

Each bit represents scale, and each non-bit represents -scale. So the sum is
simply

```
(number of bits set) * scale + (number of non-bits set) * (-scale)
= (number of bits set) * scale - (number of non-bits set) * scale
= ((number of bits set) - (number of non-bits set)) * scale
= ((number of bits set) - (total number of bits - number of bits set)) * scale
= (2 * (number of bits set) - (total number of bits)) * scale
```

Take a look at the sample implementation in
[quantized_sum.py](quantized_sum.py). Notice that:

*   `torch.tpu.pallas.jax_op` wraps JAX inside a PyTorch custom op.
*   The caller of the PyTorch custom op has no visibility into the fact that the
    op is implemented in terms of JAX.

Using the logging techniques demonstrated previously, you can even trace down
the [`popcnt`](https://openxla.org/stablehlo/spec#popcnt) in the dump file
`module_0039.tt_jit_custom_kernel.before_optimizations.txt`.

```
%test_example__popcount_0xb92b15e71d7bc5ec.1 (packed_vector.1: u32[]) -> u32[] {
  %packed_vector.1 = u32[] parameter(0), sharding={replicated}, metadata={op_name="packed_vector"}
  ROOT %population_count.1 = u32[] popcnt(%packed_vector.1), metadata={op_name="jit(jax_popcount)/population_count" stack_frame_id=10}
}
```

`jax.lax.population_count` is not differentiable so you would only use it for
inference. For a more realistic `jax.lax` op you might want to expose, consider
`jax.lax.approx_max_k`.

For a canonical example of implementing a custom op using a JAX kernel,
including use of `jax.jvp`/`jax.grad` to automatically create a backward
function, see the unit test
github.com/.../torch_tpu/tests/pallas/pallas_test.py, specifically
`test_jax_dot_grad_for_backwards()`.

### Custom ops via existing Pallas kernels from Tokamax

Coming soon!

### Custom ops via Pallas

TorchTPU exposes lower-level control of TPUs via Pallas, the domain-specific
language for writing custom kernels. Like the Triton kernel you saw earlier,
Pallas allows fine-grained control of memory.

This guide will only scratch the surface of Pallas. Refer to the
[JAX documentation on Pallas](https://docs.jax.dev/en/latest/pallas/index.html)
for a more complete introduction.

#### Basic example: Vector addition across modular targets

Before looking at a more complex quantization kernel, consider a minimal,
modular example that demonstrates the plumbing required to take a Pallas kernel
and make it callable from PyTorch.

This example is split into 3 distinct targets to illustrate the clean separation
of responsibilities:

1.  **Pallas kernel definition** in [pallas_add.py](pallas_add.py): Defines the
    low-level Pallas kernel body (`add_vectors_kernel`) that operates on TPU
    memory references, and wraps it in a typed JAX function (`add_vectors_jax`)
    using `pl.pallas_call`.
2.  **TorchTPU adapter** in [torch_pallas_add.py](torch_pallas_add.py): Adapts
    the JAX function into a PyTorch-compatible custom op using
    `torch.tpu.pallas.jax_op` and exposes a Pythonic PyTorch function
    `pallas_add_vectors`.
3.  **User-facing call and test** in [call_pallas_add.py](call_pallas_add.py):
    Simulates how an end-user calls the kernel from PyTorch, verifying that it
    works in both eager mode and with `@torch.compile(backend="tpu")`.

#### Advanced example: Quantization and packing in Pallas

As a toy problem, suppose you want to explore speeding up the quantization of
the one-bit format you previously developed. This toy problem is inspired by the
[DeepSeek v3's activation quantization kernel in Triton](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py#L10).

The DeepSeek act_quant kernel takes higher precision activations and both
quantizes them down to fp8 blockwise (with an f32 scaling factor per block), and
packs them to one float per byte. In your design [qat_linear.py](qat_linear.py),
you created two separate functions, one for quantization (quantize) and one for
packing (pack). You decide to create a single, merged function quantize and
pack.

Since writing kernels is about performance, you want to understand the
performance of an unoptimized baseline. This baseline also acts as a test of
numerical correctness.

Look at [quant.py](quant.py). It copies the code from
[qat_linear.py](qat_linear.py) and adds the necessary boilerplate to dump HLO
ops and LLO ops. It adds `quant_and_pack` to match the behavior of DeepSeek's
act_quant kernel. This is the baseline. In a realistic example, you would review
the generated aten/HLO/LLO, as well as profile the code. For this example,
assume you've done that and decide you think you can do better.

Look at the sample implementation in [quant_pallas.py](quant_pallas.py). Notice
that:

*   The wrapper around the JAX function is similar to
    [quantized_sum.py](quantized_sum.py), using `torch.tpu.pallas.jax_op`.
*   The JAX code itself calls into Pallas to implement the kernel.

## Conclusion

In this guide, you have learned how to extend PyTorch functionality. You started
with the most basic approach: composing existing functionality together.

You took a detour into understanding some internals of PyTorch in preparation
for deeper control of PyTorch.

You took the first steps by implementing a custom backward pass.

Then you learned how to implement a custom op, starting with a lower level of
complexity and gradually building up to the most complex approach for TPU:
Pallas.
