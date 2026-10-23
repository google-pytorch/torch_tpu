# Copyright 2026 Google LLC
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

# pylint: skip-file
import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
  import marimo as mo

  return (mo,)


@app.cell(hide_code=True)
def _(mo):
  mo.md(r"""
    # TorchTPU and the TPU memory model
    PyTorch has evolved alongside GPUs and effectively unlocks their power to Python users. TorchTPU extends PyTorch to express the power of XLA on TPUs.

    The physical hardware of GPUs and TPUs are different, and so are the programming paradigms targeting each hardware architecture. TorchTPU exists to bridge this gap in both directions. We aim to support GPU-native interfaces like tensor views and streams in XLA, while also exposing the unique power of the XLA compiler in familiar PyTorch interfaces.

    Understanding how TorchTPU accomplishes these goals can help users optimize TPU-native codebases for maximum benefit.
    
    ## Hardware: CPUs, GPUs, TPUs
    A general-purpose CPU (such as the one powering your computer or mobile device) consists of multiple cores, each supporting one or more threads. Each thread is capable of executing in parallel, while also accessing shared memory with random-access read/write capabilities.

    A GPU is in many ways similar to a CPU; however, a GPU has many more threads than a CPU. A modern high-power CPU may have a few dozen threads; but a high-power GPU has *thousands*. The tradeoff is that each GPU thread is individually somewhat weaker than a CPU thread, and there are some restrictions on thread independence. Still, this makes GPUs useful for workloads with highly parallelizable operations, such as graphics rendering or linear algebra.

    A TPU is a fundamentally different design. The heart of the TPU is the TensorCore and its matrix multiplication units (MXUs). These MXUs operate as a [systolic array](https://en.wikipedia.org/wiki/Systolic_array), which is somewhat similar to a SIMD register on a CPU. These use wide, two-dimensional registers that can perform significantly more dense computational work per clock cycle than a CPU or GPU thread.

    However, the TensorCore is fundamentally single-threaded; it is not possible to subdivide a systolic array to do multiple simultaneous computations. Additionally, having wide registers means that a TensorCore is very poor at performing sparse, random-access read-write operations, as a large memory block needs to be loaded at once.

    The most recent generations of TPU augment their TensorCores with SparseCores, which are (as the name suggests) specialized for sparse operations, including more precise read/write operations. The separate cores also allow for a limited amount of parallelism, with each SparseCore able to perform operations at the same time as each TensorCore.

    ## Programming model: CUDA vs XLA
    The native software interfaces for GPUs and TPUs are tailored to the capabilities of those platforms.

    Take CUDA as an example, as the most commonly used GPU programming model backing PyTorch. CUDA is a low-level programming interface, similar to C or C++. CUDA buffers are, like CPU memory, intrinsically one-dimensional byte arrays. A CUDA kernel is an imperative procedure–a void function, to use the C terminology. Each thread is provided with pointers to some buffers, and is able to read from any offset into any buffer, and write back into them, byte-by-byte.

    This programming model does not work for TPUs. Memory on TPUs is aligned not as one-dimensional byte arrays, but as two-dimensional register-aligned tiles. And as mentioned, TensorCores do not have threads and are poor at random-access.

    Instead, the TPU compiler, XLA, uses a declarative, functional model. XLA accepts programs written in a language called Multi-Level Intermediate Representation (MLIR) which contains multiple dialects. Using the StableHLO dialect, it is possible to describe the desired logical tensor output of a program. XLA will compile that program into lower-level internal operations that are optimized for your specific TPU generation.
    
    ## Buffers: `torch.UntypedStorage`

    PyTorch's core tools are very closely aligned with the CUDA programming model.

    A storage object, using the Python class `torch.UntypedStorage`, is little more than a memory address (`torch.UntypedStorage.data_ptr()`) and a size in bytes (`torch.UntypedStorage.nbytes()`). This is a close match for both the CPU and CUDA interfaces.

    However, this is not sufficient for execution on TPU; a buffer in XLA is always well-typed and has a known multi-dimensional shape. Because a TPU’s internal memory is aligned to 2D registers, it is critical to know how a buffer is laid out. This means that a `torch.UntypedStorage` alone does not carry enough information for TPU execution.

    To solve this problem, TorchTPU adds an additional layer. When on TPU, a `torch.UntypedStorage.data_ptr` does not point directly to TPU memory; instead, it points at a location in CPU memory which contains additional metadata about the TPU buffer. **This is the key to TorchTPU’s operation**; this additional metadata allows for active interpretation between PyTorch's existing CUDA-like API and the realities of XLA's functional design and the TPU hardware.

    ## `torch.Tensor`
    A `torch.Tensor` can be thought of as two things simultaneously.

    First, a `torch.Tensor` is a *physical* representation of an access pattern on an underlying `torch.UntypedStorage`. A `Tensor` contains essential metadata for the strides (multiplicative factors), sizes (index bounds), offset (location of the 0-element), and element type (set the bytes per logical element). This metadata provides a way to map from an index tuple into a tensor, into a byte offset into the linear 1D buffer array backing that `Tensor`.

    Second, a `torch.Tensor` is a *logical* representation of a mathematical object. When you matrix-multiply two tensors, the result is another `Tensor`, with an appropriate shape, representing the outcome. The exact pattern of strides and storage offset are of minimal importance here; what the `Tensor` means as a computation is more relevant than what the tensor is as a memory reference.

    When moving from CUDA to TPU, understanding this distinction is critical–the **logical** behavior of PyTorch operations on Tensors can be correctly performed on XLA/TPU, even if the **physical** reality of the device is intrinsically different!
    
    ## View decomposition
    When running on CUDA, the stride language of a `torch.Tensor` is used directly by the internal CUDA kernels; the strides are provided to the internal thread procedures to describe how to offset into the relevant buffer. These offsets are performed dynamically at runtime.

    But on TPU, the physical strides are not directly usable; a one-dimensional stride description is not usable for two-dimensional register-aligned memory. However, the logical behavior of views *is* supported through StableHLO. Most Tensor view operations have a direct StableHLO equivalent: reshaping, transposing, slicing, and broadcasting.

    When provided with a `torch.Tensor` stride pattern, TorchTPU will perform an operation called *view decomposition*. This interprets a stride pattern and element type as a sequence of logical operations which will produce the desired output Tensor with every element in the right place.

    As an example:

    ```python
    # Logical values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    # sizes=(10,), strides=(1,), storage_offset=0
    base = torch.arange(10, dtype=torch.float)

    # Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
    physical_view = base.as_strided(
      sizes=(2, 2, 2),
      strides=(2, 3, 0),
      storage_offset=1
    )

    # Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
    # StableHLO: slice, reshape, slice, transpose, reshape, broadcast_in_dim
    logical_view = base[1:].view(3, 3)[:2, ::2].t().unsqueeze(-1).expand(2, 2, 2)
    ```

    The low-level stride manipulation version using `torch.Tensor.as_strided` can be equivalently expressed using a sequence of more familiar operations, which then can be expressed in StableHLO and compiled by XLA.

    It is important to note, however, that **logical equivalence does not mean equivalent performance**. A reshape or transpose operation may, in some cases, require a full copy on TPU to map from one register-aligned layout to another. This means that, on TorchTPU, **views are not "free"** as they often are on CUDA, due to the physical differences between the GPU and TPU hardware.

    ## In-place mutations and buffer donation
    StableHLO is a functional programming language. Even more specifically, it is a [static single assignment (SSA) language](https://en.wikipedia.org/wiki/Static_single-assignment_form). This means that values can only be created fresh–never changed or reassigned. This assists the XLA compiler in producing optimized execution plans.

    Neither GPU nor CPU IRs are purely functional, and as a consequence, neither is PyTorch. Tensor operations suffixed with an underscore, or which use the "`out=`" keyword argument, will overwrite the bytes in their base `torch.UntypedStorage`. Because multiple `torch.Tensors` can share a common storage buffer, a write to any tensor becomes immediately visible to all other Tensors.

    TorchTPU handles this by exploiting our metadata layer. When a PyTorch operation requests an in-place update or out-tensor assignment, TorchTPU will:
     * Describe the logical update to XLA, producing a new buffer
     * Swap the buffer inside the metadata object pointed to by the `torch.UntypedStorage`

    This is essentially automatic functionalization: all mutations are expressed instead as out-of-place operations.

    Achieving automatic functionalization requires being able to run the view decomposition algorithm in reverse, which we call view inversion. This is only possible for non-overlapping views; writing to a tensor that resulted from a broadcast is invalid as multiple logical locations correspond to a single physical location, making the instruction ambiguous. On the CUDA backend, the operation would be considered undefined behavior but may or may not explicitly fail; on TorchTPU we always explicitly fail.

    In the case of a partial write, the view inversion uses a StableHLO operation called [`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) to compute a merged output tensor, containing both the overwritten and non-overwritten values in the correct locations.

    Of course, an out-of-place operation results in double the peak memory; both the old and new versions briefly coexist before the old value is discarded. To avoid this, TorchTPU will opportunistically try to apply XLA’s closest equivalent to in-place mutations: **buffer donation**. When the last reference to an obsolete buffer is removed, TorchTPU will annotate the StableHLO to “donate” the input, allowing XLA to consume the input, preventing the memory increase from auto-functionalization.

    Buffer donation is not always possible to do safely. If TorchTPU cannot prove a buffer is safe to donate, it will fall back to an out-of-place update. This can be particularly expensive for small writes to large buffers; in this case, the out-of-place update will copy the entire base storage tensor, not just the part being written to!
    
    ## Copy-on-write
    Functionalization has advantages, however. One benefit is the ability to embrace copy-on-write behavior.

    On the CUDA backend, calling `torch.Tensor.clone()` will immediately allocate a new physical memory buffer large enough to hold the Tensor’s data. After this, any in-place writes to the different `torch.UntypedStorages` will be fully independent, but the memory cost is paid immediately.

    On the TorchTPU backend, however, the immutability of a statically-assigned value means that a copy can simply be a reference to the original; with no immediate physical copy required.

    As soon as there is an in-place update we must reflect that update only on the `torch.UntypedStorage` that was modified and not the other. This prevents buffer donation (as we still require both the old and new values), so in this case the in-place update is performed out-of-place with no donation. This out-of-place update will allocate new memory–but this memory is only allocated lazily on an as-needed basis, unlike PyTorch/CUDA which pays the memory cost up-front.
    
    ## Custom kernels
    TorchTPU makes a best-effort attempt to apply memory-optimizing buffer donations, and provides a logically-equivalent transformation from CUDA strided views to StableHLO view operations. That said, these are still high-level operations, not directly tied to the physical layout of a TPU buffer, and do not guarantee any particular low-level semantics.

    For users who want the ultimate low-level control, TorchTPU also supports custom kernel integration. These allow for surgical replacement of critical operations, or even full-model rewrites, with manual specification of TPU buffer behavior. This can be used to ensure optimal donation and in-place update behavior.
    
    ## Takeaways
    When working with TorchTPU, users should know the following key facts:

     * **PyTorch code will produce the same logical result on both CUDA and TorchTPU:** You can use familiar low-level CUDA operations (shared-storage views, stride manipulation, in-place updates) and achieve the same results on TPU, although the performance may not be the same.
     * **Copies are inexpensive:** Copying a tensor into a contiguous form is often computationally free, due to copy-on-write behavior.
     * **Views are expensive:** Reading a tensor as a different shape or non-contiguous pattern may force a relayout on the TPU, which may be expensive. Prefer contiguous tensors when possible.
     * **Overwriting a contiguous tensor is usually memory-efficient:** A contiguous tensor can usually be overwritten with a buffer donation if all old values are erased.
     * **Partial writes are sometimes memory-expensive:** if buffer donation is not possible, the memory cost of a write is proportional to the size of the torch.UntypedStorage, not the size of the written data being modified. Prefer using many small tensors rather than slices of large ones.
     * **Custom kernels give full control:** where the CUDA-to-XLA translation fails or is insufficient, implementing a custom TPU-native kernel can optimize for specific low-level performance.

    """)
  return


if __name__ == "__main__":
  app.run()
