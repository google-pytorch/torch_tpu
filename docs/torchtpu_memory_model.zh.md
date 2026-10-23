# TorchTPU 与 TPU 内存模型 { #torchtpu-and-the-tpu-memory-model }

PyTorch 一直与 GPU 共同发展，有效地将 GPU 的强大能力提供给 Python 用户。TorchTPU 扩展了 PyTorch，使其能够在 TPU 上发挥 XLA 的强大能力。

GPU 和 TPU 的物理硬件不同，面向这两种硬件架构的编程范式也不同。TorchTPU 的存在就是为了从两个方向弥合这一差距。我们的目标是在 XLA 中支持张量视图和 stream 等 GPU 原生接口，同时在熟悉的 PyTorch 接口中展现 XLA 编译器的独特能力。

了解 TorchTPU 如何实现这些目标，可以帮助用户优化 TPU 原生代码库，从而获得最大收益。

## 硬件：CPU、GPU、TPU { #hardware-cpus-gpus-tpus }

通用 CPU（例如为您的计算机或移动设备提供算力的 CPU）由多个核心组成，每个核心支持一个或多个线程。每个线程都能并行执行，同时还能以随机读写的方式访问共享内存。

GPU 在许多方面与 CPU 类似；但是，GPU 的线程数量远多于 CPU。一个现代高性能 CPU 可能有几十个线程，而一个高性能 GPU 则有*数千个*线程。代价是每个 GPU 线程单独来看都比 CPU 线程弱一些，而且线程的独立性也存在一些限制。尽管如此，这仍使 GPU 非常适合具有高度可并行化操作的工作负载，例如图形渲染或线性代数。

TPU 的设计则截然不同。TPU 的核心是 TensorCore 及其矩阵乘法单元（MXU）。这些 MXU 以[脉动阵列](https://en.wikipedia.org/wiki/Systolic_array)的方式运行，这在某种程度上类似于 CPU 上的 SIMD 寄存器。它们使用宽的二维寄存器，每个时钟周期能执行的密集计算工作量远远超过 CPU 或 GPU 线程。

然而，TensorCore 本质上是单线程的；无法将脉动阵列细分为多个部分来同时执行多项计算。此外，由于寄存器很宽，TensorCore 非常不擅长执行稀疏的随机读写操作，因为需要一次性加载一大块内存。

最新几代 TPU 在 TensorCore 之外增加了 SparseCore，顾名思义，SparseCore 专门用于稀疏运算，包括更精确的读写操作。这些独立的核心还提供了有限的并行能力，每个 SparseCore 都能与每个 TensorCore 同时执行操作。

## 编程模型：CUDA 与 XLA { #programming-model-cuda-vs-xla }

GPU 和 TPU 的原生软件接口是根据这些平台的能力量身定制的。

以 CUDA 为例，它是支撑 PyTorch 的最常用的 GPU 编程模型。CUDA 是一种底层编程接口，类似于 C 或 C++。与 CPU 内存一样，CUDA 缓冲区本质上是一维字节数组。CUDA kernel 是一个命令式过程，用 C 语言的术语来说，就是一个 void 函数。每个线程都会获得指向某些缓冲区的指针，并且能够从任意缓冲区的任意偏移位置读取数据，并逐字节写回。

这种编程模型不适用于 TPU。TPU 上的内存不是按一维字节数组对齐的，而是按与寄存器对齐的二维 tile 对齐。而且如前所述，TensorCore 没有线程，也不擅长随机访问。

相反，TPU 编译器 XLA 使用的是一种声明式的函数式模型。XLA 接受用一种名为多级中间表示（MLIR）的语言编写的程序，MLIR 包含多种方言。使用 StableHLO 方言，可以描述程序期望的逻辑张量输出。XLA 会将该程序编译为针对您的特定 TPU 代际优化的底层内部操作。

## 缓冲区：`torch.UntypedStorage` { #buffers-torchuntypedstorage }

PyTorch 的核心工具与 CUDA 编程模型非常契合。

使用 Python 类 `torch.UntypedStorage` 的存储对象，基本上只是一个内存地址（`torch.UntypedStorage.data_ptr()`）加上一个以字节为单位的大小（`torch.UntypedStorage.nbytes()`）。这与 CPU 和 CUDA 接口都非常匹配。

然而，这不足以支持在 TPU 上执行；XLA 中的缓冲区始终具有明确的类型和已知的多维形状。由于 TPU 的内部内存与 2D 寄存器对齐，了解缓冲区的布局方式至关重要。这意味着仅靠 `torch.UntypedStorage` 无法携带足够的信息来支持 TPU 执行。

为了解决这个问题，TorchTPU 增加了一个额外的层。在 TPU 上，`torch.UntypedStorage.data_ptr` 不会直接指向 TPU 内存，而是指向 CPU 内存中的一个位置，该位置包含有关 TPU 缓冲区的额外元数据。**这是 TorchTPU 运作的关键**；这些额外的元数据使得 PyTorch 现有的类 CUDA API 与 XLA 的函数式设计和 TPU 硬件的实际情况之间能够进行主动转换。

## `torch.Tensor` { #torchtensor }
`torch.Tensor` 可以同时被看作两种东西。

首先，`torch.Tensor` 是底层 `torch.UntypedStorage` 上某种访问模式的*物理*表示。`Tensor` 包含以下基本元数据：stride（乘法因子）、size（索引边界）、offset（第 0 个元素的位置）以及元素类型（设置每个逻辑元素的字节数）。这些元数据提供了一种映射方式，可以将张量的索引元组映射为支撑该 `Tensor` 的线性 1D 缓冲区数组中的字节偏移。

其次，`torch.Tensor` 是数学对象的*逻辑*表示。将两个张量进行矩阵乘法时，结果是另一个具有适当形状、表示运算结果的 `Tensor`。
stride 和存储偏移的确切模式在这里并不重要；`Tensor` 作为一项计算所代表的含义，比张量作为内存引用是什么更重要。

从 CUDA 迁移到 TPU 时，理解这一区别至关重要：即使设备的**物理**实际情况本质上不同，PyTorch 对张量执行操作的**逻辑**行为也能在 XLA/TPU 上正确实现！

## 视图分解 { #view-decomposition }

在 CUDA 上运行时，`torch.Tensor` 的 stride 描述会被内部 CUDA kernel 直接使用；stride 会提供给内部线程过程，用于描述如何在相关缓冲区中计算偏移。这些偏移是在运行时动态计算的。

但在 TPU 上，物理 stride 无法直接使用；一维的 stride 描述不适用于与寄存器对齐的二维内存。不过，StableHLO *确实*支持视图的逻辑行为。大多数张量视图操作都有直接对应的 StableHLO 操作：reshape、transpose、slice 和 broadcast。

当收到一个 `torch.Tensor` stride 模式时，TorchTPU 会执行一项名为*视图分解*（view decomposition）的操作。该操作会将 stride 模式和元素类型解释为一系列逻辑操作，这些操作会生成所需的输出张量，并使每个元素都位于正确的位置。

例如：

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

使用 `torch.Tensor.as_strided` 的底层 stride 操作版本可以等效地表示为一系列更熟悉的操作，这些操作随后可以用 StableHLO 表示并由 XLA 编译。

但需要注意的是，**逻辑上等价并不意味着性能相同**。在某些情况下，reshape 或 transpose 操作可能需要在 TPU 上进行完整复制，才能从一种与寄存器对齐的布局映射到另一种布局。这意味着，由于 GPU 与 TPU 硬件的物理差异，在 TorchTPU 上，**视图并不像在 CUDA 上通常那样是“免费”的**。

## 原地修改和缓冲区捐赠 { #in-place-mutations-and-buffer-donation }

StableHLO 是一种函数式编程语言。更具体地说，它是一种[静态单赋值（SSA）语言](https://en.wikipedia.org/wiki/Static_single-assignment_form)。这意味着值只能新建，永远不能更改或重新赋值。这有助于 XLA 编译器生成经过优化的执行计划。

GPU 和 CPU 的 IR 都不是纯函数式的，因此 PyTorch 也不是。以下划线为后缀或使用“`out=`”关键字参数的张量操作会覆盖其底层 `torch.UntypedStorage` 中的字节。由于多个 `torch.Tensors` 可以共享同一个存储缓冲区，对任何张量的写入都会立即对所有其他张量可见。

TorchTPU 利用我们的元数据层来处理这一问题。当 PyTorch 操作请求原地更新或 out 张量赋值时，TorchTPU 会：

*   向 XLA 描述该逻辑更新，生成一个新的缓冲区
*   替换 `torch.UntypedStorage` 所指向的元数据对象中的缓冲区

这本质上是自动函数化（functionalization）：所有修改都改为以非原地操作的形式表示。

要实现自动函数化，需要能够反向运行视图分解算法，我们称之为视图反演（view inversion）。这仅适用于不重叠的视图；写入由广播产生的张量是无效的，因为多个逻辑位置对应同一个物理位置，会使指令产生歧义。在 CUDA 后端上，这种操作被视为未定义行为，可能会也可能不会显式失败；而在 TorchTPU 上，我们始终会显式失败。

对于部分写入，视图反演会使用一个名为 [`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice) 的 StableHLO 操作来计算合并后的输出张量，其中被覆盖的值和未被覆盖的值都位于正确的位置。

当然，非原地操作会使峰值内存翻倍；在旧值被丢弃之前，新旧两个版本会短暂共存。为避免这种情况，TorchTPU 会伺机尝试应用 XLA 中最接近原地修改的机制：**缓冲区捐赠**（buffer donation）。当对某个过时缓冲区的最后一个引用被移除时，TorchTPU 会在 StableHLO 中添加注解以“捐赠”该输入，使 XLA 可以直接占用该输入，从而避免自动函数化带来的内存增长。

缓冲区捐赠并不总能安全地进行。如果 TorchTPU 无法证明某个缓冲区可以安全捐赠，就会回退为非原地更新。对于向大缓冲区进行的小规模写入，这种方式的开销可能特别大；在这种情况下，非原地更新会复制整个底层存储张量，而不仅仅是被写入的部分！

## 写时复制 { #copy-on-write }

不过，函数化也有其优势。其中一个好处是能够采用写时复制（copy-on-write）行为。

在 CUDA 后端上，调用 `torch.Tensor.clone()` 会立即分配一个足以容纳该张量数据的新物理内存缓冲区。此后，对不同 `torch.UntypedStorages` 的任何原地写入都将完全独立，但内存开销会立即产生。

而在 TorchTPU 后端上，静态赋值的值是不可变的，这意味着副本可以只是对原始值的引用，无需立即进行物理复制。

一旦发生原地更新，我们就必须只在被修改的 `torch.UntypedStorage` 上反映该更新，而不能影响另一个。这会阻止缓冲区捐赠（因为我们仍然需要新旧两个值），因此在这种情况下，原地更新会以非原地的方式执行，且不进行捐赠。这种非原地更新会分配新的内存，但这些内存只会在需要时延迟分配，而不像 PyTorch/CUDA 那样预先支付内存开销。

## 自定义 kernel { #custom-kernels }

TorchTPU 会尽力应用能够优化内存的缓冲区捐赠，并提供从 CUDA 带 stride 视图到 StableHLO 视图操作的逻辑等价转换。尽管如此，这些仍然是高层操作，并不直接与 TPU 缓冲区的物理布局绑定，也不保证任何特定的底层语义。

对于希望获得极致底层控制的用户，TorchTPU 还支持集成自定义 kernel。借助自定义 kernel，您可以精确地替换关键操作，甚至重写整个模型，并手动指定 TPU 缓冲区的行为。这可以用来确保最佳的捐赠和原地更新行为。

## 要点总结 { #takeaways }
使用 TorchTPU 时，用户应了解以下关键事实：

*   **PyTorch 代码在 CUDA 和 TorchTPU 上会产生相同的逻辑结果**：您可以使用熟悉的 CUDA 底层操作（共享存储视图、stride 操作、原地更新），并在 TPU 上获得相同的结果，但性能可能不同。
*   **复制的开销很低**：由于采用写时复制行为，将张量复制为连续形式在计算上通常是免费的。
*   **视图的开销很高**：以不同形状或非连续模式读取张量，可能会强制在 TPU 上重新布局，而这可能开销很大。请尽可能使用连续张量。
*   **覆盖连续张量通常能高效利用内存**：如果所有旧值都会被擦除，通常可以通过缓冲区捐赠来覆盖连续张量。
*   **部分写入有时会占用大量内存**：如果无法进行缓冲区捐赠，写入的内存开销将与 torch.UntypedStorage 的大小成正比，而不是与被修改的写入数据的大小成正比。请优先使用许多小张量，而不是大张量的切片。
*   **自定义 kernel 提供完全的控制**：如果从 CUDA 到 XLA 的转换失败或不够理想，可以实现自定义的 TPU 原生 kernel，以针对特定的底层性能进行优化。
