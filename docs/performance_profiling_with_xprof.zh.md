# 使用 XProf 进行性能分析 { #performance-profiling-with-xprof }

在 TPU 上训练大型模型时，了解时间花在了哪里至关重要。本指南介绍如何使用 TPU 性能分析器（TPU Profiler）捕获执行跟踪记录（trace），并在 TensorBoard 中进行分析。

> **Warning:** **前提条件**：TensorBoard 需要 `setuptools<81`。最新版本移除了 TensorBoard 所依赖的 `pkg_resources`。请先安装：
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **PyTorch 版本**：原生 `torch.profiler` TPU 集成需要 PyTorch 2.12 或更高版本。

## 1. TPU 性能分析器 { #1-the-tpu-profiler }

标准的 `torch.profiler`（已与 `torch_tpu` 集成）提供了一个**上下文管理器**接口，用于跟踪 CPU 和 TPU 的活动。您只需将要分析的代码封装在 `torch.profiler.profile()` 块中，收集到的跟踪记录就会保存到磁盘，以便在 TensorBoard 中查看。

**主要组件：**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 组件                                  | 作用                            |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | 用于启动/停止跟踪记录收集的 <br> 标准上下文管理器 |
| `torch.profiler.ProfilerActivity.CPU`      | 跟踪 CPU 端操作（分派、数据 <br> 加载） |
| `torch.profiler.ProfilerActivity.TPU`      | 跟踪 TPU 硬件执行（计算、 <br> 内存、互连） |
| `torch.profiler.tensorboard_trace_handler` | 将跟踪记录保存到目录中供 <br> TensorBoard 使用的回调       |
| `TpuProfilerConfig`                        | 用于自定义 TPU 性能分析配置的辅助类                   |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. 自定义性能分析配置（`TpuProfilerConfig`） { #2-custom-profiling-configuration-tpuprofilerconfig }

通过 `TpuProfilerConfig` 传递的选项可以精细地控制主机和设备跟踪器的级别以及输出目录结构。

### 认可的配置选项 { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 选项键            | 类型           | 默认值 | 说明               |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | 控制 CPU 主机跟踪器的详细程度 <br>（0=关闭，1=关键，2=全部）。 |
| `device_tracer_level` | `int`          | `1`     | 控制 TPU 硬件设备跟踪器的 <br> 详细程度（0=关闭，1=开启）。 |
| `python_tracer_level` | `int`          | `0`     | 控制 Python 函数跟踪的 <br> 级别（0=关闭，1=开启）。 |
| `run_dir`             | `pathlib.Path` | `None`  | 保存 TPU 硬件跟踪记录 <br> 文件（`.xplane.pb`）的输出目录。 |
| `worker_rank`         | `str`          | `None`  | 作为后缀附加到 <br> XPlane 文件名（`<hostname>_<rank>.xplane.pb`）的 worker rank。 |
<!-- mdformat on -->

```python
import pathlib
import torch
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
    python_tracer_level=0,
    run_dir=pathlib.Path("./profiler_output"),
    worker_rank="0",
)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
) as prof:
    model(inputs)
```

## 3. 捕获性能分析数据 { #3-capturing-a-profile }

下面的代码单元在性能分析器上下文中运行一个小型训练循环。要点如下：

1.  将训练循环封装在 `torch.profiler.profile()` 中
2.  指定活动：`CPU` 用于主机端操作，`TPU` 用于硬件执行
3.  将 `on_trace_ready` 设置为使用 `torch.profiler.tensorboard_trace_handler` 将跟踪记录保存到目录中
4.  （可选）传入带有 `TpuProfilerConfig` 的 `experimental_config`，以控制跟踪器级别

> **Important:** 性能分析器会增加开销。请仅对具有代表性的部分步骤（例如 100–1000 步）进行分析，而不是整个训练运行。

```python
# Build a small model and optimizer
model = (
    torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )
    .to(device)
    .to(torch.bfloat16)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

# Where to save the profiler traces
log_dir = "/tmp/profiler_output"

# Configure TPU-specific tracer levels
config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
)

# Profile 100 training steps with native PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
) as prof:
  for step in range(100):
    optimizer.zero_grad()
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    torch.accelerator.synchronize()
    prof.step()

print(f"✅ Profile captured: {100} steps saved to {log_dir}")
```

## 4. 在 TensorBoard 中查看跟踪记录 { #4-viewing-traces-in-tensorboard }

运行以下命令：

**第 1 步**：安装性能分析插件（显示信息中心所必需）：

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**第 2 步**：启动 TensorBoard：

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

XProf 也可以作为独立服务器运行。

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**第 3 步**：在浏览器中打开 `http://localhost:6006`，然后选择 **“Profile”** 标签页。

> **Note:** 您必须从 VS Code 集成终端运行 TensorBoard，端口转发才能自动生效。

**在 GKE 上**：将跟踪记录目录写入使用 [Cloud Storage FUSE CSI 驱动程序](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)装载的 Cloud Storage 存储桶，使跟踪记录在 Pod 结束后仍然保留，然后使用 `xprof
--logdir=gs://<bucket>/profiler_output` 打开它们。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 视图               | 显示内容                                          |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | 概要：步骤时间、设备利用率、耗时最多的算子 |
| **Trace Viewer**   | CPU 和 TPU 上每个算子的时间线：查找空隙（硬件空闲） |
| **Op Profile**     | 每个操作的时间明细：找出开销最大的算子 |
| **Memory Profile** | HBM 用量随时间的变化：检测内存峰值 |
<!-- mdformat on -->

## 5. 需要关注的内容 { #5-what-to-look-for }

### 跟踪记录中的危险信号 { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 模式                 | 含义                 | 解决方法                      |
| ----------------------- | ----------------------- | ------------------------ |
| TPU 算子之间存在较大空隙 | 主机端瓶颈 <br>（数据加载、Python 开销） | 预取数据，减少循环中的 Python 逻辑 |
| 频繁出现短暂的 TPU 突发   | 图中断过多 → 频繁重新编译 | 减少 `.item()` / `.cpu()` 调用，使用 `torch.compile` |
| MXU 利用率低 | 张量维度未与 tile 大小（128）对齐 | 将维度填充为 128 的倍数 |
<!-- mdformat on -->

### 理想的性能分析结果 { #the-ideal-profile }

经过充分优化的 TPU 工作负载会呈现：

-   **较长且连续的 TPU 计算块**：编译器将许多算子融合到了一次执行中
-   **CPU→TPU 空隙极小**：数据加载与计算相互重叠
-   **MXU 利用率高（>60%）**：张量形状与硬件对齐

## 6. 验证是否已捕获跟踪记录 { #6-verifying-the-traces-were-captured }

下面的代码单元会检查性能分析器输出目录中是否包含跟踪记录文件。

```python
import os

trace_files = os.listdir(log_dir)
print(f"Trace directory: {log_dir}")
print(f"Files found: {len(trace_files)}")
for f in sorted(trace_files)[:10]:
  size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
  print(f"  {f} ({size_kb:.1f} KB)")

if trace_files:
  print(f"\n✅ Traces ready. View with:")
  print(f"   tensorboard --logdir={log_dir}")
else:
  print("\n❌ No trace files found. Check profiler configuration.")
```

## 7. Kernel 性能分析（细粒度性能计数器） { #7-kernel-profiling-fine-grained-performance-counters }

Kernel 性能分析可以收集细粒度的硬件性能计数器（例如针对 Pallas kernel），以诊断在标准事件跟踪记录中无法看到的性能瓶颈。

> **Note:** 细粒度性能计数器采样是一项高级硬件功能，适用于 **Ironwood（TPU v7/TPU7x）**及后续代际的 TPU。

### 启用性能计数器 { #enabling-performance-counters }

您可以通过 `TpuProfilerConfig` 中的 `experimental_options` 传入计数器配置，从而启用性能计数器：

```python
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    experimental_options={
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
            " indices:11 indices:56 indices:57 indices:58"
        ),
    }
)
```

如需查看配置参数的详细列表（例如采样间隔、缩放和触发模式），以及如何确定计数器索引的说明，请参阅官方的 [OpenXLA Kernel 性能分析指南](https://openxla.org/xprof/kernel-profiling)。

## 8. 快速参考 { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
步骤        | 命令/代码
----------- | --------------
**导入**  | `from torch.tpu.profiler import TpuProfilerConfig`
**分析** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**保存**    | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**查看**    | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** 请仅分析 100–1000 步，以使跟踪记录文件保持在可管理的大小。性能分析器会为每一步增加约 5–10% 的开销。
