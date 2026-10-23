# TorchTPU の分散サポート { #torchtpu-distributed-support }

## 分散サポート { #distributed-support }

TorchTPU は、PyTorch の ProcessGroup レベル（`c10d::Backend`）で、`tpu_dist` というバックエンド名で分散通信を実装しています。ProcessGroup レイヤで集団通信をインターセプトすることで、TorchTPU は eager モードと `torch.compile` の両方で標準の `torch.distributed` API をサポートし、DTensor、DistributedDataParallel（DDP）、FSDPv2、[SimpleFSDP](https://arxiv.org/abs/2411.00284) などの PyTorch 分散フレームワークとそのまま連携します。

GPU 上の PyTorch と同様に、TorchTPU はデバイスごとに 1 プロセスという実行モデルを採用しています。内部的には、TPU 間の通信に別個のランタイム通信ライブラリ（NCCL など）は使いません。代わりに、集団通信演算は [StableHLO の集団通信演算](https://openxla.org/stablehlo/spec#collective_ops)（`stablehlo.all_reduce`、`stablehlo.all_gather`、`stablehlo.reduce_scatter`、`stablehlo.all_to_all` など）に直接 lowering され、XLA によって TPU 実行可能プログラムにコンパイルされます。

### GPU とのアーキテクチャ上の主な違い { #key-architectural-differences-from-gpu }

TorchTPU は標準の `torch.distributed` のセマンティクスを維持していますが、StableHLO と PjRT を介して TPU をプログラミングする場合、GPU バックエンドと比べて 3 つの重要な違いがあります。

*   **初期化前のハードウェア トポロジの検出:** マルチ GPU ホストでは、torchrun が起動した各プロセスは接続されているすべての GPU を調べ、`cuda:{local_rank}` で割り当てられたデバイスを選び、`torch.distributed.init_process_group()` を呼び出す前にモデルをそのデバイスに移動できます。TPU ホストでは、各ワーカー プロセスは単一の TPU デバイスにバインドされ、単独でデバイス全体を所有することはできません。PjRT ランタイムを初期化する前、または `torch.distributed.init_process_group()` を呼び出す前に、環境で [TPU スライス](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)の物理メッシュのシェイプ（`TORCH_TPU_TOPOLOGY`）、Inter-Chip Interconnect（ICI）メッシュの構築に使うローカル コーディネータのアドレス（`TORCH_TPU_SLICEBUILDER_ADDRESSES`）、参加する TPU デバイスの総数（`WORLD_SIZE`）を指定する必要があります。
*   **サブグループに対するグローバルなレプリカ グループの解決:** PyTorch では、[`torch.distributed.all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) のような集団通信の呼び出しは、呼び出し元のランクが参加している ProcessGroup（サブグループ）だけを受け取ります。これに対して、[`stablehlo.all_reduce`](https://openxla.org/stablehlo/spec#all_reduce) では、現在のプロセスが属していないものも含め、デフォルトのプロセス グループ全体にわたる参加サブグループすべての完全な記述（`replica_groups`）が必要です。

    ```
    %0 = "stablehlo.all_reduce"(%arg0) <{
      replica_groups = dense<[[4, 0, 1, 3], [5, 2, 6, 7]]> : tensor<2x4xi64>
    }> ({ ... })
    ```

    TorchTPU の `tpu_dist` バックエンドは、集団通信のセットアップ中にデフォルトのプロセス グループ内の全ランク間でサブグループのメンバーシップを自動的に交換し、XLA が必要とするグローバルな `replica_groups` と `device_assignment` を構築します。
*   **SPMD コンパイルと MPMD 実行:** XLA コンパイラと TPU の通信スタックは Single Program, Multiple Data（SPMD）実行向けに設計されており、すべてのランクが、対応する集団通信演算を含む同一の StableHLO モジュールをコンパイルして実行します。しかし PyTorch のプログラムは、ランク間で Multiple Program, Multiple Data（MPMD）の動作を示すことがあります。たとえば、ランクに依存する分岐（`if dist.get_rank() == 0:`）や、次元がデバイス数で割り切れない場合の不均等なテンソル シャーディングが原因になります。異なるランクが集団通信を含む異なる StableHLO モジュールを実行すると、ランク間で同期フラグの ID やバッファ オフセットが一致せず、実行時のデッドロックやメモリ破損が発生する可能性があります。

### 集団通信の実行モードと融合モード { #collective-execution-and-fusion-modes }

TorchTPU は、各集団通信をスキニー集団通信（skinny collective）として実行します。XLA が計算と通信をオーバーラップできるようにするには、集団通信を周囲の計算と融合する 2 つの仕組みのいずれか、または両方をオプトインで有効にできます。この 2 つは独立しており、組み合わせることもできます。`@spmd_safe` 関数内の集団通信は常に融合され、有効にした場合は、各 `torch.compile` グラフの残りの集団通信についてコンパイル ハンドシェイクが判断します。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| モード | 設定 | グラフの動作 | 最適な用途 |
| :---- | :---- | :---- | :---- |
| スキニー集団通信（デフォルトの動作） | 以下のオプトインで融合されない集団通信には常に適用 | 集団通信の前後にグラフ ブレークを挿入し、各集団通信演算を独立した（「スキニー」な）StableHLO モジュールにコンパイルします。 | 追加設定なしで一般的な MPMD の安全性を確保できます。ただし、XLA による計算と通信のオーバーラップができなくなります。 |
| `@spmd_safe` デコレータ（オプトイン） | `@torch.tpu.distributed.spmd_safe` | デコレートした関数内の集団通信の前後でグラフ ブレークを無効にし、計算と集団通信を単一の StableHLO モジュールに融合します。 | SPMD であることがわかっている関数やトレーニング ステップ（例: SimpleFSDP や手動のテンソル並列）で、XLA による計算と通信のオーバーラップを最大化します。 |
| コンパイル ハンドシェイク（オプトイン） | `TORCH_TPU_SPMD_HANDSHAKE_STAGE=COMPILE_STAGE` または `DISPATCH_STAGE`（デフォルト: OFF） | torch.compile のグラフで、コンパイルされた FX グラフがすべてのランク間で一致するかを確認します。一致すれば集団通信を融合し、一致しなければスキニー集団通信にフォールバックします。 | 偶発的な MPMD の分岐を防ぎつつ、`torch.compile` で集団通信の融合を自動的に有効にします。 |
<!-- mdformat on -->

#### `@torch.tpu.distributed.spmd_safe` を使う { #using-torchtpudistributedspmd_safe }

関数やモデルのステップがすべてのランクで厳密に SPMD である（ランクに依存する制御フローや不均等なテンソル シェイプがない）ことがわかっている場合は、`@torch.tpu.distributed.spmd_safe` でデコレートします。`torch.compile`（および `DEFER_AND_FUSE` の eager モード）では、これにより集団通信の前後のグラフ ブレークが取り除かれ、XLA が非同期通信を行列計算と並行してスケジュールできるようになります。

```python
import torch
import torch.distributed as dist

@torch.compile
@torch.tpu.distributed.spmd_safe
def spmd_train_step(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
  out = x @ weight
  dist.all_reduce(out, op=dist.ReduceOp.AVG)
  return torch.nn.functional.silu(out)
```

`@torch.tpu.distributed.spmd_safe` は、すべてのランクでまったく同じ順序のテンソル演算と集団通信演算を実行するコードにのみ適用してください。MPMD のコードに適用すると、ランク間で異なる StableHLO モジュールが生成され、TPU の実行がデッドロックする可能性があります。

#### コンパイル ハンドシェイクを設定する { #configuring-the-compilation-handshake }

`@spmd_safe` で関数に手動でアノテーションを付ける代わりに、`TORCH_TPU_SPMD_HANDSHAKE_STAGE` 環境変数で TorchTPU のコンパイル ハンドシェイクを有効にできます。グローバルな集団通信を含むグラフを `torch.compile` がトレースすると、TorchTPU は集団通信をグラフに融合する前に、すべてのランク間で合意チェックを実行し、コンパイルされた FX グラフが一致することを確認します。

*   `OFF`（デフォルト）: ハンドシェイクを無効にします。`@torch.tpu.distributed.spmd_safe` でマークされていない限り、集団通信はデフォルトのスキニー集団通信を使います。
*   `COMPILE_STAGE`: PjRT コンパイルの前に、ランク間でグラフの合意を確認します。すべてのランクが同一のグラフを生成した場合、集団通信の前後のグラフ ブレークは省略されます。すべてのランクが一緒にコンパイルする標準的な `torch.compile` ワークロードにおすすめです。
*   `DISPATCH_STAGE`: 集団通信をグラフ ブレークなしでコンパイルし、ディスパッチ時にランク間の合意を確認します。ランク間で分岐した場合は、分割した集団通信で再コンパイルします。このモードは、ランク固有の再コンパイルを引き起こす可能性があるワークロードに必要です。`COMPILE_STAGE` よりもオーバーヘッドが大きくなります。

以下は、`DISPATCH_STAGE` ではサポートされるランク固有の再コンパイルの例です。`COMPILE_STAGE` では、イテレーション 1 で `divergent_rank` だけが再コンパイルするため、タイムアウトします。

```python
iteration = 0
divergent_rank = ...

def divergence_after_iteration(x, y):
  z = torch.add(x, y)
  if rank == divergent_rank and iteration == 1:
    z = torch.mul(z, 5.0)
  dist.all_reduce(z)
  return z

compiled_fn = torch.compile(divergence_after_iteration, backend=backend)
res0 = compiled_fn(x, y)
iteration += 1
# IMPORTANT: If Handshake is in COMPILE_STAGE a recompilation happens only on `divergent_rank`
# this would initiate a Handshake ONLY ON THAT RANK which would timeout which is why we provide
# a DISPATCH_STAGE mode.
res1 = compiled_fn(x, y)
```

## 分散の初期化 { #distributed-initialization }

TorchTPU で分散ワークロードを初期化するには、標準の `torch.distributed` API と、TPU トポロジとスライス ビルダーの環境変数を設定する TorchTPU の起動環境ヘルパーを組み合わせて使います。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 環境変数 | 説明 | 例 |
| :---- | :---- | :---- |
| `TORCH_TPU_TOPOLOGY` | TPU スライスの物理メッシュのシェイプです。 | `"2,2,1"`、`"2,2,1,2"`、`"4,4,4"` |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | ICI メッシュの構築に使う `host:port` 形式のアドレス（デバイスごとに 1 つ）をカンマ区切りで指定します。 | `"localhost:8471,localhost:8472,..."` |
<!-- mdformat on -->

これらの変数は、ワーカーを起動する前に設定する必要があります。一般的な構成（単一ホスト、または GKE の TPU スライス）では、TorchTPU が正しい値を判断できます。次のコマンドを

```bash
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
```

torchrun の前に起動スクリプト内で実行するか、ワーカーを起動する前に親の Python ランチャー プロセスで `torch.tpu.distributed.set_tpu_launch_env()` を呼び出してください。どちらの方法でも、すでに設定済みの変数は変更されないため、個々の値をオーバーライドできます。

以下は、分散 TorchTPU ワークロードの初期化フローと、ワーカーとランチャーの完全な例です。

![分散初期化のフロー](./images/distributed.png)

### 1. ワーカー スクリプト（`train.py`） { #1-worker-script-trainpy }

各ワーカー プロセス内で、`tpu_dist` バックエンドを初期化し、`torch.device("tpu")` でローカルの TPU デバイスを取得します。

```python
import torch
import torch.distributed as dist

def worker_main():
  # 1. Initialize the distributed process group using the tpu_dist backend`
  dist.init_process_group(backend="tpu_dist")

  # 2. Query rank, world size, and acquire the local TPU device
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = torch.device("tpu")

  print(f"[Rank {rank}/{world_size}] Initialized on {device}")

  # 3. Perform a distributed collective operation across TPU workers
  x = torch.ones(4, 4, device=device) * (rank + 1)
  dist.all_reduce(x, op=dist.ReduceOp.SUM)

  print(f"[Rank {rank}] Reduced sum: {x[0, 0].item()}")

  # 4. Clean up process group resources
  dist.destroy_process_group()

if __name__ == "__main__":
  worker_main()
```

### 2. torchrun で起動する { #2-launching-with-torchrun }

`torchrun` を呼び出す前に、`torch.tpu.distributed` CLI モジュールを使って TPU ハードウェアのトポロジを検出し、必要な起動変数をエクスポートします。

```bash
# Detect and export TORCH_TPU_TOPOLOGY, TORCH_TPU_SLICEBUILDER_ADDRESSES,
# WORLD_SIZE, NODE_RANK, and multi-host coordination variables:
eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
# Launch 1 worker process per TPU device on the host:
torchrun --nproc_per_node=$WORLD_SIZE train.py
```

または、Python からワーカーを起動する場合（例: `torch.distributed.launcher.elastic_launch` を使う場合）は、ワーカーを起動する前に親プロセスで `torch.tpu.distributed.set_tpu_launch_env()` を呼び出します。デフォルトのプロセス グループには、[TPU スライス](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)内のすべてのデバイスが含まれることに注意してください。

## 最適なデバイス メッシュのマッピング { #optimal-device-mesh-mapping }

集団通信の帯域幅を最大化し、レイテンシを最小化するには、モデルの並列化の次元を TPU の物理的な相互接続の階層に合わせます。

*   **Inter-Chip Interconnect（ICI）**: スライス内で、2D/3D のトーラスまたはツイステッド トーラスのメッシュとして TPU チップ同士をつなぐ高速な直接リンクです（リンクあたり数十から数百 GB/s）。
*   **Data Center Network（DCN）**: マルチスライスのワークロードで、別々の TPU スライスをつなぐホストレベルのイーサネットです（ICI より帯域幅が低く、レイテンシが高くなります）。

`torch.tpu.distributed.topology_aware_mesh()` を使うと、物理的な ICI リングに合わせたランク レイアウトを計算できます。論理メッシュの次元は、**通信負荷が最も低いもの**（最も外側）から**通信負荷が最も高いもの**（最も内側）の順に並べます。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 並列化の種類 | 代表的な集団通信 | 通信頻度 | 推奨される物理配置 |
| :---- | :---- | :---- | :---- |
| **テンソル並列（TP）** | `all_reduce`、`reduce_scatter`、`all_gather` | Transformer の各レイヤ（フォワード パス + バックワード パス） | 最速のローカル ICI リング（同じトレイ/ホスト、1 ホップで隣接するデバイス） |
| **コンテキスト / シーケンス並列（CP/SP）** | `all_to_all`、Ring Attention | アテンション レイヤごと | ローカルの ICI 軸（スライス内） |
| **エキスパート並列（EP）** | `all_to_all`、`ragged_all_to_all` | MoE ルーター レイヤごと | ローカルの ICI 軸（スライス内） |
| **完全シャーディング データ並列（FSDP）** | `all_gather`、`reduce_scatter` | モジュール / Transformer ブロックごと | ICI 軸（スライス内） |
| **レプリケーション データ並列（DP）** | `all_reduce`（勾配のみ） | トレーニング ステップごとに 1 回 | 外側の ICI 軸または DCN（スライス間） |
<!-- mdformat on -->

### 例: トポロジを考慮した 2D DeviceMesh を構築する { #example-constructing-a-topology-aware-2d-devicemesh }

```python
import torch
from torch.distributed.device_mesh import DeviceMesh

# Logical mesh shape: (dp=4, tp=2) on an 8-device slice,
# ordered from least communication-intensive (dp) to most intensive (tp):
mesh_shape = (4, 2)
dim_names = ("dp", "tp")

# Compute an optimal rank tensor aligned with physical ICI rings
optimal_rank_tensor = torch.tpu.distributed.topology_aware_mesh(
    mesh_shape=mesh_shape,
)

# Instantiate the PyTorch DeviceMesh
mesh_2d = DeviceMesh(
    "tpu",
    mesh=optimal_rank_tensor,
    mesh_dim_names=dim_names,
    preserve_rank_order=True,
)
dp_mesh = mesh_2d["dp"]
tp_mesh = mesh_2d["tp"]
```

マルチスライス（DCN + ICI のハイブリッド）のワークロードでは、`mesh_shape` と一緒に `dcn_mesh_shape` を渡すことは任意で、分割を明示的に選びたい場合にのみ必要です。デフォルトでは、`torch.tpu.distributed.topology_aware_mesh()` がスライスを自動的に検出し、最も外側の自明でないメッシュ次元（通常はデータ並列）を DCN 経由でスライス間に、内側の次元を ICI 経由で各スライス内にマッピングします。

## 分散 Pallas カーネル { #distributed-pallas-kernels }

TorchTPU は、ホスト CPU を介さずに ICI 経由でデバイス間の直接 Remote DMA（RDMA）を行うカスタムの [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) カーネルをサポートしています。これにより、Ring Attention、カスタムの All-to-All シャッフル、パイプライン並列のシフト、ハロー交換などのカスタムのポイントツーポイント通信カーネルやリング通信カーネルを作成し、eager モードと `torch.compile` の両方でネイティブな PyTorch カスタム演算として呼び出せます。

### 通信を行う Pallas カーネルの作成と登録 { #writing-and-registering-a-communicating-pallas-kernel }

分散 Pallas カーネルの統合は、3 つのステップで行います。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 段階 | 主なステップと演算 |
| :---- | :---- |
| **1. Pallas カーネルの本体** | `jax.lax.axis_index` と `jax.lax.axis_size` で論理座標を取得し、`pltpu.make_async_remote_copy(..., device_id=(target,))` で非同期転送記述子を作成して、`.start()` と `.wait()` で完了を管理します。 |
| **2. `pl.pallas_call` と `jax.shard_map`** | `scratch_shapes` に DMA セマフォ（`pltpu.SemaphoreType.DMA`）を割り当て、ランクに揃えたワールド サイズの `jax.sharding.Mesh` と `PartitionSpec` を使って、カーネルを `jax.shard_map` でラップします。 |
| **3. TorchTPU への登録** | `torch.tpu.pallas.jax_op` を使って型アノテーション付きの JAX ラッパー関数を登録し、`mesh`、`input_partition_specs`、`output_partition_specs` を渡します。 |
<!-- mdformat on -->

```python
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch

P = jax.sharding.PartitionSpec

# 1. Pallas Kernel Body: Asynchronously sends local buffer to neighbor rank (rank + 1)
def pallas_send_to_next_kernel(x_ref, y_ref, send_sem, recv_sem):
  rank = jax.lax.axis_index("x")
  world_size = jax.lax.axis_size("x")
  dest_id = (rank + 1) % world_size
  # Trigger hardware Remote DMA over TPU interconnect
  async_copy = pltpu.make_async_remote_copy(
      src_ref=x_ref,
      dst_ref=y_ref,
      send_sem=send_sem,
      recv_sem=recv_sem,
      device_id=(dest_id,),
  )
  async_copy.start()
  async_copy.wait()

# 2. JAX Wrapper with ShardMap and DMA Semaphores
def build_jax_fn(mesh):
  def local_pallas_call(x: jax.Array) -> jax.Array:
    return pl.pallas_call(
        pallas_send_to_next_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(x.shape)],
            out_specs=pl.BlockSpec(x.shape),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
            ),  # send & recv semaphores
        ),
    )(x)

  return jax.shard_map(
      local_pallas_call,
      mesh=mesh,
      in_specs=P("x"),
      out_specs=P("x"),
      check_vma=False,
  )

# 3. Create the TorchTPU callable wrapper
def get_send_to_next_op(world_size: int):
  mesh = torch.tpu.pallas.get_pallas_mesh(
      axis_names=("x",), mesh_shape=(world_size,)
  )
  jax_fn = build_jax_fn(mesh)
  return torch.tpu.pallas.jax_op(
      "pallas::send_to_next",
      jax_fn,
      mesh=mesh,
      input_partition_specs=[P("x")],
      output_partition_specs=[P("x")],
  )
```

### ランクに揃えたメッシュとデバイスの割り当て { #rank-aligned-meshes-and-device-assignment }

多くの TPU アーキテクチャでは、デバイス ID と割り当てられたランクは 1 対 1 に対応しません。ランクは隣接するデバイスが隣り合うように割り当てられます（ランク 1 はランク 0 とランク 2 の隣）が、このシナリオでの v7x のデバイス ID は 0、1、4 になります。この変換を簡単にするために、TorchTPU は `torch.tpu.pallas.get_pallas_mesh` を提供しています。これは、Torch アプリケーションの他の部分で使われるランク ID と正確に一致する Pallas/JAX メッシュを返します。

*   **ランクに揃えたメッシュの構築**: `torch.tpu.pallas.get_pallas_mesh(axis_names=..., mesh_shape=...,
    process_group=...)`
    は、アクティブな PyTorch の `ProcessGroup` のランクからデバイスへのマッピングを照会し、JAX の PjRT バックエンドを初期化せずに、コンパイル専用の抽象デバイスからなる `jax.sharding.Mesh` を構築します。
*   **ワールド サイズのメッシュの要件**: 分散 Pallas カーネルには、デバイスの総数が `dist.get_world_size()` と等しいメッシュが必要です。多次元の並列トポロジ（データ並列 + テンソル並列など）では、積が `world_size` と等しくなる `mesh_shape` を渡します。例: `torch.tpu.pallas.get_pallas_mesh(axis_names=("dp", "tp"), mesh_shape=(2,
    4))`。
*   **集団通信とカーネルのデバイス割り当ての統一**: `torch.tpu.pallas.jax_op` に `mesh` を渡すと、TorchTPU はメッシュのランクからデバイスへのマッピングをカーネルの XLA `DeviceAssignment` として取り出し、同じコンパイル済みモジュールに融合された標準の `torch.distributed` 集団通信と整合させます。これにより、`jax.lax.axis_index`、メッシュでアドレス指定された DMA（`device_id={axis: target}`）、PyTorch の集団通信がすべて、PyTorch のランク順に従って一貫してルーティングされることが保証されます。
