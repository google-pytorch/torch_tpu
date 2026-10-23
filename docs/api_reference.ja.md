# TorchTPU API リファレンス { #torchtpu-api-reference }

このドキュメントは、TorchTPU の公開 Python API と環境変数のリファレンス ガイドです。次のことに役立つように作られています。

*   **CUDA から TPU への移行**: よく使われる `torch.cuda` の呼び出しを、PyTorch のデバイス非依存の [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API に対応付けます。
*   **TPU 固有の機能の使用**: **`torch.tpu`** モジュールを使って、実行モード、MXU の数値精度、分散トポロジ、カスタムの Pallas/TPU 演算子、トレース可能なオプティマイザ、XProf プロファイリングを設定します。
*   **ランタイムの設定**: 環境変数を使って、コンパイル キャッシュ、デバッグ チェック、マルチホストの起動設定を調整します。

> **Note:** すべての公開 Python API と環境変数を網羅し、自動生成される信頼できる一覧については、[TorchTPU の公開 API（`public_apis.md`）](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md)をご覧ください。

## **コアのデバイス クエリ** { #core-device-queries }

TorchTPU がインストールされた状態で `torch` をインポートすると、バックエンドが `"tpu"` デバイスと `torch.tpu` モジュールを自動的に登録します。PyTorch がデバイス非依存の [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API を提供している場合は、バックエンド固有の API ではなくそちらを使ってください。そうすると、コードをアクセラレータ間で移植可能に保てます。汎用の同等 API がない TPU 固有のハードウェア プロパティには `torch.tpu` を使います。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / アクセラレータ API | 説明 |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | TPU がアクティブなアクセラレータの場合に `torch.device("tpu")` を返します。 |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | TPU バックエンドが利用可能かどうかを確認します。 |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | TorchTPU のランタイム状態が初期化済みかどうかを確認します。 |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | 現在のプロセスから見えるデバイスの数です。 |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | アクティブな TPU デバイスのインデックスを返します。 |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | 現在のプロセスのアクティブな TPU デバイスのインデックスを設定します。 |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | アクティブなデバイスのインデックスを一時的に設定するコンテキスト マネージャです。 |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | 接続されている TPU の人が読めるモデル名（例: `"TPU v6e"`）を返します。 |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | TPU デバイスの静的なプロパティ（`name`、`devices_per_chip`、`hbm_bytes_per_chip`、`hbm_bytes_per_device`、`bf16_flops_per_chip`、`bf16_flops_per_device`、`sparse_cores_per_device`、`sparse_core_simd_width` を含む）を返します。 |
<!-- mdformat on -->

この表の `torch.accelerator` API には PyTorch 2.11 以降が必要です。まれに古いバージョンの PyTorch を使う場合は、対応する `torch.tpu` API（例: `torch.tpu.device_count()` や `torch.tpu.current_device()`）を代わりに使ってください。PyTorch 2.11 以降では、これらの重複する `torch.tpu` API は TorchTPU の内部 API となり、ユーザー コードから直接呼び出すとエラーが発生します。

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **乱数生成（RNG）** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / アクセラレータ API | 説明 |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | アクティブな TPU デバイスの乱数生成器にシードを設定します。 |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | すべてのローカル TPU デバイスの RNG にシードを設定します。 |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | 現在の TPU デバイスの初期乱数シードを返します。 |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | 現在のデバイスの RNG 状態テンソルを返します。 |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | 現在のプロセスのアクティブな TPU デバイスの RNG 状態テンソルを設定します。 |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | ローカル デバイスの序数でインデックス付けされた、デフォルトの `torch.Generator` インスタンスのタプルです（遅延初期化されます）。 |
<!-- mdformat on -->

この表の `torch.accelerator.random` API には PyTorch 2.13 以降が必要です。まれに古いバージョンの PyTorch を使う場合は、代わりに `torch.tpu.initial_seed()` または `torch.tpu.get_rng_state()` を使ってください。PyTorch 2.13 以降では、これらの `torch.tpu` API は TorchTPU の内部 API となり、コードから呼び出すとエラーが発生します。

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **自動混合精度（AMP）** { #automatic-mixed-precision-amp }

TorchTPU は、`"tpu"` デバイスタイプ文字列を使って PyTorch 標準の [`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) フレームワークと統合されています。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU API | 説明 |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | 自動混合精度を有効にするコンテキスト マネージャ / デコレータです（TPU ではデフォルトで `bfloat16`）。 |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | TPU でサポートされている dtype（`bfloat16`、`float16`、`float32`、`float64`、`float8_*`、整数型、ブール型、複素数型）を一覧表示します。 |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **ストリームとイベント** { #streams-and-events }

ストリーム管理とデバイスの同期には、バックエンド固有の API ではなく標準の [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) API を使ってください。`torch.tpu` は、TPU の `Stream` や `Event` オブジェクトを直接作成する場合や、TPU コアのピン留めを設定する場合にのみ使います。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| CUDA API | TPU / アクセラレータ API | 説明 |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | デバイスのアクティブなストリームを返します。 |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | アクティブなストリームを設定します。 |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | デバイスのデフォルトの `TpuStream` を返します。 |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | 指定した `TpuStream` を選択するコンテキスト マネージャです。 |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | TPU デバイス上の保留中の処理がすべて完了するまで待機します。 |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | `torch.Stream` のサブクラスです。`synchronize()`、`record_event()`、`wait_event()`、`wait_stream()`、コアのピン留め（`pinned="dense"` または `"sparse"`。`is_dense` / `is_sparse` で確認）をサポートします。 |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | `torch.Event` のサブクラスです。`record()`、`synchronize()`、`query()`、`elapsed_time(end_event)` をサポートします。 |
<!-- mdformat on -->

この表の `torch.accelerator` API には PyTorch 2.11 以降が必要です。まれに古いバージョンの PyTorch を使う場合は、代わりに `torch.tpu.current_stream()`、`torch.tpu.set_stream()`、または `torch.tpu.synchronize()` を使ってください。

> **Note:** ストリームは CUDA の概念であり、TPU には直接対応するものがありません。TorchTPU の遅延実行モデルでは、XLA コンパイラと PjRt ランタイムがデータの依存関係に基づいて演算を自動的にスケジュールし、オーバーラップさせるため、複数のストリームを手動で管理する必要はありません。

## **TPU 固有の API** { #tpu-specific-apis }

`torch.tpu` は、標準的な CUDA の同等 API に加えて、TPU の実行モード、MXU の精度、XLA コンパイル、分散トポロジ、カスタム Pallas カーネル、トレース可能なオプティマイザ、プロファイリングのために特別に設計された API を提供します。

### **実行モードとコンパイラ オプション** { #execution-modes-and-compiler-options }

eager 演算を XLA がどのようにキューに入れ、融合し、コンパイルするかを制御します。アーキテクチャの解説については、[TorchTPU の実行モード](./torchtpu_execution_model.md)をご覧ください。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.EagerMode` | eager 実行を制御する列挙型です: `DEFER_AND_FUSE`、`DEFER_NEVER`、`DEFER_NEVER_AND_LAUNCH_BLOCKING`。 |
| `torch.tpu.eager_mode` | アクティブな `EagerMode` をグローバルに取得または設定します。 |
| `torch.tpu.set_eager_mode(mode)` | アクティブな `EagerMode` を一時的に設定するコンテキスト マネージャです。 |
| `torch.tpu.compiler.custom_compiler_options(options)` | ブロック内で実体化される eager 演算に、カスタムの XLA コンパイラ フラグを渡すコンテキスト マネージャです。 |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **数値精度** { #numerical-precision }

行列乗算ユニット（MXU）の精度と、中間の `float32` 累積を制御します。詳しいガイダンスについては、[数値精度](./numerical_precision.md)をご覧ください。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.Precision` | `float32` の行列乗算と畳み込みの MXU 精度を制御する列挙型です: `Precision.DEFAULT`、`Precision.HIGH`、`Precision.HIGHEST`。 |
| `torch.tpu.precision(level)` | ブロック内の行列乗算と畳み込みの演算に `Precision` レベルを設定するコンテキスト マネージャです。 |
| `torch.tpu.allow_excess_precision` | XLA の `--xla_allow_excess_precision` フラグをグローバルに取得または設定し、融合された中間値を `float32` のまま保持します。 |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **分散トレーニングとトポロジ** { #distributed-training-and-topology }

TorchTPU は、`"tpu"` デバイスのデフォルトの `torch.distributed` バックエンドとして `"tpu_dist"`（エイリアスは `"tpu"`）を登録します。[GKE で TorchTPU を使ってみる](./get_started_on_gke.md)と [FSDP のベスト プラクティス](./case_studies_fsdp.md)をご覧ください。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | `DeviceMesh("tpu", mesh=...)` 用に、シェイプ `mesh_shape` のトポロジを考慮したランク レイアウト テンソルを計算します。 |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | ワーカーを起動する前に、TPU トポロジとマルチホスト起動の環境変数を設定します。 |
| `torch.tpu.distributed.spmd_safe(func)` | `func` 内ですべてのランクが同一の TPU 演算を実行することを宣言するデコレータです。これにより eager の集団通信の融合が有効になります。 |
| `torch.tpu.window_num_stripes` | ピアツーピア転送の同時 DMA ストライプの最大数を取得または設定します。 |
| `torch.tpu.window_stripe_chunk_mb` | ピアツーピア転送の DMA ストライプあたりの最小ペイロード サイズ（MB 単位）を取得または設定します。 |
<!-- mdformat on -->

### **TPU カスタム演算子（`torch.tpu.ops`）と Pallas（`torch.tpu.pallas`）** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU は、ハードウェアに特化した演算子を `torch.tpu.ops` で、カスタムの JAX/Pallas カーネル ラッパーを `torch.tpu.pallas` で提供します。[Pallas を使用してカスタム カーネルを追加する](./custom_kernels.md)をご覧ください。

#### **汎用、アテンション、MoE、集団通信の演算子** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | インデックス テンソルを実体化せずに、TPU 上で 1D、2D、または 3D の max pooling を適用します。 |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | インデックス テンソルを使わずに `max_pool{1,2,3}d` のバックワード パスを計算します。 |
| `torch.tpu.ops.optimization_barrier(inputs)` | XLA がバリアをまたいで計算を移動、融合、重複排除しないようにします。 |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | TPU Pallas の Splash Attention カーネルを使った scaled dot-product attention です。 |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Mixture-of-Experts（MoE）ワークロード向けのグループ化された行列乗算です。 |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | `ragged_dot` の重みの勾配を計算します。 |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | TPU のアライメントのために、`torch.jagged` の `NestedTensor` バッファを指定した倍数までパディングします。 |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | `align_jagged_to_multiple` で追加された末尾のパディングを削除します。 |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | `group` 全体で `input` を all-gather し、次元 `dim` に沿って連結します。 |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | `group` 全体で `input` をリデュースし、次元 `dim` に沿ってチャンクを scatter します。 |
| `torch.tpu.ops.ragged_all_to_all(...)` | プロセス グループ内のランク間で可変長のテンソル スライスを交換します。 |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | デフォルトの TPU プロセス グループ上でのポイントツーポイントのテンソル送受信です。 |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | JAX/Pallas 関数をトレース可能な PyTorch カスタム演算子として登録します。 |
<!-- mdformat on -->

#### **SparseCore 埋め込み演算子** { #sparsecore-embedding-operators }

これらの演算子には SparseCore を搭載した TPU が必要です。`torch.tpu.get_device_properties()` の `sparse_cores_per_device` プロパティはデバイスあたりの SparseCore の数（SparseCore のない TPU では 0）で、`sparse_core_simd_width` はその SIMD 幅です。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | TPU SparseCore 上で CSR 埋め込みのルックアップと集約を実行します。 |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | `sparse_dense_matmul` のバックワード パスの勾配を計算し、SparseCore 上で融合されたインプレースの SGD、Adagrad、または Adam によるテーブル更新を適用します。 |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | スタックされた SparseCore の埋め込みアクティベーションを、特徴ごとのテンソルに分割します。 |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | SparseCore 用に、特徴ごとのアクティベーションの勾配を 1 つのテンソルにスタックします。 |
| `torch.tpu.ops.sparse_gather(...)` | TPU SparseCore 上で、CSR インデックスを使って 2D テンソルから行を gather します。 |
| `torch.tpu.ops.sparse_gather_backward(...)` | TPU SparseCore 上で、`sparse_gather` の勾配を scatter-add します。 |
| `torch.tpu.ops.sparse_iota(...)` | SparseCore 上で、CSR 行列の非ゼロ要素について行内の位置インデックスを生成します。 |
| `torch.tpu.ops.sparse_map_row(...)` | SparseCore 上で、スパースな CSR 行列に対してカスタムの MLIR 行単位の計算をマッピングします。 |
<!-- mdformat on -->

### **融合トレーニング ステップとオプティマイザ（`torch.tpu.optimizers`）** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` は、フォワード パス、損失の計算、バックワード パス、関数型オプティマイザのステップをトレースして、コンパイル済みの 1 つの TPU グラフにまとめます。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | 関数型のパラメータ（`params`）とオプティマイザの状態（`param_group`）を管理し、`make_compiled_train_step(...)` を介してトレーニング ステップ全体をコンパイルします。 |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | ステートレスな関数型オプティマイザの基底クラスです（`defaults`、`init_param_group`、`step` を公開します）。 |
| `torch.tpu.optimizers.AdamW(...)` | トレース可能な関数型の AdamW オプティマイザです（`AdamW.ParamGroup`）。 |
| `torch.tpu.optimizers.Muon(...)` | トレース可能な関数型の Muon オプティマイザです（`Muon.ParamGroup`）。 |
| `torch.tpu.optimizers.Sgd(...)` | トレース可能な関数型の SGD オプティマイザです（`Sgd.ParamGroup`）。 |
<!-- mdformat on -->

### **プロファイリング（`torch.tpu.profiler`）** { #profiling-torchtpuprofiler }

XProf と TensorBoard 用に、TPU ハードウェアと Python のトレースを設定します。[TorchTPU 向けの XProf](./performance_profiling_with_xprof.md)をご覧ください。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | 説明 |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | `torch.profiler.profile(..., experimental_config=...)` に渡す試験運用版の設定で、XProf 用にホスト、TPU デバイス、Python のトレースを設定します。 |
<!-- mdformat on -->

## **環境変数** { #environment-variables }

TorchTPU の動作は、XLA コンパイラ、TPU ドライバ（`libtpu`）、PyTorch 自身のデバッグ サブシステムを設定するいくつかの環境変数で制御されます。

> **重要: プロセス全体での初期化** TorchTPU の環境変数は、バックエンドが初期化された時点（通常は `import torch` または `torch.device("tpu")` の実行時）で**固定**されます。

### **デバイスの可視性、ランタイム、分散起動** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 環境変数 | 説明 |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | 現在のプロセスから見える TPU チップのインデックスをカンマ区切りで指定します（`CUDA_VISIBLE_DEVICES` に相当）。 |
| `RANK` | 分散ジョブにおける現在のプロセスのグローバル ランクです。 |
| `LOCAL_RANK` | 現在のプロセスのホスト内でのローカル ランクです。 |
| `WORLD_SIZE` | 分散ジョブのプロセスの総数です。 |
| `NNODES` | マルチホスト ジョブのホストノードの総数です。 |
| `NODE_RANK` | マルチホスト ジョブにおける現在のホストノードのランクです。 |
| `MASTER_ADDR` | `torch.distributed` のランク 0 のコーディネータ ノードのアドレスです。 |
| `MASTER_PORT` | `torch.distributed` のランデブーに使う `MASTER_ADDR` 上のポートです。 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | `"0"` に設定すると、`torch_tpu` バックエンドの自動登録が無効になります。 |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | 有効にすると、内部の `torch.tpu` デバイス API（`torch.accelerator` に同等の API があるもの）を PyTorch コアの外部から呼び出した際に、`RuntimeError` を発生させる代わりに 1 回だけ警告を出して実行を許可します。これは一時的な回避策としてのみ使ってください。`torch.accelerator.*` への移行をおすすめします。この変数は今後のリリースで削除される可能性が高いです。 |
| `LIBTPU_INIT_ARGS` | `libtpu` と PJRT ランタイムに渡す初期化フラグをスペース区切りで指定します。 |
| `TPU_PREMAPPED_BUFFER_SIZE` | PJRT クライアント用の、事前マッピングされたホストからデバイスへのメモリバッファのサイズ（バイト単位）です。 |
| `TMPDIR` | ランタイムとコンパイラが使う一時ディレクトリです。 |
<!-- mdformat on -->

### **ハードウェア トポロジとマルチホストの連携** { #hardware-topology-and-multi-host-coordination }

これらの変数は、TPU ハードウェアのトポロジとプロセスのネットワーキングを指定します。ほとんどの場合、`torch.tpu.distributed.set_tpu_launch_env()`、TorchTPU ランチャー、または Google Cloud 環境によって自動的に設定されます。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 環境変数 | 説明 |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | TPU スライスの物理トポロジです（例: `2x2x1`、`2x2x4`）。 |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | TPU スライス ビルダー サービスのワーカー アドレスを `host:port` 形式のカンマ区切りで指定します。 |
| `TORCH_TPU_DEVICES_PER_HOST` | スライス ビルダーのアドレスを導出するときの、ホストあたりの TPU チップ数です。 |
| `TORCH_TPU_BASE_PORT` | スライス ビルダーのアドレスを導出するときの開始ポート番号です。 |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | 集団通信のグラフがランク間で一致しているかを各ランクが検証する段階です。 |
| `TORCHCOMM_RANK` | `torchcomm` のプロセス ランクです（`RANK` より優先されます）。 |
| `TORCHCOMM_SIZE` | `torchcomm` のワールド サイズです（`WORLD_SIZE` より優先されます）。 |
<!-- mdformat on -->

### **実行モード、コンパイル、キャッシュ** { #execution-mode-compilation-and-caching }

これらの変数は、PyTorch の演算とグラフがどのようにディスパッチされ、lowering され、XLA によって最適化され、実行間やランク間でキャッシュされるかを制御します。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 環境変数 | 説明 |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | 有効にすると（`"1"`/`"true"`）、デフォルトの eager モードが `EagerMode.DEFER_AND_FUSE` に設定されます。 |
| `TORCH_TPU_LAUNCH_BLOCKING` | 有効にすると（`"1"`/`"true"`）、デフォルトの eager モードが `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING` に設定されます。 |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | ホストローカルのコンパイル キャッシュ用の、`/dev/shm/torch_tpu_cache/` 配下のディレクトリ名です（または `"disabled"`）。 |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | ホスト間と実行間で共有される永続的な Tier 3 コンパイル キャッシュへのパスです（例: `gs://...` またはローカルパス）。 |
| `XLA_FLAGS` | XLA コンパイラのフラグをスペース区切りで指定します（例: `--xla_dump_hlo_as_text --xla_dump_to=...`）。 |
<!-- mdformat on -->

### **デバッグ、安全性、プロファイリング、トレース** { #debugging-safety-profiling-and-tracing }

これらの変数は、TorchTPU のデバッグ コンテキスト、ランタイムの安全性チェック、プロファイラの出力を制御します。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 環境変数 | 説明 |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | `"1"` に設定すると、Python の `RuntimeError` 例外に C++ のバックトレースが追加されます。 |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | `"1"` に設定すると、TPU 上でランタイムのデバッグ チェックと境界チェックが有効になります。 |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | `TpuProfilerConfig.run_dir` が設定されていない場合に使う、XProf トレース ファイルの代替出力ディレクトリです。 |
| `TORCH_TPU_XPROF_SESSION_ID` | マルチホストのワーカー間で共有する XProf プロファイリング セッション ID です。 |
| `TORCH_TRACE` | 構造化されたコンパイラとディスパッチャのトレースログ（`tlparse`）の出力ディレクトリです。 |
<!-- mdformat on -->

### **環境変数の設定スニペット** { #environment-variable-configuration-snippets }

#### **1. C++ スタックトレース（`TORCH_SHOW_CPP_STACKTRACES`）** { #1-c-stacktraces-torch_show_cpp_stacktraces }

`torch` をインポートする前に `TORCH_SHOW_CPP_STACKTRACES=1` を設定すると、C++ バックエンドや XLA の深い部分で発生するエラーの診断がずっと簡単になります。単純な Python の `RuntimeError` の代わりに、正確な実行フレームを示す詳細な `C++
CapturedTraceback` が表示されます。

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. 内部デバッグ チェックと境界の検証（`TORCH_TPU_ENABLE_DEBUG_CHECKS`）** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

デフォルトでは、TPU は速度を優先して境界チェックを省略します。`TORCH_TPU_ENABLE_DEBUG_CHECKS=1` を有効にすると、範囲外のインデックス（`torch.take` など）に関連する見つけにくいバグを検出できます。ただし、CPU/TPU 間の同期が増えるため実行が遅くなります。本来なら何も通知されずに失敗したりデータ破損を引き起こしたりする演算で、`take(): expected indices to be
in range` のような明示的な例外が発生します。

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. XLA IR / HLO のダンプ（`XLA_FLAGS`）** { #3-dumping-xla-ir-hlo-xla_flags }

`XLA_FLAGS` を設定すると、コンパイラに High-Level Optimizer（HLO）のテキストをディレクトリへ強制的にダンプさせることができます。ダンプ ディレクトリ（例: `/tmp/hlo_dump`）には次のものが含まれます。

*   **`*.after_optimizations.txt`**: ハードウェア レベルの演算を示す最終的な HLO グラフです。
*   **`*.flagfile`**: 有効な XLA コンパイラ フラグの完全な一覧です（`xla_optimization_level` の確認に使えます）。
*   **`*tpu_comp_env.txt`**: コンパイル時に使われた内部の TPU 環境設定です。

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. 多層コンパイル キャッシュ（`TORCH_TPU_TIER2_COMPILATION_CACHE` と `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`）** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU は多層キャッシュを使って、実行間やランク間での重複したコンパイルを回避します。

*   **Tier 2（`TORCH_TPU_TIER2_COMPILATION_CACHE`）** は、ホスト内の共有メモリにローカルに保存されます。ディレクトリの**名前**（例: `"tpu_tier2_cache"`）を指定する必要があり、バックエンドが自動的に `/dev/shm/torch_tpu_cache/<name>` 内に配置します。
*   **Tier 3（`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`）** は、ホスト間とセッション間で共有される永続的なコンパイル キャッシュです（通常は `gs://` バケットの URI または共有ファイル システムのパス）。

コンパイル済みのバイナリ アーティファクト（`*.bin` 実行可能ファイル）はキャッシュ ディレクトリに非同期で保存され、以降の実行で再利用されるため、JIT コンパイルのレイテンシが大幅に短縮されます。

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```
