# XProf によるパフォーマンス プロファイリング { #performance-profiling-with-xprof }

TPU で大規模なモデルをトレーニングするには、どこに時間がかかっているかを把握することが不可欠です。このガイドでは、TPU プロファイラで実行トレースをキャプチャし、TensorBoard で分析する方法について説明します。

> **Warning:** **前提条件:** TensorBoard には `setuptools<81` が必要です。最新バージョンでは、TensorBoard が依存している `pkg_resources` が削除されています。最初にインストールしてください。
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **PyTorch のバージョン:** ネイティブの `torch.profiler` TPU 統合には、PyTorch 2.12 以降が必要です。

## 1. TPU プロファイラ { #1-the-tpu-profiler }

標準の `torch.profiler`（`torch_tpu` と統合済み）は、CPU と TPU の両方のアクティビティをトレースするための **コンテキスト マネージャ** インターフェースを提供します。プロファイリングするコードを `torch.profiler.profile()` ブロックでラップすると、収集されたトレースがディスクに保存され、TensorBoard で表示できます。

**主なコンポーネント:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| コンポーネント                                  | 役割                            |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | トレースの収集を開始 / 停止する <br> 標準のコンテキスト マネージャ |
| `torch.profiler.ProfilerActivity.CPU`      | CPU 側の演算（ディスパッチ、データの <br> 読み込み）をトレース |
| `torch.profiler.ProfilerActivity.TPU`      | TPU ハードウェアの実行（計算、 <br> メモリ、インターコネクト）をトレース |
| `torch.profiler.tensorboard_trace_handler` | TensorBoard 用にトレースを <br> ディレクトリに保存するコールバック       |
| `TpuProfilerConfig`                        | TPU のプロファイリングをカスタム構成するためのヘルパー                   |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. プロファイリングのカスタム構成（`TpuProfilerConfig`） { #2-custom-profiling-configuration-tpuprofilerconfig }

`TpuProfilerConfig` で渡すオプションを使用すると、ホストとデバイスのトレーサー レベルや、出力ディレクトリの構造を細かく制御できます。

### サポートされている構成オプション { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| オプションキー            | 型           | デフォルト | 説明               |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | CPU ホスト トレーサーの詳細度を制御します <br>（0=オフ、1=重要なもののみ、2=すべて）。 |
| `device_tracer_level` | `int`          | `1`     | TPU ハードウェアのデバイス トレーサーの <br> 詳細度を制御します（0=オフ、1=オン）。 |
| `python_tracer_level` | `int`          | `0`     | Python 関数のトレース <br> レベルを制御します（0=オフ、1=オン）。 |
| `run_dir`             | `pathlib.Path` | `None`  | TPU ハードウェアのトレース ファイル <br>（`.xplane.pb`）が保存される出力ディレクトリ。 |
| `worker_rank`         | `str`          | `None`  | XPlane のファイル名に接尾辞として <br> 追加されるワーカーのランク（`<hostname>_<rank>.xplane.pb`）。 |
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

## 3. プロファイルをキャプチャする { #3-capturing-a-profile }

次のセルは、プロファイラのコンテキスト内で小さなトレーニング ループを実行します。主なポイントは次のとおりです。

1.  トレーニング ループを `torch.profiler.profile()` の内側でラップします
2.  アクティビティを指定します。ホスト側の演算には `CPU`、ハードウェアの実行には `TPU` を指定します
3.  `torch.profiler.tensorboard_trace_handler` を使用してトレースをディレクトリに保存するように `on_trace_ready` を設定します
4.  （省略可）トレーサー レベルを制御するには、`TpuProfilerConfig` を指定した `experimental_config` を渡します

> **Important:** プロファイラを使用するとオーバーヘッドが発生します。トレーニングの実行全体ではなく、代表的なステップのサブセット（100～1,000 ステップなど）のみをプロファイリングしてください。

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

## 4. TensorBoard でトレースを表示する { #4-viewing-traces-in-tensorboard }

次のコマンドを実行します。

**ステップ 1:** プロファイラ プラグインをインストールします（ダッシュボードを表示するために必要です）。

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**ステップ 2:** TensorBoard を起動します。

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

XProf はスタンドアロン サーバーとして実行することもできます。

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**ステップ 3:** ブラウザで `http://localhost:6006` を開き、**「Profile」** タブを選択します。

> **Note:** ポート転送が自動的に機能するように、TensorBoard は VS Code の統合ターミナルから実行する必要があります。

**GKE の場合:** トレースが Pod よりも長く保持されるように、[Cloud Storage FUSE CSI ドライバ](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)でマウントした Cloud Storage バケットにトレース ディレクトリを書き込み、`xprof
--logdir=gs://<bucket>/profiler_output` でトレースを開きます。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| ビュー               | 表示内容                                          |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | 概要: ステップ時間、デバイスの使用率、上位の演算 |
| **Trace Viewer**   | CPU と TPU 上のすべての演算のタイムライン。ギャップ（ハードウェアのアイドル状態）を探します |
| **Op Profile**     | 演算ごとの時間の内訳。最もコストの高い演算を見つけます |
| **Memory Profile** | HBM 使用量の推移。メモリの急増を検出します |
<!-- mdformat on -->

## 5. 確認すべきポイント { #5-what-to-look-for }

### トレースの危険信号 { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| パターン                 | 意味                 | 対処法                      |
| ----------------------- | ----------------------- | ------------------------ |
| TPU の演算間の大きなギャップ | ホスト側のボトルネック <br>（データの読み込み、Python のオーバーヘッド） | データをプリフェッチし、ループ内の Python ロジックを減らす |
| 頻繁な短い TPU のバースト   | グラフ ブレークが多すぎる → 頻繁な再コンパイル | `.item()` / `.cpu()` の呼び出しを減らし、`torch.compile` を使用する |
| MXU の使用率が低い | テンソルの次元がタイルサイズ（128）に揃っていない | 次元を 128 の倍数にパディングする |
<!-- mdformat on -->

### 理想的なプロファイル { #the-ideal-profile }

適切に最適化された TPU ワークロードでは、次のような特徴が見られます。

-   **長く連続した TPU の計算ブロック**: コンパイラが多数の演算を 1 つの実行に融合しています
-   **CPU→TPU のギャップが最小限**: データの読み込みが計算とオーバーラップしています
-   **MXU の使用率が高い（60% 超）**: テンソルのシェイプがハードウェアに揃っています

## 6. トレースがキャプチャされたことを確認する { #6-verifying-the-traces-were-captured }

次のセルは、プロファイラの出力ディレクトリにトレース ファイルが含まれていることを確認します。

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

## 7. カーネルのプロファイリング（きめ細かいパフォーマンス カウンタ） { #7-kernel-profiling-fine-grained-performance-counters }

カーネルのプロファイリングを使用すると、きめ細かいハードウェア パフォーマンス カウンタ（Pallas カーネル用など）を収集して、標準のイベント トレースでは見えないパフォーマンスのボトルネックを診断できます。

> **Note:** きめ細かいパフォーマンス カウンタのサンプリングは、**Ironwood（TPU v7/TPU7x）** 以降の TPU 世代で利用できる高度なハードウェア機能です。

### パフォーマンス カウンタを有効にする { #enabling-performance-counters }

パフォーマンス カウンタを有効にするには、`TpuProfilerConfig` の `experimental_options` でカウンタの構成を渡します。

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

構成パラメータ（サンプリング間隔、スケーリング、トリガーモードなど）の詳細なリストと、カウンタのインデックスを決定する方法については、公式の [OpenXLA カーネル プロファイリング ガイド](https://openxla.org/xprof/kernel-profiling)をご覧ください。

## 8. クイック リファレンス { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
ステップ        | コマンド / コード
----------- | --------------
**インポート**  | `from torch.tpu.profiler import TpuProfilerConfig`
**プロファイリング** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**保存**    | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**表示**    | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** トレース ファイルを管理しやすいサイズに保つために、プロファイリングは 100～1,000 ステップのみにしてください。プロファイラによって、ステップごとに約 5～10% のオーバーヘッドが発生します。
