# TorchTPU の仕組み { #how-torchtpu-works }

## TPU の基礎 { #tpu-primer }

[Google TPU](https://cloud.google.com/tpu) は、ディープ ラーニングの基盤となる大規模なテンソル演算を実行するために特別に設計された専用の ASIC です。TPU の[アーキテクチャは世代によって異なります](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)が、大まかに言うと、TPU チップには次のものが含まれます。

*   1 つ以上の TensorCore。それぞれが、行列乗算ユニット（MXU）、ベクトル ユニット（VPU）、スカラー ユニットで構成されるシングルスレッドのプロセッサです。
*   1 つ以上の SparseCore。スパース演算に特化しており、TensorCore から演算をオフロードできます。
*   大容量の High Bandwidth Memory（HBM）。[TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive) では 288 GB、8.6 TB/s で、ML データを保存し、CPU と共有するために使用できます。
*   双方向の Inter-Chip Interconnect（ICI）。[TPU 8t と 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) では 19.2 TBps で、ユーザーが構成した [TPU スライス](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)内の TPU チップ間で高速な通信を可能にします。TPU スライスは [TPU Pod](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod) の論理的なサブグループです。TPU Pod は、あらかじめ構成された数の TPU チップが 3D トーラス（TPU 7、TPU 8t）または Boardfly（TPU 8i）構成で接続されたものです。
*   双方向の Data-Center-Network（DCN）。[TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) では 400 GBps で、スライスをまたいだ TPU チップ間の通信を可能にします。

TPU は、[XLA コンパイラ](https://github.com/openxla/xla) スタックを使用して、[StableHLO](https://openxla.org/stablehlo/spec) と [Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu) の 2 つの異なるエントリ ポイントからプログラミングされます。実際の TPU ISA は独自仕様で、変更される可能性があり、一般には公開されていません。実質的には、StableHLO、そしてそれ以上に Mosaic は、PTX に相当するものと考えることができます。概要については、次の図をご覧ください。

![TorchTPU の仕組み](./images/torch_tpu_inputs_to_tpu.svg)

C++ プログラムは、[PjRT クライアント](https://openxla.org/xla/pjrt/cpp_api_overview)をインスタンス化することで、ローカルの TPU にアクセスできます。その API には、MLIR（StableHLO または Mosaic）を TPU 実行可能プログラムにコンパイルするメソッド、その実行可能プログラムを起動するメソッド、TPU との間でデータをコピーするメソッドがあります。TPU の実行は非同期であり、CPU は関連する PjRTBuffer を待機することで TPU の結果の準備ができるまで待機できます。StableHLO は [SSA パラダイム](https://en.wikipedia.org/wiki/Static_single-assignment_form)に従っているため、インプレースの変更はできません。つまり、書き込み演算は常に（少なくとも概念的には）新しい値を生成します。これは TorchTPU にとって特に重要です。TorchTPU は、ビューの分解と逆変換によって、変更を含むおなじみの PyTorch の「ビュー」セマンティクスを提供しているためです。XLA コンパイラは、テンソルのシェイプがコンパイル時にわかっていることを前提としています。TorchTPU では、この制約は[有界の動的性](./dynamic_shapes.md)によって緩和されています。これが PyTorch ワークロードにとっての制限であることを認識し、有界の動的性のサポートを追加してきました。有界の動的性とは、シェイプの次元の最大値はコンパイル時にわかっているものの、実際の次元は実行時にしかわからない場合を指します。

## TorchTPU: 自動読み込みをサポートするツリー外のバックエンド { #torchtpu-an-out-of-tree-backend-with-autoload-support }

追加のバックエンドをツリー外で構築し、実行時に登録するという[現在のベスト プラクティス](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)に従い、PyTorch ディスパッチャを新しい C++ バックエンドで拡張し、`PrivateUse1` として登録しています。さらに、標準の `at::Tensor` メカニズムを使用して、テンソル ストレージのデバイス固有の実装を定義しています（`c10::StorageImpl` 内の `c10::UniqueVoidPtr` を使用）。

TorchTPU は使いやすさを重視して設計されています。TorchTPU の wheel がインストールされた TPU マシンでユーザーが PyTorch をインポートすると、Python が TorchTPU の wheel を自動的に読み込みます。wheel が自動的に読み込まれると、バックエンドがセットアップされ、`PrivateUse1` という名前が `tpu` に置き換えられます。そのため、簡単な例では TorchTPU プログラムはツリー内のバックエンドと変わらず、PyTorch プログラムはほとんど、またはまったく変更せずに TPU 上で実行できます。

```python
import torch
import torch.distributed as dist

def main():
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

TPU が現在のアクセラレータになり、`tpu_dist` が TorchTPU のデフォルトのバックエンドになるため、これらを明示的に指定する必要はありません。

```python
import torch
import torch.distributed as dist

def main():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

## グラフをキャプチャする { #capturing-the-graph }

TorchTPU は、バックエンド インフラストラクチャの一部である PyTorch の演算ディスパッチ フックを利用して、eager の演算を ATen レベルでトレースします。TorchTPU の演算は StableHLO プログラムとして構築され、単独またはグループで XLA に渡されます。これにより、コンパイラは演算を大幅に融合して、ハードウェアをより効果的に利用できます。TorchTPU が PyTorch ディスパッチャからコールバックの呼び出しを受け取ると、ランタイムは次の処理を行います。

*   次のものを含む Deferred Op インスタンスを作成します。
    *   StableHLO ビルダー オブジェクト。呼び出されると、次に示すように `aten::mul` のセマンティクスを実装する StableHLO 関数を返します（暗黙的なブロードキャストと dtype の昇格は別の場所で処理されます）。

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   キャッシュ キーの計算に使用される、静的な演算パラメータ（`aten::native_batch_norm` の training や momentum など）のマップ（省略可）。
*   その Deferred Op インスタンスを（間接参照によって）指す `c10::StorageImpl` を持つ、新しい `at::Tensor` を返します。

実行モードに応じた適切な同期ポイントに到達すると、TorchTPU は次の処理を行います。

*   演算の領域を分析します。
*   どの演算を実行可能コードに実体化するかを決定します。
*   グラフのコンパイル キャッシュ キーを作成します。
*   キャッシュにすでに存在する場合は、実行可能プログラムを取得して TPU にディスパッチします。
*   キャッシュに存在しない場合は、オブジェクトを非同期でコンパイルし、キャッシュに追加して、実行のためにディスパッチします。
*   到達した同期ポイントで必要な場合は、値の準備ができるまで待機して、値をコピーして戻します。

さまざまな実行モードの詳細については、[TorchTPU の実行モデル](./torchtpu_execution_model.md)で説明します。

## TorchTPU のコンパイル キャッシュ { #the-torchtpu-compilation-cache }

不要なコンパイルを避けるために、TorchTPU は多層のコンパイル キャッシュを使用します。

*   **Tier 1:** 同じプロセス内でコンパイル結果を共有するための、プロセスローカルのインメモリ キャッシュです。3 つの層の中で最も高速です。常に有効になっています。
*   **Tier 2:** 同じホスト上のプロセス間での共有を可能にする、マシンローカルの共有メモリです。マルチデバイスのトレーニングや推論に役立ちます。Tier 2 は分散プログラムではデフォルトで有効になっていますが、単一デバイスのプログラムではメリットがないため無効になっています。
*   **Tier 3:** リモート ファイル システム（Google Cloud Storage など）を基盤とする永続キャッシュで、ホスト、プログラム、ジョブの再起動をまたいで再利用できます。Colab やデータセンターなどの実行環境で TPU プログラムを実行する場合に役立ちます。デフォルトでは無効になっていますが、ストレージの場所を指定することで有効にできます。

各エントリは、1 つの ATen 演算または一連の ATen 演算から生成された StableHLO 計算グラフを表す、コンパイル済みの TPU 実行可能プログラムです。StableHLO グラフは、グラフの構造、入出力のテンソル シェイプ、コンパイラ オプション（TPU ハードウェア、libtpu、PjRt ランタイムのバージョンを含む）を組み合わせたフィンガープリントで表されます。

## `torch.compile` のサポート { #torchcompile-support }

`torch.compile()` をサポートする他のハードウェアと一貫したユーザー エクスペリエンスを維持するために、新しい TorchDynamo [バックエンド](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)を登録しています。このバックエンドは、`compiled_model = torch.compile(model,
backend="tpu")` のように使用できます。

Dynamo は、`torch.fx.GraphModule` とサンプル入力テンソルのリストを渡して TorchTPU バックエンドを呼び出します。その時点で、ランタイムは提供されたグラフを eager モードと同じようにトレースしますが、すべての演算は遅延され、TPU 上では実行されません。ランタイムは FX グラフ全体に対して単一の StableHLO プログラムを生成してコンパイルし、その呼び出しをラップする Python の callable を返します。TorchTPU は `TorchInductor` を使用せず、TPU アーキテクチャとともに開発されてきた成熟した XLA コンパイラ スタックを再利用しています。

## 詳細 { #more-detail }

以降のドキュメントでは、次の内容について詳しく説明します。

*   TPU でサポートされる[メモリ モデル](./torchtpu_memory_model.md)と、それが PyTorch に与える影響。
*   TorchTPU がサポートする[実行モード](./torchtpu_execution_model.md)。
*   [カスタム カーネル](./custom_kernels.md)のサポート。
*   TPU クラスタでの[分散実行](./distributed_support.md)。
