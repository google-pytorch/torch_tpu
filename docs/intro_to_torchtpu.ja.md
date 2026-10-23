# TorchTPU の概要 { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU は、[Google Cloud TPU](https://cloud.google.com/tpu) 向けのオープンソースのネイティブ PyTorch バックエンドで、Google と Meta の PyTorch チームが共同で開発しています。TorchTPU を使用すると、デベロッパーは既存の PyTorch ワークロードを最小限のコード変更で TPU ハードウェア上で実行でき、トレーニングと推論のワークロードを単一のチップからマルチスライスの TPU Pod までスケールできます。

## TorchTPU を使用する理由 { #why-use-torchtpu }

Google TPU は、ディープ ラーニングを支える行列演算とベクトル演算のために専用に構築された特定用途向け集積回路（ASIC）で、大規模な環境で高いスループット、エネルギー効率、コスト パフォーマンスを提供します。TPU はこれまで JAX や TensorFlow のエコシステムと組み合わせて使用されてきましたが、TorchTPU は TPU のネイティブ サポートを [PyTorch エコシステム](https://github.com/google-pytorch/torch_tpu)に直接もたらします。TorchTPU は、次の 3 つの指針に基づいて意図的に設計されています。

*   **PyTorch へのネイティブな統合:** TorchTPU は JAX ランタイムに依存せず、PyTorch と直接統合されます。標準の PyTorch のテンソル ビューとインプレース変更のセマンティクスを維持し、PyTorch の 1 デバイスにつき 1 プロセスの実行モデルを採用し、Dynamo を通じて `torch.compile` をサポートします。
*   **すぐに使える使いやすさ:** ワークロードを GPU から TPU に移植するには、まず `device="cuda"` を `device="tpu"` に、`backend="nccl"` を `backend="tpu_dist"` に変更します。変更を加えていない PyTorch モデルがすぐに eager モードで実行され、ランタイムやコンパイラの問題を診断する際にはわかりやすいエラー メッセージが表示されます。
*   **本番環境規模のパフォーマンス:** TorchTPU は、演算ごとの eager 実行に加えて、遅延融合の eager モード、Dynamo から XLA へのコンパイラ パイプライン、シェイプに起因する再コンパイルを避けるための有界の動的性、カスタム カーネル API を導入しており、ハードウェアの使用率をピークまで高めるのに役立ちます。

## クイックスタート: TPU で実行する { #quickstart-running-on-tpu }

次のセルは、TPU 上で実行される PyTorch プログラムの簡単な例です。TPU ランタイムを有効にすると、[この例を Colab で](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)実行できます。

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## 主な機能 { #key-features }

*   **演算子とモデルの包括的なカバレッジ:** 450 個のコア ATen 演算子（フォワード パスとバックワード パスの両方）をサポートしており、LLM（Llama、Qwen、Gemma、DeepSeek、Mistral）、ビジョンモデル（ViT、ResNet、ConvNeXt、DINOv2）、音声モデル（Whisper、Wav2Vec2）、拡散パイプライン（FLUX、Stable Diffusion）にわたる 5,000 個の人気の HuggingFace モデルで継続的に検証されています。
*   **3 つの eager 実行モード:**
    *   **Strict（デフォルト）:** ATen 演算子を 1 つずつ非同期で TPU にディスパッチします。GPU での標準的な PyTorch の eager 動作と同じです。
    *   **Debug（`TORCH_TPU_LAUNCH_BLOCKING=1`）:** 各演算子を CPU と同期して実行し、実行時の境界チェックを有効にしてデバッグを容易にします。
    *   **DeferAndFuse（`TORCH_TPU_DEFER_AND_FUSE=1`）:** 実体化ポイント（`.cpu()` や `.item()` など）まで、一連の ATen 演算子を遅延させて最適化された計算ブロックに融合します。これにより、High Bandwidth Memory（HBM）のトラフィックが大幅に減少し、eager 実行が高速化されます。
*   **`torch.compile` によるグラフ コンパイル:** `"tpu"` Dynamo バックエンド（`torch.compile(model)`）を登録して、PyTorch FX グラフ全体を [StableHLO](https://openxla.org/stablehlo/spec) に lowering し、[OpenXLA コンパイラ](https://openxla.org/xla)でコンパイルします。動的シェイプ（`torch._dynamo.mark_dynamic`）を積極的にサポートしているため、コンパイル済みのグラフは、指定された上限までのさまざまなバッチサイズやシーケンス長を再コンパイルせずに処理できます。
*   **有界の動的性:** 動的な上限に対してコンパイルすることで、可変のバッチサイズやシーケンス長によって発生する再コンパイルをネイティブに軽減します。
*   **多層のコンパイル キャッシュ:** 3 つのキャッシュ層で冗長な XLA コンパイルを排除します。
    *   **Tier 1:** プロセスローカルのインメモリ キャッシュ（常に有効）。
    *   **Tier 2:** マルチデバイスのジョブでプロセス間の再利用を可能にする、ホストローカルの共有メモリ。
    *   **Tier 3:** ホスト間やジョブの再起動をまたいで再利用できる永続的なリモート キャッシュ（Cloud Storage など）。
*   **ネイティブの分散プリミティブ:** `backend="tpu_dist"` を通じて `torch.distributed` プロセス グループ（`c10d::Backend`）レベルで TPU の集団通信を実装しており、`DistributedDataParallel`（DDP）、`FullyShardedDataParallel`（FSDP2 と SimpleFSDP）、`DTensor` をサポートしています。また、`torch.tpu.distributed.topology_aware_mesh` を使用して、並列処理の次元を物理的な TPU インターコネクトに合わせることもできます。
*   **カスタム TPU カーネル:** [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) でカスタム カーネルを作成する（チップ間の直接 RDMA 集団通信をサポートする `torch.tpu.pallas.jax_op` から呼び出し可能）、試験運用版の [Helion](https://pytorch.org/blog/helion/) DSL を使用してタイルベースのカーネルを記述する、または Tokamax から最適化されたビルド済みの TPU カーネルをインポートすることができます。
*   **エコシステムとの統合:** `torch.profiler`（TensorBoard/XProf）、vLLM、SGLang、TorchTitan、Ray、TorchRec など、標準の PyTorch のサービングとトレーニングのフレームワーク、プロファイリング スイートとすぐに連携できます。

## 制限事項と今後の取り組み { #limitations-and-future-work }

他のプロダクトと同様に、TorchTPU にも既知の制限事項があり、今後対処していく予定です。最初のリリースでは主要なワークロードを堅牢かつすぐに使える形でカバーしていますが、TorchTPU はコミュニティ主導の共同の取り組みです。今後の予定については[機能とロードマップ](./feature_set.md)をご覧ください。また、GitHub で [PR を送信](https://github.com/google-pytorch/torch_tpu)して、これらの制限事項の解決にぜひご協力ください。これはコミュニティ プロジェクトです。[GitHub](https://github.com/google-pytorch/torch_tpu)、[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu)、[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)、[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) を通じて、皆様からのご意見とご協力をお待ちしています。

*   動的シェイプは再コンパイルを引き起こします
    *   TorchTPU では、`torch.compile` はデフォルトで `dynamic=False` を設定します。`dynamic=None` と `dynamic=True` は、本番環境ではまだサポートされていません。[動的シェイプに関する説明をご覧ください](./dynamic_shapes.md)
    *   たとえば、Hugging Face の LLM で decode/generate ステップを実行すると、有界の動的性を有効にしない限り、ステップごとに再コンパイルが発生します。
*   TPU ではビュー演算のコストが高くなる場合があります（[TorchTPU のメモリ モデル](./torchtpu_memory_model.md)をご覧ください）。
*   集団通信を使用する MPMD は機能としてはサポートされていますが、まだピーク パフォーマンスには達しません。集団通信と計算のオーバーラップには[いくつかの制約](./distributed_support.md)があります。
    *   SparseCore オフロードによるストリームの完全なサポートは近い将来に提供される予定です。
    *   eager または `torch.compile` で FSDPv2 を使用するのではなく、`torch.compile()` で SimpleFSDP を使用してください。[FSDP のベスト プラクティス](./case_studies_fsdp.md)をご覧ください
    *   通信を行うモジュールで XLA の最適化からピーク パフォーマンスを得る（つまり、計算と通信のオーバーラップを有効にする）には、完全に同一であることを **ユーザーが保証する** 領域に `spmd_safe` デコレータを使用することをおすすめします
*   TorchTPU は現在、FlexAttention をネイティブにサポートしていません
*   現世代の TPU は、fp32 より低い精度（tf32 など）での matmul の累積をサポートしていません。
*   TPU デバイスにアクセスできるのは 1 つのプロセスのみであるため、同じ場所に配置する RL のサポートが制限されます（TPU チップには 1 つ以上のデバイスを関連付けることができ、それらには個別の OS プロセスからアクセスできます）。

**その他の推奨事項:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 制限事項 | 推奨事項 |
| :--- | :--- |
| eager モードにはゼロではないウォームアップ時間がある | 永続コンパイル キャッシュを有効にする |
| Torch.compile に時間がかかる | 永続コンパイル キャッシュを有効にする |
| PyTorch の CNN はデフォルトでチャネル ファーストのレイアウトを使用するため、TPU では非常に非効率 | パフォーマンスを向上させるには、こちらの [PyTorch オプション](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)を使用してチャネル ラストのレイアウトを設定する |
| TPU と他のアーキテクチャではトポロジが異なるため、デフォルトのデバイス メッシュのマッピングが最適でない場合がある | 最適なデバイス メッシュのマッピングと、デバイス メッシュのマッピングをカスタマイズする機能のために `topology_aware_mesh` を使用する |
<!-- mdformat on -->

## 使ってみる { #getting-started }

このドキュメント スイートは、最初のセットアップから分散環境での本番デプロイまで進めるのに役立つ、4 つの構造化されたセクションで構成されています。

*   **使ってみる:** 環境を構成し、[VM 上](./get_started_on_a_vm.md)または [GKE を通じて](./get_started_on_gke.md)ワークロードをデプロイするための手順を説明します。
*   **詳細:** [TPU のメモリ モデル](./torchtpu_memory_model.md)、[分散実行](./distributed_support.md)、さまざまな[実行モード](./torchtpu_execution_model.md)について詳しく説明します。
*   **ケーススタディ:** [Flux モデル](./case_studies_flux.md)の最適化、[Muse Glimmer](./case_studies_muse_glimmer_30b_training.md) のトレーニング、[エンドツーエンドの LLM 推論](./case_studies_muse_glimmer_30b_inference.md)、[FSDP](./case_studies_fsdp.md) の効果的な適用を扱う、エンドツーエンドのレシピです。
*   **ワークロードを分析する:** [プロファイリング](./performance_profiling_with_xprof.md)、[数値精度](./numerical_precision.md)の管理
