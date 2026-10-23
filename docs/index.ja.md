<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>ディスカッション フォーラム</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>TPU 上の PyTorch</b></a>
</p>

## 概要 { #about }

TorchTPU は、Google Cloud TPU 向けのオープンソースのネイティブ PyTorch バックエンドで、Google と Meta が共同で開発しています。TorchTPU を使用すると、既存の PyTorch モデルを最小限のコード変更で TPU ハードウェア上で実行でき、トレーニングと推論のワークロードを単一のチップからマルチスライスの TPU Pod までスケールできます。

## 今日は何をしたいですか？ { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> 使ってみる**

    環境をプロビジョニングし、基本的なアーキテクチャを理解して、最初のワークロードを実行します。

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU の概要](./intro_to_torchtpu.md): TorchTPU が作られた理由、TorchTPU が提供するもの、TorchTPU がどのように役立つかを説明します。
    - [<!--mkdocs::octicons-arrow-right-24:--> VM で TorchTPU を使ってみる](./get_started_on_a_vm.md): TPU VM に TorchTPU をセットアップします。
    - [<!--mkdocs::octicons-arrow-right-24:--> GKE で TorchTPU を使ってみる](./get_started_on_gke.md): GKE を通じて TorchTPU を使用します。
    - [<!--mkdocs::octicons-arrow-right-24:--> サポートされている機能とロードマップ](./feature_set.md): 現在サポートされている機能と、今後のリリースの計画です。
    - [<!--mkdocs::octicons-arrow-right-24:--> API リファレンス](./api_reference.md): 現在のリリースの TorchTPU API と環境変数です。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU FAQ](./faq.md): TorchTPU に関するよくある質問です。

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> アーキテクチャとコア コンセプトの詳細**

    メモリ レイアウト、実行エンジン、分散実行の仕組みを詳しく説明します。

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU の仕組み](./how_torchtpu_works.md): TorchTPU が PyTorch および TPU ハードウェアとどのように統合されているかを、上位層から下位層まで説明します。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU の実行モデル](./torchtpu_execution_model.md): TorchTPU が PyTorch コードを実行する方法と、サポートしているさまざまな実行モードを詳しく説明します。
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU のメモリ モデル](./torchtpu_memory_model.md): TPU と XLA コンパイラによるメモリの扱いに TorchTPU がどのように対処しているかを説明します。
    - [<!--mkdocs::octicons-arrow-right-24:--> 分散処理のサポート](./distributed_support.md): TorchTPU が分散ワークロードをどのようにサポートしているかを説明します。
    - [<!--mkdocs::octicons-arrow-right-24:--> カスタム カーネルのサポート](./custom_kernels.md): TorchTPU 用のカスタム カーネルを作成する方法です。
    - [<!--mkdocs::octicons-arrow-right-24:--> 動的シェイプ](./dynamic_shapes.md): 動的なテンソル シェイプのサポートです。

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> ケーススタディ**

    TorchTPU を使用して実際の問題を解決する実践的な例を紹介します。

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX の最適化](./case_studies_flux.md): FLUX で複数の最適化を特定して適用する方法を示す例です。
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B の推論](./case_studies_muse_glimmer_30b_inference.md): 実際の重みと静的 / 動的 KV キャッシュを使用した反復デコードの例です。
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B のトレーニング](./case_studies_muse_glimmer_30b_training.md): Muse Glimmer 30B のスタンドアロン実装でのトレーニング ループです。
    - [<!--mkdocs::octicons-arrow-right-24:--> FSDP の使用](./case_studies_fsdp.md): TorchTPU で FSDP を使用するための推奨アプローチです。

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> ワークロードを分析する**

    - [<!--mkdocs::octicons-arrow-right-24:--> 数値精度とパリティ](./numerical_precision.md): TorchTPU で精度を制御する方法です。
    - [<!--mkdocs::octicons-arrow-right-24:--> XProf でパフォーマンスをプロファイリングする](./performance_profiling_with_xprof.md): XProf を TorchTPU のプロファイラとして使用する方法です。

<!--mkdocs:</div>-->
<!-- mdformat on -->
