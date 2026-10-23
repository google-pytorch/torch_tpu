# TorchTPU に関するよくある質問 { #torchtpu-faq }

## PyTorch/XLA と TorchTPU の違いは何ですか？ { #what-is-the-difference-between-pytorchxla-and-torchtpu }

PyTorch/XLA は従来、JAX のランタイム コンポーネントと専用の API に大きく依存するブリッジ フレームワークとして機能してきました。TorchTPU は [Google Cloud TPU](https://cloud.google.com/tpu) 向けのネイティブな PyTorch バックエンドを提供し、JAX ランタイムへの依存はまったくありません。既存の PyTorch ワークロードは、ごくわずかなコード変更（例: `.to("cuda")` → `.to("tpu")`）だけでそのまま動作します。

## TorchAX と TorchTPU の違いは何ですか？ { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX（torch_xla2 とも呼ばれます）は、PyTorch の演算を Python レベルでインターセプトし、XLA に（多くの場合 JAX を介して）lowering する実装です。TorchAX は推論のベンチマークやプロトタイピングで広く使われてきましたが、TPU でそのまま置き換えて使える PyTorch ネイティブな体験を提供する新しい公式標準は TorchTPU です。

## PyTorch/XLA は今後どうなりますか？ { #what-is-happening-to-pytorchxla }

PyTorch/XLA は非推奨となり、現在公開されている [TorchTPU](https://github.com/google-pytorch/torch_tpu) に置き換えられます。

## TorchTPU は JAX に取って代わるものですか？ { #is-torchtpu-replacing-jax }

いいえ。TorchTPU は、Cloud TPU へのネイティブな PyTorch の入り口を提供することで TPU エコシステムを補完するよう設計されています。既存の PyTorch コードベースを持つデベロッパーは、書き換えなし、または最小限の書き換えで TPU 上で直接実行できます。モデルを JAX で書き直したり、GPU と TPU のインフラストラクチャ用に別々のコードベースを管理したりする必要はありません。

*   **TorchTPU を検討すべきなのはどんなときですか？** 既存の PyTorch コードベースがある場合、リファクタリングなしで複数のハードウェア（GPU/TPU）間の移植性が必要な場合、PyTorch ネイティブなフレームワーク（例: TorchTitan、vLLM、SGLang、Hugging Face）を使っている場合、または GPU から TPU へ迅速に移行したい場合です。

## TorchTPU の開発者と連絡を取るにはどうすればよいですか？ { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) にメールでフィードバックを送れます。また、リポジトリ（[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu)）への直接のコントリビューションも歓迎しています。[discuss.pytorch.org](https://discuss.pytorch.org/) で「TPU」タグを使うこともできます。

## コミュニティと交流するフォーラムはありますか？ { #will-there-be-any-community-engagement-forums }

はい。コミュニティとの交流は私たちにとって非常に重要であり、そうしたフォーラムの詳細を今後お知らせします。さらに、毎月オフィスアワーを開催する予定です。

## TorchTPU はカーネル開発をどのようにサポートしていますか？ { #what-support-does-torchtpu-provide-for-kernel-development }

[TorchTPU のカスタム カーネルのサポート](./custom_kernels.md)をご覧ください。

## どの TPU 世代が現在サポートされていますか？今後のハードウェアのサポート予定はどうなっていますか？ { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU は v5e、v5p、Trillium（TPU v6e）、Ironwood（TPU v7x）を現在サポートしています。TPU v8 と今後のチップの検証と最適化は、ロードマップの一環として進行中です。

## TorchTPU のサポート対象機能と今後のロードマップはどこで確認できますか？ { #where-can-we-find-torchtpus-supported-features-future-roadmap }

現在の機能セットと今後予定されている機能セットについては、[サポートされている機能](./feature_set.md)をご覧ください。

## オンボーディングを簡単にするエージェントはありますか？ { #is-there-an-agent-to-make-my-onboarding-easier }

TPU Agent Suite（試験運用版）をご覧ください: [https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

TPU Agent Suite は、Cloud TPU v6e と v7x 上の PyTorch モデルのオンボーディング、診断、推論パフォーマンスのチューニングを自動化する専用のサブエージェントを提供するオープンソースのプラグインです。

## 独自のサービング ソリューションで TorchTPU を使えますか？ { #can-i-use-torchtpu-with-our-custom-serving-solution }

はい。TorchTPU は vLLM や SGLang など業界標準のサービング エンジンとすぐに使える統合を提供していますが、独自または社内のサービング ソリューションも、ネイティブな PyTorch 実行 API と下位レベルのコンパイラ バックエンドを介して TorchTPU と統合できます。高度なオーケストレーションが必要な場合は、llm-d を使って GKE 上に独自のサービング構成をデプロイすることもできます。
