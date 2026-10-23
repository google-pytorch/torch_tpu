# TorchTPU でサポートされている機能と今後の取り組み { #torchtpu-supported-feature-set-and-future-work }

## 現在サポートされている機能 { #current-supported-feature-set }

以下の表は、最初のオープンソース リリース時点での機能のカバレッジを示しています。

ワークロードで TorchTPU を評価しやすいように、機能を 2 つの段階に分類しています。

*   **検証済み:** エンドツーエンドの継続的なテスト、自動化された CI/CD のカバレッジ、標準的なモデル アーキテクチャ全体でのベンチマーク検証が行われている機能です。
*   **試験運用版:** 初期段階の実装、新しい API、プロトタイプの機能です。これらは動作し、早期フィードバックのために利用できますが、コミュニティからの意見に基づいて動作が調整されたり API が変更されたりする可能性があります。

> **Note:** TorchTPU はオープンな環境で急速に進化しています。最新のステータスを確認したり、特定の機能をリクエストしたりするには、[GitHub](https://github.com/google-pytorch/torch_tpu) にアクセスしてください。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **検証済み** | **試験運用版** |
| ----- | ----- | ----- |
| **PyTorch コミュニティとの連携** | <ul><li>TorchTPU、vLLM、SGLang、TorchTitan、TorchRec の OSS リリース</li><li>外部 PR の受け入れ</li><li>ドキュメント / チュートリアル</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>ATEN 移行用の TPU エージェント</li></ul> |
| **TPU のサポート** | <ul><li>v5e、v5p、v6e、v7</li></ul> |  |
| **基本機能** | <ul><li>容易な移行</li><li>eager とコンパイルの混在</li><li>FW \+ BW</li><li>Torch.profile</li><li>動的な演算（再コンパイルあり）</li></ul> | <ul><li>SymInt</li><li>Jagged テンソル</li></ul> |
| **モデル アーキテクチャ** | <ul><li>Transformer（Dense \+ MoE）: Qwen、Llama、Muse Glimmer、Gemma</li><li>拡散モデル: WAN、FLUX</li><li>CNN</li></ul> | <ul><li>GLM、Kimi、DeepSeek、GPT OSS など多数（段階的に改善中）</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **カーネルのサポート** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **量子化** | <ul><li>テンソル全体の FP8/FP4</li><li>チャネルごとの FP8</li></ul> |  |
| **分散処理** | <ul><li>最適なメッシュ</li><li>SimpleFSDP</li><li>チェックポイント</li><li>アクティベーションのオフロード</li><li>最大 8K チップのマルチスライス</li></ul> | <ul><li>FSDPv2（低パフォーマンス）</li><li>MPMD（低パフォーマンス）</li><li>TorchComm</li></ul> |
| **エコシステムとの統合** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **本番環境に対応したソリューション** | <ul><li>推論（マルチホスト、P/D 分離）</li><li>ファインチューニング（最大 \~8K チップ）</li></ul> |  |
<!-- mdformat on -->

## 今後の取り組み { #future-work }

このセクションでは、今後の開発の展望と調査について説明します。確定したリリース スケジュールではありません。

コミュニティ ファーストのオープンソース プロジェクトとして、開発の優先順位はコミュニティからのフィードバックとデベロッパーによる採用状況に意図的に合わせています。どのワークロードが重要かを学ぶにつれて、機能、タイムライン、アーキテクチャのアプローチは変わる可能性があります。

今後の取り組みは、大きく 2 つの展望に分けています。

*   **短期的な重点領域:** コアチームが現在優先的に取り組んでいる、設計、プロトタイピング、パフォーマンスの強化を進めている領域です。
*   **今後の調査:** 長期的なアーキテクチャの調査、将来的なエコシステムとの統合、コミュニティからリクエストされたバックログ項目のうち、現在積極的に検討しているものです。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **短期的な重点領域** | **今後の調査** |
| ----- | ----- | ----- |
| **PyTorch コミュニティとの連携** | <ul><li>ドキュメント / チュートリアル / ブログの拡充</li></ul> | <ul><li>カーネル移行用の TPU エージェント</li></ul> |
| **TPU のサポート** | <ul><li>v8i、v8t</li></ul> |  |
| **基本機能** | <ul><li>ストリーム</li><li>グラフ キャプチャ</li><li>起動時間が短い eager モード</li><li>動的な演算（再コンパイルなし）</li></ul> |  |
| **モデル アーキテクチャ** | <ul><li>リリース初日からのモデル サポート</li></ul> |  |
| **量子化** | <ul><li>torchao によるブロック単位の FP8/FP4（TPUv8t でピーク パフォーマンス）</li></ul> |  |
| **分散処理** | <ul><li>FSDPv2（高パフォーマンス）</li><li>MPMD（高パフォーマンス）</li><li>8K チップ超でのトレーニング</li><li>マルチティア チェックポイント</li><li>Goodput</li></ul> | <ul><li>8K チップを大きく超えるトレーニング</li></ul>  |
| **エコシステムとの統合** | <ul><li>VeRL</li></ul> |  |
| **本番環境に対応したソリューション** | <ul><li>RL</li></ul> | <ul><li>事前トレーニング</li></ul> |
<!-- mdformat on -->
