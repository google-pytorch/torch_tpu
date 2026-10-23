# FSDP のベスト プラクティス { #fsdp-best-practices }

Fully Sharded Data Parallel（FSDP）は、モデルのパラメータ、勾配、オプティマイザの状態をデータ並列のランク間でシャーディングし、各 TPU コアがモデルの `1 / world_size` のスライスのみを保持するようにします。レイヤを実行する前に、そのレイヤの完全な重みが **all-gather** によって再構築されます。バックワード パスでは、勾配が **reduce-scatter** によって結合され、再びシャーディングされます。

このガイドでは、TorchTPU で FSDP を行う推奨の方法を説明し、[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp) にある実行可能なサンプルを紹介します。

このガイドでは、次のことを学びます。

1.  TorchTPU で **どの FSDP アプローチを選択するか**。
2.  **SimpleFSDP（推奨）**: 1 回のラップ呼び出しで使える、コンパイル優先の FSDP。
3.  **手動 FSDP**: 集団通信をモデル定義に直接記述する方法。
4.  すべてのアプローチに当てはまる **ベスト プラクティス**。
5.  **サンプルの実行**: 起動コマンドと、各サンプルで確認する内容。

## 1. FSDP アプローチを選択する { #1-choosing-an-fsdp-approach }

```
On TPU, performance comes from compiling the whole training step into a single
XLA program with `torch.compile(backend="tpu")`. When the XLA compiler sees all
forward, backward, and collective operations at once, it can schedule
collectives to overlap with computation and combine small collectives when
profitable.

Choose an FSDP approach based on how well it composes with whole-graph
compilation:
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
    | アプローチ | 使用する場面 | サンプル |
    | :------- | :------------- | :------ |
    | **SimpleFSDP**（`simple_fsdp.data_parallel`） | **推奨のデフォルト。** モデル定義を変更せずに FSDP または HSDP を使用したい場合。| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **手動 FSDP**（手動の集団通信） | モデルのコードを自分で管理しており、パフォーマンス チューニングのために通信コードを計算の近くに置きたい場合。 | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2**（`fully_shard`） | まだ移行されていない既存のコード。動作はしますが、ランタイム フックによってグラフ ブレークが発生します。また、競争力のあるパフォーマンスを得るには、TPU の SparseCore へのストリーム オフロードの今後のサポートが必要です。
.| [移行ガイド](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
<!-- mdformat on -->

```
SimpleFSDP and manual FSDP both express the collectives as part of the traced
graph, so XLA can optimize them together with the rest of the step. PyTorch
FSDPv2 (`fully_shard`) relies on Python pre-forward and post-backward hooks.
These hooks split the step into many smaller graphs, even when the model is
compiled.

For a side-by-side comparison of compiled FSDPv2 and compiled SimpleFSDP, see
the
[FSDPv2 to SimpleFSDP tutorial](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md).
```

## 2. 推奨: SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py) は、`torch.compile` と連携するように設計された FSDP の実装です。TorchTitan で導入されたもので、PyTorch のみに依存する自己完結型のコピーが `examples/fsdp/simple_fsdp.py` で提供されています。

SimpleFSDP は、各パラメータをシャーディングされた `DTensor` として保存します。パラメータにアクセスすると、パラメータ化（parametrization）によって `DTensor.redistribute` が呼び出され、パラメータが all-gather されます。`redistribute` の autograd のバックワード パスが勾配の reduce-scatter を実行します。ランタイム フックがないため、ステップ全体がグラフ ブレークなしでコンパイルされます。モデル全体は 1 回の `simple_fsdp.data_parallel(...)` 呼び出しでシャーディングされ、レイヤごとのラップやプリフェッチのチューニングは不要です。また、2D の `DeviceMesh` で `mode="hybrid_shard"` を使用すると HSDP がサポートされます。

SimpleFSDP の使用方法については、以下をご覧ください。

*   [SimpleFSDP README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md): クイックスタートと利用可能なオプション。
*   [FSDPv2 から SimpleFSDP へのチュートリアル](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md): 既存の `fully_shard` コードを移行する方法。

## 3. 代替手段: 手動 FSDP { #3-alternative-manual-fsdp }

モデル定義を管理できる場合は、手動の集団通信（`torch.distributed._functional_collectives`）と `torch.compile` を使用して FSDP を直接記述できます。各レイヤは自身の重みのシャードのみを保持し、フォワード パスで完全な重みを all-gather します。関数型集団通信には autograd の式があるため、all-gather のバックワード パスは reduce-scatter になり、カスタムの `torch.autograd.Function` を記述する必要はありません。

この方法では通信コードを計算コードの近くに置けるため、パフォーマンス チューニングが容易になる場合があります。ただし、モデル定義を変更する必要があるというトレードオフがあります。

手動 FSDP の実装方法については、[手動 FSDP ガイド](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md)をご覧ください。

## 4. ベスト プラクティス { #4-best-practices }

以下の推奨事項は、SimpleFSDP と手動 FSDP に当てはまります。

1.  **`torch.compile(backend="tpu")` でトレーニング ステップ全体をコンパイルします。** ステップ全体をコンパイルできない場合は、少なくともフォワード パス、損失、バックワード パスをまとめてコンパイルします。モデル モジュールだけをコンパイルしても単一のグラフにはならないため、XLA が集団通信を計算とオーバーラップさせる効果が制限されます。
2.  **シャーディングされた各レイヤでアクティベーション チェックポイントを使用します。** 各レイヤのフォワード パスを `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` でラップします。これにより、シャーディングされていない重みがバックワード パスまで保持されず、各レイヤのフォワード パスの後に解放されるため、FSDP に期待される ZeRO-3 のメモリ動作が得られます。
3.  **シャーディングする次元をメッシュサイズで割り切れるようにします。** 手動 FSDP ではこれが必須です。SimpleFSDP は、シャーディングするには小さすぎるパラメータを複製します。`shard_dim` を使用して、均等に分割できる次元を選択してください。
4.  **シャーディングされた重みを初期化するときは、ランクごとに異なるシードを設定します。** すべてのランクが同じシードで自身のシャードを初期化すると、完全な重みは同じランダムなブロックがランク間で繰り返されたものになり、収束に悪影響を与える可能性があります。再現可能で互いに異なるシャードにするには、`torch.manual_seed(base_seed + dist.get_rank())` のように、ランクに依存する値で各ランクのシードを設定します。
5.  **すべてのランクで同じプログラムを実行し続けます。** 実体化を強制する演算（`.item()` など）を、`if
    rank == 0` のようなランク固有の分岐の中に置かないでください。
6.  **メッシュ全体での完全なシャーディングがスケールしない場合は、HSDP を使用します。** ICI スライスが大きすぎてモデルを効果的にシャーディングできない場合や、スライス全体でシャーディングするとネットワーク トラフィックが多くなりすぎる場合は、2D メッシュで `mode="hybrid_shard"` を使用します。複数のスライスにまたがって実行する場合は、ほとんどの場合、低速な DCN 間では複製（データ並列）し、各スライス内でシャーディングするのが最適です。
7.  **スケールアップする前に、メモリと数値を確認します。** `assert_close` を使用して、シャーディングされていないリファレンスまたは FSDPv2 のリファレンスと比較し、フォワード パスとバックワード パスの境界をまたいで保持されるメモリを測定します。サンプルには、パリティ チェック（`fsdp_to_simple_fsdp_worker.py`）とメモリテスト（`manual_fsdp_memory_test.py`）が含まれています。

## 5. サンプルを実行する { #5-running-the-examples }

各サンプルには、8 個の TPU コア（`v6e-8` など）で実行するランチャー スクリプトが付属しています。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| サンプル | 起動コマンド | 確認する内容 |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | コンパイルされた FSDPv2 と <br> コンパイルされた SimpleFSDP の損失と <br> 予測が一致すること。また、それぞれの <br> ステップ時間が報告されます。 |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` または `python3 launch.py` | 手動 FSDP の勾配と予測が <br> 単一デバイスの CPU リファレンスと一致すること。 |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | シャーディングされていない重みが <br> フォワード パスの後に解放されること。 |
<!-- mdformat on -->
