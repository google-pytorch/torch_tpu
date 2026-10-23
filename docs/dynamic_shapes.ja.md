# 動的シェイプ { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

`torch.compile(model)` でモデルをコンパイルすると、TorchTPU は計算を StableHLO に lowering し、XLA コンパイラでコンパイルします（[TorchTPU の仕組み](./how_torchtpu_works.md)をご覧ください）。XLA コンパイラは従来、**静的に既知のテンソル シェイプ** を前提に設計されています。これにより、High Bandwidth Memory（HBM）のバッファを事前にレイアウトし、演算子のパイプラインを融合し、TPU のシストリック アレイ（MXU）向けに行列乗算をタイル化できます。`torch.compile` ワークロードで入力テンソルの次元が呼び出しごとに変わる場合、正確なシェイプごとにコンパイルすると Dynamo のシェイプ ガードが無効になり、新しいシェイプが現れるたびにコストの高い XLA の完全な再コンパイルがトリガーされます。

絶え間ない再コンパイルなしで **動的シェイプ** をサポートするために、TorchTPU は **有界の動的性（Bounded Dynamism）** を使用します。有界の動的性では、動的な次元のコンパイル時の上限は固定され、実行時の実際の次元サイズは XLA によって動的に追跡されます。

## TorchTPU による動的シェイプの処理方法 { #how-dynamic-shapes-are-handled-by-torchtpu }

デフォルトの動的性、つまり `torch.compile(model, dynamic=None)` でモデルや関数をコンパイルする場合（入力シェイプが呼び出しごとに変わる場合）や、`torch._dynamo.mark_dynamic` で動的な次元を明示的にマークする場合、Dynamo は計算をトレースして FX グラフを生成します。この FX グラフでは、変化する次元が固定の定数ではなくシンボリックな整数変数（`torch.SymInt`）として表されます。TorchTPU は、これらのシンボリックな次元を次のように処理します。

### コンパイル { #compilation }

*   TorchTPU はトレースされた FX グラフをインターセプトし、各シンボリックな次元について全体の `[lower_bound, upper_bound]` 範囲を決定します（`mark_dynamic(..., min=..., max=...)` から取得するか、最初の実行時サイズ `S` から Dynamo が推論します）。次に、その範囲を `bucket_size`（デフォルトは `128` で、TPU の 128 レーンのタイル次元に一致）の倍数に揃えた連続するバケットに、最大 `num_dynamic_buckets`（デフォルトは `1`、最大は `8`）個まで分割します。
    *   最初のバケットは `lower_bound` から `S` 以上の次の `128` の倍数（`U_1`）までを、2 番目のバケットは `U_1 + 1` から次の `128` の倍数（`U_2`）までをカバーし、以降も同様です（たとえば、`[lower_bound, 128]`、`[129, 256]`、...）。
    *   TorchTPU はすべてのバケットの全範囲にわたる単一の範囲ガードを Dynamo に登録します。そのため、Dynamo は次元が全体の `upper_bound` を超えて大きくなった場合にのみ再コンパイルをトリガーします。
*   TorchTPU は各バケットについて、シンボリックな次元をバケットの静的な上限 `U` に置き換え、実行時の実際の次元サイズを入力として渡し、静的な上限 `U` とアクティブな実行時サイズの両方をすべての演算で追跡する有界テンソル型を使用してグラフを StableHLO に lowering します。
*   生成されたモジュールは、構成された各バケットの上限 `U` ごとに XLA によってコンパイルされます。

### 実行 { #execution }

*   実行時には、動的な次元の実際のサイズに基づいて、TorchTPU が対応するコンパイル済みのバケット実行可能プログラムを選択します。

*   コンパイル済みのモジュールを実行するとき、XLA は動的な入力テンソルをバケットの静的な上限 `U` までパディングし、アクティブな実行時サイズを超えるパディング領域の値を代数的な単位元の値にリセットします。これにより、パディングされた要素が出力結果に影響しないようにします。

*   動的な出力テンソルでは、パディングはバッファの末尾に残ります。そのため、TorchTPU はテンソルを物理的にスライスするのではなく、軽量なシェイプ メタデータの更新だけでバッファを PyTorch に返すことができます。

## 動機となるユースケース: 自己回帰デコード { #motivating-use-case-autoregressive-decoding }

有界の動的性の主な動機となるワークロードは、動的に増加する KV キャッシュを使用した **自己回帰 LLM デコード** です。標準的なデコード ループでは、各ステップで新しいトークンの key と value の状態が、`torch.cat` によってシーケンスの次元に沿ってキャッシュされた履歴に追加されます。

*   **プレフィル:** 入力プロンプトのシェイプは `[B, S]`（バッチサイズ `B`、プロンプトのシーケンス長 `S`）で、シェイプ `[B, H_kv, S, D]`（`H_kv` 個の key/value ヘッド、ヘッドの次元 `D`）の KV キャッシュを生成します。
*   **デコード ステップ 1:** 入力トークンのシェイプは `[B, 1]` で、キャッシュと連結されてシェイプ `[B, H_kv, S + 1, D]` になります。
*   **デコード ステップ `t`:** KV キャッシュのシーケンス長は `S + t` まで増加します。

静的シェイプのコンパイル（`dynamic=False`）では、Dynamo は KV キャッシュの正確なシーケンス長に対してガードを設定します。デコード ステップごとにテンソルのシェイプが変わり、ガードが無効になり、**生成されるトークンごとに XLA の完全な再コンパイル** がトリガーされます。その結果、静的なバッファを事前に割り当ててインプレース変更を行うようにモデルを書き換えない限り、ミリ秒単位のデコード ステップが数秒のコンパイルになってしまいます。

**有界の動的性** は、モデルを書き換えることなくこの問題を解決します。デコード ステップが進んで KV キャッシュが増加しても（`S + 1`、`S + 2`、...）、コンパイル済みのバケット（`<= 128`、`<= 256`、...）内のすべてのシェイプが同じ XLA 実行可能プログラムを再利用します。

有界の動的性を LLM のデコード ループ全体に適用する実践的なチュートリアル（静的な KV キャッシュの事前割り当てとの比較を含む）については、[エンドツーエンドの LLM 推論](./case_studies_muse_glimmer_30b_inference.md)をご覧ください。

## 例: 増加するテンソルを使用した自己回帰ループ { #example-autoregressive-loop-with-a-growing-tensor }

静的コンパイル（`dynamic=False`）と、デフォルトの動的性（`dynamic=None`）を使用した有界の動的性が実際にどのように動作するかを確認するために、各ステップで `torch.cat` を使用してシーケンスの次元（`dim=2`）に沿って新しいトークンを追加する、最小限の自己回帰ループを考えてみましょう。

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. `dynamic=False` で実行する { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

出力例:

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**何が起きたか:** `dynamic=False` では、Dynamo は `kv_cache` の正確なシーケンス長（`16`、`17`、`18`、...）に対してガードを設定します。`torch.cat` は反復ごとに `dim=2` を `1` ずつ増やすため、シェイプ ガードは呼び出しのたびに失敗し、すべてのステップで XLA の完全な再コンパイルがトリガーされます。

### 2. `dynamic=None` で実行する（有界の動的性） { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

出力例:

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**何が起きたか:** `dynamic=None` では、コンパイルのオーバーヘッドが発生するのは最初の 3 回の反復だけです。ステップ 4 以降は、実行時間がステップあたり約 0.4 ms（約 100 倍高速）に下がり、再コンパイルは発生しません。

*   **ステップ 1（`seq_len=16 -> 17`）:** Dynamo は入力を初めて認識し、`seq_len=16` 用の静的な実行可能プログラムをコンパイルします。
*   **ステップ 2（`seq_len=17 -> 18`）:** `dim=2` が `16` から `17` に変わったため、Dynamo は `dim=2` を自動的に動的（`SymInt`）としてマークし、TorchTPU は最初の入力テンソルのレイアウトを使用して `[17, 128]` バケット用の有界動的な実行可能プログラムをコンパイルします。
*   **ステップ 3（`seq_len=18 -> 19`）:** ステップ 2 で生成された有界動的な出力バッファは XLA のパディングされた出力レイアウトを持つため、TorchTPU はその定常状態の XLA レイアウト用に `[17,
    128]` バケットをもう一度コンパイルします。
*   **ステップ 4～10（`seq_len=19 -> 26`）:** シーケンス長（`<= 128` バケット内）と XLA バッファのレイアウトの両方がキャッシュされた実行可能プログラムと一致するため、後続のステップはすべて、再コンパイルせずにコンパイル済みの XLA 実行可能プログラムを直接ディスパッチします。

> **Tip:** どの次元が変化するかが事前にわかっている場合は、最初のステップの前に `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` を呼び出すことができます。これにより、Dynamo はステップ 1 から直ちに `dim=2` を動的としてトレースします。ステップ 1 での最初の静的コンパイルがスキップされ、明示的な `[min, max]` の範囲を指定できます。

## 現在の制限事項 { #current-limitations }

TorchTPU の有界の動的性は積極的に開発が進められており、現在は **自己回帰の KV キャッシュ** のユースケースを中心に最適化されています。モデルに適用する際は、次の制限事項に留意してください。

*   **バケットの上限と非有界の動的性:** トレースごとにコンパイルされるバケットは最大 **8 個**（`num_dynamic_buckets <= 8`）です。非有界の動的性はまだサポートされていません。動的な次元が、コンパイル済みの最も大きいバケットの上限を超えて増加すると、再コンパイルがトリガーされます。
*   **Dynamo のシェイプ制約:** バケットの上限は、有効な Dynamo のシェイプ制約（`reshape` や `view` による割り切れる必要があるという要件など）を満たす必要があります。満たさない場合、TorchTPU は上限が `2 * S`（最初の実行時サイズ `S` の 2 倍）の単一のバケットにフォールバックします。
*   **バケットの上限までの計算と HBM のオーバーヘッド:** 実行はバケットの静的な上限 `U` までパディングされたバッファ上で行われるため、HBM のフットプリントと FLOP は実際の実行時サイズではなく `U` に応じて増加します。実際のサイズが `U` よりはるかに小さいときに、非常に大きな `bucket_size` や `mark_dynamic(..., max=U)` を設定すると、`U` の計算コストとメモリコストを支払うことになります。
*   **XLA のレイアウトの変更:** テンソルのシェイプがコンパイル済みのバケット内に収まっていても、入力バッファの基盤となる XLA の物理レイアウトが変わると、バケットの再コンパイルがトリガーされます。
*   **永続コンパイル キャッシュ（`AOTAutogradCache`）なし:** 有界動的な実行可能プログラムはまだ `AOTAutogradCache` にシリアル化されないため、新しいプロセスが起動するたびに再コンパイルされます。
*   **演算子のカバレッジ:** すべての PyTorch 演算子が有界の動的シェイプをサポートしているわけではありません。
*   **モデル入力としてのビュー:** テンソルのビューをコンパイル済みのモデルに入力として直接渡すことのサポートは限られています。
*   **データ依存の動的性:** 出力シェイプが実行時のテンソルの値に依存する演算（`torch.nonzero` など）はサポートされていません。
*   **カスタム カーネル:** 有界の動的シェイプは、カスタム TPU カーネル（`torch.tpu.pallas.jax_op` で登録された Pallas カーネルなど）ではサポートされていません。
*   **dynamic=True:** torch.compile() の *dynamic=True* モードはサポートされていません。
