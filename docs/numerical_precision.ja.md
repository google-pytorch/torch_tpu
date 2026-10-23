# TorchTPU で数値精度を管理する { #manage-numerical-precision-in-torchtpu }

Google TPU は、スループットと数値精度のバランスを取るために、浮動小数点演算を 2 つのハードウェア ユニットに分けて実行します。

*   **行列乗算ユニット（MXU）:** 縮約演算（`torch.matmul`、`torch.mm`、`torch.bmm`、畳み込みなど）を、ネイティブの `bfloat16` 入力（8 ビットの仮数）と 32 ビットの `float32` 累積を使用して実行します。
*   **ベクトル実行ユニット（VPU）:** 縮約以外の演算（要素ごとの算術演算、活性化関数、レイヤ正規化、softmax など）を、完全な IEEE-754 `float32`（24 ビットの仮数）で実行します。

このガイドでは、`float32` の行列乗算の精度モードを構成する方法、CPU と TPU の間の数値パリティを検証する方法、出力のアキュムレータの型と超過精度を制御する方法、コードで MXU のビット単位の動作を検証する方法について説明します。

### CUDA と TPU での PyTorch の精度を比較する { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU は、標準の PyTorch の精度 API（`torch.set_float32_matmul_precision` と `torch.get_float32_matmul_precision`）に加えて、TPU 固有の制御（`torch.tpu.Precision` と `torch.tpu.precision`）をサポートしています。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 観点          | CUDA 上の PyTorch                      | TPU 上の PyTorch（TorchTPU）              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **デフォルトの精度モード** | `"highest"`（純粋な `float32`） | `"medium"`（`Precision.DEFAULT`）       |
| **高速演算のアクセラレータ プリミティブ** | Tensor Core（`TF32`） | シストリック MXU（`bfloat16`）             |
| **高速演算のビット** | 符号 1 + 指数 8 + 仮数 10 ビット（19 ビットの `TF32`） | 符号 1 + 指数 8 + 仮数 7 ビット（16 ビットの `bfloat16`） |
| **累積の精度** | `float32`（`"highest"`、`"high"`）、`TF32`（`"medium"`） | すべてのモードで `float32` |
| **より高い精度を実現する仕組み** | 専用ハードウェアの `float32`/`TF32` ユニット | マルチパス分解（MXU を 3 パスまたは 6 パス） |
| **`high`/`highest` のパフォーマンスへの影響** | ベースラインの速度またはわずかな低下（約 1～2 倍） | `"high"` で MXU の計算コストが約 3 倍、`"highest"` で約 6 倍 |
| **スコープの API** | グローバルな `torch.set_float32_matmul_precision` | グローバルな <br> `torch.set_float32_matmul_precision` + スコープ付きの `torch.tpu.precision` |
<!-- mdformat on -->

## 精度モードを指定する { #specify-precision-modes }

MXU で `float32` テンソルを乗算する場合、TorchTPU は PyTorch の精度レベルを `torch.tpu.Precision` 列挙型にマッピングします。

*   `"medium"` / `Precision.DEFAULT`（1 パス）: **デフォルト。** `float32` 入力を `bfloat16` に丸め（仮数の上位 8 ビットを乗算）、`float32` で累積します。ほぼすべてのトレーニングと推論のワークロードにおすすめです。
*   `"high"` / `Precision.HIGH`（3 パス）: `float32` 入力を上位と下位の `bfloat16` 成分に分割し、3 つのクロス項を計算します（仮数の最初の 16 ビットを乗算）。MXU の計算コストは約 3 倍になります。
*   `"highest"` / `Precision.HIGHEST`（6 パス）: `float32` 入力を 3 つの `bfloat16` 成分に分割し、6 つのクロス項を計算して（仮数の 24 ビットすべてを乗算）、完全な IEEE-754 `float32` の精度を近似します。MXU の計算コストは約 6 倍になります。

行列乗算の精度は、次の 2 つの方法で構成できます。

*   **グローバル構成:** `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` を呼び出して、プロセスのデフォルトモードを設定します。
*   **スコープ付きのコンテキスト マネージャ:** `with torch.tpu.precision(mode):` を使用して、数値的に影響を受けやすいレイヤやサブモジュールにのみ高い精度を適用します。
*   **パフォーマンスに関する警告:** `"high"` または `"highest"` を選択すると、1 回限りの警告（`TORCH_WARN_ONCE`）が出力されます。精度に関連する収束の問題が見られない限り、デフォルトの `"medium"` モードのままにしてください。

> **Note:** `torch.tpu.precision` に入ったり出たりすると、`torch.compile` のグラフ ブレークが発生します。`torch.compile(..., fullgraph=True)` を使用する場合は、コンパイルされる関数の外側で `torch.set_float32_matmul_precision(...)` を使用して精度を設定するか、外側の呼び出しを `with
> torch.tpu.precision(...):` でラップしてください。

`torch.tpu.precision` を使用して、3 つのモードの定常状態の実行時間を比較します。

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### 精度モード間で精度を比較する { #compare-accuracy-across-precision-modes }

同じ入力に対する CPU の `float32` のリファレンスと TPU の出力を比較して、各精度モードで数値誤差がどのように減少するかを確認します。

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## 数値パリティを検証する { #verify-numerical-parity }

モデルを TPU に移植する場合や、収束の違いをデバッグする場合は、重みを同期し、同一の入力を使用して、同じレイヤまたはサブモジュールを CPU と TPU で実行し、`torch.testing.assert_close` で出力を比較して数値パリティを検証します。

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### 不一致の詳細を解釈する { #interpret-detailed-mismatches }

テンソルが許容誤差を超えると、`torch.testing.assert_close` は概要を出力します。この概要は、通常の浮動小数点のずれとロジックのバグを区別するのに役立ちます。

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** 絶対差が小さい要素の割合が高い場合、通常はバグではなく、標準的な丸めや、コンパイラの融合による累積順序の違いを示しています。
*   **Greatest absolute / relative difference:** 最大の誤差とそのテンソル インデックスを報告します。これを使用して外れ値を調べたり、適切な `atol` と `rtol` のしきい値を選択したりできます。

### パリティのベスト プラクティス { #best-practices-for-parity }

*   **両方のデバイスでデータ型を一致させる:** TPU の `bfloat16` は CPU の `bfloat16` と比較し、TPU の `float32`（`Precision.HIGHEST` を使用）は CPU の `float32` と比較します。`bfloat16` を `float32` と比較すると、ハードウェアのずれではなく量子化誤差を測定することになります。
*   **アクティベーションだけでなく勾配も確認する:** `loss.backward()` を呼び出した後、同じパターンを使用して `model_tpu.weight.grad.cpu()` を `model_cpu.weight.grad` と比較します。
*   **標準の許容誤差から始める:** [PyTorch のテストに関するドキュメント](https://docs.pytorch.org/docs/stable/testing.html)のデフォルトの許容誤差をベースラインとして使用し、精度を下げた累積で想定される箇所でのみ `atol` または `rtol` を緩めます。

## 出力の精度を指定する { #specify-output-precision }

`torch.mm` または `torch.bmm` で `bfloat16` テンソルを乗算する場合、MXU は積を `float32` で累積してから、最終的な出力を `bfloat16` に丸めて戻します。ダウンキャストせずに `float32` のアキュムレータを直接返すには、`out_dtype=torch.float32` を渡します。

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## 超過精度を制御する { #control-excess-precision }

TPU の MXU は `bfloat16` の行列乗算を `float32` で累積するため、XLA コンパイラは、ステップ間で中間値を `bfloat16` に丸めずに `float32` のまま保持することで、融合された演算を最適化できます。このような追加の中間ビットを保持することを *超過精度*（excess precision）と呼びます。

この動作は、`torch.tpu.allow_excess_precision`（基盤となる `--xla_allow_excess_precision` コンパイラ フラグを設定し、デフォルトは `False`）を使用してグローバルに構成できます。

*   **超過精度を無効にする（`False`、デフォルト）:** `torch.tpu.allow_excess_precision = False` を設定すると、コンパイラは `bfloat16` への中間のダウンキャストを保持するため、融合された演算が eager の `bfloat16` の丸め動作と一致します。
*   **超過精度を許可する（`True`）:** コンパイラが中間のダウンキャストを省略し、融合された中間結果を `float32` のまま保持できるようにするには、`torch.tpu.allow_excess_precision =
    True` を設定します。

> **Important:** `torch.tpu.allow_excess_precision` は、TPU の演算を実行またはコンパイルする前に、スクリプトの冒頭で設定してください。XLA ランタイムは最初のコンパイルが開始された時点でこのフラグをロックします。その後に別の値に変更しようとすると、`RuntimeError` が発生します。

## TPU の MXU の精度を理解する { #understand-tpu-mxu-precision }

TPU の MXU が浮動小数点数をどのように扱うかを理解すると、超過精度とマルチパスの `float32` モードがなぜそのように動作するのかがわかります。

*   **範囲と精度（`bfloat16`）:** [`bfloat16` に関する Google Cloud の研究](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)では、ニューラル ネットワークは、`float32` の広いダイナミック レンジ（ノルムや勾配などの演算に不可欠）を維持していれば、仮数の精度を下げてもモデルの精度を損なうことなくトレーニングできることが示されています。`float32` の仮数は 23 ビット（暗黙の先頭ビットを含めると仮数の精度は 24 ビット）ですが、`bfloat16` は同じ 8 ビットの指数範囲を維持しつつ、仮数は 7 ビットのみです（仮数の精度は 8 ビットで、マシン イプシロンは `2**-7`、つまり約 0.78%）。
*   **交換形式と算術形式:** 浮動小数点システムでは、**交換形式**（数値をメモリに保存する方法）と **算術形式**（ハードウェアが中間演算を計算して丸める方法。標準的な ALU の 3 つの追加ビットである guard、round、sticky ビットなど）を区別します。`bfloat16` はどちらも指す場合があります。
*   **縮約次元に沿った累積:** シェイプ `[I, J] @ [J, K]` の 2 つの行列を乗算すると、シェイプ `[1,
    J] @ [J, 1]` の内積のグリッドが計算されます。各内積は、**縮約次元** `J` に沿って
    `J` 個の要素ごとの積を合計します。`bfloat16` のマシン イプシロンは `2**-7`（約 `0.008`）であるため、`J` 個の積を `bfloat16` で累積すると大きな丸め誤差が生じます。代わりに、MXU は `bfloat16` 入力（TPU v7x などの新しい世代では `fp8` も）を受け付け、中間の合計を `float32` で累積します。これは、出力形式に丸める前の ALU の guard、round、sticky ビットと同じ役割を果たします。
*   **入力と出力の形式の不一致:** `bfloat16` の入力と `float32` のアキュムレータを持つことが、*超過精度*（融合された演算間で `float32` の中間値を保持すること）と *想定よりも低い精度*（1 パスの `float32` matmul で入力が 8 ビットの `bfloat16` の仮数に切り捨てられること）の両方の根本原因です。対照的に、TPU のベクトル実行ユニット（VPU）は、`float32` の 23 ビットの仮数すべてを使用して演算します（非正規化数などの IEEE-754 のわずかなエッジケースを除く）。

### `float32` のマルチパス多項式展開 { #multi-pass-polynomial-expansion-for-float32 }

MXU は `bfloat16` 入力（8 ビットの仮数）しか受け付けないため、24 ビットの完全な `float32` の行列乗算を 1 パスで計算することはできません。実際には、最新のモデルのほとんどは重みを `bfloat16` で保存しており、`float32` の matmul を実行することはまれです。完全な `float32` の matmul の精度が必要な場合、XLA は [Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)（Henry、Tang、Heinecke）のアルゴリズムを使用して、24 ビットの各 `float32` の仮数を 3 つの 8 ビットの `bfloat16` 値の合計（`High + Med + Low`）として表し、それらの積を展開します。

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

3 項の和を 2 つ乗算すると 9 つの `bfloat16` の積が得られ、そのうち最も低次の 3 つの項は `float32` で常にアンダーフローします。

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *（1 パス）* | High x Med *（3 パス）*    | High x Low *（6 パス）*    |
| **Med**  | Med x High *（3 パス）* | Med x Med *（6 パス）*     | Med x Low *（アンダーフロー）* |
| **Low**  | Low x High *（6 パス）* | Low x Med *（アンダーフロー）* | Low x Low *（アンダーフロー）* |
<!-- mdformat on -->

### 3 パスと 6 パスのパフォーマンスを評価する { #evaluate-3-pass-and-6-pass-performance }

`Precision.HIGH`（3 パス）と `Precision.HIGHEST`（6 パス）では MXU のパスが 3 倍と 6 倍必要になりますが、エンドツーエンドの影響はワークロードによって異なります。

*   **大きな縮約次元:** `float32` の累積でトレーニングされ（DeepSeek-V3 など）、大きな縮約次元を持つ最新のモデルでは、数千の要素にわたる累積のノイズと非決定性が、仮数の低次のクロス項からの寄与を上回るため、1 パス（`Precision.DEFAULT`）で十分です。
*   **メモリ帯域幅がボトルネックとなるモデル:** 古い `float32` モデルは、High Bandwidth Memory（HBM）から転送される 1 バイトあたりの算術演算が少なく、MXU（または GPU の Tensor Core）を飽和させられないことがよくあります。このようなメモリがボトルネックとなるワークロードでは、3 パスと 6 パスの実行での追加の MXU パスは、本来はアイドル状態の MXU サイクルを使用するため、エンドツーエンドの速度低下は 3 倍や 6 倍よりも小さくなります。

## ビット単位の精度の動作を検証する { #validate-bitwise-precision-behavior }

2 つの 2 進浮動小数点のルールを使用して入力を構築することで、MXU が仮数ビットと超過精度をどのように扱うかを検証できます。

*   `1 + 2**-n` は、2 進の仮数が `1.000...1000...` の数値を作成し、仮数の `n` 番目のビットに小数部の `1` を配置します。
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: 仮数の 6 番目と 7 番目のビットが設定された 2 つの数値を乗算すると、6、7、`6 + 7 = 13` 番目のビットが設定された結果が得られます。`bfloat16` は 7 ビットの仮数しか保持しないため、13 番目のビットは `bfloat16` からあふれ、`float32` のアキュムレータに保持されている場合にのみ維持されます。

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

これらの値を `256 x 256` 行列でラップすると MXU で実行されるため、13 番目の仮数ビットが丸められて失われるか、保持されるかを調べることができます。

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```
