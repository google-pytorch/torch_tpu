# TPU VM で TorchTPU を使ってみる { #get-started-with-torchtpu-on-a-tpu-vm }

<!-- Link to colab and to generated notebook -->

[![Colab で開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/get_started_on_a_vm.ipynb)

TorchTPU を使用する最も簡単な方法は、TPU VM 上で使用することです。この例では、シンプルなテンソル並列の例を実行します。例そのものではなく、VM 上で実行するためのセットアップ コードに焦点を当てます。次のチュートリアルでは、GKE を通じて同様のことを行う方法を説明します。最新の TPU ハードウェアにアクセスするには GKE が必要です。

## 1. インストールとセットアップ { #1-install-and-set-up }

このノートブックを実行する前に、[Google Cloud TPU VM](https://docs.cloud.google.com/tpu/docs/intro-to-tpu?hl=en)（v6e など）上で実行していることを確認してください。

TPU VM に TorchTPU とその依存関係をインストールします。PyTorch ディスパッチャと TPU ドライバが正しく連携するように、コンポーネントは次の順序でインストールする必要があります。

### ステップ 1: Python 3.12 を初期化する { #step-1-initialize-python-312 }

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv env
source env/bin/activate
```

### ステップ 2: TorchTPU と依存関係をインストールする { #step-2-install-torchtpu-and-dependencies }

```bash
pip install torch_tpu
```

## 2. ハードウェアを初期化する { #2-initialize-the-hardware }

CPU や GPU と同様に、TPU バックエンドは明示的に初期化する必要があります。これにより、"tpu" デバイス文字列が PyTorch に登録されます。

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu

print(f"TPU Initialization complete. Primary device: {device}")
```

## 3. テンソルを操作する { #3-work-with-tensors }

初期化が完了すると、GPU とほぼ同じように TPU を操作できます。既存の CPU テンソルを TPU に移動することも、ハードウェア上に新しいテンソルを直接作成することもできます。

TPU デバイスを指定する標準的な方法は 2 つあります。

1.  `torch.device()` から返される `device` オブジェクトを使用する。
2.  `"tpu"` 文字列識別子を使用する。

```python
# 1. Create a tensor on the CPU
cpu_tensor = torch.ones((5, 5))

# 2. Move it to the TPU
tpu_tensor = cpu_tensor.to(device)  # Using the device object
# OR
tpu_tensor_alt = cpu_tensor.to("tpu")  # Using the "tpu" string

# 3. Create a tensor directly on the TPU
direct_tpu_tensor = torch.randn((5, 5), device="tpu")

print(f"cpu_tensor is on: {cpu_tensor.device}")
print(f"tpu_tensor is on: {tpu_tensor.device}")
print(f"direct_tpu_tensor is on: {direct_tpu_tensor.device}")
```

## 4. 計算して結果を実体化する { #4-compute-and-materialize-results }

TPU 上での計算はシームレスです。TPU テンソルに対して演算を実行すると、計算はハードウェア上で行われます。出力したり標準の Python ライブラリで扱ったりするためにデータを CPU に戻すには、`.cpu()` メソッドを使用します。

```python
# Perform a simple operation on the TPU
result_on_tpu = tpu_tensor + direct_tpu_tensor

# Bring the result back to the CPU to print or use in other libraries
result_on_cpu = result_on_tpu.cpu()

print("Computation complete.")
print(f"Result Checksum (Sum): {result_on_cpu.sum():.4f}")
```

### 🎉 成功です！ { #success }

TPU の初期化、デバイスへのテンソルのマッピング、計算の実行、結果の取得が完了しました。
