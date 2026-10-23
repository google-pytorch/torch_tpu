# TorchTPU 작동 방식 { #how-torchtpu-works }

## TPU 기초 { #tpu-primer }

[Google TPU](https://cloud.google.com/tpu)는 딥러닝의 기반이 되는 대규모 텐서
연산을 수행하도록 특별히 설계된 전용 ASIC입니다. TPU는 세대마다
[아키텍처가 다르지만](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm),
넓게 정의하면 TPU 칩에는 다음이 포함됩니다.

*   하나 이상의 TensorCore. 각 TensorCore는 행렬 곱셈 유닛(MXU), 벡터
    유닛(VPU), 스칼라 유닛으로 구성된 단일 스레드 프로세서입니다.
*   하나 이상의 SparseCore. 희소 연산에 특화되어 있으며, TensorCore가 연산을
    오프로드할 수 있습니다.
*   대용량 HBM(High Bandwidth Memory).
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)의
    경우 288GB, 8.6TB/s이며, ML 데이터를 저장하고 CPU와 공유하는 데 사용할 수
    있습니다.
*   양방향 ICI(Inter-Chip Interconnect).
    [TPU 8t 및 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)의
    경우 19.2TBps이며, 사용자가 구성한
    [TPU 슬라이스](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    내의 TPU 칩 간에 빠른 통신을 가능하게 합니다. TPU 슬라이스는
    [TPU Pod](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod)의
    논리적 하위 그룹으로, 3D 토러스(TPU 7, TPU 8t) 또는 Boardfly(TPU 8i)
    구성으로 연결된 사전 구성된 수의 TPU 칩으로 이루어집니다.
*   양방향 DCN(Data-Center-Network).
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/)의
    경우 400GBps이며, 슬라이스 간 TPU 칩 통신을 가능하게 합니다.

TPU는 [XLA 컴파일러](https://github.com/openxla/xla) 스택을 통해 두 가지 진입점,
즉 [StableHLO](https://openxla.org/stablehlo/spec)와
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu)로
프로그래밍됩니다. 실제 TPU ISA는 독점 사양이며, 변경될 수 있고, 공개되지
않습니다. 실질적으로 StableHLO와 (특히) Mosaic는 PTX에 해당한다고 생각할 수
있습니다. 대략적인 개요는 아래 그림을 참조하세요.

![TorchTPU 작동 방식](./images/torch_tpu_inputs_to_tpu.svg)

C++ 프로그램은 [PjRT 클라이언트](https://openxla.org/xla/pjrt/cpp_api_overview)를
인스턴스화하여 로컬 TPU에 액세스할 수 있습니다. PjRT 클라이언트 API는 MLIR
(StableHLO 또는 Mosaic)을 TPU 실행 파일로 컴파일하고, 그 실행 파일을 실행하고,
TPU와 데이터를 주고받는 메서드를 제공합니다. TPU 실행은 비동기적이며, CPU는
연결된 PjRTBuffer를 대기하여 TPU 결과가 준비될 때까지 기다릴 수 있습니다.
StableHLO는 [SSA 패러다임](https://en.wikipedia.org/wiki/Static_single-assignment_form)을
따르므로 인플레이스 변경을 허용하지 않습니다. 즉, 쓰기 연산은 항상 (적어도
개념적으로는) 새 값을 생성합니다. 이는 뷰 분해 및 역변환을 통해 변경을 포함한
익숙한 PyTorch "뷰" 시맨틱스를 제공하는 TorchTPU에 특히 중요합니다. XLA
컴파일러는 텐서 형상을 컴파일 시간에 알 수 있다고 가정합니다. TorchTPU에서는
[제한된 동적성](./dynamic_shapes.md)으로 이 제약을 완화합니다. 이것이 PyTorch
워크로드의 제한사항임을 인식하고, 제한된 동적성 지원을 추가해 왔습니다. 제한된
동적성은 형상 차원의 최대 경계는 컴파일 시간에 알 수 있지만 실제 차원은
런타임에만 알 수 있는 경우를 말합니다.

## TorchTPU: 자동 로드를 지원하는 out-of-tree 백엔드 { #torchtpu-an-out-of-tree-backend-with-autoload-support }

추가 백엔드를 out-of-tree로 구성하고 런타임에 등록하는
[현재 권장사항](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html)에
따라, `PrivateUse1`로 등록되는 새 C++ 백엔드로 PyTorch 디스패처를 확장합니다.
또한 표준 `at::Tensor` 메커니즘을 사용하여 텐서 스토리지의 디바이스별 구현을
정의했습니다(`c10::StorageImpl` 내부의 `c10::UniqueVoidPtr` 사용).

TorchTPU는 사용성을 염두에 두고 설계되었습니다. TorchTPU wheel이 설치된 TPU
머신에서 사용자가 PyTorch를 가져오면 Python이 TorchTPU wheel을 자동으로
로드합니다. wheel을 자동 로드하면 백엔드가 설정되고 `PrivateUse1` 이름이 `tpu`로
대체됩니다. 덕분에 간단한 예에서는 TorchTPU 프로그램이 in-tree 백엔드와 다르지
않으며, PyTorch 프로그램을 거의 또는 전혀 변경하지 않고 TPU에서 실행할 수
있습니다.

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

TPU가 현재 가속기가 되고 `tpu_dist`가 TorchTPU의 기본 백엔드이므로, 이를
명시적으로 지정할 필요가 없습니다.

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

## 그래프 캡처 { #capturing-the-graph }

TorchTPU는 백엔드 인프라의 일부인 PyTorch 연산 디스패치 훅을 활용하여 ATen
수준에서 eager 연산을 트레이싱합니다. TorchTPU 연산은 StableHLO 프로그램으로
구성되어 XLA에 하나씩 또는 그룹으로 전달되므로, 컴파일러가 연산을 상당 부분
퓨전하여 하드웨어를 더 효과적으로 활용할 수 있습니다. TorchTPU가 PyTorch
디스패처로부터 콜백 호출을 받으면 런타임은 다음을 수행합니다.

*   다음을 포함하는 Deferred Op 인스턴스를 만듭니다.
    *   StableHLO 빌더 객체. 호출되면 아래와 같이 `aten::mul`의 시맨틱스를
        구현하는 StableHLO 함수를 반환합니다(암시적 브로드캐스팅과 dtype
        승격은 다른 곳에서 처리됨).

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   캐시 키 계산에 사용할 정적 연산 파라미터(예: `aten::native_batch_norm`의
    training 및 momentum)의 선택적 맵
*   (간접 참조를 통해) 해당 Deferred Op 인스턴스를 가리키는
    `c10::StorageImpl`을 가진 새 `at::Tensor`를 반환합니다.

실행 모드에 따라 적절한 동기화 지점에 도달하면 TorchTPU는 다음을 수행합니다.

*   연산 영역을 분석합니다.
*   실행 가능한 코드로 구체화할 연산을 결정합니다.
*   그래프의 컴파일 캐시 키를 만듭니다.
*   캐시에 이미 있으면 실행 파일을 가져와 TPU에 디스패치합니다.
*   캐시에 없으면 객체를 비동기적으로 컴파일하고, 캐시에 추가한 다음, 실행을
    위해 디스패치합니다.
*   도달한 동기화 지점에 적합한 경우, 값이 준비될 때까지 기다렸다가 다시
    복사할 수 있습니다.

다양한 실행 모드에 관한 자세한 내용은
[TorchTPU 실행 모델](./torchtpu_execution_model.md)에서 설명합니다.

## TorchTPU 컴파일 캐시 { #the-torchtpu-compilation-cache }

불필요한 컴파일을 피하기 위해 TorchTPU는 다중 계층 컴파일 캐시를 사용합니다.

*   **계층 1:** 같은 프로세스 내에서 컴파일을 공유하기 위한 프로세스 로컬
    인메모리 캐시입니다. 세 계층 중 가장 빠르며, 항상 사용 설정되어 있습니다.
*   **계층 2:** 같은 호스트에서 프로세스 간 공유를 가능하게 하는 머신 로컬 공유
    메모리입니다. 멀티 디바이스 학습/추론에 유용합니다. 계층 2는 분산
    프로그램에서 기본적으로 사용 설정되지만, 단일 디바이스 프로그램에서는
    이점이 없으므로 사용 중지됩니다.
*   **계층 3:** 원격 파일 시스템(예: Google Cloud Storage)을 기반으로 하는 영구
    캐시로, 호스트, 프로그램, 작업 재시작 간에 재사용할 수 있습니다. Colab이나
    데이터 센터와 같은 실행 환경에서 TPU 프로그램을 실행할 때 유용합니다.
    기본적으로 사용 중지되어 있지만 저장 위치를 지정하여 사용 설정할 수
    있습니다.

각 항목은 ATen 연산 하나 또는 ATen 연산 시퀀스에서 생성된 StableHLO 계산
그래프를 나타내는 컴파일된 TPU 실행 파일입니다. StableHLO 그래프는 그래프 구조,
입력/출력의 텐서 형상, 그리고 TPU 하드웨어, libtpu, PjRt 런타임 버전을 포함한
컴파일러 옵션을 결합한 지문으로 표현됩니다.

## `torch.compile` 지원 { #torchcompile-support }

`torch.compile()`을 지원하는 다른 하드웨어와 일관된 사용자 경험을 유지하기 위해
새 TorchDynamo
[백엔드](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)를
등록합니다. 이 백엔드는 다음과 같이 사용할 수 있습니다.
`compiled_model = torch.compile(model,
backend="tpu")`

Dynamo는 `torch.fx.GraphModule`과 예시 입력 텐서 목록을 전달하여 TorchTPU
백엔드를 호출합니다. 이 시점에 런타임은 제공된 그래프를 eager 모드인 것처럼
트레이싱하지만, 모든 연산은 지연되며 TPU에서 실행되지 않습니다. 런타임은 전체 FX
그래프에 대해 단일 StableHLO 프로그램을 생성하고 컴파일한 후, 호출을 래핑하는
Python 호출 가능 객체를 반환합니다. TorchTPU는 `TorchInductor`를 사용하지 않고,
대신 TPU 아키텍처와 함께 발전해 온 성숙한 XLA 컴파일러 스택을 재사용합니다.

## 자세히 알아보기 { #more-detail }

다음 문서에서는 다음 내용을 자세히 다룹니다.

*   TPU가 지원하는 [메모리 모델](./torchtpu_memory_model.md)과 이것이 PyTorch에
    미치는 영향
*   TorchTPU가 지원하는 [실행 모드](./torchtpu_execution_model.md)
*   [커스텀 커널](./custom_kernels.md) 지원
*   TPU 클러스터에서의 [분산 실행](./distributed_support.md)
