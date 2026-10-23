# TorchTPU 소개 { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU는 Google과 Meta의 PyTorch 팀이 협력하여 개발한
[Google Cloud TPU](https://cloud.google.com/tpu)용 오픈소스 네이티브 PyTorch
백엔드입니다. TorchTPU를 사용하면 개발자는 기존 PyTorch 워크로드를 최소한의 코드
변경으로 TPU 하드웨어에서 실행하고, 학습 및 추론 워크로드를 단일 칩에서 멀티슬라이스
TPU Pod까지 확장할 수 있습니다.

## TorchTPU를 사용하는 이유 { #why-use-torchtpu }

Google TPU는 딥러닝을 구동하는 행렬 및 벡터 연산을 위해 특별히 설계된 주문형
반도체(ASIC)로, 대규모 환경에서 높은 처리량, 에너지 효율, 비용 대비 성능을
제공합니다. TPU는 지금까지 주로 JAX 및 TensorFlow 생태계와 함께 사용되어 왔지만,
TorchTPU는 다음 세 가지 원칙을 중심으로 의도적으로 설계되어
[PyTorch 생태계](https://github.com/google-pytorch/torch_tpu)에 네이티브 TPU
지원을 직접 제공합니다.

*   **PyTorch 생태계의 네이티브 구성원:** TorchTPU는 JAX 런타임 종속성 없이 PyTorch와 직접
    통합됩니다. 표준 PyTorch 텐서 뷰와 인플레이스 변경 시맨틱스를 유지하고,
    PyTorch의 디바이스당 하나의 프로세스 실행 모델을 채택하며, Dynamo를 통해
    `torch.compile`을 지원합니다.
*   **바로 쓸 수 있는 사용성:** 워크로드를 GPU에서 TPU로 이식하는 작업은
    `device="cuda"`를 `device="tpu"`로, `backend="nccl"`을
    `backend="tpu_dist"`로 바꾸는 것에서 시작합니다. 수정하지 않은 PyTorch 모델이
    eager 모드에서 바로 실행되며, 런타임 또는 컴파일러 문제를 진단할 때 명확한
    오류 메시지를 제공합니다.
*   **프로덕션 규모의 성능:** TorchTPU는 연산별 eager 실행 외에도 지연 퓨전 eager
    모드, Dynamo-XLA 컴파일러 파이프라인, 형상 때문에 발생하는 재컴파일을 피하기 위한
    제한된 동적성(bounded dynamism), 커스텀 커널 API를 제공하여 하드웨어
    사용률을 최대로 끌어올릴 수 있도록 돕습니다.

## 빠른 시작: TPU에서 실행하기 { #quickstart-running-on-tpu }

다음 셀에서 TPU에서 실행되는 간단한 PyTorch 프로그램 예를 볼 수 있습니다. TPU
런타임을 사용 설정하면
[Colab에서 이 예를 실행](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)할
수 있습니다.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## 주요 기능 { #key-features }

*   **폭넓은 연산자 및 모델 지원:** 450개의 핵심 ATen 연산자(순전파 및 역전파
    모두)를 지원하며, LLM(Llama, Qwen, Gemma, DeepSeek, Mistral), 비전
    모델(ViT, ResNet, ConvNeXt, DINOv2), 오디오 모델(Whisper, Wav2Vec2),
    diffusion 파이프라인(FLUX, Stable Diffusion)을 아우르는 5,000개의 인기
    HuggingFace 모델에서 지속적으로 검증됩니다.
*   **세 가지 eager 실행 모드:**
    *   **Strict(기본값):** ATen 연산자를 한 번에 하나씩 TPU에 비동기적으로
        디스패치하며, GPU의 표준 PyTorch eager 동작과 일치합니다.
    *   **Debug(`TORCH_TPU_LAUNCH_BLOCKING=1`):** 각 연산자를 CPU와 동기적으로
        실행하고 런타임 경계 검사를 사용 설정하여 디버깅을 단순화합니다.
    *   **DeferAndFuse(`TORCH_TPU_DEFER_AND_FUSE=1`):** 구체화 지점(예: `.cpu()`
        또는 `.item()`)까지 ATen 연산자 시퀀스를 지연시키고 최적화된 계산
        블록으로 퓨전하여 HBM(High Bandwidth Memory) 트래픽을 크게 줄이고 eager
        실행을 가속합니다.
*   **`torch.compile`을 사용한 그래프 컴파일:** `"tpu"` Dynamo
    백엔드(`torch.compile(model)`)를 등록하여 전체 PyTorch FX 그래프를
    [StableHLO](https://openxla.org/stablehlo/spec)로 로어링하고
    [OpenXLA 컴파일러](https://openxla.org/xla)로 컴파일합니다. 동적
    형상(`torch._dynamo.mark_dynamic`)을 적극적으로 지원하므로 컴파일된 그래프가
    재컴파일 없이 지정된 상한까지 다양한 배치 크기나 시퀀스 길이를 처리할 수
    있습니다.
*   **제한된 동적성:** 동적 상한을 기준으로 컴파일하여 가변 배치 크기와 시퀀스
    길이로 인한 재컴파일을 네이티브로 줄입니다.
*   **다중 계층 컴파일 캐시:** 세 가지 캐시 계층에 걸쳐 중복 XLA 컴파일을
    제거합니다.
    *   **계층 1:** 프로세스 로컬 인메모리 캐시(항상 사용 설정됨)
    *   **계층 2:** 멀티 디바이스 작업에서 프로세스 간 재사용을 위한 호스트 로컬
        공유 메모리
    *   **계층 3:** 호스트 간 및 작업 재시작 시 재사용을 위한 영구 원격
        캐시(예: Cloud Storage)
*   **네이티브 분산 프리미티브:** `backend="tpu_dist"`를 통해
    `torch.distributed` 프로세스 그룹(`c10d::Backend`) 수준에서 TPU 집합 통신을
    구현하며, `DistributedDataParallel`(DDP),
    `FullyShardedDataParallel`(FSDP2 및 SimpleFSDP), `DTensor`를 지원합니다.
    `torch.tpu.distributed.topology_aware_mesh`를 사용하여 병렬 처리 차원을
    물리적 TPU 인터커넥트에 맞출 수도 있습니다.
*   **커스텀 TPU 커널:**
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)로 커스텀 커널을
    작성하거나(칩 간 직접 RDMA 집합 통신을 지원하는 `torch.tpu.pallas.jax_op`로
    호출 가능), 실험적인 [Helion](https://pytorch.org/blog/helion/) DSL로 타일
    기반 커널을 작성하거나, Tokamax에서 최적화된 사전 빌드 TPU 커널을 가져올 수
    있습니다.
*   **생태계 통합:** `torch.profiler`(TensorBoard/XProf), vLLM, SGLang,
    TorchTitan, Ray, TorchRec을 포함한 표준 PyTorch 서빙 및 학습 프레임워크, 프로파일링
    도구 모음과 별도 설정 없이 함께 작동합니다.

## 제한사항 및 향후 작업 { #limitations-and-future-work }

다른 제품과 마찬가지로 TorchTPU에도 알려진 제한사항이 있으며, 시간이 지나면서
해결될 예정입니다. 최초 릴리스는 주요 워크로드를 별도 설정 없이 안정적으로
지원하지만, TorchTPU는 커뮤니티 중심의 협업 프로젝트입니다.
[지원 기능 및 로드맵](./feature_set.md)을 검토하여 앞으로 무엇이 계획되어 있는지
확인하거나, GitHub에서
[PR을 제출](https://github.com/google-pytorch/torch_tpu)하여 이러한 제한사항
해결에 직접 참여하세요. 이 프로젝트는 커뮤니티 프로젝트이며, 여러분의 의견을
듣고 싶습니다.
[GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
및 [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com)을 통해
함께 협업해 주세요.

*   동적 형상은 재컴파일을 유발합니다.
    *   TorchTPU에서 `torch.compile`은 기본적으로 `dynamic=False`를 설정합니다.
        `dynamic=None` 또는 `dynamic=True`는 아직 프로덕션용으로 지원되지
        않습니다. [동적 형상에 관한 논의 보기](./dynamic_shapes.md)
    *   예를 들어 Hugging Face LLM에서 디코딩/생성 단계를 실행하면 제한된
        동적성을 사용 설정하지 않는 한 단계마다 재컴파일이 발생합니다.
*   TPU에서는 뷰 연산의 비용이 클 수 있습니다
    ([TorchTPU 메모리 모델](./torchtpu_memory_model.md) 참조).
*   집합 통신을 사용하는 MPMD는 기능적으로 지원되지만 아직 최고 성능에
    도달하지는 못합니다. 집합 통신과 계산의 오버랩에는
    [몇 가지 제약](./distributed_support.md)이 있습니다.
    *   SparseCore 오프로드를 포함한 전체 스트림 지원이 가까운 시일 내에
        제공될 예정입니다.
    *   eager 또는 `torch.compile`과 함께 FSDPv2를 사용하는 대신
        `torch.compile()`과 함께 SimpleFSDP를 사용하세요.
        [FSDP 권장사항](./case_studies_fsdp.md)을 참조하세요.
    *   통신하는 모듈에서 XLA 최적화로 최고 성능을 얻는 권장 방법(즉,
        계산/통신 오버랩을 사용 설정하는 방법)은 정확히 동일하다고 **사용자가
        보장하는** 영역에 `spmd_safe` 데코레이터를 사용하는 것입니다.
*   TorchTPU는 현재 네이티브 FlexAttention을 지원하지 않습니다.
*   현재 세대 TPU는 fp32보다 낮은 정밀도(예: tf32)의 matmul 누적을 지원하지
    않습니다.
*   하나의 프로세스만 TPU 디바이스에 액세스할 수 있으므로, 같은 위치에 배치된
    RL에 대한 지원이 제한됩니다(TPU 칩 하나에는 하나 이상의 디바이스가 연결될
    수 있으며, 각 디바이스는 별도의 OS 프로세스에서 액세스할 수 있습니다).

**기타 권장사항:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 제한사항 | 권장사항 |
| :--- | :--- |
| eager 모드에도 워밍업 시간이 필요합니다 | 영구 컴파일 캐시를 사용 설정하세요 |
| Torch.compile에 시간이 오래 걸립니다 | 영구 컴파일 캐시를 사용 설정하세요 |
| PyTorch의 CNN은 기본적으로 채널 우선 레이아웃을 사용하며, 이는 TPU에서 매우 비효율적입니다 | 이 [PyTorch 옵션](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)을 사용하여 채널 마지막 레이아웃을 설정하면 성능이 향상됩니다. |
| TPU와 다른 아키텍처 간 토폴로지 차이로 인해 기본 디바이스 메시 매핑이 최적이 아닐 수 있습니다 | 최적의 디바이스 메시 매핑과 디바이스 메시 매핑 맞춤설정을 위해 `topology_aware_mesh`를 사용하세요 |
<!-- mdformat on -->

## 시작하기 { #getting-started }

이 문서 모음은 첫 설정부터 분산 프로덕션 배포까지 진행할 수 있도록 네 개의
섹션으로 구성되어 있습니다.

*   **시작하기:** 환경을 구성하고 [VM](./get_started_on_a_vm.md)에서 또는
    [GKE를 통해](./get_started_on_gke.md) 워크로드를 배포하는 단계별
    안내입니다.
*   **심층 분석:** [TPU 메모리 모델](./torchtpu_memory_model.md),
    [분산 실행](./distributed_support.md), 다양한
    [실행 모드](./torchtpu_execution_model.md)를 자세히 살펴봅니다.
*   **사례 연구:** [Flux 모델](./case_studies_flux.md) 최적화,
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md) 학습,
    [엔드 투 엔드 LLM 추론](./case_studies_muse_glimmer_30b_inference.md),
    [FSDP](./case_studies_fsdp.md)의 효과적인 적용을 다루는 엔드 투 엔드
    레시피입니다.
*   **워크로드 분석:**
    [프로파일링](./performance_profiling_with_xprof.md),
    [수치 정밀도](./numerical_precision.md) 관리
