# TorchTPU FAQ { #torchtpu-faq }

## PyTorch/XLA와 TorchTPU의 차이점은 무엇인가요? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

PyTorch/XLA는 그동안 JAX 런타임 구성요소와 전용 API에 크게 의존하는 브리지
프레임워크 역할을 해 왔습니다. TorchTPU는
[Google Cloud TPU](https://cloud.google.com/tpu)용 네이티브 PyTorch 백엔드를
제공하며 JAX 런타임 종속성이 전혀 없습니다. 기존 PyTorch 워크로드는 사소한 코드
변경(예: `.to("cuda")` → `.to("tpu")`)만으로 바로 실행됩니다.

## TorchAX와 TorchTPU의 차이점은 무엇인가요? { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX(torch_xla2라고도 함)는 Python 수준에서 PyTorch 연산을 가로채 XLA로(주로
JAX를 통해) 로어링하는 구현입니다. TorchAX는 추론 벤치마킹과 프로토타이핑에
많이 사용되어 왔지만, TPU에서 기존 코드를 그대로 쓸 수 있는(drop-in)
PyTorch 네이티브 환경을 위한 새로운 공식 표준은 TorchTPU입니다.

## PyTorch/XLA는 어떻게 되나요? { #what-is-happening-to-pytorchxla }

PyTorch/XLA는 지원 중단되었으며, 현재 공개된
[TorchTPU](https://github.com/google-pytorch/torch_tpu)로 대체되었습니다.

## TorchTPU가 JAX를 대체하나요? { #is-torchtpu-replacing-jax }

아니요. TorchTPU는 Cloud TPU에 대한 네이티브 PyTorch 진입점을 제공하여 TPU
생태계를 보완하도록 설계되었습니다. 기존 PyTorch 코드베이스를 가진 개발자는 코드를
다시 작성하지 않거나 최소한으로만 수정하여 TPU에서 직접 실행할 수 있습니다.
따라서 모델을 JAX로 다시 작성하거나 GPU와 TPU 인프라용 코드베이스를 따로 관리할
필요가 없습니다.

*   **TorchTPU는 언제 고려해야 하나요?** 기존 PyTorch 코드베이스가 있거나, 리팩터링
    없이 여러 하드웨어(GPU/TPU) 간 이식성을 원하거나, PyTorch 네이티브
    프레임워크(예: TorchTitan, vLLM, SGLang, Hugging Face)를 사용하거나, GPU에서
    TPU로 빠르게 마이그레이션하려는 경우입니다.

## TorchTPU 개발자와 소통하는 가장 좋은 방법은 무엇인가요? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com)으로
이메일을 보내 의견을 공유할 수 있습니다. 저장소에 직접 기여하는 것도 환영합니다:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
[discuss.pytorch.org](https://discuss.pytorch.org/)에서 "TPU" 태그를 사용할 수도
있습니다.

## 커뮤니티 참여 포럼이 있나요? { #will-there-be-any-community-engagement-forums }

예. 커뮤니티 참여는 저희에게 매우 중요하며, 이러한 포럼에 대한 세부정보를 공유할
예정입니다. 또한 매월 오피스 아워를 운영할 계획입니다.

## TorchTPU는 커널 개발을 어떻게 지원하나요? { #what-support-does-torchtpu-provide-for-kernel-development }

[TorchTPU 커스텀 커널 지원](./custom_kernels.md)을 참고하세요.

## 현재 지원되는 TPU 세대는 무엇인가요? 향후 하드웨어 지원 계획은 무엇인가요? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU는 v5e, v5p, Trillium(TPU v6e), Ironwood(TPU v7x)를 지원합니다. TPU v8과
향후 칩에 대한 검증과 최적화는 로드맵의 일부로 진행 중입니다.

## TorchTPU의 지원 기능과 향후 로드맵은 어디에서 확인할 수 있나요? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

현재 기능 세트와 예정된 기능 세트는 [지원 기능](./feature_set.md)을 참고하세요.

## 온보딩을 쉽게 해 주는 에이전트가 있나요? { #is-there-an-agent-to-make-my-onboarding-easier }

TPU Agent Suite(실험용)를 확인해 보세요.
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

TPU Agent Suite는 Cloud TPU v6e 및 v7x에서 PyTorch 모델의 온보딩, 진단, 추론
성능 튜닝을 자동화하는 전문 하위 에이전트를 제공하는 오픈소스 플러그인입니다.

## 자체 서빙 솔루션에서 TorchTPU를 사용할 수 있나요? { #can-i-use-torchtpu-with-our-custom-serving-solution }

예. TorchTPU는 vLLM, SGLang 같은 업계 표준 서빙 엔진과 바로 통합되며, 커스텀
또는 사내 서빙 솔루션도 네이티브 PyTorch 실행 API와 하위 수준 컴파일러 백엔드를
통해 TorchTPU와 통합할 수 있습니다. 고급 오케스트레이션이 필요하면 llm-d를
사용하여 GKE에 커스텀 서빙 구성을 배포할 수도 있습니다.
