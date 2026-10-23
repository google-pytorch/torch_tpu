<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>토론 포럼</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>TPU에서 PyTorch 사용</b></a>
</p>

## 정보 { #about }

TorchTPU는 Google과 Meta가 협력하여 개발한 Google Cloud TPU용 오픈소스 네이티브
PyTorch 백엔드입니다. TorchTPU를 사용하면 기존 PyTorch 모델을 최소한의 코드
변경으로 TPU 하드웨어에서 실행하고, 학습 및 추론 워크로드를 단일 칩에서 멀티슬라이스
TPU Pod까지 확장할 수 있습니다.

## 오늘 무엇을 하려고 하시나요? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> 시작하기**

    환경을 프로비저닝하고, 기본 아키텍처를 이해하고, 첫 워크로드를 실행합니다.

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 소개](./intro_to_torchtpu.md): TorchTPU가 만들어진 이유, 제공하는 기능, 그리고 TorchTPU가 어떻게 도움이 되는지 설명합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> VM에서 TorchTPU 시작하기](./get_started_on_a_vm.md): TPU VM에 TorchTPU를 설정합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> GKE에서 TorchTPU 시작하기](./get_started_on_gke.md): GKE를 통해 TorchTPU를 사용합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> 지원 기능 및 로드맵](./feature_set.md): 현재 지원되는 기능과 향후 릴리스 계획입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> API 참조](./api_reference.md): 현재 릴리스의 TorchTPU API와 환경 변수입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU FAQ](./faq.md): TorchTPU에 관해 자주 묻는 질문입니다.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> 아키텍처 심층 분석 및 핵심 개념**

    메모리 레이아웃, 실행 엔진, 분산 실행의 작동 방식을 자세히 살펴봅니다.

    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 작동 방식](./how_torchtpu_works.md): TorchTPU가 위에서 아래까지 PyTorch 및 TPU 하드웨어와 어떻게 통합되는지 설명합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 실행 모델](./torchtpu_execution_model.md): TorchTPU가 PyTorch 코드를 실행하는 방식과 지원하는 여러 실행 모드를 자세히 설명합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> TorchTPU 메모리 모델](./torchtpu_memory_model.md): TPU와 XLA 컴파일러가 메모리를 다루는 방식을 TorchTPU가 어떻게 처리하는지 설명합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> 분산 지원](./distributed_support.md): TorchTPU가 분산 워크로드를 지원하는 방식입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> 커스텀 커널 지원](./custom_kernels.md): TorchTPU용 커스텀 커널을 작성하는 방법입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> 동적 형상](./dynamic_shapes.md): 동적 텐서 형상 지원입니다.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> 사례 연구**

    TorchTPU를 사용하여 실제 문제를 해결하는 실용적인 예를 살펴봅니다.

    - [<!--mkdocs::octicons-arrow-right-24:--> FLUX 최적화](./case_studies_flux.md): FLUX에서 여러 최적화를 식별하고 적용하는 방법을 보여 주는 예입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B 추론](./case_studies_muse_glimmer_30b_inference.md): 정적/동적 KV 캐시와 실제 가중치를 사용한 반복 디코딩 예입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> Muse Glimmer 30B 학습](./case_studies_muse_glimmer_30b_training.md): Muse Glimmer 30B의 독립형 구현에서 실행하는 학습 루프입니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> FSDP 사용하기](./case_studies_fsdp.md): TorchTPU에서 FSDP를 사용하는 권장 방법입니다.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> 워크로드 분석**

    - [<!--mkdocs::octicons-arrow-right-24:--> 수치 정밀도 및 일치성](./numerical_precision.md): TorchTPU에서 정밀도를 제어합니다.
    - [<!--mkdocs::octicons-arrow-right-24:--> XProf로 성능 프로파일링](./performance_profiling_with_xprof.md): XProf를 TorchTPU 프로파일러로 사용하는 방법입니다.

<!--mkdocs:</div>-->
<!-- mdformat on -->
