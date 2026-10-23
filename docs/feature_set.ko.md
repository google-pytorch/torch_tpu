# TorchTPU 지원 기능 및 향후 작업 { #torchtpu-supported-feature-set-and-future-work }

## 현재 지원 기능 { #current-supported-feature-set }

아래 표는 최초 오픈소스 릴리스 시점의 기능 지원 범위를 보여 줍니다.

워크로드에 TorchTPU가 적합한지 평가할 수 있도록 기능을 두 단계로 분류합니다.

*   **검증됨:** 표준 모델 아키텍처 전반에서 엔드 투 엔드 지속 테스트, 자동화된
    CI/CD 적용, 벤치마크 검증을 거친 기능입니다.
*   **실험적:** 초기 단계의 구현, 새로운 API 또는 프로토타입 기능입니다. 작동하며
    초기 피드백을 위해 제공되지만, 커뮤니티 의견에 따라 동작이 개선되거나 API가
    변경될 수 있습니다.

> **Note:** TorchTPU는 공개적으로 빠르게 발전하고 있습니다. 최신 상태를
> 확인하거나 특정 기능을 요청하려면
> [GitHub](https://github.com/google-pytorch/torch_tpu)를 방문하세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **검증됨** | **실험적** |
| ----- | ----- | ----- |
| **PyTorch 커뮤니티 참여** | <ul><li>TorchTPU, vLLM, SGLang, TorchTitan, TorchRec의 OSS 릴리스</li><li>외부 PR 수락</li><li>문서 / 튜토리얼</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>ATEN 마이그레이션용 TPU 에이전트</li></ul> |
| **TPU 지원** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **기본 기능** | <ul><li>손쉬운 마이그레이션</li><li>eager \+ 컴파일 혼합</li><li>순전파 \+ 역전파</li><li>Torch.profile</li><li>동적 연산(재컴파일)</li></ul> | <ul><li>SymInts</li><li>Jagged 텐서</li></ul> |
| **모델 아키텍처** | <ul><li>Transformer(Dense \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Diffusion: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS 외 다수(점진적 개선)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **커널 지원** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **양자화** | <ul><li>전체 텐서 FP8/FP4</li><li>채널별 FP8</li></ul> |  |
| **분산** | <ul><li>최적 메시</li><li>SimpleFSDP</li><li>체크포인팅</li><li>활성화 오프로딩</li><li>최대 8K 칩 멀티슬라이스</li></ul> | <ul><li>FSDPv2(낮은 성능)</li><li>MPMD(낮은 성능)</li><li>TorchComm</li></ul> |
| **생태계 통합** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **프로덕션용 솔루션** | <ul><li>추론(멀티 호스트, P/D 분리)</li><li>미세 조정(최대 \~8K 칩)</li></ul> |  |
<!-- mdformat on -->

## 향후 작업 { #future-work }

이 섹션은 향후 개발 방향과 탐색 영역을 설명하며, 확정된 릴리스 일정이
아닙니다.

커뮤니티 중심의 오픈소스 프로젝트로서 개발 우선순위는 커뮤니티 피드백과 개발자
채택에 의도적으로 맞춰 조정됩니다. 기능, 일정, 아키텍처 접근 방식은 어떤
워크로드가 가장 중요한지 파악해 가면서 바뀔 수 있습니다.

향후 작업은 두 가지 큰 방향으로 구성됩니다.

*   **단기 중점 사항:** 핵심 팀이 현재 우선적으로 진행하는 적극적인 설계,
    프로토타입 제작, 성능 강화 영역입니다.
*   **향후 탐색:** 장기적인 아키텍처 탐색, 잠재적인 생태계 통합, 그리고
    커뮤니티가 요청하여 적극적으로 조사 중인 백로그 항목입니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **단기 중점 사항** | **향후 탐색** |
| ----- | ----- | ----- |
| **PyTorch 커뮤니티 참여** | <ul><li>더 많은 문서 / 튜토리얼 / 블로그</li></ul> | <ul><li>커널 마이그레이션용 TPU 에이전트</li></ul> |
| **TPU 지원** | <ul><li>v8i, v8t</li></ul> |  |
| **기본 기능** | <ul><li>스트림</li><li>그래프 캡처</li><li>빠른 시작 시간의 eager 모드</li><li>동적 연산(재컴파일 없음)</li></ul> |  |
| **모델 아키텍처** | <ul><li>출시 당일 모델 지원</li></ul> |  |
| **양자화** | <ul><li>torchao를 사용한 블록 단위 FP8/FP4(TPUv8t에서 최고 성능)</li></ul> |  |
| **분산** | <ul><li>FSDPv2(성능 개선)</li><li>MPMD(성능 개선)</li><li>8K 칩 초과 규모의 학습</li><li>다중 계층 체크포인팅</li><li>Goodput</li></ul> | <ul><li>8K 칩을 훨씬 넘는 규모의 학습</li></ul>  |
| **생태계 통합** | <ul><li>VeRL</li></ul> |  |
| **프로덕션용 솔루션** | <ul><li>RL</li></ul> | <ul><li>사전 학습</li></ul> |
<!-- mdformat on -->
