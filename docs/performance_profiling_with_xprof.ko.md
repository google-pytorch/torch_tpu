# XProf를 사용한 성능 프로파일링 { #performance-profiling-with-xprof }

TPU에서 대규모 모델을 학습하려면 시간이 어디에 소요되는지 파악하는 것이 매우
중요합니다. 이 가이드에서는 TPU 프로파일러로 실행 트레이스를 캡처하고
TensorBoard에서 분석하는 방법을 설명합니다.

> **Warning:** **기본 요건:** TensorBoard에는 `setuptools<81`이 필요합니다.
> 최신 버전에서는 TensorBoard가 의존하는 `pkg_resources`가 제거되었습니다. 먼저
> 다음을 설치하세요.
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **PyTorch 버전:** 네이티브 `torch.profiler` TPU 통합을 사용하려면
> PyTorch 2.12 이상이 필요합니다.

## 1. TPU 프로파일러 { #1-the-tpu-profiler }

표준 `torch.profiler`(`torch_tpu`와 통합됨)는 CPU 및 TPU 활동을 모두 트레이싱하는
**컨텍스트 관리자** 인터페이스를 제공합니다. 프로파일링할 코드를
`torch.profiler.profile()` 블록으로 감싸면, 수집된 트레이스가 디스크에 저장되어
TensorBoard에서 볼 수 있습니다.

**주요 구성요소:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 구성요소                                  | 역할                            |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | 트레이스 수집을 시작/중지하는 <br> 표준 컨텍스트 관리자 |
| `torch.profiler.ProfilerActivity.CPU`      | CPU 측 연산(디스패치, 데이터 <br> 로드)을 트레이싱 |
| `torch.profiler.ProfilerActivity.TPU`      | TPU 하드웨어 실행(계산, <br> 메모리, 인터커넥트)을 트레이싱 |
| `torch.profiler.tensorboard_trace_handler` | TensorBoard용 디렉터리에 트레이스를 <br> 저장하는 콜백       |
| `TpuProfilerConfig`                        | 커스텀 TPU 프로파일링 구성을 위한 도우미                   |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. 커스텀 프로파일링 구성(`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

`TpuProfilerConfig`를 통해 전달하는 옵션으로 호스트 및 디바이스 트레이서 수준과
출력 디렉터리 구조를 세밀하게 제어할 수 있습니다.

### 지원되는 구성 옵션 { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 옵션 키            | 타입           | 기본값 | 설명               |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | CPU 호스트 트레이서 상세 수준을 <br> 제어합니다(0=사용 안 함, 1=중요, 2=전체). |
| `device_tracer_level` | `int`          | `1`     | TPU 하드웨어 디바이스 트레이서 <br> 상세 수준을 제어합니다(0=사용 안 함, 1=사용). |
| `python_tracer_level` | `int`          | `0`     | Python 함수 트레이싱 <br> 수준을 제어합니다(0=사용 안 함, 1=사용). |
| `run_dir`             | `pathlib.Path` | `None`  | TPU 하드웨어 트레이스 <br> 파일(`.xplane.pb`)이 저장되는 출력 디렉터리입니다. |
| `worker_rank`         | `str`          | `None`  | XPlane 파일 이름(`<hostname>_<rank>.xplane.pb`)에 <br> 접미사로 추가되는 워커 랭크입니다. |
<!-- mdformat on -->

```python
import pathlib
import torch
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
    python_tracer_level=0,
    run_dir=pathlib.Path("./profiler_output"),
    worker_rank="0",
)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
) as prof:
    model(inputs)
```

## 3. 프로필 캡처 { #3-capturing-a-profile }

아래 셀은 프로파일러 컨텍스트 안에서 작은 학습 루프를 실행합니다. 핵심 사항은
다음과 같습니다.

1.  학습 루프를 `torch.profiler.profile()` 안에 넣습니다.
2.  활동을 지정합니다. 호스트 측 연산에는 `CPU`, 하드웨어 실행에는 `TPU`를
    지정합니다.
3.  `torch.profiler.tensorboard_trace_handler`를 사용하여 트레이스를 디렉터리에
    저장하도록 `on_trace_ready`를 설정합니다.
4.  (선택사항) 트레이서 수준을 제어하려면 `TpuProfilerConfig`와 함께
    `experimental_config`를 전달합니다.

> **Important:** 프로파일러는 오버헤드를 추가합니다. 전체 학습 실행이 아니라
> 대표적인 일부 단계(예: 100~1000단계)만 프로파일링하세요.

```python
# Build a small model and optimizer
model = (
    torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )
    .to(device)
    .to(torch.bfloat16)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

# Where to save the profiler traces
log_dir = "/tmp/profiler_output"

# Configure TPU-specific tracer levels
config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
)

# Profile 100 training steps with native PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
) as prof:
  for step in range(100):
    optimizer.zero_grad()
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    torch.accelerator.synchronize()
    prof.step()

print(f"✅ Profile captured: {100} steps saved to {log_dir}")
```

## 4. TensorBoard에서 트레이스 보기 { #4-viewing-traces-in-tensorboard }

다음 명령어를 실행하세요.

**1단계:** 프로파일러 플러그인을 설치합니다(대시보드가 표시되려면 필요함).

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**2단계:** TensorBoard를 실행합니다.

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

XProf는 독립형 서버로도 실행할 수 있습니다.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**3단계:** 브라우저에서 `http://localhost:6006`을 열고 **"Profile"** 탭을
선택합니다.

> **Note:** 포트 전달이 자동으로 작동하도록 VS Code 통합 터미널에서
> TensorBoard를 실행해야 합니다.

**GKE에서:** 트레이스가 포드보다 오래 유지되도록
[Cloud Storage FUSE CSI 드라이버](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)로
마운트된 Cloud Storage 버킷에 트레이스 디렉터리를 쓴 다음, `xprof
--logdir=gs://<bucket>/profiler_output`로 트레이스를 여세요.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 뷰               | 표시 내용                                          |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | 대략적인 요약: 단계 시간, 디바이스 사용률, 상위 연산 |
| **Trace Viewer**   | CPU와 TPU의 모든 연산 타임라인. 간격(유휴 하드웨어)을 찾아보세요. |
| **Op Profile**     | 연산별 시간 분석. 가장 비용이 큰 연산을 찾으세요. |
| **Memory Profile** | 시간에 따른 HBM 사용량. 메모리 급증을 감지하세요. |
<!-- mdformat on -->

## 5. 살펴볼 사항 { #5-what-to-look-for }

### 트레이스의 위험 신호 { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 패턴                 | 의미                 | 해결 방법                      |
| ----------------------- | ----------------------- | ------------------------ |
| TPU 연산 사이의 큰 간격 | 호스트 측 병목 <br>(데이터 로드, Python 오버헤드) | 데이터를 프리페치하고, 루프 내 Python 로직을 줄이세요 |
| 짧은 TPU 버스트가 잦음   | 그래프 중단이 너무 많음 → 잦은 재컴파일 | `.item()` / `.cpu()` 호출을 줄이고, `torch.compile`을 사용하세요 |
| 낮은 MXU 사용률 | 텐서 차원이 타일 크기(128)에 맞춰 정렬되지 않음 | 차원을 128의 배수로 패딩하세요 |
<!-- mdformat on -->

### 이상적인 프로필 { #the-ideal-profile }

잘 최적화된 TPU 워크로드는 다음과 같은 특징을 보입니다.

-   **길고 연속적인 TPU 계산 블록**: 컴파일러가 많은 연산을 하나의 실행으로
    퓨전했습니다.
-   **최소한의 CPU→TPU 간격**: 데이터 로드가 계산과 오버랩됩니다.
-   **높은 MXU 사용률(>60%)**: 텐서 형상이 하드웨어에 맞춰 정렬되어 있습니다.

## 6. 트레이스 캡처 확인 { #6-verifying-the-traces-were-captured }

아래 셀은 프로파일러 출력 디렉터리에 트레이스 파일이 있는지 확인합니다.

```python
import os

trace_files = os.listdir(log_dir)
print(f"Trace directory: {log_dir}")
print(f"Files found: {len(trace_files)}")
for f in sorted(trace_files)[:10]:
  size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
  print(f"  {f} ({size_kb:.1f} KB)")

if trace_files:
  print(f"\n✅ Traces ready. View with:")
  print(f"   tensorboard --logdir={log_dir}")
else:
  print("\n❌ No trace files found. Check profiler configuration.")
```

## 7. 커널 프로파일링(세분화된 성능 카운터) { #7-kernel-profiling-fine-grained-performance-counters }

커널 프로파일링을 사용하면 세분화된 하드웨어 성능 카운터(예: Pallas 커널용)를
수집하여 표준 이벤트 트레이스에서는 보이지 않는 성능 병목을 진단할 수 있습니다.

> **Note:** 세분화된 성능 카운터 샘플링은 **Ironwood(TPU v7/TPU7x)** 및 이후 TPU
> 세대에서 사용할 수 있는 고급 하드웨어 기능입니다.

### 성능 카운터 사용 설정 { #enabling-performance-counters }

`TpuProfilerConfig`의 `experimental_options`를 통해 카운터 구성을 전달하여 성능
카운터를 사용 설정할 수 있습니다.

```python
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    experimental_options={
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
            " indices:11 indices:56 indices:57 indices:58"
        ),
    }
)
```

구성 파라미터(예: 샘플링 간격, 스케일링, 트리거 모드)의 자세한 목록과 카운터
인덱스를 확인하는 방법은 공식
[OpenXLA 커널 프로파일링 가이드](https://openxla.org/xprof/kernel-profiling)를
참조하세요.

## 8. 빠른 참조 { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
단계        | 명령어 / 코드
----------- | --------------
**가져오기**  | `from torch.tpu.profiler import TpuProfilerConfig`
**프로파일링** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**저장**    | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**보기**    | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** 트레이스 파일을 관리 가능한 크기로 유지하려면 100~1000단계만
> 프로파일링하세요. 프로파일러는 단계당 약 5~10%의 오버헤드를 추가합니다.
