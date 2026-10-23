# FSDP 권장사항 { #fsdp-best-practices }

FSDP(Fully Sharded Data Parallel)는 모델 파라미터, 그래디언트, 옵티마이저 상태를
데이터 병렬 랭크 전체에 샤딩하므로, 각 TPU 코어는 모델의 `1 / world_size`
슬라이스만 저장합니다. 레이어가 실행되기 전에 **all-gather**로 전체 가중치를
재구성하고, 역전파 중에는 **reduce-scatter**로 그래디언트를 결합하고 다시
샤딩합니다.

이 가이드에서는 TorchTPU에서 FSDP를 사용하는 권장 방법을 설명하고
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp)에
있는 실행 가능한 예제를 안내합니다.

이 가이드에서 학습할 내용은 다음과 같습니다.

1.  TorchTPU에서 **어떤 FSDP 접근 방식을 선택할지**
2.  **SimpleFSDP(권장)**: 단일 래핑 호출로 사용하는 컴파일 우선 FSDP
3.  **수동 FSDP**: 모델 정의에 집합 통신을 직접 넣는 방법
4.  모든 접근 방식에 적용되는 **권장사항**
5.  **예제 실행하기**: 실행 명령어와 각 예제가 확인하는 내용

## 1. FSDP 접근 방식 선택 { #1-choosing-an-fsdp-approach }

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
    | 접근 방식 | 사용 시기 | 예제 |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **권장 기본값입니다.** 모델 정의를 변경하지 않고 FSDP 또는 HSDP를 사용하려는 경우입니다.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **수동 FSDP**(수동 집합 통신) | 모델 코드를 직접 관리하며, 성능 조정을 위해 통신 코드를 계산 옆에 두려는 경우입니다. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | 아직 마이그레이션하지 않은 기존 코드입니다. 작동은 하지만 런타임 훅이 그래프 중단을 일으키며, 경쟁력 있는 성능을 내려면 TPU의 SparseCore로 스트림을 오프로드하는 향후 지원이 필요합니다.
.| [마이그레이션 가이드](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
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

## 2. 권장: SimpleFSDP { #2-recommended-simplefsdp }

[`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)는
`torch.compile`과 함께 작동하도록 설계된 FSDP 구현입니다. TorchTitan에서
도입되었으며, PyTorch에만 의존하는 독립형 사본이 `examples/fsdp/simple_fsdp.py`에
제공됩니다.

SimpleFSDP는 각 파라미터를 샤딩된 `DTensor`로 저장합니다. 파라미터에 액세스하면
파라미터화(parametrization)가 `DTensor.redistribute`를 호출하여 파라미터를 all-gather합니다.
`redistribute`의 autograd 역전파가 그래디언트 reduce-scatter를 수행합니다. 런타임
훅이 없으므로 전체 단계가 그래프 중단 없이 컴파일됩니다. 레이어별 래핑이나
프리페치 조정 없이 단일 `simple_fsdp.data_parallel(...)` 호출로 전체 모델을
샤딩하며, 2D `DeviceMesh`에서 `mode="hybrid_shard"`로 HSDP를 지원합니다.

SimpleFSDP 사용 방법은 다음을 참조하세요.

*   [SimpleFSDP README](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md):
    빠른 시작 및 사용 가능한 옵션
*   [FSDPv2에서 SimpleFSDP로 전환 튜토리얼](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    기존 `fully_shard` 코드를 마이그레이션하는 방법

## 3. 대안: 수동 FSDP { #3-alternative-manual-fsdp }

모델 정의를 제어할 수 있다면 수동 집합 통신
(`torch.distributed._functional_collectives`)과 `torch.compile`을 사용하여 FSDP를
직접 작성할 수 있습니다. 각 레이어는 자신의 가중치 샤드만 저장하고, 순전파에서
전체 가중치를 all-gather합니다. 함수형 집합 통신에는 autograd 공식이 있으므로
all-gather의 역전파는 reduce-scatter가 되며, 커스텀
`torch.autograd.Function`을 작성할 필요가 없습니다.

이렇게 하면 통신 코드를 계산 코드 옆에 둘 수 있어 성능 조정이 더 쉬워질 수
있습니다. 대신 모델 정의를 변경해야 합니다.

수동 FSDP를 구현하는 방법은
[수동 FSDP 가이드](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md)를
참조하세요.

## 4. 권장사항 { #4-best-practices }

이 권장사항은 SimpleFSDP와 수동 FSDP에 적용됩니다.

1.  **`torch.compile(backend="tpu")`로 전체 학습 단계를 컴파일하세요.** 전체
    단계를 컴파일할 수 없다면 최소한 순전파, 손실, 역전파를 함께 컴파일하세요.
    모델 모듈만 컴파일하면 단일 그래프가 만들어지지 않으므로, XLA가 집합 통신을
    계산과 오버랩하는 능력이 제한됩니다.
2.  **샤딩된 각 레이어에 활성화 체크포인팅을 사용하세요.** 각 레이어의 순전파를
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`로 래핑하세요.
    이렇게 하면 샤딩되지 않은 가중치를 역전파까지 유지하지 않고 각 레이어의
    순전파 후에 해제하므로, FSDP에서 기대하는 ZeRO-3 메모리 동작을 얻을 수
    있습니다.
3.  **샤딩되는 차원을 메시 크기로 나누어떨어지게 하세요.** 수동 FSDP에서는
    필수입니다. SimpleFSDP는 샤딩하기에 너무 작은 파라미터를 복제합니다.
    `shard_dim`을 사용하여 균등하게 나누어지는 차원을 선택하세요.
4.  **샤딩된 가중치를 초기화할 때 랭크마다 다른 시드를 사용하세요.** 모든
    랭크가 같은 시드로 자신의 샤드를 초기화하면 전체 가중치는 같은 무작위 블록이
    랭크 전체에 반복된 형태가 되어 수렴에 악영향을 줄 수 있습니다. 재현 가능하고
    서로 다른 샤드를 얻으려면 `torch.manual_seed(base_seed + dist.get_rank())`와
    같이 랭크에 따라 달라지는 값으로 각 랭크의 시드를 설정하세요.
5.  **모든 랭크가 같은 프로그램을 실행하도록 하세요.** 구체화를 강제하는
    연산(예: `.item()`)을 `if
    rank == 0`과 같은 랭크별 분기 안에 넣지 마세요.
6.  **전체 메시에 대한 완전 샤딩이 확장되지 않으면 HSDP를 사용하세요.** ICI
    슬라이스가 너무 커서 모델을 효과적으로 샤딩할 수 없거나, 그 전체에 걸친
    샤딩이 네트워크 트래픽을 과도하게 만든다면 2D 메시와 함께
    `mode="hybrid_shard"`를 사용하세요. 여러 슬라이스에서 실행할 때는 거의
    항상 더 느린 DCN에 걸쳐서는 복제(데이터 병렬)하고 각 슬라이스 내에서는
    샤딩하는 것이 가장 좋습니다.
7.  **확장하기 전에 메모리와 수치를 확인하세요.** `assert_close`로 샤딩되지
    않은 참조 또는 FSDPv2 참조와 비교하고, 순전파-역전파 경계를 넘어 유지되는
    메모리를 측정하세요. 예제에는 일치성 검사(`fsdp_to_simple_fsdp_worker.py`)와
    메모리 테스트(`manual_fsdp_memory_test.py`)가 포함되어 있습니다.

## 5. 예제 실행하기 { #5-running-the-examples }

각 예제에는 8개의 TPU 코어(예: `v6e-8`)에서 실행되는 런처 스크립트가 함께
제공됩니다.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| 예제 | 실행 명령어 | 확인 내용 |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | 컴파일된 FSDPv2와 컴파일된 <br> SimpleFSDP의 손실과 예측이 <br> 일치하는지 확인하고, 각 단계 <br> 시간을 보고합니다. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` 또는 `python3 launch.py` | 수동 FSDP의 그래디언트와 예측이 <br> 단일 디바이스 CPU 참조와 일치하는지 확인합니다. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | 샤딩되지 않은 가중치가 순전파 후 <br> 해제되는지 확인합니다. |
<!-- mdformat on -->
