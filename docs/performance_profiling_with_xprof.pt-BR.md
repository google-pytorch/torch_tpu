# Criação de perfil de desempenho com o XProf { #performance-profiling-with-xprof }

Entender onde o tempo é gasto é fundamental para treinar modelos grandes na
TPU. Este guia mostra como capturar traces de execução com o TPU Profiler e
analisá-los no TensorBoard.

> **Warning:** **Pré-requisito:** o TensorBoard exige `setuptools<81`. A
> versão mais recente remove `pkg_resources`, do qual o TensorBoard depende.
> Instale-o primeiro:
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **Versão do PyTorch:** a integração nativa de TPU com o
> `torch.profiler` exige o PyTorch 2.12 ou mais recente.

## 1. O profiler da TPU { #1-the-tpu-profiler }

O `torch.profiler` padrão (integrado ao `torch_tpu`) oferece uma interface de
**gerenciador de contexto** para rastrear as atividades da CPU e da TPU. Você
coloca o código que quer analisar em um bloco `torch.profiler.profile()`, e os
traces coletados são salvos em disco para visualização no TensorBoard.

**Principais componentes:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Componente                                 | Função                          |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | Gerenciador de contexto padrão que inicia/interrompe <br> a coleta de traces |
| `torch.profiler.ProfilerActivity.CPU`      | Rastreia operações do lado da CPU (despacho, carregamento <br> de dados) |
| `torch.profiler.ProfilerActivity.TPU`      | Rastreia a execução no hardware da TPU (computação, <br> memória, interconexão) |
| `torch.profiler.tensorboard_trace_handler` | Callback que salva os traces em um diretório para o <br> TensorBoard       |
| `TpuProfilerConfig`                        | Auxiliar para a configuração personalizada da criação de perfil da TPU |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. Configuração personalizada da criação de perfil (`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

As opções passadas com `TpuProfilerConfig` permitem um controle detalhado dos
níveis dos tracers do host e do dispositivo, além da estrutura dos diretórios de
saída.

### Opções de configuração aceitas { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Chave da opção        | Tipo           | Padrão  | Descrição                 |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | Controla o nível de detalhe do tracer do host da CPU <br> (0=desativado, 1=crítico, 2=tudo). |
| `device_tracer_level` | `int`          | `1`     | Controla o nível de detalhe do tracer do dispositivo <br> de hardware da TPU (0=desativado, 1=ativado). |
| `python_tracer_level` | `int`          | `0`     | Controla o nível de rastreamento de funções <br> Python (0=desativado, 1=ativado). |
| `run_dir`             | `pathlib.Path` | `None`  | Diretório de saída em que os arquivos de trace do hardware <br> da TPU (`.xplane.pb`) são salvos. |
| `worker_rank`         | `str`          | `None`  | Rank do worker adicionado como sufixo ao <br> nome do arquivo XPlane (`<hostname>_<rank>.xplane.pb`). |
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

## 3. Como capturar um perfil { #3-capturing-a-profile }

A célula abaixo executa um pequeno loop de treinamento dentro do contexto do
profiler. Os pontos principais são:

1.  Coloque seu loop de treinamento dentro de `torch.profiler.profile()`
2.  Especifique as atividades: `CPU` para operações do lado do host e `TPU`
    para a execução no hardware
3.  Defina `on_trace_ready` para salvar os traces em um diretório usando
    `torch.profiler.tensorboard_trace_handler`
4.  (Opcional) Passe `experimental_config` com `TpuProfilerConfig` para
    controlar os níveis dos tracers

> **Important:** o profiler adiciona sobrecarga. Analise apenas um subconjunto
> representativo de etapas (por exemplo, de 100 a 1.000), e não toda a execução
> do treinamento.

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

## 4. Como visualizar traces no TensorBoard { #4-viewing-traces-in-tensorboard }

Execute os seguintes comandos:

**Etapa 1:** instale o plug-in do profiler (necessário para que o painel
apareça):

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**Etapa 2:** inicie o TensorBoard:

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

O XProf também pode ser executado como um servidor independente.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**Etapa 3:** abra `http://localhost:6006` no navegador e selecione a guia
**"Profile"**.

> **Note:** você precisa executar o TensorBoard no terminal integrado do VS Code
> para que o encaminhamento de portas funcione automaticamente.

**No GKE:** grave o diretório de traces em um bucket do Cloud Storage montado
com o
[driver CSI do Cloud Storage FUSE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
para que os traces continuem existindo depois do pod e, em seguida, abra-os com
`xprof
--logdir=gs://<bucket>/profiler_output`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Visualização       | O que ela mostra                                       |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | Resumo geral: tempo de etapa, utilização do dispositivo, principais operações |
| **Trace Viewer**   | Linha do tempo de cada operação na CPU e na TPU: procure lacunas (hardware ocioso) |
| **Op Profile**     | Detalhamento do tempo por operação: encontre as operações mais caras |
| **Memory Profile** | Uso de HBM ao longo do tempo: detecte picos de memória |
<!-- mdformat on -->

## 5. O que observar { #5-what-to-look-for }

### Sinais de alerta nos traces { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Padrão                  | Significado             | Correção                 |
| ----------------------- | ----------------------- | ------------------------ |
| Grandes lacunas entre operações da TPU | Gargalo no lado do host <br> (carregamento de dados, sobrecarga do Python) | Faça prefetch dos dados e reduza a lógica Python no loop |
| Rajadas curtas e frequentes na TPU | Muitas quebras de grafo → recompilação frequente | Reduza as chamadas `.item()` / `.cpu()` e use `torch.compile` |
| Baixa utilização da MXU | Dimensões dos tensores não alinhadas ao tamanho do tile (128) | Preencha as dimensões até múltiplos de 128 |
<!-- mdformat on -->

### O perfil ideal { #the-ideal-profile }

Uma carga de trabalho de TPU bem otimizada mostra:

-   **Blocos de computação da TPU longos e contínuos**: o compilador fundiu
    muitas operações em uma única execução
-   **Lacunas mínimas entre CPU→TPU**: o carregamento de dados se sobrepõe à
    computação
-   **Alta utilização da MXU (>60%)**: os formatos dos tensores estão alinhados
    ao hardware

## 6. Como verificar se os traces foram capturados { #6-verifying-the-traces-were-captured }

A célula abaixo verifica se o diretório de saída do profiler contém arquivos de
trace.

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

## 7. Criação de perfil de kernels (contadores de desempenho detalhados) { #7-kernel-profiling-fine-grained-performance-counters }

A criação de perfil de kernels permite coletar contadores de desempenho
detalhados do hardware (por exemplo, para kernels Pallas) para diagnosticar
gargalos de desempenho que não aparecem nos traces de eventos padrão.

> **Note:** a amostragem detalhada de contadores de desempenho é um recurso
> avançado de hardware disponível na **Ironwood (TPU v7/TPU7x)** e nas gerações
> de TPU seguintes.

### Como ativar os contadores de desempenho { #enabling-performance-counters }

Você pode ativar os contadores de desempenho passando a configuração deles com
`experimental_options` em `TpuProfilerConfig`:

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

Para ver a lista detalhada de parâmetros de configuração (como intervalos de
amostragem, escalonamento e modos de acionamento) e as instruções sobre como
determinar os índices dos contadores, consulte o
[guia oficial de criação de perfil de kernels do OpenXLA](https://openxla.org/xprof/kernel-profiling).

## 8. Referência rápida { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
Etapa       | Comando / código
----------- | --------------
**Importar** | `from torch.tpu.profiler import TpuProfilerConfig`
**Criar perfil** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**Salvar**   | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**Visualizar** | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** analise apenas de 100 a 1.000 etapas para manter os arquivos de
> trace com um tamanho razoável. O profiler adiciona uma sobrecarga de ~5 a 10%
> por etapa.
