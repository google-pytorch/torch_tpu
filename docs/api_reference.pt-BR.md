# Referência da API do TorchTPU { #torchtpu-api-reference }

Este documento é o guia de referência das APIs Python públicas e das variáveis
de ambiente do TorchTPU. Ele foi criado para ajudar você a:

*   **Migrar de CUDA para TPU**: mapear chamadas comuns de `torch.cuda` para as
    APIs independentes de dispositivo do PyTorch em
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html).
*   **Usar recursos específicos de TPU**: configurar modos de execução,
    precisão numérica da MXU, topologia distribuída, operadores personalizados
    de Pallas/TPU, otimizadores rastreáveis e criação de perfil com XProf por
    meio do módulo **`torch.tpu`**.
*   **Configurar o runtime**: ajustar o cache de compilação, as verificações de
    depuração e as configurações de inicialização multi-host por meio de
    variáveis de ambiente.

> **Note:** para a lista completa e gerada automaticamente, que é a fonte
> oficial de todas as APIs Python públicas e variáveis de ambiente, consulte
> [APIs públicas do TorchTPU (`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md).

## **Consultas básicas de dispositivo** { #core-device-queries }

Quando você importa `torch` com o TorchTPU instalado, o backend registra
automaticamente o dispositivo `"tpu"` e o módulo `torch.tpu`. Quando o PyTorch
oferece uma API independente de dispositivo em
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html),
use-a em vez de uma API específica do backend para que seu código continue
portável entre aceleradores. Use `torch.tpu` para propriedades de hardware
específicas de TPU que não têm equivalente genérico.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API de TPU / Accelerator | Descrição |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | Retorna `torch.device("tpu")` quando a TPU é o acelerador ativo. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Verifica se o backend de TPU está disponível. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Verifica se o estado do runtime do TorchTPU foi inicializado. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Número de dispositivos visíveis para o processo atual. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Retorna o índice do dispositivo TPU ativo. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Define o índice do dispositivo TPU ativo para o processo atual. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | Gerenciador de contexto que define temporariamente o índice do dispositivo ativo. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | Retorna o nome legível do modelo da TPU conectada (por exemplo, `"TPU v6e"`). |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | Retorna as propriedades estáticas do dispositivo TPU (incluindo `name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device` e `sparse_core_simd_width`). |
<!-- mdformat on -->

As APIs `torch.accelerator` desta tabela exigem o PyTorch 2.11 ou mais recente.
No caso raro de você usar uma versão mais antiga do PyTorch, use a API
`torch.tpu` correspondente (por exemplo, `torch.tpu.device_count()` ou
`torch.tpu.current_device()`). No PyTorch 2.11 e versões mais recentes, essas
APIs `torch.tpu` redundantes são internas do TorchTPU, e chamá-las diretamente
no código do usuário gera um erro.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **Geração de números aleatórios (RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API de TPU / Accelerator | Descrição |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Define a semente do gerador de números aleatórios do dispositivo TPU ativo. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Define a semente do RNG em todos os dispositivos TPU locais. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Retorna a semente aleatória inicial do dispositivo TPU atual. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Retorna o tensor de estado do RNG do dispositivo atual. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | Define o tensor de estado do RNG do dispositivo TPU ativo do processo atual. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | Tupla inicializada sob demanda com as instâncias padrão de `torch.Generator`, indexada pelo ordinal do dispositivo local. |
<!-- mdformat on -->

As APIs `torch.accelerator.random` desta tabela exigem o PyTorch 2.13 ou mais
recente. No caso raro de você usar uma versão mais antiga do PyTorch, use
`torch.tpu.initial_seed()` ou `torch.tpu.get_rng_state()`. No PyTorch 2.13 e
versões mais recentes, essas APIs `torch.tpu` são internas do TorchTPU, e
chamá-las no seu código gera um erro.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Precisão mista automática (AMP)** { #automatic-mixed-precision-amp }

O TorchTPU se integra ao framework padrão
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) do
PyTorch usando a string de tipo de dispositivo `"tpu"`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API de TPU | Descrição |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Gerenciador de contexto / decorador que ativa a precisão mista automática (`bfloat16` por padrão na TPU). |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Lista os dtypes com suporte na TPU (`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, tipos inteiros, booleanos e complexos). |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Streams e eventos** { #streams-and-events }

Para gerenciar streams e sincronizar dispositivos, use as APIs padrão de
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
em vez de APIs específicas do backend. Use `torch.tpu` somente para construir
objetos `Stream` ou `Event` de TPU diretamente ou para configurar a fixação
(pinning) de núcleos da TPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API CUDA | API de TPU / Accelerator | Descrição |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Retorna o stream ativo de um dispositivo. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | Define o stream ativo. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | Retorna o `TpuStream` padrão do dispositivo. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | Gerenciador de contexto que seleciona um determinado `TpuStream`. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | Aguarda a conclusão de todo o trabalho pendente no dispositivo TPU. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | Subclasse de `torch.Stream`. Oferece suporte a `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()` e fixação de núcleo (`pinned="dense"` ou `"sparse"`, consultada via `is_dense` / `is_sparse`). |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | Subclasse de `torch.Event`. Oferece suporte a `record()`, `synchronize()`, `query()` e `elapsed_time(end_event)`. |
<!-- mdformat on -->

As APIs `torch.accelerator` desta tabela exigem o PyTorch 2.11 ou mais recente.
No caso raro de você usar uma versão mais antiga do PyTorch, use
`torch.tpu.current_stream()`, `torch.tpu.set_stream()` ou
`torch.tpu.synchronize()`.

> **Note:** os streams são um conceito de CUDA sem analogia direta na TPU. No
> modelo de execução adiada do TorchTPU, o compilador XLA e o runtime PjRt
> agendam e sobrepõem as operações automaticamente com base nas dependências de
> dados, então não é necessário gerenciar vários streams manualmente.

## **APIs específicas de TPU** { #tpu-specific-apis }

Além dos equivalentes padrão de CUDA, `torch.tpu` expõe APIs criadas
especificamente para modos de execução da TPU, precisão da MXU, compilação XLA,
topologia distribuída, kernels Pallas personalizados, otimizadores rastreáveis
e criação de perfil.

### **Modos de execução e opções do compilador** { #execution-modes-and-compiler-options }

Controle como as operações eager são enfileiradas, fundidas e compiladas pelo
XLA. Para uma explicação da arquitetura, consulte
[Modos de execução do TorchTPU](./torchtpu_execution_model.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.EagerMode` | Enum que controla a execução eager: `DEFER_AND_FUSE`, `DEFER_NEVER` e `DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `torch.tpu.eager_mode` | Obtém ou define o `EagerMode` ativo globalmente. |
| `torch.tpu.set_eager_mode(mode)` | Gerenciador de contexto que define temporariamente o `EagerMode` ativo. |
| `torch.tpu.compiler.custom_compiler_options(options)` | Gerenciador de contexto que passa flags personalizadas do compilador XLA para as operações eager materializadas dentro do bloco. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **Precisão numérica** { #numerical-precision }

Controle a precisão da Matrix Multiply Unit (MXU) e a acumulação intermediária
em `float32`. Para orientações detalhadas, consulte
[Precisão numérica](./numerical_precision.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.Precision` | Enum que controla a precisão da MXU para matmuls e convoluções em `float32`: `Precision.DEFAULT`, `Precision.HIGH` e `Precision.HIGHEST`. |
| `torch.tpu.precision(level)` | Gerenciador de contexto que define o nível de `Precision` para operações de matmul e convolução dentro do bloco. |
| `torch.tpu.allow_excess_precision` | Obtém ou define globalmente a flag `--xla_allow_excess_precision` do XLA para manter em `float32` os valores intermediários fundidos. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **Treinamento distribuído e topologia** { #distributed-training-and-topology }

O TorchTPU registra `"tpu_dist"` (também com o alias `"tpu"`) como o backend
padrão de `torch.distributed` para dispositivos `"tpu"`. Consulte
[Primeiros passos com o TorchTPU no GKE](./get_started_on_gke.md) e
[Práticas recomendadas de FSDP](./case_studies_fsdp.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | Calcula um tensor de layout de ranks com reconhecimento de topologia, com formato `mesh_shape`, para `DeviceMesh("tpu", mesh=...)`. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | Preenche as variáveis de ambiente de topologia da TPU e de inicialização multi-host antes de iniciar os workers. |
| `torch.tpu.distributed.spmd_safe(func)` | Decorador que declara que todos os ranks executam operações de TPU idênticas em `func`, o que permite a fusão eager de coletivas. |
| `torch.tpu.window_num_stripes` | Obtém ou define o número máximo de stripes de DMA simultâneos para transferências ponto a ponto. |
| `torch.tpu.window_stripe_chunk_mb` | Obtém ou define o tamanho mínimo de payload (em MB) por stripe de DMA para transferências ponto a ponto. |
<!-- mdformat on -->

### **Operadores personalizados de TPU (`torch.tpu.ops`) e Pallas (`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

O TorchTPU expõe operadores especializados para o hardware em `torch.tpu.ops`
e um wrapper de kernels JAX/Pallas personalizados em `torch.tpu.pallas`.
Consulte [Como adicionar kernels personalizados usando Pallas](./custom_kernels.md).

#### **Operadores gerais, de atenção, de MoE e coletivos** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | Aplica max pooling 1D, 2D ou 3D na TPU sem materializar um tensor de índices. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | Calcula o backward pass de `max_pool{1,2,3}d` sem um tensor de índices. |
| `torch.tpu.ops.optimization_barrier(inputs)` | Impede que o XLA mova, funda ou elimine duplicatas de computações através da barreira. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | Atenção por produto escalar escalonado (scaled dot-product attention) com base no kernel Pallas Splash Attention da TPU. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Multiplicação de matrizes agrupada para cargas de trabalho Mixture-of-Experts (MoE). |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | Calcula o gradiente dos pesos para `ragged_dot`. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | Preenche (padding) um buffer `NestedTensor` `torch.jagged` até um múltiplo especificado para alinhamento na TPU. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | Remove o padding final adicionado por `align_jagged_to_multiple`. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | Faz all-gather de `input` em `group` e concatena ao longo da dimensão `dim`. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | Reduz `input` em `group` e distribui (scatter) os blocos ao longo da dimensão `dim`. |
| `torch.tpu.ops.ragged_all_to_all(...)` | Troca fatias de tensor de comprimento variável entre os ranks de um grupo de processos. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | Envio e recebimento ponto a ponto de tensores no grupo de processos padrão da TPU. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | Registra uma função JAX/Pallas como um operador personalizado rastreável do PyTorch. |
<!-- mdformat on -->

#### **Operadores de embedding do SparseCore** { #sparsecore-embedding-operators }

Estes operadores exigem uma TPU com SparseCores. A propriedade
`sparse_cores_per_device` de `torch.tpu.get_device_properties()` é o número de
SparseCores por dispositivo (0 em TPUs sem SparseCores), e
`sparse_core_simd_width` é a largura SIMD deles.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | Executa a busca e a agregação de embeddings CSR no SparseCore da TPU. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | Calcula os gradientes do backward de `sparse_dense_matmul` e aplica uma atualização fundida e in-place da tabela com SGD, Adagrad ou Adam no SparseCore. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | Divide ativações de embedding empilhadas do SparseCore em tensores por feature. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | Empilha gradientes de ativação por feature em um único tensor para o SparseCore. |
| `torch.tpu.ops.sparse_gather(...)` | Coleta linhas de um tensor 2D usando índices CSR no SparseCore da TPU. |
| `torch.tpu.ops.sparse_gather_backward(...)` | Faz scatter-add dos gradientes de `sparse_gather` no SparseCore da TPU. |
| `torch.tpu.ops.sparse_iota(...)` | Gera índices de posição dentro da linha para os elementos não nulos de uma matriz CSR no SparseCore. |
| `torch.tpu.ops.sparse_map_row(...)` | Mapeia uma computação MLIR personalizada, linha a linha, sobre matrizes CSR esparsas no SparseCore. |
<!-- mdformat on -->

### **Etapa de treinamento fundida e otimizadores (`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` rastreia o forward pass, o cálculo da loss, o backward
pass e a etapa funcional do otimizador em um único grafo de TPU compilado.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | Gerencia os parâmetros funcionais (`params`) e o estado do otimizador (`param_group`), e compila a etapa de treinamento completa via `make_compiled_train_step(...)`. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | Classe base para otimizadores funcionais sem estado (expõe `defaults`, `init_param_group` e `step`). |
| `torch.tpu.optimizers.AdamW(...)` | Otimizador AdamW funcional e rastreável (`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | Otimizador Muon funcional e rastreável (`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | Otimizador SGD funcional e rastreável (`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **Criação de perfil (`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

Configure o rastreamento do hardware da TPU e do Python para XProf e
TensorBoard. Consulte [XProf para TorchTPU](./performance_profiling_with_xprof.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descrição |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | Configuração experimental passada para `torch.profiler.profile(..., experimental_config=...)` para configurar o rastreamento do host, do dispositivo TPU e do Python para o XProf. |
<!-- mdformat on -->

## **Variáveis de ambiente** { #environment-variables }

O comportamento do TorchTPU é controlado por diversas variáveis de ambiente que
configuram o compilador XLA, o driver da TPU (`libtpu`) e o próprio subsistema
de depuração do PyTorch.

> **CRÍTICO: inicialização global do processo** As variáveis de ambiente do
> TorchTPU ficam **bloqueadas** no momento em que o backend é inicializado
> (geralmente durante `import torch` ou `torch.device("tpu")`).

### **Visibilidade de dispositivos, runtime e inicialização distribuída** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variável de ambiente | Descrição |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | Índices de chips TPU, separados por vírgula, visíveis para o processo atual (análogo a `CUDA_VISIBLE_DEVICES`). |
| `RANK` | Rank global do processo atual em um job distribuído. |
| `LOCAL_RANK` | Rank local do processo atual no host. |
| `WORLD_SIZE` | Número total de processos no job distribuído. |
| `NNODES` | Número total de nós host em um job multi-host. |
| `NODE_RANK` | Rank do nó host atual em um job multi-host. |
| `MASTER_ADDR` | Endereço do nó coordenador de rank 0 para `torch.distributed`. |
| `MASTER_PORT` | Porta em `MASTER_ADDR` usada para o rendezvous de `torch.distributed`. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Defina como `"0"` para desativar o registro automático do backend `torch_tpu`. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | Quando ativada, permite chamar as APIs internas de dispositivo de `torch.tpu` (as que têm equivalente em `torch.accelerator`) de fora do núcleo do PyTorch, com um aviso único em vez de gerar um `RuntimeError`. Use-a somente como correção temporária. Prefira migrar para `torch.accelerator.*`, pois esta variável provavelmente será removida em uma versão futura. |
| `LIBTPU_INIT_ARGS` | Flags de inicialização, separadas por espaço, passadas para a `libtpu` e o runtime PJRT. |
| `TPU_PREMAPPED_BUFFER_SIZE` | Tamanho, em bytes, do buffer de memória pré-mapeado de host para dispositivo do cliente PJRT. |
| `TMPDIR` | Diretório temporário usado pelo runtime e pelo compilador. |
<!-- mdformat on -->

### **Topologia de hardware e coordenação multi-host** { #hardware-topology-and-multi-host-coordination }

Estas variáveis especificam a topologia de hardware da TPU e a rede entre os
processos. Na maioria dos casos, elas são definidas automaticamente por
`torch.tpu.distributed.set_tpu_launch_env()`, pelo inicializador do TorchTPU
ou pelo ambiente do Google Cloud.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variável de ambiente | Descrição |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | Topologia física da fração (slice) de TPU (por exemplo, `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Endereços `host:port` dos workers, separados por vírgula, para o serviço slice builder da TPU. |
| `TORCH_TPU_DEVICES_PER_HOST` | Número de chips TPU por host ao derivar os endereços do slice builder. |
| `TORCH_TPU_BASE_PORT` | Número da porta inicial ao derivar os endereços do slice builder. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | Etapa em que os ranks verificam se os grafos de coletivas são iguais entre os ranks. |
| `TORCHCOMM_RANK` | Rank do processo para `torchcomm` (tem precedência sobre `RANK`). |
| `TORCHCOMM_SIZE` | Tamanho do mundo (world size) para `torchcomm` (tem precedência sobre `WORLD_SIZE`). |
<!-- mdformat on -->

### **Modo de execução, compilação e cache** { #execution-mode-compilation-and-caching }

Estas variáveis controlam como as operações e os grafos do PyTorch são
despachados, convertidos (lowering), otimizados pelo XLA e armazenados em cache
entre execuções e ranks.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variável de ambiente | Descrição |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | Quando ativada (`"1"`/`"true"`), define o modo eager padrão como `EagerMode.DEFER_AND_FUSE`. |
| `TORCH_TPU_LAUNCH_BLOCKING` | Quando ativada (`"1"`/`"true"`), define o modo eager padrão como `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | Nome do diretório em `/dev/shm/torch_tpu_cache/` para o cache de compilação local do host (ou `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Caminho (por exemplo, `gs://...` ou um caminho local) para o cache de compilação persistente de nível 3, compartilhado entre hosts e execuções. |
| `XLA_FLAGS` | Flags do compilador XLA separadas por espaço (por exemplo, `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **Depuração, segurança, criação de perfil e rastreamento** { #debugging-safety-profiling-and-tracing }

Estas variáveis controlam o contexto de depuração, as verificações de segurança
em tempo de execução e a saída do profiler do TorchTPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variável de ambiente | Descrição |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | Quando definida como `"1"`, enriquece as exceções `RuntimeError` do Python com backtraces de C++. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Quando definida como `"1"`, ativa as verificações de depuração e de limites em tempo de execução na TPU. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | Diretório de saída alternativo para os arquivos de trace do XProf quando `TpuProfilerConfig.run_dir` não está definido. |
| `TORCH_TPU_XPROF_SESSION_ID` | ID da sessão de criação de perfil do XProf compartilhada entre os workers multi-host. |
| `TORCH_TRACE` | Diretório de saída para logs estruturados de trace do compilador e do dispatcher (`tlparse`). |
<!-- mdformat on -->

### **Exemplos de configuração de variáveis de ambiente** { #environment-variable-configuration-snippets }

#### **1. Stack traces de C++ (`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

Definir `TORCH_SHOW_CPP_STACKTRACES=1` antes de importar `torch` facilita muito
o diagnóstico de erros que ocorrem nas camadas mais profundas do backend em C++
ou do XLA. Em vez de um simples `RuntimeError` do Python, você recebe um `C++
CapturedTraceback` detalhado que mostra os frames de execução exatos.

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. Verificações internas de depuração e validação de limites (`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

Por padrão, a TPU ignora as verificações de limites para ganhar velocidade.
Ativar `TORCH_TPU_ENABLE_DEBUG_CHECKS=1` detecta bugs sutis relacionados a
índices fora dos limites (como em `torch.take`), embora deixe a execução mais
lenta devido à sincronização extra entre CPU e TPU. Operações que, de outra
forma, falhariam silenciosamente ou corromperiam dados geram uma exceção
explícita como `take(): expected indices to be
in range`:

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. Como gerar dumps de IR / HLO do XLA (`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

Ao definir `XLA_FLAGS`, você pode forçar o compilador a gerar um dump do texto
do High-Level Optimizer (HLO) em um diretório. O diretório de dump (por
exemplo, `/tmp/hlo_dump`) vai conter:

*   **`*.after_optimizations.txt`**: o grafo HLO final, que mostra as operações
    no nível do hardware.
*   **`*.flagfile`**: a lista completa de flags ativas do compilador XLA
    (verifica `xla_optimization_level`).
*   **`*tpu_comp_env.txt`**: configurações internas do ambiente da TPU usadas
    durante a compilação.

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. Cache de compilação em vários níveis (`TORCH_TPU_TIER2_COMPILATION_CACHE` e `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

O TorchTPU usa cache em vários níveis para evitar compilações redundantes entre
execuções e ranks:

*   **Nível 2 (`TORCH_TPU_TIER2_COMPILATION_CACHE`)** fica armazenado
    localmente na memória compartilhada do host. Ele espera um **nome** de
    diretório (por exemplo, `"tpu_tier2_cache"`), que o backend coloca
    automaticamente em `/dev/shm/torch_tpu_cache/<name>`.
*   **Nível 3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** é um cache de
    compilação persistente compartilhado entre hosts e sessões (geralmente um
    URI de bucket `gs://` ou um caminho em um sistema de arquivos
    compartilhado).

Os artefatos binários compilados (executáveis `*.bin`) são salvos de forma
assíncrona nos diretórios de cache e reutilizados nas execuções seguintes, o
que reduz muito a latência de compilação JIT:

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```
