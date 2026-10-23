# Introdução ao TorchTPU { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

O TorchTPU é um backend nativo do PyTorch, de código aberto, para
[Google Cloud TPUs](https://cloud.google.com/tpu), desenvolvido em
colaboração entre o Google e a equipe do PyTorch na Meta. Com o TorchTPU, os
desenvolvedores podem executar cargas de trabalho PyTorch existentes em hardware
TPU com mudanças mínimas no código e escalonar cargas de trabalho de treinamento
e inferência de um único chip até pods de TPU com várias frações (multi-slice).

## Por que usar o TorchTPU? { #why-use-torchtpu }

As TPUs do Google são circuitos integrados de aplicação específica (ASICs)
criados especialmente para as operações de matrizes e vetores que alimentam o
deep learning, com alta taxa de transferência, eficiência energética e
desempenho por dólar em grande escala. Historicamente, as TPUs foram usadas com
os ecossistemas JAX e TensorFlow, mas o TorchTPU traz suporte nativo a TPUs
diretamente para o
[ecossistema PyTorch](https://github.com/google-pytorch/torch_tpu), projetado
intencionalmente com base nestes três princípios orientadores:

*   **Integração nativa com o PyTorch:** o TorchTPU se integra diretamente ao
    PyTorch, sem dependências do runtime do JAX. Ele preserva a semântica padrão
    do PyTorch de views de tensores e de mutação in-place, adota o modelo de
    execução do PyTorch com um processo por dispositivo e oferece suporte a
    `torch.compile` por meio do Dynamo.
*   **Pronto para uso:** para portar uma carga de trabalho de uma GPU para
    uma TPU, comece mudando `device="cuda"` para `device="tpu"` e
    `backend="nccl"` para `backend="tpu_dist"`. Seu modelo PyTorch sem
    modificações é executado imediatamente no modo eager, com mensagens de erro
    claras para diagnosticar problemas de runtime ou do compilador.
*   **Desempenho em escala de produção:** além da execução eager operação por
    operação, o TorchTPU traz um modo eager com fusão adiada, um pipeline de
    compilação do Dynamo para o XLA, dinamismo limitado para evitar
    recompilações causadas por formatos e APIs de kernels personalizados para
    ajudar você a alcançar a utilização máxima do hardware.

## Guia de início rápido: como executar na TPU { #quickstart-running-on-tpu }

Na célula a seguir, você pode ver um exemplo simples de um programa PyTorch
executado na TPU. Você pode executar
[este exemplo no Colab](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)
ativando o ambiente de execução de TPU.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## Principais recursos { #key-features }

*   **Cobertura ampla de operadores e modelos:** oferece suporte a 450
    operadores principais do ATen (tanto no forward quanto no backward pass) e é
    validado continuamente em 5.000 modelos populares do HuggingFace, que abrangem
    LLMs (Llama, Qwen, Gemma, DeepSeek, Mistral), modelos de visão (ViT, ResNet,
    ConvNeXt, DINOv2), modelos de áudio (Whisper, Wav2Vec2) e pipelines de
    difusão (FLUX, Stable Diffusion).
*   **Três modos de execução eager:**
    *   **Strict (padrão):** despacha um operador ATen por vez, de forma
        assíncrona, para a TPU, igual ao comportamento eager padrão do PyTorch
        em GPUs.
    *   **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** executa cada operador de
        forma síncrona com a CPU e ativa a verificação de limites em tempo de
        execução para facilitar a depuração.
    *   **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** adia e funde
        sequências de operadores ATen em blocos de computação otimizados até um
        ponto de materialização (como `.cpu()` ou `.item()`), o que reduz muito
        o tráfego da memória de alta largura de banda (HBM) e acelera a
        execução eager.
*   **Compilação de grafos com `torch.compile`:** registra o backend
    `"tpu"` do Dynamo (`torch.compile(model)`) para converter (lowering) grafos
    FX inteiros do PyTorch em
    [StableHLO](https://openxla.org/stablehlo/spec) e compilá-los com o
    [compilador OpenXLA](https://openxla.org/xla). O suporte ativo a formatos
    dinâmicos (`torch._dynamo.mark_dynamic`) permite que grafos compilados lidem
    com tamanhos de batch ou comprimentos de sequência variáveis até um limite
    especificado, sem recompilar.
*   **Dinamismo limitado:** reduz de forma nativa as recompilações causadas por
    tamanhos de batch e comprimentos de sequência variáveis, compilando com base
    em limites superiores dinâmicos.
*   **Cache de compilação em vários níveis:** elimina compilações redundantes do
    XLA com três níveis de cache:
    *   **Nível 1:** cache em memória local do processo (sempre ativado).
    *   **Nível 2:** memória compartilhada local do host para reutilização entre
        processos em jobs com vários dispositivos.
    *   **Nível 3:** cache remoto persistente (como o Cloud Storage) para
        reutilização entre hosts e reinicializações de jobs.
*   **Primitivas distribuídas nativas:** implementa operações coletivas de TPU
    no nível do Process Group do `torch.distributed` (`c10d::Backend`) com
    `backend="tpu_dist"`, com suporte a `DistributedDataParallel` (DDP),
    `FullyShardedDataParallel` (FSDP2 e SimpleFSDP) e `DTensor`. Você também
    pode usar `torch.tpu.distributed.topology_aware_mesh` para alinhar as
    dimensões de paralelismo às interconexões físicas da TPU.
*   **Kernels de TPU personalizados:** escreva kernels personalizados em
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (que podem ser
    chamados com `torch.tpu.pallas.jax_op` e têm suporte direto a coletivas RDMA
    entre chips), escreva kernels baseados em tiles com a DSL experimental
    [Helion](https://pytorch.org/blog/helion/) ou importe kernels de TPU
    otimizados e prontos do Tokamax.
*   **Integrações com o ecossistema:** funciona de imediato com os frameworks
    padrão de serving e treinamento do PyTorch e com ferramentas de criação de
    perfil,
    como `torch.profiler` (TensorBoard/XProf), vLLM, SGLang, TorchTitan, Ray e
    TorchRec.

## Limitações e trabalhos futuros { #limitations-and-future-work }

Como qualquer produto, o TorchTPU tem limitações conhecidas, que serão
resolvidas com o tempo. Nossa primeira versão oferece cobertura robusta e
imediata para as principais cargas de trabalho, mas o TorchTPU é um esforço
colaborativo e conduzido pela comunidade. Confira nosso
[conjunto de recursos e roadmap](./feature_set.md) para ver o que vem por aí ou
participe agora mesmo
[enviando PRs](https://github.com/google-pytorch/torch_tpu) no GitHub para nos
ajudar a resolver essas limitações. Este é um projeto da comunidade, e
adoraríamos receber seu feedback. Convidamos você a colaborar conosco pelo
[GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
e [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

*   Formatos dinâmicos causam recompilações
    *   No TorchTPU, `torch.compile` define `dynamic=False` por padrão.
        `dynamic=None` ou `dynamic=True` ainda não têm suporte para uso em
        produção.
        [Veja a discussão sobre formatos dinâmicos](./dynamic_shapes.md)
    *   Por exemplo, executar a etapa de decode/generate em LLMs do Hugging
        Face envolve recompilações a cada etapa, a menos que o dinamismo
        limitado esteja ativado.
*   Operações de view podem ser caras na TPU (consulte
    [Modelo de memória do TorchTPU](./torchtpu_memory_model.md)).
*   O MPMD com coletivas tem suporte funcional, mas ainda
    não alcança o desempenho máximo. A sobreposição de coletivas com computação
    tem [algumas restrições](./distributed_support.md):
    *   O suporte completo a streams com offload para o SparseCore será lançado
        em breve.
    *   Use o SimpleFSDP com `torch.compile()` em vez do FSDPv2 com o modo eager
        ou com `torch.compile`. Consulte
        [Práticas recomendadas de FSDP](./case_studies_fsdp.md)
    *   A forma recomendada de obter o desempenho máximo das otimizações do XLA
        em módulos que se comunicam (ou seja, de permitir a sobreposição entre
        computação e comunicação) é usar o decorador `spmd_safe` em regiões que
        **os usuários garantem** que são exatamente iguais
*   No momento, o TorchTPU não tem suporte nativo a FlexAttention
*   As TPUs da geração atual não oferecem suporte a acumulação de matmul com
    precisão menor que fp32 (por exemplo, tf32).
*   Só um processo pode acessar um dispositivo TPU, o que limita o suporte a RL
    colocalizado (observe que um chip de TPU pode ter um ou mais dispositivos
    associados, e eles podem ser acessados por processos separados do sistema
    operacional).

**Outras recomendações:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Limitação | Recomendação |
| :--- | :--- |
| Os modos eager têm um tempo de aquecimento maior que zero | Ative o cache de compilação persistente |
| O Torch.compile demora muito | Ative o cache de compilação persistente |
| CNNs no PyTorch usam por padrão o layout channel-first, que é muito ineficiente na TPU | Use esta [opção do PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) para definir um layout channel-last e ter um desempenho melhor. |
| O mapeamento da malha de dispositivos pode não ser ideal por padrão devido a diferenças de topologia entre TPUs e outras arquiteturas | Use `topology_aware_mesh` para ter um mapeamento ideal da malha de dispositivos e poder personalizar esse mapeamento |
<!-- mdformat on -->

## Primeiros passos { #getting-started }

Esta documentação está organizada em quatro seções estruturadas para ajudar você
a passar da primeira configuração até a implantação distribuída em produção:

*   **Primeiros passos:** tutoriais passo a passo para configurar seu ambiente e
    implantar cargas de trabalho em [uma VM](./get_started_on_a_vm.md) ou
    [por meio do GKE](./get_started_on_gke.md).
*   **Detalhes técnicos:** explicações aprofundadas sobre o
    [modelo de memória da TPU](./torchtpu_memory_model.md), a
    [execução distribuída](./distributed_support.md) e os diferentes
    [modos de execução](./torchtpu_execution_model.md).
*   **Estudos de caso:** receitas de ponta a ponta que abordam otimizações para
    o [modelo Flux](./case_studies_flux.md), o treinamento do
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md), a
    [inferência de LLM de ponta a ponta](./case_studies_muse_glimmer_30b_inference.md)
    e a aplicação eficaz do [FSDP](./case_studies_fsdp.md).
*   **Analise sua carga de trabalho:**
    [criação de perfil](./performance_profiling_with_xprof.md) e gerenciamento da
    [precisão numérica](./numerical_precision.md)
