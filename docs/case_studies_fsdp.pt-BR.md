# Práticas recomendadas de FSDP { #fsdp-best-practices }

O Fully Sharded Data Parallel (FSDP) fragmenta (sharding) os parâmetros do
modelo, os gradientes e o estado do otimizador entre os ranks de paralelismo de
dados, de modo que cada núcleo de TPU armazene apenas uma parte de
`1 / world_size` do modelo. Antes de uma camada ser executada, os pesos
completos dela são reconstruídos com um **all-gather**. Durante o backward, os
gradientes são combinados e fragmentados novamente com um **reduce-scatter**.

Este guia explica a forma recomendada de usar FSDP no TorchTPU e indica
exemplos executáveis em
[`examples/fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp).

Neste guia, você vai aprender:

1.  **Qual abordagem de FSDP escolher** no TorchTPU.
2.  **SimpleFSDP (recomendado)**: FSDP pensado para compilação, com uma única
    chamada de encapsulamento.
3.  **FSDP manual**: como colocar as operações coletivas diretamente na
    definição do modelo.
4.  **Práticas recomendadas** que se aplicam a todas as abordagens.
5.  **Como executar os exemplos**: comandos de inicialização e o que cada
    exemplo verifica.

## 1. Como escolher uma abordagem de FSDP { #1-choosing-an-fsdp-approach }

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
    | Abordagem | Quando usar | Exemplo |
    | :------- | :------------- | :------ |
    | **SimpleFSDP** (`simple_fsdp.data_parallel`) | **Padrão recomendado.** Você quer FSDP ou HSDP sem mudar a definição do modelo.| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) |
    | **FSDP manual** (coletivas manuais) | Você é responsável pelo código do modelo e quer o código de comunicação ao lado da computação para ajustar o desempenho. | [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) |
    | **PyTorch FSDPv2** (`fully_shard`) | Código existente que ainda não foi migrado. Ele funciona, mas os hooks de runtime causam quebras de grafo e, para ter desempenho competitivo, ele depende do suporte futuro a offload de streams para o SparseCore da TPU.
.| [Guia de migração](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md) |
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

## 2. Recomendado: SimpleFSDP { #2-recommended-simplefsdp }

O [`simple_fsdp`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/simple_fsdp.py)
é uma implementação de FSDP projetada para funcionar com `torch.compile`. Ele
foi introduzido no TorchTitan, e uma cópia independente que depende só do
PyTorch está disponível em `examples/fsdp/simple_fsdp.py`.

O SimpleFSDP armazena cada parâmetro como um `DTensor` fragmentado. Quando um
parâmetro é acessado, uma parametrização chama `DTensor.redistribute` para
fazer o all-gather dele. O backward do autograd de `redistribute` faz o
reduce-scatter dos gradientes. Não há hooks de runtime, então a etapa inteira
é compilado sem quebras de grafo. O modelo inteiro é fragmentado com uma única
chamada `simple_fsdp.data_parallel(...)`, sem encapsulamento por camada nem
ajuste de prefetch, e há suporte a HSDP com `mode="hybrid_shard"` em um
`DeviceMesh` 2D.

Para aprender a usar o SimpleFSDP, consulte:

*   [README do SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/README.md):
    guia de início rápido e opções disponíveis.
*   [Tutorial de FSDPv2 para SimpleFSDP](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp/fsdp_to_simple_fsdp.md):
    como migrar código existente que usa `fully_shard`.

## 3. Alternativa: FSDP manual { #3-alternative-manual-fsdp }

Se você controla a definição do modelo, pode escrever o FSDP diretamente com
operações coletivas manuais (`torch.distributed._functional_collectives`) e
`torch.compile`. Cada camada armazena apenas o fragmento do próprio peso e faz o
all-gather do peso completo no forward. As coletivas funcionais têm fórmulas de
autograd, então o backward do all-gather é um reduce-scatter, e você não
precisa escrever uma `torch.autograd.Function` personalizada.

Isso mantém o código de comunicação ao lado do código de computação, o que pode
facilitar o ajuste de desempenho. A desvantagem é que você precisa mudar a
definição do modelo.

Para aprender a implementar o FSDP manual, consulte o
[guia de FSDP manual](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual/manual_fsdp.md).

## 4. Práticas recomendadas { #4-best-practices }

Estas recomendações se aplicam ao SimpleFSDP e ao FSDP manual.

1.  **Compile a etapa de treinamento inteira com `torch.compile(backend="tpu")`.**
    Se não for possível compilar a etapa inteira, compile pelo menos o forward,
    a loss e o backward juntos. Compilar só o módulo do modelo não gera um
    único grafo, o que limita a capacidade do XLA de sobrepor as coletivas à
    computação.
2.  **Use activation checkpointing em cada camada fragmentada.** Encapsule o
    forward de cada camada em
    `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Isso libera
    os pesos não fragmentados após o forward pass de cada camada, em vez de
    mantê-los até o backward, o que gera o comportamento de memória do ZeRO-3
    esperado do FSDP.
3.  **Faça com que as dimensões fragmentadas sejam divisíveis pelo tamanho da
    malha.** O FSDP manual exige isso. O SimpleFSDP replica parâmetros pequenos
    demais para serem fragmentados. Use `shard_dim` para escolher uma dimensão
    que se divida igualmente.
4.  **Use uma seed diferente em cada rank ao inicializar pesos fragmentados.**
    Se todos os ranks inicializarem o próprio fragmento com a mesma seed, o
    peso completo será o mesmo bloco aleatório repetido entre os ranks, o que
    pode prejudicar a convergência. Para ter fragmentos reproduzíveis e
    distintos, use em cada rank uma seed que dependa do rank, como
    `torch.manual_seed(base_seed + dist.get_rank())`.
5.  **Mantenha todos os ranks executando o mesmo programa.** Não coloque
    operações que forçam a materialização (como `.item()`) dentro de
    ramificações específicas de um rank, como `if
    rank == 0`.
6.  **Use HSDP quando a fragmentação completa na malha inteira não escalar.**
    Use `mode="hybrid_shard"` com uma malha 2D quando sua fração (slice) de ICI
    for grande demais para fragmentar o modelo de forma eficaz ou quando a
    fragmentação nela gerar tráfego de rede demais. Ao executar em várias
    frações, quase sempre é melhor replicar (paralelismo de dados) pela DCN,
    que é mais lenta, e fragmentar dentro de cada fração.
7.  **Verifique a memória e os resultados numéricos antes de escalonar.**
    Compare com uma referência não fragmentada ou com FSDPv2 usando
    `assert_close` e meça a memória retida na fronteira entre forward e
    backward. Os exemplos incluem uma verificação de paridade
    (`fsdp_to_simple_fsdp_worker.py`) e um teste de memória
    (`manual_fsdp_memory_test.py`).

## 5. Como executar os exemplos { #5-running-the-examples }

Cada exemplo vem com um script de inicialização que é executado em 8 núcleos de
TPU (por exemplo, `v6e-8`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Exemplo | Comando de inicialização | O que ele verifica |
:-------- | :------------- | :------------- |
| [`fsdp_to_simple_fsdp/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/fsdp_to_simple_fsdp) | `./run_fsdp_to_simple_fsdp.sh` | O FSDPv2 compilado e o SimpleFSDP <br> compilado têm losses e previsões <br> iguais, e os tempos de etapa <br> de cada um são informados. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `./launch.sh` ou `python3 launch.py` | Os gradientes e as previsões do FSDP manual <br> correspondem a uma referência de CPU com um único dispositivo. |
| [`manual/`](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/examples/fsdp/manual) | `python3 manual_fsdp_memory_test.py` | Os pesos não fragmentados são liberados após o <br> forward pass. |
<!-- mdformat on -->
