# Como o TorchTPU funciona { #how-torchtpu-works }

## Noções básicas de TPU { #tpu-primer }

As [TPUs do Google](https://cloud.google.com/tpu) são ASICs especializados,
projetados especificamente para executar as enormes operações de tensores que
são a base do deep learning. A
[arquitetura das TPUs varia](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
entre gerações, mas, de modo geral, um chip de TPU contém:

*   Um ou mais TensorCores. Cada um é um processador de thread único composto
    por unidades de multiplicação de matrizes (MXUs), uma unidade vetorial (VPU)
    e uma unidade escalar.
*   Um ou mais SparseCores, que são especializados em operações esparsas e
    para os quais o TensorCore pode transferir (offload) operações.
*   Uma grande memória de alta largura de banda (HBM), com 288 GB a 8,6 TB/s na
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive),
    que pode ser usada para armazenar dados de ML e compartilhá-los com a CPU.
*   Interconexão bidirecional entre chips (ICI), com 19,2 TBps na
    [TPU 8t e na 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
    que permite a comunicação rápida entre chips de TPU dentro de
    [frações (slices) de TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    configuradas pelo usuário. As frações são subgrupos lógicos de um
    [pod de TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
    formado por um número pré-configurado de chips de TPU conectados em uma
    configuração de toro 3D (na TPU 7 e na TPU 8t) ou Boardfly (na TPU 8i).
*   Rede de data center (DCN) bidirecional, com 400 GBps na
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
    que permite a comunicação entre chips de TPU em frações diferentes.

As TPUs são programadas com a pilha do
[compilador XLA](https://github.com/openxla/xla) por meio de dois pontos de
entrada diferentes: [StableHLO](https://openxla.org/stablehlo/spec) e
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
A ISA real da TPU é proprietária, pode mudar e não está disponível
publicamente. Na prática, você pode pensar no StableHLO e (mais ainda) no
Mosaic como equivalentes ao PTX. Veja a imagem abaixo para uma visão geral:

![Como o TorchTPU funciona](./images/torch_tpu_inputs_to_tpu.svg)

Um programa em C++ pode acessar uma TPU local instanciando um
[PjRT Client](https://openxla.org/xla/pjrt/cpp_api_overview), cuja API oferece
métodos para compilar MLIR (StableHLO ou Mosaic) em um executável de TPU,
iniciar esse executável e copiar dados para a TPU e a partir dela. Observe que a
execução na TPU é assíncrona e a CPU pode esperar que um resultado da TPU fique
pronto aguardando o PjRTBuffer associado. O StableHLO segue o
[paradigma SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) e,
por isso, não permite mutações in-place, ou seja, uma operação de escrita sempre
produz um novo valor (pelo menos conceitualmente). Isso é especialmente
importante para o TorchTPU, que oferece a semântica conhecida de "view" do
PyTorch, incluindo mutações, por meio da decomposição e da inversão de views. O
compilador XLA pressupõe que os formatos dos tensores são conhecidos no momento
da compilação. No TorchTPU, essa restrição é flexibilizada pelo
[dinamismo limitado](./dynamic_shapes.md). Como reconhecemos que isso é uma
limitação para cargas de trabalho PyTorch, estamos adicionando suporte ao
dinamismo limitado, ou seja, quando os limites máximos das dimensões de um
formato são conhecidos no momento da compilação, mas as dimensões reais só são
conhecidas em tempo de execução.

## TorchTPU: um backend fora da árvore com suporte a carregamento automático { #torchtpu-an-out-of-tree-backend-with-autoload-support }

Seguindo a
[prática recomendada atual](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html),
em que backends adicionais são criados fora da árvore e registrados em tempo de
execução, estendemos o dispatcher do PyTorch com um novo backend em C++,
registrado como `PrivateUse1`. Além disso, usamos o mecanismo padrão de
`at::Tensor` para definir uma implementação de armazenamento de tensores
específica do dispositivo (usando o `c10::UniqueVoidPtr` dentro de
`c10::StorageImpl`).

Projetamos o TorchTPU pensando na usabilidade. Quando o usuário importa o
PyTorch em uma máquina com TPU em que o wheel do TorchTPU está instalado, o
Python carrega automaticamente o wheel do TorchTPU. Esse carregamento automático
configura o backend e substitui o nome `PrivateUse1` por `tpu`. Assim, em um
exemplo simples, um programa TorchTPU não é diferente dos backends que estão na
árvore, e um programa PyTorch pode ser executado na TPU com pouca ou nenhuma
mudança:

```python
import torch
import torch.distributed as dist

def main():
    device = torch.device("tpu")
    dist.init_process_group(backend="tpu_dist")
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

Como a TPU se torna o acelerador atual e `tpu_dist` é o backend padrão do
TorchTPU, não é necessário especificá-los de forma explícita:

```python
import torch
import torch.distributed as dist

def main():
    device = torch.accelerator.current_accelerator()
    dist.init_process_group()
    rank = dist.get_rank()
    word_size = dist.get_world_size()
    ...
    model = model.to(device)
    ...
```

## Como capturar o grafo { #capturing-the-graph }

O TorchTPU rastreia operações eager no nível do ATen, usando os hooks de
despacho de operações do PyTorch que fazem parte da infraestrutura de backends.
As operações do TorchTPU são criadas como programas StableHLO e enviadas ao XLA
uma de cada vez ou em grupos, o que permite que o compilador faça uma fusão
significativa de operações para aproveitar melhor o hardware. Quando o TorchTPU
recebe uma invocação de callback do dispatcher do PyTorch, o runtime:

*   Cria uma instância de operação adiada (Deferred Op) que contém:
    *   Um objeto builder de StableHLO que, quando invocado, retorna uma função
        StableHLO que implementa a semântica de `aten::mul`, como mostrado
        abaixo (observe que qualquer broadcasting implícito e promoção de dtype
        são tratados em outro lugar)

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   Um mapa opcional de parâmetros estáticos da operação (por exemplo, training
    e momentum para `aten::native_batch_norm`) usado no cálculo de uma chave de
    cache;
*   Retorna um novo `at::Tensor` com `c10::StorageImpl` apontando (por meio de
    indireção) para essa instância de operação adiada.

Ao chegar a um ponto de sincronização adequado, que depende do modo de execução,
o TorchTPU:

*   Analisa a região de operações;
*   Determina quais operações materializar em código executável;
*   Cria uma chave de cache de compilação para o grafo;
*   Se ela já existir no cache, busca o executável e o despacha na TPU;
*   Se ela não existir no cache, compila o objeto de forma assíncrona, adiciona
    ao cache e despacha para execução;
*   Se for o caso para o ponto de sincronização alcançado, pode esperar o valor
    ficar pronto e copiá-lo de volta.

Mais detalhes sobre os vários modos de execução estão em
[Modelo de execução do TorchTPU](./torchtpu_execution_model.md).

## O cache de compilação do TorchTPU { #the-torchtpu-compilation-cache }

Para evitar compilações desnecessárias, o TorchTPU usa um cache de compilação
em vários níveis:

*   **Nível 1:** cache em memória local do processo para compartilhar
    compilações dentro do mesmo processo. É o mais rápido dos três níveis e está
    sempre ativado.
*   **Nível 2:** memória compartilhada local da máquina que permite o
    compartilhamento entre processos no mesmo host. É útil para
    treinamento/inferência com vários dispositivos. O nível 2 vem ativado por
    padrão em programas distribuídos, mas fica desativado em programas com um
    único dispositivo, porque não traz benefício nesse caso.
*   **Nível 3:** cache persistente armazenado em um sistema de arquivos remoto
    (por exemplo, o Google Cloud Storage), que permite a reutilização entre
    hosts, programas e reinicializações de jobs. É útil ao executar programas de
    TPU em ambientes como o Colab e data centers. Ele vem desativado por padrão,
    mas pode ser ativado com a especificação de um local de armazenamento.

Cada entrada é um executável de TPU compilado que representa um grafo de
computação StableHLO gerado a partir de uma operação ATen ou de uma sequência
de operações ATen. Os grafos StableHLO são representados por fingerprints que
combinam a estrutura do grafo, os formatos dos tensores de entrada/saída e as
opções do compilador, incluindo as versões do hardware da TPU, da libtpu e do
runtime PjRt.

## Suporte a `torch.compile` { #torchcompile-support }

Para manter uma experiência do usuário consistente com outros hardwares
compatíveis com `torch.compile()`, registramos um novo
[backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)
do TorchDynamo, que pode ser usado assim: `compiled_model = torch.compile(model,
backend="tpu")`.

O Dynamo invoca o backend do TorchTPU passando um `torch.fx.GraphModule` e uma
lista de tensores de entrada de exemplo. Nesse momento, o runtime rastreia o
grafo recebido como se estivesse no modo eager, mas com todas as operações
adiadas e sem nenhuma execução na TPU. O runtime gera e compila um único
programa StableHLO para todo o grafo FX e retorna um callable Python que
encapsula a invocação. Observe que o TorchTPU não usa `TorchInductor` e, em vez
disso, reutiliza a pilha madura do compilador XLA, que foi desenvolvida junto
com a arquitetura da TPU.

## Mais detalhes { #more-detail }

Nos documentos a seguir, vamos detalhar:

*   O [modelo de memória](./torchtpu_memory_model.md) compatível com a TPU e as
    implicações dele no PyTorch;
*   Os [modos de execução](./torchtpu_execution_model.md) compatíveis com o
    TorchTPU;
*   O suporte a [kernels personalizados](./custom_kernels.md);
*   A [execução distribuída](./distributed_support.md) em clusters de TPUs.
