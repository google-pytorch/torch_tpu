# O TorchTPU e o modelo de memória da TPU { #torchtpu-and-the-tpu-memory-model }

O PyTorch evoluiu junto com as GPUs e coloca todo o poder delas ao alcance dos
usuários de Python. O TorchTPU estende o PyTorch para expressar o poder do XLA
nas TPUs.

O hardware físico das GPUs e das TPUs é diferente, assim como os paradigmas de
programação voltados para cada arquitetura de hardware. O TorchTPU existe para
diminuir essa distância nos dois sentidos. Nosso objetivo é oferecer suporte no
XLA a interfaces nativas de GPU, como views de tensores e streams, e, ao mesmo
tempo, expor o poder exclusivo do compilador XLA em interfaces conhecidas do
PyTorch.

Entender como o TorchTPU alcança esses objetivos pode ajudar os usuários a
otimizar bases de código nativas de TPU para obter o máximo benefício.

## Hardware: CPUs, GPUs e TPUs { #hardware-cpus-gpus-tpus }

Uma CPU de uso geral (como a que alimenta seu computador ou dispositivo móvel)
tem vários núcleos, e cada um oferece suporte a uma ou mais threads. Cada thread
pode ser executada em paralelo e, ao mesmo tempo, acessar a memória
compartilhada com recursos de leitura/escrita de acesso aleatório.

Uma GPU é parecida com uma CPU em muitos aspectos, mas tem muito mais threads.
Uma CPU moderna de alta potência pode ter algumas dezenas de threads, mas uma
GPU de alta potência tem *milhares*. A contrapartida é que cada thread de GPU é,
individualmente, um pouco mais fraca que uma thread de CPU, e há algumas
restrições quanto à independência das threads. Mesmo assim, isso torna as GPUs
úteis para cargas de trabalho com operações altamente paralelizáveis, como
renderização gráfica ou álgebra linear.

Uma TPU tem um design fundamentalmente diferente. O coração da TPU é o
TensorCore e as unidades de multiplicação de matrizes (MXUs) dele. Essas MXUs
operam como um [array sistólico](https://en.wikipedia.org/wiki/Systolic_array),
que é um pouco parecido com um registrador SIMD em uma CPU. Elas usam
registradores largos e bidimensionais que conseguem fazer muito mais trabalho
computacional denso por ciclo de clock do que uma thread de CPU ou de GPU.

No entanto, o TensorCore é fundamentalmente single-threaded. Não é possível
subdividir um array sistólico para fazer várias computações simultâneas. Além
disso, ter registradores largos significa que um TensorCore é muito ruim em
operações esparsas de leitura/escrita com acesso aleatório, porque um grande
bloco de memória precisa ser carregado de uma vez.

As gerações mais recentes de TPU complementam os TensorCores com SparseCores,
que são (como o nome sugere) especializados em operações esparsas, incluindo
operações de leitura/escrita mais precisas. Os núcleos separados também
permitem uma quantidade limitada de paralelismo, já que cada SparseCore pode
executar operações ao mesmo tempo que cada TensorCore.

## Modelo de programação: CUDA e XLA { #programming-model-cuda-vs-xla }

As interfaces de software nativas das GPUs e das TPUs são adaptadas aos
recursos dessas plataformas.

Veja o exemplo do CUDA, o modelo de programação de GPU mais usado como base do
PyTorch. O CUDA é uma interface de programação de baixo nível, parecida com C
ou C++. Os buffers do CUDA são, como a memória da CPU, intrinsecamente arrays de
bytes unidimensionais. Um kernel CUDA é um procedimento imperativo (uma função
void, para usar a terminologia de C). Cada thread recebe ponteiros para alguns
buffers e pode ler de qualquer deslocamento em qualquer buffer e escrever de
volta neles, byte por byte.

Esse modelo de programação não funciona nas TPUs. A memória das TPUs não é
alinhada como arrays de bytes unidimensionais, mas como tiles bidimensionais
alinhados a registradores. E, como mencionado, os TensorCores
não têm threads e são ruins em acesso aleatório.

Em vez disso, o compilador da TPU, o XLA, usa um modelo declarativo e funcional.
O XLA aceita programas escritos em uma linguagem chamada Multi-Level
Intermediate Representation (MLIR), que contém vários dialetos. Com o dialeto
StableHLO, é possível descrever a saída lógica de tensor desejada de um
programa. O XLA compila esse programa em operações internas de nível mais baixo
otimizadas para a sua geração específica de TPU.

## Buffers: `torch.UntypedStorage` { #buffers-torchuntypedstorage }

As ferramentas principais do PyTorch são muito alinhadas ao modelo de
programação do CUDA.

Um objeto de armazenamento, que usa a classe Python `torch.UntypedStorage`, é
pouco mais que um endereço de memória (`torch.UntypedStorage.data_ptr()`) e um
tamanho em bytes (`torch.UntypedStorage.nbytes()`). Isso corresponde bem às
interfaces da CPU e do CUDA.

No entanto, isso não é suficiente para a execução na TPU. Um buffer no XLA
sempre tem um tipo bem definido e um formato multidimensional conhecido. Como a
memória interna da TPU é alinhada a registradores 2D, é fundamental saber como
um buffer está organizado. Isso significa que um `torch.UntypedStorage`
sozinho não traz informações suficientes para a execução na TPU.

Para resolver esse problema, o TorchTPU adiciona uma camada extra. Na TPU, um
`torch.UntypedStorage.data_ptr` não aponta diretamente para a memória da TPU.
Em vez disso, ele aponta para um local na memória da CPU que contém metadados
adicionais sobre o buffer da TPU. **Essa é a chave do funcionamento do
TorchTPU**: esses metadados adicionais permitem uma interpretação ativa entre a
API existente do PyTorch, parecida com a do CUDA, e a realidade do design
funcional do XLA e do hardware da TPU.

## `torch.Tensor` { #torchtensor }
Um `torch.Tensor` pode ser visto como duas coisas ao mesmo tempo.

Primeiro, um `torch.Tensor` é uma representação *física* de um padrão de acesso
a um `torch.UntypedStorage` subjacente. Um `Tensor` contém metadados essenciais
sobre os strides (fatores multiplicativos), os tamanhos (limites dos índices), o
deslocamento (local do elemento 0) e o tipo de elemento (define os bytes por
elemento lógico). Esses metadados fornecem uma forma de mapear uma tupla de
índices em um tensor para um deslocamento em bytes no array de buffer linear 1D
que armazena esse `Tensor`.

Segundo, um `torch.Tensor` é uma representação *lógica* de um objeto matemático.
Quando você faz a multiplicação de matrizes de dois tensores, o resultado é
outro `Tensor`, com um formato adequado, que representa o resultado. O padrão exato de
strides e o deslocamento no armazenamento têm pouca importância aqui. O que o
`Tensor` significa como computação é mais relevante do que o que o tensor é como
referência de memória.

Ao migrar do CUDA para a TPU, é fundamental entender essa diferença: o
comportamento **lógico** das operações do PyTorch em Tensors pode ser executado
corretamente no XLA/TPU, mesmo que a realidade **física** do dispositivo seja
intrinsecamente diferente!

## Decomposição de views { #view-decomposition }

Na execução no CUDA, a linguagem de strides de um `torch.Tensor` é usada
diretamente pelos kernels CUDA internos. Os strides são fornecidos aos
procedimentos internos das threads para descrever como calcular o deslocamento
no buffer relevante. Esses deslocamentos são calculados dinamicamente em tempo
de execução.

Mas, na TPU, os strides físicos não podem ser usados diretamente. Uma descrição
de strides unidimensional não serve para uma memória bidimensional alinhada a
registradores. No entanto, o comportamento lógico das views *tem* suporte por
meio do StableHLO. A maioria das operações de view de Tensor tem um equivalente
direto no StableHLO: reshape, transposição, slicing e broadcasting.

Quando recebe um padrão de strides de um `torch.Tensor`, o TorchTPU executa uma
operação chamada *decomposição de views*. Ela interpreta um padrão de strides e
um tipo de elemento como uma sequência de operações lógicas que produz o Tensor
de saída desejado, com cada elemento no lugar certo.

Por exemplo:

```python
# Logical values: [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
# sizes=(10,), strides=(1,), storage_offset=0
base = torch.arange(10, dtype=torch.float)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
physical_view = base.as_strided(
  sizes=(2, 2, 2),
  strides=(2, 3, 0),
  storage_offset=1
)

# Logical values: [[[1., 1.], [4., 4.]], [[3., 3.], [6., 6.]]]
# StableHLO: slice, reshape, slice, transpose, reshape, broadcast_in_dim
logical_view = base[1:].view(3, 3)[:2, ::2].t().unsqueeze(-1).expand(2, 2, 2)
```

A versão de manipulação de strides de baixo nível, que usa
`torch.Tensor.as_strided`, pode ser expressa de forma equivalente com uma
sequência de operações mais conhecidas, que então podem ser expressas em
StableHLO e compiladas pelo XLA.

No entanto, é importante observar que **equivalência lógica não significa
desempenho equivalente**. Em alguns casos, uma operação de reshape ou de
transposição pode exigir uma cópia completa na TPU para mapear de um layout
alinhado a registradores para outro. Isso significa que, no TorchTPU, **as
views não são "gratuitas"**, como costumam ser no CUDA, por causa das
diferenças físicas entre o hardware da GPU e o da TPU.

## Mutações in-place e doação de buffers { #in-place-mutations-and-buffer-donation }

O StableHLO é uma linguagem de programação funcional. Mais especificamente, é
uma
[linguagem de atribuição única estática (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form).
Isso significa que os valores só podem ser criados do zero, e nunca alterados
ou reatribuídos. Isso ajuda o compilador XLA a produzir planos de execução
otimizados.

Nem as IRs de GPU nem as de CPU são puramente funcionais e, como consequência,
o PyTorch também não é. As operações de tensor com sufixo de sublinhado, ou que
usam o argumento de palavra-chave "`out=`", sobrescrevem os bytes no
`torch.UntypedStorage` base. Como vários `torch.Tensors` podem compartilhar um
buffer de armazenamento em comum, uma escrita em qualquer tensor fica
imediatamente visível para todos os outros Tensors.

O TorchTPU lida com isso aproveitando nossa camada de metadados. Quando uma
operação do PyTorch pede uma atualização in-place ou uma atribuição a um tensor
de saída, o TorchTPU:

*   Descreve a atualização lógica para o XLA, produzindo um novo buffer
*   Troca o buffer dentro do objeto de metadados apontado pelo
    `torch.UntypedStorage`

Isso é, essencialmente, uma funcionalização automática: todas as mutações são
expressas como operações out-of-place.

Para conseguir a funcionalização automática, é preciso executar o algoritmo de
decomposição de views ao contrário, o que chamamos de inversão de views. Isso só
é possível em views sem sobreposição. Escrever em um tensor resultante de um
broadcast é inválido, porque vários locais lógicos correspondem a um único
local físico, o que torna a instrução ambígua. No backend CUDA, a operação seria
considerada comportamento indefinido, mas pode ou não falhar de forma
explícita. No TorchTPU, ela sempre falha de forma explícita.

No caso de uma escrita parcial, a inversão de views usa uma operação do
StableHLO chamada
[`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice)
para calcular um tensor de saída mesclado, que contém tanto os valores
sobrescritos quanto os não sobrescritos nos locais corretos.

Claro que uma operação out-of-place dobra o pico de memória: as versões antiga
e nova coexistem por um breve período antes de o valor antigo ser descartado.
Para evitar isso, o TorchTPU tenta, sempre que possível, aplicar o equivalente
mais próximo que o XLA tem de mutações in-place: a **doação de buffers**. Quando
a última referência a um buffer obsoleto é removida, o TorchTPU anota o
StableHLO para "doar" a entrada, o que permite que o XLA consuma a entrada e
evita o aumento de memória causado pela funcionalização automática.

Nem sempre é possível fazer a doação de buffers com segurança. Se o TorchTPU
não conseguir provar que é seguro doar um buffer, ele volta a usar uma
atualização out-of-place. Isso pode ser especialmente caro em escritas pequenas
em buffers grandes. Nesse caso, a atualização out-of-place copia todo o tensor
de armazenamento base, e não apenas a parte que está sendo escrita!

## Copy-on-write { #copy-on-write }

No entanto, a funcionalização tem vantagens. Uma delas é a possibilidade de
adotar o comportamento de copy-on-write (cópia na escrita).

No backend CUDA, chamar `torch.Tensor.clone()` aloca imediatamente um novo
buffer de memória física grande o suficiente para conter os dados do Tensor.
Depois disso, qualquer escrita in-place nos diferentes
`torch.UntypedStorages` é totalmente independente, mas o custo de memória é
pago imediatamente.

Já no backend do TorchTPU, a imutabilidade de um valor atribuído
estaticamente significa que uma cópia pode ser simplesmente uma referência ao
original, sem precisar de uma cópia física imediata.

Assim que há uma atualização in-place, precisamos refletir essa atualização
apenas no `torch.UntypedStorage` que foi modificado, e não no outro. Isso impede
a doação de buffers (já que ainda precisamos dos valores antigo e novo), então,
nesse caso, a atualização in-place é feita out-of-place, sem doação. Essa
atualização out-of-place aloca memória nova, mas essa memória só é alocada de
forma lazy, conforme a necessidade, ao contrário do PyTorch/CUDA, que paga o
custo de memória antecipadamente.

## Kernels personalizados { #custom-kernels }

O TorchTPU faz o possível para aplicar doações de buffers que otimizam a
memória e oferece uma transformação logicamente equivalente de views com
strides do CUDA para operações de view do StableHLO. Ainda assim, essas são
operações de alto nível, não ligadas diretamente ao layout físico de um buffer
da TPU, e não garantem nenhuma semântica específica de baixo nível.

Para usuários que querem o máximo controle de baixo nível, o TorchTPU também
oferece suporte à integração de kernels personalizados. Eles permitem substituir
com precisão operações críticas, ou até reescrever o modelo inteiro, com a
especificação manual do comportamento dos buffers da TPU. Isso pode ser usado
para garantir o comportamento ideal de doação e de atualização in-place.

## Principais conclusões { #takeaways }
Ao trabalhar com o TorchTPU, os usuários precisam saber os seguintes fatos
principais:

*   **O código PyTorch produz o mesmo resultado lógico no CUDA e no
    TorchTPU:** você pode usar operações conhecidas de baixo nível do CUDA
    (views com armazenamento compartilhado, manipulação de strides,
    atualizações in-place) e obter os mesmos resultados na TPU, embora o
    desempenho possa não ser o mesmo.
*   **Cópias são baratas:** copiar um tensor para uma forma contígua costuma
    ter custo computacional zero, por causa do comportamento de copy-on-write.
*   **Views são caras:** ler um tensor com um formato diferente ou com um padrão
    não contíguo pode forçar uma reorganização do layout (relayout) na TPU, o
    que pode ser caro. Prefira tensores contíguos sempre que possível.
*   **Sobrescrever um tensor contíguo costuma ser eficiente em memória:** um
    tensor contíguo geralmente pode ser sobrescrito com uma doação de buffer se
    todos os valores antigos forem apagados.
*   **Escritas parciais às vezes são caras em memória:** se a doação de buffer
    não for possível, o custo de memória de uma escrita é proporcional ao
    tamanho do torch.UntypedStorage, e não ao tamanho dos dados escritos que
    estão sendo modificados. Prefira usar muitos tensores pequenos em vez de
    fatias de tensores grandes.
*   **Kernels personalizados dão controle total:** quando a tradução do CUDA
    para o XLA falhar ou não for suficiente, implementar um kernel
    personalizado nativo de TPU permite otimizar para um desempenho específico
    de baixo nível.
