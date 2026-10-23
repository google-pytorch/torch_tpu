# Modelo de execução do TorchTPU { #torchtpu-execution-model }

Quando você executa um programa PyTorch com o TorchTPU, muitas coisas acontecem
internamente para otimizar a execução. Entender esse modelo ajuda você a saber
como ajustar sua experiência e aproveitar alguns pontos fortes exclusivos do
TorchTPU.

## Pipeline de execução { #execution-pipeline }

### Criar a operação adiada { #create-deferred-op }

Quando uma operação do PyTorch (ou apenas "op") é executada no backend do
TorchTPU, inicialmente nenhum trabalho é feito na própria TPU.

Em vez disso, o resultado lógico da computação é empacotado em uma operação
adiada (deferred op). Ela é um objeto na memória do lado da CPU que contém
todas as informações necessárias para executar essa operação: as entradas, as
etapas lógicas a seguir, o formato e o tipo das saídas, entre outras coisas.

O restante do pipeline, que substitui essa operação adiada simbólica por dados
reais, se chama materialização.

### Enfileirar a operação em um stream { #enqueue-op-on-a-stream }

O TorchTPU é compatível com a interface `torch.Stream`. Sempre que uma
DeferredOp é criada, ela é enviada ao Stream ativo atual para ficar na fila e
ser executada depois.

Nesse ponto, a thread que fez a chamada termina o trabalho. Tudo o que vem
depois acontece de forma assíncrona em segundo plano (exceto no modo de launch
blocking, que será explicado mais adiante).

### Analisar o stream e preparar o plano de execução { #analyze-stream-and-prepare-execution-plan }

Quando há trabalho a ser feito em um stream, em algum momento uma thread em
segundo plano pega esse trabalho do stream. Essa thread em segundo plano então
analisa em conjunto o trabalho no stream, considerando vários fatores
diferentes, como:

*   Em quantas execuções individuais isso precisa ser dividido?
*   Quais são as entradas e saídas de cada bloco de execução?
*   Que nível de otimização de compilação deve ser aplicado a cada um?
*   Já vimos cada unidade de computação antes ou ela é nova?
*   Em qual núcleo (TensorCore ou SparseCore) esse trabalho deve ser executado?

Ao considerar o trabalho a ser feito como um todo, a thread em segundo plano
pode tomar decisões de execução melhores.

Depois que a thread em segundo plano identifica as saídas finais de cada bloco,
ela marca essas saídas como pendentes, para indicar que elas serão executadas no
futuro, e segue para a próxima etapa.

### Compilar (ou buscar) os executáveis { #compile-or-fetch-executables }

Para cada estágio de execução, a thread em segundo plano consulta um cache de
compilação para ver se um executável já foi compilado com as configurações
corretas. Por padrão, esse é um cache na memória local (nível 1), com opções
para ativar um cache persistente local do dispositivo (nível 2) ou caches
remotos (nível 3) para minimizar os custos de recompilação.

Quando há um cache miss, o grafo de compilação da unidade é convertido em MLIR
(principalmente StableHLO, exceto em kernels personalizados definidos pelo
usuário) e entregue a um threadpool de compilação. Esse threadpool compila de
forma assíncrona muitas unidades de compilação em paralelo e as devolve à thread
de trabalho em segundo plano quando ficam prontas.

### Iniciar a execução { #start-execution }

Depois que as entradas são coletadas e o executável é preparado, uma tarefa de
execução é enfileirada no runtime do XLA, o
[PjRt](https://openxla.org/xla/pjrt).

O PjRt retorna promessas para essa saída futura, que são então conectadas às
saídas para indicar que elas estão sendo materializadas.

### (Opcional) Esperar a conclusão { #optional-wait-for-completion }

Para saber quando o trabalho termina, você pode registrar um `torch.Event` no
stream e usar o método `torch.Event.query` para inspecioná-lo.

Como alternativa ou além disso, você pode bloquear o stream com qualquer
instrução de "synchronize" (`torch.Event.synchronize`,
`torch.Stream.synchronize` ou `torch.accelerator.synchronize` para o
dispositivo inteiro), o que pausa seu código Python até a TPU sinalizar a
conclusão.

## Modos de execução { #execution-modes }

O pipeline de programação do TorchTPU é flexível e oferece suporte a várias
modalidades de execução diferentes.

### Modo eager estrito { #strict-eager-mode }

Também conhecido como "eager padrão" ou "defer never", esse modo é a
experiência conhecida do PyTorch, pronta para uso. Nesse modo, cada operação
adicionada à fila marca imediatamente o stream para execução. Com isso, as
operações são processadas uma de cada vez pela thread de trabalho de
materialização em segundo plano.

Esses grafos de execução com uma única operação têm potencial mínimo de
otimização e, por isso, são compilados com o nível de compilação mais baixo,
`-O1`, para maximizar a velocidade de compilação. Como há muitas compilações
individuais, elas são paralelizadas de forma muito eficaz no threadpool de
compilação, e o nível de reutilização de operações individuais é muito alto.
Isso gera uma sobrecarga mínima de compilação durante o "aquecimento" do cache de
compilação.

Observe que, embora esse modo execute de forma eager e contínua, ele ainda faz
isso de forma assíncrona. Ainda é necessário sincronizar para esperar a TPU
terminar se você quiser ver os resultados, e qualquer erro de runtime pode
aparecer muito depois da operação que o causou.

### Modo eager de depuração { #debug-eager-mode }

Também conhecido como "launch blocking", esse modo é ativado com a variável de
ambiente `TORCH_TPU_LAUNCH_BLOCKING=1`. Ele é destinado apenas à depuração
interativa, e não ao uso em produção.

Esse modo é igual ao eager estrito (execução operação por operação), exceto
que o host bloqueia e sincroniza a cada operação. Isso deixa a execução
artificialmente mais lenta, mas garante que as falhas sejam informadas assim
que acontecem, o que ajuda a localizar os erros.

### `torch.compile` { #torchcompile }

Para ter o desempenho máximo, o `torch.compile` oferece tanto um potencial de
otimização maior quanto uma sobrecarga de runtime menor.

Ao compilar uma função ou um módulo específico com `torch.compile`, o backend
do TorchTPU recebe todo o escopo do alvo da compilação como um
[grafo FX](https://docs.pytorch.org/docs/stable/fx.html). Isso fornece o máximo
de informações sobre a execução pretendida, com a garantia de que ela não vai
mudar, o que nos permite fazer otimizações agressivas que seriam arriscadas
demais no modo eager.

Como backend de modo compilado, o TorchTPU faz um rastreamento inicial (usando
tensores "placeholder", semelhantes ao FakeTensor, mas que mantêm metadados
específicos da TPU) e produz um grafo de computação totalmente adiado que
replica o grafo FX. Depois que ele é criado, entregamos esse programa StableHLO
ao XLA para compilação com o nível de otimização mais alto, `-O2`. Esse
programa é então empacotado em um callable Python e devolvido ao usuário para
execução posterior.

Quando esse callable é invocado, pulamos imediatamente a maior parte das etapas
da execução eager. Nenhuma DeferredOp nova é criada, nenhuma análise adicional
do grafo é necessária e o executável já está disponível, então não é preciso
usar cache. Em vez disso, o executável é enviado imediatamente ao PjRt para
execução, o que minimiza o custo do lado do host com um grafo de computação
totalmente otimizado pelo compilador XLA.

### Modo eager otimizado { #optimized-eager-mode }

Também conhecido como modo "Defer and Fuse", ele é ativado com
`TORCH_TPU_DEFER_AND_FUSE=1`. Esse é um modo híbrido e novo, que traz muitas
das vantagens do `torch.compile` para o modo eager.

Nesse modo, criar uma nova DeferredOp não inicia a execução imediatamente.
Deixamos as operações adiadas entre chamadas Python, esperando na fila do
Stream.

A materialização começa quando uma destas duas condições é atendida:

*   O usuário faz alguma ação que força a execução. As mais comuns são iniciar
    uma transferência do dispositivo para o host ou uma chamada explícita de
    sincronização.
*   Detectamos uma repetição nas operações despachadas, o que indica que
    estamos em um loop e devemos compilar uma iteração desse loop para
    reutilizá-la com facilidade.

Isso sinaliza uma materialização em segundo plano, semelhante ao modo eager
padrão, exceto que o grafo compilado é maior que uma única operação. Os tensores
que já tiverem sido descartados ou sobrescritos quando o sinal de
materialização for gerado são ignorados. Normalmente, eles são fundidos com
as operações seguintes ou totalmente eliminados se a saída for descartada sem
ser lida. (Observação: a eliminação de código morto é ignorada em operações
coletivas distribuídas para evitar deadlocks entre TPUs).

Como isso gera grafos de compilação maiores, usamos o nível de otimização 2
(`-O2`) para melhorar o tempo de execução. Como compilamos um número menor de
grafos maiores, o threadpool de compilação traz menos benefício de
paralelização, então os tempos de compilação tendem a ser mais longos que no
eager estrito.

Embora não seja tão otimizado quanto o modo `torch.compile`, o modo eager
otimizado tem uma melhoria de desempenho de até 50% em relação ao eager padrão,
sem nenhuma mudança no código do modelo.

## Armadilhas { #pitfalls }

Esse pipeline de execução é otimizado tanto para latência quanto para taxa de
transferência, mas há casos em que o desempenho pode ser prejudicado.

### Todos os modos: fluxo de controle no Python e formatos dependentes de dados { #all-modes-python-side-control-flow-and-data-dependent-shapes }

No PyTorch, é fácil forçar sem querer, e sem nenhum aviso, uma transferência do
dispositivo para o host. Uma operação como "`if my_tensor.all()`" cria
silenciosamente um tensor de valor booleano, sincroniza o stream e copia esse
booleano para a CPU antes de continuar com o fluxo de controle.

No modo eager padrão, isso não força novas quebras de grafo, mas força o
esvaziamento de todo o pipeline de execução do stream antes de continuar. Pior
ainda: no modo eager otimizado ou no modo compilado, esse ponto de
sincronização forçado causa uma quebra de grafo no stream fundido de forma
eager ou no grafo FX, criando uma barreira à otimização.

Um comportamento parecido também pode acontecer com operações do PyTorch que
têm formatos de saída dependentes de dados, como `torch.masked_select`. Como o
tamanho do tensor de saída só é conhecido depois que a entrada de máscara é
avaliada, isso também força uma sincronização totalmente bloqueante do stream e
uma transferência dos metadados de formato do dispositivo para o host antes de
continuar.

### Modo eager estrito: executáveis minúsculos causam microbolhas { #strict-eager-mode-tiny-executables-cause-microbubbles }

Enfileirar um executável no PjRt leva um pouco de tempo, normalmente 50
microssegundos ou menos. Na maioria dos casos, cada executável leva mais tempo
que isso (mesmo que apenas 1 ou 2 milissegundos) e esconde totalmente essa
latência, o que leva a um estado estável em que a TPU opera continuamente com a
eficiência máxima.

No entanto, se forem enfileirados muitos executáveis abaixo desse limite (como
ao usar o modo eager estrito com tensores pequenos), a TPU pode ficar sem
trabalho. Quando a TPU executa mais rápido do que a CPU consegue enfileirar
trabalho, a execução na TPU é interrompida por "microbolhas" minúsculas, mas
frequentes, porque fica limitada pela velocidade de despacho.

Para evitar isso, garanta que seus modelos façam trabalho suficiente em cada
operação para compensar essa pequena sobrecarga de inicialização ou use o modo
eager otimizado para fundir automaticamente as execuções em blocos de compilação
de forma dinâmica.

### Modo compilado: views como argumentos ou saídas { #compiled-mode-views-as-arguments-or-outputs }

Ao chamar um executável compilado, o PjRt exige que os valores de entrada sejam
contíguos, tenham deslocamento zero e sejam de um tipo de dados específico. Se
um argumento Tensor for uma view não trivial ou tiver passado por bitcast a
partir do tipo de dados original, o TorchTPU vai fazer uma cópia temporária do
tensor no modo eager (estrito ou otimizado) para chamar corretamente o
executável compilado.

Para evitar essa cópia temporária, garanta que todas as entradas de uma função
compilada sejam contíguas. Se você planeja chamar várias funções compiladas em
sequência, garanta que as saídas das funções anteriores sejam contíguas para
evitar atrito do modo eager entre as funções otimizadas.

### Modo eager otimizado: retenção de tensores redundantes { #optimized-eager-mode-retaining-redundant-tensors }

Em todos os modos eager, as operações são executadas na ordem em que são
criadas e, sempre que um sinal de materialização é gerado, todos os Tensors
vivos naquele momento são materializados.

Se algum desses tensores for um tensor intermediário sem importância, isso pode
fazer o mecanismo de defer-and-fuse identificá-lo por engano como uma saída
necessária e materializá-lo, o que pode reduzir a eficiência do grafo de
compilação.

Para melhorar a fusão e minimizar o uso de memória, libere os tensores assim
que eles não forem mais necessários ou use operações in-place para substituir
tensores obsoletos pelos valores atuais.

## Principais conclusões { #takeaways }

O modelo de execução adiada do TorchTPU oferece um pipeline eficiente com
vários equilíbrios entre tempo de compilação e tempo de execução.

*   **Para o menor tempo de compilação:** use o modo eager estrito (o padrão).
    Esse modo inicia a execução imediatamente e paraleliza a compilação para
    chegar ao primeiro resultado o mais rápido possível.
*   **Para a maior taxa de transferência depois do aquecimento e a menor
    latência:** use `torch.compile`. Ele elimina quase toda a sobrecarga do
    modo eager e maximiza o potencial de otimização do compilador.
*   **Para casos em que você prefere uma execução mais rápida, mas
    `torch.compile` não é uma opção:** considere usar o modo eager otimizado.
*   **Para depuração:** use o modo de launch blocking para forçar a CPU e a TPU
    a avançar em sincronia.
*   **Evite sincronizações forçadas:** mantenha um conjunto mínimo de tensores
    contíguos e use operações densas em computação para manter a TPU sempre
    abastecida com trabalho produtivo.
