# Formatos dinâmicos { #dynamic-shapes }

<!-- Link to colab and to generated notebook -->

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/dynamic_shapes.ipynb)

Quando você compila um modelo com `torch.compile(model)`, o TorchTPU converte
(lowering) a computação para StableHLO e a compila com o compilador XLA
(consulte [Como o TorchTPU funciona](./how_torchtpu_works.md)).
Tradicionalmente, o compilador XLA é projetado com base em **formatos (shapes)
de tensores conhecidos estaticamente**, para que os buffers da memória de alta
largura de banda (HBM) possam ser organizados com antecedência, os pipelines de
operadores possam ser fundidos e as multiplicações de matrizes possam ser
divididas em tiles para os arrays sistólicos (MXUs) da TPU. Quando as dimensões
dos tensores de entrada variam entre chamadas em uma carga de trabalho com
`torch.compile`, compilar para cada formato exato invalidaria as guardas de
formato do Dynamo e acionaria uma recompilação completa e cara do XLA a cada
novo formato.

Para oferecer suporte a **formatos dinâmicos** sem recompilações constantes, o
TorchTPU usa o **dinamismo limitado** (bounded dynamism), em que os limites
superiores das dimensões dinâmicas são fixados no momento da compilação,
enquanto o XLA acompanha dinamicamente os tamanhos reais das dimensões em tempo
de execução.

## Como o TorchTPU lida com formatos dinâmicos { #how-dynamic-shapes-are-handled-by-torchtpu }

Quando você compila um modelo ou uma função com o dinamismo padrão, ou seja,
`torch.compile(model, dynamic=None)` (conforme os formatos de entrada variam
entre chamadas), ou marca dimensões dinâmicas de forma explícita com
`torch._dynamo.mark_dynamic`, o Dynamo rastreia a computação e produz um grafo
FX em que as dimensões que mudam são representadas como variáveis inteiras
simbólicas (`torch.SymInt`) em vez de constantes fixas. O TorchTPU trata essas
dimensões simbólicas da seguinte forma:

### Compilação { #compilation }

*   O TorchTPU intercepta o grafo FX rastreado, determina um intervalo geral
    `[lower_bound, upper_bound]` para cada dimensão simbólica (a partir de
    `mark_dynamic(..., min=..., max=...)` ou inferido pelo Dynamo a partir do
    tamanho inicial `S` em tempo de execução) e divide esse intervalo em buckets
    contíguos alinhados a múltiplos de `bucket_size` (padrão `128`, igual à
    dimensão de tile de 128 lanes da TPU), até `num_dynamic_buckets` (padrão
    `1`, máximo `8`):
    *   O primeiro bucket vai de `lower_bound` até o próximo múltiplo de
        `128` igual ou maior que `S` (`U_1`), o segundo bucket vai de `U_1 + 1`
        até o próximo múltiplo de `128` (`U_2`), e assim por diante (por
        exemplo, `[lower_bound, 128]`, `[129, 256]`, ...).
    *   O TorchTPU registra no Dynamo uma única guarda de intervalo que cobre
        toda a extensão de todos os buckets, de modo que o Dynamo só aciona a
        recompilação se a dimensão crescer além do `upper_bound` geral.
*   Para cada bucket, o TorchTPU substitui a dimensão simbólica pelo limite
    superior estático `U` do bucket, passa o tamanho real da dimensão em tempo de
    execução como entrada e converte o grafo para StableHLO com tipos de tensor
    limitados que acompanham tanto o limite superior estático `U` quanto o
    tamanho ativo em tempo de execução em todas as operações.
*   O XLA compila o módulo resultante para o limite superior `U` de cada bucket
    configurado.

### Execução { #execution }

*   Em tempo de execução, com base no tamanho real da dimensão dinâmica, o
    TorchTPU seleciona o executável compilado do bucket correspondente.

*   Ao executar o módulo compilado, o XLA preenche (padding) o tensor de
    entrada dinâmico até o limite superior estático `U` do bucket e redefine os
    valores na região preenchida, além do tamanho ativo em tempo de execução,
    para valores de identidade algébrica. Isso garante que os elementos de
    preenchimento não afetem o resultado.

*   Em tensores de saída dinâmicos, o preenchimento fica no fim do buffer, o
    que permite que o TorchTPU devolva o buffer ao PyTorch com uma atualização
    leve dos metadados de formato, em vez de fatiar fisicamente o tensor.

## Caso de uso motivador: decodificação autorregressiva { #motivating-use-case-autoregressive-decoding }

Uma das principais cargas de trabalho que motivam o dinamismo limitado é a
**decodificação autorregressiva de LLMs** com um KV cache que cresce
dinamicamente. Em um loop de decodificação padrão, cada etapa anexa os estados
de chave e valor do novo token ao histórico em cache ao longo da dimensão de
sequência com `torch.cat`:

*   **Prefill:** o prompt de entrada tem o formato `[B, S]` (tamanho de batch
    `B`, comprimento de sequência do prompt `S`) e produz um KV cache com o
    formato `[B, H_kv, S, D]` (`H_kv` cabeças de chave/valor, dimensão de cabeça
    `D`).
*   **Etapa de decodificação 1:** o token de entrada tem o formato `[B, 1]` e é
    concatenado ao cache, produzindo o formato `[B, H_kv, S + 1, D]`.
*   **Etapa de decodificação `t`:** o comprimento de sequência do KV cache
    cresce para `S + t`.

Com a compilação com formatos estáticos (`dynamic=False`), o Dynamo cria guardas
para o comprimento exato da sequência do KV cache. Cada etapa de decodificação
muda o formato do tensor, invalida a guarda e aciona uma **recompilação completa
do XLA a cada token gerado**. Isso transforma etapas de decodificação de
milissegundos em compilações de vários segundos, a menos que você reescreva o
modelo para pré-alocar buffers estáticos e fazer mutações in-place.

O **dinamismo limitado** resolve isso sem exigir que você reescreva o modelo:
conforme o KV cache cresce entre as etapas de decodificação (`S + 1`, `S + 2`,
...), todos os formatos dentro do bucket compilado (`<= 128`, `<= 256`, ...)
reutilizam o mesmo executável do XLA.

Para um tutorial prático que aplica o dinamismo limitado a um loop completo de
decodificação de LLM (e o compara com a pré-alocação de um KV cache estático),
consulte
[Inferência de LLM de ponta a ponta](./case_studies_muse_glimmer_30b_inference.md).

## Exemplo: loop autorregressivo com um tensor que cresce { #example-autoregressive-loop-with-a-growing-tensor }

Para ver como a compilação estática (`dynamic=False`) e o dinamismo limitado
com o dinamismo padrão (`dynamic=None`) se comportam na prática, considere um
loop autorregressivo mínimo em que cada etapa anexa um novo token ao longo da
dimensão de sequência (`dim=2`) com `torch.cat`:

```python
import time
import torch
from torch._functorch._aot_autograd import autograd_cache

torch._logging.set_logs()  # pylint: disable=protected-access
device = torch.device("tpu")

def decode_step(
    kv_cache: torch.Tensor, new_token: torch.Tensor
) -> torch.Tensor:
  return torch.cat([kv_cache, new_token], dim=2)

def run_decode_loop(
    compiled_step, num_steps: int = 10, prefix: str = ""
) -> None:
  kv_cache = torch.ones((1, 8, 16, 128), dtype=torch.bfloat16, device=device)
  new_token = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16, device=device)
  torch.accelerator.synchronize()

  for step in range(num_steps):
    start = time.perf_counter()
    kv_cache = compiled_step(kv_cache, new_token)
    torch.accelerator.synchronize()
    step_ms = (time.perf_counter() - start) * 1000
    print(
        f"{prefix}Step {step + 1}/{num_steps} (seq_len={kv_cache.shape[2]}):"
        f" {step_ms:.2f} ms",
        flush=True,
    )
```

### 1. Execução com `dynamic=False` { #1-running-with-dynamicfalse }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()
torch._dynamo.config.recompile_limit = 64

step_static = torch.compile(decode_step, backend="tpu", dynamic=False)
run_decode_loop(step_static, prefix="[dynamic=False] ")
```

Exemplo de saída:

```text
[dynamic=False] Step 1/10 (seq_len=17): 49.54 ms
[dynamic=False] Step 2/10 (seq_len=18): 43.51 ms
[dynamic=False] Step 3/10 (seq_len=19): 39.52 ms
[dynamic=False] Step 4/10 (seq_len=20): 38.63 ms
[dynamic=False] Step 5/10 (seq_len=21): 38.40 ms
[dynamic=False] Step 6/10 (seq_len=22): 38.23 ms
[dynamic=False] Step 7/10 (seq_len=23): 38.11 ms
[dynamic=False] Step 8/10 (seq_len=24): 38.55 ms
[dynamic=False] Step 9/10 (seq_len=25): 37.89 ms
[dynamic=False] Step 10/10 (seq_len=26): 37.75 ms
```

**O que aconteceu:** com `dynamic=False`, o Dynamo cria guardas para o
comprimento exato da sequência de `kv_cache` (`16`, `17`, `18`, ...). Como
`torch.cat` aumenta `dim=2` em `1` a cada iteração, a guarda de formato falha em
todas as chamadas e aciona uma recompilação completa do XLA em todas as etapas.

### 2. Execução com `dynamic=None` (dinamismo limitado) { #2-running-with-dynamicnone-bounded-dynamism }

```python
torch._dynamo.reset()
autograd_cache.AOTAutogradCache.clear()

step_dynamic = torch.compile(decode_step, backend="tpu", dynamic=None)
run_decode_loop(step_dynamic, prefix="[dynamic=None] ")
```

Exemplo de saída:

```text
[dynamic=None] Step 1/10 (seq_len=17): 49.59 ms
[dynamic=None] Step 2/10 (seq_len=18): 206.68 ms
[dynamic=None] Step 3/10 (seq_len=19): 64.29 ms
[dynamic=None] Step 4/10 (seq_len=20): 0.51 ms
[dynamic=None] Step 5/10 (seq_len=21): 0.43 ms
[dynamic=None] Step 6/10 (seq_len=22): 0.38 ms
[dynamic=None] Step 7/10 (seq_len=23): 0.40 ms
[dynamic=None] Step 8/10 (seq_len=24): 0.44 ms
[dynamic=None] Step 9/10 (seq_len=25): 0.45 ms
[dynamic=None] Step 10/10 (seq_len=26): 0.37 ms
```

**O que aconteceu:** com `dynamic=None`, só as 3 primeiras iterações têm
sobrecarga de compilação e, a partir da etapa 4, a execução cai para cerca de
0,4 ms por etapa (~100x mais rápido), sem nenhuma recompilação:

*   **Etapa 1 (`seq_len=16 -> 17`):** o Dynamo vê as entradas pela primeira vez
    e compila um executável estático para `seq_len=16`.
*   **Etapa 2 (`seq_len=17 -> 18`):** como `dim=2` mudou de `16` para `17`, o
    Dynamo marca automaticamente `dim=2` como dinâmica (`SymInt`) e o TorchTPU
    compila um executável com dinamismo limitado para o bucket `[17, 128]`
    usando o layout do tensor de entrada inicial.
*   **Etapa 3 (`seq_len=18 -> 19`):** o buffer de saída com dinamismo limitado
    produzido pela etapa 2 tem o layout de saída preenchido do XLA, então o
    TorchTPU compila o bucket `[17,
    128]` mais uma vez para esse layout estável do XLA.
*   **Etapas 4 a 10 (`seq_len=19 -> 26`):** tanto o comprimento da sequência
    (dentro do bucket `<= 128`) quanto o layout do buffer do XLA correspondem ao
    executável em cache, então cada etapa seguinte despacha diretamente o
    executável do XLA já compilado, sem recompilar.

> **Tip:** se você já sabe com antecedência qual dimensão vai variar, pode
> chamar `torch._dynamo.mark_dynamic(kv_cache, 2, min=16, max=128)` antes da
> primeira etapa. Isso diz ao Dynamo para rastrear `dim=2` como dinâmica logo
> na etapa 1, o que pula a compilação estática inicial da etapa 1 e permite que
> você especifique limites `[min, max]` explícitos.

## Limitações atuais { #current-limitations }

O dinamismo limitado no TorchTPU está em evolução ativa e, no momento, é
otimizado para o caso de uso de **KV cache autorregressivo**. Lembre-se das
seguintes limitações ao aplicá-lo aos seus modelos:

*   **Limite de buckets e dinamismo ilimitado:** no máximo **8 buckets**
    (`num_dynamic_buckets <= 8`) são compilados por rastreamento. O dinamismo
    ilimitado ainda não tem suporte. Se uma dimensão dinâmica crescer além do
    limite superior do maior bucket compilado, uma recompilação é acionada.
*   **Restrições de formato do Dynamo:** os limites superiores dos buckets
    precisam atender às restrições de formato ativas do Dynamo (como requisitos
    de divisibilidade de `reshape` ou `view`). Caso contrário, o TorchTPU usa
    um único bucket com limite superior `2 * S` (o dobro do tamanho inicial `S`
    em tempo de execução).
*   **Sobrecarga de computação e de HBM até o limite superior do bucket:** como a
    execução acontece em buffers preenchidos até o limite superior estático `U`
    do bucket, o uso de HBM e os FLOPs escalam com `U`, e não com o tamanho
    real em tempo de execução. Se você definir um `bucket_size` muito grande ou
    `mark_dynamic(..., max=U)` quando o tamanho real for muito menor que `U`,
    vai pagar o custo de computação e de memória de `U`.
*   **Mudanças de layout do XLA:** mesmo quando o formato de um tensor está dentro
    de um bucket já compilado, uma mudança no layout físico subjacente do XLA
    de um buffer de entrada aciona a recompilação do bucket.
*   **Sem cache de compilação persistente (`AOTAutogradCache`):** os
    executáveis com dinamismo limitado ainda não são serializados no
    `AOTAutogradCache`, então são recompilados a cada novo processo.
*   **Cobertura de operadores:** nem todos os operadores do PyTorch oferecem
    suporte a formatos com dinamismo limitado ainda.
*   **Views como entradas do modelo:** o suporte para passar views de tensores
    diretamente como entradas do modelo compilado é limitado.
*   **Dinamismo dependente de dados:** operações cujos formatos de saída dependem
    dos valores dos tensores em tempo de execução (como `torch.nonzero`) não
    têm suporte.
*   **Kernels personalizados:** formatos com dinamismo limitado não têm
    suporte com kernels de TPU personalizados (como kernels Pallas
    registrados com `torch.tpu.pallas.jax_op`).
*   **dynamic=True:** o modo *dynamic=True* de torch.compile() não tem suporte.
