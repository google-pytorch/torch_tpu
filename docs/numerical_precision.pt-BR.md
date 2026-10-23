# Gerencie a precisão numérica no TorchTPU { #manage-numerical-precision-in-torchtpu }

As TPUs do Google dividem as operações de ponto flutuante entre duas unidades de
hardware para equilibrar a taxa de transferência e a exatidão numérica:

*   **Unidade de multiplicação de matrizes (MXU):** executa operações de
    contração (como `torch.matmul`, `torch.mm`, `torch.bmm` e convoluções) com
    entradas nativas em `bfloat16` (significando de 8 bits) e acumulação em
    `float32` de 32 bits.
*   **Unidade de execução vetorial (VPU):** executa operações sem contração
    (como aritmética elemento a elemento, ativações, normalização de camada e
    softmax) em `float32` IEEE-754 completo (significando de 24 bits).

Este guia mostra como configurar os modos de precisão da multiplicação de
matrizes em `float32`, verificar a paridade numérica entre CPU e TPU, controlar
os tipos do acumulador de saída e o excesso de precisão e validar no código o
comportamento bit a bit da MXU.

### Compare a precisão do PyTorch no CUDA e na TPU { #compare-pytorch-precision-on-cuda-and-tpu }

O TorchTPU é compatível com as APIs de precisão padrão do PyTorch
(`torch.set_float32_matmul_precision` e `torch.get_float32_matmul_precision`),
além de controles específicos da TPU (`torch.tpu.Precision` e
`torch.tpu.precision`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Dimensão           | PyTorch no CUDA                      | PyTorch na TPU (TorchTPU)              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **Modo de precisão padrão** | `"highest"` (`float32` puro) | `"medium"` (`Precision.DEFAULT`)       |
| **Primitiva do acelerador para cálculo rápido** | Tensor Cores (`TF32`) | MXUs sistólicas (`bfloat16`)             |
| **Bits do cálculo rápido** | 1 bit de sinal + 8 de expoente + 10 de mantissa (`TF32` de 19 bits) | 1 bit de sinal + 8 de expoente + 7 de mantissa (`bfloat16` de 16 bits) |
| **Precisão da acumulação** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` em todos os modos |
| **Mecanismo de precisão maior** | Unidades de hardware dedicadas para `float32`/`TF32` | Decomposição em várias passagens (3x ou 6x passagens na MXU) |
| **Impacto de `high`/`highest` no desempenho** | Velocidade de referência ou pequena redução (~1–2x) | Custo de computação na MXU ~3x maior com `"high"` e ~6x com `"highest"` |
| **API de escopo** | `torch.set_float32_matmul_precision` global | `torch.set_float32_matmul_precision` <br> global + `torch.tpu.precision` com escopo |
<!-- mdformat on -->

## Especifique os modos de precisão { #specify-precision-modes }

Ao multiplicar tensores `float32` na MXU, o TorchTPU mapeia os níveis de
precisão do PyTorch para o enum `torch.tpu.Precision`:

*   `"medium"` / `Precision.DEFAULT` (1 passagem): **padrão.** Arredonda as
    entradas `float32` para `bfloat16` (multiplicando os 8 bits mais altos do
    significando) e acumula em `float32`. Recomendado para quase todas as cargas
    de trabalho de treinamento e inferência.
*   `"high"` / `Precision.HIGH` (3 passagens): divide as entradas `float32` em
    componentes `bfloat16` alto e baixo e calcula três termos cruzados
    (multiplicando os primeiros 16 bits do significando), com cerca de 3x o
    custo de computação na MXU.
*   `"highest"` / `Precision.HIGHEST` (6 passagens): divide as entradas
    `float32` em três componentes `bfloat16` e calcula seis termos cruzados
    (multiplicando todos os 24 bits do significando) para se aproximar da
    exatidão completa do `float32` IEEE-754, com cerca de 6x o custo de
    computação na MXU.

Você pode configurar a precisão da multiplicação de matrizes de duas formas:

*   **Configuração global:** chame `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` para definir o modo padrão do processo.
*   **Gerenciador de contexto com escopo:** use `with torch.tpu.precision(mode):`
    para aplicar uma precisão maior apenas a camadas ou submódulos sensíveis do
    ponto de vista numérico.
*   **Aviso de desempenho:** selecionar `"high"` ou `"highest"` emite um aviso
    único (`TORCH_WARN_ONCE`). Mantenha o modo padrão `"medium"`, a menos que
    você observe problemas de convergência relacionados à precisão.

> **Note:** entrar em `torch.tpu.precision` ou sair dele causa uma quebra de
> grafo no `torch.compile`. Ao usar `torch.compile(..., fullgraph=True)`,
> defina a precisão fora da função compilada com
> `torch.set_float32_matmul_precision(...)` ou coloque a chamada externa em `with
> torch.tpu.precision(...):`.

Use `torch.tpu.precision` para comparar o tempo de execução em estado estável
nos três modos:

```python
import time
import torch

device = torch.device("tpu")
precision = torch.tpu.precision  # Context manager
Precision = torch.tpu.Precision  # Enum class

print(f"Default matmul precision: {torch.get_float32_matmul_precision()}")

size = (8, 2048, 2048)
a = torch.randn(size, dtype=torch.float32, device=device)
b = torch.randn(size, dtype=torch.float32, device=device)

def run_matmul():
  res = torch.matmul(a, b)
  torch.accelerator.synchronize()
  return res

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    # Warm up to compile the graph before timing steady-state execution
    run_matmul()
    start = time.perf_counter()
    run_matmul()
    print(f"{mode} time: {time.perf_counter() - start:.4f}s")
```

### Compare a exatidão entre os modos de precisão { #compare-accuracy-across-precision-modes }

Compare as saídas da TPU com uma referência `float32` da CPU usando as mesmas
entradas para ver como cada modo de precisão reduz o erro numérico:

```python
# CPU float32 reference result
c_ref = torch.matmul(a.cpu(), b.cpu())

for mode in [Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST]:
  with precision(mode):
    res = torch.matmul(a, b)
    diff = res.cpu() - c_ref
    print(f"Mode: {mode}")
    print(f"  Max Absolute Error: {diff.abs().max():.6e}")
    print(
        "  Mean Relative Error: "
        f" {(diff.abs() / (c_ref.abs() + 1e-8)).mean():.6e}\n"
    )
```

## Verifique a paridade numérica { #verify-numerical-parity }

Ao portar um modelo para a TPU ou depurar diferenças de convergência, verifique
a paridade numérica executando a mesma camada ou o mesmo submódulo na CPU e na
TPU com pesos sincronizados e entradas idênticas. Depois, compare as saídas com
`torch.testing.assert_close`:

```python
import torch.nn as nn

# 1. Set up identical models in bfloat16 for a fair comparison
model_cpu = nn.Linear(10, 10).cpu().to(torch.bfloat16)
model_tpu = nn.Linear(10, 10).to(device).to(torch.bfloat16)

# 2. Synchronize weights (mandatory for parity checks)
model_tpu.load_state_dict(model_cpu.state_dict())

# 3. Create identical input data for both devices
input_data = torch.randn(5, 10, dtype=torch.bfloat16)
input_tpu = input_data.to(device)

# 4. Run the forward pass
out_cpu = model_cpu(input_data)
out_tpu = model_tpu(input_tpu)

# 5. Compare results
# Note: TorchTPU's bfloat16 accumulation and fusion differences can exceed
# default tolerances, generating the `assert_close` diagnostic report below.
try:
  torch.testing.assert_close(out_tpu.cpu(), out_cpu)
  print("Parity check passed!")
except AssertionError as e:
  print("Parity check failed (expected for hardware precision drift):\n")
  print(e)
```

### Interprete as divergências detalhadas { #interpret-detailed-mismatches }

Quando os tensores excedem as tolerâncias, `torch.testing.assert_close` imprime
um resumo que ajuda a distinguir o desvio normal de ponto flutuante de um bug
de lógica:

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** uma porcentagem alta de elementos com pequenas
    diferenças absolutas geralmente indica diferenças normais de arredondamento
    ou de ordem de acumulação causadas pelas fusões do compilador, e não um
    bug.
*   **Greatest absolute / relative difference:** informa o maior erro e o
    índice dele no tensor, para que você possa inspecionar valores atípicos ou
    escolher limites adequados de `atol` e `rtol`.

### Práticas recomendadas para paridade { #best-practices-for-parity }

*   **Use os mesmos tipos de dados nos dois dispositivos:** compare `bfloat16`
    na TPU com `bfloat16` na CPU, ou `float32` na TPU (com
    `Precision.HIGHEST`) com `float32` na CPU. Comparar `bfloat16` com
    `float32` mede o erro de quantização, e não o desvio do hardware.
*   **Verifique os gradientes, além das ativações:** depois de chamar
    `loss.backward()`, compare `model_tpu.weight.grad.cpu()` com
    `model_cpu.weight.grad` usando o mesmo padrão.
*   **Comece pelas tolerâncias padrão:** use as tolerâncias padrão da
    [documentação de testes do PyTorch](https://docs.pytorch.org/docs/stable/testing.html)
    como referência e flexibilize `atol` ou `rtol` apenas onde for esperado por
    causa da acumulação com precisão reduzida.

## Especifique a precisão da saída { #specify-output-precision }

Ao multiplicar tensores `bfloat16` com `torch.mm` ou `torch.bmm`, a MXU acumula
os produtos em `float32` antes de arredondar a saída final de volta para
`bfloat16`. Passe `out_dtype=torch.float32` para retornar o acumulador
`float32` diretamente, sem reduzir a precisão:

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## Controle o excesso de precisão { #control-excess-precision }

Como a MXU da TPU acumula as multiplicações de matrizes em `bfloat16` usando
`float32`, o compilador XLA pode otimizar operações fundidas mantendo os
valores intermediários em `float32`, em vez de arredondá-los para `bfloat16`
entre as etapas. Manter esses bits intermediários extras se chama *excesso de
precisão* (excess precision).

Você pode configurar esse comportamento globalmente com
`torch.tpu.allow_excess_precision` (que define a flag de compilador subjacente
`--xla_allow_excess_precision` e tem `False` como padrão):

*   **Desativar o excesso de precisão (`False`, padrão):** defina
    `torch.tpu.allow_excess_precision = False` para forçar o compilador a
    manter as conversões intermediárias para `bfloat16`, de modo que as operações
    fundidas correspondam ao comportamento de arredondamento em `bfloat16` do
    modo eager.
*   **Permitir o excesso de precisão (`True`):** defina
    `torch.tpu.allow_excess_precision =
    True` para permitir que o compilador omita as conversões intermediárias para
    menor precisão e mantenha os resultados intermediários fundidos em
    `float32`.

> **Important:** defina `torch.tpu.allow_excess_precision` logo no início do
> script, antes de executar ou compilar qualquer operação de TPU. O runtime do
> XLA bloqueia essa flag quando a primeira compilação começa. Tentar mudá-la
> para outro valor depois disso gera um `RuntimeError`.

## Entenda a precisão da MXU da TPU { #understand-tpu-mxu-precision }

Entender como a MXU da TPU lida com números de ponto flutuante explica por que o
excesso de precisão e os modos `float32` com várias passagens funcionam como
funcionam:

*   **Intervalo e precisão (`bfloat16`):**
    [A pesquisa do Google Cloud sobre o `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    mostrou que as redes neurais podem ser treinadas com precisão de mantissa
    reduzida sem perder exatidão do modelo, desde que mantenham o amplo
    intervalo dinâmico do `float32` (fundamental para operações como normas e
    gradientes). O `float32` tem 23 bits de mantissa (24 bits de precisão do
    significando, incluindo o bit inicial implícito), enquanto o `bfloat16`
    mantém o mesmo intervalo de expoente de 8 bits com apenas 7 bits de mantissa
    (8 bits de precisão do significando, com um épsilon de máquina de `2**-7`,
    ou ~0,78%).
*   **Formatos de intercâmbio e formatos aritméticos:** os sistemas de ponto
    flutuante diferenciam **formatos de intercâmbio** (como os números são
    armazenados na memória) e **formatos aritméticos** (como o hardware calcula
    e arredonda as operações intermediárias, como os 3 bits extras de guarda,
    arredondamento e sticky em uma ALU padrão). `bfloat16` pode se referir a
    qualquer um dos dois.
*   **Acumulação ao longo da dimensão de contração:** multiplicar duas matrizes
    com os formatos `[I, J] @ [J, K]` calcula uma grade de produtos internos com o
    formato `[1,
    J] @ [J, 1]`, e cada um soma `J` produtos elemento a elemento ao longo da
    **dimensão de contração** `J`. Como o `bfloat16` tem um épsilon de máquina
    de `2**-7` (~`0.008`), acumular `J` produtos em `bfloat16` causaria grandes
    erros de arredondamento. Em vez disso, a MXU aceita entradas `bfloat16` (e
    `fp8` em gerações mais novas, como a TPU v7x) e acumula as somas
    intermediárias em `float32`, cumprindo o mesmo papel dos bits de guarda,
    arredondamento e sticky em uma ALU antes de arredondar para o formato de
    saída.
*   **Diferença entre os formatos de entrada e saída:** ter entradas `bfloat16`
    e um acumulador `float32` é a causa principal tanto do *excesso de
    precisão* (manter valores intermediários em `float32` entre operações
    fundidas) quanto da *precisão menor que a esperada* (a matmul `float32` de
    1 passagem trunca as entradas para significandos `bfloat16` de 8 bits). Já a
    unidade de execução vetorial (VPU) da TPU opera em todos os 23 bits de
    mantissa do `float32` (exceto em casos extremos menores do IEEE-754, como
    números subnormais).

### Expansão polinomial com várias passagens para `float32` { #multi-pass-polynomial-expansion-for-float32 }

Como uma MXU só aceita entradas `bfloat16` (significando de 8 bits), ela não
consegue calcular uma multiplicação de matrizes `float32` completa de 24 bits
em uma única passagem. Na prática, a maioria dos modelos modernos armazena os
pesos em `bfloat16` e raramente executa matmuls em `float32`. Quando é
necessária a precisão completa de `float32` na matmul, o XLA usa o algoritmo
de
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang e Heinecke) para representar cada significando `float32` de 24
bits como uma soma de três valores `bfloat16` de 8 bits (`High + Med + Low`) e
expandir o produto deles:

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

Multiplicar duas somas de 3 termos gera nove produtos `bfloat16`, dos quais os
três termos de ordem mais baixa sempre sofrem underflow em `float32`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1 passagem)* | High x Med *(3 passagens)*    | High x Low *(6 passagens)*    |
| **Med**  | Med x High *(3 passagens)* | Med x Med *(6 passagens)*     | Med x Low *(underflow)* |
| **Low**  | Low x High *(6 passagens)* | Low x Med *(underflow)* | Low x Low *(underflow)* |
<!-- mdformat on -->

### Avalie o desempenho com 3 e 6 passagens { #evaluate-3-pass-and-6-pass-performance }

Embora `Precision.HIGH` (3 passagens) e `Precision.HIGHEST` (6 passagens)
exijam 3x e 6x mais passagens na MXU, o impacto de ponta a ponta depende da sua
carga de trabalho:

*   **Dimensões de contração grandes:** em modelos modernos treinados com
    acumulação em `float32` (como o DeepSeek-V3) que têm dimensões de contração
    grandes, o ruído de acumulação e o não determinismo em milhares de
    elementos superam a contribuição dos termos cruzados de mantissa de ordem
    mais baixa. Por isso, 1 passagem (`Precision.DEFAULT`) é suficiente.
*   **Modelos limitados pela largura de banda de memória:** modelos `float32`
    mais antigos muitas vezes não fazem aritmética suficiente por byte
    transferido da memória de alta largura de banda (HBM) para saturar a MXU
    (ou os Tensor Cores da GPU). Nessas cargas de trabalho limitadas pela
    memória, as passagens extras na MXU da execução com 3 e 6 passagens usam
    ciclos da MXU que ficariam ociosos, o que resulta em uma lentidão de ponta a
    ponta menor que 3x ou 6x.

## Valide o comportamento da precisão bit a bit { #validate-bitwise-precision-behavior }

Você pode verificar como a MXU lida com os bits de mantissa e com o excesso de
precisão construindo entradas com duas regras de ponto flutuante binário:

*   `1 + 2**-n` cria um número com o significando binário `1.000...1000...`,
    colocando um `1` fracionário no `n`-ésimo bit da mantissa.
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: multiplicar dois números com bits
    ativados nas posições 6 e 7 da mantissa produz um resultado com bits
    ativados nas posições 6, 7 e `6 + 7 = 13`. Como o `bfloat16` só tem 7 bits
    de mantissa, a posição 13 ultrapassa o `bfloat16` e só é preservada quando
    mantida no acumulador `float32`:

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

Colocar esses valores em matrizes `256 x 256` aciona a execução na MXU e
permite verificar se o 13º bit da mantissa é descartado no arredondamento ou
mantido:

```python
# Python uses float64 by default.
bit_6 = 1 + 2**-6
bit_7 = 1 + 2**-7
bits_6_7 = 1.0 + 2**-6 + 2**-7
bits_6_7_13 = 1.0 + 2**-6 + 2**-7 + 2**-13

print("=== Python CPU values ===")
print(f"{bit_6=:.20f}")
print(f"{bit_7=:.20f}")
print(f"{bits_6_7=:.20f}")
print(f"{bits_6_7_13=:.20f}")
print(f"{(bit_6 * bit_7)=:.20f}")
assert bit_6 * bit_7 == bits_6_7_13

print("=== PyTorch TPU values ===")

eye = torch.eye(256, dtype=torch.bfloat16, device=device)
bit_6_tensor = eye.clone() * bit_6
bit_7_tensor = eye.clone() * bit_7
bits_6_7_tensor = eye.float().clone() * bits_6_7
bits_6_7_13_tensor = eye.float().clone() * bits_6_7_13

# Use a 1-element vector rather than a 0D scalar because PyTorch's type
# promotion rules ignore 0D scalar tensor dtypes when paired with 2D tensors.
one_float32 = torch.tensor([1.0], dtype=torch.float32, device=device)

print(f"{bit_6_tensor[0,0]=:.20f}")
print(f"{bit_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_tensor[0,0]=:.20f}")
print(f"{bits_6_7_13_tensor[0,0]=:.20f}")

def mul_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
  return a @ b * c

# In eager execution, `a @ b` rounds to bfloat16 before `* c`, dropping bit 13.
print("=== TPU values: Eager ===")
eager_res = mul_add(bit_6_tensor, bit_7_tensor, one_float32)[0, 0]
print(f"{eager_res=:.20f}")
assert eager_res == bits_6_7_tensor[0, 0]

# With allow_excess_precision = False, the compiled graph also rounds `a @ b`
# to bfloat16 before multiplying by `c`, dropping bit 13.
print("=== TPU values: compiled with allow_excess_precision = False ===")
torch.compiler.reset()
torch.tpu.allow_excess_precision = False
mul_add_compiled_false = torch.compile(mul_add, backend="tpu")
print(f"{torch.tpu.allow_excess_precision=}")
res_false = mul_add_compiled_false(bit_6_tensor, bit_7_tensor, one_float32)[
    0, 0
]
print(f"{res_false=:.20f}")

try:
  assert res_false == bits_6_7_tensor[0, 0]
  print("  [PASSED] Successfully disabled excess precision.")
except AssertionError:
  print(
      "  [WARNING] Failed to disable excess precision (XLA compiled another"
      " graph with default 'True' earlier in this session)."
  )

# Changing allow_excess_precision after XLA compilation has already started in
# this process raises a RuntimeError because compiler options are locked on
# the first compilation.
#
# When `torch.tpu.allow_excess_precision = True` is set at the start of a
# fresh Python session before any TPU operations or compilations run, XLA
# keeps `a @ b` in float32 when fusing with `* c`, preserving the 13th
# mantissa bit (res_true == bits_6_7_13_tensor[0, 0] == 1.0235595703125).
print("=== TPU values: compiled with allow_excess_precision = True ===")
torch.compiler.reset()
try:
  torch.tpu.allow_excess_precision = True
  mul_add_compiled_true = torch.compile(mul_add, backend="tpu")
  print(f"{torch.tpu.allow_excess_precision=}")
  res_true = mul_add_compiled_true(bit_6_tensor, bit_7_tensor, one_float32)[
      0, 0
  ]
  print(f"{res_true=:.20f}")
  assert res_true == bits_6_7_13_tensor[0, 0]
  print("  [PASSED] Successfully enabled excess precision.")
except (RuntimeError, AssertionError) as e:
  print(
      "  [EXPECTED] Cannot change excess precision after first compilation:"
      f" {e}"
  )
```
