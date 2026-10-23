# Gestire la precisione numerica in TorchTPU { #manage-numerical-precision-in-torchtpu }

Le TPU di Google suddividono le operazioni in virgola mobile tra due unità
hardware per bilanciare throughput e accuratezza numerica:

*   **Matrix Multiply Unit (MXU):** esegue le operazioni di contrazione (come
    `torch.matmul`, `torch.mm`, `torch.bmm` e le convoluzioni) usando input
    `bfloat16` nativi (significando a 8 bit) con accumulo `float32` a 32 bit.
*   **Vector Execution Unit (VPU):** esegue le operazioni senza contrazione
    (come l'aritmetica elemento per elemento, le attivazioni, la layer
    normalization e la softmax) in `float32` IEEE-754 completo (significando a
    24 bit).

Questa guida spiega come configurare le modalità di precisione della
moltiplicazione di matrici `float32`, verificare la parità numerica tra CPU e
TPU, controllare i tipi dell'accumulatore di output e la precisione in
eccesso, e convalidare nel codice il comportamento bit per bit della MXU.

### Confrontare la precisione di PyTorch su CUDA e su TPU { #compare-pytorch-precision-on-cuda-and-tpu }

TorchTPU supporta le API di precisione standard di PyTorch
(`torch.set_float32_matmul_precision` e `torch.get_float32_matmul_precision`)
insieme a controlli specifici per TPU (`torch.tpu.Precision` e
`torch.tpu.precision`):

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Aspetto            | PyTorch su CUDA                      | PyTorch su TPU (TorchTPU)              |
| :----------------- | :----------------------------------- | :------------------------------------- |
| **Modalità di precisione predefinita** | `"highest"` (`float32` puro) | `"medium"` (`Precision.DEFAULT`)       |
| **Primitiva dell'acceleratore per la matematica veloce** | Tensor Core (`TF32`) | MXU sistoliche (`bfloat16`)             |
| **Bit della matematica veloce** | 1 bit di segno + 8 di esponente + 10 di mantissa (`TF32` a 19 bit) | 1 bit di segno + 8 di esponente + 7 di mantissa (`bfloat16` a 16 bit) |
| **Precisione di accumulo** | `float32` (`"highest"`, `"high"`), `TF32` (`"medium"`) | `float32` in tutte le modalità |
| **Meccanismo per una precisione superiore** | Unità hardware dedicate `float32`/`TF32` | Decomposizione multi-pass (3 o 6 passate sulla MXU) |
| **Impatto sulle prestazioni di `high`/`highest`** | Velocità di base o rallentamento modesto (~1–2x) | Costo di calcolo sulla MXU ~3x per `"high"`; ~6x per `"highest"` |
| **API di ambito** | `torch.set_float32_matmul_precision` globale | `torch.set_float32_matmul_precision` globale <br> + `torch.tpu.precision` con ambito |
<!-- mdformat on -->

## Specificare le modalità di precisione { #specify-precision-modes }

Quando moltiplichi tensori `float32` sulla MXU, TorchTPU mappa i livelli di
precisione di PyTorch sull'enum `torch.tpu.Precision`:

*   `"medium"` / `Precision.DEFAULT` (1 passata): **predefinita.** Arrotonda
    gli input `float32` a `bfloat16` (moltiplicando gli 8 bit più alti del
    significando) e accumula in `float32`. Consigliata per quasi tutti i
    carichi di lavoro di addestramento e inferenza.
*   `"high"` / `Precision.HIGH` (3 passate): suddivide gli input `float32` in
    componenti `bfloat16` alta e bassa e calcola tre termini incrociati
    (moltiplicando i primi 16 bit del significando) con un costo di calcolo
    sulla MXU di circa 3 volte.
*   `"highest"` / `Precision.HIGHEST` (6 passate): suddivide gli input
    `float32` in tre componenti `bfloat16` e calcola sei termini incrociati
    (moltiplicando tutti i 24 bit del significando) per approssimare
    l'accuratezza completa del `float32` IEEE-754 con un costo di calcolo
    sulla MXU di circa 6 volte.

Puoi configurare la precisione della moltiplicazione di matrici in due modi:

*   **Configurazione globale:** chiama `torch.set_float32_matmul_precision("medium"
    | "high" | "highest")` per impostare la modalità predefinita del processo.
*   **Context manager con ambito:** usa `with torch.tpu.precision(mode):` per
    applicare una precisione superiore solo ai layer o ai sottomoduli
    numericamente sensibili.
*   **Avviso sulle prestazioni:** se selezioni `"high"` o `"highest"` viene
    emesso un avviso una sola volta (`TORCH_WARN_ONCE`). Mantieni la modalità
    predefinita `"medium"` a meno che tu non osservi problemi di convergenza
    legati alla precisione.

> **Note:** entrare o uscire da `torch.tpu.precision` introduce un graph break
> in `torch.compile`. Quando usi `torch.compile(..., fullgraph=True)`,
> imposta la precisione fuori dalla funzione compilata con
> `torch.set_float32_matmul_precision(...)` oppure racchiudi la chiamata
> esterna in `with
> torch.tpu.precision(...):`.

Usa `torch.tpu.precision` per confrontare il tempo di esecuzione a regime
nelle tre modalità:

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

### Confrontare l'accuratezza tra le modalità di precisione { #compare-accuracy-across-precision-modes }

Confronta gli output della TPU con un riferimento `float32` su CPU sugli
stessi input per vedere come ogni modalità di precisione riduce l'errore
numerico:

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

## Verificare la parità numerica { #verify-numerical-parity }

Quando esegui il porting di un modello su TPU o analizzi differenze di
convergenza, verifica la parità numerica eseguendo lo stesso layer o
sottomodulo su CPU e su TPU con pesi sincronizzati e input identici, quindi
confronta gli output con `torch.testing.assert_close`:

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

### Interpretare le discrepanze dettagliate { #interpret-detailed-mismatches }

Quando i tensori superano le tolleranze, `torch.testing.assert_close` stampa un
riepilogo che aiuta a distinguere la normale deriva in virgola mobile da un
bug logico:

```text
Tensor-likes are not close!

Mismatched elements: 1 / 50 (2.0%)
Greatest absolute difference: 0.006596 at index (3, 9) (up to 1e-05 allowed)
Greatest relative difference: 0.165375 at index (3, 9) (up to 0.016 allowed)
```

*   **Mismatched elements:** un'alta percentuale di elementi con piccole
    differenze assolute indica in genere normali differenze di arrotondamento
    o di ordine di accumulo dovute alle fusioni del compilatore, non un bug.
*   **Greatest absolute / relative difference:** riporta l'errore più grande e
    il relativo indice nel tensore, così puoi esaminare i valori anomali o
    scegliere soglie `atol` e `rtol` appropriate.

### Best practice per la parità { #best-practices-for-parity }

*   **Usa gli stessi tipi di dati su entrambi i dispositivi:** confronta
    `bfloat16` su TPU con `bfloat16` su CPU, oppure `float32` su TPU (con
    `Precision.HIGHEST`) con `float32` su CPU. Confrontare `bfloat16` con
    `float32` misura l'errore di quantizzazione, non la deriva dell'hardware.
*   **Controlla sia i gradienti sia le attivazioni:** dopo aver chiamato
    `loss.backward()`, confronta `model_tpu.weight.grad.cpu()` con
    `model_cpu.weight.grad` usando lo stesso schema.
*   **Parti dalle tolleranze standard:** usa come base le tolleranze
    predefinite indicate nella
    [documentazione sui test di PyTorch](https://docs.pytorch.org/docs/stable/testing.html)
    e allenta `atol` o `rtol` solo dove è previsto per l'accumulo a precisione
    ridotta.

## Specificare la precisione dell'output { #specify-output-precision }

Quando moltiplichi tensori `bfloat16` con `torch.mm` o `torch.bmm`, la MXU
accumula i prodotti in `float32` prima di arrotondare l'output finale di
nuovo a `bfloat16`. Passa `out_dtype=torch.float32` per restituire
direttamente l'accumulatore `float32` senza conversione verso il basso:

```python
a_bmm = torch.randn((2, 256, 256), dtype=torch.bfloat16, device=device)
# Using out_dtype to force float32 output accumulator precision
res_bmm = torch.bmm(a_bmm, a_bmm, out_dtype=torch.float32)
print(f"Input dtype: {a_bmm.dtype}")
print(f"Output dtype: {res_bmm.dtype}")
```

## Controllare la precisione in eccesso { #control-excess-precision }

Poiché la MXU della TPU accumula le moltiplicazioni di matrici `bfloat16` in
`float32`, il compilatore XLA può ottimizzare le operazioni fuse mantenendo i
valori intermedi in `float32` invece di arrotondarli a `bfloat16` tra un
passaggio e l'altro. Il mantenimento di questi bit intermedi aggiuntivi si
chiama *precisione in eccesso* (excess precision).

Puoi configurare questo comportamento a livello globale con
`torch.tpu.allow_excess_precision` (che imposta il flag del compilatore
sottostante `--xla_allow_excess_precision` e ha come valore predefinito
`False`):

*   **Disattiva la precisione in eccesso (`False`, predefinito):** imposta
    `torch.tpu.allow_excess_precision = False` per obbligare il compilatore a
    mantenere le conversioni intermedie a `bfloat16`, in modo che le
    operazioni fuse corrispondano al comportamento di arrotondamento
    `bfloat16` della modalità eager.
*   **Consenti la precisione in eccesso (`True`):** imposta `torch.tpu.allow_excess_precision =
    True` per consentire al compilatore di eliminare le conversioni intermedie
    e mantenere in `float32` i risultati intermedi fusi.

> **Important:** imposta `torch.tpu.allow_excess_precision` all'inizio dello
> script, prima di eseguire o compilare qualsiasi operazione TPU. Il runtime
> XLA blocca questo flag all'avvio della prima compilazione; se in seguito
> provi a cambiarlo con un valore diverso, viene generato un `RuntimeError`.

## Capire la precisione della MXU della TPU { #understand-tpu-mxu-precision }

Capire come la MXU della TPU gestisce i numeri in virgola mobile spiega perché
la precisione in eccesso e le modalità `float32` multi-pass funzionano in
questo modo:

*   **Intervallo e precisione (`bfloat16`):** la
    [ricerca di Google Cloud su `bfloat16`](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    ha dimostrato che le reti neurali possono essere addestrate con una
    precisione della mantissa ridotta senza perdere accuratezza del modello,
    purché mantengano l'ampio intervallo dinamico del `float32` (fondamentale
    per operazioni come norme e gradienti). Mentre `float32` ha 23 bit di
    mantissa (24 bit di precisione del significando, incluso il bit iniziale
    implicito), `bfloat16` mantiene lo stesso intervallo dell'esponente a 8
    bit con solo 7 bit di mantissa (8 bit di precisione del significando, per
    un epsilon di macchina di `2**-7`, circa lo 0,78%).
*   **Formati di interscambio e formati aritmetici:** i sistemi in virgola
    mobile distinguono tra **formati di interscambio** (come i numeri vengono
    memorizzati in memoria) e **formati aritmetici** (come l'hardware calcola
    e arrotonda le operazioni intermedie, ad esempio i 3 bit aggiuntivi guard,
    round e sticky in una ALU standard). `bfloat16` può riferirsi a entrambi.
*   **Accumulo lungo la dimensione di contrazione:** moltiplicare due matrici
    di forma `[I, J] @ [J, K]` calcola una griglia di prodotti interni di forma `[1,
    J] @ [J, 1]`, ognuno dei quali somma `J` prodotti elemento per elemento
    lungo la **dimensione di contrazione** `J`. Poiché `bfloat16` ha un
    epsilon di macchina di `2**-7` (~`0.008`), accumulare `J` prodotti in
    `bfloat16` causerebbe grandi errori di arrotondamento. La MXU accetta
    invece input `bfloat16` (e `fp8` nelle generazioni più recenti come TPU
    v7x) e accumula le somme intermedie in `float32`, con lo stesso ruolo dei
    bit guard, round e sticky in una ALU prima dell'arrotondamento al formato
    di output.
*   **Discrepanza tra formato di input e di output:** avere input `bfloat16` e
    un accumulatore `float32` è la causa principale sia della *precisione in
    eccesso* (mantenimento di valori intermedi `float32` tra operazioni fuse)
    sia della *precisione inferiore al previsto* (matmul `float32` a 1
    passata che tronca gli input a significandi `bfloat16` a 8 bit). Al
    contrario, la Vector Execution Unit (VPU) della TPU opera su tutti i 23
    bit di mantissa del `float32` (tranne che per casi limite minori di
    IEEE-754, come i numeri subnormali).

### Espansione polinomiale multi-pass per `float32` { #multi-pass-polynomial-expansion-for-float32 }

Poiché una MXU accetta solo input `bfloat16` (significando a 8 bit), non può
calcolare in una sola passata una moltiplicazione di matrici `float32`
completa a 24 bit. In pratica, la maggior parte dei modelli moderni
memorizza i pesi in `bfloat16` ed esegue raramente matmul `float32`. Quando
serve la precisione completa delle matmul `float32`, XLA usa l'algoritmo
descritto in
[Leveraging the bfloat16 Artificial Intelligence Datatype For Higher-Precision Computations](https://arxiv.org/pdf/1904.06376)
(Henry, Tang e Heinecke) per rappresentare ogni significando `float32` a 24
bit come somma di tre valori `bfloat16` a 8 bit (`High + Med + Low`) ed
espanderne il prodotto:

```text
(a + b + c)(d + e + f) = ad + ae + af + bd + be + bf + cd + ce + cf
```

La moltiplicazione di due somme di 3 termini produce nove prodotti
`bfloat16`, di cui i tre termini di ordine più basso vanno sempre in
underflow in `float32`:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|          | High        | Med                      | Low                      |
| :------- | :---------- | :----------------------- | :----------------------- |
| **High** | High x High *(1 passata)* | High x Med *(3 passate)*    | High x Low *(6 passate)*    |
| **Med**  | Med x High *(3 passate)* | Med x Med *(6 passate)*     | Med x Low *(underflow)* |
| **Low**  | Low x High *(6 passate)* | Low x Med *(underflow)* | Low x Low *(underflow)* |
<!-- mdformat on -->

### Valutare le prestazioni a 3 e 6 passate { #evaluate-3-pass-and-6-pass-performance }

Anche se `Precision.HIGH` (3 passate) e `Precision.HIGHEST` (6 passate)
richiedono un numero di passate sulla MXU 3 e 6 volte maggiore, l'impatto
end-to-end dipende dal tuo carico di lavoro:

*   **Dimensioni di contrazione grandi:** nei modelli moderni addestrati con
    accumulo `float32` (come DeepSeek-V3) che hanno dimensioni di contrazione
    grandi, il rumore di accumulo e il non determinismo su migliaia di
    elementi prevalgono sul contributo dei termini incrociati di ordine
    inferiore della mantissa, quindi 1 passata (`Precision.DEFAULT`) è
    sufficiente.
*   **Modelli limitati dalla banda di memoria:** i modelli `float32` meno
    recenti spesso non eseguono abbastanza operazioni aritmetiche per ogni
    byte trasferito dalla High Bandwidth Memory (HBM) da saturare la MXU (o i
    Tensor Core delle GPU). In questi carichi di lavoro limitati dalla
    memoria, le passate aggiuntive sulla MXU dell'esecuzione a 3 e 6 passate
    usano cicli della MXU che altrimenti resterebbero inattivi, con un
    rallentamento end-to-end inferiore a 3 o 6 volte.

## Convalidare il comportamento della precisione bit per bit { #validate-bitwise-precision-behavior }

Puoi verificare come la MXU gestisce i bit della mantissa e la precisione in
eccesso costruendo input secondo due regole binarie della virgola mobile:

*   `1 + 2**-n` crea un numero con significando binario `1.000...1000...`,
    con un `1` frazionario nell'`n`-esimo bit della mantissa.
*   `(1 + m) * (1 + n) = 1 + m + n + mn`: moltiplicando due numeri con bit
    impostati nelle posizioni 6 e 7 della mantissa si ottiene un risultato con
    bit impostati nelle posizioni 6, 7 e `6 + 7 = 13`. Poiché `bfloat16`
    contiene solo 7 bit di mantissa, la posizione 13 va oltre la capacità di
    `bfloat16` e viene mantenuta solo se resta nell'accumulatore `float32`:

```py
value = (1 + 2**-6) * (1 + 2**-7)  # 1.0000_01 * 1.0000_001 = 1.0000_0110_0000_1
```

Inserire questi valori in matrici `256 x 256` attiva l'esecuzione sulla MXU e
ti permette di verificare se il 13° bit della mantissa viene eliminato
dall'arrotondamento o mantenuto:

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
