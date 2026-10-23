# Come funziona TorchTPU { #how-torchtpu-works }

## Nozioni di base sulle TPU { #tpu-primer }

Le [TPU di Google](https://cloud.google.com/tpu) sono ASIC specializzati,
progettati appositamente per eseguire le enormi operazioni sui tensori alla
base del deep learning. L'architettura delle TPU
[varia](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) da una
generazione all'altra ma, in linea generale, un chip TPU contiene:

*   Uno o più TensorCore, ognuno dei quali è un processore single-thread
    composto da unità di moltiplicazione di matrici (MXU), un'unità vettoriale
    (VPU) e un'unità scalare.
*   Uno o più SparseCore, specializzati per le operazioni sparse, ai quali il
    TensorCore può delegare (offload) operazioni.
*   Un'ampia High Bandwidth Memory (HBM), 288 GB a 8,6 TB/s nella
    [TPU 8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive),
    che puoi usare per memorizzare i dati di ML e condividerli con la CPU.
*   Un'interconnessione bidirezionale tra chip (Inter-Chip Interconnect, ICI),
    19,2 TBps nelle
    [TPU 8t e 8i](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
    che consente una comunicazione rapida tra i chip TPU all'interno di
    [slice TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices)
    configurate dall'utente. Le slice sono sottogruppi logici di un
    [pod TPU](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod),
    composto da un numero preconfigurato di chip TPU collegati in una
    configurazione a toro 3D (per TPU 7 e TPU 8t) o Boardfly (TPU 8i).
*   Una rete di data center bidirezionale (Data-Center-Network, DCN), 400 GBps
    nella
    [TPU 8t](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/),
    che consente la comunicazione tra chip TPU di slice diverse.

Le TPU si programmano con lo stack del
[compilatore XLA](https://github.com/openxla/xla) tramite due diversi punti di
ingresso: [StableHLO](https://openxla.org/stablehlo/spec) e
[Mosaic](https://docs.jax.dev/en/latest/pallas/design/design.html#lowering-pallas-to-mosaic-for-tpu).
L'ISA effettiva della TPU è proprietaria, soggetta a modifiche e non
disponibile pubblicamente. In pratica, puoi considerare StableHLO e (ancora di
più) Mosaic come equivalenti a PTX. L'immagine seguente mostra uno schema di
alto livello:

![Come funziona TorchTPU](./images/torch_tpu_inputs_to_tpu.svg)

Un programma C++ può accedere a una TPU locale istanziando un
[client PjRT](https://openxla.org/xla/pjrt/cpp_api_overview), la cui API
fornisce metodi per compilare MLIR (StableHLO o Mosaic) in un eseguibile TPU,
per avviare quell'eseguibile e per copiare dati da e verso la TPU. Nota che
l'esecuzione sulla TPU è asincrona e la CPU può attendere che un risultato
della TPU sia pronto attendendo il PjRTBuffer associato. StableHLO segue il
[paradigma SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) e
quindi non consente modifiche in-place: un'operazione di scrittura produce
sempre un nuovo valore (almeno a livello concettuale). Questo aspetto è
particolarmente importante per TorchTPU, che offre la familiare semantica
delle "viste" di PyTorch, comprese le modifiche, tramite la decomposizione e
l'inversione delle viste. Il compilatore XLA presuppone che le forme dei
tensori siano note in fase di compilazione. In TorchTPU questo vincolo viene
attenuato dal [dinamismo limitato](./dynamic_shapes.md). Poiché questo
vincolo è una limitazione per i carichi di lavoro PyTorch, stiamo
aggiungendo il supporto per il dinamismo limitato, ovvero il caso in cui i
limiti massimi delle dimensioni di una forma sono noti in fase di
compilazione, ma le dimensioni effettive sono note solo a runtime.

## TorchTPU: un backend out-of-tree con supporto per il caricamento automatico { #torchtpu-an-out-of-tree-backend-with-autoload-support }

Seguendo la
[best practice attuale](https://docs.pytorch.org/tutorials/advanced/extend_dispatcher.html),
secondo cui i backend aggiuntivi vengono costruiti out-of-tree e registrati a
runtime, estendiamo il dispatcher di PyTorch con un nuovo backend C++,
registrato come `PrivateUse1`. Inoltre, abbiamo usato il meccanismo standard
`at::Tensor` per definire un'implementazione dello storage dei tensori
specifica per il dispositivo (usando il `c10::UniqueVoidPtr` all'interno di
`c10::StorageImpl`).

Abbiamo progettato TorchTPU pensando all'usabilità. Quando importi PyTorch su
una macchina TPU in cui è installata la wheel di TorchTPU, Python carica
automaticamente la wheel di TorchTPU. Il caricamento automatico della wheel
configura il backend e sostituisce il nome `PrivateUse1` con `tpu`. In questo
modo, in un esempio semplice, un programma TorchTPU non è diverso da quelli
per i backend in-tree, e un programma PyTorch può essere eseguito sulla TPU
con modifiche minime o nulle:

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

Poiché la TPU diventa l'acceleratore corrente e `tpu_dist` è il backend
predefinito di TorchTPU, non è necessario specificarli in modo esplicito:

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

## Acquisizione del grafo { #capturing-the-graph }

TorchTPU traccia le operazioni eager a livello ATen, basandosi sugli hook di
dispatch delle operazioni di PyTorch che fanno parte dell'infrastruttura dei
backend. Le operazioni di TorchTPU vengono costruite come programmi StableHLO
e passate a XLA singolarmente o in gruppi, in modo che il compilatore possa
eseguire una fusione significativa delle operazioni e usare l'hardware in modo
più efficace. Quando TorchTPU riceve una chiamata di callback dal dispatcher
di PyTorch, il runtime:

*   Crea un'istanza di Deferred Op che contiene:
    *   Un oggetto builder StableHLO che, quando viene invocato, restituisce
        una funzione StableHLO che implementa la semantica di `aten::mul`, come
        mostrato di seguito (nota che il broadcasting implicito e la promozione
        dei dtype vengono gestiti altrove)

```
func.func @main(%arg0: tensor<4x16xbfloat16>, %arg1: tensor<4x16xbfloat16>) -> tensor<4x16xbfloat16> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x16xbfloat16>
  return %0 : tensor<4x16xbfloat16>
}
```

*   Una mappa facoltativa di parametri statici dell'operazione (ad esempio
    training e momentum per `aten::native_batch_norm`) da usare per il calcolo
    di una chiave di cache;
*   Restituisce un nuovo `at::Tensor` con un `c10::StorageImpl` che punta
    (tramite indirezione) a quell'istanza di Deferred Op.

Quando raggiunge un punto di sincronizzazione appropriato, che dipende dalla
modalità di esecuzione, TorchTPU:

*   Analizza la regione di operazioni;
*   Determina quali operazioni materializzare in codice eseguibile;
*   Crea una chiave della cache di compilazione per il grafo;
*   Se esiste già nella cache, recupera l'eseguibile e lo invia alla TPU;
*   Se non esiste nella cache, compila l'oggetto in modo asincrono, lo aggiunge
    alla cache e lo invia per l'esecuzione;
*   Eventualmente attende che il valore sia pronto e lo copia indietro, se il
    punto di sincronizzazione raggiunto lo richiede.

Descriviamo nel dettaglio le varie modalità di esecuzione in
[Il modello di esecuzione di TorchTPU](./torchtpu_execution_model.md).

## La cache di compilazione di TorchTPU { #the-torchtpu-compilation-cache }

Per evitare compilazioni non necessarie, TorchTPU usa una cache di
compilazione multilivello:

*   **Livello 1:** cache in memoria locale al processo per condividere le
    compilazioni all'interno dello stesso processo. È il più veloce dei tre
    livelli ed è sempre attivo.
*   **Livello 2:** memoria condivisa locale alla macchina, che consente la
    condivisione tra processi sullo stesso host. È utile per l'addestramento o
    l'inferenza multi-dispositivo. Il livello 2 è attivo per impostazione
    predefinita per i programmi distribuiti, mentre per i programmi su singolo
    dispositivo non offre vantaggi ed è quindi disattivato.
*   **Livello 3:** cache persistente basata su un file system remoto (ad
    esempio Google Cloud Storage), che consente il riutilizzo tra host,
    programmi e riavvii dei job. È utile quando esegui programmi TPU in
    ambienti di esecuzione come Colab e i data center. È disattivato per
    impostazione predefinita, ma puoi attivarlo specificando una posizione di
    archiviazione.

Ogni voce è un eseguibile TPU compilato che rappresenta un grafo di calcolo
StableHLO generato da un'operazione ATen o da una sequenza di operazioni ATen.
I grafi StableHLO sono rappresentati da fingerprint che combinano la
struttura del grafo, le forme dei tensori di input e output e le opzioni del
compilatore, comprese le versioni dell'hardware TPU, di libtpu e del runtime
PjRt.

## Supporto di `torch.compile` { #torchcompile-support }

Per mantenere un'esperienza utente coerente con altri hardware che supportano
`torch.compile()`, registriamo un nuovo
[backend](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)
TorchDynamo, che puoi usare in questo modo: `compiled_model = torch.compile(model,
backend="tpu")`.

Dynamo invoca il backend TorchTPU passando un `torch.fx.GraphModule` e un
elenco di tensori di input di esempio. A quel punto il runtime traccia il
grafo fornito come se fossimo in modalità eager, ma con tutte le operazioni
differite e senza alcuna esecuzione sulla TPU. Il runtime genera e compila un
unico programma StableHLO per l'intero grafo FX e restituisce un callable
Python che racchiude l'invocazione. Nota che TorchTPU non usa
`TorchInductor`, ma riutilizza il maturo stack del compilatore XLA, sviluppato
insieme all'architettura TPU.

## Ulteriori dettagli { #more-detail }

Nei documenti seguenti approfondiamo:

*   Il [modello di memoria](./torchtpu_memory_model.md) supportato dalla TPU e
    le sue implicazioni per PyTorch;
*   Le [modalità di esecuzione](./torchtpu_execution_model.md) supportate da
    TorchTPU;
*   Il supporto per i [kernel personalizzati](./custom_kernels.md);
*   L'[esecuzione distribuita](./distributed_support.md) su cluster di TPU.
