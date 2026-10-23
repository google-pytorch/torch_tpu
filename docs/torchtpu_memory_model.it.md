# TorchTPU e il modello di memoria della TPU { #torchtpu-and-the-tpu-memory-model }

PyTorch si è evoluto insieme alle GPU e ne mette efficacemente a disposizione
la potenza agli utenti Python. TorchTPU estende PyTorch per esprimere la
potenza di XLA sulle TPU.

L'hardware fisico delle GPU e delle TPU è diverso, e lo sono anche i paradigmi
di programmazione pensati per ciascuna architettura. TorchTPU esiste per
colmare questo divario in entrambe le direzioni. Il nostro obiettivo è
supportare in XLA interfacce native delle GPU come le viste dei tensori e gli
stream, esponendo al tempo stesso la potenza unica del compilatore XLA tramite
le interfacce PyTorch che già conosci.

Capire come TorchTPU raggiunge questi obiettivi può aiutarti a ottimizzare
codebase native per TPU e ottenerne il massimo beneficio.

## Hardware: CPU, GPU, TPU { #hardware-cpus-gpus-tpus }

Una CPU general-purpose (come quella del tuo computer o del tuo dispositivo
mobile) è composta da più core, ognuno dei quali supporta uno o più thread.
Ogni thread può essere eseguito in parallelo e accedere alla memoria condivisa
con funzionalità di lettura e scrittura ad accesso casuale.

Una GPU è per molti versi simile a una CPU, ma ha molti più thread. Una CPU
moderna ad alte prestazioni può avere qualche decina di thread, mentre una GPU
ad alte prestazioni ne ha *migliaia*. Il compromesso è che ogni thread della
GPU, preso singolarmente, è un po' meno potente di un thread della CPU, e ci
sono alcune restrizioni sull'indipendenza dei thread. Ciononostante, questo
rende le GPU utili per i carichi di lavoro con operazioni altamente
parallelizzabili, come il rendering grafico o l'algebra lineare.

Una TPU ha un design fondamentalmente diverso. Il cuore della TPU è il
TensorCore con le sue unità di moltiplicazione di matrici (MXU). Queste MXU
funzionano come un
[array sistolico](https://en.wikipedia.org/wiki/Systolic_array), in qualche
modo simile a un registro SIMD su una CPU. Usano registri ampi e
bidimensionali che possono svolgere, per ciclo di clock, molto più lavoro di
calcolo denso rispetto a un thread di CPU o GPU.

Tuttavia, il TensorCore è fondamentalmente single-thread: non è possibile
suddividere un array sistolico per eseguire più calcoli simultanei. Inoltre,
avere registri ampi significa che un TensorCore è molto poco efficiente
nell'eseguire operazioni di lettura e scrittura sparse e ad accesso casuale,
perché deve caricare un grande blocco di memoria alla volta.

Le generazioni più recenti di TPU affiancano ai TensorCore gli SparseCore, che
(come suggerisce il nome) sono specializzati per le operazioni sparse, comprese
operazioni di lettura e scrittura più precise. I core separati consentono
anche una quantità limitata di parallelismo, poiché ogni SparseCore può
eseguire operazioni nello stesso momento di ogni TensorCore.

## Modello di programmazione: CUDA e XLA a confronto { #programming-model-cuda-vs-xla }

Le interfacce software native per GPU e TPU sono adattate alle capacità di
queste piattaforme.

Prendi come esempio CUDA, il modello di programmazione per GPU più usato alla
base di PyTorch. CUDA è un'interfaccia di programmazione di basso livello,
simile a C o C++. I buffer CUDA, come la memoria della CPU, sono
intrinsecamente array di byte monodimensionali. Un kernel CUDA è una procedura
imperativa: una funzione void, per usare la terminologia del C. Ogni thread
riceve i puntatori ad alcuni buffer e può leggere da qualsiasi offset di
qualsiasi buffer e riscriverci, byte per byte.

Questo modello di programmazione non funziona per le TPU. La memoria delle TPU
non è allineata come array di byte monodimensionali, ma come tile
bidimensionali allineati ai registri. Inoltre, come già detto, i TensorCore
non hanno thread e sono poco efficienti con l'accesso casuale.

Il compilatore della TPU, XLA, usa invece un modello dichiarativo e
funzionale. XLA accetta programmi scritti in un linguaggio chiamato
Multi-Level Intermediate Representation (MLIR), che contiene più dialetti. Con
il dialetto StableHLO puoi descrivere l'output logico del tensore che vuoi
ottenere da un programma. XLA compila quel programma in operazioni interne di
livello più basso, ottimizzate per la tua specifica generazione di TPU.

## Buffer: `torch.UntypedStorage` { #buffers-torchuntypedstorage }

Gli strumenti di base di PyTorch sono strettamente allineati al modello di
programmazione CUDA.

Un oggetto di storage, che usa la classe Python `torch.UntypedStorage`, è poco
più di un indirizzo di memoria (`torch.UntypedStorage.data_ptr()`) e di una
dimensione in byte (`torch.UntypedStorage.nbytes()`). Questo corrisponde bene
sia all'interfaccia della CPU sia a quella di CUDA.

Tuttavia, questo non basta per l'esecuzione su TPU: in XLA un buffer ha sempre
un tipo ben definito e una forma multidimensionale nota. Poiché la memoria
interna di una TPU è allineata a registri 2D, è fondamentale sapere come è
disposto un buffer. Questo significa che un `torch.UntypedStorage` da solo non
contiene informazioni sufficienti per l'esecuzione su TPU.

Per risolvere questo problema, TorchTPU aggiunge un ulteriore livello. Su TPU,
un `torch.UntypedStorage.data_ptr` non punta direttamente alla memoria della
TPU, ma a una posizione nella memoria della CPU che contiene metadati
aggiuntivi sul buffer TPU. **Questa è la chiave del funzionamento di
TorchTPU**: questi metadati aggiuntivi consentono un'interpretazione attiva
tra l'API esistente di PyTorch, simile a CUDA, e la realtà del design
funzionale di XLA e dell'hardware TPU.

## `torch.Tensor` { #torchtensor }
Un `torch.Tensor` può essere considerato contemporaneamente come due cose.

In primo luogo, un `torch.Tensor` è una rappresentazione *fisica* di un
pattern di accesso su un `torch.UntypedStorage` sottostante. Un `Tensor`
contiene metadati essenziali per gli stride (fattori moltiplicativi), le
dimensioni (limiti degli indici), l'offset (posizione dell'elemento 0) e il
tipo di elemento (che stabilisce i byte per elemento logico). Questi metadati
permettono di mappare una tupla di indici di un tensore in un offset in byte
nell'array di buffer lineare 1D che sta alla base di quel `Tensor`.

In secondo luogo, un `torch.Tensor` è una rappresentazione *logica* di un
oggetto matematico. Quando moltiplichi due tensori come matrici, il risultato
è un altro `Tensor`, con una forma appropriata, che rappresenta il risultato. Lo
schema esatto di stride e offset dello storage ha un'importanza minima: conta
più ciò che il `Tensor` significa come calcolo che ciò che il tensore è come
riferimento di memoria.

Quando passi da CUDA a TPU, è fondamentale capire questa distinzione: il
comportamento **logico** delle operazioni PyTorch sui Tensor può essere
eseguito correttamente su XLA/TPU, anche se la realtà **fisica** del
dispositivo è intrinsecamente diversa!

## Decomposizione delle viste { #view-decomposition }

Su CUDA, il linguaggio degli stride di un `torch.Tensor` viene usato
direttamente dai kernel CUDA interni: gli stride vengono forniti alle
procedure dei thread interni per descrivere come calcolare l'offset nel buffer
pertinente. Questi offset vengono calcolati dinamicamente a runtime.

Su TPU, invece, gli stride fisici non sono utilizzabili direttamente: una
descrizione monodimensionale degli stride non è utilizzabile per una memoria
bidimensionale allineata ai registri. Tuttavia, il comportamento logico delle
viste *è* supportato tramite StableHLO. La maggior parte delle operazioni di
vista sui Tensor ha un equivalente diretto in StableHLO: reshape, trasposizione,
slicing e broadcasting.

Quando riceve un pattern di stride di un `torch.Tensor`, TorchTPU esegue
un'operazione chiamata *decomposizione delle viste* (view decomposition), che
interpreta un pattern di stride e un tipo di elemento come una sequenza di
operazioni logiche che producono il Tensor di output desiderato, con ogni
elemento al posto giusto.

Ad esempio:

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

La versione di basso livello con manipolazione degli stride, che usa
`torch.Tensor.as_strided`, si può esprimere in modo equivalente con una
sequenza di operazioni più familiari, che possono poi essere espresse in
StableHLO e compilate da XLA.

È importante notare, tuttavia, che **l'equivalenza logica non implica
prestazioni equivalenti**. Un'operazione di reshape o di trasposizione può, in
alcuni casi, richiedere una copia completa su TPU per passare da un layout
allineato ai registri a un altro. Questo significa che, su TorchTPU, **le
viste non sono "gratuite"** come spesso accade su CUDA, a causa delle
differenze fisiche tra l'hardware GPU e TPU.

## Modifiche in-place e donazione dei buffer { #in-place-mutations-and-buffer-donation }

StableHLO è un linguaggio di programmazione funzionale. Più precisamente, è un
[linguaggio ad assegnazione singola statica (SSA)](https://en.wikipedia.org/wiki/Static_single-assignment_form).
Questo significa che i valori possono solo essere creati da zero, mai
modificati o riassegnati. Questo aiuta il compilatore XLA a produrre piani di
esecuzione ottimizzati.

Né le IR delle GPU né quelle delle CPU sono puramente funzionali e, di
conseguenza, nemmeno PyTorch lo è. Le operazioni sui tensori con il suffisso
underscore, o che usano l'argomento keyword "`out=`", sovrascrivono i byte nel
loro `torch.UntypedStorage` di base. Poiché più `torch.Tensors` possono
condividere un buffer di storage comune, una scrittura su qualsiasi tensore
diventa immediatamente visibile a tutti gli altri Tensor.

TorchTPU gestisce questa situazione sfruttando il proprio livello di metadati.
Quando un'operazione PyTorch richiede un aggiornamento in-place o
un'assegnazione a un tensore out, TorchTPU:

*   Descrive l'aggiornamento logico a XLA, producendo un nuovo buffer
*   Sostituisce il buffer all'interno dell'oggetto di metadati a cui punta il
    `torch.UntypedStorage`

Si tratta essenzialmente di una funzionalizzazione automatica: tutte le
modifiche vengono invece espresse come operazioni out-of-place.

Per ottenere la funzionalizzazione automatica bisogna poter eseguire
l'algoritmo di decomposizione delle viste al contrario, operazione che
chiamiamo inversione delle viste. Questo è possibile solo per le viste non
sovrapposte: scrivere su un tensore risultato di un broadcast non è valido,
perché più posizioni logiche corrispondono a un'unica posizione fisica e
l'istruzione diventa ambigua. Sul backend CUDA, l'operazione sarebbe
considerata un comportamento non definito, ma potrebbe fallire o meno in modo
esplicito; su TorchTPU fallisce sempre in modo esplicito.

Nel caso di una scrittura parziale, l'inversione delle viste usa un'operazione
StableHLO chiamata
[`stablehlo.dynamic_update_slice`](https://openxla.org/stablehlo/spec#dynamic_update_slice)
per calcolare un tensore di output unificato, che contiene nelle posizioni
corrette sia i valori sovrascritti sia quelli non sovrascritti.

Naturalmente, un'operazione out-of-place raddoppia il picco di memoria: la
vecchia e la nuova versione coesistono brevemente prima che il vecchio valore
venga scartato. Per evitarlo, TorchTPU cerca, quando possibile, di applicare
l'equivalente più vicino di XLA alle modifiche in-place: la **donazione dei
buffer** (buffer donation). Quando viene rimosso l'ultimo riferimento a un
buffer obsoleto, TorchTPU annota lo StableHLO per "donare" l'input,
consentendo a XLA di consumarlo ed evitando l'aumento di memoria dovuto alla
funzionalizzazione automatica.

Non sempre è possibile donare un buffer in modo sicuro. Se TorchTPU non riesce
a dimostrare che un buffer può essere donato in sicurezza, ripiega su un
aggiornamento out-of-place. Questo può essere particolarmente costoso per
piccole scritture su buffer di grandi dimensioni: in questo caso,
l'aggiornamento out-of-place copia l'intero tensore di storage di base, non
solo la parte su cui si scrive!

## Copy-on-write { #copy-on-write }

La funzionalizzazione ha però dei vantaggi. Uno di questi è la possibilità di
adottare un comportamento copy-on-write.

Sul backend CUDA, chiamare `torch.Tensor.clone()` alloca immediatamente un
nuovo buffer di memoria fisica abbastanza grande da contenere i dati del
Tensor. Dopodiché, le scritture in-place sui diversi `torch.UntypedStorages`
sono completamente indipendenti, ma il costo in memoria viene pagato subito.

Sul backend TorchTPU, invece, l'immutabilità di un valore assegnato
staticamente fa sì che una copia possa essere semplicemente un riferimento
all'originale, senza bisogno di una copia fisica immediata.

Non appena avviene un aggiornamento in-place, dobbiamo riflettere
quell'aggiornamento solo sul `torch.UntypedStorage` modificato e non
sull'altro. Questo impedisce la donazione del buffer (perché servono ancora
sia il vecchio sia il nuovo valore), quindi in questo caso l'aggiornamento
in-place viene eseguito out-of-place senza donazione. Questo aggiornamento
out-of-place alloca nuova memoria, ma solo in modo lazy e quando serve, a
differenza di PyTorch/CUDA, che paga il costo in memoria in anticipo.

## Kernel personalizzati { #custom-kernels }

TorchTPU fa il possibile per applicare donazioni dei buffer che ottimizzano la
memoria e fornisce una trasformazione logicamente equivalente dalle viste
CUDA con stride alle operazioni di vista di StableHLO. Detto questo, si tratta
comunque di operazioni di alto livello, non legate direttamente al layout
fisico di un buffer TPU, e non garantiscono una particolare semantica di basso
livello.

Se vuoi il massimo controllo di basso livello, TorchTPU supporta anche
l'integrazione di kernel personalizzati. Questi permettono di sostituire in
modo mirato le operazioni critiche, o persino di riscrivere l'intero modello,
specificando manualmente il comportamento dei buffer TPU. Puoi usarli per
garantire un comportamento ottimale di donazione e di aggiornamento in-place.

## Punti chiave { #takeaways }
Quando lavori con TorchTPU, tieni presenti questi fatti fondamentali:

*   **Il codice PyTorch produce lo stesso risultato logico sia su CUDA sia su
    TorchTPU:** puoi usare le familiari operazioni CUDA di basso livello (viste
    con storage condiviso, manipolazione degli stride, aggiornamenti in-place)
    e ottenere gli stessi risultati su TPU, anche se le prestazioni potrebbero
    non essere le stesse.
*   **Le copie sono economiche:** copiare un tensore in forma contigua spesso
    non ha alcun costo di calcolo, grazie al comportamento copy-on-write.
*   **Le viste sono costose:** leggere un tensore con una forma diversa o con
    un pattern non contiguo può forzare un relayout sulla TPU, che può essere
    costoso. Quando possibile, preferisci i tensori contigui.
*   **Sovrascrivere un tensore contiguo di solito è efficiente in termini di
    memoria:** di solito un tensore contiguo può essere sovrascritto con una
    donazione del buffer, se tutti i vecchi valori vengono cancellati.
*   **Le scritture parziali a volte sono costose in termini di memoria:** se la
    donazione del buffer non è possibile, il costo in memoria di una scrittura
    è proporzionale alla dimensione del torch.UntypedStorage, non alla
    dimensione dei dati scritti. Preferisci usare molti tensori piccoli invece
    di slice di tensori grandi.
*   **I kernel personalizzati offrono il controllo completo:** quando la
    traduzione da CUDA a XLA non riesce o non è sufficiente, implementare un
    kernel personalizzato nativo per TPU permette di ottimizzare specifiche
    prestazioni di basso livello.
