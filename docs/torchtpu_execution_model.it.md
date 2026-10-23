# Il modello di esecuzione di TorchTPU { #torchtpu-execution-model }

Quando esegui un programma PyTorch con TorchTPU, dietro le quinte avvengono
molte operazioni per ottimizzare l'esecuzione. Comprendere questo modello può
aiutarti a capire come ottimizzare la tua esperienza e a sfruttare alcuni dei
punti di forza esclusivi di TorchTPU.

## Pipeline di esecuzione { #execution-pipeline }

### Creazione della deferred op { #create-deferred-op }

Quando un'operazione PyTorch (o semplicemente "op") viene eseguita sul backend
TorchTPU, inizialmente sulla TPU non viene svolto alcun lavoro.

Il risultato logico del calcolo viene invece racchiuso in una deferred op
(operazione differita). Si tratta di un oggetto in memoria lato CPU che
contiene tutte le informazioni necessarie per eseguire l'operazione: i suoi
input, i passaggi logici da eseguire, la forma e il tipo degli output, e altro
ancora.

Il resto della pipeline, che sostituisce questa deferred op simbolica con dati
reali, si chiama materializzazione.

### Accodamento della op su uno stream { #enqueue-op-on-a-stream }

TorchTPU supporta l'interfaccia `torch.Stream`. Ogni volta che viene creata
una DeferredOp, questa viene inserita nello Stream attivo corrente, in coda per
l'esecuzione successiva.

A questo punto il thread chiamante ha terminato; tutto ciò che segue avviene
in modo asincrono in background (tranne che in modalità launch-blocking; ne
parliamo più avanti).

### Analisi dello stream e preparazione del piano di esecuzione { #analyze-stream-and-prepare-execution-plan }

Quando c'è del lavoro da svolgere su uno stream, a un certo punto un thread in
background preleva il lavoro da quello stream. Questo thread analizza poi il
lavoro sullo stream nel suo insieme, considerando molti fattori diversi, tra
cui:

*   In quante esecuzioni separate deve essere suddiviso?
*   Quali sono gli input e gli output di ogni blocco di esecuzione?
*   Quale livello di ottimizzazione della compilazione applicare a ciascuno?
*   Abbiamo già visto ogni unità di calcolo o è nuova?
*   Su quale core (TensorCore o SparseCore) deve essere eseguito il lavoro?

Considerando il lavoro da svolgere nel suo insieme, il thread in background
può prendere decisioni di esecuzione migliori.

Dopo aver identificato gli output finali di ogni blocco, il thread in
background li contrassegna come in attesa (pending) per indicare che verranno
eseguiti in futuro e passa alla fase successiva.

### Compilazione (o recupero) degli eseguibili { #compile-or-fetch-executables }

Per ogni fase di esecuzione, il thread in background controlla in una cache di
compilazione se è già stato compilato un eseguibile con le impostazioni
corrette. Per impostazione predefinita si tratta di una cache in memoria
locale (livello 1), con opzioni per attivare una cache persistente locale al
dispositivo (livello 2) o cache remote (livello 3) per ridurre al minimo i
costi di ricompilazione.

In caso di cache miss, il grafo di compilazione dell'unità viene convertito in
MLIR (principalmente StableHLO, tranne che per i kernel personalizzati definiti
dall'utente) e passato a un threadpool di compilazione. Questo threadpool
compila in modo asincrono molte unità di compilazione in parallelo e le
restituisce al thread worker in background quando sono pronte.

### Avvio dell'esecuzione { #start-execution }

Dopo aver raccolto gli input e preparato l'eseguibile, un'attività di
esecuzione viene accodata nel runtime di XLA,
[PjRt](https://openxla.org/xla/pjrt).

PjRt restituisce delle promise per questo output futuro, che vengono poi
collegate agli output per indicare che sono in fase di materializzazione.

### (Facoltativo) Attesa del completamento { #optional-wait-for-completion }

Per sapere quando il lavoro è completato, puoi registrare un `torch.Event`
sullo stream e usare il metodo `torch.Event.query` per esaminarlo.

In alternativa o in aggiunta, puoi bloccare l'esecuzione sullo stream con
qualsiasi istruzione di sincronizzazione (`torch.Event.synchronize`,
`torch.Stream.synchronize` o `torch.accelerator.synchronize` per l'intero
dispositivo), che mette in pausa il tuo codice Python finché la TPU non
segnala il completamento.

## Modalità di esecuzione { #execution-modes }

La pipeline di programmazione di TorchTPU è flessibile e supporta diverse
modalità di esecuzione.

### Modalità eager strict { #strict-eager-mode }

Nota anche come "eager predefinita" o "defer never", questa modalità è la
familiare esperienza PyTorch pronta all'uso. In questa modalità, ogni op
aggiunta alla coda contrassegna immediatamente lo stream per l'esecuzione. Di
conseguenza, il worker di materializzazione in background elabora le op una
alla volta.

Questi grafi di esecuzione a singola op hanno un potenziale di ottimizzazione
minimo, quindi vengono compilati con il livello di compilazione inferiore
`-O1` per massimizzare la velocità di compilazione. Poiché ci sono molte
compilazioni singole, queste vengono parallelizzate in modo molto efficace sul
threadpool di compilazione e il livello di riutilizzo per le singole
operazioni è molto alto. Questo comporta un overhead di compilazione minimo
durante il "riscaldamento" della cache di compilazione.

Nota che, anche se questa modalità esegue in modo eager e continuo, lo fa
comunque in modo asincrono: per osservare i risultati è comunque necessaria
una sincronizzazione che attenda il completamento della TPU, e gli eventuali
errori a runtime possono emergere molto dopo la op che li ha causati.

### Modalità eager di debug { #debug-eager-mode }

Nota anche come "launch blocking", questa modalità si attiva impostando la
variabile di ambiente `TORCH_TPU_LAUNCH_BLOCKING=1`. È pensata solo per il
debug interattivo, non per l'uso in produzione.

Questa modalità è uguale alla modalità eager strict (esecuzione op per op),
tranne per il fatto che l'host si blocca e si sincronizza a ogni operazione.
Questo rallenta artificialmente l'esecuzione, ma garantisce che gli errori
vengano segnalati non appena si verificano, per aiutarti a individuarli con
precisione.

### `torch.compile` { #torchcompile }

Per ottenere le massime prestazioni, `torch.compile` offre sia un maggiore
potenziale di ottimizzazione sia un overhead a runtime ridotto.

Quando compili una funzione o un modulo specifico con `torch.compile`, il
backend di TorchTPU riceve l'intero ambito del target di compilazione come
[grafo FX](https://docs.pytorch.org/docs/stable/fx.html). In questo modo
dispone della massima quantità di informazioni sull'esecuzione prevista,
insieme alla garanzia che non cambierà, e possiamo applicare ottimizzazioni
aggressive che sarebbero troppo rischiose in modalità eager.

Come backend in modalità compilata, TorchTPU esegue un tracciamento iniziale
(usando tensori "segnaposto", simili a FakeTensor ma che mantengono i metadati
specifici della TPU), producendo un grafo di calcolo completamente differito
che replica il grafo FX. Una volta creato, forniamo questo programma StableHLO
a XLA, che lo compila con il livello di ottimizzazione superiore `-O2`. Questo
programma viene quindi racchiuso in un callable Python e restituito
all'utente per l'esecuzione successiva.

Quando invochi questo callable, la maggior parte dei passaggi dell'esecuzione
eager viene saltata. Non vengono create nuove DeferredOp, non è necessaria
alcuna ulteriore analisi del grafo e l'eseguibile è già disponibile, quindi
non serve alcuna cache. L'eseguibile viene invece inviato immediatamente a
PjRt per l'esecuzione, riducendo al minimo il costo lato host con un grafo di
calcolo completamente ottimizzato dal compilatore XLA.

### Modalità eager ottimizzata { #optimized-eager-mode }

Nota anche come modalità "Defer and Fuse", si attiva impostando
`TORCH_TPU_DEFER_AND_FUSE=1`. È una nuova modalità ibrida che porta in
modalità eager molti dei vantaggi di `torch.compile`.

In questa modalità, la creazione di una nuova DeferredOp non avvia subito
l'esecuzione: le op restano differite tra una chiamata Python e l'altra, in
attesa nella coda dello Stream.

La materializzazione inizia quando si verifica una di queste due condizioni:

*   L'utente esegue un'azione che forza l'esecuzione. Le più comuni sono
    l'avvio di un trasferimento dal dispositivo all'host o una chiamata
    esplicita di sincronizzazione.
*   Rileviamo una ripetizione nelle op inviate, segno che ci troviamo in un
    loop e che conviene compilare un'iterazione di quel loop per riutilizzarla
    facilmente.

Questo segnala una materializzazione in background, simile alla modalità eager
predefinita, tranne per il fatto che il grafo compilato è più grande di una
singola operazione. I tensori già eliminati o sovrascritti al momento del
segnale di materializzazione vengono saltati: di solito vengono fusi nelle
operazioni a valle, oppure eliminati del tutto se l'output viene scartato
senza essere letto. (Nota: l'eliminazione del codice morto viene saltata per
le operazioni collettive distribuite, per evitare deadlock tra TPU).

Poiché questo produce grafi di compilazione più grandi, usiamo il livello di
ottimizzazione 2 (`-O2`) per migliorare il tempo di esecuzione. Dato che
compiliamo un numero minore di grafi più grandi, il threadpool di compilazione
offre un vantaggio di parallelizzazione ridotto, quindi i tempi di
compilazione saranno più lunghi rispetto alla modalità eager strict.

Anche se non è ottimizzata quanto la modalità `torch.compile`, la modalità
eager ottimizzata offre un miglioramento delle prestazioni fino al 50%
rispetto alla modalità eager predefinita, senza modifiche al codice del
modello.

## Problemi comuni { #pitfalls }

Anche se questa pipeline di esecuzione è ottimizzata sia per la latenza sia
per il throughput, ci sono casi in cui le prestazioni possono peggiorare.

### Tutte le modalità: flusso di controllo lato Python e forme dipendenti dai dati { #all-modes-python-side-control-flow-and-data-dependent-shapes }

In PyTorch è facile forzare, accidentalmente e senza accorgersene, un
trasferimento dal dispositivo all'host. Un'operazione come
"`if my_tensor.all()`" costruisce in modo silenzioso un tensore a valori
booleani, sincronizza lo stream e copia quel valore booleano sulla CPU prima
di proseguire con il flusso di controllo.

Nella modalità eager predefinita, questo non forza nuovi graph break, ma forza
lo svuotamento dell'intera pipeline di esecuzione dello stream prima di
proseguire. Peggio ancora, nella modalità eager ottimizzata o nella modalità
compilata, questo punto di sincronizzazione forzato causa un graph break nello
stream fuso in modo eager o nel grafo FX, creando una barriera
all'ottimizzazione.

Un comportamento simile può verificarsi anche con le operazioni PyTorch che
hanno forme di output dipendenti dai dati, come `torch.masked_select`. Poiché
la dimensione del tensore di output non è nota finché non viene valutato
l'input della maschera, anche in questo caso viene forzata una
sincronizzazione dello stream completamente bloccante e un trasferimento dal
dispositivo all'host dei metadati della forma prima di proseguire.

### Modalità eager strict: gli eseguibili molto piccoli causano microbolle { #strict-eager-mode-tiny-executables-cause-microbubbles }

Accodare un eseguibile con PjRt richiede poco tempo, di solito 50 microsecondi
o meno. Nella maggior parte dei casi, ogni eseguibile impiega più di questo
tempo (anche solo 1 o 2 millisecondi) e nasconde completamente questa
latenza, portando a uno stato stazionario in cui la TPU lavora continuamente
alla massima efficienza.

Tuttavia, se vengono accodati troppi eseguibili al di sotto di questa soglia,
ad esempio quando usi la modalità eager strict con tensori di piccole
dimensioni, la TPU può rimanere senza lavoro (starvation). Quando la TPU è più
veloce della CPU nell'accodare nuovo lavoro, l'esecuzione sulla TPU viene
interrotta da "microbolle" piccole ma frequenti, perché il collo di bottiglia
diventa la velocità di dispatch.

Per evitarlo, assicurati che i tuoi modelli svolgano una quantità di lavoro
sufficiente in ogni operazione da compensare questo piccolo overhead di
avvio, oppure usa la modalità eager ottimizzata per fondere automaticamente le
esecuzioni in blocchi di compilazione al volo.

### Modalità compilata: viste come argomenti o output { #compiled-mode-views-as-arguments-or-outputs }

Quando chiami un eseguibile compilato, PjRt richiede che i valori di input
siano contigui, con offset zero e di un tipo di dati specifico. Se un
argomento Tensor è una vista non banale o è stato sottoposto a bitcast dal
tipo di dati originale, TorchTPU crea una copia temporanea del tensore in
modalità eager (strict o ottimizzata) per chiamare correttamente l'eseguibile
compilato.

Per evitare questa copia temporanea, assicurati che tutti gli input di una
funzione compilata siano contigui. Se prevedi di chiamare più funzioni
compilate in sequenza, assicurati che gli output delle funzioni precedenti
siano contigui, per evitare attriti dovuti alla modalità eager tra le funzioni
ottimizzate.

### Modalità eager ottimizzata: mantenimento di tensori ridondanti { #optimized-eager-mode-retaining-redundant-tensors }

In tutte le modalità eager, le operazioni vengono eseguite nell'ordine in cui
vengono create e, ogni volta che viene emesso un segnale di
materializzazione, tutti i Tensor attivi in quel momento vengono
materializzati.

Se alcuni di questi tensori sono tensori intermedi non importanti, il
meccanismo defer-and-fuse può identificarli erroneamente come output
necessari e materializzarli, riducendo potenzialmente l'efficienza del grafo
di compilazione.

Per migliorare la fusione e ridurre al minimo l'uso della memoria, rilascia i
tensori non appena non ti servono più, oppure usa operazioni in-place per
sostituire i tensori obsoleti con i valori correnti.

## Punti chiave { #takeaways }

Il modello di esecuzione differita di TorchTPU offre un pipelining efficiente
con diversi compromessi tra tempo di compilazione e tempo di esecuzione.

*   **Per il tempo di compilazione minimo:** usa la modalità eager strict
    (quella predefinita). Questa modalità avvia subito l'esecuzione e
    parallelizza la compilazione per ottenere il primo risultato nel minor
    tempo possibile.
*   **Per il massimo throughput a regime e la latenza minima:** usa
    `torch.compile`. Elimina quasi tutto l'overhead della modalità eager e
    massimizza il potenziale di ottimizzazione del compilatore.
*   **Per i casi in cui preferisci un runtime più veloce ma `torch.compile` non
    è un'opzione:** valuta la modalità eager ottimizzata.
*   **Per il debug:** usa la modalità launch-blocking per far procedere CPU e
    TPU in sincronia.
*   **Evita le sincronizzazioni forzate:** mantieni un insieme minimo di
    tensori contigui e usa operazioni ad alta densità di calcolo per fornire
    alla TPU lavoro produttivo in modo costante.
