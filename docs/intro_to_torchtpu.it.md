# Introduzione a TorchTPU { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU è un backend PyTorch nativo e open source per le
[Cloud TPU di Google](https://cloud.google.com/tpu), sviluppato in
collaborazione tra Google e il team PyTorch di Meta. Con TorchTPU, gli
sviluppatori possono eseguire carichi di lavoro PyTorch esistenti su hardware
TPU con modifiche minime al codice e scalare i carichi di lavoro di
addestramento e inferenza da un singolo chip fino a pod TPU multi-slice.

## Perché usare TorchTPU? { #why-use-torchtpu }

Le TPU di Google sono circuiti integrati per applicazioni specifiche (ASIC)
progettati appositamente per le operazioni su matrici e vettori alla base del
deep learning, e offrono throughput elevato, efficienza energetica e un ottimo
rapporto prestazioni/costo su larga scala. Mentre in passato le TPU sono state
abbinate agli ecosistemi JAX e TensorFlow, TorchTPU porta il supporto nativo
delle TPU direttamente
nell'[ecosistema PyTorch](https://github.com/google-pytorch/torch_tpu), ed è
progettato intenzionalmente intorno a tre principi guida:

*   **Integrazione nativa con PyTorch:** TorchTPU si integra direttamente con
    PyTorch senza dipendenze dal runtime JAX. Mantiene la semantica standard
    di PyTorch per le viste dei tensori e le modifiche in-place, adotta il
    modello di esecuzione di PyTorch con un processo per dispositivo e
    supporta `torch.compile` tramite Dynamo.
*   **Usabilità immediata:** il porting di un carico di lavoro da GPU a TPU
    inizia sostituendo `device="cuda"` con `device="tpu"` e `backend="nccl"`
    con `backend="tpu_dist"`. Il tuo modello PyTorch, senza modifiche, viene
    eseguito subito in modalità eager, con messaggi di errore chiari per
    diagnosticare i problemi di runtime o del compilatore.
*   **Prestazioni su scala di produzione:** oltre all'esecuzione eager
    operazione per operazione, TorchTPU introduce una modalità eager con
    fusione differita, una pipeline di compilazione da Dynamo a XLA, un
    dinamismo limitato (bounded dynamism) per evitare le ricompilazioni dovute
    alle forme e API per kernel personalizzati che ti aiutano a raggiungere il
    massimo utilizzo dell'hardware.

## Guida rapida: esecuzione su TPU { #quickstart-running-on-tpu }

Nella cella seguente puoi vedere un semplice esempio di programma PyTorch
eseguito su TPU. Puoi eseguire
[questo esempio su Colab](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)
attivando il runtime TPU.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## Funzionalità principali { #key-features }

*   **Copertura completa di operatori e modelli:** supporta 450 operatori ATen
    core (sia per il forward pass sia per il backward pass) ed è convalidato
    in modo continuo su 5.000 modelli HuggingFace popolari, tra cui LLM
    (Llama, Qwen, Gemma, DeepSeek, Mistral), modelli di visione (ViT, ResNet,
    ConvNeXt, DINOv2), modelli audio (Whisper, Wav2Vec2) e pipeline di
    diffusione (FLUX, Stable Diffusion).
*   **Tre modalità di esecuzione eager:**
    *   **Strict (predefinita):** invia alla TPU un operatore ATen alla volta
        in modo asincrono, replicando il comportamento eager standard di
        PyTorch sulle GPU.
    *   **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** esegue ogni operatore in
        modo sincrono con la CPU e attiva il controllo dei limiti a runtime per
        semplificare il debug.
    *   **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** differisce e fonde
        sequenze di operatori ATen in blocchi di calcolo ottimizzati fino a un
        punto di materializzazione (come `.cpu()` o `.item()`), riducendo in
        modo significativo il traffico verso la High Bandwidth Memory (HBM) e
        accelerando l'esecuzione eager.
*   **Compilazione del grafo con `torch.compile`:** registra il backend Dynamo
    `"tpu"` (`torch.compile(model)`) per abbassare (lowering) interi grafi FX
    di PyTorch in [StableHLO](https://openxla.org/stablehlo/spec) e compilarli
    con il [compilatore OpenXLA](https://openxla.org/xla). Il supporto attivo
    per le forme dinamiche (`torch._dynamo.mark_dynamic`) consente ai grafi
    compilati di gestire batch size o lunghezze di sequenza variabili fino a
    un limite specificato senza ricompilare.
*   **Dinamismo limitato:** riduce in modo nativo le ricompilazioni causate da
    batch size e lunghezze di sequenza variabili, compilando rispetto a limiti
    superiori dinamici.
*   **Cache di compilazione multilivello:** elimina le compilazioni XLA
    ridondanti con tre livelli di cache:
    *   **Livello 1:** cache in memoria locale al processo (sempre attiva).
    *   **Livello 2:** memoria condivisa locale all'host per il riutilizzo tra
        processi nei job multi-dispositivo.
    *   **Livello 3:** cache remota persistente (come Cloud Storage) per il
        riutilizzo tra host e tra riavvii dei job.
*   **Primitive distribuite native:** implementa le operazioni collettive TPU
    a livello di Process Group di `torch.distributed` (`c10d::Backend`)
    tramite `backend="tpu_dist"`, con supporto per `DistributedDataParallel`
    (DDP), `FullyShardedDataParallel` (FSDP2 e SimpleFSDP) e `DTensor`. Puoi
    anche usare `torch.tpu.distributed.topology_aware_mesh` per allineare le
    dimensioni del parallelismo alle interconnessioni fisiche delle TPU.
*   **Kernel TPU personalizzati:** scrivi kernel personalizzati in
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (richiamabili
    tramite `torch.tpu.pallas.jax_op` con supporto diretto per le operazioni
    collettive RDMA tra chip), scrivi kernel basati su tile con il DSL
    sperimentale [Helion](https://pytorch.org/blog/helion/) oppure importa
    kernel TPU ottimizzati e già pronti da Tokamax.
*   **Integrazioni con l'ecosistema:** funziona subito con i framework standard
    di PyTorch per serving e addestramento e con gli strumenti di profilazione,
    tra cui `torch.profiler` (TensorBoard/XProf), vLLM, SGLang, TorchTitan, Ray
    e TorchRec.

## Limitazioni e lavori futuri { #limitations-and-future-work }

Come ogni prodotto, TorchTPU ha delle limitazioni note, che verranno risolte
nel tempo. Anche se la prima release offre una copertura solida e immediata
per i principali carichi di lavoro, TorchTPU è un progetto collaborativo
guidato dalla community. Consulta le nostre
[funzionalità e roadmap](./feature_set.md) per vedere cosa è in arrivo, oppure
contribuisci subito [inviando PR](https://github.com/google-pytorch/torch_tpu)
su GitHub per aiutarci a risolvere queste limitazioni. Questo è un progetto
della community: ci farebbe piacere ricevere il tuo parere e ti invitiamo a
collaborare con noi tramite
[GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
e [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

*   Le forme dinamiche causano ricompilazioni
    *   Su TorchTPU, `torch.compile` imposta `dynamic=False` per impostazione
        predefinita. `dynamic=None` o `dynamic=True` non sono ancora supportati
        in produzione.
        [Consulta la discussione sulle forme dinamiche](./dynamic_shapes.md)
    *   Ad esempio, l'esecuzione dello step di decodifica/generazione sugli LLM
        di Hugging Face comporta ricompilazioni a ogni step, a meno che non sia
        attivo il dinamismo limitato.
*   Le operazioni di vista possono essere costose su TPU (consulta
    [Il modello di memoria di TorchTPU](./torchtpu_memory_model.md)).
*   MPMD con operazioni collettive è supportato a livello funzionale, ma non
    raggiunge ancora le prestazioni di picco. La sovrapposizione delle
    operazioni collettive con il calcolo ha
    [alcuni vincoli](./distributed_support.md):
    *   Il supporto completo degli stream con offload su SparseCore arriverà
        nel prossimo futuro.
    *   Usa SimpleFSDP con `torch.compile()` invece di FSDPv2 in modalità eager
        o con `torch.compile`; consulta
        [Best practice per FSDP](./case_studies_fsdp.md)
    *   Usare il decorator `spmd_safe` sulle regioni che **gli utenti
        garantiscono** essere esattamente identiche è il modo consigliato per
        ottenere le massime prestazioni dalle ottimizzazioni XLA sui moduli che
        comunicano (ovvero per abilitare la sovrapposizione tra calcolo e
        comunicazione)
*   Al momento TorchTPU non supporta in modo nativo FlexAttention
*   Le TPU della generazione attuale non supportano l'accumulo delle matmul in
    una precisione inferiore a fp32 (ad esempio tf32).
*   Solo un processo può accedere a un dispositivo TPU, il che limita il
    supporto per l'RL co-locato (nota che a un chip TPU possono essere
    associati uno o più dispositivi, accessibili da processi del sistema
    operativo separati).

**Altri consigli:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Limitazione | Consiglio |
| :--- | :--- |
| Le modalità eager hanno un tempo di warmup non nullo | Attiva la cache di compilazione persistente |
| Torch.compile richiede molto tempo | Attiva la cache di compilazione persistente |
| Le CNN in PyTorch, che per impostazione predefinita usano il layout channel-first, sono molto inefficienti su TPU | Usa questa [opzione di PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) per impostare un layout channel-last e ottenere prestazioni migliori. |
| La mappatura della device mesh può essere non ottimale per impostazione predefinita a causa delle differenze di topologia tra le TPU e le altre architetture | Usa `topology_aware_mesh` per una mappatura ottimale della device mesh e per poterla personalizzare |
<!-- mdformat on -->

## Per iniziare { #getting-started }

Questa documentazione è organizzata in quattro sezioni strutturate per
accompagnarti dalla prima configurazione fino al deployment distribuito in
produzione:

*   **Inizia:** procedure dettagliate per configurare l'ambiente ed eseguire il
    deployment dei carichi di lavoro su [una VM](./get_started_on_a_vm.md) o
    [tramite GKE](./get_started_on_gke.md).
*   **Approfondimenti:** analisi dettagliate del
    [modello di memoria della TPU](./torchtpu_memory_model.md),
    dell'[esecuzione distribuita](./distributed_support.md) e delle diverse
    [modalità di esecuzione](./torchtpu_execution_model.md).
*   **Casi di studio:** procedure end-to-end che coprono le ottimizzazioni per
    il [modello Flux](./case_studies_flux.md), l'addestramento di
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md),
    l'[inferenza LLM end-to-end](./case_studies_muse_glimmer_30b_inference.md)
    e l'applicazione efficace di [FSDP](./case_studies_fsdp.md).
*   **Analizza il tuo carico di lavoro:**
    [profilazione](./performance_profiling_with_xprof.md) e gestione della
    [precisione numerica](./numerical_precision.md)
