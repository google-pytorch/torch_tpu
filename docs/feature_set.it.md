# Funzionalità supportate da TorchTPU e lavori futuri { #torchtpu-supported-feature-set-and-future-work }

## Funzionalità attualmente supportate { #current-supported-feature-set }

La tabella seguente riporta la copertura attuale delle funzionalità alla data
della prima release open source.

Per aiutarti a valutare TorchTPU per i tuoi carichi di lavoro, suddividiamo le
funzionalità in due livelli:

*   **Convalidate:** funzionalità con test continui end-to-end, copertura
    CI/CD automatizzata e verifica tramite benchmark su architetture di
    modelli standard.
*   **Sperimentali:** implementazioni in fase iniziale, API emergenti o
    funzionalità prototipali. Sono funzionanti e disponibili per ricevere
    feedback iniziali, ma il loro comportamento o le loro API possono cambiare
    in base ai contributi della community.

> **Note:** TorchTPU si evolve rapidamente in modo aperto. Per conoscere lo
> stato più aggiornato o per richiedere funzionalità specifiche, visita il
> nostro [GitHub](https://github.com/google-pytorch/torch_tpu).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Convalidate** | **Sperimentali** |
| ----- | ----- | ----- |
| **Coinvolgimento della community PyTorch** | <ul><li>Release OSS di TorchTPU, vLLM, SGLang, TorchTitan, TorchRec</li><li>Accettazione di PR esterne</li><li>Documentazione / tutorial</li><li>CI/CD</li><li>Colab</li></ul> | <ul><li>Agente TPU per la migrazione di ATEN</li></ul> |
| **Supporto TPU** | <ul><li>v5e, v5p, v6e, v7</li></ul> |  |
| **Funzionalità di base** | <ul><li>Facilità di migrazione</li><li>Eager e compilato combinati</li><li>FW \+ BW</li><li>Torch.profile</li><li>Operazioni dinamiche (ricompilazione)</li></ul> | <ul><li>SymInts</li><li>Jagged tensor</li></ul> |
| **Architetture dei modelli** | <ul><li>Transformer (denso \+ MoE): Qwen, Llama, Muse Glimmer, Gemma</li><li>Diffusione: WAN, FLUX</li><li>CNN</li></ul> | <ul><li>GLM, Kimi, DeepSeek, GPT OSS e molti altri (hill climb)</li><li>Mamba</li><li>LSTM/GRU</li><li>DLRM</li></ul> |
| **Supporto dei kernel** | <ul><li>Pallas</li><li>Tokamax</li></ul> | <ul><li>Helion</li></ul> |
| **Quantizzazione** | <ul><li>FP8/FP4 a tensore intero</li><li>FP8 per canale</li></ul> |  |
| **Calcolo distribuito** | <ul><li>Mesh ottimale</li><li>SimpleFSDP</li><li>Checkpointing</li><li>Offloading delle attivazioni</li><li>Multi-slice fino a 8K chip</li></ul> | <ul><li>FSDPv2 (prestazioni basse)</li><li>MPMD (prestazioni basse)</li><li>TorchComm</li></ul> |
| **Integrazioni con l'ecosistema** | <ul><li>Hugging Face</li><li>Ray</li><li>vLLM / SGLang</li><li>TorchTitan</li><li>TorchRec</li></ul> |  |
| **Soluzioni pronte per la produzione** | <ul><li>Inferenza (multi-host, disaggregazione P/D)</li><li>Fine-tuning (fino a \~8K chip)</li></ul> |  |
<!-- mdformat on -->

## Lavori futuri { #future-work }

Questa sezione descrive i nostri orizzonti di sviluppo e le esplorazioni
future, non un calendario di release vincolante.

In quanto progetto open source orientato alla community, le nostre priorità di
sviluppo rispondono intenzionalmente al feedback della community e
all'adozione da parte degli sviluppatori. Funzionalità, tempistiche e approcci
architetturali possono evolvere man mano che capiamo quali carichi di lavoro
contano di più per te.

Organizziamo il lavoro in arrivo in due grandi orizzonti:

*   **Obiettivi a breve termine:** aree di progettazione attiva, prototipazione
    in corso e consolidamento delle prestazioni a cui il team principale sta
    dando priorità.
*   **Esplorazione futura:** esplorazioni architetturali a lungo termine,
    possibili integrazioni con l'ecosistema e richieste della community in
    backlog che stiamo valutando attivamente.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
|  | **Obiettivi a breve termine** | **Esplorazione futura** |
| ----- | ----- | ----- |
| **Coinvolgimento della community PyTorch** | <ul><li>Più documentazione / tutorial / blog</li></ul> | <ul><li>Agente TPU per la migrazione dei kernel</li></ul> |
| **Supporto TPU** | <ul><li>v8i, v8t</li></ul> |  |
| **Funzionalità di base** | <ul><li>Stream</li><li>Acquisizione del grafo</li><li>Modalità eager con avvio rapido</li><li>Operazioni dinamiche (senza ricompilazione)</li></ul> |  |
| **Architetture dei modelli** | <ul><li>Supporto dei modelli dal giorno 1</li></ul> |  |
| **Quantizzazione** | <ul><li>FP8/FP4 a blocchi con torchao (prestazioni di picco su TPUv8t)</li></ul> |  |
| **Calcolo distribuito** | <ul><li>FSDPv2 (con buone prestazioni)</li><li>MPMD (con buone prestazioni)</li><li>Addestramento \> 8K chip</li><li>Checkpointing multilivello</li><li>Goodput</li></ul> | <ul><li>Addestramento >> 8K chip</li></ul>  |
| **Integrazioni con l'ecosistema** | <ul><li>VeRL</li></ul> |  |
| **Soluzioni pronte per la produzione** | <ul><li>RL</li></ul> | <ul><li>Pre-addestramento</li></ul> |
<!-- mdformat on -->
