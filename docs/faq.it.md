# Domande frequenti su TorchTPU { #torchtpu-faq }

## Qual è la differenza tra PyTorch/XLA e TorchTPU? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

In passato PyTorch/XLA ha fatto da framework ponte, fortemente dipendente dai
componenti del runtime JAX e da API specializzate. TorchTPU offre un backend
PyTorch nativo per le [Google Cloud TPU](https://cloud.google.com/tpu) senza
alcuna dipendenza dal runtime JAX, e i carichi di lavoro PyTorch esistenti funzionano
subito con modifiche minime al codice (ad esempio, `.to("cuda")` → `.to("tpu")`)

## Qual è la differenza tra TorchAX e TorchTPU? { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX (noto anche come torch_xla2) è un'implementazione che intercetta le
operazioni PyTorch a livello Python e le abbassa (lowering) in XLA (spesso
tramite JAX). TorchAX è stato molto usato per benchmark di inferenza e
prototipazione, ma TorchTPU è il nuovo standard ufficiale per un'esperienza
PyTorch nativa e drop-in sulle TPU.

## Cosa succede a PyTorch/XLA? { #what-is-happening-to-pytorchxla }

PyTorch/XLA è deprecato e sostituito da
[TorchTPU](https://github.com/google-pytorch/torch_tpu), che ora è pubblico.

## TorchTPU sostituisce JAX? { #is-torchtpu-replacing-jax }

No. TorchTPU è pensato per completare l'ecosistema TPU offrendo un punto di
accesso PyTorch nativo alle Cloud TPU. Permette agli sviluppatori con codebase
PyTorch esistenti di eseguirle direttamente sulle TPU senza riscritture o con riscritture
minime, senza dover riscrivere i modelli in JAX o gestire codebase separate
per l'infrastruttura GPU e TPU.

*   **Quando considerare TorchTPU?** Se hai una codebase PyTorch esistente, vuoi
    la portabilità tra hardware diversi (GPU/TPU) senza refactoring, usi
    framework PyTorch nativi (ad esempio TorchTitan, vLLM, SGLang, Hugging Face)
    o vuoi migrare rapidamente da GPU a TPU.

## Qual è il modo migliore per entrare in contatto con gli sviluppatori di TorchTPU? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

Puoi inviare feedback via email a
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com) e i
contributi diretti al repository sono benvenuti:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
Puoi anche usare il tag "TPU" su
[discuss.pytorch.org](https://discuss.pytorch.org/).

## Ci saranno forum per la community? { #will-there-be-any-community-engagement-forums }

Sì, il coinvolgimento della community è molto importante per noi e
condivideremo i dettagli su questi forum. Inoltre, prevediamo di organizzare
office hours mensili.

## Che supporto offre TorchTPU per lo sviluppo di kernel? { #what-support-does-torchtpu-provide-for-kernel-development }

Consulta [Supporto dei kernel personalizzati di TorchTPU](./custom_kernels.md).

## Quali generazioni di TPU sono supportate attivamente? Qual è il piano per il supporto dell'hardware futuro? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU supporta attivamente v5e, v5p, Trillium (TPU v6e) e Ironwood (TPU
v7x). La convalida e l'ottimizzazione per TPU v8 e i chip futuri sono in corso
come parte della nostra roadmap.

## Dove trovo le funzionalità supportate e la roadmap futura di TorchTPU? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

Consulta [Funzionalità supportate](./feature_set.md) per il set di funzionalità
attuale e quello previsto in futuro.

## Esiste un agente che semplifica l'onboarding? { #is-there-an-agent-to-make-my-onboarding-easier }

Prova la TPU agent suite (sperimentale):
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

La TPU Agent Suite è un plug-in open source che fornisce sub-agenti
specializzati per automatizzare onboarding, diagnostica e ottimizzazione delle
prestazioni di inferenza per i modelli PyTorch su Cloud TPU v6e e v7x.

## Posso usare TorchTPU con la nostra soluzione di serving personalizzata? { #can-i-use-torchtpu-with-our-custom-serving-solution }

Sì. TorchTPU offre un'integrazione pronta all'uso con motori di serving standard
del settore come vLLM e SGLang, ma anche le soluzioni di serving personalizzate o
interne possono integrarsi con TorchTPU tramite le API di esecuzione PyTorch
native e i backend del compilatore di livello più basso. Per un'orchestrazione
avanzata, i clienti possono anche eseguire il deployment di configurazioni di
serving personalizzate su GKE con llm-d.
