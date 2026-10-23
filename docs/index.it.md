<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Forum di discussione</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch su TPU</b></a>
</p>

## Informazioni { #about }

TorchTPU è un backend PyTorch nativo e open source per le Cloud TPU di Google,
sviluppato in collaborazione tra Google e Meta. Con TorchTPU puoi eseguire
modelli PyTorch esistenti su hardware TPU con modifiche minime al codice e
scalare i carichi di lavoro di addestramento e inferenza da un singolo chip
fino a pod TPU multi-slice.

## Cosa vuoi fare oggi? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Inizia**

    Esegui il provisioning degli ambienti, comprendi le architetture fondamentali ed esegui i tuoi primi carichi di lavoro.

    - [<!--mkdocs::octicons-arrow-right-24:--> Introduzione a TorchTPU](./intro_to_torchtpu.md): perché è nato TorchTPU, cosa offre e come può aiutarti.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inizia a utilizzare TorchTPU su una VM](./get_started_on_a_vm.md): configura TorchTPU su una VM TPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inizia a utilizzare TorchTPU su GKE](./get_started_on_gke.md): usa TorchTPU tramite GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Funzionalità supportate e roadmap](./feature_set.md): le funzionalità attualmente supportate e i piani per le release future.
    - [<!--mkdocs::octicons-arrow-right-24:--> Riferimento API](./api_reference.md): l'API TorchTPU e le variabili di ambiente della release attuale.
    - [<!--mkdocs::octicons-arrow-right-24:--> Domande frequenti su TorchTPU](./faq.md): le domande più frequenti su TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> Architettura e concetti fondamentali in dettaglio**

    Approfondisci il funzionamento dei layout di memoria, dei motori di esecuzione e dell'esecuzione distribuita.

    - [<!--mkdocs::octicons-arrow-right-24:--> Come funziona TorchTPU](./how_torchtpu_works.md): come TorchTPU si integra con PyTorch e con l'hardware TPU, dall'alto verso il basso.
    - [<!--mkdocs::octicons-arrow-right-24:--> Il modello di esecuzione di TorchTPU](./torchtpu_execution_model.md): una descrizione dettagliata di come TorchTPU esegue il codice PyTorch e delle diverse modalità di esecuzione che supporta.
    - [<!--mkdocs::octicons-arrow-right-24:--> Il modello di memoria di TorchTPU](./torchtpu_memory_model.md): come TorchTPU gestisce il modo in cui la TPU e il compilatore XLA trattano la memoria.
    - [<!--mkdocs::octicons-arrow-right-24:--> Supporto per il calcolo distribuito](./distributed_support.md): come TorchTPU supporta i carichi di lavoro distribuiti.
    - [<!--mkdocs::octicons-arrow-right-24:--> Supporto dei kernel personalizzati](./custom_kernels.md): come scrivere kernel personalizzati per TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Forme dinamiche](./dynamic_shapes.md): supporto per le forme dinamiche dei tensori.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Casi di studio**

    Esplora esempi pratici di come usare TorchTPU per risolvere problemi reali.

    - [<!--mkdocs::octicons-arrow-right-24:--> Ottimizzazione di FLUX](./case_studies_flux.md): un esempio che mostra come individuare e applicare più ottimizzazioni a FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inferenza di Muse Glimmer 30B](./case_studies_muse_glimmer_30b_inference.md): un esempio di decodifica iterativa con pesi reali e KV cache statica/dinamica.
    - [<!--mkdocs::octicons-arrow-right-24:--> Addestramento di Muse Glimmer 30B](./case_studies_muse_glimmer_30b_training.md): un loop di addestramento su un'implementazione autonoma di Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Uso di FSDP](./case_studies_fsdp.md): l'approccio consigliato per usare FSDP con TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analizza il tuo carico di lavoro**

    - [<!--mkdocs::octicons-arrow-right-24:--> Precisione numerica e parità](./numerical_precision.md): come controllare la precisione in TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Profila le prestazioni con XProf](./performance_profiling_with_xprof.md): come usare XProf come profiler per TorchTPU.

<!--mkdocs:</div>-->
<!-- mdformat on -->
