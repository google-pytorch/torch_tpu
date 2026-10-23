<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Foro de discusión</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch en TPU</b></a>
</p>

## Acerca de { #about }

TorchTPU es un backend nativo de PyTorch de código abierto para Cloud TPU,
desarrollado en colaboración entre Google y Meta. Con TorchTPU, puedes ejecutar
modelos de PyTorch existentes en hardware de TPU con cambios mínimos en el
código y escalar cargas de trabajo de entrenamiento e inferencia desde un solo
chip hasta pods de TPU de varias porciones (multi-slice).

## ¿Qué quieres hacer hoy? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Primeros pasos**

    Aprovisiona entornos, comprende las arquitecturas fundamentales y ejecuta tus primeras cargas de trabajo.

    - [<!--mkdocs::octicons-arrow-right-24:--> Introducción a TorchTPU](./intro_to_torchtpu.md): Por qué se creó TorchTPU, qué ofrece y cómo puede ayudarte.
    - [<!--mkdocs::octicons-arrow-right-24:--> Primeros pasos con TorchTPU en una VM](./get_started_on_a_vm.md): Configura TorchTPU en una VM de TPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Primeros pasos con TorchTPU en GKE](./get_started_on_gke.md): Usa TorchTPU a través de GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Conjunto de funciones compatibles y hoja de ruta](./feature_set.md): Funciones compatibles actualmente y planes para versiones futuras.
    - [<!--mkdocs::octicons-arrow-right-24:--> Referencia de la API](./api_reference.md): La API de TorchTPU y las variables de entorno de la versión actual.
    - [<!--mkdocs::octicons-arrow-right-24:--> Preguntas frecuentes sobre TorchTPU](./faq.md): Preguntas frecuentes sobre TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> Arquitectura en detalle y conceptos fundamentales**

    Profundiza en el funcionamiento de los diseños de memoria, los motores de ejecución y la ejecución distribuida.

    - [<!--mkdocs::octicons-arrow-right-24:--> Cómo funciona TorchTPU](./how_torchtpu_works.md): Cómo se integra TorchTPU con PyTorch y el hardware de TPU, de principio a fin.
    - [<!--mkdocs::octicons-arrow-right-24:--> El modelo de ejecución de TorchTPU](./torchtpu_execution_model.md): Detalles de cómo TorchTPU ejecuta código de PyTorch y los distintos modos de ejecución que admite.
    - [<!--mkdocs::octicons-arrow-right-24:--> El modelo de memoria de TorchTPU](./torchtpu_memory_model.md): Cómo TorchTPU maneja la forma en que la TPU y el compilador XLA gestionan la memoria.
    - [<!--mkdocs::octicons-arrow-right-24:--> Compatibilidad con la ejecución distribuida](./distributed_support.md): Cómo TorchTPU admite cargas de trabajo distribuidas.
    - [<!--mkdocs::octicons-arrow-right-24:--> Compatibilidad con kernels personalizados](./custom_kernels.md): Cómo escribir kernels personalizados para TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Formas dinámicas](./dynamic_shapes.md): Compatibilidad con formas de tensores dinámicas.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Casos prácticos**

    Explora ejemplos prácticos de cómo usar TorchTPU para resolver problemas reales.

    - [<!--mkdocs::octicons-arrow-right-24:--> Optimización de FLUX](./case_studies_flux.md): Ejemplo que muestra cómo identificar y aplicar varias optimizaciones en FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inferencia de Muse Glimmer 30B](./case_studies_muse_glimmer_30b_inference.md): Ejemplo de decodificación iterativa con pesos reales y caché de KV estática o dinámica.
    - [<!--mkdocs::octicons-arrow-right-24:--> Entrenamiento de Muse Glimmer 30B](./case_studies_muse_glimmer_30b_training.md): Un bucle de entrenamiento en una implementación independiente de Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Uso de FSDP](./case_studies_fsdp.md): El enfoque recomendado para usar FSDP con TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analiza tu carga de trabajo**

    - [<!--mkdocs::octicons-arrow-right-24:--> Precisión numérica y paridad](./numerical_precision.md): Cómo controlar la precisión en TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Genera perfiles de rendimiento con XProf](./performance_profiling_with_xprof.md): Cómo usar XProf como generador de perfiles de TorchTPU.

<!--mkdocs:</div>-->
<!-- mdformat on -->
