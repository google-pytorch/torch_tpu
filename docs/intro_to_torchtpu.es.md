# Introducción a TorchTPU { #introduction-to-torchtpu }

<!-- Link to colab and to generated notebook -->

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)

TorchTPU es un backend nativo de PyTorch de código abierto para
[Cloud TPU](https://cloud.google.com/tpu), desarrollado en colaboración entre
Google y el equipo de PyTorch de Meta. Con TorchTPU, los desarrolladores pueden
ejecutar cargas de trabajo de PyTorch existentes en hardware de TPU con cambios
mínimos en el código y escalar cargas de trabajo de entrenamiento e inferencia
desde un solo chip hasta pods de TPU de varias porciones (multi-slice).

## ¿Por qué usar TorchTPU? { #why-use-torchtpu }

Las TPU de Google son circuitos integrados de aplicación específica (ASIC)
diseñados para las operaciones matriciales y vectoriales en las que se basa el
aprendizaje profundo. Ofrecen alta capacidad de procesamiento, eficiencia
energética y buen rendimiento por dólar a gran escala. Aunque históricamente
las TPU se usaron con los ecosistemas de JAX y TensorFlow, TorchTPU lleva la
compatibilidad nativa con TPU directamente al
[ecosistema de PyTorch](https://github.com/google-pytorch/torch_tpu), con un
diseño basado de forma intencional en los siguientes tres principios:

*   **Integración nativa con PyTorch:** TorchTPU se integra directamente con
    PyTorch, sin dependencias del entorno de ejecución de JAX. Conserva la
    semántica estándar de PyTorch para vistas de tensores y mutaciones in situ
    (in-place), adopta el modelo de ejecución de PyTorch de un proceso por
    dispositivo y admite `torch.compile` a través de Dynamo.
*   **Facilidad de uso desde el primer momento:** Para portar una carga de
    trabajo de una GPU a una TPU, primero cambias `device="cuda"` por
    `device="tpu"` y `backend="nccl"` por `backend="tpu_dist"`. Tu modelo de
    PyTorch sin modificar se ejecuta de inmediato en modo eager, con mensajes de
    error claros para diagnosticar problemas del entorno de ejecución o del
    compilador.
*   **Rendimiento a escala de producción:** Además de la ejecución eager
    operación por operación, TorchTPU incorpora un modo eager con fusión
    diferida, una canalización de compilación de Dynamo a XLA, dinamismo
    acotado para evitar recompilaciones causadas por las formas y APIs de kernels
    personalizados que te ayudan a alcanzar el uso máximo del hardware.

## Guía de inicio rápido: Ejecución en TPU { #quickstart-running-on-tpu }

En la siguiente celda, puedes ver un ejemplo simple de un programa de PyTorch
que se ejecuta en TPU. Puedes ejecutar
[este ejemplo en Colab](https://colab.research.google.com/github/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/notebooks/intro_to_torchtpu.ipynb)
si habilitas el entorno de ejecución de TPU.

```python
import torch

device = torch.device("tpu")
x = torch.randn((4, 4), device=device)
y = torch.ones((4, 4), device=device)

result = torch.matmul(x, y).cpu()
print(f"Executed on device: {x.device}")
print(f"Result shape on CPU: {result.shape}, sum: {result.sum():.4f}")
```

## Funciones clave { #key-features }

*   **Cobertura amplia de operadores y modelos:** Admite 450 operadores
    principales de ATen (tanto en el pase hacia adelante como en el pase hacia
    atrás) y se valida de forma continua con 5,000 modelos populares de
    Hugging Face que abarcan LLM (Llama, Qwen, Gemma, DeepSeek, Mistral),
    modelos de visión (ViT, ResNet, ConvNeXt, DINOv2), modelos de audio
    (Whisper, Wav2Vec2) y canalizaciones de difusión (FLUX, Stable Diffusion).
*   **Tres modos de ejecución eager:**
    *   **Strict (predeterminado):** Despacha un operador de ATen a la vez de
        forma asíncrona a la TPU, igual que el comportamiento eager estándar de
        PyTorch en GPU.
    *   **Debug (`TORCH_TPU_LAUNCH_BLOCKING=1`):** Ejecuta cada operador de
        forma síncrona con la CPU y habilita la verificación de límites en el
        entorno de ejecución para simplificar la depuración.
    *   **DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`):** Difiere y fusiona
        secuencias de operadores de ATen en bloques de cálculo optimizados hasta
        un punto de materialización (como `.cpu()` o `.item()`). Esto reduce
        significativamente el tráfico de la memoria de alto ancho de banda (HBM)
        y acelera la ejecución eager.
*   **Compilación de grafos con `torch.compile`:** Registra el backend de
    Dynamo `"tpu"` (`torch.compile(model)`) para reducir grafos completos de
    PyTorch FX a [StableHLO](https://openxla.org/stablehlo/spec) y compilarlos
    con el [compilador de OpenXLA](https://openxla.org/xla). La compatibilidad
    activa con formas dinámicas (`torch._dynamo.mark_dynamic`) permite que los
    grafos compilados manejen tamaños de batch o longitudes de secuencia
    variables hasta un límite especificado sin recompilar.
*   **Dinamismo acotado:** Mitiga de forma nativa las recompilaciones causadas
    por tamaños de batch y longitudes de secuencia variables, ya que compila
    con límites superiores dinámicos.
*   **Caché de compilación de varios niveles:** Elimina compilaciones de XLA
    redundantes en tres niveles de caché:
    *   **Nivel 1:** Caché en memoria local del proceso (siempre habilitada).
    *   **Nivel 2:** Memoria compartida local del host para reutilizar entre
        procesos en trabajos de varios dispositivos.
    *   **Nivel 3:** Caché remota persistente (como Cloud Storage) para
        reutilizar entre hosts y reinicios de trabajos.
*   **Primitivas distribuidas nativas:** Implementa operaciones colectivas de
    TPU a nivel del grupo de procesos de `torch.distributed`
    (`c10d::Backend`) mediante `backend="tpu_dist"`, con compatibilidad con
    `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP2 y
    SimpleFSDP) y `DTensor`. También puedes usar
    `torch.tpu.distributed.topology_aware_mesh` para alinear las dimensiones de
    paralelismo con las interconexiones físicas de la TPU.
*   **Kernels de TPU personalizados:** Escribe kernels personalizados en
    [Pallas](https://docs.jax.dev/en/latest/pallas/index.html) (que se pueden
    llamar mediante `torch.tpu.pallas.jax_op`, con compatibilidad directa con
    operaciones colectivas RDMA entre chips), escribe kernels basados en tiles
    con el DSL experimental [Helion](https://pytorch.org/blog/helion/) o importa
    kernels de TPU optimizados y precompilados de Tokamax.
*   **Integraciones con el ecosistema:** Funciona desde el primer momento con
    frameworks estándar de serving y entrenamiento de PyTorch, y con paquetes
    de generación de perfiles, como `torch.profiler`
    (TensorBoard/XProf), vLLM, SGLang, TorchTitan, Ray y TorchRec.

## Limitaciones y trabajo futuro { #limitations-and-future-work }

Como cualquier producto, TorchTPU tiene limitaciones conocidas que se
resolverán con el tiempo. Aunque nuestra versión inicial ofrece una cobertura
sólida desde el primer momento para las principales cargas de trabajo, TorchTPU
es un esfuerzo colaborativo impulsado por la comunidad. Revisa nuestro
[conjunto de funciones y hoja de ruta](./feature_set.md) para ver lo que viene,
o participa directamente
[enviando PR](https://github.com/google-pytorch/torch_tpu) en GitHub para
ayudarnos a resolver estas limitaciones. Este es un proyecto de la comunidad, y
nos encantaría saber de ti e invitarte a colaborar con nosotros a través de
[GitHub](https://github.com/google-pytorch/torch_tpu),
[https://discuss.pytorch.org/c/tpu](https://discuss.pytorch.org/c/tpu),
[https://dev-discuss.pytorch.org/c/tpu/](https://dev-discuss.pytorch.org/c/tpu/)
y [torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com).

*   Las formas dinámicas causan recompilaciones
    *   `torch.compile` establece `dynamic=False` de forma predeterminada en
        TorchTPU. `dynamic=None` y `dynamic=True` todavía no son compatibles
        para producción.
        [Consulta el análisis sobre formas dinámicas](./dynamic_shapes.md)
    *   Por ejemplo, ejecutar el paso de decodificación o generación en LLM de
        Hugging Face implica recompilaciones en cada paso, a menos que se
        habilite el dinamismo acotado.
*   Las operaciones de vista pueden ser costosas en TPU (consulta
    [El modelo de memoria de TorchTPU](./torchtpu_memory_model.md)).
*   MPMD con operaciones colectivas es compatible a nivel funcional, pero aún no
    alcanza el rendimiento máximo. La superposición de operaciones colectivas
    con el cálculo tiene [algunas restricciones](./distributed_support.md):
    *   La compatibilidad completa con streams y la descarga a SparseCore
        llegarán en un futuro cercano.
    *   Usa SimpleFSDP con `torch.compile()` en lugar de FSDPv2 con eager o
        `torch.compile`; consulta
        [Prácticas recomendadas para FSDP](./case_studies_fsdp.md)
    *   La forma recomendada de obtener el máximo rendimiento de las
        optimizaciones de XLA en módulos que se comunican (es decir, de
        habilitar la superposición de cálculo y comunicación) es usar el
        decorador `spmd_safe` en regiones que **los usuarios garantizan** que
        son exactamente iguales
*   Actualmente, TorchTPU no tiene compatibilidad nativa con FlexAttention
*   Las TPU de la generación actual no admiten la acumulación de matmul con una
    precisión inferior a fp32 (por ejemplo, tf32).
*   Solo un proceso puede acceder a un dispositivo TPU, lo que limita la
    compatibilidad con RL coubicado (ten en cuenta que un chip TPU puede tener
    uno o más dispositivos asociados, y procesos separados del SO pueden
    acceder a ellos).

**Otras recomendaciones:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Limitación | Recomendación |
| :--- | :--- |
| Los modos eager tienen un tiempo de preparación distinto de cero | Habilita la caché de compilación persistente |
| Torch.compile tarda mucho tiempo | Habilita la caché de compilación persistente |
| Las CNN en PyTorch usan de forma predeterminada el diseño channel-first, que es muy ineficiente en TPU | Usa esta [opción de PyTorch](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) para establecer un diseño channel-last y obtener un mejor rendimiento. |
| La asignación de la malla de dispositivos puede ser subóptima de forma predeterminada debido a las diferencias de topología entre las TPU y otras arquitecturas | Usa `topology_aware_mesh` para obtener una asignación óptima de la malla de dispositivos y poder personalizarla |
<!-- mdformat on -->

## Primeros pasos { #getting-started }

Esta documentación se organiza en cuatro secciones estructuradas para ayudarte
a pasar de la primera configuración a la implementación distribuida en
producción:

*   **Primeros pasos:** Explicaciones paso a paso para configurar tu entorno e
    implementar cargas de trabajo en [una VM](./get_started_on_a_vm.md) o
    [a través de GKE](./get_started_on_gke.md).
*   **Información detallada:** Análisis detallados del
    [modelo de memoria de TPU](./torchtpu_memory_model.md), la
    [ejecución distribuida](./distributed_support.md) y los distintos
    [modos de ejecución](./torchtpu_execution_model.md).
*   **Casos prácticos:** Recetas de extremo a extremo que abarcan optimizaciones
    para el [modelo Flux](./case_studies_flux.md), el entrenamiento de
    [Muse Glimmer](./case_studies_muse_glimmer_30b_training.md), la
    [inferencia de LLM de extremo a extremo](./case_studies_muse_glimmer_30b_inference.md)
    y la aplicación eficaz de [FSDP](./case_studies_fsdp.md).
*   **Analiza tu carga de trabajo:**
    [Generación de perfiles](./performance_profiling_with_xprof.md) y gestión
    de la [precisión numérica](./numerical_precision.md)
