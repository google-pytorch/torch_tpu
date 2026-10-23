# Referencia de la API de TorchTPU { #torchtpu-api-reference }

Este documento es la guía de referencia de las APIs públicas de Python y las
variables de entorno de TorchTPU. Está diseñado para ayudarte a hacer lo
siguiente:

*   **Migrar de CUDA a TPU**: Asigna las llamadas comunes de `torch.cuda` a las
    APIs
    [`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
    de PyTorch, que son independientes del dispositivo.
*   **Usar funciones específicas de TPU**: Configura los modos de ejecución, la
    precisión numérica de la MXU, la topología distribuida, los operadores
    personalizados de Pallas/TPU, los optimizadores trazables y la generación
    de perfiles con XProf mediante el módulo **`torch.tpu`**.
*   **Configurar el entorno de ejecución**: Ajusta la caché de compilación, las
    verificaciones de depuración y la configuración de lanzamiento en varios
    hosts mediante variables de entorno.

> **Note:** Para ver la lista exhaustiva y generada automáticamente de todas las
> APIs públicas de Python y las variables de entorno, que es la fuente de
> referencia, consulta
> [APIs públicas de TorchTPU (`public_apis.md`)](https://github.com/google-pytorch/torch_tpu/blob/0.2.0-docs/docs/public_apis.md).

## **Consultas básicas de dispositivos** { #core-device-queries }

Cuando se importa `torch` con TorchTPU instalado, el backend registra
automáticamente el dispositivo `"tpu"` y el módulo `torch.tpu`. Cuando PyTorch
proporcione una API
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html)
independiente del dispositivo, úsala en lugar de una API específica del backend
para que tu código siga siendo portable entre aceleradores. Usa `torch.tpu`
para las propiedades de hardware específicas de TPU que no tienen un equivalente
genérico.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API de CUDA | API de TPU / Accelerator | Descripción |
| :--- | :--- | :--- |
| `torch.device("cuda")` | `torch.accelerator.current_accelerator()` | Devuelve `torch.device("tpu")` cuando la TPU es el acelerador activo. |
| `torch.cuda.is_available()` | `torch.accelerator.is_available()` | Verifica si el backend de TPU está disponible. |
| `torch.cuda.is_initialized()` | `torch.tpu.is_initialized()` | Verifica si se inicializó el estado del entorno de ejecución de TorchTPU. |
| `torch.cuda.device_count()` | `torch.accelerator.device_count()` | Cantidad de dispositivos visibles para el proceso actual. |
| `torch.cuda.current_device()` | `torch.accelerator.current_device_index()` | Devuelve el índice del dispositivo de TPU activo. |
| `torch.cuda.set_device(rank)` | `torch.accelerator.set_device_index(rank)` | Establece el índice del dispositivo de TPU activo para el proceso actual. |
| `torch.cuda.device(rank)` | `torch.accelerator.device_index(rank)` | Administrador de contexto que establece temporalmente el índice del dispositivo activo. |
| `torch.cuda.get_device_name(device=None)` | `torch.tpu.get_device_properties(device=None).name` | Devuelve el nombre legible del modelo de la TPU conectada (por ejemplo, `"TPU v6e"`). |
| `torch.cuda.get_device_properties(device=None)` | `torch.tpu.get_device_properties(device=None)` | Devuelve las propiedades estáticas del dispositivo de TPU (incluidas `name`, `devices_per_chip`, `hbm_bytes_per_chip`, `hbm_bytes_per_device`, `bf16_flops_per_chip`, `bf16_flops_per_device`, `sparse_cores_per_device` y `sparse_core_simd_width`). |
<!-- mdformat on -->

Las APIs de `torch.accelerator` de esta tabla requieren PyTorch 2.11 o una
versión posterior. En el caso poco común de que uses una versión anterior de
PyTorch, usa la API de `torch.tpu` correspondiente (por ejemplo,
`torch.tpu.device_count()` o `torch.tpu.current_device()`). En PyTorch 2.11 y
versiones posteriores, esas APIs redundantes de `torch.tpu` son internas de
TorchTPU, y llamarlas directamente desde el código de usuario genera un error.

```python
import torch

# Initialize hardware
device = torch.device("tpu")

print(f"torch.accelerator.is_available(): {torch.accelerator.is_available()}")
print(f"torch.accelerator.device_count(): {torch.accelerator.device_count()}")
print(
    "torch.accelerator.current_device_index():"
    f" {torch.accelerator.current_device_index()}"
)
print(
    "torch.tpu.get_device_properties().name:"
    f" {torch.tpu.get_device_properties().name}"
)
```

## **Generación de números aleatorios (RNG)** { #random-number-generation-rng }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API de CUDA | API de TPU / Accelerator | Descripción |
| :--- | :--- | :--- |
| `torch.cuda.manual_seed(seed)` | `torch.tpu.manual_seed(seed)` | Establece la semilla del generador de números aleatorios del dispositivo de TPU activo. |
| `torch.cuda.manual_seed_all(seed)` | `torch.tpu.manual_seed_all(seed)` | Establece la semilla del RNG en todos los dispositivos de TPU locales. |
| `torch.cuda.initial_seed()` | `torch.accelerator.random.initial_seed()` | Devuelve la semilla aleatoria inicial del dispositivo de TPU actual. |
| `torch.cuda.get_rng_state()` | `torch.accelerator.random.get_rng_state()` | Devuelve el tensor de estado del RNG del dispositivo actual. |
| `torch.cuda.set_rng_state(new_state, device=...)` | `torch.tpu.set_rng_state(new_state, device=...)` | Establece el tensor de estado del RNG del dispositivo de TPU activo del proceso actual. |
| `torch.cuda.default_generators` | `torch.tpu.default_generators` | Tupla, inicializada de forma diferida, de instancias predeterminadas de `torch.Generator` indexadas por el ordinal del dispositivo local. |
<!-- mdformat on -->

Las APIs de `torch.accelerator.random` de esta tabla requieren PyTorch 2.13 o
una versión posterior. En el caso poco común de que uses una versión anterior
de PyTorch, usa `torch.tpu.initial_seed()` o `torch.tpu.get_rng_state()`. En
PyTorch 2.13 y versiones posteriores, estas APIs de `torch.tpu` son internas de
TorchTPU, y llamarlas desde tu código genera un error.

```python
# Set the seed across all cores
torch.tpu.manual_seed_all(42)
print("Seeded all TPU cores with value 42.")
```

## **Precisión mixta automática (AMP)** { #automatic-mixed-precision-amp }

TorchTPU se integra con el framework estándar
[`torch.amp.autocast`](https://docs.pytorch.org/docs/stable/amp.html) de
PyTorch mediante la cadena de tipo de dispositivo `"tpu"`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API de CUDA | API de TPU | Descripción |
| :--- | :--- | :--- |
| `torch.amp.autocast("cuda", ...)` | `torch.amp.autocast("tpu", ...)` | Administrador de contexto o decorador que habilita la precisión mixta automática (`bfloat16` de forma predeterminada en TPU). |
| `torch.cuda.get_amp_supported_dtype()` | `torch.tpu.get_amp_supported_dtype()` | Enumera los dtypes compatibles en TPU (`bfloat16`, `float16`, `float32`, `float64`, `float8_*`, y los tipos enteros, booleanos y complejos). |
<!-- mdformat on -->

```python
supported_dtypes = torch.tpu.get_amp_supported_dtype()
print(f"AMP supported dtypes: {supported_dtypes}")
```

## **Streams y eventos** { #streams-and-events }

Para la administración de streams y la sincronización de dispositivos, usa las
APIs estándar de
[`torch.accelerator`](https://docs.pytorch.org/docs/stable/accelerator.html) en
lugar de APIs específicas del backend. Usa `torch.tpu` solo cuando construyas
objetos `Stream` o `Event` de TPU directamente o cuando configures la fijación
(pinning) de núcleos de TPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API de CUDA | API de TPU / Accelerator | Descripción |
| :--- | :--- | :--- |
| `torch.cuda.current_stream()` | `torch.accelerator.current_stream()` | Devuelve el stream activo de un dispositivo. |
| `torch.cuda.set_stream(stream)` | `torch.accelerator.set_stream(stream)` | Establece el stream activo. |
| `torch.cuda.default_stream(device=None)` | `torch.tpu.default_stream(device=None)` | Devuelve el `TpuStream` predeterminado del dispositivo. |
| `torch.cuda.stream(stream)` | `torch.tpu.stream(stream)` | Administrador de contexto que selecciona un `TpuStream` determinado. |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` | Espera a que se complete todo el trabajo pendiente en el dispositivo de TPU. |
| `torch.cuda.Stream` | `torch.tpu.Stream(device=None, priority=0, *, pinned=None)` | Es una subclase de `torch.Stream`. Admite `synchronize()`, `record_event()`, `wait_event()`, `wait_stream()` y la fijación de núcleos (`pinned="dense"` o `"sparse"`, que se consulta mediante `is_dense` / `is_sparse`). |
| `torch.cuda.Event` | `torch.tpu.Event(enable_timing=False)` | Es una subclase de `torch.Event`. Admite `record()`, `synchronize()`, `query()` y `elapsed_time(end_event)`. |
<!-- mdformat on -->

Las APIs de `torch.accelerator` de esta tabla requieren PyTorch 2.11 o una
versión posterior. En el caso poco común de que uses una versión anterior de
PyTorch, usa `torch.tpu.current_stream()`, `torch.tpu.set_stream()` o
`torch.tpu.synchronize()`.

> **Note:** Los streams son un concepto de CUDA sin una analogía directa en TPU.
> En el modelo de ejecución diferida de TorchTPU, el compilador XLA y el entorno
> de ejecución de PjRt programan y superponen las operaciones automáticamente
> según las dependencias de datos, por lo que no se necesita administrar varios
> streams de forma manual.

## **APIs específicas de TPU** { #tpu-specific-apis }

Además de los equivalentes estándar de CUDA, `torch.tpu` expone APIs diseñadas
específicamente para los modos de ejecución de TPU, la precisión de la MXU, la
compilación de XLA, la topología distribuida, los kernels personalizados de
Pallas, los optimizadores trazables y la generación de perfiles.

### **Modos de ejecución y opciones del compilador** { #execution-modes-and-compiler-options }

Controla cómo XLA pone en cola, fusiona y compila las operaciones eager. Para
ver una explicación de la arquitectura, consulta
[Modos de ejecución de TorchTPU](./torchtpu_execution_model.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.EagerMode` | Enum que controla la ejecución eager: `DEFER_AND_FUSE`, `DEFER_NEVER` y `DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `torch.tpu.eager_mode` | Obtiene o establece el `EagerMode` activo de forma global. |
| `torch.tpu.set_eager_mode(mode)` | Administrador de contexto que establece temporalmente el `EagerMode` activo. |
| `torch.tpu.compiler.custom_compiler_options(options)` | Administrador de contexto que pasa marcas personalizadas del compilador XLA a las operaciones eager que se materializan dentro del bloque. |
<!-- mdformat on -->

```python
print(f"Default eager_mode: {torch.tpu.eager_mode}")
with torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE):
  x = torch.ones((4, 4), device="tpu") + 1
  print(
      f"Inside context manager: {torch.tpu.eager_mode}, sum={x.sum().item()}"
  )
```

### **Precisión numérica** { #numerical-precision }

Controla la precisión de la unidad de multiplicación de matrices (MXU) y la
acumulación intermedia en `float32`. Para obtener orientación detallada,
consulta [Precisión numérica](./numerical_precision.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.Precision` | Enum que controla la precisión de la MXU para las multiplicaciones de matrices y convoluciones en `float32`: `Precision.DEFAULT`, `Precision.HIGH` y `Precision.HIGHEST`. |
| `torch.tpu.precision(level)` | Administrador de contexto que establece el nivel de `Precision` para las operaciones de multiplicación de matrices y convolución dentro del bloque. |
| `torch.tpu.allow_excess_precision` | Obtiene o establece de forma global la marca `--xla_allow_excess_precision` de XLA para mantener en `float32` los valores intermedios fusionados. |
<!-- mdformat on -->

```python
a = torch.randn(64, 64, device="tpu", dtype=torch.float32)
b = torch.randn(64, 64, device="tpu", dtype=torch.float32)
with torch.tpu.precision(torch.tpu.Precision.HIGH):
  c = torch.matmul(a, b)
print(f"Computed HIGH-precision matmul with output shape: {tuple(c.shape)}")
```

### **Entrenamiento distribuido y topología** { #distributed-training-and-topology }

TorchTPU registra `"tpu_dist"` (también con el alias `"tpu"`) como el backend
predeterminado de `torch.distributed` para los dispositivos `"tpu"`. Consulta
[Primeros pasos con TorchTPU en GKE](./get_started_on_gke.md) y
[Prácticas recomendadas para FSDP](./case_studies_fsdp.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.topology_aware_mesh(mesh_shape, ...)` | Calcula un tensor de distribución de rangos que tiene en cuenta la topología, con la forma `mesh_shape`, para `DeviceMesh("tpu", mesh=...)`. |
| `torch.tpu.distributed.set_tpu_launch_env(...)` | Completa las variables de entorno de la topología de TPU y del lanzamiento en varios hosts antes de generar los workers. |
| `torch.tpu.distributed.spmd_safe(func)` | Decorador que declara que todos los rangos ejecutan operaciones de TPU idénticas en `func`, lo que habilita la fusión de operaciones colectivas en modo eager. |
| `torch.tpu.window_num_stripes` | Obtiene o establece la cantidad máxima de franjas (stripes) de DMA simultáneas para las transferencias punto a punto. |
| `torch.tpu.window_stripe_chunk_mb` | Obtiene o establece el tamaño mínimo de carga útil (en MB) por franja de DMA para las transferencias punto a punto. |
<!-- mdformat on -->

### **Operadores personalizados de TPU (`torch.tpu.ops`) y Pallas (`torch.tpu.pallas`)** { #tpu-custom-operators-torchtpuops-and-pallas-torchtpupallas }

TorchTPU expone operadores especializados para el hardware en `torch.tpu.ops` y
un wrapper de kernels personalizados de JAX/Pallas en `torch.tpu.pallas`.
Consulta [Agrega kernels personalizados con Pallas](./custom_kernels.md).

#### **Operadores generales, de atención, de MoE y colectivos** { #general-attention-moe-and-collective-operators }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.ops.max_pool{1,2,3}d(...)` | Aplica max pooling 1D, 2D o 3D en TPU sin materializar un tensor de índices. |
| `torch.tpu.ops.max_pool{1,2,3}d_backward(...)` | Calcula el pase hacia atrás de `max_pool{1,2,3}d` sin un tensor de índices. |
| `torch.tpu.ops.optimization_barrier(inputs)` | Impide que XLA mueva, fusione o elimine duplicados de cómputos a través de la barrera. |
| `torch.tpu.ops.splash_sdpa(q, k, v, ...)` | Atención de producto escalar escalado respaldada por el kernel Splash Attention de Pallas para TPU. |
| `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes, ...)` | Multiplicación de matrices agrupada para cargas de trabajo de Mixture-of-Experts (MoE). |
| `torch.tpu.ops.ragged_dot_weight_grad(lhs, grad_output, group_sizes)` | Calcula el gradiente de los pesos de `ragged_dot`. |
| `torch.tpu.ops.align_jagged_to_multiple(nt, ...)` | Rellena un búfer `NestedTensor` `torch.jagged` hasta un múltiplo especificado para la alineación en TPU. |
| `torch.tpu.ops.strip_jagged_padding(nt, ...)` | Quita el relleno final que agrega `align_jagged_to_multiple`. |
| `torch.tpu.ops.all_gather_dim(input, dim, group=None)` | Realiza un all-gather de `input` en `group` y concatena a lo largo de la dimensión `dim`. |
| `torch.tpu.ops.reduce_scatter_dim(input, dim, group=None, ...)` | Reduce `input` en `group` y distribuye fragmentos a lo largo de la dimensión `dim`. |
| `torch.tpu.ops.ragged_all_to_all(...)` | Intercambia porciones de tensores de longitud variable entre los rangos de un grupo de procesos. |
| `torch.tpu.ops.send(tensors, dst, tag)` / `torch.tpu.ops.recv(tensors, src, tag)` | Envío y recepción punto a punto de tensores en el grupo de procesos de TPU predeterminado. |
| `torch.tpu.pallas.jax_op(name, fn=None, ...)` | Registra una función de JAX/Pallas como un operador personalizado trazable de PyTorch. |
<!-- mdformat on -->

#### **Operadores de embeddings de SparseCore** { #sparsecore-embedding-operators }

Estos operadores requieren una TPU con SparseCores. La propiedad
`sparse_cores_per_device` de `torch.tpu.get_device_properties()` es la cantidad
de SparseCores por dispositivo (0 en las TPUs sin SparseCores), y
`sparse_core_simd_width` es su ancho SIMD.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.ops.sparse_dense_matmul(...)` | Realiza la búsqueda y la agregación de embeddings CSR en SparseCore de TPU. |
| `torch.tpu.ops.sparse_dense_matmul_grad_with_sgd(...)` / `_with_adagrad(...)` / `_with_adam(...)` | Calcula los gradientes del pase hacia atrás de `sparse_dense_matmul` y aplica una actualización fusionada e in situ de la tabla con SGD, Adagrad o Adam en SparseCore. |
| `torch.tpu.ops.sparse_dense_matmul_activation_unstack(...)` | Divide las activaciones apiladas de embeddings de SparseCore en tensores por atributo. |
| `torch.tpu.ops.sparse_dense_matmul_gradient_stack(...)` | Apila los gradientes de activación por atributo en un solo tensor para SparseCore. |
| `torch.tpu.ops.sparse_gather(...)` | Recopila filas de un tensor 2D con índices CSR en SparseCore de TPU. |
| `torch.tpu.ops.sparse_gather_backward(...)` | Realiza scatter-add de los gradientes de `sparse_gather` en SparseCore de TPU. |
| `torch.tpu.ops.sparse_iota(...)` | Genera índices de posición dentro de cada fila para los elementos distintos de cero de una matriz CSR en SparseCore. |
| `torch.tpu.ops.sparse_map_row(...)` | Aplica un cómputo personalizado de MLIR fila por fila sobre matrices CSR dispersas en SparseCore. |
<!-- mdformat on -->

### **Paso de entrenamiento fusionado y optimizadores (`torch.tpu.optimizers`)** { #fused-training-step-and-optimizers-torchtpuoptimizers }

`torch.tpu.optimizers` traza el pase hacia adelante, el cálculo de la pérdida,
el pase hacia atrás y el paso funcional del optimizador en un solo grafo de TPU
compilado.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.optimizers.SingleTraceTrainer(model, optimizer=None, ...)` | Administra los parámetros funcionales (`params`) y el estado del optimizador (`param_group`), y compila el paso de entrenamiento completo mediante `make_compiled_train_step(...)`. |
| `torch.tpu.optimizers.Optimizer(lr=..., weight_decay=...)` | Clase base para optimizadores funcionales sin estado (expone `defaults`, `init_param_group` y `step`). |
| `torch.tpu.optimizers.AdamW(...)` | Optimizador funcional AdamW trazable (`AdamW.ParamGroup`). |
| `torch.tpu.optimizers.Muon(...)` | Optimizador funcional Muon trazable (`Muon.ParamGroup`). |
| `torch.tpu.optimizers.Sgd(...)` | Optimizador funcional SGD trazable (`Sgd.ParamGroup`). |
<!-- mdformat on -->

### **Generación de perfiles (`torch.tpu.profiler`)** { #profiling-torchtpuprofiler }

Configura el trazado del hardware de TPU y de Python para XProf y
TensorBoard. Consulta
[XProf para TorchTPU](./performance_profiling_with_xprof.md).

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| API | Descripción |
| :--- | :--- |
| `torch.tpu.profiler.TpuProfilerConfig(...)` | Configuración experimental que se pasa a `torch.profiler.profile(..., experimental_config=...)` para configurar el trazado del host, del dispositivo de TPU y de Python para XProf. |
<!-- mdformat on -->

## **Variables de entorno** { #environment-variables }

El comportamiento de TorchTPU se controla mediante varias variables de entorno
que configuran el compilador XLA, el controlador de TPU (`libtpu`) y el propio
subsistema de depuración de PyTorch.

> **CRÍTICO: Inicialización global del proceso** Las variables de entorno de
> TorchTPU quedan **bloqueadas** en el momento en que se inicializa el backend
> (por lo general, durante `import torch` o `torch.device("tpu")`).

### **Visibilidad de dispositivos, entorno de ejecución y lanzamiento distribuido** { #device-visibility-runtime-and-distributed-launch }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable de entorno | Descripción |
| :--- | :--- |
| `TPU_VISIBLE_DEVICES` | Índices de chips de TPU separados por comas que son visibles para el proceso actual (análogo a `CUDA_VISIBLE_DEVICES`). |
| `RANK` | Rango global del proceso actual en un trabajo distribuido. |
| `LOCAL_RANK` | Rango local del proceso actual dentro del host. |
| `WORLD_SIZE` | Cantidad total de procesos en el trabajo distribuido. |
| `NNODES` | Cantidad total de nodos host en un trabajo de varios hosts. |
| `NODE_RANK` | Rango del nodo host actual en un trabajo de varios hosts. |
| `MASTER_ADDR` | Dirección del nodo coordinador del rango 0 para `torch.distributed`. |
| `MASTER_PORT` | Puerto en `MASTER_ADDR` que se usa para el punto de encuentro (rendezvous) de `torch.distributed`. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | Configúrala como `"0"` para desactivar el registro automático del backend `torch_tpu`. |
| `TORCH_TPU_ALLOW_LEGACY_DEVICE_APIS` | Cuando está habilitada, permite llamar a las APIs internas de dispositivo de `torch.tpu` (las que tienen un equivalente en `torch.accelerator`) desde fuera del núcleo de PyTorch con una advertencia única, en lugar de generar un `RuntimeError`. Úsala solo como una solución temporal; se recomienda migrar a `torch.accelerator.*`, y es probable que esta variable se quite en una versión futura. |
| `LIBTPU_INIT_ARGS` | Marcas de inicialización separadas por espacios que se pasan a `libtpu` y al entorno de ejecución de PJRT. |
| `TPU_PREMAPPED_BUFFER_SIZE` | Tamaño en bytes del búfer de memoria premapeado de host a dispositivo para el cliente de PJRT. |
| `TMPDIR` | Directorio temporal que usan el entorno de ejecución y el compilador. |
<!-- mdformat on -->

### **Topología de hardware y coordinación entre varios hosts** { #hardware-topology-and-multi-host-coordination }

Estas variables especifican la topología de hardware de TPU y la red de los
procesos. En la mayoría de los casos, las configura automáticamente
`torch.tpu.distributed.set_tpu_launch_env()`, el lanzador de TorchTPU o el
entorno de Google Cloud.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable de entorno | Descripción |
| :--- | :--- |
| `TORCH_TPU_TOPOLOGY` | Topología física de la porción de TPU (por ejemplo, `2x2x1`, `2x2x4`). |
| `TORCH_TPU_SLICEBUILDER_ADDRESSES` | Direcciones `host:port` de los workers, separadas por comas, para el servicio constructor de porciones de TPU. |
| `TORCH_TPU_DEVICES_PER_HOST` | Cantidad de chips de TPU por host al derivar las direcciones del constructor de porciones. |
| `TORCH_TPU_BASE_PORT` | Número de puerto inicial al derivar las direcciones del constructor de porciones. |
| `TORCH_TPU_SPMD_HANDSHAKE_STAGE` | Etapa en la que los rangos verifican que los grafos colectivos coincidan entre los rangos. |
| `TORCHCOMM_RANK` | Rango del proceso para `torchcomm` (tiene prioridad sobre `RANK`). |
| `TORCHCOMM_SIZE` | Tamaño del mundo para `torchcomm` (tiene prioridad sobre `WORLD_SIZE`). |
<!-- mdformat on -->

### **Modo de ejecución, compilación y almacenamiento en caché** { #execution-mode-compilation-and-caching }

Estas variables controlan cómo se despachan las operaciones y los grafos de
PyTorch, cómo se reducen (lowering), cómo los optimiza XLA y cómo se almacenan
en caché entre ejecuciones y rangos.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable de entorno | Descripción |
| :--- | :--- |
| `TORCH_TPU_DEFER_AND_FUSE` | Cuando está habilitada (`"1"`/`"true"`), establece el modo eager predeterminado en `EagerMode.DEFER_AND_FUSE`. |
| `TORCH_TPU_LAUNCH_BLOCKING` | Cuando está habilitada (`"1"`/`"true"`), establece el modo eager predeterminado en `EagerMode.DEFER_NEVER_AND_LAUNCH_BLOCKING`. |
| `TORCH_TPU_TIER2_COMPILATION_CACHE` | Nombre del directorio dentro de `/dev/shm/torch_tpu_cache/` para la caché de compilación local del host (o `"disabled"`). |
| `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` | Ruta (por ejemplo, `gs://...` o una ruta local) a la caché de compilación persistente de nivel 3 que se comparte entre hosts y ejecuciones. |
| `XLA_FLAGS` | Marcas del compilador XLA separadas por espacios (por ejemplo, `--xla_dump_hlo_as_text --xla_dump_to=...`). |
<!-- mdformat on -->

### **Depuración, seguridad, generación de perfiles y trazado** { #debugging-safety-profiling-and-tracing }

Estas variables controlan el contexto de depuración, las verificaciones de
seguridad en el entorno de ejecución y la salida del generador de perfiles de
TorchTPU.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Variable de entorno | Descripción |
| :--- | :--- |
| `TORCH_SHOW_CPP_STACKTRACES` | Cuando se configura como `"1"`, enriquece las excepciones `RuntimeError` de Python con seguimientos de pila (backtraces) de C++. |
| `TORCH_TPU_ENABLE_DEBUG_CHECKS` | Cuando se configura como `"1"`, habilita las verificaciones de depuración y de límites en el entorno de ejecución de TPU. |
| `TORCH_TPU_PROFILER_OUTPUT_DIR` | Directorio de salida alternativo para los archivos de traza de XProf cuando `TpuProfilerConfig.run_dir` no está configurado. |
| `TORCH_TPU_XPROF_SESSION_ID` | ID de sesión de generación de perfiles de XProf compartido entre los workers de varios hosts. |
| `TORCH_TRACE` | Directorio de salida para los registros estructurados de trazas del compilador y del despachador (`tlparse`). |
<!-- mdformat on -->

### **Fragmentos de configuración de variables de entorno** { #environment-variable-configuration-snippets }

#### **1. Seguimientos de pila de C++ (`TORCH_SHOW_CPP_STACKTRACES`)** { #1-c-stacktraces-torch_show_cpp_stacktraces }

Si configuras `TORCH_SHOW_CPP_STACKTRACES=1` antes de importar `torch`, es mucho
más fácil diagnosticar los errores que ocurren en lo profundo del backend de
C++ o de XLA. En lugar de un simple `RuntimeError` de Python, recibes un `C++
CapturedTraceback` detallado que muestra los marcos de ejecución exactos.

```python
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
import torch

device = torch.device("tpu")
a = torch.randn(10, device=device)
b = torch.randn(20, device=device)
# Raises RuntimeError enriched with C++ Error Trace and Captured Traceback:
c = a + b
```

#### **2. Verificaciones de depuración internas y validación de límites (`TORCH_TPU_ENABLE_DEBUG_CHECKS`)** { #2-internal-debug-checks-boundary-validation-torch_tpu_enable_debug_checks }

De forma predeterminada, la TPU omite las verificaciones de límites para ganar
velocidad. Si habilitas `TORCH_TPU_ENABLE_DEBUG_CHECKS=1`, se detectan errores
sutiles relacionados con índices fuera de los límites (como en `torch.take`),
aunque la ejecución se vuelve más lenta debido a la sincronización adicional
entre la CPU y la TPU. Las operaciones que, de otro modo, fallarían de forma
silenciosa o dañarían los datos generan una excepción explícita, como
`take(): expected indices to be
in range`:

```python
import os
os.environ["TORCH_TPU_ENABLE_DEBUG_CHECKS"] = "1"
import torch

device = torch.device("tpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
indices = torch.tensor([5], device=device)  # Out of bounds
# Raises an explicit out-of-bounds exception when materialized:
y = torch.take(x, indices).cpu()
```

#### **3. Volcado de IR / HLO de XLA (`XLA_FLAGS`)** { #3-dumping-xla-ir-hlo-xla_flags }

Si configuras `XLA_FLAGS`, puedes forzar al compilador a volcar el texto del
High-Level Optimizer (HLO) en un directorio. El directorio de volcado (por
ejemplo, `/tmp/hlo_dump`) contendrá lo siguiente:

*   **`*.after_optimizations.txt`**: El grafo de HLO final, que muestra las
    operaciones a nivel de hardware.
*   **`*.flagfile`**: Una lista completa de las marcas activas del compilador
    XLA (verifica `xla_optimization_level`).
*   **`*tpu_comp_env.txt`**: La configuración interna del entorno de TPU que se
    usa durante la compilación.

```python
import os

dump_dir = "/tmp/hlo_dump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = f"--xla_dump_hlo_as_text --xla_dump_to={dump_dir}"

import torch
device = torch.device("tpu")
x = torch.randn(10, device=device)
y = (x + 1).to("cpu")
```

#### **4. Almacenamiento en caché de compilación en varios niveles (`TORCH_TPU_TIER2_COMPILATION_CACHE` y `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** { #4-multi-tiered-compilation-caching-torch_tpu_tier2_compilation_cache-torch_tpu_tier3_compilation_cache_root }

TorchTPU usa un almacenamiento en caché de varios niveles para evitar
compilaciones redundantes entre ejecuciones y rangos:

*   **El nivel 2 (`TORCH_TPU_TIER2_COMPILATION_CACHE`)** se almacena de forma
    local en la memoria compartida del host. Espera un **nombre** de directorio
    (por ejemplo, `"tpu_tier2_cache"`), que el backend coloca automáticamente
    dentro de `/dev/shm/torch_tpu_cache/<name>`.
*   **El nivel 3 (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`)** es una caché de
    compilación persistente que se comparte entre hosts y sesiones (por lo
    general, un URI de bucket `gs://` o una ruta de un sistema de archivos
    compartido).

Los artefactos binarios compilados (ejecutables `*.bin`) se guardan en los
directorios de caché de forma asíncrona y se reutilizan en ejecuciones
posteriores, lo que reduce drásticamente la latencia de la compilación JIT:

```python
import os

tier2_name = "tpu_tier2_cache"
tier2_path = os.path.join("/dev/shm/torch_tpu_cache", tier2_name)
tier3_path = "/tmp/tpu_tier3_cache"  # Or "gs://<bucket_name>/<path_to_cache>"
os.makedirs(tier2_path, exist_ok=True)
os.makedirs(tier3_path, exist_ok=True)

# TORCH_TPU_TIER2_COMPILATION_CACHE expects just the directory NAME inside /dev/shm/torch_tpu_cache/
os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = tier2_name
os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = tier3_path

import torch
device = torch.device("tpu")
a = torch.randn(1024, 1024, device=device)
_ = (a @ a).to("cpu")
```
