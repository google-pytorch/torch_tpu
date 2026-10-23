# Generación de perfiles de rendimiento con XProf { #performance-profiling-with-xprof }

Comprender en qué se invierte el tiempo es fundamental para entrenar modelos
grandes en TPU. En esta guía, se explica cómo capturar trazas de ejecución con el
generador de perfiles de TPU y analizarlas en TensorBoard.

> **Warning:** **Requisito previo:** TensorBoard requiere `setuptools<81`. La
> versión más reciente quita `pkg_resources`, del que depende TensorBoard.
> Instálalo primero:
>
> ```shell
> pip install "setuptools<81"
> pip install xprof tensorboard
> ```

> **Note:** **Versión de PyTorch:** La integración nativa de `torch.profiler`
> con TPU requiere PyTorch 2.12 o una versión posterior.

## 1. El generador de perfiles de TPU { #1-the-tpu-profiler }

El `torch.profiler` estándar (integrado con `torch_tpu`) proporciona una
interfaz de **administrador de contexto** para trazar las actividades de la CPU
y de la TPU. Encapsulas el código del que quieres generar un perfil en un bloque
`torch.profiler.profile()`, y las trazas recopiladas se guardan en el disco para
verlas en TensorBoard.

**Componentes clave:**

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Componente                                 | Función                         |
| ------------------------------------------ | ------------------------------- |
| `torch.profiler.profile()`                 | Administrador de contexto estándar que inicia y detiene <br> la recopilación de trazas |
| `torch.profiler.ProfilerActivity.CPU`      | Traza las operaciones del lado de la CPU (despacho, carga <br> de datos) |
| `torch.profiler.ProfilerActivity.TPU`      | Traza la ejecución en el hardware de TPU (cálculo, <br> memoria, interconexión) |
| `torch.profiler.tensorboard_trace_handler` | Devolución de llamada que guarda las trazas en un directorio para <br> TensorBoard       |
| `TpuProfilerConfig`                        | Asistente para la configuración personalizada de la generación de perfiles de TPU |
<!-- mdformat on -->

```python
import torch
from torch.tpu.profiler import TpuProfilerConfig

device = torch.device("tpu")
```

## 2. Configuración personalizada de la generación de perfiles (`TpuProfilerConfig`) { #2-custom-profiling-configuration-tpuprofilerconfig }

Las opciones que se pasan mediante `TpuProfilerConfig` permiten un control
detallado de los niveles de los trazadores del host y del dispositivo, así como
de las estructuras del directorio de salida.

### Opciones de configuración admitidas { #sanctioned-configuration-options }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Clave de la opción    | Tipo           | Predeterminado | Descripción        |
| :-------------------- | :------------- | :------ | :------------------------ |
| `host_tracer_level`   | `int`          | `2`     | Controla el nivel de detalle del trazador del host de CPU <br> (0=desactivado, 1=crítico, 2=todo). |
| `device_tracer_level` | `int`          | `1`     | Controla el nivel de detalle del trazador del dispositivo <br> de hardware de TPU (0=desactivado, 1=activado). |
| `python_tracer_level` | `int`          | `0`     | Controla el nivel de trazado de funciones <br> de Python (0=desactivado, 1=activado). |
| `run_dir`             | `pathlib.Path` | `None`  | Directorio de salida donde se guardan los archivos de traza <br> del hardware de TPU (`.xplane.pb`). |
| `worker_rank`         | `str`          | `None`  | Rango del worker que se agrega como sufijo al <br> nombre del archivo XPlane (`<hostname>_<rank>.xplane.pb`). |
<!-- mdformat on -->

```python
import pathlib
import torch
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
    python_tracer_level=0,
    run_dir=pathlib.Path("./profiler_output"),
    worker_rank="0",
)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
) as prof:
    model(inputs)
```

## 3. Captura un perfil { #3-capturing-a-profile }

La siguiente celda ejecuta un bucle de entrenamiento pequeño dentro del contexto
del generador de perfiles. Los puntos clave son los siguientes:

1.  Encapsula tu bucle de entrenamiento dentro de `torch.profiler.profile()`
2.  Especifica las actividades: `CPU` para las operaciones del lado del host y
    `TPU` para la ejecución en el hardware
3.  Establece `on_trace_ready` para guardar las trazas en un directorio con
    `torch.profiler.tensorboard_trace_handler`
4.  (Opcional) Pasa `experimental_config` con `TpuProfilerConfig` para controlar
    los niveles de los trazadores

> **Important:** El generador de perfiles agrega sobrecarga. Genera perfiles
> solo de un subconjunto representativo de pasos (por ejemplo, de 100 a 1,000),
> no de toda la ejecución del entrenamiento.

```python
# Build a small model and optimizer
model = (
    torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )
    .to(device)
    .to(torch.bfloat16)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(32, 128, device=device, dtype=torch.bfloat16)

# Where to save the profiler traces
log_dir = "/tmp/profiler_output"

# Configure TPU-specific tracer levels
config = TpuProfilerConfig(
    host_tracer_level=2,
    device_tracer_level=1,
)

# Profile 100 training steps with native PyTorch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.TPU,
    ],
    experimental_config=config,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
) as prof:
  for step in range(100):
    optimizer.zero_grad()
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    torch.accelerator.synchronize()
    prof.step()

print(f"✅ Profile captured: {100} steps saved to {log_dir}")
```

## 4. Visualiza las trazas en TensorBoard { #4-viewing-traces-in-tensorboard }

Ejecuta los siguientes comandos:

**Paso 1:** Instala el complemento del generador de perfiles (es necesario para
que aparezca el panel):

```shell
pip install tensorboard-plugin-profile "setuptools<81"
```

**Paso 2:** Inicia TensorBoard:

```shell
tensorboard --logdir=/tmp/profiler_output --port=6006
```

XProf también se puede ejecutar como un servidor independiente.

```shell
xprof --logdir=/tmp/profiler_output --port=6006
```

**Paso 3:** Abre `http://localhost:6006` en tu navegador y selecciona la
pestaña **"Profile"**.

> **Note:** Debes ejecutar TensorBoard desde la terminal integrada de VS Code
> para que la redirección de puertos funcione automáticamente.

**En GKE:** Escribe el directorio de trazas en un bucket de Cloud Storage
montado con el
[controlador de CSI de Cloud Storage FUSE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
para que las trazas sobrevivan al pod y, luego, ábrelas con `xprof
--logdir=gs://<bucket>/profiler_output`.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Vista              | Qué muestra                                            |
| ------------------ | ------------------------------------------------------ |
| **Overview**       | Resumen general: tiempo por paso, uso del dispositivo y operaciones principales |
| **Trace Viewer**   | Cronograma de cada operación en la CPU y la TPU: busca brechas (hardware inactivo) |
| **Op Profile**     | Desglose del tiempo por operación: encuentra las operaciones más costosas |
| **Memory Profile** | Uso de HBM a lo largo del tiempo: detecta picos de memoria |
<!-- mdformat on -->

## 5. Qué buscar { #5-what-to-look-for }

### Señales de alerta en las trazas { #red-flags-in-traces }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Patrón                  | Significado             | Solución                 |
| ----------------------- | ----------------------- | ------------------------ |
| Brechas grandes entre operaciones de TPU | Cuello de botella del lado del host <br> (carga de datos, sobrecarga de Python) | Precarga los datos y reduce la lógica de Python en el bucle |
| Ráfagas cortas y frecuentes de TPU   | Demasiadas interrupciones de grafo → recompilación frecuente | Reduce las llamadas a `.item()` / `.cpu()` y usa `torch.compile` |
| Bajo uso de las MXU | Dimensiones de los tensores no alineadas con el tamaño de tile (128) | Rellena las dimensiones hasta múltiplos de 128 |
<!-- mdformat on -->

### El perfil ideal { #the-ideal-profile }

Una carga de trabajo de TPU bien optimizada muestra lo siguiente:

-   **Bloques de cálculo de TPU largos y continuos**: el compilador fusionó
    muchas operaciones en una sola ejecución
-   **Brechas mínimas entre la CPU y la TPU**: la carga de datos se superpone
    con el cálculo
-   **Uso alto de las MXU (>60%)**: las formas de los tensores están alineadas
    con el hardware

## 6. Verifica que se hayan capturado las trazas { #6-verifying-the-traces-were-captured }

La siguiente celda verifica que el directorio de salida del generador de
perfiles contenga archivos de traza.

```python
import os

trace_files = os.listdir(log_dir)
print(f"Trace directory: {log_dir}")
print(f"Files found: {len(trace_files)}")
for f in sorted(trace_files)[:10]:
  size_kb = os.path.getsize(os.path.join(log_dir, f)) / 1024
  print(f"  {f} ({size_kb:.1f} KB)")

if trace_files:
  print(f"\n✅ Traces ready. View with:")
  print(f"   tensorboard --logdir={log_dir}")
else:
  print("\n❌ No trace files found. Check profiler configuration.")
```

## 7. Generación de perfiles de kernels (contadores de rendimiento detallados) { #7-kernel-profiling-fine-grained-performance-counters }

La generación de perfiles de kernels permite recopilar contadores de
rendimiento de hardware detallados (por ejemplo, para kernels de Pallas) a fin
de diagnosticar cuellos de botella de rendimiento que, de otro modo, no se ven
en las trazas de eventos estándar.

> **Note:** El muestreo detallado de contadores de rendimiento es una función
> avanzada de hardware disponible para **Ironwood (TPU v7/TPU7x)** y las
> generaciones de TPU posteriores.

### Habilita los contadores de rendimiento { #enabling-performance-counters }

Para habilitar los contadores de rendimiento, pasa la configuración de los
contadores mediante `experimental_options` en `TpuProfilerConfig`:

```python
from torch.tpu.profiler import TpuProfilerConfig

config = TpuProfilerConfig(
    experimental_options={
        "tpu_enable_periodic_counter_sampling": True,
        "tpu_tc_perf_counter_sampling_options": (
            "interval_us:1 scaling:0 counter_size_bits:1 indices:10"
            " indices:11 indices:56 indices:57 indices:58"
        ),
    }
)
```

Para ver la lista detallada de parámetros de configuración (como los intervalos
de muestreo, el escalamiento y los modos de activación) y las instrucciones para
determinar los índices de los contadores, consulta la
[Guía oficial de generación de perfiles de kernels de OpenXLA](https://openxla.org/xprof/kernel-profiling).

## 8. Referencia rápida { #8-quick-reference }

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
Paso        | Comando o código
----------- | --------------
**Importar**  | `from torch.tpu.profiler import TpuProfilerConfig`
**Generar perfil** | `with torch.profiler.profile(activities=[...], on_trace_ready=...):`
**Guardar**    | `torch.profiler.tensorboard_trace_handler("/tmp/profiler_output")`
**Ver**    | `tensorboard --logdir=/tmp/profiler_output`
<!-- mdformat on -->

> **Note:** Genera perfiles de solo 100 a 1,000 pasos para que los archivos de
> traza sean manejables. El generador de perfiles agrega una sobrecarga de
> ~5–10% por paso.
