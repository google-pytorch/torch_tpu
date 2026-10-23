# Primeros pasos con TorchTPU en GKE { #get-started-with-torchtpu-on-gke }

En este instructivo, ejecutarás cargas de trabajo de PyTorch en un clúster de
TPU de Google Kubernetes Engine (GKE) en tres partes:

1.  **Ejecuta un ejemplo simple de un solo dispositivo**
    (`examples/simple_pytorch/simple_pytorch.py`) como un `Job` de Kubernetes.
2.  **Ejecuta un ejemplo distribuido de varios dispositivos**
    (`examples/tensor_parallel/manual/tp_worker.py`) en un solo host de 4 chips
    (`Job`) y en una porción (slice) de varios hosts y 64 chips (`JobSet`).
3.  **Ejecuta un ejemplo de PyTorch upstream** de
    [`pytorch/examples`](https://github.com/pytorch/examples) (como MNIST).

Suponemos que ya tienes un clúster de GKE Standard con un grupo de nodos de TPU.
Si necesitas crear uno, consulta
[Implementa cargas de trabajo de TPU en GKE Standard](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpus).

## Antes de comenzar { #before-you-begin }

### Requisitos previos { #prerequisites }

*   Un clúster de GKE con un grupo de nodos `tpu7x`. En este instructivo, se
    usa un grupo de nodos de 4 chips (`2x2x1`).
*   La [gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk).

### Configura las variables de entorno { #set-environment-variables }

Todos los pasos de este instructivo dependen de las siguientes variables;
configúralas según corresponda. `LOCATION` es la zona o región de tu clúster.

```bash
export PROJECT=<YOUR_PROJECT_ID>
export CLUSTER=<YOUR_CLUSTER_NAME>
export LOCATION=<YOUR_CLUSTER_ZONE_OR_REGION>
export NODEPOOL_NAME=<YOUR_4CHIP_NODEPOOL_NAME>
export DOCKER_IMAGE=us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### Autentícate y obtén las credenciales del clúster { #authenticate-and-get-cluster-credentials }

```bash
gcloud auth login
gcloud auth application-default login
gcloud components install kubectl gke-gcloud-auth-plugin
gcloud container clusters get-credentials ${CLUSTER} --location=${LOCATION} --project=${PROJECT}
```

Para obtener más detalles, consulta
[Instala kubectl y configura el acceso al clúster](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl).

## Envía cargas de trabajo con kubectl { #submit-workloads-with-kubectl }

Las cargas de trabajo se envían como manifiestos estándar de Kubernetes `Job` o
`JobSet`. Las secuencias de comandos de ejemplo de Python ya vienen
preinstaladas en la imagen de contenedor pública en `/workspace/examples/`,
mientras que los manifiestos `.yaml` de Kubernetes de referencia
(`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`,
`examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` y
`examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml`) están en el
directorio `examples/` del repositorio de TorchTPU. Ejecuta los comandos
`sed` y `kubectl
apply` que aparecen a continuación desde la raíz de una copia local del
repositorio `torch_tpu`.

## Ejecuta una carga de trabajo de un solo dispositivo (un TensorCore de un host de 4 chips) { #run-a-single-device-workload-one-tensorcore-of-a-4-chip-host }

El manifiesto `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`
reserva un host con 4 chips de TPU v7x. Cada chip v7x tiene dos TensorCores, y
TorchTPU expone cada TensorCore como un dispositivo, por lo que el host tiene 8
dispositivos de TPU y la topología de su porción es `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
    ...
    cloud.google.com/gke-tpu-topology: 2x2x1
```

El ejemplo ejecuta un solo proceso en un solo dispositivo de TPU (un
TensorCore) y deja inactivos los otros 7 dispositivos del host. Esto se expresa
como una topología de TorchTPU `1x1x1x1`, en la que el cuarto número es la
cantidad de TensorCores por chip que usa el trabajo:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "1,1,1,1"
```

El contenedor ejecuta `simple_pytorch.py`, que se incluye en la imagen en
`/workspace/examples/`:

```yaml
  command:
    - /bin/bash
    - -c
    - |
      set -ex
      export PYTHONUNBUFFERED=1
      export TPU_LAUNCH_BLOCKING=1
      export ABSL_LOGGING_MIN_LOG_LEVEL=0
      echo "=== Starting Single-Device 1-Chip, 1-Device TPU v7x Workload ==="
      cd /workspace/examples/simple_pytorch/
      python3 ./simple_pytorch.py
      echo "=== Single-Host single-device Workload Completed Successfully ==="
```

Este es el programa completo:

```python
import torch

device = torch.device("tpu")  # Change from cuda to tpu
print(f"TPU Initialization complete. Primary device: {device}")

# 1. Create a tensor on the CPU
cpu_tensor = torch.ones((5, 5))

# 2. Move it to the TPU
tpu_tensor = cpu_tensor.to(device)  # Using the device object
# OR
tpu_tensor_alt = cpu_tensor.to("tpu")  # Using the "tpu" string

# 3. Create a tensor directly on the TPU
direct_tpu_tensor = torch.randn((5, 5), device="tpu")

print(f"cpu_tensor is on: {cpu_tensor.device}")
print(f"tpu_tensor is on: {tpu_tensor.device}")
print(f"direct_tpu_tensor is on: {direct_tpu_tensor.device}")
```

Inserta la imagen y el grupo de nodos en el manifiesto y aplícalo:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
```

Verifica el estado del trabajo y del pod, y luego lee los registros:

```bash
kubectl get job,pods -n default
kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
```

El trabajo se ejecuta hasta completarse y muestra las cuatro líneas anteriores.

## Ejecuta cargas de trabajo distribuidas en GKE { #run-distributed-workloads-on-gke }

### Un solo host, varios chips (4 chips / 8 dispositivos en un host) { #single-host-multiple-chips-4-chips-8-devices-on-one-host }

El siguiente paso después del ejemplo de un solo dispositivo es usar todos los
dispositivos de TPU de un host. Envía un `Job` de Kubernetes de un solo host:
primero, actualiza el manifiesto con tu imagen de Docker y el nombre de tu
grupo de nodos, y luego inicia el trabajo con `kubectl`.

El manifiesto `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` se
aplica a un solo host de 4 TPU v7x (8 TensorCores), por lo que la topología de
la porción es `2x2x1`:

```yaml
    cloud.google.com/gke-tpu-slice-topology: 2x2x1
```

TorchTPU interpreta esto como una topología `2x2x1x2` porque inicia un proceso
por cada TensorCore de TPU:

```yaml
    - name: TORCH_TPU_TOPOLOGY
      value: "2,2,1,2"
```

Ejecuta el ejemplo manual de paralelismo de tensores desde la imagen de Docker:

```yaml
  command:
    - /bin/bash
    - -c
    - |
      set -ex
      export PYTHONUNBUFFERED=1
      export TPU_LAUNCH_BLOCKING=1
      export ABSL_LOGGING_MIN_LOG_LEVEL=0
      echo "=== Starting Single-Host 4-Chip (8-Device) TPU v7x Workload ==="
      cd /workspace/examples/tensor_parallel/manual/
      bash ./launch_singlehost.sh
      echo "=== Single-Host 4-Chip Workload Completed Successfully ==="
```

`examples/tensor_parallel/manual/launch_singlehost.sh` detecta la topología de
TPU, la exporta e inicia un proceso por cada dispositivo de TPU con
`torch.distributed.run` (`torchrun`), que ejecuta
`examples/tensor_parallel/manual/tp_worker.py`:

```bash
# Detect and export TPU configuration dynamically.
eval "$(python3 -m torch.tpu.distributed)"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES

python3 -m torch.distributed.run --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    --nproc_per_node=8 ./tp_worker.py
```

Las secuencias de comandos del worker y de inicio están en
`examples/tensor_parallel/manual/`.

Inserta la imagen y el grupo de nodos en el manifiesto y aplícalo:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml
```

Verifica el estado del trabajo y del pod:

```bash
kubectl get job,pods -n default
```

### Varios hosts (64 chips / 128 dispositivos en 16 hosts) { #multiple-hosts-64-chips-128-devices-across-16-hosts }

Para escalar más allá de un host, el mismo ejemplo se ejecuta como un `JobSet`
de Kubernetes: un pod por host, 16 hosts para una porción `4x4x4`, y
`examples/tensor_parallel/manual/launch_multihost.sh` deriva el rango (rank) de
cada host a partir de su posición en la porción. Actualiza el manifiesto con tu
imagen de Docker y el nombre de tu grupo de nodos de 64 chips, y aplícalo con
`kubectl`:

```bash
sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
kubectl apply -f examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml
```

Verifica el estado del `JobSet` y del pod:

```bash
kubectl get jobset,pods -n default
```

## Ejecuta un ejemplo de PyTorch upstream { #run-an-upstream-pytorch-example }

Como TorchTPU es un backend nativo de PyTorch (`device="tpu"`,
`torch.accelerator`, `backend="tpu_dist"` distribuido), puedes obtener y
ejecutar secuencias de comandos estándar del repositorio upstream
[`pytorch/examples`](https://github.com/pytorch/examples) directamente en GKE.

### Ejemplo: Ejecuta MNIST upstream en GKE { #example-running-upstream-mnist-on-gke }

La secuencia de comandos de entrenamiento upstream
[`pytorch/examples/mnist`](https://github.com/pytorch/examples/tree/main/mnist)
(`mnist/main.py`) usa la API `torch.accelerator` de PyTorch, que es
independiente del dispositivo:

```python
if use_accel:
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")
```

Cuando `torch_tpu` está instalado en el contenedor,
`torch.accelerator.current_accelerator()` selecciona automáticamente el
dispositivo `"tpu"`, por lo que `mnist/main.py` no requiere **ningún cambio en
el código**.

Para ejecutar `mnist/main.py` en un solo dispositivo de TPU en GKE a partir de
`examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml`, sigue estos
pasos:

1.  **Actualiza el bloque `command:` del contenedor** en
    `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` para instalar
    `git`, clonar
    [`https://github.com/pytorch/examples.git`](https://github.com/pytorch/examples),
    mantener `num_workers` de `DataLoader` en su valor predeterminado (`0`) para
    que no bifurque un subproceso worker que intente volver a adquirir el
    archivo de bloqueo del dispositivo de `libtpu`, y ejecutar `mnist/main.py`:

    ```yaml
      command:
        - /bin/bash
        - -c
        - |
          set -ex
          export PYTHONUNBUFFERED=1
          apt-get update && apt-get install -y git
          git clone --depth 1 https://github.com/pytorch/examples.git /tmp/pytorch_examples
          cd /tmp/pytorch_examples/mnist
          sed -i "/'num_workers': 1,/d; /'persistent_workers': True,/d" main.py
          python3 main.py --epochs=1 --dry-run
    ```

2.  **Borra cualquier ejecución anterior del trabajo** (porque las
    especificaciones de `Job` de Kubernetes son inmutables una vez creadas),
    reemplaza los marcadores de posición si es necesario, aplica el manifiesto
    actualizado y transmite los registros de entrenamiento de MNIST:

    ```bash
    kubectl delete -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml --ignore-not-found
    sed -i "s|<DOCKER_IMAGE>|${DOCKER_IMAGE}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    sed -i "s|<NODEPOOL_NAME>|${NODEPOOL_NAME}|g" examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl apply -f examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml
    kubectl logs -n default -l job-name=torchtpu-singledevice-1chip-job -f
    ```

### Elige una plantilla de manifiesto para otros ejemplos upstream { #choosing-a-manifest-template-for-other-upstream-examples }

Para ejecutar otros ejemplos upstream de un solo dispositivo o distribuidos,
elige la plantilla de manifiesto de referencia que coincida con tu topología de
destino:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Tipo de carga de trabajo | Ruta del manifiesto | Topología de TPU (`gke-tpu-topology`) | `TORCH_TPU_TOPOLOGY` | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| **Un solo dispositivo** (1 TensorCore en un host de 4 chips) | `examples/simple_pytorch/torchtpu_singledevice_slice_job.yaml` | `2x2x1` | `"1,1,1,1"` | Ejecuta un solo proceso de Python en 1 dispositivo de TPU. |
| **Distribuido en un solo host** (4 chips / 8 núcleos) | `examples/tensor_parallel/torchtpu_singlehost_slice_job.yaml` | `2x2x1` | `"2,2,1,2"` | Ejecuta 8 rangos distribuidos (`WORLD_SIZE=8`) en un solo host de 4 chips mediante `torchrun`. |
| **Distribuido en varios hosts** (64 chips / 128 núcleos) | `examples/tensor_parallel/torchtpu_multihost_slice_jobset.yaml` | `4x4x4` | `"4,4,4,2"` | `JobSet` de Kubernetes que coordina 16 pods worker en una porción de 64 chips. |
<!-- mdformat on -->

Para usar otras topologías de porción de TPU, actualiza
`cloud.google.com/gke-tpu-slice-topology`, `cloud.google.com/gke-tpu-topology`
y `TORCH_TPU_TOPOLOGY` en el manifiesto (consulta
[Arquitectura del sistema de VM de TPU](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
y
[Tipos de máquinas y topologías de TPU](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)).

### Adapta un ejemplo de PyTorch upstream para ejecutarlo en GKE { #porting-an-upstream-pytorch-example-to-run-on-gke }

Como TorchTPU es un backend nativo de PyTorch, ejecutar un ejemplo o una prueba
de PyTorch upstream en GKE requiere un proceso mínimo:

*   **Los ejemplos no distribuidos no necesitan cambios en el código**: Los
    ejemplos no distribuidos escritos para dispositivos estándar de PyTorch
    (mediante `torch.accelerator` o `torch.device`) funcionan con TorchTPU sin
    cambios en el código.
*   **Se espera que los ejemplos no obsoletos funcionen**: Los ejemplos de
    PyTorch upstream que tienen mantenimiento activo y no son obsoletos, así
    como las APIs modernas, funcionan de inmediato (consulta
    [Conjunto de funciones compatibles](./feature_set.md) para obtener más
    detalles).
*   **Los ejemplos distribuidos pueden necesitar modificaciones menores**:
    Algunos ejemplos distribuidos upstream están demasiado ajustados para otros
    dispositivos (por ejemplo, backends, marcas o supuestos de fragmentación
    específicos de GPU o NCCL) y es posible que requieran modificaciones
    menores para ejecutarse en TPU (consulta las secciones sobre ejecución
    distribuida anteriores y
    [Prácticas recomendadas para FSDP](./case_studies_fsdp.md)).
