# Preguntas frecuentes sobre TorchTPU { #torchtpu-faq }

## ¿Cuál es la diferencia entre PyTorch/XLA y TorchTPU? { #what-is-the-difference-between-pytorchxla-and-torchtpu }

Históricamente, PyTorch/XLA funcionó como un framework puente que dependía en
gran medida de componentes del entorno de ejecución de JAX y de APIs
especializadas. TorchTPU ofrece un backend nativo de PyTorch para
[Google Cloud TPUs](https://cloud.google.com/tpu) sin dependencias del entorno
de ejecución de JAX, y las cargas de trabajo de PyTorch existentes funcionan
de inmediato con cambios mínimos en el código (por ejemplo, `.to("cuda")` →
`.to("tpu")`)

## ¿Cuál es la diferencia entre TorchAX y TorchTPU? { #what-is-the-difference-between-torchax-and-torchtpu }

TorchAX (también conocido como torch_xla2) es una implementación que intercepta
las operaciones de PyTorch a nivel de Python y las reduce (lowering) a XLA (a
menudo a través de JAX). Aunque TorchAX se usó mucho para comparativas de
rendimiento de inferencia y para prototipos, TorchTPU es el nuevo estándar
oficial para una experiencia nativa de PyTorch en TPUs que funciona como
reemplazo directo.

## ¿Qué pasará con PyTorch/XLA? { #what-is-happening-to-pytorchxla }

PyTorch/XLA está obsoleto y lo reemplaza
[TorchTPU](https://github.com/google-pytorch/torch_tpu), que ya es público.

## ¿TorchTPU reemplaza a JAX? { #is-torchtpu-replacing-jax }

No. TorchTPU está diseñado para complementar el ecosistema de TPU con un punto
de entrada nativo de PyTorch a Cloud TPU. Permite que los desarrolladores con
bases de código de PyTorch existentes las ejecuten directamente en TPUs sin
reescrituras o con reescrituras mínimas. Así, no necesitan reescribir modelos
en JAX ni mantener bases de código separadas para infraestructura de GPU y TPU.

*   **¿Cuándo conviene considerar TorchTPU?** Si tienes una base de código de
    PyTorch existente, si quieres portabilidad entre varios tipos de hardware
    (GPU/TPU) sin refactorizar, si usas frameworks nativos de PyTorch (por
    ejemplo, TorchTitan, vLLM, SGLang, Hugging Face) o si quieres migrar
    rápidamente de GPU a TPU.

## ¿Cuál es la mejor forma de contactar a los desarrolladores de TorchTPU? { #what-is-the-best-way-for-me-to-connect-with-torchtpu-developers }

Puedes enviar comentarios por correo electrónico a
[torchtpu-feedback@google.com](mailto:torchtpu-feedback@google.com), y
recibimos con gusto contribuciones directamente en el repositorio:
[https://github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu).
También puedes usar la etiqueta "TPU" en
[discuss.pytorch.org](https://discuss.pytorch.org/).

## ¿Habrá foros de participación de la comunidad? { #will-there-be-any-community-engagement-forums }

Sí, la participación de la comunidad es muy importante para nosotros, y
compartiremos los detalles de esos foros. Además, planeamos organizar horas de
consulta (office hours) mensuales.

## ¿Qué soporte ofrece TorchTPU para el desarrollo de kernels? { #what-support-does-torchtpu-provide-for-kernel-development }

Consulta [Compatibilidad con kernels personalizados de TorchTPU](./custom_kernels.md).

## ¿Qué generaciones de TPU son compatibles actualmente? ¿Cuál es el plan para el hardware futuro? { #which-tpu-generations-are-actively-supported-whats-the-plan-for-future-hardware-support }

TorchTPU es compatible actualmente con v5e, v5p, Trillium (TPU v6e) e Ironwood
(TPU v7x). La validación y la optimización para TPU v8 y los chips futuros
están en curso como parte de nuestra hoja de ruta.

## ¿Dónde puedo encontrar las funciones compatibles y la hoja de ruta de TorchTPU? { #where-can-we-find-torchtpus-supported-features-future-roadmap }

Consulta [Funciones compatibles](./feature_set.md) para ver el conjunto de
funciones actual y el conjunto de funciones previsto.

## ¿Hay algún agente que facilite mi incorporación? { #is-there-an-agent-to-make-my-onboarding-easier }

Prueba el conjunto de agentes para TPU (experimental):
[https://github.com/google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite)

TPU Agent Suite es un plugin de código abierto que ofrece subagentes
especializados para automatizar la incorporación, el diagnóstico y el ajuste
del rendimiento de inferencia de modelos de PyTorch en Cloud TPU v6e y v7x.

## ¿Puedo usar TorchTPU con nuestra solución de serving personalizada? { #can-i-use-torchtpu-with-our-custom-serving-solution }

Sí. TorchTPU ofrece integración lista para usar con motores de serving
estándar de la industria, como vLLM y SGLang. Además, las soluciones de serving
personalizadas o internas pueden integrarse con TorchTPU mediante las APIs de
ejecución nativas de PyTorch y los backends de compilador de nivel más bajo.
Para una orquestación avanzada, los clientes también pueden implementar
configuraciones de serving personalizadas en GKE con llm-d.
