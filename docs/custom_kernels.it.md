# Kernel personalizzati di TorchTPU { #torchtpu-custom-kernels }

Sebbene TorchTPU acceleri in modo nativo gli operatori standard del framework
PyTorch (come
[`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)),
potresti dover integrare kernel personalizzati per accelerare operazioni
specifiche del modello, fondere operazioni per ridurre il carico sulla banda
di memoria o implementare nuove architetture non ancora coperte dagli
operatori standard di PyTorch.

## Scegliere un approccio di integrazione { #choosing-an-integration-approach }

TorchTPU offre tre modi per usare kernel personalizzati su TPU, a seconda del
tuo caso d'uso:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Approccio | Stato | Ideale per |
| :---- | :---- | :---- |
| **Tokamax** | Pronto per la produzione | Accelerazione immediata dei blocchi comuni di LLM/MoE senza scrivere codice di kernel. |
| **Pallas (`torch.tpu.pallas.jax_op`)** | Pronto per la produzione | Scrivere kernel TPU su misura e con le massime prestazioni in JAX/Pallas e chiamarli da PyTorch. |
| **Helion** | Sperimentale | Scrivere kernel portabili tra acceleratori (ad esempio GPU, TPU) con un DSL nativo di PyTorch. |
<!-- mdformat on -->

## 1. Usare kernel già pronti di Tokamax { #1-using-pre-built-kernels-from-tokamax }

Prima di scrivere da zero un kernel personalizzato, verifica se l'operatore è
disponibile in [Tokamax](http://openxla.org/tokamax), una libreria di kernel
TPU ad alte prestazioni con interoperabilità nativa con PyTorch.

### Quando usarlo { #when-to-use }

Usa Tokamax quando il tuo modello richiede blocchi standard ad alte
prestazioni, tra cui:

*   **Mixture-of-Experts (MoE):** moltiplicazione di matrici raggruppata,
    gather e scatter dei token.
*   **Varianti di attention:** Paged attention, Gated DeltaNet, Kimi Delta
    Attention e Multi-head Latent Attention (MLA).
*   **Selezione:** calcolo ottimizzato di `TopK`.

### Come funziona { #how-it-works }

I kernel di Tokamax sono scritti in JAX/Pallas e supportati da autotuning
integrato (ad esempio la ricerca automatica della dimensione dei tile),
benchmarking XProf a basso overhead e test di regressione su tutte le
generazioni di TPU. Internamente, Tokamax usa `torch.tpu.pallas.jax_op` per
gestire in modo trasparente l'interoperabilità tra framework ed espone API
PyTorch native direttamente al tuo modello.

## 2. Scrivere kernel Pallas personalizzati { #2-writing-custom-pallas-kernels }

Se ti serve un'operazione su misura non coperta da PyTorch o Tokamax, puoi
scrivere un kernel TPU in
[Pallas](https://docs.jax.dev/en/latest/pallas/index.html) e registrarlo come
operatore PyTorch con `torch.tpu.pallas.jax_op`.

### Come integrare un kernel Pallas { #how-to-integrate-a-pallas-kernel }

1.  **Definisci il kernel Pallas** usando riferimenti di memoria (`Ref`) per
    input e output.
2.  **Racchiudi il kernel in una funzione JAX** con `pl.pallas_call` per
    specificare forme e dtype dell'output.
3.  **Registra la funzione JAX in TorchTPU** tramite `torch.tpu.pallas.jax_op`.

```python
import jax
from jax.experimental import pallas as pl
import torch

# 1. Define the Pallas kernel
def add_vectors_pallas_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y

# 2. Wrap the Pallas kernel into a JAX function
def add_vectors_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_pallas_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)

# 3. Wrap it for PyTorch using the functional interface
add_vectors_fn = torch.tpu.pallas.jax_op(
    "pallas::add_vectors",
    add_vectors_jax,
)

# 4. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors_fn(x, y)
```

## 3. Scrivere kernel portabili con Helion (sperimentale) { #3-writing-portable-kernels-with-helion-experimental }

Se vuoi che i tuoi kernel siano portabili tra acceleratori (ad esempio TPU,
GPU), puoi usare [Helion](https://helionlang.com/), un DSL nativo di PyTorch
per scrivere kernel.

> **Note:** il supporto di Helion in TorchTPU è attualmente **sperimentale**.
> Per i carichi di lavoro di produzione, usa Tokamax o
> `torch.tpu.pallas.jax_op`.

### Come scrivere un kernel Helion { #how-to-write-a-helion-kernel }

Decora la funzione con `@helion.kernel`. Usa le operazioni PyTorch standard
per la configurazione lato host e `hl.tile` per definire il loop di
esecuzione sul dispositivo sui tile partizionati:

```python
import helion
import helion.language as hl
import torch

# 1. Define the kernel in Helion
@helion.kernel
def add_vectors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # Standard PyTorch host-side setup
  x, y = torch.broadcast_tensors(x, y)
  out = torch.empty_like(x)

  # Device execution loop over automatically partitioned tiles
  for tile in hl.tile(out.size()):
    out[tile] = x[tile] + y[tile]

  return out

# 2. Call it like a standard PyTorch function
x = torch.tensor([1.0, 2.0, 3.0], device="tpu")
y = torch.tensor([4.0, 5.0, 6.0], device="tpu")
z = add_vectors(x, y)
```

Per maggiori informazioni su Helion, consulta la
[documentazione di Helion](https://helionlang.com/), fai domande nel
[forum di discussione per sviluppatori PyTorch](https://dev-discuss.pytorch.org/c/helion/31)
oppure [segnala problemi su GitHub](https://github.com/pytorch/helion/issues).
