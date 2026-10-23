<p align="center">
 <img src="images/torchtpu_banner.png" alt="TorchTPU" style="width: 90%;">
</p>
<p align="center">
<a href="https://github.com/google-pytorch/torch_tpu"><b>GitHub</b></a> | <a href="https://discuss.pytorch.org/c/tpu/45"><b>Fórum de discussão</b></a> | <a href="https://docs.cloud.google.com/tpu/docs/pytorch-on-tpu"><b>PyTorch na TPU</b></a>
</p>

## Sobre { #about }

O TorchTPU é um backend nativo do PyTorch, de código aberto, para Google Cloud
TPUs, desenvolvido em colaboração entre o Google e a Meta. Com o TorchTPU, você
pode executar modelos PyTorch existentes em hardware TPU com mudanças mínimas
no código e escalonar cargas de trabalho de treinamento e inferência de um único
chip até pods de TPU com várias frações (multi-slice).

## O que você quer fazer hoje? { #what-are-you-trying-to-do-today }

<!-- mdformat off(reason: preserve MkDocs grid card comments and 4-space list indentation) -->
<!--mkdocs:<div class="grid cards two-columns" markdown>-->

- **<!--mkdocs::material-rocket-launch:{ .lg .middle }--> Primeiros passos**

    Provisione ambientes, entenda as arquiteturas fundamentais e execute suas primeiras cargas de trabalho.

    - [<!--mkdocs::octicons-arrow-right-24:--> Introdução ao TorchTPU](./intro_to_torchtpu.md): por que o TorchTPU foi criado, o que ele oferece e como ele pode ajudar você.
    - [<!--mkdocs::octicons-arrow-right-24:--> Primeiros passos com o TorchTPU em uma VM](./get_started_on_a_vm.md): configure o TorchTPU em uma VM de TPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Primeiros passos com o TorchTPU no GKE](./get_started_on_gke.md): use o TorchTPU por meio do GKE.
    - [<!--mkdocs::octicons-arrow-right-24:--> Recursos compatíveis e roadmap](./feature_set.md): recursos compatíveis no momento e planos para versões futuras.
    - [<!--mkdocs::octicons-arrow-right-24:--> Referência da API](./api_reference.md): a API do TorchTPU e as variáveis de ambiente da versão atual.
    - [<!--mkdocs::octicons-arrow-right-24:--> Perguntas frequentes sobre o TorchTPU](./faq.md): perguntas frequentes sobre o TorchTPU.

- **<!--mkdocs::material-sine-wave:{ .lg .middle }--> Arquitetura em detalhes e conceitos principais**

    Aprofunde-se no funcionamento dos layouts de memória, dos mecanismos de execução e da execução distribuída.

    - [<!--mkdocs::octicons-arrow-right-24:--> Como o TorchTPU funciona](./how_torchtpu_works.md): como o TorchTPU se integra ao PyTorch e ao hardware TPU, de cima a baixo.
    - [<!--mkdocs::octicons-arrow-right-24:--> O modelo de execução do TorchTPU](./torchtpu_execution_model.md): detalhes aprofundados de como o TorchTPU executa código PyTorch e dos diferentes modos de execução que ele oferece.
    - [<!--mkdocs::octicons-arrow-right-24:--> O modelo de memória do TorchTPU](./torchtpu_memory_model.md): como o TorchTPU lida com a forma como a TPU e o compilador XLA gerenciam a memória.
    - [<!--mkdocs::octicons-arrow-right-24:--> Suporte a execução distribuída](./distributed_support.md): como o TorchTPU oferece suporte a cargas de trabalho distribuídas.
    - [<!--mkdocs::octicons-arrow-right-24:--> Suporte a kernels personalizados](./custom_kernels.md): como escrever kernels personalizados para o TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Formatos dinâmicos](./dynamic_shapes.md): suporte a formatos (shapes) dinâmicos de tensores.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Estudos de caso**

    Veja exemplos práticos de como usar o TorchTPU para resolver problemas reais.

    - [<!--mkdocs::octicons-arrow-right-24:--> Otimização do FLUX](./case_studies_flux.md): exemplo que mostra como identificar e aplicar várias otimizações no FLUX.
    - [<!--mkdocs::octicons-arrow-right-24:--> Inferência do Muse Glimmer 30B](./case_studies_muse_glimmer_30b_inference.md): exemplo de decodificação iterativa com pesos reais e KV cache estático/dinâmico.
    - [<!--mkdocs::octicons-arrow-right-24:--> Treinamento do Muse Glimmer 30B](./case_studies_muse_glimmer_30b_training.md): um loop de treinamento em uma implementação independente do Muse Glimmer 30B.
    - [<!--mkdocs::octicons-arrow-right-24:--> Como usar FSDP](./case_studies_fsdp.md): a abordagem recomendada para usar FSDP com o TorchTPU.

- **<!--mkdocs::material-book-open-variant:{ .lg .middle }--> Analise sua carga de trabalho**

    - [<!--mkdocs::octicons-arrow-right-24:--> Precisão e paridade numéricas](./numerical_precision.md): como controlar a precisão no TorchTPU.
    - [<!--mkdocs::octicons-arrow-right-24:--> Crie perfis de desempenho com o XProf](./performance_profiling_with_xprof.md): como usar o XProf como profiler do TorchTPU.

<!--mkdocs:</div>-->
<!-- mdformat on -->
