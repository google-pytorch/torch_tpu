# StableHLO and CHLO Builder APIs

To use StableHLO and CHLO builder APIs, you may need to reference the generated
header files. If they are not present, run the corresponding `bazel build`
command.

The generated files live in the StableHLO external repository under
`bazel-bin/external/`, in a directory whose name Bazel chooses; run
`ls bazel-bin/external/ | grep stablehlo` to find it (`<stablehlo_repo>` below).

## StableHLO Builder APIs

*   **File**:
    `bazel-bin/external/<stablehlo_repo>/stablehlo/integrations/cpp/builder/StablehloBuilder.h.inc`
*   **Build Command**: `bazel build
    @stablehlo//stablehlo/integrations/cpp/builder:stablehlo_builder_inc`

## CHLO Builder APIs

*   **File**:
    `bazel-bin/external/<stablehlo_repo>/stablehlo/integrations/cpp/builder/ChloBuilder.h.inc`
*   **Build Command**: `bazel build
    @stablehlo//stablehlo/integrations/cpp/builder:chlo_builder_inc`
