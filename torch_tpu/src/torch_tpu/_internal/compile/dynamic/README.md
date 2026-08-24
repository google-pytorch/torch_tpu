# TPU Dynamic Compilation Backend

A backend compiler for handling dynamic shapes in PyTorch on TPU. It intercepts
the FX graph produced by AOTAutograd, modifies it to replace dynamic dimensions
(represented by `SymInt`) with their upper bounds, and inserts
`set_dimension_logical_size` operations to communicate the actual runtime sizes
to the XLA compiler. This allows generating a single static executable that can
handle variable shapes within the specified bounds, avoiding expensive
recompilations.

**Caveat:** While main model recompilation is avoided, the padding subgraph is
currently still recompiled for every unique set of input shapes during
execution.

## Compilation

-   **Initialize `SymShapeManager`**

    `SymShapeManager` tracks and resolves symbolic integers (`SymInt`) across
    input and output tensors. It stores input shape metadata, symbolic bounds,
    and output shape expressions needed by the compilation passes and execution
    wrapper.

    What it tracks:

    -   **Input Shape Metadata:** Identifies which input tensor dimensions are
        dynamic vs. static, resolving their positions across the signature. It
        tracks this via `InputTensorMetadata` (containing `static_shape`,
        `dynamic_bounds`, and `dynamic_dims`).
    -   **Symbolic Bounds (`SymInt` limits):** Determines lower and upper bounds
        for every dynamic size variable (either from `shape_env` or estimated
        from hints). Essential for calculating conservative static memory
        limits.
    -   **Output Shape Expressions:** Records formulas mapping output tensor
        shapes as algebraic expressions of the input dimensions (e.g., computing
        how shapes evolve through operations).

        -   Example:

            ```python
            def forward(self, arg0_1: "Sym(s77)", arg1_1: "i64[s77, 1]"):
              cat: "i64[2*s77, 1]" = torch.ops.aten.cat.default([arg1_1, arg1_1])
              return (cat,)
            ```

            Output shape info:

            ```python
            [ {'expr': '2*s77', 'deps': {'s77': (0, 0)}},  # dynamic dim
               1 ]  # static dim
            ```

    -   **Generative Op Support:** Creates and tracks specialized FX placeholder
        nodes linked to `SymInt` variables, enabling purely dynamic generative
        operations like `torch.arange`.

-   Modify FX Graph

    -   Pass to handle dynamic input tensors (`HandleDynamicInputTensorPass`)

        -   Identifies all input tensor placeholders with dynamic dimensions.
        -   Adds a new placeholder for runtime size.
        -   Inserts `set_dimension_logical_size` operation for dynamic
            placeholders and replaces the usage of those placeholders in the
            graph with the output of the `set_dimension_logical_size` operation.
        -   Example:

            ```python
            def forward(x: torch.Tensor):
                z = x + 3.0
                return z
            ```

            Input Fx Graph:

            ```python
            def forward(self, arg0_1: "Sym(s27)", arg1_1: "i64[1, s27]"):
                add: "f32[1, s27]" = torch.ops.aten.add.Tensor(arg1_1, 3.0)
                return (add,)
            ```

            Modified Fx Graph:

            ```python
            def forward(self, arg0_1: "Sym(s27)", arg1_1: "i64[1, s27]", dyn_size_1_dim1: "i64[]"):
                set_dimension_logical_size: "i64[1, s27]" = torch.ops.tpu.set_dimension_logical_size(arg1_1, 1, dyn_size_1_dim1)
                add: "f32[1, s27]" = torch.ops.aten.add.Tensor(set_dimension_logical_size, 3.0)
                return (add,)
            ```

    -   Pass to handle scalar symints as input to generative ops
        (`HandleGenerativeOpsPass`)

        -   Identifies generative ops (currently only `torch.arange`) that have
            dynamic scalar inputs.
        -   Adds a new placeholder for runtime size.
        -   Replaces the scalar symint with its static upper bound.
        -   Adds `set_dimension_logical_size` op on the output of generative
            ops.
        -   Replaces the usage of generative op output tensors in the graph with
            the output of the `set_dimension_logical_size` operation.
        -   Example:

            ```python
            def forward(x: torch.Tensor):
                z = torch.arange(0, x.shape[0])
                return z
            ```

            Test case: `x` = `torch.tensor([8])`

            Input Fx Graph:

            ```python
            def forward(self, arg0_1: "Sym(s77)"):
                arange: "i64[s77]" = torch.ops.aten.arange.start(0, arg0_1)
                return (arange,)
            ```

            Modified Fx Graph:

            ```python
            def forward(self, arg0_1: "Sym(s77)", dyn_size_0: "i32[]"):
                arange: "i64[s77]" = torch.ops.aten.arange.start(0, 16)  # Upper bound = 2 * 8 = 16
                set_dimension_logical_size: "i64[s77]" = torch.ops.tpu.set_dimension_logical_size(arange, 0, dyn_size_0)
                return (set_dimension_logical_size,)
            ```

-   Create example inputs by replacing `SymInt` with upper bounds for the
    updated FX graph to trace and generate MLIR. Scalar `SymInt` is converted to
    a tensor.

-   Use the static TPU backend to compile the updated FX graph using the upper
    bound inputs to generate static executable.

-   Create a composite executable that wraps the bound information and the
    static executable.

    ```python
    _DynamicTpuCompiledExecutable(
        model_executable=static_model_executable,
        sym_shape_manager=sym_shape_manager)
    ```

## Execution

Dynamic TPU Backend Execution Flow:

-   Example model

    ```python
    def simple_add(x):
      return x + 3
    ```

-   Scalar inputs are converted to tensors iff a placeholder was created for it,
    otherwise they are ignored.

-   Compile and execute the pad subgraph (`_compile_and_execute_pad_subgraph`).

    -   Converts dynamic runtime tensors into statically padded tensors and size
        tensors.
    -   Non-dynamic inputs are passed through unchanged.
    -   MLIR for Input `[1, 3]`, `Dim 1 upper bound = 2*3 = 6`

    ```mlir
    module @pad_module {
      func.func @main(%arg0: tensor<1x3xi64>) -> (tensor<1x6xi64>, tensor<i32>) {
        %c = stablehlo.constant dense<0> : tensor<i64>
        %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 3], interior = [0, 0] : (tensor<1x3xi64>, tensor<i64>) -> tensor<1x6xi64>
        %1 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x3xi64>) -> tensor<i32>
        return %0, %1 : tensor<1x6xi64>, tensor<i32>
      }
    }
    ```

    -   MLIR for Input `[1, 4]`

    ```mlir
    module @pad_module {
      func.func @main(%arg0: tensor<1x4xi64>) -> (tensor<1x6xi64>, tensor<i32>) {
        %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
        %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 2], interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x6xi64>
        %1 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x4xi64>) -> tensor<i32>
        return %0, %1 : tensor<1x6xi64>, tensor<i32>
      }
    }
    ```

-   Compute output shapes based on the input argument shapes.

    -   Extract output metadata containing dynamic expressions and solve them
        via `SymPy` substitution using runtime input shapes.

-   Model execution:

    -   Output DeviceBufferRefs are created based on the runtime output shapes.
    -   The model is executed using the padded module outputs as inputs.

    ```mlir
    module @simple_add {
     func.func @main(%arg0: tensor<1x6xi64> loc(unknown), %arg1: tensor<i32> loc(unknown)) -> tensor<1x?xf32, #stablehlo.bounds<?, 6>>
       %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 1 : (tensor<1x6xi64>, tensor<i32>) -> tensor<1x?xi64, #stablehlo.bounds<?, 6>>
       ...
       return %6 : tensor<1x?xf32, #stablehlo.bounds<?, 6>>
    }
    ```

    -   Slicing only needed for relayouts.

## Recompilations

Utilizes `torch._check` on SymInt bounds to prevent unnecessary recompilations.
However, during execution there is still compilation on the pad module.

## Performance

-   Experiment with `stablehlo.dynamic_update_slice` to remove pad module
    compilation during execution.
