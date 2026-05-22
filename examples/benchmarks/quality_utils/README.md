# Adding New Models to the Quality Benchmark Suite

This directory contains utilities for the quality benchmark suite. Below are the
steps to add a new model to the suite.

## Step 1: Create Model Wrapper

Create a new file for your model wrapper and inherit from
`QualityBenchmarkModel`. You will need to implement the following methods: *
`initialize`: Load the model (e.g., from shared file system), set it to eval
mode, and handle the data type. * `max_seq_len`: Return the maximum sequence
length supported by the model. * `encode`: Tokenize and encode input text. *
`format`: Format the inputs, handling padding appropriately, and fallback to EOS
token if necessary. * `get_logits_and_targets`: Get logits from the model,
upcast them to float32 if needed, and handle attention masks for the targets. *
**Crucial Note on Attention Mask**: Some models (like Qwen) strictly require an
`attention_mask` in this method to avoid incorrect results or OOM on padding. *
**Crucial Note on Stability**: Always ensure that logits are upcast to `float32`
for loss calculation stability.

### Data Loading and Chunking

When processing data for evaluation, keep in mind: * `PerplexityMetric` drops
chunks shorter than `max_seq_len + 1`. * Therefore, the data loader must provide
long enough sequences, or concatenate documents (as done for datasets like
WikiText) to avoid `NaN` scores and ensure meaningful metrics.

## Step 2: Add Unit Test

Create a unit test for your model wrapper to ensure correctness. At a minimum,
verify the `format` and `get_logits_and_targets` methods by using a mocked
model. * **Mocking Recommendation**: Suggest mocking the model's `forward`
method to return dummy logits with expected shapes. This avoids running the
actual model in unit tests.

## Step 3: Update BUILD Files

Update the corresponding `BUILD` files to make your new model available: * Add a
target for your model wrapper in `models/BUILD`. * Add the new model wrapper
dependency in `e2e/BUILD`.

## Step 4: Integrate into E2E Suite

Finally, integrate your model into the end-to-end (E2E) testing suite: * Add a
configuration helper for your model in `model_quality_benchmarks.py`. * Add the
relevant test methods to execute benchmarks for the newly added model.
