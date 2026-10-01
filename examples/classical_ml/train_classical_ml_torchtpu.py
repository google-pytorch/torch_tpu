# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classical ML (MLP) Training & Profiling on Cloud TPU v6e using TorchTPU.

Demonstrates:
1. Native eager mode training on TPU
   (`device = torch.accelerator.current_accelerator()`).
2. Ahead-of-time compiled mode training on TPU
   (`torch.compile(model, backend='tpu', dynamic=False)`).
3. Asynchronous Micro-Batching:
   - Executes micro-batch windows (`micro_batches=8`) continuously on device
     using `non_blocking=True` transfers and deferred on-device loss reduction
     (`losses.append(loss.detach())` followed by `torch.stack(losses).sum()`).
   - Avoids per-step eager addition dispatches and host-device synchronization
     stalls (`synchronize()` or `.item()` anti-patterns), synchronizing only
     once per window via `.item()`.
4. Comprehensive Monitoring & Telemetry:
   - Model Metrics: Loss, Step Latency (ms), and Test Accuracy in TensorBoard.
   - TPU Profiler: Step-scheduled XProf trace capture (`xplane.pb`) via
     `torch.profiler.profile` and `torch.profiler.schedule`.
5. TPU Best Practices: `DataLoader(num_workers=0)` fork-safety, static shapes,
   MXU tile alignment.
"""

import argparse
from collections.abc import Iterator
import importlib.metadata
import os
import sys
import time
from typing import Dict, Optional, Tuple, cast

import torch
from torch import nn
from torch.tpu.profiler import TpuProfilerConfig  # pyrefly: ignore[missing-import]
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch_tpu
from torchvision import datasets
from torchvision import transforms


class ClassicalMLP(nn.Module):
  """3-layer Multi-Layer Perceptron for classification.

  Dimensions (784 -> 256 -> 256 -> 10) are chosen to align cleanly with
  TPU Matrix Execution Unit (MXU) systolic array tiles (multiples of 128/256).
  """

  def __init__(
      self,
      input_dim: int = 784,
      hidden_dim: int = 256,
      num_classes: int = 10,
  ):
    super().__init__()
    self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


def get_data_loaders(
    data_dir: str = "./data", batch_size: int = 256
) -> Tuple[DataLoader, DataLoader]:
  """Prepares standard MNIST dataset.

  NOTE: On TPU, num_workers must be 0 to prevent process fork deadlocks
  after PJRT runtime initialization.

  Args:
    data_dir: Local path to download or load MNIST dataset.
    batch_size: Batch size per iteration.

  Returns:
    A tuple of (train_loader, test_loader).
  """
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
  ])

  train_dataset = datasets.MNIST(
      data_dir, train=True, download=True, transform=transform
  )
  test_dataset = datasets.MNIST(
      data_dir, train=False, download=True, transform=transform
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      drop_last=True,  # Avoids dynamic-shaped partial trailing batches on TPU
      num_workers=0,  # Avoids PJRT fork deadlock
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      drop_last=True,
      num_workers=0,
  )
  return train_loader, test_loader


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    data_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    micro_batches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
  """Executes a single macro step comprising multiple micro-batches on TPU.

  Avoids host-device synchronization inside the inner micro-batch loop by
  transferring tensors with `non_blocking=True` and accumulating loss locally
  on the TPU device.

  Args:
    model: The PyTorch neural network module to train.
    optimizer: PyTorch optimizer.
    criterion: Loss function module.
    data_iterator: Iterator yielding (inputs, targets) batches.
    micro_batches: Number of micro-batch iterations per macro step.
    device: Target execution device (`torch.device('tpu')`).

  Returns:
    Tuple of (accumulated_loss_on_device, actual_steps, samples_processed).
  """
  losses = []
  actual_steps = 0
  samples_processed = 0

  for _ in range(micro_batches):
    try:
      inputs, targets = next(data_iterator)
    except StopIteration:
      break

    # 1. Asynchronous Host-to-Device transfer
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    # 2. Forward & backward pass
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # 3. Store detached loss tensor in list and reduce once after the loop
    losses.append(loss.detach())
    actual_steps += 1
    samples_processed += inputs.size(0)

  accumulated_loss = (
      torch.stack(losses).sum() if losses else torch.tensor(0.0, device=device)
  )
  return accumulated_loss, actual_steps, samples_processed


def log_metrics(
    writer: Optional[SummaryWriter],
    metrics: Dict[str, float],
    step: int,
) -> None:
  """Logs a dictionary of scalar metrics to TensorBoard if writer is present."""
  if writer is None:
    return
  for tag, value in metrics.items():
    writer.add_scalar(tag, value, step)


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 3,
    micro_batches: int = 8,
    mode_name: str = "Eager",
    writer: Optional[SummaryWriter] = None,
    profiler: Optional[torch.profiler.profile] = None,
) -> Dict[str, float]:
  """Executes the training loop with micro-batching & telemetry.

  Args:
    model: The PyTorch neural network module to train.
    train_loader: DataLoader providing training batches.
    optimizer: PyTorch optimizer.
    criterion: Loss function module.
    device: Target execution device (`torch.device('tpu')`).
    num_epochs: Number of training epochs.
    micro_batches: Number of iterations per asynchronous micro-batch window.
    mode_name: Display label for logging ('Eager' or 'Compiled').
    writer: Optional TensorBoard SummaryWriter for scalar logging.
    profiler: Optional PyTorch profiler instance for step boundary marking.

  Returns:
    Dictionary containing throughput, duration, and latency metrics.
  """
  print(f"\n{'='*25} Running {mode_name} Training {'='*25}", flush=True)
  model.train()

  total_samples = 0
  total_steps = 0
  overall_start_time = time.perf_counter()
  step_latencies = []
  configured_batch_size = train_loader.batch_size or 256
  total_batches_per_epoch = len(train_loader)

  for epoch in range(num_epochs):
    epoch_start = time.perf_counter()
    running_loss = 0.0
    num_batches = 0
    epoch_samples = 0
    data_iter = iter(train_loader)
    macro_idx = 0

    while True:
      window_start = time.perf_counter()
      accumulated_loss, actual_steps, window_samples = train_step(
          model=model,
          optimizer=optimizer,
          criterion=criterion,
          data_iterator=data_iter,
          micro_batches=micro_batches,
          device=device,
      )
      if actual_steps == 0:
        break

      # Moving accumulated loss to CPU via .item() implicitly synchronizes
      # the TPU execution pipeline across the entire micro-batch window.
      window_avg_loss = float((accumulated_loss / actual_steps).item())
      window_duration = time.perf_counter() - window_start

      step_duration = window_duration / actual_steps
      for _ in range(actual_steps):
        step_latencies.append(step_duration)

      total_samples += window_samples
      epoch_samples += window_samples
      total_steps += actual_steps
      num_batches += actual_steps
      running_loss += window_avg_loss * actual_steps
      macro_idx += 1

      step_throughput = (
          window_samples / window_duration if window_duration > 0 else 0.0
      )

      log_metrics(
          writer,
          {
              "Loss/train": window_avg_loss,
              "Perf/step_time_ms": step_duration * 1000.0,
          },
          total_steps,
      )

      if profiler is not None:
        profiler.step()

      if macro_idx % 6 == 1 or num_batches >= total_batches_per_epoch:
        print(
            f"[{mode_name}] Epoch {epoch+1:2d}/{num_epochs:2d} | Step"
            f" {num_batches:3d}/{total_batches_per_epoch} | Loss:"
            f" {window_avg_loss:.4f} | Latency: {step_duration*1000:5.2f}ms |"
            f" Throughput: {step_throughput:7.1f} samples/s",
            flush=True,
        )

    epoch_duration = time.perf_counter() - epoch_start
    epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
    epoch_throughput = (
        epoch_samples / epoch_duration if epoch_duration > 0 else 0.0
    )

    print(
        f"[{mode_name}] Epoch {epoch+1:2d} Finished in {epoch_duration:.3f}s |"
        f" Avg Loss: {epoch_loss:.4f} | Epoch Throughput:"
        f" {epoch_throughput:7.1f} samples/s",
        flush=True,
    )

  total_duration = time.perf_counter() - overall_start_time
  avg_throughput = total_samples / total_duration if total_duration > 0 else 0.0
  avg_latency_ms = (
      (sum(step_latencies) / len(step_latencies)) * 1000.0
      if step_latencies
      else 0.0
  )
  cold_window_step_ms = step_latencies[0] * 1000.0 if step_latencies else 0.0
  # Measure steady-state warm performance across Epoch 2+ (after Epoch 1
  # compilation and scheduled profiler trace serialization have completed).
  steady_start = (
      total_batches_per_epoch
      if len(step_latencies) > total_batches_per_epoch
      else micro_batches
  )
  warm_latencies = (
      step_latencies[steady_start:]
      if len(step_latencies) > steady_start
      else step_latencies
  )
  warm_avg_latency_ms = (
      (sum(warm_latencies) / len(warm_latencies)) * 1000.0
      if warm_latencies
      else 0.0
  )
  warm_avg_throughput = (
      (configured_batch_size / (warm_avg_latency_ms / 1000.0))
      if warm_avg_latency_ms > 0
      else 0.0
  )

  print(f"\n{'='*25} {mode_name} Summary {'='*25}", flush=True)
  print(f"Total Time:          {total_duration:.3f}s", flush=True)
  print(f"Warm Avg Step Time:  {warm_avg_latency_ms:.2f}ms", flush=True)
  print(
      f"Warm Avg Throughput: {warm_avg_throughput:7.1f} samples/sec\n",
      flush=True,
  )

  return {
      "throughput": avg_throughput,
      "duration": total_duration,
      "avg_latency_ms": avg_latency_ms,
      "cold_window_step_ms": cold_window_step_ms,
      "warm_avg_latency_ms": warm_avg_latency_ms,
      "warm_avg_throughput": warm_avg_throughput,
  }


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    tag: str = "Test",
) -> float:
  """Evaluates model classification accuracy on the test dataset.

  Accumulates correct predictions locally on the TPU device and performs
  a single device-to-host synchronization via `.item()` at the end.

  Args:
    model: The PyTorch neural network module to evaluate.
    test_loader: DataLoader providing test batches.
    device: Target execution device (`torch.device('tpu')`).
    writer: Optional TensorBoard SummaryWriter for scalar logging.
    tag: Display tag for logging.

  Returns:
    Top-1 accuracy percentage on the test dataset.
  """
  model.eval()
  correct_counts = []
  total = 0
  with torch.no_grad():
    for inputs, targets in test_loader:
      inputs = inputs.to(device, non_blocking=True)
      targets = targets.to(device, non_blocking=True)
      outputs = model(inputs)
      preds = outputs.argmax(dim=1)
      correct_counts.append((preds == targets).sum())
      total += targets.size(0)

  # Single on-device reduction and host synchronization after the loop
  correct = (
      int(torch.stack(correct_counts).sum().item()) if correct_counts else 0
  )
  accuracy = 100.0 * correct / total if total > 0 else 0.0
  print(f"Test Accuracy ({tag}): {accuracy:.2f}%\n", flush=True)

  log_metrics(writer, {f"Accuracy/{tag}": accuracy}, 1)

  return accuracy


def main():
  parser = argparse.ArgumentParser(
      description="TorchTPU Classical ML Training & Profiling Benchmark"
  )
  parser.add_argument(
      "--data_dir",
      type=str,
      default="./data",
      help="Directory where MNIST dataset is staged",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default="./runs/tpu_classical_ml",
      help="Base directory for TensorBoard logs",
  )
  parser.add_argument(
      "--batch_size", type=int, default=256, help="Training batch size"
  )
  parser.add_argument(
      "--micro_batches",
      type=int,
      default=8,
      help="Number of iterations per asynchronous micro-batch window",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=3,
      help="Number of epochs per experiment",
  )
  parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
  parser.add_argument(
      "--enable_profiler",
      action="store_true",
      help=(
          "Capture step-scheduled TPU XProf hardware traces (*.xplane.pb) into"
          " log_dir."
      ),
  )
  parser.add_argument(
      "--profile_wait",
      type=int,
      default=2,
      help="Number of macro-step windows to skip before profiling",
  )
  parser.add_argument(
      "--profile_warmup",
      type=int,
      default=1,
      help="Number of macro-step windows for profiler warmup",
  )
  parser.add_argument(
      "--profile_active",
      type=int,
      default=3,
      help="Number of active macro-step windows to capture in XProf trace",
  )
  args = parser.parse_args()

  resolved_log_dir = os.path.expanduser(args.log_dir)

  print(
      "================================================================",
      flush=True,
  )
  print(
      " TorchTPU Classical ML Training & Profiling Benchmark on TPU v6e",
      flush=True,
  )
  try:
    torchtpu_ver = importlib.metadata.version("torch_tpu")
  except importlib.metadata.PackageNotFoundError:
    torchtpu_ver = "installed"
  print(
      f" Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} |"
      f" TorchTPU: {torchtpu_ver}",
      flush=True,
  )
  print(
      "================================================================",
      flush=True,
  )

  # 1. Setup TensorBoard writers
  eager_writer: Optional[SummaryWriter] = None
  compiled_writer: Optional[SummaryWriter] = None
  if resolved_log_dir:
    eager_logdir = os.path.join(resolved_log_dir, "eager")
    compiled_logdir = os.path.join(resolved_log_dir, "compiled")
    os.makedirs(eager_logdir, exist_ok=True)
    os.makedirs(compiled_logdir, exist_ok=True)
    eager_writer = SummaryWriter(log_dir=eager_logdir)
    compiled_writer = SummaryWriter(log_dir=compiled_logdir)
    print(f"[TensorBoard] Logging metrics to {resolved_log_dir}", flush=True)
    print(
        f"[TensorBoard] View via: tensorboard --logdir={resolved_log_dir}"
        " --port=6006",
        flush=True,
    )

  # 2. Setup TPU Device and Data Loaders
  device = torch.accelerator.current_accelerator() or torch.device("tpu")
  print(f"\n[Hardware] Target Device: {device}", flush=True)
  train_loader, test_loader = get_data_loaders(
      data_dir=args.data_dir, batch_size=args.batch_size
  )
  criterion = nn.CrossEntropyLoss()

  # -------------------------------------------------------------------------
  # Experiment 1: Eager Mode Execution on TPU
  # -------------------------------------------------------------------------
  print("\n--- Phase 1: Eager Mode Execution on TPU ---", flush=True)
  torch.manual_seed(42)
  eager_model = ClassicalMLP().to(device)
  eager_optimizer = torch.optim.Adam(eager_model.parameters(), lr=args.lr)

  eager_metrics = run_training_loop(
      eager_model,
      train_loader,
      eager_optimizer,
      criterion,
      device,
      num_epochs=args.epochs,
      micro_batches=args.micro_batches,
      mode_name="Eager",
      writer=eager_writer,
  )
  print("Evaluating Eager Model...", flush=True)
  eager_accuracy = evaluate(
      eager_model, test_loader, device, writer=eager_writer, tag="Eager"
  )

  # -------------------------------------------------------------------------
  # Experiment 2: Compiled Mode (torch.compile) + Profiler Trace
  # -------------------------------------------------------------------------
  print(
      "\n--- Phase 2: Compiled Mode Execution on TPU (torch.compile) ---",
      flush=True,
  )
  torch.manual_seed(42)
  raw_model = ClassicalMLP().to(device)
  compiled_model = cast(
      nn.Module, torch.compile(raw_model, backend="tpu", dynamic=False)
  )
  compiled_optimizer = torch.optim.Adam(compiled_model.parameters(), lr=args.lr)

  if args.enable_profiler:
    try:
      print(
          "[Profiler] Capturing scheduled TPU XProf trace"
          f" (wait={args.profile_wait}, warmup={args.profile_warmup},"
          f" active={args.profile_active}) into {resolved_log_dir}...",
          flush=True,
      )
      tpu_activity = getattr(torch.profiler.ProfilerActivity, "TPU")
      with torch.profiler.profile(
          activities=[
              torch.profiler.ProfilerActivity.CPU,
              tpu_activity,
          ],
          schedule=torch.profiler.schedule(
              wait=args.profile_wait,
              warmup=args.profile_warmup,
              active=args.profile_active,
              repeat=1,
          ),
          on_trace_ready=torch.profiler.tensorboard_trace_handler(
              resolved_log_dir
          ),
          experimental_config=TpuProfilerConfig(run_dir=resolved_log_dir),
      ) as prof:
        compiled_metrics = run_training_loop(
            compiled_model,
            train_loader,
            compiled_optimizer,
            criterion,
            device,
            num_epochs=args.epochs,
            micro_batches=args.micro_batches,
            mode_name="Compiled (torch.compile)",
            writer=compiled_writer,
            profiler=prof,
        )
      print(
          "[Profiler] TPU XProf trace successfully captured and saved to disk!",
          flush=True,
      )
    except (RuntimeError, OSError) as e:
      print(
          f"[Profiler] Notice: Trace capture encountered: {e}. Running standard"
          " loop...",
          flush=True,
      )
      compiled_metrics = run_training_loop(
          compiled_model,
          train_loader,
          compiled_optimizer,
          criterion,
          device,
          num_epochs=args.epochs,
          micro_batches=args.micro_batches,
          mode_name="Compiled (torch.compile)",
          writer=compiled_writer,
      )
  else:
    compiled_metrics = run_training_loop(
        compiled_model,
        train_loader,
        compiled_optimizer,
        criterion,
        device,
        num_epochs=args.epochs,
        micro_batches=args.micro_batches,
        mode_name="Compiled (torch.compile)",
        writer=compiled_writer,
    )

  print("Evaluating Compiled Model...", flush=True)
  compiled_accuracy = evaluate(
      compiled_model,
      test_loader,
      device,
      writer=compiled_writer,
      tag="Compiled",
  )

  # 4. Flush TensorBoard writers
  if eager_writer:
    eager_writer.close()
  if compiled_writer:
    compiled_writer.close()

  # 5. Final Speedup & Comparative Summary
  eager_warm_tp = eager_metrics["warm_avg_throughput"]
  compiled_warm_tp = compiled_metrics["warm_avg_throughput"]
  warm_speedup = compiled_warm_tp / eager_warm_tp if eager_warm_tp > 0 else 1.0

  print("\n" + "=" * 65, flush=True)
  print("                FINAL BENCHMARK COMPARISON", flush=True)
  print("=" * 65, flush=True)
  print(
      " Metric                     | Eager Mode       | Compiled Mode",
      flush=True,
  )
  print(
      " ---------------------------+------------------+-----------------",
      flush=True,
  )
  print(
      " Warm Step Latency (ms)     |"
      f" {eager_metrics['warm_avg_latency_ms']:12.2f}     |"
      f" {compiled_metrics['warm_avg_latency_ms']:12.2f}",
      flush=True,
  )
  print(
      f" Warm Throughput (samples/s)| {eager_warm_tp:12.1f}     |"
      f" {compiled_warm_tp:12.1f}",
      flush=True,
  )
  print(
      " Cold Window Step Time (ms) |"
      f" {eager_metrics['cold_window_step_ms']:12.2f}     |"
      f" {compiled_metrics['cold_window_step_ms']:12.2f}",
      flush=True,
  )
  print(
      f" Total Train Duration (s)   | {eager_metrics['duration']:12.3f}     |"
      f" {compiled_metrics['duration']:12.3f}",
      flush=True,
  )
  print(
      f" Test Accuracy (%)          | {eager_accuracy:12.2f}%    |"
      f" {compiled_accuracy:12.2f}%",
      flush=True,
  )
  print(
      " ---------------------------+------------------+-----------------",
      flush=True,
  )
  print(
      " Steady-State Speedup Factor (Warm Compiled vs Eager):"
      f" {warm_speedup:.2f}x",
      flush=True,
  )
  print("=" * 65, flush=True)
  print(
      "\nTo inspect the visual dashboards & TPU Profile trace in TensorBoard:",
      flush=True,
  )
  print(f"  tensorboard --logdir={resolved_log_dir} --port=6006", flush=True)
  print("=" * 65 + "\n", flush=True)


if __name__ == "__main__":
  main()
