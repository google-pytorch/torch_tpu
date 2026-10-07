/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TORCH_TPU_CSRC_COMMON_REROUTE_WARNINGS_H_
#define TORCH_TPU_CSRC_COMMON_REROUTE_WARNINGS_H_

// Utilities to surface C++ warnings raised on background threads to Python.
//
// A C++ warning (e.g. from `TT_WARN`) is dispatched to the warning handler
// of the thread that raises it. It reaches Python only when it is raised on a
// dispatch thread called by Python, where PyTorch's `PyWarningHandler` is
// installed. On any other thread, it is printed to stderr, consequently users
// cannot filter, capture or escalate it.
//
// To work around it, TorchTPU background threads reroute their warnings to a
// single process-wide buffer, and dispatch threads surface them after waiting
// for background work to complete. Rerouted warnings that are never surfaced
// are dropped, so dispatch threads must surface them after every wait on
// background work.

namespace torch_tpu {

// Makes the calling thread reroute its c10 warnings to dispatch threads, i.e.
// buffer them until a dispatch thread calls `SurfaceReroutedWarnings`. This
// function is thread-safe.
//
// Calling it replaces the thread's current c10 warning handler until another
// one is installed. Call it once when starting a background thread.
void RouteWarningsOnThisThreadToDispatchThread();

// Raises all rerouted warnings, from every background thread, through the
// calling thread's current c10 warning handler, in the order they were
// rerouted, and empties the buffer. This function is thread-safe.
//
// Call it on dispatch threads after waiting on background work. Python then
// attributes the warnings to the Python frame that waited, not the one that
// enqueued the work.
//
// Crashes if called on a thread that reroutes its warnings.
void SurfaceReroutedWarnings();

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_REROUTE_WARNINGS_H_
