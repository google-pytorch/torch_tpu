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

#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "pybind11/pybind11.h"
#include "torch_tpu/csrc/common/context_manager.h"
#include "torch_tpu/csrc/common/context_states.h"
#include "torch_tpu/csrc/common/error_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/rpc/profiler_server.h"

namespace torch_tpu {

namespace py = pybind11;

// Thread-safe wrapper for tsl::profiler::ProfilerServer.
class TpuProfilerServer {
 public:
  // Creates a server in not-started state.
  TpuProfilerServer() = default;

  // This class is neither copyable nor movable.
  TpuProfilerServer(const TpuProfilerServer&) = delete;
  TpuProfilerServer& operator=(const TpuProfilerServer&) = delete;
  TpuProfilerServer(TpuProfilerServer&&) = delete;
  TpuProfilerServer& operator=(TpuProfilerServer&&) = delete;

  // Starts the profiler server on the given port. Fails if the server has
  // already been started.
  absl::Status Start(int port) ABSL_LOCKS_EXCLUDED(mutex_);

  // Stops the profiler server. Fails if the server has not been started or has
  // already been stopped.
  absl::Status Stop() ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global profiler server instance.
  static TpuProfilerServer& GetInstance() {
    static absl::NoDestructor<TpuProfilerServer> global_server;
    return *global_server;
  }

 private:
  mutable absl::Mutex mutex_;
  absl_nullable std::unique_ptr<tsl::profiler::ProfilerServer> server_
      ABSL_GUARDED_BY(mutex_);
};

// Thread-safe wrapper for tsl::ProfilerSession.
class TpuProfilerSession {
 public:
  // Creates a session in not-started state.
  TpuProfilerSession() = default;

  // This class is neither copyable nor movable.
  TpuProfilerSession(const TpuProfilerSession&) = delete;
  TpuProfilerSession& operator=(const TpuProfilerSession&) = delete;
  TpuProfilerSession(TpuProfilerSession&&) = delete;
  TpuProfilerSession& operator=(TpuProfilerSession&&) = delete;

  // Starts the profiler session on the given log directory with the given
  // options. Fails if the session has already been started.
  absl::Status Start(const std::string& logdir, py::object options_obj)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Stops the profiler session and writes the trace data to the given file.
  // Fails if the session has not been started or has already been stopped.
  absl::Status Stop(const std::string& filename) ABSL_LOCKS_EXCLUDED(mutex_);

  // Returns the global profiler session instance.
  static TpuProfilerSession& GetInstance() {
    static absl::NoDestructor<TpuProfilerSession> global_session;
    return *global_session;
  }

 private:
  mutable absl::Mutex mutex_;
  absl_nullable std::unique_ptr<tsl::ProfilerSession> session_
      ABSL_GUARDED_BY(mutex_);
};

absl::Status TpuProfilerServer::Start(int port) ABSL_LOCKS_EXCLUDED(mutex_) {
  absl::MutexLock lock(mutex_);
  TT_RET_CHECK(server_ == nullptr, error::kFailedPrecondition)
      << "the profiler server has already been started";
  server_ = std::make_unique<tsl::profiler::ProfilerServer>();
  server_->StartProfilerServer(port);
  return absl::OkStatus();
}

absl::Status TpuProfilerServer::Stop() ABSL_LOCKS_EXCLUDED(mutex_) {
  absl::MutexLock lock(mutex_);
  TT_RET_CHECK(server_ != nullptr, error::kFailedPrecondition)
      << "the profiler server has not been started or has already been "
         "stopped";
  server_.reset();
  return absl::OkStatus();
}

absl::Status TpuProfilerSession::Start(const std::string& logdir,
                                       py::object options_obj) {
  absl::MutexLock lock(mutex_);
  TT_RET_CHECK(session_ == nullptr, error::kFailedPrecondition)
      << "the profiler session has already been started";

  tensorflow::ProfileOptions opts = tsl::ProfilerSession::DefaultOptions();
  if (!options_obj.is_none()) {
    py::dict options = options_obj.cast<py::dict>();
    if (options.contains("host_tracer_level")) {
      opts.set_host_tracer_level(options["host_tracer_level"].cast<int>());
    }
    if (options.contains("device_tracer_level")) {
      opts.set_device_tracer_level(options["device_tracer_level"].cast<int>());
    }
    if (options.contains("python_tracer_level")) {
      opts.set_python_tracer_level(options["python_tracer_level"].cast<int>());
    }
  }
  if (!logdir.empty()) {
    opts.set_repository_path(logdir);
  }
  ABSL_LOG(INFO) << "Starting trace, logdir: " << logdir;

  // Put the new session in a temporary variable first. Only update
  // session_ if the session is successfully created. This way, if the new
  // session has a bad status, session_ will still be null and the user
  // can try to create a session again.
  auto session = tsl::ProfilerSession::Create(opts);
  TT_RETURN_IF_ERROR(session->Status()).SetPrepend()
      << "failed to start profiler session: ";

  session_ = std::move(session);
  return absl::OkStatus();
}

absl::Status TpuProfilerSession::Stop(const std::string& filename) {
  absl::MutexLock lock(mutex_);
  TT_RET_CHECK(session_ != nullptr, error::kFailedPrecondition)
      << "the profiler session has not been started or has already been "
         "stopped";

  // Set an auto-cleanup callback to ensure that the session is deleted
  // regardless of whether an error occurs below.
  absl::Cleanup cleanup = [this] {
    // cleanup is guaranteed to destruct before the lock destructs,
    // so accessing the session_ here is safe. However, the compiler
    // isn't smart enough to know this, so we need to assert that the mutex is
    // held. Without this, the code won't compile.
    mutex_.AssertHeld();
    session_.reset();
  };

  tensorflow::profiler::XSpace xspace;
  TT_RETURN_IF_ERROR(session_->CollectData(&xspace))
      << "failed to collect trace data";
  ABSL_LOG(INFO) << "Collected " << xspace.planes_size() << " planes.";
  tsl::Env* env = tsl::Env::Default();

  // Ensure the parent directory exists
  std::string dirname(tsl::io::Dirname(filename));
  if (!dirname.empty()) {
    TT_RETURN_IF_ERROR(env->RecursivelyCreateDir(dirname))
        << "failed to create directory: " << dirname;
  }

  std::unique_ptr<tsl::WritableFile> outfile;
  TT_RETURN_IF_ERROR(env->NewWritableFile(filename, &outfile))
      << "failed to create file: " << filename;

  // Serialize the collected XSpace data to a string.
  std::string serialized_proto;
  TT_RET_CHECK(xspace.SerializeToString(&serialized_proto), error::kInternal)
      << "failed to serialize profile data";
  ABSL_LOG(INFO) << "Writing " << serialized_proto.size() << " bytes to "
                 << filename;

  // Write the serialized XSpace to the file.
  TT_RETURN_IF_ERROR(outfile->Append(serialized_proto))
      << "failed to write data to " << filename;
  TT_RETURN_IF_ERROR(outfile->Close()) << "failed to close file: " << filename;

  return absl::OkStatus();
}

static void PyStartProfilerServer(int port) {
  TT_THROW_IF_ERROR(TpuProfilerServer::GetInstance().Start(port));
}

static void PyStopProfilerServer() {
  TT_THROW_IF_ERROR(TpuProfilerServer::GetInstance().Stop());
}

static void PyStartTrace(const std::string& logdir, py::object options_obj) {
  TT_THROW_IF_ERROR(
      TpuProfilerSession::GetInstance().Start(logdir, options_obj));
}

static void PyStopTrace(const std::string& filename) {
  TT_THROW_IF_ERROR(TpuProfilerSession::GetInstance().Stop(filename));
}

PYBIND11_MODULE(_profiler_backend, m) {
  m.doc() = "PjRt backend for PyTorch profiler.";

  m.def("start_trace", &PyStartTrace, py::arg("logdir"),
        py::arg("options") = py::none(),
        "Starts a profiler trace with the given options; fails if the trace "
        "has already been started.");
  m.def("stop_trace", &PyStopTrace,
        "Stops the profiler trace and writes the trace data to the log dir; "
        "fails if the trace has not been started or has already been stopped.");
  m.def("start_profiler_server", &PyStartProfilerServer, py::arg("port"),
        "Starts the profiler gRPC server on the given port; fails if the "
        "server has already been started.");
  m.def("stop_profiler_server", &PyStopProfilerServer,
        "Stops the profiler gRPC server; fails if the server has not been "
        "started or has already been stopped.");

  m.def("_push_enable_profiler",
        []() { PushContextState(ProfilerStatus::kEnabled); });
  m.def("_pop_enable_profiler",
        []() { PopContextState<ProfileContextState>(); });
}

}  // namespace torch_tpu
