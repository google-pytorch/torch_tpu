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

#ifndef TORCH_TPU_COMMON_UNIQUE_FILE_DESCRIPTOR_H_
#define TORCH_TPU_COMMON_UNIQUE_FILE_DESCRIPTOR_H_

#include <unistd.h>

namespace torch_tpu {

// RAII class for managing a file descriptor. A UniqueFileDescriptor object
// owns a file descriptor and automatically closes it when the object goes out
// of scope. The interface is intentionally similar to the unique_ptr class.
class UniqueFileDescriptor {
 public:
  static constexpr int kInvalidFd = -1;

  // Takes ownership of the given file descriptor.
  explicit UniqueFileDescriptor(const int fd = kInvalidFd) : fd_(fd) {}

  // Closes the file descriptor if it is valid.
  ~UniqueFileDescriptor() { reset(); }

  // The class is not copyable but movable.
  UniqueFileDescriptor(const UniqueFileDescriptor&) = delete;
  UniqueFileDescriptor& operator=(const UniqueFileDescriptor&) = delete;
  UniqueFileDescriptor(UniqueFileDescriptor&& other) : fd_(other.release()) {}
  UniqueFileDescriptor& operator=(UniqueFileDescriptor&& other) {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  // Returns the underlying file descriptor without releasing ownership.
  [[nodiscard]] int get() const { return fd_; }

  // Releases ownership of the file descriptor.
  [[nodiscard]] int release() {
    const int fd = fd_;
    fd_ = kInvalidFd;
    return fd;
  }

  // Returns true if the file descriptor is valid.
  [[nodiscard]] bool valid() const { return fd_ != kInvalidFd; }

  // Sets the file descriptor to the given value. The existing file descriptor
  // is closed if it is valid.
  void reset(const int fd = kInvalidFd) {
    if (fd == fd_) return;
    if (valid()) {
      ::close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_ = kInvalidFd;
};

}  // namespace torch_tpu

#endif  // TORCH_TPU_COMMON_UNIQUE_FILE_DESCRIPTOR_H_
