/*
 * Copyright 2025 Google LLC
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

#if defined(BUILD_SCM_REVISION)
#define TT_STR_HELPER(x) #x
#define TT_STR(x) TT_STR_HELPER(x)
const char* torch_tpu_git_commit = TT_STR(BUILD_SCM_REVISION);
#elif defined(BUILD_CHANGELIST)
#define TT_STR_HELPER(x) #x
#define TT_STR(x) TT_STR_HELPER(x)
const char* torch_tpu_git_commit = TT_STR(BUILD_CHANGELIST);
#else
const char* torch_tpu_git_commit = "unknown";
#endif

extern "C" {
const char* get_torch_tpu_git_commit() { return torch_tpu_git_commit; }
}
