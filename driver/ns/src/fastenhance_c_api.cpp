#include "fastenhance_c_api.h"
#include "fastenhancer.hpp"

#include <new>
#include <string>

namespace {
thread_local std::string g_create_error;
}

extern "C" void* fastenhance_create(const char* model_path, int npu_core) {
  g_create_error.clear();
  if (npu_core != 0) {
    g_create_error = "FastEnhance requires NPU core 0";
    return nullptr;
  }
  if (model_path == nullptr || *model_path == '\0') {
    g_create_error = "FastEnhance model path is empty";
    return nullptr;
  }
  auto* enhancer = new (std::nothrow) FastEnhancer();
  if (enhancer == nullptr || !enhancer->load(model_path)) {
    if (enhancer != nullptr) g_create_error = enhancer->last_error();
    if (g_create_error.empty()) g_create_error = "FastEnhancer allocation/load failed";
    delete enhancer;
    return nullptr;
  }
  return enhancer;
}

extern "C" const char* fastenhance_create_last_error(void) {
  return g_create_error.c_str();
}

extern "C" int fastenhance_process(void* handle, const float* input,
                                    float* output) {
  if (handle == nullptr || input == nullptr || output == nullptr) return -1;
  return static_cast<FastEnhancer*>(handle)->process(input, output) ? 0 : -1;
}

extern "C" const char* fastenhance_last_error(void* handle) {
  if (handle == nullptr) return "invalid FastEnhance handle";
  return static_cast<FastEnhancer*>(handle)->last_error().c_str();
}

extern "C" void fastenhance_destroy(void* handle) {
  delete static_cast<FastEnhancer*>(handle);
}
