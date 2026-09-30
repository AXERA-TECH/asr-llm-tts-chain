/* Copyright (c) 2011 The WebRTC project authors. BSD-style license. */
#ifndef RTC_BASE_MEMORY_ALIGNED_MALLOC_H_
#define RTC_BASE_MEMORY_ALIGNED_MALLOC_H_

#include <stddef.h>

namespace webrtc {

void* GetRightAlign(const void* ptr, size_t alignment);
void* AlignedMalloc(size_t size, size_t alignment);
void AlignedFree(void* mem_block);

template <typename T>
T* GetRightAlign(const T* ptr, size_t alignment) {
  return reinterpret_cast<T*>(
      GetRightAlign(reinterpret_cast<const void*>(ptr), alignment));
}
template <typename T>
T* AlignedMalloc(size_t size, size_t alignment) {
  return reinterpret_cast<T*>(AlignedMalloc(size, alignment));
}

struct AlignedFreeDeleter {
  inline void operator()(void* ptr) const { AlignedFree(ptr); }
};

}  // namespace webrtc

#endif  // RTC_BASE_MEMORY_ALIGNED_MALLOC_H_
