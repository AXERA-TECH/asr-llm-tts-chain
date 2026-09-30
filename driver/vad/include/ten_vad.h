//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef TEN_VAD_H
#define TEN_VAD_H

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__linux__)
#define TENVAD_API __attribute__((visibility("default")))
#elif defined(_WIN32) || defined(__CYGWIN__)
#ifdef TENVAD_EXPORTS
#define TENVAD_API __declspec(dllexport)
#else
#define TENVAD_API __declspec(dllimport)
#endif
#else
#define TENVAD_API
#endif

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *ten_vad_handle_t;

TENVAD_API int ten_vad_create(ten_vad_handle_t *handle, size_t hop_size,
                              float threshold);

TENVAD_API int ten_vad_process(ten_vad_handle_t handle,
                               const int16_t *audio_data,
                               size_t audio_data_length,
                               float *out_probability, int *out_flag);

TENVAD_API int ten_vad_destroy(ten_vad_handle_t *handle);

TENVAD_API const char *ten_vad_get_version(void);

#ifdef __cplusplus
}
#endif

#endif
