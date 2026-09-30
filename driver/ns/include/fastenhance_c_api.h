#ifndef AX_AUDIO_SDK_FASTENHANCE_C_API_H_
#define AX_AUDIO_SDK_FASTENHANCE_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Stable SDK boundary around the validated FastEnhancer implementation.
 * input/output contain 512 normalized float samples in [-1, 1]. */
void* fastenhance_create(const char* model_path, int npu_core);
/* Returns the thread-local error from the most recent create attempt. */
const char* fastenhance_create_last_error(void);
int fastenhance_process(void* handle, const float* input, float* output);
/* Returns a pointer owned by the handle; it remains valid until the next
 * operation on the handle or fastenhance_destroy(). */
const char* fastenhance_last_error(void* handle);
void fastenhance_destroy(void* handle);

#ifdef __cplusplus
}
#endif

#endif
