cmake_minimum_required(VERSION 3.16)
project(ax_audio_qwen3_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_SKIP_RPATH TRUE)

set(AXLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ax-llm")
set(AX650_SDK_DIR "" CACHE PATH "AX650 board SDK containing include/ and lib/")
if(NOT EXISTS "${AX650_SDK_DIR}/include/ax_engine_api.h")
    message(FATAL_ERROR "Set AX650_SDK_DIR to an AX650 board SDK")
endif()

set(TOKENIZER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZER_WITH_JINJA ON CACHE BOOL "" FORCE)
set(TOKENIZER_WITH_HF_LOADER OFF CACHE BOOL "" FORCE)
set(SIMPLECV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${AXLLM_DIR}/third_party/tokenizer.axera"
                 "${CMAKE_BINARY_DIR}/tokenizer")
add_subdirectory("${AXLLM_DIR}/third_party/SimpleCV"
                 "${CMAKE_BINARY_DIR}/simplecv")

set(AXLLM_RUNTIME_SOURCES
    "${AXLLM_DIR}/src/runner/utils/memory_utils.cpp"
    "${AXLLM_DIR}/src/runner/utils/audio_processor.cpp"
    "${AXLLM_DIR}/src/runner/utils/cqdm.cpp"
    "${AXLLM_DIR}/src/runner/utils/logger.cpp"
    "${AXLLM_DIR}/src/runner/utils/net_utils.cpp"
    "${AXLLM_DIR}/src/runner/LLM.cpp"
    "${AXLLM_DIR}/src/runner/MediaHistory.cpp"
    "${AXLLM_DIR}/src/runner/Qwen3_5Runtime.cpp"
    "${AXLLM_DIR}/src/runner/LLM_embed.cpp"
    "${AXLLM_DIR}/src/runner/LLM_init.cpp"
    "${AXLLM_DIR}/src/runner/LLM_decode.cpp"
    "${AXLLM_DIR}/src/runner/utils/files.cpp"
    "${AXLLM_DIR}/src/runner/utils/image_processor.cpp"
    "${AXLLM_DIR}/src/runner/utils/mrope.cpp"
    "${AXLLM_DIR}/src/runner/vision/vision_module.cpp"
    "${AXLLM_DIR}/src/runner/vision/vision_adapters.cpp"
    "${AXLLM_DIR}/src/runner/vision/audio_encoder.cpp"
    "${AXLLM_DIR}/src/runner/ax_model_runner/ax_model_runner_ax650.cpp")

add_library(axllm_runtime STATIC ${AXLLM_RUNTIME_SOURCES})
target_include_directories(axllm_runtime
    PUBLIC
        "${AXLLM_DIR}/src"
        "${AXLLM_DIR}/src/runner"
        "${AXLLM_DIR}/src/runner/utils"
        "${AXLLM_DIR}/third_party/tokenizer.axera/include"
        "${AXLLM_DIR}/third_party/magic_enum/include"
    PRIVATE "${AX650_SDK_DIR}/include")

set(AX_ENGINE_LIBRARY "${AX650_SDK_DIR}/lib/libax_engine.so")
set(AX_INTERPRETER_LIBRARY "${AX650_SDK_DIR}/lib/libax_interpreter.so")
set(AX_SYS_LIBRARY "${AX650_SDK_DIR}/lib/libax_sys.so")
foreach(AX_LIBRARY IN ITEMS AX_ENGINE_LIBRARY AX_INTERPRETER_LIBRARY AX_SYS_LIBRARY)
    if(NOT EXISTS "${${AX_LIBRARY}}")
        message(FATAL_ERROR "Required AX650 library not found: ${${AX_LIBRARY}}")
    endif()
endforeach()

find_package(Threads REQUIRED)
find_package(OpenMP QUIET)
target_link_libraries(axllm_runtime PUBLIC
    tokenizer SimpleCV::simplecv
    "${AX_ENGINE_LIBRARY}" "${AX_INTERPRETER_LIBRARY}" "${AX_SYS_LIBRARY}"
    Threads::Threads ${CMAKE_DL_LIBS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(axllm_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

add_library(qwen3_runner STATIC src/qwen3_runner.cpp)
target_include_directories(qwen3_runner
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${AX650_SDK_DIR}/include")
target_link_libraries(qwen3_runner PUBLIC axllm_runtime)

add_executable(qwen3-worker src/qwen3_worker.cpp)
target_link_libraries(qwen3-worker PRIVATE qwen3_runner)
