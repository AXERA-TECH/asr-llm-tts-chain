cmake_minimum_required(VERSION 3.16)

project(SimpleCV
  VERSION 0.1.0
  DESCRIPTION "A tiny OpenCV-like wrapper powered by stb"
  LANGUAGES CXX)

option(SIMPLECV_BUILD_TESTS "Build SimpleCV tests" ON)
option(SIMPLECV_WARNINGS_AS_ERRORS "Treat warnings as errors when building SimpleCV" OFF)

add_library(simplecv STATIC
  src/SimpleCV_Codec.cpp
  src/SimpleCV_Proc.cpp
  src/SimpleCV_Draw.cpp
  src/SimpleCV_Text.cpp
  src/SimpleCV_Utils.cpp
  src/SimpleCV_Kalman.cpp
  src/SimpleCV_Warp.cpp
)

add_library(SimpleCV::simplecv ALIAS simplecv)

target_include_directories(simplecv
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(simplecv PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(simplecv PRIVATE /W4)
  if(SIMPLECV_WARNINGS_AS_ERRORS)
    target_compile_options(simplecv PRIVATE /WX)
  endif()
else()
  target_compile_options(simplecv PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
  if(SIMPLECV_WARNINGS_AS_ERRORS)
    target_compile_options(simplecv PRIVATE -Werror)
  endif()
  # 第三方 stb 头里有 by-design 的整型 wrap / shift，UBSan 会误报。
  # 关掉这两个 TU 的 UBSan（在 sanitizer 关闭时此 flag 无副作用）。
  set_source_files_properties(
    src/SimpleCV_Codec.cpp
    src/SimpleCV_Proc.cpp
    PROPERTIES COMPILE_OPTIONS "-fno-sanitize=undefined"
  )
endif()

include(GNUInstallDirs)

install(TARGETS simplecv
  EXPORT SimpleCVTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SimpleCVTargets
  NAMESPACE SimpleCV::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SimpleCV)

if(SIMPLECV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
