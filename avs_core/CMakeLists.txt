CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

if (WIN32)
  set(CoreName "AviSynth")
else()
  set(CoreName "avisynth")
endif()

# Create library
project("AvsCore" VERSION "${PROJECT_VERSION}" LANGUAGES CXX)
Include("Files.cmake")
add_library("AvsCore" ${AvsCore_Sources})
set_target_properties("AvsCore" PROPERTIES "OUTPUT_NAME" "${CoreName}")
if (NOT WIN32)
  set_target_properties("AvsCore" PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION "${AVISYNTH_INTERFACE_VERSION}")
endif()
if (MINGW)
  set_target_properties("AvsCore" PROPERTIES PREFIX "")
  set_target_properties("AvsCore" PROPERTIES IMPORT_PREFIX "")
endif()

# Automatically group source files according to directory structure
foreach(FILE ${AvsCore_Sources})
  get_filename_component(PARENT_DIR "${FILE}" PATH)

  string(REGEX REPLACE "(\\./)" "" GROUP "${PARENT_DIR}")
  string(REPLACE "/" "\\" GROUP "${GROUP}")

  # group into "Source Files" and "Header Files"
  if ("${FILE}" MATCHES ".*\\.cpp")
    set(GROUP "Source Files\\${GROUP}")
  elseif("${FILE}" MATCHES ".*\\.h")
    set(GROUP "Header Files\\${GROUP}")
  endif()

  source_group("${GROUP}" FILES "${FILE}")
endforeach()

if (MSVC_IDE)
  IF(CLANG_IN_VS STREQUAL "1")
      # special SSSE3 option for source files with *_ssse3.cpp pattern
      file(GLOB_RECURSE SRCS_SSSE3 "*_ssse3.cpp")
      set_source_files_properties(${SRCS_SSSE3} PROPERTIES COMPILE_FLAGS " -mssse3 ")

      # special SSE4.1 option for source files with *_sse41.cpp pattern
      file(GLOB_RECURSE SRCS_SSE41 "*_sse41.cpp")
      set_source_files_properties(${SRCS_SSE41} PROPERTIES COMPILE_FLAGS " -msse4.1 ")

      # special AVX option for source files with *_avx.cpp pattern
      file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
      set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS " -mavx ")

      # special AVX2 option for source files with *_avx2.cpp pattern
      file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
      set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS " -mavx2 -mfma ")
  ELSE()
      # special AVX option for source files with *_avx.cpp pattern
      file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
      set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS " /arch:AVX ")

      # special AVX2 option for source files with *_avx2.cpp pattern
      file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
      set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS " /arch:AVX2 ")
  ENDIF()
else()
  # special SSSE3 option for source files with *_ssse3.cpp pattern
  file(GLOB_RECURSE SRCS_SSSE3 "*_ssse3.cpp")
  set_source_files_properties(${SRCS_SSSE3} PROPERTIES COMPILE_FLAGS " -mssse3 ")

  # special SSE4.1 option for source files with *_sse41.cpp pattern
  file(GLOB_RECURSE SRCS_SSE41 "*_sse41.cpp")
  set_source_files_properties(${SRCS_SSE41} PROPERTIES COMPILE_FLAGS " -msse4.1 ")

  # special AVX option for source files with *_avx.cpp pattern
  file(GLOB_RECURSE SRCS_AVX "*_avx.cpp")
  set_source_files_properties(${SRCS_AVX} PROPERTIES COMPILE_FLAGS -mavx )

  # special AVX2 option for source files with *_avx2.cpp pattern
  file(GLOB_RECURSE SRCS_AVX2 "*_avx2.cpp")
  set_source_files_properties(${SRCS_AVX2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
endif()

# Specify include directories
target_include_directories("AvsCore" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Specify preprocessor definitions
target_compile_definitions("AvsCore" PRIVATE BUILDING_AVSCORE)

if(NOT ${BUILD_SHARED_LIBS})
    target_compile_definitions("AvsCore" PRIVATE AVS_STATIC_LIB)
endif()

# If checked out with compat filesystem submodule, add that to system include directories
get_filename_component(
    GHS_FILESYSTEM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../filesystem/include
    ABSOLUTE
)
if (EXISTS ${GHS_FILESYSTEM_INCLUDE_DIR})
    target_include_directories("AvsCore" SYSTEM PRIVATE ${GHS_FILESYSTEM_INCLUDE_DIR})
endif()

# stdc++fs was mainlined into stdc++ in GCC 9, but GCC 8 can build it too
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set(FSLIB "stdc++fs")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Haiku")
    set(SYSLIB "root")
    set(SYSLIBS "-lroot")
elseif (CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
    set(SYSLIB "pthread")
    set(SYSLIBS "-lpthread")
elseif (MINGW)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)

  set(SYSLIB "Threads::Threads" )
  set(SYSLIBS "-lpthread" )
else()
    set(SYSLIB "pthread" "dl")
    set(SYSLIBS "-lpthread -ldl")
endif()

# Per-OS link dependencies
if (MSVC)
  target_link_libraries("AvsCore" "uuid" "winmm" "vfw32" "msacm32" "gdi32" "user32" "advapi32" "ole32" "imagehlp")
elseif (MINGW)
  target_link_libraries("AvsCore" "uuid" "winmm" "msacm32" "gdi32" "ole32" "vfw32" "imagehlp" "${SYSLIB}" "${FSLIB}" -static)
else()
  target_link_libraries("AvsCore" "${SYSLIB}" "${FSLIB}")
endif()


if(ENABLE_CUDA)
  if(${CMAKE_VERSION} VERSION_LESS "3.19.5")
    # deprecated since 3.10 but works
    find_package(CUDA)
    if(CUDA_FOUND)
      target_include_directories("AvsCore" PRIVATE ${CUDA_INCLUDE_DIRS})
      #target_link_libraries ("AvsCore" ${CUDA_LIBRARIES})
      target_link_libraries ("AvsCore" ${CUDA_cudart_static_LIBRARY})
      target_compile_definitions("AvsCore" PUBLIC ENABLE_CUDA)
    endif()
  else()
    # Fixed in 3.19.5: https://gitlab.kitware.com/cmake/cmake/-/issues/21740
    include(FindCUDAToolkit)
    # FIXME: check supported compiler/platform/CUDA SDK version combinations
    if(CUDAToolkit_FOUND)
      target_include_directories("AvsCore" PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_link_libraries ("AvsCore" CUDA::cudart_static)
      target_compile_definitions("AvsCore" PUBLIC ENABLE_CUDA)
    endif()
  endif()
endif()

if (MSVC_IDE)
  # Copy output to a common folder for easy deployment
  add_custom_command(
    TARGET AvsCore
    POST_BUILD
    COMMAND xcopy /Y \"$(TargetPath)\" \"${CMAKE_BINARY_DIR}/Output\"
  )
  IF(NOT CLANG_IN_VS STREQUAL "1")
    # LLVM does not generate exp or don't know how to do it
    add_custom_command(
      TARGET AvsCore
      POST_BUILD
      COMMAND xcopy /Y \"$(TargetDir)AviSynth.exp\" \"${CMAKE_BINARY_DIR}/Output/c_api\"
    )
  ENDIF()
  add_custom_command(
    TARGET AvsCore
    POST_BUILD
    COMMAND xcopy /Y \"$(TargetDir)AviSynth.lib\" \"${CMAKE_BINARY_DIR}/Output/c_api\"
  )
endif()

# Determine target architecture
include("${CMAKE_CURRENT_LIST_DIR}/TargetArch.cmake")
target_architecture(AVS_ARCH)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/core/arch.h.in ${CMAKE_CURRENT_BINARY_DIR}/arch.h @ONLY)

# Dynamically generate the sequential version info from Git
# Based on the example here: http://www.cmake.org/pipermail/cmake/2010-July/038015.html
FIND_PACKAGE(Git)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
ADD_CUSTOM_TARGET(
    VersionGen
    ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/core/version.h.in
                     -D DST=${CMAKE_CURRENT_BINARY_DIR}/version.h
                     -D GIT=${GIT_EXECUTABLE}
                     -D REPO=${CMAKE_SOURCE_DIR}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/Version.cmake
)
ADD_DEPENDENCIES("AvsCore" VersionGen)

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRELEASE_TARBALL")
endif()

# Generate pkg-config file
get_target_property(LIB_NAME AvsCore OUTPUT_NAME)
set(LIBS "-l${LIB_NAME}")
CONFIGURE_FILE("avisynth.pc.in" "avisynth.pc" @ONLY)

# Generate avisynth.conf
CONFIGURE_FILE("avisynth_conf.h.in" "avisynth_conf.h" @ONLY)

INSTALL(TARGETS AvsCore
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

INSTALL(DIRECTORY "include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/avisynth")

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/avisynth.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
