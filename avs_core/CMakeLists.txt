CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2...3.11 )

if (WIN32)
  if(BUILD_SHARED_LIBS)
    set(CoreName "AviSynth")
  else()
    set(CoreName "avisynth")
  endif()
else()
  set(CoreName "avisynth")
endif()

# Create library
project("AvsCore" VERSION "${PROJECT_VERSION}" LANGUAGES CXX)
Include("Files.cmake")

# Initialize a variable to control INTEL_INTRINSICS_AVX512 definition
# AVX512 support is only available on Intel architectures and 64-bit builds and specific compilers
set(DEFINE_AVX512 FALSE)
if(ENABLE_INTEL_SIMD)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit build
        if(MSVC_IDE AND 
           NOT CLANG_IN_VS AND 
           NOT IntelLLVM_IN_VS AND 
           NOT IntelClassic_IN_VS) # Native MSVC, need to check version
            string(REGEX MATCH "^19\\.([0-9]+)\\.([0-9]+)" MSVC_VERSION_MATCH "${CMAKE_CXX_COMPILER_VERSION}")
            if(MSVC_VERSION_MATCH)
                set(MSVC_MAJOR 19)
                set(MSVC_MINOR ${CMAKE_MATCH_1})
                set(MSVC_PATCH ${CMAKE_MATCH_2})

                if(MSVC_MAJOR GREATER_EQUAL 19 AND MSVC_MINOR GREATER_EQUAL 22) # MSVC 2019 16.2 (19.22) or newer
                    set(DEFINE_AVX512 TRUE)
                endif()
            else()
                message(WARNING "Could not parse native 64-bit MSVC version. AVX-512 support might be incomplete.")
            endif()
        else() # Not native old MSVC (likely GCC, Clang, or newer MSVC via Clang/IntelLLVM)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                # GCC 9.0 is a reasonable safe minimum for VNNI/VBMI/VBMI2/BITALG
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
                    set(DEFINE_AVX512 TRUE)
                else()
                    message(STATUS "Disabling AVX-512: GCC version ${CMAKE_CXX_COMPILER_VERSION} is too old (requires 9.0+).")
                endif()
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # Clang 9.0 is a reasonable safe minimum for VNNI/VBMI/VBMI2/BITALG
                if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
                    set(DEFINE_AVX512 TRUE)
                else()
                    message(STATUS "Disabling AVX-512: Clang version ${CMAKE_CXX_COMPILER_VERSION} is too old (requires 9.0+).")
                endif()
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
                # Assume modern Intel compilers (icc, icx) support it
                set(DEFINE_AVX512 TRUE)
            else()
                # Default case for unknown/other compilers (like Clang/Intel in VS)
                # Since we exclude native MSVC, we assume modern compiler support.
                set(DEFINE_AVX512 TRUE)
            endif()
       endif()
    endif()

    # Note: AVX512 related source must be guarded with #ifdef INTEL_INTRINSICS_AVX512 within the
    # already existing #ifdef INTEL_INTRINSICS
    # Though INTEL_INTRINSICS conditional directive can exist for 32 bit and old compiler versions,
    # but INTEL_INTRINSICS_AVX512 exists only on x64 platforms for compilers supporting AVX512 features required by Avisynth
endif()

if(NOT DEFINE_AVX512)
    # Remove AVX512 specific files if it's a 32-bit build
    # or not supported compiler version
    list(FILTER AvsCore_Sources EXCLUDE REGEX ".*_avx512\\.(cpp|h)$")
endif()

add_library("AvsCore" ${AvsCore_Sources})

if(DEFINE_AVX512)
    # dont' use add_definitions(-DINTEL_INTRINSICS_AVX512)
    target_compile_definitions("AvsCore" PRIVATE INTEL_INTRINSICS_AVX512)
endif()

set_target_properties("AvsCore" PROPERTIES "OUTPUT_NAME" "${CoreName}")
if (NOT WIN32)
  set_target_properties("AvsCore" PROPERTIES VERSION "${PROJECT_VERSION}" SOVERSION "${AVISYNTH_INTERFACE_VERSION}")
endif()
if (MINGW)
  if(BUILD_SHARED_LIBS)
    set_target_properties("AvsCore" PROPERTIES PREFIX "")
    set_target_properties("AvsCore" PROPERTIES IMPORT_PREFIX "")
  endif()
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

function(handle_arch_flags ARCH_SUFFIX GCC_FLAGS MSVC_FLAGS)
    string(TOLOWER "${ARCH_SUFFIX}" ARCH_SUFFIX_LOWER)
    file(GLOB_RECURSE SRCS_${ARCH_SUFFIX} "*_${ARCH_SUFFIX_LOWER}.cpp")
    if(SRCS_${ARCH_SUFFIX})
        if (MSVC_IDE)
            IF(CLANG_IN_VS STREQUAL "1" OR IntelLLVM_IN_VS STREQUAL "1" OR IntelClassic_IN_VS STREQUAL "1")
                set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
            ELSE()
                # MSVC gives warning in x64 when SSE2 flag is added, it's just required as a minimum
                if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8 AND "${MSVC_FLAGS}" STREQUAL " /arch:SSE2 "))
                    set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${MSVC_FLAGS}")
                endif()
            ENDIF()
        else()
            set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
        endif()
        list(APPEND AvsCore_Sources ${SRCS_${ARCH_SUFFIX}})
    endif()
endfunction()

# gcc/llvm/clang-like flags, MSVC flags
handle_arch_flags(SSSE3 " -mssse3 " " /arch:SSE2 ") # no special SSSE3 option in MSVC
handle_arch_flags(SSE41 " -msse4.1 " " /arch:SSE2 ") # no special SSE4.1 option in MSVC
handle_arch_flags(AVX " -mavx " " /arch:AVX ")
handle_arch_flags(AVX2 " -mavx2 -mfma " " /arch:AVX2 ")

if(DEFINE_AVX512)
    # The CPUF_AVX512_FAST feature flag in Avisynth guarantees a modern, low-throttling architecture (ICL/RKL/AVX10.2).
    # This feature set is the minimal intersection of ICL and AVX10.1's guaranteed features 
    # (F, CD, BW, DQ, VL, VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ).
    # Function used from *_avx512.cpp files should be dispatched on CPUF_AVX512_FAST to match these compile flags.
    set(AVX512_FAST_GCC_FLAGS
        #" -march=icelake-client "  # PF test
        #" -march=graniterapids " # PF test avx10.1
        " -mfma -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "
    )
    # Note: MSVC's /arch:AVX512 enables the basic F, CD, ... and does not have individual flags for VBMI, VBMI2, etc.
    # GCC/Clang/Intel compilers use the -mavx512... flags inside handle_arch_flags when running in MSVC_IDE
    handle_arch_flags(AVX512 "${AVX512_FAST_GCC_FLAGS}" " /arch:AVX512 ")
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

if (CMAKE_SYSTEM_NAME STREQUAL "Haiku")
    set(SYSLIB "root")
elseif(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
    set(SYSLIB "pthread")
elseif(MSVC OR MINGW)
    set(SYSLIB "uuid" "winmm" "vfw32" "msacm32" "gdi32" "user32" "advapi32" "ole32" "imagehlp")
else()
    set(SYSLIB "pthread" "dl" "m")
endif()

if(MINGW)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    list(APPEND SYSLIB "pthread")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND SYSLIB "stdc++")
    # stdc++fs was mainlined into stdc++ in GCC 9, but GCC 8 can build it too
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
        list(APPEND SYSLIB "stdc++fs")
    endif()
elseif(NOT MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND SYSLIB "c++")
endif()

target_link_libraries("AvsCore" ${SYSLIB})

if(ENABLE_CUDA)
  if(${CMAKE_VERSION} VERSION_LESS "3.19.5")
    # deprecated since 3.10 but works
    find_package(CUDA)
    if(CUDA_FOUND)
      target_include_directories("AvsCore" PRIVATE ${CUDA_INCLUDE_DIRS})
      #target_link_libraries ("AvsCore" ${CUDA_LIBRARIES})
      target_link_libraries ("AvsCore" ${CUDA_cudart_static_LIBRARY})
      target_compile_definitions("AvsCore" PUBLIC ENABLE_CUDA)
      list(APPEND SYSLIB "cudart_static")
    endif()
  else()
    # Fixed in 3.19.5: https://gitlab.kitware.com/cmake/cmake/-/issues/21740
    include(FindCUDAToolkit)
    # FIXME: check supported compiler/platform/CUDA SDK version combinations
    if(CUDAToolkit_FOUND)
      target_include_directories("AvsCore" PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_link_libraries ("AvsCore" CUDA::cudart_static)
      target_compile_definitions("AvsCore" PUBLIC ENABLE_CUDA)
      list(APPEND SYSLIB "cudart_static")
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
  IF(NOT CLANG_IN_VS STREQUAL "1" AND
     NOT IntelLLVM_IN_VS STREQUAL "1" AND 
     NOT IntelClassic_IN_VS STREQUAL "1")
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
  add_custom_command(
    TARGET AvsCore
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/Output/include
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

IF(MSVC_IDE)
  add_custom_command(
    TARGET AvsCore
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/avs_core/version.h ${CMAKE_BINARY_DIR}/avs_core/arch.h ${CMAKE_BINARY_DIR}/Output/include/avs
  )
endif()

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/.git" OR NOT GIT_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRELEASE_TARBALL")
endif()

# Generate pkg-config file
get_target_property(LIB_NAME AvsCore OUTPUT_NAME)
set(LIBS "-l${LIB_NAME}")
set(_SYSLIBS_ITEMS ${SYSLIB})
list(TRANSFORM _SYSLIBS_ITEMS PREPEND "-l")
list(JOIN _SYSLIBS_ITEMS " " SYSLIBS)

if(MINGW)
set(SYSLIBS "${SYSLIBS} -static")
endif()

CONFIGURE_FILE("avisynth.pc.in" "avisynth.pc" @ONLY)

# Generate avisynth.conf
CONFIGURE_FILE("avisynth_conf.h.in" "avisynth_conf.h" @ONLY)

INSTALL(TARGETS AvsCore
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

INSTALL(DIRECTORY "include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/avisynth")

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
              "${CMAKE_CURRENT_BINARY_DIR}/arch.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/avisynth/avs")

INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/avisynth.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
