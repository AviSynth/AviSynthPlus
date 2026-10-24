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
                else()
                    # set(DEFINE_AVX512 TRUE) # perhaps compiles but not recommended
                    message(STATUS "Disabling AVX-512: MSVC version ${CMAKE_CXX_COMPILER_VERSION} is too old (MSVC 2019 16.2 (19.22) or newer).")
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
    list(FILTER AvsCore_Sources EXCLUDE REGEX ".*_avx512\\.(cpp|hpp|h)$")
    list(FILTER AvsCore_Sources EXCLUDE REGEX ".*_avx512b\\.(cpp|hpp|h)$")
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
  elseif("${FILE}" MATCHES ".*\\.(h|hpp)$")
    set(GROUP "Header Files\\${GROUP}")
  endif()

  source_group("${GROUP}" FILES "${FILE}")
endforeach()

function(handle_arch_flags ARCH_SUFFIX GCC_FLAGS MSVC_FLAGS)
    string(TOLOWER "${ARCH_SUFFIX}" ARCH_SUFFIX_LOWER)
    file(GLOB_RECURSE SRCS_${ARCH_SUFFIX} "*_${ARCH_SUFFIX_LOWER}.cpp")
    if(SRCS_${ARCH_SUFFIX})
        # Check for any MSVC-based toolchain (VS IDE or cl.exe with Ninja) on Windows.
        if (MSVC) # MSVC can do Win32 and ARM64 as well.
            # not this: if (MSVC_IDE), Ninja leaves MSVC_IDE off
            # Check for MSVC-frontends that use GCC/Clang-like flags (e.g., ClangCL, IntelLLVM)
            IF(CLANG_IN_VS STREQUAL "1" OR IntelLLVM_IN_VS STREQUAL "1" OR IntelClassic_IN_VS STREQUAL "1")
                set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
            ELSE()
                # Native MSVC (cl.exe, including when used with Ninja)
                IF(WIN32)
                # MSVC gives warning in x64 when SSE2 flag is added, it's just required as a minimum
                if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8 AND "${MSVC_FLAGS}" STREQUAL " /arch:SSE2 "))
                    set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${MSVC_FLAGS}")
                endif()
                endif()
            ENDIF()
        else()
            # Fallback for all non-MSVC compilers (GCC, Clang on Linux/macOS, MinGW)
            set_source_files_properties(${SRCS_${ARCH_SUFFIX}} PROPERTIES COMPILE_FLAGS "${GCC_FLAGS}")
        endif()
        list(APPEND AvsCore_Sources ${SRCS_${ARCH_SUFFIX}})
    endif()
endfunction()

if(ENABLE_NEON_SIMD)
    # =========================================================================
    # AArch64 (ARM64) SIMD Flags
    # =========================================================================
    # See: Arch64 SIMD Compilation and Source Code Rules
    # Files must be located in 'aarch64' subdirectory.
    
    # Baseline NEON (ASIMD)
    # GCC/Clang: Implicit (or -march=armv8-a) MSVC: /arch:armv8.0
    # asm output debug, leave it here pls.: handle_arch_flags(NEON " -march=armv8-a -S -g -fverbose-asm -Wa,-adhln " " /arch:armv8.0 ")
    handle_arch_flags(NEON " -march=armv8-a " " /arch:armv8.0 ")

    # Dot Product (FEAT_DotProd: optional from ARMv8.2-A, compulsory from ARMv8.4-A)
    # armv8.2-a base: GCC 11/12 arm_neon.h define vdotq_u32 for arch=armv8.2-a+dotprod, armv8.1-a+dotprod fails there.
    handle_arch_flags(DP " -march=armv8.2-a+dotprod " " /arch:armv8.2 ") 

    # Int8 Matrix Multiply (FEAT_I8MM / ARMv8.2-A) Compulsory from ARMv8.6-A
    # Use the base arch flag to enable the features, relying on modern toolchain support.
    handle_arch_flags(I8MM " -march=armv8.2-a+i8mm " " /arch:armv8.2 ") 

    # SVE2 (Scalable Vector Extension 2) (FEAT_SVE2 / ARMv8.5-A) Compulsory from ARMv9.0-A
    handle_arch_flags(SVE2 " -march=armv8.5-a+sve2 " " /arch:armv8.5 ") 

    # SVE2.1 (FEAT_SVE2p1 / ARMv9.1-A)
    # Needs GCC 15 or a newer Clang (Clang 22 OK, Clang 15 not).
    # No *_sve2_1.cpp files yet. Once there are, an older compiler fails on the unknown +sve2p1 modifier:
    # check the flag first (check_cxx_compiler_flag), skip the files and guard their dispatch with a define
    # when unsupported, like DEFINE_AVX512 / INTEL_INTRINSICS_AVX512 on x86. (+sve2 and +i8mm: GCC 10+)
    handle_arch_flags(SVE2_1 " -march=armv9.1-a+sve2p1 " " /arch:armv9.1 ")

else()
    # =========================================================================
    # Intel (x86/x64) SIMD Flags
    # =========================================================================  
# gcc/llvm/clang-like flags, MSVC flags
handle_arch_flags(SSSE3 " -mssse3 " " /arch:SSE2 ") # no special SSSE3 option in MSVC
handle_arch_flags(SSE41 " -msse4.1 " " /arch:SSE2 ") # no special SSE4.1 option in MSVC
handle_arch_flags(AVX " -mavx " " /arch:AVX ")
handle_arch_flags(AVX2 " -mavx2 -mfma " " /arch:AVX2 ")

if(DEFINE_AVX512)
        # Similar to AVX2, FMA must be added explicitly for GCC/Clang.
        # We also enable BMI2, as it is standard since AVX2 and useful for 
        # AVX-512 mask operations like _bzhi_u64/u32.
        # The core subsets (F, CD, BW, DQ, VL) comprising the CPUF_AVX512_BASE 
        # flag are present on all AVX-512-capable architectures.
        #
        # Note: Early Xeon (e.g., Skylake-X/Cascadelake) may exhibit severe 
        # thermal throttling even on a single thread. In AviSynth+, AVX512_BASE 
        # must be manually enabled in-script unless CPUF_AVX512_FAST is also detected.
        # Functions in *_avx512b.cpp should be dispatched on CPUF_AVX512_BASE.
        set(AVX512_BASE_GCC_FLAGS
            " -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl "
        )

        # CPUF_AVX512_FAST indicates a modern, low-throttling architecture (ICL/RKL/AVX10.x).
        # This set represents the intersection of Icelake and AVX10.1 features:
    # (F, CD, BW, DQ, VL, VNNI, VBMI, VBMI2, BITALG, VPOPCNTDQ).
        # Functions in *_avx512.cpp should be dispatched on CPUF_AVX512_FAST.
    set(AVX512_FAST_GCC_FLAGS
        #" -march=icelake-client "  # PF test
        #" -march=graniterapids " # PF test avx10.1
            " -mfma -mbmi2 -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "
    )

        # Note: MSVC's /arch:AVX512 enables the basic subset (F, CD, BW, DQ, VL).
        # Unlike GCC/Clang, MSVC does not require individual flags for extensions 
        # like VBMI or VNNI; if the intrinsic is used, it is compiled directly. 
        # GCC/Clang require either file-level flags (set here) or function-level 
        # target attributes.
        # GCC/Clang/Intel compilers will use -mavx512... flags via handle_arch_flags 
        # even when running within the MSVC_IDE.
        handle_arch_flags(AVX512B "${AVX512_BASE_GCC_FLAGS}" " /arch:AVX512 ") # *_avx512b.cpp
        handle_arch_flags(AVX512 "${AVX512_FAST_GCC_FLAGS}" " /arch:AVX512 ") # *_avx512.cpp
endif()
ENDIF()

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
