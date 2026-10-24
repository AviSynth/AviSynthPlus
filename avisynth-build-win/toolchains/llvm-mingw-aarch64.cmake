# ---------------------------------------------------------------------------
#  CMake toolchain: Windows on ARM64 (aarch64) via llvm-mingw
#
#  llvm-mingw (https://github.com/mstorsjo/llvm-mingw) is an x86_64-hosted
#  toolchain that targets aarch64-w64-mingw32 (and x86_64/i686 as well),
#  so no ARM64 hardware and no ARM64 Windows is needed to build for
#  Windows on ARM64.
#
#  The UCRT flavour of llvm-mingw is expected here (the "-ucrt-" build):
#  it links against ucrtbase.dll and the api-ms-win-crt-* forwarders,
#  which are the CRT of Windows 10/11 -- the same CRT the MSVC build
#  uses.  The msvcrt flavour (msvcrt.dll, frozen since VS2005) is not
#  used for the release builds.
#
#  Note on the "gcc" driver: llvm-mingw also ships aarch64-w64-mingw32-g++.exe,
#  but that is the very same clang in its GCC-compatible driver mode -- a
#  separate build with it would produce the same code.  A real GNU GCC for
#  aarch64-w64-mingw32 has to be built from source (in WSL); that build is
#  documented in distrib/docs/.../release_build_automation.rst.
#
#  Usage:
#    cmake -S . -B build -G Ninja ^
#          -DCMAKE_TOOLCHAIN_FILE=.../llvm-mingw-aarch64.cmake ^
#          -DAVS_LLVM_MINGW_ROOT=C:/avsplus_build_deps/llvm-mingw-20261020-ucrt-x86_64
# ---------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(AVS_LLVM_MINGW_ROOT "C:/avsplus_build_deps/llvm-mingw-20261020-ucrt-x86_64" CACHE PATH
    "Root directory of the llvm-mingw installation")

# The toolchain file is re-read in the try_compile projects, where the
# cache variables of the outer project are not visible -- so they have to
# be passed on explicitly.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
     AVS_LLVM_MINGW_ROOT
     CMAKE_POLICY_VERSION_MINIMUM)

# CMake 4.x: the try_compile projects that CMake generates call
#   cmake_minimum_required(VERSION 3)
# and CMake 4.3 rejects a bare major version there
#   "Invalid CMAKE_POLICY_VERSION_MINIMUM value "3""
# CMake itself pre-sets the variable to that bare "3" before the toolchain
# file is read, so it has to be replaced with a full major.minor version
# (and kept that way in the try_compile projects).
if(NOT CMAKE_POLICY_VERSION_MINIMUM MATCHES "^[0-9]+\\.[0-9]+")
  set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING
      "Oldest CMake policy version to assume (CMake 4.x requires major.minor)" FORCE)
endif()

get_filename_component(_avs_llvmmingw_root "${AVS_LLVM_MINGW_ROOT}" ABSOLUTE)
set(_avs_bin "${_avs_llvmmingw_root}/bin")

set(_avs_cc  "${_avs_bin}/aarch64-w64-mingw32-clang.exe")
set(_avs_cxx "${_avs_bin}/aarch64-w64-mingw32-clang++.exe")

foreach(_need "${_avs_cc}" "${_avs_cxx}" "${_avs_bin}/llvm-rc.exe" "${_avs_bin}/llvm-ar.exe")
  if(NOT EXISTS "${_need}")
    message(FATAL_ERROR "llvm-mingw toolchain: missing '${_need}'.\n"
                        "Check AVS_LLVM_MINGW_ROOT (currently '${AVS_LLVM_MINGW_ROOT}').")
  endif()
endforeach()
message(STATUS "llvm-mingw aarch64: using ${_avs_cxx}")

set(CMAKE_C_COMPILER   "${_avs_cc}")
set(CMAKE_CXX_COMPILER "${_avs_cxx}")
set(CMAKE_RC_COMPILER  "${_avs_bin}/llvm-rc.exe")
set(CMAKE_AR           "${_avs_bin}/llvm-ar.exe"     CACHE FILEPATH "")
set(CMAKE_RANLIB       "${_avs_bin}/llvm-ranlib.exe" CACHE FILEPATH "")

# llvm-mingw's drivers know their target already, but setting it explicitly
# keeps CMake's compiler identification happy.
set(CMAKE_C_COMPILER_TARGET   aarch64-w64-mingw32)
set(CMAKE_CXX_COMPILER_TARGET aarch64-w64-mingw32)

# --- flags ------------------------------------------------------------------
#  The produced DLLs import libc++.dll and libunwind.dll (llvm-mingw's C++
#  runtime, not part of Windows): ship them next to AviSynth.dll and in
#  plugins\ (build_release.bat copies them).  The C runtime is the UCRT.
set(AVS_MINGW_COMMON_FLAGS "")

# --- search paths -----------------------------------------------------------
#  llvm-mingw keeps the target libraries under lib/aarch64-w64-mingw32;
#  the driver finds them by itself, so nothing has to be set up here.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_FLAGS_INIT   "${AVS_MINGW_COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${AVS_MINGW_COMMON_FLAGS}")

# COFF has no .so -- DLLs are named foo.dll, and their import libraries
# (the link-time stubs, MSVC calls them .lib) are named foo.dll.a
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dll")
set(CMAKE_IMPORT_LIBRARY_PREFIX "")
set(CMAKE_IMPORT_LIBRARY_SUFFIX ".dll.a")
