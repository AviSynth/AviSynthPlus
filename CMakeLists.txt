# Tested generators:
# Command line makefile generator
# "MinGW Makefiles": MSYS2/Mingw32 GCC 8.3, 10.3 build
# IDE project file generators
# "Visual Studio 15 2017" optional platform generator Win32 and x64
# "Visual Studio 16 2019" optional platform generator Win32 and x64. optional toolset "v141_xp"
# "Visual Studio 17 2022" optional platform generator Win32 and x64. optional toolset "v141_xp"

# Intel C++ Compilers (Cmake 3.20 minimum):
# Howto: https://www.intel.com/content/www/us/en/developer/articles/technical/using-oneapi-compilers-with-cmake-in-visual-studio.html
# "Intel(R) oneAPI DPC++ Compiler", two flavours. DPCPP is not compatible.
# Intel® NextGen Compiler (in base kit, LLVM based): TOOLSET = "Intel C++ Compiler 2021", COMPILER EXE NAME = icx.exe
# Intel® Classic Compiler (in extra HPC kit): TOOLSET = "Intel C++ Compiler 19.2", COMPILER EXE NAME = icl.exe
# CMake support files (info from: c:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\cmake\SYCL\)
#   copy c:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\cmake\SYCL\FindIntelDPCPP.cmake to c:\Program Files\CMake\share\cmake-3.20\Modules\
# CMake GUI: 
# - Generator: "Visual Studio 16 2019"
# - Optional toolset to use (-T option): 
#   For LLVM based icx: Intel C++ Compiler 2021
#   For classic 19.2 icl: Intel C++ Compiler 19.2
# - Specify native compilers: browse for the appropriate compiler executable path.
#   icx: C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\icx.exe
#   icl: C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64\icl.exe
# If you have errors like "xilink: : error : Assertion failed (shared/driver/drvutils.c, line 312" then
# as a workaround you must copy clang.exe (by default it is located in C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin)
# to the folder beside xilink (for x64 configuration it is in C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\intel64).
# Successful log looks like:
#   The CXX compiler identification is IntelLLVM 2021.4.0 with MSVC-like command-line
#   Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/icx.exe
# or
#   The CXX compiler identification is Intel 2021.4.0.20210910
#   Check for working CXX compiler: C:/Program Files (x86)/Intel/oneAPI/compiler/2021.4.0/windows/bin/intel64/icl.exe
#
# command line:
#   run "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"   to configure the environment
#   cmake -T "Intel C++ Compiler 2021" -DCMAKE_CXX_COMPILER="icx.exe" ../ 
#   cmake -T "Intel C++ Compiler 19.2" -DCMAKE_CXX_COMPILER="icl.exe" ../ 

# "Visual Studio 16 2019" + LLVM 8.0 (clang) optional platform generator Win32 and x64

CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )
# VS2019: 3.14.1
# Intel 2021: 3.20
# VS2022: 3.21

# Get PROJECT_VERSION property from 'avs_core/core/version.h.in'
file(READ "avs_core/core/version.h.in" versioning)
string(REGEX MATCH "AVS_MAJOR_VER     ([0-9]*)" _ ${versioning})
set(version_major ${CMAKE_MATCH_1})
string(REGEX MATCH "AVS_MINOR_VER     ([0-9]*)" _ ${versioning})
set(version_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "AVS_BUGFIX_VER    ([0-9]*)" _ ${versioning})
set(version_bugfix ${CMAKE_MATCH_1})

# Get AVISYNTH_INTERFACE_VERSION from avs_core/include/avisynth.h
file(READ "avs_core/include/avisynth.h" versioning)
string(REGEX MATCH "AVISYNTH_INTERFACE_VERSION = ([0-9]*)" _ ${versioning})
set(AVISYNTH_INTERFACE_VERSION ${CMAKE_MATCH_1})

option(BUILD_SHARED_LIBS "Build shared libraries instead of static ones." ON)
if(NOT ${BUILD_SHARED_LIBS})
  message(WARNING "You must satisfy the conditions of the GPL license when linking against the AviSynth library.")
endif()

option(HEADERS_ONLY "Install only the Headers" ${INSTALL_ONLY_HEADER})
if(${INSTALL_ONLY_HEADER})
  set(INSTALL_ONLY_HEADER OFF)
endif()

if(NOT HEADERS_ONLY)

  project("AviSynth+" VERSION ${version_major}.${version_minor}.${version_bugfix} LANGUAGES CXX)

  # message("Compiler ID: ${CMAKE_CXX_COMPILER_ID} ") 

  include(GNUInstallDirs)

  # Avoid uselessly linking to unused libraries
  set(CMAKE_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
  set(CMAKE_C_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)
  set(CMAKE_CXX_STANDARD_LIBRARIES "" CACHE STRING "" FORCE)

  # We require C++17 or higher.
if(CMAKE_VERSION VERSION_GREATER 3.7)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS FALSE)
endif()

  # Detect Intel processors and turn Intel SIMD on or off automatically.
  message("-- Detected target processor as: ${CMAKE_SYSTEM_PROCESSOR}")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCHID)
  if( ("${ARCHID}" STREQUAL "x86") OR
      ("${ARCHID}" STREQUAL "x64") OR
      ("${ARCHID}" STREQUAL "i686") OR
      ("${ARCHID}" STREQUAL "amd64") OR
      ("${ARCHID}" STREQUAL "x86_64") )
    set(INTEL_SIMD "ON")
  else()
    set(INTEL_SIMD "OFF")
  endif()

  option(ENABLE_PLUGINS "Build set of default external plugins" ON)
  option(ENABLE_INTEL_SIMD "Enable SIMD intrinsics for Intel processors" "${INTEL_SIMD}")
  set(USER_AVS_PLUGINDIR_LOCATION "$ENV{HOME}/.local/lib/avisynth" CACHE STRING "Override path for user-local plugins (default: $HOME/.local/lib/avisynth)")
  option(ENABLE_CUDA "Enable CUDA support" OFF)

  if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES Debug Release RelWithDebInfo)
    set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "Reset the configurations to what we need" FORCE)
  endif()

  IF( MSVC )  # Check for Visual Studio

    #1910-1919 = VS 15.0 (v141 toolset) Visual Studio 2017
    #1920-1929 = VS 16.0 (v142 toolset) Visual Studio 2019
    #1930-1939 = VS 17.0 (v143 toolset) Visual Studio 2022

    IF( MSVC_VERSION VERSION_LESS 1910 )
      MESSAGE(FATAL_ERROR "Visual C++ 2017 or newer required.")
    ENDIF()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Output/plugins")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Output/system")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/Output/c_api")

    # Needed to properly handle __has_include(<filesystem>) in avs_core/filesystem.h
    # See note in filesystem/README.md
    add_definitions("/Zc:__cplusplus")

    IF(MSVC_IDE)
      message("MSVC_IDE support found, reported CMAKE_GENERATOR_TOOLSET is: ${CMAKE_GENERATOR_TOOLSET}")
      string( TOLOWER "${CMAKE_GENERATOR_TOOLSET}" cmake_gentoolset_lower)

      IF(cmake_gentoolset_lower STREQUAL "intel c++ compiler 2021")
        # IntelLLVM
        set(IntelLLVM_IN_VS "1")
      ELSEIF(cmake_gentoolset_lower STREQUAL "intel c++ compiler 19.2")
        # Intel Classic
        set(IntelClassic_IN_VS "1")
      ELSEIF(cmake_gentoolset_lower MATCHES "intel")
        MESSAGE(FATAL_ERROR "Possibly unknown or unsupported Intel compiler") # version update needed in the above lines
      ELSEIF(cmake_gentoolset_lower STREQUAL "llvm" OR cmake_gentoolset_lower STREQUAL "clangcl")
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")  # hope: always
          message("LLVM toolset was specified via -T. Compiler ID is: ${CMAKE_CXX_COMPILER_ID}; CMAKE_CXX_COMPILER_VERSION is: ${CMAKE_CXX_COMPILER_VERSION}")
          # Clang; 9.0.0
          # These are probably not supported when clang is downloaded as a ready-made binary: CLANG_VERSION_MAJOR CLANG_VERSION_MINOR CLANG_VERSION_STRING
          # string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
          if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0.1 )
            MESSAGE(FATAL_ERROR "Clang 7.0.1 or newer required") # as of November 2021 actually we are using 12.0
          endif()
        endif()
        set(CLANG_IN_VS "1")
      ELSEIF(cmake_gentoolset_lower STREQUAL "v141_clang_c2")
         #1900 is reported
        message("v141_clang_c2 toolset was specified via -T. Reported MSVC_VERSION is: ${MSVC_VERSION}")
        message("May not work, try clangcl or LLVM")
        set(CLANG_IN_VS "1")
        # For LLVM Clang installed separately, specify llvm
        # Since Visual Studio 2019 v16.4, LLVM 9.0 or newer is integrated, for this use Toolset: clangcl
      ENDIF()

      option(WINXP_SUPPORT "Make binaries compatible with Windows XP and Vista" OFF)
      if(WINXP_SUPPORT)
        # We want our project to also run on Windows XP
        # Not for LLVM: Clang stopped XP support in 2016
        # 1900 (VS2015) is not supported but we leave here
        IF(MSVC_VERSION VERSION_LESS 1910 )
          IF(NOT CLANG_IN_VS STREQUAL "1")
            set(CMAKE_GENERATOR_TOOLSET "v140_xp" CACHE STRING "The compiler toolset to use for Visual Studio." FORCE) # VS2015
            # https://connect.microsoft.com/VisualStudio/feedback/details/1789709/visual-c-2015-runtime-broken-on-windows-server-2003-c-11-magic-statics
            message("CMAKE_GENERATOR_TOOLSET is forced to: ${CMAKE_GENERATOR_TOOLSET}")
            add_definitions("/Zc:threadSafeInit-")
          ENDIF()
        ELSE()
          IF(NOT CLANG_IN_VS STREQUAL "1")
            # Setting CMAKE_GENERATOR_TOOLSET here has no effect, only when passed (-T option) or set directly, so we just check it
            IF(CMAKE_GENERATOR_TOOLSET STREQUAL "v141_xp")
              # v141_xp is still available in Visual Studio 2022
              message("CMAKE_GENERATOR_TOOLSET is XP compatible: ${CMAKE_GENERATOR_TOOLSET}, extra XP options added")
              # https://connect.microsoft.com/VisualStudio/feedback/details/1789709/visual-c-2015-runtime-broken-on-windows-server-2003-c-11-magic-statics
              add_definitions("/Zc:threadSafeInit-")
            ELSE()
              message(FATAL_ERROR "For XP you must specify v141_xp toolset with -T option (or 'Optional toolset to use' in CMake GUI)!")
            ENDIF()
          ENDIF()
        ENDIF()
      endif()
    ENDIF()

    IF(CLANG_IN_VS STREQUAL "1")
        #these are unknown
        #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        STRING( REPLACE "/EHsc" "/EHa" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        STRING( REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-inconsistent-missing-override")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
    ELSEIF(IntelLLVM_IN_VS STREQUAL "1")
        # The CXX compiler identification is IntelLLVM 2021.4.0 with MSVC-like command-line
        message("IntelLLVM in VS environment chosen, setting additional flags")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-inconsistent-missing-override")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-inconsistent-missing-override")
        # contrary to MSVC-like commandline interface, these are not set
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHa")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHa")
        # for some reason, this was not set
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    ELSEIF(IntelClassic_IN_VS STREQUAL "1")
        # Intel C++ Compiler 19.2 # The CXX compiler identification is Intel 2021.4.0.20210910
        message("Intel Classic chosen, setting additional flags")
        set(DELETE_THIS "/std:c++17") # if it would co-exist with /Qstd=c++17
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        
        # Workaround as of 19.2: 
        # Statically link with /MT instead of /MD against linker error message
        # "unresolved external symbol wmemcmp referenced in function std::filesystem ... lexically_normal"
        # Linker bug?
        set(CompilerFlags
                CMAKE_CXX_FLAGS
                CMAKE_CXX_FLAGS_DEBUG
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_MINSIZEREL
                CMAKE_CXX_FLAGS_RELWITHDEBINFO
                CMAKE_C_FLAGS
                CMAKE_C_FLAGS_DEBUG
                CMAKE_C_FLAGS_RELEASE
                CMAKE_C_FLAGS_MINSIZEREL
                CMAKE_C_FLAGS_RELWITHDEBINFO
                )
        foreach(CompilerFlag ${CompilerFlags})
            string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
            set(${CompilerFlag} "${${CompilerFlag}}" CACHE STRING "msvc compiler flags" FORCE)
            message("MSVC flags: ${CompilerFlag}:${${CompilerFlag}}")
        endforeach()    
    ELSE()
        # MSVC
        # Enable C++ with SEH exceptions
        # Avoid an obnoxious 'overriding /EHsc with /EHa' warning when
        # using something other than MSBuild
        STRING( REPLACE "/EHsc" "/EHa" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        STRING( REPLACE "/EHsc" "/EHa" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        
        # Behavior is new in Visual Studio 2022:
        # Floating-point contractions (mul+add to fma) aren't generated by default under /fp:precise
        # Enable it manually
        if (NOT (MSVC_VERSION LESS 1930)) # at least VS2022
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:contract")
          set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:contract")
        endif()
    ENDIF()
    # Prevent VC++ from complaining about not using MS-specific functions
    add_definitions("/D _CRT_SECURE_NO_WARNINGS /D _SECURE_SCL=0")

    # Enable CRT heap debugging - only effective in debug builds
    add_definitions("/D _CRTDBG_MAP_ALLOC")

    # if missing, some modules inhibit source containing assembler/simd parts
    add_definitions("/D __SSE2__") # fixme: does it really need anymore?

    # Set additional optimization flags
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Oy /Ot /GS- /Oi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oy /Ot /GS- /Oi")

    # CPU_ARCH can be overridden with the corresponding values when using MSVC:
    # IA32 (disabled),
    # SSE (Pentium III and higher, 1999),
    # SSE2 (Pentium 4 and higher, 2000/2001),
    # AVX (Sandy Bridge and higher, 2011),
    # AVX2 (Haswell and higher, 2013)
    set(MSVC_CPU_ARCH "SSE2" CACHE STRING "Set MSVC architecture optimization level (default: SSE2)")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:${MSVC_CPU_ARCH}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:${MSVC_CPU_ARCH}")

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      # MSVC doesn't allow 64-bit builds to have their /arch set to SSE2 (no-op) or below
      if("${MSVC_CPU_ARCH}" MATCHES "(SSE2)")
        set(DELETE_THIS "/arch:${MSVC_CPU_ARCH}")
        message("MSVC doesn't need x86-64 builds to define /arch:${MSVC_CPU_ARCH}. Setting will be ignored.")
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      elseif("${MSVC_CPU_ARCH}" MATCHES "(IA32|SSE)")
        set(DELETE_THIS "/arch:${MSVC_CPU_ARCH}")
        message("MSVC doesn't allow x86-64 builds to define /arch:${MSVC_CPU_ARCH}. Setting will be ignored.")
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
        STRING( REPLACE "${DELETE_THIS}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      endif()
    endif()

    IF(CLANG_IN_VS STREQUAL "1" OR IntelLLVM_IN_VS STREQUAL "1")
      # suppress other frequent but harmless/unavoidable warnings
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-reorder")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-value")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-value")
      # allow per-function attributes like __attribute__((__target__("sse4.1")))
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-gcc-compat")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-gcc-compat")
    ENDIF()

    # Enable standards-conformance mode for MSVC compilers that support this
    # flag (Visual C++ 2017 and later). Default. DirectShowSource will remove if needed.
    # The headers in the XP-side SDK also have errors if built in conformance mode,
    # so if we're building for XP, don't turn that on.
    if (NOT WINXP_SUPPORT)
      if (NOT (MSVC_VERSION LESS 1910))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /permissive-")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /permissive-")
      endif()
    endif()

    if(ENABLE_INTEL_SIMD)
      add_definitions("/D INTEL_INTRINSICS")
    endif()

  ELSE()
    # not MS Visual Studio IDE

    # CMAKE_CXX_STANDARD doesn't cover the use-case of pre-final C++17 support,
    # but I'd assume most setups with a new enough version of CMake to use
    # CMAKE_CXX_STANDARD 17 would also be running a version of GCC/Clang new enough
    # to not need this.  So this will most likely only ever be used by setups running
    # older versions of CMake; regardless, it shouldn't be necessary to force a
    # CMAKE_VERSION check on this part unless the mere presence of CMAKE_CXX_STANDARD 17
    # ends up causing problems for the older compilers here.
    if( ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU")   AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)) OR
        ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)) )
      set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z" )
    endif()

    if(ENABLE_INTEL_SIMD)
      SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -msse2 -DINTEL_INTRINSICS" )
    endif()

    if(WIN32)
      SET( CMAKE_SHARED_LINKER_FLAGS "-Wl,--enable-stdcall-fixup" )
    elseif(APPLE)
      # macOS uses Clang's linker, doesn't like --no-undefined
      SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,error" )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      # make sure there are no undefined symbols
      SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined" )
    endif()

  ENDIF()

  add_subdirectory("avs_core")
  if(ENABLE_PLUGINS)
    add_subdirectory("plugins")
  endif()

else()

  project(AviSynth-Headers VERSION ${version_major}.${version_minor}.${version_bugfix} LANGUAGES CXX)
  message(STATUS "Install Only Headers: ON")

  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/avs_core/include")
  add_library(AviSynth::Headers ALIAS ${PROJECT_NAME})

  # Determine target architecture
  include("${CMAKE_CURRENT_LIST_DIR}/avs_core/TargetArch.cmake")
  target_architecture(AVS_ARCH)
  CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/avs_core/core/arch.h.in ${CMAKE_CURRENT_BINARY_DIR}/arch.h @ONLY)

  # Dynamically generate the sequential version info from Git
  # Based on the example here: http://www.cmake.org/pipermail/cmake/2010-July/038015.html
  FIND_PACKAGE(Git)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ADD_CUSTOM_TARGET(
      VersionGen
      ${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/avs_core/core/version.h.in
                       -D DST=${CMAKE_CURRENT_BINARY_DIR}/version.h
                       -D GIT=${GIT_EXECUTABLE}
                       -D REPO=${CMAKE_SOURCE_DIR}
                       -P ${CMAKE_CURRENT_SOURCE_DIR}/avs_core/Version.cmake
  )

  include(GNUInstallDirs)

  install(
          FILES ${CMAKE_CURRENT_SOURCE_DIR}/avs_core/include/avisynth.h
                ${CMAKE_CURRENT_SOURCE_DIR}/avs_core/include/avisynth_c.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/avisynth
  )

  install(
          DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/avs_core/include/avs
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/avisynth
  )

  install(
          FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
                "${CMAKE_CURRENT_BINARY_DIR}/arch.h"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/avisynth/avs"
  )


endif()

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
