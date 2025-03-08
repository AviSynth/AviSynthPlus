include_directories(${AvsCore_SOURCE_DIR}/include)

if (WIN32)
# These plugins are for Windows only
option(BUILD_DIRECTSHOWSOURCE "Build DirectShowSource plugin" OFF)
option(BUILD_VDUBFILTER "Build VDubFilter plugin" ON)
#option(BUILD_VFAPIFILTER "Build VFAPIFilter plugin" ON)
endif()

include(FindDevIL)
if(DevIL_FOUND)
  set(DEVIL_PLUGIN_STATE ON)
else()
  MESSAGE(WARNING "DevIL library not found. ImageSeq plugin will not be built.")
  set(DEVIL_PLUGIN_STATE OFF)
endif()

find_package(PkgConfig)
if(PkgConfig_FOUND)
  pkg_check_modules(SOUNDTOUCH soundtouch)
endif()

if(SOUNDTOUCH_FOUND)
  set(SOUNDTOUCH_PLUGIN_STATE ON)
else()
  # check to see if SoundTouch's CMake files were installed and whether the library itself exists
  find_package(SoundTouch)
  find_library(SOUNDTOUCH SoundTouch PATH ${SoundTouch_LIBRARY_DIRS})
  if(SoundTouch_FOUND)
    # override the SOUNDTOUCH_LINK_LIBRARIES and SOUNDTOUCH_INCLUDE_DIRS variables
    # that pkg-config would have set, and allow these to be overridden at configure time as well
    set(SOUNDTOUCH_LINK_LIBRARIES ${SOUNDTOUCH} CACHE PATH "Path to SoundTouch library")
    set(SOUNDTOUCH_INCLUDE_DIRS ${CMAKE_PREFIX_PATH}/include/soundtouch CACHE PATH "Path to SoundTouch headers")
    set(SOUNDTOUCH_PLUGIN_STATE ON)
  else()
    MESSAGE(WARNING "SoundTouch library not found. TimeStretch plugin will not be built.")
    set(SOUNDTOUCH_PLUGIN_STATE OFF)
  endif()
endif()

option(BUILD_IMAGESEQ "Build ImageSeq plugin" ${DEVIL_PLUGIN_STATE})
option(BUILD_TIMESTRETCH "Build TimeStretch plugin" ${SOUNDTOUCH_PLUGIN_STATE})
option(BUILD_SHIBATCH "Build Shibatch plugin" ON)
option(BUILD_CONVERTSTACKED "Build ConvertStacked plugin" ON)

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  if(NOT MSVC)
    add_definitions("-DAVSC_WIN32_GCC32")
  endif()
endif()

if (WIN32)
  if(BUILD_DIRECTSHOWSOURCE)
    add_subdirectory("DirectShowSource")
  endif()

if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "AMD64" OR
    "${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "X86")
# obviously depends on VirtualDub being available natively,
# which is only true for X86 and AMD64
  if (BUILD_VDUBFILTER)
    add_subdirectory("VDubFilter")
  endif()
endif()

  #if (BUILD_VFAPIFILTER)
    #add_subdirectory("VFAPIFilter")
  #endif()
endif()

if (BUILD_IMAGESEQ)
add_subdirectory("ImageSeq")
endif()

if (BUILD_TIMESTRETCH)
add_subdirectory("TimeStretch")
endif()

if (BUILD_SHIBATCH)
add_subdirectory("Shibatch")
endif()

if (BUILD_CONVERTSTACKED)
add_subdirectory("ConvertStacked")
endif()
