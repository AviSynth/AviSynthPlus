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
pkg_check_modules(SOUNDTOUCH soundtouch)
if(SOUNDTOUCH_FOUND)
  set(SOUNDTOUCH_PLUGIN_STATE ON)
else()
  MESSAGE(WARNING "SoundTouch library not found. TimeStretch plugin will not be built.")
  set(SOUNDTOUCH_PLUGIN_STATE OFF)
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

  if (BUILD_VDUBFILTER)
    add_subdirectory("VDubFilter")
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
