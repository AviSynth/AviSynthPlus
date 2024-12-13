CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

set(PluginName "ImageSeq")
set(ProjectName "Plugin${PluginName}")

if (NOT WIN32)
  string(TOLOWER "${PluginName}" PluginName)
endif()

# Create library
project(${ProjectName} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
list (APPEND SourceFiles
    "ImageSeq.cpp"
    "ImageSeq.h"
    "ImageReader.cpp"
    "ImageWriter.cpp"
)
add_library(${ProjectName} SHARED ${SourceFiles})
set_target_properties(${ProjectName} PROPERTIES "OUTPUT_NAME" ${PluginName})
if (MINGW)
  set_target_properties(${ProjectName} PROPERTIES PREFIX "")
  set_target_properties(${ProjectName} PROPERTIES IMPORT_PREFIX "")
endif()

# If checked out with compat filesystem submodule, add that to system include directories
get_filename_component(
    GHS_FILESYSTEM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../../filesystem/include
    ABSOLUTE
)
if (EXISTS ${GHS_FILESYSTEM_INCLUDE_DIR})
    target_include_directories(${ProjectName} SYSTEM PRIVATE ${GHS_FILESYSTEM_INCLUDE_DIR})
endif()

# stdc++fs was mainlined into stdc++ in GCC 9, but GCC 8 can build it too
if (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    set(FSLIB "stdc++fs")
endif()

  # linux: sudo apt-get install libdevil-dev
  # fixme: put it to dependancy?
include(FindDevIL) # official CMake module

  # This module locates the developer's image library. http://openil.sourceforge.net/
  # IL_LIBRARIES -   the name of the IL library. These include the full path to
  #                  the core DevIL library. This one has to be linked into the
  #                  application.
  # IL_INCLUDE_DIR - where to find the il.h, ilu.h and ilut.h files.
  # DevIL_FOUND    - this is set to TRUE if all the above variables were set.
  #                  This will be set to false if ILU or ILUT are not found,
  #                  even if they are not needed. In most systems, if one
  #                  library is found all the others are as well. That's the
  #                  way the DevIL developers release it.
if(DevIL_FOUND)
    message("Found DevIL library: ${IL_LIBRARIES}") 
    target_link_libraries(${ProjectName} "${IL_LIBRARIES}" "${FSLIB}")
    target_include_directories(${ProjectName} PRIVATE "${IL_INCLUDE_DIR}" ${AvsCore_SOURCE_DIR})
else()
    target_link_libraries(${ProjectName} "${FSLIB}")
endif()

if (MSVC_IDE)
  # Copy output to a common folder for easy deployment
  add_custom_command(
    TARGET ${ProjectName}
    POST_BUILD
    COMMAND xcopy /Y \"$(TargetPath)\" \"${CMAKE_BINARY_DIR}/Output/plugins\"
  )
endif()

INSTALL(TARGETS "${ProjectName}"
        DESTINATION "${CORE_PLUGIN_INSTALL_PATH}/avisynth")
