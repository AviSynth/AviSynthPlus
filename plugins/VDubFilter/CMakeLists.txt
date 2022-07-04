CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

set(PluginName "VDubFilter")
set(ProjectName "Plugin${PluginName}")

# Create library
project(${ProjectName} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
list (APPEND SourceFiles
    "VDubFilter.cpp"
    "GetCPUFlags.cpp"
    "ScriptInterpreter.h"
    "ScriptValue.h"
    "VBitmap.h"
    "vdplugin.h"
    "vdvideofilt.h"
)
add_library(${ProjectName} SHARED ${SourceFiles})
set_target_properties(${ProjectName} PROPERTIES "OUTPUT_NAME" ${PluginName})
if (MINGW)
  set_target_properties(${ProjectName} PROPERTIES PREFIX "")
  set_target_properties(${ProjectName} PROPERTIES IMPORT_PREFIX "")
endif()

# Library dependencies
target_link_libraries(${ProjectName} "user32")

# Include directories
target_include_directories(${ProjectName} PRIVATE ${AvsCore_SOURCE_DIR})

if (MSVC_IDE)
  # Copy output to a common folder for easy deployment
  add_custom_command(
    TARGET ${ProjectName}
    POST_BUILD
    COMMAND xcopy /Y \"$(TargetPath)\" \"${CMAKE_BINARY_DIR}/Output/plugins\"
  )
endif()

INSTALL(TARGETS "${ProjectName}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")
