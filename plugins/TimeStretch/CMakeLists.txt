CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

set(PluginName "TimeStretch")
set(ProjectName "Plugin${PluginName}")

if (NOT WIN32)
  string(TOLOWER "${PluginName}" PluginName)
endif()

# Create library
project(${ProjectName} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
list (APPEND SourceFiles
    "TimeStretch.cpp"
)
add_library(${ProjectName} SHARED ${SourceFiles})
set_target_properties(${ProjectName} PROPERTIES "OUTPUT_NAME" ${PluginName})
if (MINGW)
  set_target_properties(${ProjectName} PROPERTIES PREFIX "")
  set_target_properties(${ProjectName} PROPERTIES IMPORT_PREFIX "")
endif()

# Library dependencies
target_link_libraries(${ProjectName} ${SOUNDTOUCH_LINK_LIBRARIES})

# Include directories
target_include_directories(${ProjectName} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${AvsCore_SOURCE_DIR} ${SOUNDTOUCH_INCLUDE_DIRS})

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
