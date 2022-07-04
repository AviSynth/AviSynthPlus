CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

set(PluginName "ConvertStacked")
set(ProjectName "Plugin${PluginName}")

if (NOT WIN32)
  string(TOLOWER "${PluginName}" PluginName)
endif()

# Create library
project(${ProjectName} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
list (APPEND SourceFiles
    "ConvertStacked.cpp"
)
add_library(${ProjectName} SHARED ${SourceFiles})
set_target_properties(${ProjectName} PROPERTIES "OUTPUT_NAME" ${PluginName})
if (MINGW)
  set_target_properties(${ProjectName} PROPERTIES PREFIX "")
  set_target_properties(${ProjectName} PROPERTIES IMPORT_PREFIX "")
endif()

# Library dependencies
target_link_libraries(${ProjectName})

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
