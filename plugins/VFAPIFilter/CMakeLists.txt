CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

set(PluginName "VFAPIFilter")
set(ProjectName "Plugin${PluginName}")

# We need these variables set by the user to compile successfully
set(DSHOWSRC_BASECLASSES_PATH "C:/Program Files/Microsoft SDKs/Windows/v7.1/Samples/multimedia/directshow/baseclasses" CACHE STRING "Folder path to the DirectShow example baseclasses.")
set(DSHOWSRC_BASECLASSES_LIB CACHE FILEPATH "File path to the DirectShow example baseclasses precompiled static library ('strmbase.lib').")
set(DSHOWSRC_DX_INCLUDE_PATH "C:/Program Files/Microsoft DirectX SDK (August 2009)/Include" CACHE STRING "Include folder path to the DirectX headers.")

# Create library
project(${ProjectName} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
list (APPEND SourceFiles
    "VFAPIFilter.cpp"
)
add_library(${ProjectName} SHARED ${SourceFiles})
set_target_properties(${ProjectName} PROPERTIES "OUTPUT_NAME" ${PluginName})
if (MINGW)
  set_target_properties(${ProjectName} PROPERTIES PREFIX "")
  set_target_properties(${ProjectName} PROPERTIES IMPORT_PREFIX "")
endif()

# Library dependencies
target_link_libraries(${ProjectName} "User32.lib")

# Include directories
target_include_directories(${ProjectName} PRIVATE ${AvsCore_SOURCE_DIR} ${DSHOWSRC_BASECLASSES_PATH} ${DSHOWSRC_DX_INCLUDE_PATH})

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
