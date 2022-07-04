CMAKE_MINIMUM_REQUIRED( VERSION 3.6.2 )

# Find all the input files
FILE(GLOB SoundTouch_Sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cpp"
  "*.h"
)

if(NOT WIN32)
# or if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") ?
# aginst relocation R_X86_64_PC32 against symbol this-and-that
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
 set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# Create library project
add_library("SoundTouch" STATIC ${SoundTouch_Sources})
