# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project (tests)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)

set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend")
message("RTNEURAL_XSIMD in ${CMAKE_PROJECT_NAME} = ${RTNEURAL_XSIMD}")

# add external libraries
add_subdirectory(../modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)

# include and link directories
include_directories (./src ../modules/RTNeural ../modules/RTNeural/modules/json)
link_directories (./src ../modules/RTNeural ../modules/RTNeural/modules/json)

# config libraries
add_executable(tests
  src/tests.cpp
)

# configure target
set_target_properties(tests PROPERTIES PREFIX "" BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(tests RTNeural)
target_compile_definitions(tests PUBLIC)
