# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project(tests)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)

set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend")
message("RTNEURAL_XSIMD in ${CMAKE_PROJECT_NAME} = ${RTNEURAL_XSIMD}")

# add external libraries
add_subdirectory(../modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)

# configure executable
add_executable(test-rtneural
    src/test_rtneural.cpp
)

# include and link directories
include_directories(test-rtneural ./src ../modules/RTNeural ../modules/RTNeural/modules/json)
link_directories(test-rtneural ./src ../modules/RTNeural ../modules/RTNeural/modules/json)

# configure target
target_link_libraries(test-rtneural RTNeural)
target_compile_definitions(test-rtneural PUBLIC)
