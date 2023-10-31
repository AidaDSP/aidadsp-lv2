# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project(rt-neural-generic)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)

set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend")
message("RTNEURAL_XSIMD in ${CMAKE_PROJECT_NAME} = ${RTNEURAL_XSIMD}")

# add external libraries
add_subdirectory(../modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)

# check for lv2 using pkgconfig
find_package(PkgConfig)
pkg_check_modules(LV2 REQUIRED lv2>=1.10.0)

# configure executable
add_library(rt-neural-generic SHARED
    src/rt-neural-generic.cpp
    ../common/Biquad.cpp
)

# include and link directories
include_directories(rt-neural-generic
    ./src
    ../common
    ${LV2_INCLUDE_DIRS}
    ../modules/RTNeural/modules/json
    ../modules/RTNeural)

link_directories(rt-neural-generic
    ./src
    ../common
    ${LV2_LIBRARY_DIRS}
    ../modules/RTNeural
    ../modules/RTNeural/modules/json)

# configure target
target_compile_definitions(rt-neural-generic PUBLIC
    AIDADSP_COMMERCIAL=0
    AIDADSP_MODEL_LOADER=1
)
target_link_libraries(rt-neural-generic ${LV2_LIBRARIES} RTNeural)
set_target_properties(rt-neural-generic PROPERTIES PREFIX "")

# setup install dir
set(LV2_INSTALL_DIR ${DESTDIR}${PREFIX}/rt-neural-generic.lv2)

# config install
install(TARGETS rt-neural-generic
    DESTINATION ${LV2_INSTALL_DIR}
)

install(DIRECTORY ttl/
    DESTINATION ${LV2_INSTALL_DIR}
)

install(DIRECTORY ../models
    DESTINATION ${LV2_INSTALL_DIR}
)
