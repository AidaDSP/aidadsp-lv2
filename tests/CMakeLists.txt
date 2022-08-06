# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project(tests)

option(RTNEURAL_ENABLE_AARCH64 "Use specific cxxflags and ldflags for aarch64" OFF)
if(RTNEURAL_ENABLE_AARCH64)
    set(ARCHITECTURE "aarch64")
endif()

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

# add external project
include(ExternalProject)
ExternalProject_Add(lsp-dsp-units-build
    SOURCE_DIR ../../modules/lsp-dsp-units
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/lsp-dsp-units
    CONFIGURE_COMMAND make config ARCHITECTURE=${ARCHITECTURE}
    BUILD_COMMAND make clean && make fetch && make ARCHITECTURE=${ARCHITECTURE} -C <SOURCE_DIR>
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND make DESTDIR=<INSTALL_DIR> install
)

# specify external project target type and location
add_library(lsp-dsp-units STATIC IMPORTED)
set_target_properties(lsp-dsp-units PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lsp-dsp-units/usr/local/lib/liblsp-dsp-units-1.0.3.a)

# configure executable
add_executable(test-lsp
  src/test_lsp.cpp
)

# setup dependencies
add_dependencies(test-lsp lsp-dsp-units-build)

# include and link directories
include_directories(test-lsp
    ./src
    ../modules/lsp-dsp-units/include
    ../modules/lsp-dsp-units/modules/lsp-dsp-lib/include
    ../modules/lsp-dsp-units/modules/lsp-common-lib/include
    ../modules/lsp-dsp-units/modules/lsp-test-fw/include
    ../modules/lsp-dsp-units/modules/lsp-runtime-lib/include
    ../modules/lsp-dsp-units/modules/lsp-lltl-lib/include)

link_directories(test-lsp ./src ${CMAKE_CURRENT_BINARY_DIR}/lsp-dsp-units/usr/local/lib)

find_library(LIBSNDFILE_LIBRARY
    NAMES sndfile libsndfile-1
    PATHS ${LIBSNDFILE_PKGCONF_LIBRARY_DIRS}
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# configure target
target_compile_definitions(test-lsp PUBLIC)
target_link_libraries(test-lsp lsp-dsp-units ${CMAKE_DL_LIBS} Threads::Threads ${LIBSNDFILE_LIBRARY})
