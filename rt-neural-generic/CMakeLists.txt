# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project (rt-neural-generic)

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

# check for lv2 using pkgconfig
find_package(PkgConfig)
pkg_check_modules(LV2 REQUIRED lv2>=1.10.0)

# find dependencies for lsp-dsp-units
find_library(LIBSNDFILE_LIBRARY
    NAMES sndfile libsndfile-1
    PATHS ${LIBSNDFILE_PKGCONF_LIBRARY_DIRS}
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# configure executable
add_library(rt-neural-generic SHARED
  src/rt-neural-generic.cpp
)

# setup dependencies
add_dependencies(rt-neural-generic lsp-dsp-units-build)

# include and link directories
include_directories (rt-neural-generic
    ./src
    ${LV2_INCLUDE_DIRS}
    ../modules/RTNeural/modules/json
    ../modules/RTNeural
    ../modules/lsp-dsp-units/include
    ../modules/lsp-dsp-units/modules/lsp-dsp-lib/include
    ../modules/lsp-dsp-units/modules/lsp-common-lib/include
    ../modules/lsp-dsp-units/modules/lsp-test-fw/include
    ../modules/lsp-dsp-units/modules/lsp-runtime-lib/include
    ../modules/lsp-dsp-units/modules/lsp-lltl-lib/include)

link_directories (rt-neural-generic
    ./src
    ${LV2_LIBRARY_DIRS}
    ../modules/RTNeural
    ../modules/RTNeural/modules/json
    ${CMAKE_CURRENT_BINARY_DIR}/lsp-dsp-units/usr/local/lib)

# configure target
target_compile_definitions(rt-neural-generic PUBLIC)
target_link_libraries(rt-neural-generic ${LV2_LIBRARIES} RTNeural lsp-dsp-units ${CMAKE_DL_LIBS} Threads::Threads ${LIBSNDFILE_LIBRARY})

# setup install dir
set(LV2_INSTALL_DIR ${DESTDIR}${PREFIX}/rt-neural-generic.lv2)

# config install
install(TARGETS rt-neural-generic
  LIBRARY
  DESTINATION ${LV2_INSTALL_DIR}
)

install (DIRECTORY ttl/
  DESTINATION ${LV2_INSTALL_DIR}
)

install (DIRECTORY ../models
  DESTINATION ${LV2_INSTALL_DIR}
)

install (FILES ../models/SIMPLE-RNN/48kHz/LSTM-12/NeuralDSP_Archetype-Plini/ElectricSunriseRiff.json
  DESTINATION ${LV2_INSTALL_DIR} RENAME model.json
)
