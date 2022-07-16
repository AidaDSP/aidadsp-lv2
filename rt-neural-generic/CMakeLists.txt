# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project (rt-neural-generic)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)

set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend")
message("RTNEURAL_XSIMD in ${CMAKE_PROJECT_NAME} = ${RTNEURAL_XSIMD}")
set(RTNEURAL_LSTM_MODEL_HIDDEN_SIZE "12" CACHE STRING "Set hidden size in RTNeuralLSTM class")
message("RTNEURAL_LSTM_MODEL_HIDDEN_SIZE in ${CMAKE_PROJECT_NAME} = ${RTNEURAL_LSTM_MODEL_HIDDEN_SIZE}")

# add external libraries
add_subdirectory(../modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)

# check for lv2 using pkgconfig
find_package(PkgConfig)
pkg_check_modules(LV2 REQUIRED lv2>=1.10.0)

# include and link directories
include_directories (${LV2_INCLUDE_DIRS} ./src ../modules/RTNeural ../modules/RTNeural/modules/json)
link_directories (${LV2_LIBRARY_DIRS} ./src ../modules/RTNeural ../modules/RTNeural/modules/json)

# config libraries
add_library(rt-neural-generic SHARED
  src/rt-neural-generic.cpp
  src/RTNeuralLSTM.cpp
)

# configure target
set_target_properties(rt-neural-generic PROPERTIES PREFIX "")
set_target_properties(rt-neural-generic PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(rt-neural-generic ${LV2_LIBRARIES} RTNeural)
target_compile_definitions(rt-neural-generic PUBLIC RTNEURAL_LSTM_MODEL_HIDDEN_SIZE=${RTNEURAL_LSTM_MODEL_HIDDEN_SIZE})

# cxxflags, release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wl,--strip-all")

# linker flags, release
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -Wl,--strip-all")

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

install (FILES ../models/48kHz/RNN3-LSTM-12/ArchetypePliniElectricSunriseRiff.json
  DESTINATION ${LV2_INSTALL_DIR} RENAME lstm-model.json
)
