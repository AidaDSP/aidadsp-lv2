# CMake file for LV2 rt-neural-generic plugin

cmake_minimum_required(VERSION 3.15)

project (rt-neural-generic)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)
set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend" FORCE)

# add external libraries
add_subdirectory(../modules/RTNeural ${CMAKE_CURRENT_BINARY_DIR}/RTNeural)
add_subdirectory(../modules/json ${CMAKE_CURRENT_BINARY_DIR}/json)

# check for lv2 using pkgconfig
find_package(PkgConfig)
pkg_check_modules(LV2 REQUIRED lv2>=1.10.0)

# include and link directories
include_directories (${LV2_INCLUDE_DIRS} ./src ../modules/RTNeural ../modules/json)
link_directories (${LV2_LIBRARY_DIRS} ./src ../modules/RTNeural ../modules/json)

# config libraries
add_library(rt-neural-generic SHARED
  src/rt-neural-generic.cpp
  src/RTNeuralLSTM.cpp
)

# configure target
set_target_properties(rt-neural-generic PROPERTIES PREFIX "")
set_target_properties(rt-neural-generic PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(rt-neural-generic ${LV2_LIBRARIES} RTNeural nlohmann_json::nlohmann_json)
target_compile_definitions(rt-neural-generic PUBLIC)

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

install (FILES ../models/ArchetypePliniElectricSunrise.json
  DESTINATION ${LV2_INSTALL_DIR} RENAME lstm-model.json
)
