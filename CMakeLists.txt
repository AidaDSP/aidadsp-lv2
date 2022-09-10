cmake_minimum_required(VERSION 3.15)

project(aidadsp-lv2 VERSION 1.0.0)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

option(GENERIC_AARCH64 "Use specific cxxflags and ldflags for a generic aarch64 machine" OFF)
option(MOD_DWARF_GCC_750 "Use specific cxxflags and ldflags for Mod Dwarf device using toolchain gcc 7.5.0" OFF)

if(GENERIC_AARCH64)
    # @TODO: -funsafe-loop-optimizations causes the plugin to sound really bad, excluding
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprefetch-loop-arrays -funroll-loops -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

if(MOD_DWARF_GCC_750)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a35 -mtune=cortex-a35 -ffast-math -fno-finite-math-only -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

message("CMAKE_CXX_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_SHARED_LINKER_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

add_subdirectory(rt-neural-generic)
