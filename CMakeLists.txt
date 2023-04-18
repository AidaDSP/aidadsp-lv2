cmake_minimum_required(VERSION 3.15)

project(aidadsp-lv2 VERSION 1.0.0)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

option(GENERIC_AARCH64 "Use specific cxxflags and ldflags for a generic aarch64 machine" OFF)
option(MPB_MOD_DUO_GCC_750 "Use specific cxxflags and ldflags for Mod Duo device using mod-plugin-builder with toolchain gcc 7.5.0" OFF)
option(MPB_MOD_DUO_GCC_750 "Use specific cxxflags and ldflags for Mod Dwarf device using mod-plugin-builder with toolchain gcc 7.5.0" OFF)
option(MPB_MOD_DUOX_GCC_750 "Use specific cxxflags and ldflags for Mod Duo X device using mod-plugin-builder with toolchain gcc 7.5.0" OFF)
option(MPB_RPI4_GCC_750 "Use specific cxxflags and ldflags for Rpi4 device using mod-plugin-builder with toolchain gcc 7.5.0" OFF)

if(GENERIC_AARCH64)
    # @TODO: -funsafe-loop-optimizations causes the plugin to sound really bad, excluding
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprefetch-loop-arrays -funroll-loops -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

if(MPB_MOD_DUO_GCC_750)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a7 -mtune=cortex-a7 -ffast-math -fno-finite-math-only -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

if(MPB_MOD_DWARF_GCC_750)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a35 -mtune=cortex-a35 -ffast-math -fno-finite-math-only -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

if(MPB_MOD_DUOX_GCC_750)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a53 -mtune=cortex-a53 -ffast-math -fno-finite-math-only -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

if(MPB_RPI4_GCC_750)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=cortex-a72 -mtune=cortex-a72 -ffast-math -fno-finite-math-only -fprefetch-loop-arrays -funroll-loops -funsafe-loop-optimizations")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
endif()

message("CMAKE_CXX_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_SHARED_LINKER_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

set(AIDADSP_MODEL "" CACHE STRING "Which commercial plugin model to build")

if (AIDADSP_MODEL)
    add_subdirectory(commercial)
else()
    add_subdirectory(rt-neural-generic)
endif()
