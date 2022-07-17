cmake_minimum_required(VERSION 3.15)

project(aidadsp-lv2 VERSION 1.0.0)

option(RTNEURAL_ENABLE_AARCH64 "Use specific cxxflags and ldflags for aarch64" OFF)
if(RTNEURAL_ENABLE_AARCH64)
    # @TODO: -funsafe-loop-optimizations causes the plugin to sound really bad, excluding
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprefetch-loop-arrays -funroll-loops -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -static-libstdc++ -Wl,-Ofast -Wl,--as-needed -Wl,--strip-all")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
endif()

message("CMAKE_CXX_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_SHARED_LINKER_FLAGS_RELEASE in ${CMAKE_PROJECT_NAME} = ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")

add_subdirectory(rt-neural-generic)
