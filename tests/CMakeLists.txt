# CMake file for rt-neural-generic plugin tests

cmake_minimum_required(VERSION 3.15)

project(tests)

# flags and definitions
set(CMAKE_CXX_STANDARD 17)

set(TEST_NAME "" CACHE STRING "Which test to build")

message("Bulding binary test-${TEST_NAME}")

if (NOT TEST_NAME)
    error("TEST_NAME cannot be empty")
else()
    if(TEST_NAME STREQUAL "rtneural")
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
    elseif(TEST_NAME STREQUAL "smoothers")
        # configure executable
        add_executable(test-smoothers
            src/test_smoothers.cpp
        )

        # include and link directories
        include_directories(test-smoothers ./src ../common)
        link_directories(test-smoothers ./src ../common)
    endif()
endif()

