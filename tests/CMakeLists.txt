# Tests need to be added as executables first
add_executable(unit_tests bs_test.cpp)

# I'm using C++17 in the test
target_compile_features(unit_tests PRIVATE cxx_std_17)

target_link_libraries(unit_tests PRIVATE gtest_main sigma)

add_test(
        NAME
        unit
        COMMAND
        ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
)
