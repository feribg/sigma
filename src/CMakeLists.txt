cmake_minimum_required(VERSION 3.15)

set(HEADER_LIST
        "${sigma_SOURCE_DIR}/include/sigma/bs.hpp"
        "${sigma_SOURCE_DIR}/include/sigma/instruments.hpp"
        "${sigma_SOURCE_DIR}/include/sigma/smath.hpp"
        "${sigma_SOURCE_DIR}/include/sigma/utils.hpp"
        "${sigma_SOURCE_DIR}/include/sigma/algo.hpp")

set(SOURCE_LIST
        "${sigma_SOURCE_DIR}/src/lib.cpp" ../include/sigma/utils.hpp)

# Make an automatic library - will be static or dynamic based on user setting
add_library(sigma ${SOURCE_LIST} ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(sigma PUBLIC
        "${sigma_SOURCE_DIR}/include"
        ${xtensor_INCLUDE_DIRS}
        "${gcem_SOURCE_DIR}/include"
        ${nlohmann_json_INCLUDE_DIRS})

# This depends on (header only) boost
target_link_libraries(sigma PRIVATE nlohmann_json xtensor xtensor::optimize xtensor::use_xsimd lapack)

# All users of this library will need at least C++17
target_compile_features(sigma PUBLIC cxx_std_17)
target_compile_options(sigma PRIVATE -march=native)


# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})