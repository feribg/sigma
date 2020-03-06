cmake_minimum_required(VERSION 3.15)
include(FetchContent)

project(sigma CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Testing library
FetchContent_Declare(
        googletest
        GIT_REPOSITORY "https://github.com/google/googletest"
        GIT_TAG        master
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
        googlebench
        GIT_REPOSITORY "https://github.com/google/benchmark.git"
        GIT_TAG        master
)
FetchContent_MakeAvailable(googlebench)

FetchContent_Declare(
        gcem
        GIT_REPOSITORY "https://github.com/kthohr/gcem.git"
        GIT_TAG        master
)

FetchContent_GetProperties(gcem)
if(NOT gcem_POPULATED)
    FetchContent_Populate(gcem)
    add_subdirectory(${gcem_SOURCE_DIR} ${gcem_BINARY_DIR})
endif()

set(XTENSOR_USE_XSIMD 1)
#set(XTENSOR_USE_TBB 1)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(benchmark REQUIRED)
find_package(blas REQUIRED)
find_package(lapack REQUIRED)


message("nlohmann_json version: ${nlohmann_json_VERSION}")
message("benchmark version: ${benchmark_VERSION}")
message("xtensor version: ${xtensor_VERSION}")
message("xsimd version: ${xsimd_VERSION}")
message("xtl version: ${xtl_VERSION}")
message("blas version: ${blas_VERSION}")
message("lapack version: ${lapack_VERSION}")

# The executable code is here
add_subdirectory(src)
add_subdirectory(bench)
enable_testing()
add_subdirectory(tests)




