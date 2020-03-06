cmake_minimum_required(VERSION 3.15)
project(bench)

set(CMAKE_CXX_STANDARD 17)

add_executable(bench bench.cpp)

target_include_directories(bench PUBLIC ${xtensor_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS}  ${benchmark_INCLUDE_DIRS})
target_link_libraries(bench PUBLIC sigma xtensor nlohmann_json benchmark::benchmark)