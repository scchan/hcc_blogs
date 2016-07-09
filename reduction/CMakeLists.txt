cmake_minimum_required( VERSION 2.6.0 )

project (saxpy)
set(CMAKE_CXX_COMPILER hcc)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

execute_process(COMMAND hcc-config  --cxxflags OUTPUT_VARIABLE HCC_COMPILER_FLAGS)
string(STRIP "${HCC_COMPILER_FLAGS}" HCC_COMPILER_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_COMPILER_FLAGS}")

execute_process(COMMAND hcc-config  --ldflags  OUTPUT_VARIABLE HCC_LINKER_FLAGS)
string(STRIP "${HCC_LINKER_FLAGS}" HCC_LINKER_FLAGS)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HCC_LINKER_FLAGS}")

add_executable(reduce_group_mem reduce_group_mem.cpp)

add_executable(reduce_dynamic_group_mem reduce_dynamic_group_mem.cpp)

add_executable(reduce_shuffle reduce_shuffle.cpp)


