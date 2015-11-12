cmake_minimum_required( VERSION 2.6.0 )

project (pstl)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

execute_process(COMMAND hcc-config  --cxxflags OUTPUT_VARIABLE HCC_COMPILER_FLAGS)
string(STRIP "${HCC_COMPILER_FLAGS}" HCC_COMPILER_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HCC_COMPILER_FLAGS}")

execute_process(COMMAND hcc-config  --ldflags  OUTPUT_VARIABLE HCC_LINKER_FLAGS)
string(STRIP "${HCC_LINKER_FLAGS}" HCC_LINKER_FLAGS)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${HCC_LINKER_FLAGS}")

include_directories(/opt/hsa/include)
link_directories(/opt/hsa/lib)
add_definitions(-DHCC_VERSION_08)

add_executable(reduce reduce.cpp hc_am.cpp)
target_link_libraries(reduce m hsa-runtime64)


