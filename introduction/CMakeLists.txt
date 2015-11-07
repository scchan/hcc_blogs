cmake_minimum_required( VERSION 2.6.0 )

project (saxpy)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

execute_process(COMMAND hcc-config  --cxxflags OUTPUT_VARIABLE HCC_COMPILER_FLAGS)
set( COMPILER_FLAGS "${COMPILER_FLAGS} ${HCC_COMPILER_FLAGS} -std=c++11" )

execute_process(COMMAND hcc-config  --ldflags  OUTPUT_VARIABLE HCC_LINKER_FLAGS)
set( LINKER_FLAGS "${LINKER_FLAGS} ${HCC_LINKER_FLAGS}")

add_executable(saxpy saxpy.cpp)
target_link_libraries(saxpy m)

set_target_properties(saxpy PROPERTIES
                      COMPILE_FLAGS ${COMPILER_FLAGS}
                      LINK_FLAGS ${LINKER_FLAGS})

