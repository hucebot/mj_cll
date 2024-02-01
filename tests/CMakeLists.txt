include(ExternalProject)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORTS_CXX17)
check_cxx_compiler_flag("-std=c++14" COMPILER_SUPPORTS_CXX14)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wmaybe-uninitialized -Wuninitialized")
elseif(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wmaybe-uninitialized -Wuninitialized")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wmaybe-uninitialized -Wuninitialized")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wmaybe-uninitialized -Wuninitialized")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 nor C++14 support. Please use a different C++ compiler.")
endif()

find_package(GTest)

include_directories(${GTEST_INCLUDE_DIRS} ${FCL_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR} )
enable_testing()

set(MJ_CLL_TESTS testMjCll)

add_definitions(-D_TESTS_FOLDER="${CMAKE_SOURCE_DIR}/tests/")
message(STATUS "D_TESTS_FOLDER is ${CMAKE_SOURCE_DIR}/tests/")

SET(TestLibs mj_cll ${GTEST_BOTH_LIBRARIES} -lpthread)

ADD_EXECUTABLE(testMjCll TestMjCll.cpp)
TARGET_LINK_LIBRARIES(testMjCll ${TestLibs})
add_dependencies(testMjCll mj_cll)
add_test(NAME mj_cll_test COMMAND testMjCll)

ADD_EXECUTABLE(testKangaroo TestKangaroo.cpp)
TARGET_LINK_LIBRARIES(testKangaroo ${TestLibs})
add_dependencies(testKangaroo mj_cll)
add_test(NAME kangaroo_test COMMAND testKangaroo)

