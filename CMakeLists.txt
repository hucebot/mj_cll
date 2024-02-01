cmake_minimum_required(VERSION 3.0.2)
project(mj_cll)
set(MJ_CLL_VERSION_MAJOR 0)
set(MJ_CLL_VERSION_MINOR 1)
set(MJ_CLL_VERSION_PATCH 0)
set(MJ_CLL_VERSION {MJ_CLL_VERSION_MAJOR}.${MJ_CLL_VERSION_MINOR}.${MJ_CLL_VERSION_PATCH})


LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

find_package(Mujoco REQUIRED)
find_package(GLFW3 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenGL REQUIRED)
if(NOT OPENGL_FOUND)
    message("ERROR: OpenGL not found")
endif(NOT OPENGL_FOUND)
set(GL_LIBRARY GL GLU X11)
set(GLFW_DEPS X11 Xrandr Xinerama Xxf86vm Xcursor GL dl pthread)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")


option(MJ_CLL_COMPILE_TESTS "Compile mj_cll tests" TRUE)

###########
## Build ##
###########

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${MUJOCO_INCLUDE_DIR}
  ${GLFW3_INCLUDE_DIR}
)


# Declare a C++ library
add_library(${PROJECT_NAME} include/mj_cll/ClosedLinkage.hpp src/Utils.cpp src/MujoKin.cpp include/mj_cll/eiQuadProg.hpp)
target_link_libraries(${PROJECT_NAME} ${MUJOCO_LIBRARY} ${GLFW3_LIBRARY})
set_target_properties(${PROJECT_NAME} PROPERTIES
                                      LINKER_LANGUAGE CXX
                                      VERSION ${MJ_CLL_VERSION}
                                      SOVERSION ${MJ_CLL_VERSION_MAJOR})

##############
## Bindings ##
##############
add_subdirectory(bindings/python)

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/${PROJECT_NAME})

#############
## Testing ##
#############

if(MJ_CLL_COMPILE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

