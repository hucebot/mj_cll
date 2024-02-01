cmake_minimum_required(VERSION 3.5)

if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    message("Please consider to switch to latest version to enable
    more reliable Python3 binaries detection")
else()
    find_package(Python ${PYBIND11_PYTHON_VERSION} COMPONENTS Development Interpreter REQUIRED)
endif()

find_package(pybind11 QUIET)

message(STATUS "check for pybind11")
if(${pybind11_FOUND})

    message(STATUS "compiling python bindings")

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    pybind11_add_module(pymj_cll pyMjCll.cpp)
    target_link_libraries(pymj_cll PUBLIC mj_cll)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    pybind11_add_module(pymujokin pyMujoKin.cpp)
    target_link_libraries(pymujokin PUBLIC mj_cll)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    pybind11_add_module(pyIK pyIK.cpp)
    target_link_libraries(pyIK PUBLIC mj_cll)

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

    install(TARGETS pymj_cll
        COMPONENT python
        LIBRARY DESTINATION "~/.local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

    install(TARGETS pymujokin
        COMPONENT python
        LIBRARY DESTINATION "~/.local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

    install(TARGETS pyIK
        COMPONENT python
        LIBRARY DESTINATION "~/.local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")

else()
    message(STATUS "pybind not found")
endif()
