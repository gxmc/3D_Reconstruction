cmake_minimum_required(VERSION 3.5)
project(Reconstruction)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)

#set (CMAKE_C_COMPILER  /usr/bin/clang-3.9)
#set (CMAKE_CXX_COMPILER  /usr/bin/clang++-3.9)
#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# For MVS as shared library. Extra for static MVS lib case.
#find_package(Boost REQUIRED system)

set(SOURCE_FILES main.cpp image_processing.cpp colmap.cpp openmvs.cpp simplify_mesh.cpp)
add_executable(Reconstruction ${SOURCE_FILES} ${HEADER_FILES})

set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")

find_package(OpenMVS REQUIRED)

#link_directories(${OPEN_MVS_LIB_DIR})
#find_library(libMVS PATHS ${OPEN_MVS_LIB_DIR})

target_link_libraries(Reconstruction ${OpenCV_LIBS} ${Boost_LIBRARIES} -lstdc++fs MVS)

# or MVS as static library
#find_package(OpenMVS REQUIRED)
#find_library(/usr/local/lib/OpenMVS/ libMVS.a)
#include_directories(/home/user/Downloads/openMVS-master/libs/)
#target_link_libraries( Reconstruction ${OpenCV_LIBS} -lstdc++fs -fopenmp MVS)