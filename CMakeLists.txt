cmake_minimum_required(VERSION 3.5)
project(Reconstruction)

find_package( Eigen3 REQUIRED )
include_directories(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3/" )

find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)

include_directories(${YOUR_DIRECTORY} "/home/user/CLionProjects/Reconstruction")
set(HEADER_FILES image_processing.h colmap.cpp colmap.h openmvs.cpp openmvs.h Simplify.h)
set(SOURCE_FILES main.cpp image_processing.cpp utils.h colmap.cpp colmap.h openmvs.cpp openmvs.h Simplify.h)
add_executable(Reconstruction ${SOURCE_FILES} ${HEADER_FILES})

set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -O3")

find_package(OpenMVS REQUIRED)
#include_directories(OpenMVS_INCLUDE_DIR "/usr/local/include/OpenMVS")
find_library(/usr/local/lib/OpenMVS/ libMVS.a)
include_directories(/home/user/Downloads/openMVS-master/libs/)


target_link_libraries( Reconstruction ${OpenCV_LIBS} -lstdc++fs ${OpenCV_LIBS} MVS)