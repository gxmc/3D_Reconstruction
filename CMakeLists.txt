cmake_minimum_required(VERSION 3.5)
project(Reconstruction)

find_package(Eigen3 REQUIRED)
include_directories(EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3/" )

find_package(OpenCV REQUIRED)

include_directories(${YOUR_DIRECTORY} "/home/user/CLionProjects/Reconstruction")
set(SOURCE_FILES main.cpp image_processing.cpp colmap.cpp openmvs.cpp simplify_mesh.cpp)
add_executable(Reconstruction ${SOURCE_FILES} ${HEADER_FILES})

set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")

find_package(OpenMVS REQUIRED)
find_library(/usr/local/lib/OpenMVS/ libMVS.a)
include_directories(/home/user/Downloads/openMVS-master/libs/)

target_link_libraries( Reconstruction ${OpenCV_LIBS} -lstdc++fs MVS)