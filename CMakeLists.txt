cmake_minimum_required(VERSION 2.8.3)
project(classification)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
    message(STATUS "No build type selected, default to ${CMAKE_BUILD_TYPE}")
endif()

find_package(Boost COMPONENTS system filesystem regex REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(TinyXML REQUIRED)

find_package( OpenCV REQUIRED )
message("OpenCV version is ${OpenCV_VERSION}")

if(CUDA_FOUND)  ##为什么没有找到CUDA
include_directories(${CUDA_INCLUDE_DIRS}) #"/usr/local/cuda-7.5/include"
endif()

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS} "/usr/local/cuda/include")
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
message("Caffe_DIR is ${Caffe_DIR}")	 #specify Caffe_DIR in /cmake/Modules/findCaffe.cmake
message("CMAKE_SOURCE_DIR is ${CMAKE_SOURCE_DIR}")
message("CMAKE_MODULE_PATH IS ${CMAKE_MODULE_PATH} ")
message("OpenCV_LIBS IS: ${OpenCV_LIBS}")

set(GLOG_LIB glog)

# Add src to include directories.
# include_directories(src/src)

add_executable (classification classification.cpp)  
target_link_libraries(classification ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG_LIB} ${Boost_LIBRARIES})

add_executable (ftSaliencyDetection2 ftSaliencyDetection2.cpp)  
target_link_libraries(ftSaliencyDetection2 ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${GLOG_LIB} ${Boost_LIBRARIES})






