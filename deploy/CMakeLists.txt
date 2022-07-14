cmake_minimum_required(VERSION 2.6)

project(yolop)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
		roscpp
		rospy
		std_msgs
		sensor_msgs
		geometry_msgs
		message_generation
		cv_bridge
		image_transport
		customized_msgs
		)





#find_package(ZED 3 REQUIRED)
#find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)
find_package(CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/opt/TensorRT-8.2.4.2/include)
link_directories(/opt/TensorRT-8.2.4.2/lib)
# zed
#include_directories(/usr/local/zed/include)
#link_directories(/usr/local/zed/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
set(ZED_LIBS ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

catkin_package(
		#  INCLUDE_DIRS include
		#  LIBRARIES yolox_ros_01
		CATKIN_DEPENDS roscpp rospy std_msgs message_runtime eigen3
		#  DEPENDS system_lib
)

include_directories(
		${catkin_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		#        ${CUDA_TOOLKIT_INCLUDE}
)

add_executable(yolop ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(yolop nvinfer)
target_link_libraries(yolop ${ZED_LIBS})
target_link_libraries(yolop cudart)
target_link_libraries(yolop myplugins)
target_link_libraries(yolop ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(main_simulation ${PROJECT_SOURCE_DIR}/main_simulation.cpp)
target_link_libraries(main_simulation nvinfer)
#target_link_libraries(main_simulation ${ZED_LIBS})
target_link_libraries(main_simulation cudart)
target_link_libraries(main_simulation myplugins)
target_link_libraries(main_simulation ${OpenCV_LIBS} ${catkin_LIBRARIES})


add_definitions(-O3 -pthread)

