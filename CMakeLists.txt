cmake_minimum_required(VERSION 3.10)
project(trt_yolo)

# Default to C++17 for ROS 2
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# Find ament_cmake and ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_compression REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)
  include_directories(/workspace/shared/TensorRT-10.2.0.19/include/)
  link_directories(/workspace/shared/TensorRT-10.2.0.19/lib/)
endif()

# Common include directories for all targets (移除了版本特定的路径)
include_directories(
    ${rclcpp_INCLUDE_DIRS}
    ${cv_bridge_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /opt/ros/humble/include
    /opt/ros/humble/include/cv_bridge
)

# Create the plugin libraries with proper include directories
add_library(yolov8plugin SHARED 
    ${PROJECT_SOURCE_DIR}/include/yolov8/plugin/yololayer.cu
)
target_include_directories(yolov8plugin PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolov8/include
    ${PROJECT_SOURCE_DIR}/include/yolov8/plugin
)
target_link_libraries(yolov8plugin nvinfer cudart)

add_library(yolo11plugin SHARED 
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin/yololayer.cu
)
target_include_directories(yolo11plugin PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolo11/include
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin
)
target_link_libraries(yolo11plugin nvinfer cudart)

# Create the main executables
file(GLOB_RECURSE YOLOV8SRCS 
    ${PROJECT_SOURCE_DIR}/src/yolov8/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/yolov8/src/*.cu
)
file(GLOB_RECURSE YOLO11SRCS 
    ${PROJECT_SOURCE_DIR}/src/yolo11/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/yolo11/src/*.cu
)

add_executable(yolov8_det ${PROJECT_SOURCE_DIR}/src/yolov8/yolov8_det.cpp ${YOLOV8SRCS})
add_executable(yolov8_seg ${PROJECT_SOURCE_DIR}/src/yolov8/yolov8_seg.cpp ${YOLOV8SRCS})
add_executable(yolov8_pose ${PROJECT_SOURCE_DIR}/src/yolov8/yolov8_pose.cpp ${YOLOV8SRCS})
add_executable(yolo11_det ${PROJECT_SOURCE_DIR}/src/yolo11/yolo11_det.cpp ${YOLO11SRCS})
add_executable(yolo11_obb ${PROJECT_SOURCE_DIR}/src/yolo11/yolo11_obb.cpp ${YOLO11SRCS})
add_executable(yolo11_pose ${PROJECT_SOURCE_DIR}/src/yolo11/yolo11_pose.cpp ${YOLO11SRCS})
add_executable(yolo11_seg ${PROJECT_SOURCE_DIR}/src/yolo11/yolo11_seg.cpp ${YOLO11SRCS})

# Set common target properties
set_target_properties(yolov8_det yolov8_seg yolov8_pose yolo11_det yolo11_obb yolo11_pose yolo11_seg PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Set version-specific include directories for each target
target_include_directories(yolov8_det PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolov8/include
    ${PROJECT_SOURCE_DIR}/include/yolov8/plugin
)
target_include_directories(yolov8_seg PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolov8/include
    ${PROJECT_SOURCE_DIR}/include/yolov8/plugin
)
target_include_directories(yolov8_pose PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolov8/include
    ${PROJECT_SOURCE_DIR}/include/yolov8/plugin
)
target_include_directories(yolo11_det PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolo11/include
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin
)
target_include_directories(yolo11_obb PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolo11/include
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin
)
target_include_directories(yolo11_pose PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolo11/include
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin
)
target_include_directories(yolo11_seg PRIVATE
    ${PROJECT_SOURCE_DIR}/include/yolo11/include
    ${PROJECT_SOURCE_DIR}/include/yolo11/plugin
)

# Common include directories for all executables
foreach(target yolov8_det yolov8_seg yolov8_pose yolo11_det yolo11_obb yolo11_pose yolo11_seg)
    target_include_directories(${target} PRIVATE
        ${rclcpp_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${vision_msgs_INCLUDE_DIRS}
        ${visualization_msgs_INCLUDE_DIRS}
        /opt/ros/humble/include
        /opt/ros/humble/include/cv_bridge
    )
endforeach()

# Common dependencies for all executables
set(COMMON_DEPENDENCIES
    rclcpp
    cv_bridge
    image_transport
    sensor_msgs
    geometry_msgs
    tf2_ros
    std_msgs
    rosbag2_cpp
    rosbag2_compression
    vision_msgs
    visualization_msgs
)

foreach(target yolov8_det yolov8_seg yolov8_pose yolo11_det yolo11_obb yolo11_pose yolo11_seg)
    ament_target_dependencies(${target} ${COMMON_DEPENDENCIES})
endforeach()

# Link libraries with version-specific plugins
target_link_libraries(yolov8_det
    nvinfer
    cudart
    yolov8plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolov8_seg
    nvinfer
    cudart
    yolov8plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolov8_pose
    nvinfer
    cudart
    yolov8plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolo11_det
    nvinfer
    cudart
    yolo11plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolo11_obb
    nvinfer
    cudart
    yolo11plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolo11_pose
    nvinfer
    cudart
    yolo11plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)
target_link_libraries(yolo11_seg
    nvinfer
    cudart
    yolo11plugin
    ${OpenCV_LIBS}
    ${cv_bridge_LIBRARIES}
    ${vision_msgs_LIBRARIES}
    ${visualization_msgs_LIBRARIES}
    cv_bridge::cv_bridge
)

ament_export_libraries(yolov8plugin yolo11plugin)

# Install targets
install(TARGETS
    yolov8_det yolov8_seg yolov8_pose yolo11_det yolo11_obb yolo11_pose yolo11_seg
    yolov8plugin yolo11plugin
    DESTINATION lib/${PROJECT_NAME}
)

# Install launch and config directories
install(DIRECTORY launch/yolo11 launch/yolov8
  DESTINATION share/${PROJECT_NAME}
  PATTERN "*.launch.py" PATTERN "*.launch.xml"
)

install(DIRECTORY config/yolo11 config/yolov8
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(${COMMON_DEPENDENCIES})

# Export include directories (移除了版本特定的路径)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()