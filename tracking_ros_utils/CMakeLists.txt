cmake_minimum_required(VERSION 3.0.2)
project(tracking_ros_utils)
find_package(catkin REQUIRED
    sensor_msgs
    geometry_msgs
    jsk_recognition_msgs
    message_generation
)
add_service_files(
    FILES
    SamPrompt.srv
)
generate_messages(
    DEPENDENCIES
    sensor_msgs
    geometry_msgs
    jsk_recognition_msgs
)
catkin_package()
