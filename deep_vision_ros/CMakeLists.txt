cmake_minimum_required(VERSION 3.20.0)
project(deep_vision_ros)

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  sensor_msgs
  dynamic_reconfigure
)

# Dynamic reconfigure setup
generate_dynamic_reconfigure_options(
  cfg/GroundingDINO.cfg
  cfg/YOLO.cfg
  cfg/VLPart.cfg
)


catkin_package(
  CATKIN_DEPENDS
)

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.txt
  PYTHON_INTERPRETER python3.9
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
  CHECK_VENV FALSE
)

# file(GLOB SCRIPT_PROGRAMS scripts/*.py)
# catkin_install_python(
#   PROGRAMS ${SCRIPT_PROGRAMS}
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts/)
file(GLOB SCRIPTS_FILES node_scripts/*.py)
catkin_install_python(
  PROGRAMS ${SCRIPTS_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(FILES
  requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# if(CATKIN_ENABLE_TESTING)
#   add_custom_target(${PROJECT_NAME}_download_image ALL
#     COMMAND ${PROJECT_SOURCE_DIR}/test/prepare_test_data.sh)
#   find_package(catkin REQUIRED COMPONENTS rostest)
#   add_rostest(test/test_node.test)
# endif()
