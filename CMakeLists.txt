cmake_minimum_required(VERSION 3.0.2)
project(tracking_ros)

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  jsk_data
  dynamic_reconfigure
)

# ------------------------------------------------------------------------------------
# Download
# ------------------------------------------------------------------------------------
add_custom_target(${PROJECT_NAME}_install_trained_data ALL COMMAND python$ENV{ROS_PYTHON_VERSION} ${PROJECT_SOURCE_DIR}/scripts/install_trained_data.py)


catkin_package(
  CATKIN_DEPENDS
  jsk_data
)

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.txt
  PYTHON_INTERPRETER python3.8
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
