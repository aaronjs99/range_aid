cmake_minimum_required(VERSION 3.0.2)
project(range_aid)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  rtabmap_msgs
  rospy
  std_msgs
  std_srvs
  tf2_ros
)

add_message_files(
  FILES
    CorrectionProposal.msg
    RangeAidStatus.msg
    RangeObservation.msg
)

# Catkin's standard Python relay requires a nested source directory named after
# the ROS package. Range Aid intentionally keeps a flat scripts/ tree, so this
# small devel-space initializer extends the generated message package to that
# source root instead.
set(range_aid_CATKIN_PYTHON_SETUP_HAS_PACKAGE_INIT TRUE)
file(MAKE_DIRECTORY "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/range_aid")
configure_file(
  cmake/range_aid_devel_init.py.in
  "${CATKIN_DEVEL_PREFIX}/${PYTHON_INSTALL_DIR}/range_aid/__init__.py"
  @ONLY
)

generate_messages(DEPENDENCIES geometry_msgs std_msgs)

catkin_package(CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs rtabmap_msgs rospy std_msgs std_srvs tf2_ros)

catkin_install_python(PROGRAMS
  scripts/range_aid_node.py
  scripts/synthetic_range_source.py
  scripts/export_cora_snapshot.py
  scripts/range_aid_archive.py
  scripts/run_official_cora.py
  scripts/run_score_baseline.py
  scripts/validation/validate_estimator_consistency.py
  scripts/validation/validate_shadow_runtime.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch tools
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Keep the source tree flat while still supporting a complete catkin install.
# The bridge becomes the installed package initializer; reusable online modules
# are copied below it. Repository-only synthetic experiment modules remain in
# source and are driven by run.py.
install(FILES scripts/range_aid.py
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  RENAME __init__.py
)

install(DIRECTORY
    scripts/archive
    scripts/certification
    scripts/estimation
    scripts/models
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)
