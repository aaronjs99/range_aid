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

catkin_python_setup()

add_message_files(
  FILES
    CorrectionProposal.msg
    RangeAidStatus.msg
    RangeObservation.msg
)

generate_messages(DEPENDENCIES geometry_msgs std_msgs)

catkin_package(CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs rtabmap_msgs rospy std_msgs std_srvs tf2_ros)

catkin_install_python(PROGRAMS
  nodes/range_aid_node.py
  nodes/synthetic_range_source.py
  scripts/export_cora_snapshot.py
  scripts/range_aid_archive.py
  scripts/run_score_baseline.py
  scripts/validation/validate_estimator_consistency.py
  scripts/validation/validate_shadow_runtime.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
