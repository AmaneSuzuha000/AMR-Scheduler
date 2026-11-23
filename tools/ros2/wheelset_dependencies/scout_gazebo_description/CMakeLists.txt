cmake_minimum_required(VERSION 3.8)
project(scout_gazebo_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".git" EXCLUDE
)

ament_package()
