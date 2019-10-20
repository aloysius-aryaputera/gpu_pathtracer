//File: cartesian_system.h
#ifndef CARTESIAN_SYSTEM_H
#define CARTESIAN_SYSTEM_H

#include "vector_and_matrix/vec3.h"

class CartesianSystem {

  public:
    __host__ __device__ CartesianSystem();
    __host__ __device__ CartesianSystem(vec3 new_z_axis_);
    __host__ __device__ vec3 to_world_system(vec3 input_vector);

    vec3 new_x_axis, new_y_axis, new_z_axis;
};

__host__ __device__ CartesianSystem::CartesianSystem() {
  new_x_axis = vec3(1, 0, 0);
  new_y_axis = vec3(0, 1, 0);
  new_z_axis = vec3(0, 0, 1);
}

__host__ __device__ CartesianSystem::CartesianSystem(vec3 new_z_axis_) {
  new_z_axis = new_z_axis_;
  if (abs(new_z_axis.x()) > abs(new_z_axis.y())) {
    new_x_axis = vec3(new_z_axis.z(), 0, -new_z_axis.x()) / \
      sqrt(new_z_axis.x() * new_z_axis.x() + new_z_axis.z() * new_z_axis.z());
  } else {
    new_x_axis = vec3(0, -new_z_axis.z(), new_z_axis.y()) / \
      sqrt(new_z_axis.y() * new_z_axis.y() + new_z_axis.z() * new_z_axis.z());
  }
  new_y_axis = unit_vector(cross(new_z_axis, new_x_axis));
}

__host__ __device__ vec3 CartesianSystem::to_world_system(vec3 input_vector) {
  vec3 v3_rand_world = unit_vector(
    vec3(
      input_vector.x() * new_x_axis.x() + input_vector.y() * new_y_axis.x() + \
      input_vector.z() * new_z_axis.x(),
      input_vector.x() * new_x_axis.y() + input_vector.y() * new_y_axis.y() + \
      input_vector.z() * new_z_axis.y(),
      input_vector.x() * new_x_axis.z() + input_vector.y() * new_y_axis.z() + \
      input_vector.z() * new_z_axis.z()
    )
  );
  return v3_rand_world;
}


#endif
