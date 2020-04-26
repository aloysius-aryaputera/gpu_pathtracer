//File: cartesian_system.h
#ifndef CARTESIAN_SYSTEM_H
#define CARTESIAN_SYSTEM_H

#include "vector_and_matrix/vec3.h"

class CartesianSystem {

  public:
    __host__ __device__ CartesianSystem();
    __host__ __device__ CartesianSystem(vec3 new_z_axis_);
    __host__ __device__ CartesianSystem(
      vec3 new_z_axis_, vec3 new_x_axis_draft
    );
    __host__ __device__ vec3 to_world_system(vec3 input_vector);

    vec3 new_x_axis, new_y_axis, new_z_axis;
};

__host__ __device__ CartesianSystem::CartesianSystem() {
  this -> new_x_axis = vec3(1, 0, 0);
  this -> new_y_axis = vec3(0, 1, 0);
  this -> new_z_axis = vec3(0, 0, 1);
}

__host__ __device__ CartesianSystem::CartesianSystem(vec3 new_z_axis_) {
  this -> new_z_axis = unit_vector(new_z_axis_);
  if (abs(new_z_axis.x()) > abs(new_z_axis.y())) {
    this -> new_x_axis = vec3(
      this -> new_z_axis.z(), 0, -this -> new_z_axis.x()) / \
      sqrt(this -> new_z_axis.x() * this -> new_z_axis.x() +
           this -> new_z_axis.z() * this -> new_z_axis.z());
  } else {
    this -> new_x_axis = vec3(
      0, -this -> new_z_axis.z(), this -> new_z_axis.y()) / \
      sqrt(this -> new_z_axis.y() * this -> new_z_axis.y() +
           this -> new_z_axis.z() * this -> new_z_axis.z());
  }
  this -> new_y_axis = unit_vector(
    cross(this -> new_z_axis, this -> new_x_axis));
}

__host__ __device__ CartesianSystem::CartesianSystem(
  vec3 new_z_axis_, vec3 new_x_axis_draft
) {
  this -> new_z_axis = unit_vector(new_z_axis_);
  this -> new_y_axis = unit_vector(
    cross(this -> new_z_axis, new_x_axis_draft));
  this -> new_x_axis = unit_vector(
    cross(this -> new_y_axis, this -> new_z_axis));
}

__host__ __device__ vec3 CartesianSystem::to_world_system(vec3 input_vector) {
  vec3 v3_rand_world = vec3(
    input_vector.x() * this -> new_x_axis.x() + \
    input_vector.y() * this -> new_y_axis.x() + \
    input_vector.z() * this -> new_z_axis.x(),
    input_vector.x() * this -> new_x_axis.y() + \
    input_vector.y() * this -> new_y_axis.y() + \
    input_vector.z() * this -> new_z_axis.y(),
    input_vector.x() * this -> new_x_axis.z() + \
    input_vector.y() * this -> new_y_axis.z() + \
    input_vector.z() * this -> new_z_axis.z()
  );
  v3_rand_world.make_unit_vector();
  if (isnan(v3_rand_world.x()) || isnan(v3_rand_world.y()) || isnan(v3_rand_world.z())) {
    printf(
      "input_vector = %5.5f, %5.5f, %5.5f; \
      new_x_axis = %5.5f, %5.5f, %5.5f;\
      new_y_axis = %5.5f, %5.5f, %5.5f;\
      new_z_axis = %5.5f, %5.5f, %5.5f;\n",
      input_vector.x(), input_vector.y(), input_vector.z(),
      this -> new_x_axis.x(), this -> new_x_axis.y(), this -> new_x_axis.z(),
      this -> new_y_axis.x(), this -> new_y_axis.y(), this -> new_y_axis.z(),
      this -> new_z_axis.x(), this -> new_z_axis.y(), this -> new_z_axis.z()
    );
  }
  return v3_rand_world;
}


#endif
