//File: ray_operations.h
#ifndef RAY_OPERATIONS_H
#define RAY_OPERATIONS_H

#include <curand_kernel.h>

#include "../../util/vector_util.h"
#include "../cartesian_system.h"
#include "../vector_and_matrix/vec3.h"
#include "ray.h"

__device__ Ray generate_ray(
  vec3 init_point, vec3 main_dir, vec3 normal, float fuziness,
  curandState *rand_state
);

__device__ Ray generate_ray(
  vec3 init_point, vec3 main_dir, vec3 normal, float fuziness,
  curandState *rand_state
) {
  CartesianSystem new_xyz_system = CartesianSystem(normal);
  vec3 v3_rand = get_random_unit_vector_hemisphere(rand_state);
  vec3 v3_rand_world = new_xyz_system.to_world_system(v3_rand);
  vec3 dir = unit_vector(main_dir + fuziness * v3_rand_world);
  return Ray(init_point, dir);
}

#endif