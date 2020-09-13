//File: ray_operations.h
#ifndef RAY_OPERATIONS_H
#define RAY_OPERATIONS_H

#include <curand_kernel.h>

#include "../../util/vector_util.h"
#include "../cartesian_system.h"
#include "../vector_and_matrix/vec3.h"
#include "ray.h"

__device__ Ray generate_ray(
  vec3 init_point, vec3 main_dir, vec3 normal, int mode, float n,
  curandState *rand_state
);

__device__ Ray generate_ray(
  vec3 init_point, vec3 main_dir, vec3 normal, int mode, float n,
  curandState *rand_state
) {
  CartesianSystem new_xyz_system;
  vec3 v3_rand, dir, v3_rand_world;
  if (mode == 0) {
    new_xyz_system = CartesianSystem(normal);
    v3_rand = get_random_unit_vector_hemisphere_cos_pdf(rand_state);
  } else if (mode == 1) {
    new_xyz_system = CartesianSystem(main_dir);
    v3_rand = get_random_unit_vector_phong(n, rand_state);
  } else {
    new_xyz_system = CartesianSystem(normal);
    v3_rand = get_random_unit_vector_hemisphere(rand_state);
  }
  v3_rand_world = new_xyz_system.to_world_system(v3_rand);

  return Ray(init_point, v3_rand_world);
}

#endif
