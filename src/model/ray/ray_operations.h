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
		v3_rand_world = new_xyz_system.to_world_system(v3_rand);
		dir = unit_vector(v3_rand_world);
  } else {

    if (main_dir.vector_is_nan()) {
		  printf("main_dir is nan\n");
		}

		new_xyz_system = CartesianSystem(main_dir);
    v3_rand = get_random_unit_vector_phong(n, rand_state);
		v3_rand_world = new_xyz_system.to_world_system(v3_rand);
		dir = unit_vector(v3_rand_world);

		if (
			(dot(main_dir, normal) >= 0 && dot(dir, normal) <= 0) ||
			(dot(main_dir, normal) <= 0 && dot(dir, normal) >= 0)
		) {
			dir = main_dir;
		}
  }

  //if (v3_rand_world.vector_is_nan())
	//	printf("Vector is nan!\n");

  //vec3 dir = unit_vector(main_dir + fuziness * v3_rand_world);
  return Ray(init_point, dir);
}

#endif
