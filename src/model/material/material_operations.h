//File: material_operations.h
#ifndef MATERIAL_OPERATIONS_H
#define MATERIAL_OPERATIONS_H

#include <curand_kernel.h>

#include "material.h"

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, curandState *rand_state_mis,
  curandState *rand_state_mis_geom
);

__device__ void pick_a_random_point_on_a_target_geom(
  hit_record rec, Primitive **target_geom_array, int num_target_geom,
  vec3 &target_point, float &pdf, curandState *rand_state_mis_geom
);

__device__ void pick_a_random_point_on_a_target_geom(
  hit_record rec, Primitive **target_geom_array, int num_target_geom,
  vec3 &target_point, float &pdf, curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis_geom[0]);
  int selected_idx = int(random_number * (num_target_geom - 1));
  hit_record random_hit_record = target_geom_array[selected_idx] ->
    get_random_point_on_surface(&rand_state_mis_geom[0]);
  float cos_value;
  target_point = random_hit_record.point;
  vec3 dir = target_point - rec.point;
  float squared_dist = dir.squared_length();
  cos_value = dot(
    unit_vector(dir), target_geom_array[selected_idx] -> get_fixed_normal()
  );
  if (cos_value < 0) {
    cos_value = -cos_value;
  } else {
    cos_value = 0;
  }
  pdf = target_geom_array[selected_idx] -> get_area() * cos_value / squared_dist;
}

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &pdf, curandState *rand_state_mis,
  curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis[0]);
  vec3 new_target_point;
  Ray default_ray = ref.ray;

  if (random_number > .5) {
    pick_a_random_point_on_a_target_geom(
      rec, target_geom_array, num_target_geom, new_target_point, pdf,
      rand_state_mis_geom
    );
    ref.ray = Ray(default_ray.p0, new_target_point - default_ray.p0);
  }
}

#endif
