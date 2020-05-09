//File: material_operations.h
#ifndef MATERIAL_OPERATIONS_H
#define MATERIAL_OPERATIONS_H

#include <curand_kernel.h>

#include "material.h"

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &factor, curandState *rand_state_mis,
  curandState *rand_state_mis_geom
);

__device__ void _pick_a_random_point_on_a_target_geom(
  hit_record rec, Primitive **target_geom_array, int num_target_geom,
  vec3 &target_point, float &pdf, curandState *rand_state_mis_geom
);

__device__ void _pick_a_random_point_on_a_target_geom(
  hit_record rec, Primitive **target_geom_array, int num_target_geom,
  vec3 &target_point, float &factor, curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis_geom[0]);
  int selected_idx = int(random_number * (num_target_geom - 1));
  // printf("random_number = %f, selected_idx = %d\n", random_number, selected_idx);
  hit_record random_hit_record = target_geom_array[selected_idx] ->
    get_random_point_on_surface(&rand_state_mis_geom[1]);
  hit_record dummy_hit_record;
  float cos_value;
  target_point = random_hit_record.point;
  vec3 dir = target_point - rec.point;
  float squared_dist = dir.squared_length();
  bool hit = target_geom_array[selected_idx] -> hit(
    Ray(ref.ray.p0, dir), INFINITY, dummy_hit_record
  );
  if (hit) {
    cos_value = -dot(dir, target_geom_array[selected_idx] -> get_fixed_normal())
  } else {
    cos_value = 0;
  }
  factor = num_target_geom * target_geom_array[selected_idx] -> get_area() * cos_value / squared_dist;
}

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &factor, curandState *rand_state_mis,
  curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis[0]);
  vec3 new_target_point;
  Ray default_ray = ref.ray;

  // printf("random_number = %f\n", random_number);

  if (random_number > 0) {
    _pick_a_random_point_on_a_target_geom(
      rec, target_geom_array, num_target_geom, new_target_point, factor,
      rand_state_mis_geom
    );
    ref.ray = Ray(default_ray.p0, new_target_point - default_ray.p0);
  }
}

#endif
