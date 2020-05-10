//File: material_operations.h
#ifndef MATERIAL_OPERATIONS_H
#define MATERIAL_OPERATIONS_H

#include <curand_kernel.h>

#include "../bvh/bvh.h"
#include "../bvh/bvh_traversal_target.h"
#include "material.h"

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &factor, Node **target_node_list,
  curandState *rand_state_mis, curandState *rand_state_mis_geom
);

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Primitive **target_geom_array, int num_target_geom,
  curandState *rand_state_mis_geom
);

__device__ float _recompute_pdf(
  hit_record rec, vec3 origin, vec3 dir, Primitive **target_geom_array,
  int num_target_geom, float hittable_pdf_weight, Node **target_node_list
);

__device__ float _recompute_pdf(
  hit_record rec, vec3 origin, vec3 dir, Primitive **target_geom_array,
  int num_target_geom, float hittable_pdf_weight, Node **target_node_list
) {
  float hittable_pdf = 0, cos_pdf;
  float weight = 1.0 / num_target_geom;
  int num_potential_targets = 0;
  int potential_target_idx[400];

  dir = unit_vector(dir);

  Ray ray = Ray(origin, dir);

  traverse_bvh_target(
    target_node_list[0], ray, potential_target_idx, num_potential_targets, 400
  );

  for(int i = 0; i < num_potential_targets; i++) {
    hittable_pdf += weight * target_geom_array[potential_target_idx[i]] ->
      get_hittable_pdf(rec.point, dir);
  }

  if (dot(rec.normal, dir) <= 0) {
    cos_pdf = 0;
  } else {
    cos_pdf = dot(rec.normal, dir);
  }

  return hittable_pdf_weight * hittable_pdf + (
    1 - hittable_pdf_weight) * cos_pdf;
}

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Primitive **target_geom_array, int num_target_geom,
  curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis_geom[0]);
  int selected_idx = int(random_number * (num_target_geom - 1));
  hit_record random_hit_record = target_geom_array[selected_idx] ->
    get_random_point_on_surface(&rand_state_mis_geom[1]);
  return random_hit_record.point;
}

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &factor, Node **target_node_list,
  curandState *rand_state_mis, curandState *rand_state_mis_geom
) {
  float random_number = curand_uniform(&rand_state_mis[0]), pdf, scattering_pdf;
  vec3 new_target_point;
  Ray default_ray = ref.ray;

  if (random_number > .5) {
    new_target_point = _pick_a_random_point_on_a_target_geom(
      target_geom_array, num_target_geom, rand_state_mis_geom
    );
    ref.ray = Ray(default_ray.p0, new_target_point - default_ray.p0);
  }

  pdf = _recompute_pdf(
    rec, ref.ray.p0, ref.ray.dir, target_geom_array, num_target_geom, .5,
    target_node_list
  );

  if (dot(rec.normal, ref.ray.dir) <= 0) {
    scattering_pdf = 0;
  } else {
    scattering_pdf = dot(rec.normal, ref.ray.dir);
  }

  factor = scattering_pdf / M_PI / pdf;
}

#endif
