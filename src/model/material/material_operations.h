//File: material_operations.h
#ifndef MATERIAL_OPERATIONS_H
#define MATERIAL_OPERATIONS_H

#include <curand_kernel.h>

#include "../bvh/bvh.h"
#include "../bvh/bvh_traversal_target.h"
#include "material.h"

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref,
  Primitive **target_geom_array,
  int num_target_geom, float &factor, 
  Node **target_node_list,
  Node **target_leaf_list,
  curandState *rand_state_mis,
  bool write
);

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Node* target_bvh_root, vec3 origin, vec3 normal, vec3 kd, 
  curandState *rand_state
);

__device__ float _recompute_pdf(
  hit_record rec, vec3 origin, vec3 dir, Primitive **target_geom_array,
  int num_target_geom, float hittable_pdf_weight, Node **target_node_list,
  Node **target_leaf_list, vec3 kd, bool diffuse, float n, 
  vec3 perfect_reflection_dir, reflection_record ref
) {
  float hittable_pdf = 0, sampling_pdf;
  float node_pdf = 0;
  int num_potential_targets = 0;
  int potential_target_idx[400];
  vec3 pivot;

  if (diffuse) 
    pivot = rec.normal;
  else 
    pivot = perfect_reflection_dir;

  dir = unit_vector(dir);

  Ray ray = Ray(origin, dir);

  traverse_bvh_target(
    target_node_list[0], ray, potential_target_idx, num_potential_targets, 400
  );

  for(int i = 0; i < num_potential_targets; i++) {
    node_pdf = get_node_pdf(
      target_leaf_list[potential_target_idx[i]], origin, pivot, kd);
    hittable_pdf += node_pdf * target_geom_array[potential_target_idx[i]] ->
      get_hittable_pdf(rec.point, dir);
    if (node_pdf > 1) printf("node_pdf = %f\n", node_pdf);
  }

  if (ref.diffuse) {
    sampling_pdf = compute_diffuse_sampling_pdf(rec.normal, ref.ray.dir);
  } else {
    sampling_pdf = compute_specular_sampling_pdf(
      rec.coming_ray.dir, ref.ray.dir, rec.normal, perfect_reflection_dir,
      n, ref.refracted);
  }

  if (isnan(hittable_pdf))
    printf("hittable_pdf is nan\n");	 

  return hittable_pdf_weight * hittable_pdf + (
    1 - hittable_pdf_weight) * sampling_pdf;
}

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Node* target_bvh_root, vec3 origin, vec3 normal, vec3 kd, 
  curandState *rand_state, bool write
) {
  Primitive* selected_target;
  selected_target = traverse_bvh_to_pick_a_target(
    target_bvh_root, origin, normal, kd, rand_state, write
  );
  hit_record random_hit_record = selected_target -> 
    get_random_point_on_surface(rand_state);
  return random_hit_record.point;
}

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref,
  Primitive **target_geom_array,
  int num_target_geom, float &factor, Node **target_node_list,
  Node **target_leaf_list,
  float hittable_pdf_weight, curandState *rand_state_mis,
  bool write=false
) {
  float random_number = curand_uniform(rand_state_mis);
  float pdf, scattering_pdf;
  vec3 new_target_point, new_dir, pivot;
  Ray default_ray = ref.ray;

  if (ref.diffuse) 
    pivot = rec.normal;
  else 
    pivot = ref.perfect_reflection_dir;

  if (isinf(ref.n))
    hittable_pdf_weight = 0;

  if (random_number < hittable_pdf_weight) {

    new_target_point = _pick_a_random_point_on_a_target_geom(
      target_node_list[0], default_ray.p0, pivot, ref.k,
      rand_state_mis, write	
    );

    new_dir = new_target_point - default_ray.p0;

    ref.ray = Ray(default_ray.p0, new_dir);

    if (ref.reflected || ref.refracted) {
      ref.filter = compute_phong_filter(ref.k, ref.n, pivot, ref.ray.dir);
    }
  }

  pdf = _recompute_pdf(
    rec, ref.ray.p0, ref.ray.dir, 
    target_geom_array, num_target_geom,
    hittable_pdf_weight, target_node_list, target_leaf_list, ref.k,
    ref.diffuse, ref.n, ref.perfect_reflection_dir, ref
  );

  scattering_pdf = compute_scattering_pdf(
    rec.normal, ref.ray.dir, ref.diffuse, rec.coming_ray.dir, ref.refracted
  );

  //if (ref.diffuse) {
  //  scattering_pdf = fmaxf(0.0, dot(rec.normal, ref.ray.dir));
  //} else {
  //  float dot_prod_1 = dot(rec.coming_ray.dir, rec.normal);
  //  float dot_prod_2 = dot(ref.ray.dir, rec.normal);
  //  scattering_pdf = (
  //    (dot_prod_1 >= 0 && dot_prod_2 <= 0 && ref.reflected) ||
  //    (dot_prod_1 <= 0 && dot_prod_2 >= 0 && ref.reflected) ||
  //    (dot_prod_1 >= 0 && dot_prod_2 >= 0 && ref.refracted) ||
  //    (dot_prod_1 <= 0 && dot_prod_2 <= 0 && ref.refracted)
  //  );
  //}

  factor = scattering_pdf / M_PI / pdf;

  if (write) {
    printf("scattering_pdf = %f, pdf = %f, factor = %f.\n", scattering_pdf, pdf, factor);
  }
}

#endif
