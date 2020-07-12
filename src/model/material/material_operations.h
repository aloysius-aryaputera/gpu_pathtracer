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
	Node **target_leaf_list,
  curandState *rand_state_mis
);

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Primitive **target_geom_array, int num_target_geom, curandState *rand_state
);

__device__ vec3 _pick_a_random_point_on_a_target_geom_2(
	Node* target_bvh_root, vec3 origin, vec3 normal, vec3 kd, 
	curandState *rand_state
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
    cos_pdf = dot(rec.normal, dir) / M_PI;
  }

  return hittable_pdf_weight * hittable_pdf + (
    1 - hittable_pdf_weight) * cos_pdf;
}

__device__ float _recompute_pdf_2(
  hit_record rec, vec3 origin, vec3 dir, Primitive **target_geom_array,
  int num_target_geom, float hittable_pdf_weight, Node **target_node_list,
	Node **target_leaf_list, vec3 kd, bool diffuse, float n, 
	vec3 perfect_reflection_dir
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
		  target_leaf_list[potential_target_idx[i]], origin, pivot, 
			kd
		);
    hittable_pdf += node_pdf * target_geom_array[potential_target_idx[i]] ->
      get_hittable_pdf(rec.point, dir);
		if (node_pdf > 1) printf("node_pdf = %f\n", node_pdf);
  }

  if (dot(rec.normal, dir) <= 0) {
    sampling_pdf = 0;
  } else if (diffuse) {
    sampling_pdf = dot(rec.normal, dir) / M_PI;
  } else {
		sampling_pdf = (n + 1) * powf(dot(perfect_reflection_dir, ray.dir), n) / (2 * M_PI);
	}

  return hittable_pdf_weight * hittable_pdf + (
			1 - hittable_pdf_weight) * sampling_pdf;
}

__device__ vec3 _pick_a_random_point_on_a_target_geom(
  Primitive **target_geom_array, int num_target_geom, curandState *rand_state
) {
	float random_number = curand_uniform(rand_state);
  int selected_idx = int(random_number * (num_target_geom - 1));
  hit_record random_hit_record = target_geom_array[selected_idx] ->
    get_random_point_on_surface(rand_state);
  return random_hit_record.point;
}

__device__ vec3 _pick_a_random_point_on_a_target_geom_2(
	Node* target_bvh_root, vec3 origin, vec3 normal, vec3 kd, 
	curandState *rand_state
) {
	Primitive* selected_target;
	selected_target = traverse_bvh_to_pick_a_target(
	  target_bvh_root, origin, normal, kd, rand_state
	);
  hit_record random_hit_record = selected_target -> 
		get_random_point_on_surface(rand_state);
	return random_hit_record.point;
}

__device__ void change_ref_ray(
  hit_record rec, reflection_record &ref, Primitive **target_geom_array,
  int num_target_geom, float &factor, Node **target_node_list,
	Node **target_leaf_list,
  float hittable_pdf_weight, curandState *rand_state_mis
) {
  float random_number = curand_uniform(rand_state_mis);
  float pdf, scattering_pdf;
  vec3 new_target_point, new_dir, pivot;
  Ray default_ray = ref.ray;

	if (ref.diffuse) 
		pivot = rec.normal;
	else 
		pivot = ref.perfect_reflection_dir;

  if (random_number < hittable_pdf_weight) {
    //new_target_point = _pick_a_random_point_on_a_target_geom(
    //  target_geom_array, num_target_geom, rand_state_mis
    //);

    new_target_point = _pick_a_random_point_on_a_target_geom_2(
		  target_node_list[0], default_ray.p0, pivot, ref.filter,
		  rand_state_mis	
		);

		new_dir = new_target_point - default_ray.p0;

		//if (dot(new_dir, rec.normal) > 0)
    ref.ray = Ray(default_ray.p0, new_dir);
  }

	if (ref.reflected) {
	  ref.filter = ref.ks * (ref.n + 2) * powf(dot(ref.ray.dir, pivot), ref.n) / 2;
		ref.filter = vec3(
			fmaxf(0, ref.filter.r()), fmaxf(0, ref.filter.g()), 
			fmaxf(0, ref.filter.b())
		);
	}

  //pdf = _recompute_pdf(
  //  rec, ref.ray.p0, ref.ray.dir, target_geom_array, num_target_geom,
  //  hittable_pdf_weight, target_node_list
  //);

  pdf = _recompute_pdf_2(
    rec, ref.ray.p0, ref.ray.dir, target_geom_array, num_target_geom,
    hittable_pdf_weight, target_node_list, target_leaf_list, ref.filter,
    ref.diffuse, ref.n, ref.perfect_reflection_dir
  );

  if (dot(rec.normal, ref.ray.dir) <= 0) {
    scattering_pdf = 0;
  } else {
    scattering_pdf = dot(rec.normal, ref.ray.dir);
  }

  factor = scattering_pdf / M_PI / pdf;
}

#endif
