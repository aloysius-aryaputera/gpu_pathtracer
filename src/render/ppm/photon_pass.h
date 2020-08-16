//File: photon_pass.h
#ifndef PHOTON_PASS_H
#define PHOTON_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/bvh/bvh_traversal.h"
#include "../../model/geometry/primitive.h"
#include "../../model/material/material.h"
#include "../../model/point/point.h"
#include "../../model/ray/ray.h"
#include "../../util/vector_util.h"
#include "../material_list_operations.h"

__global__
void gather_recorded_photons(
  Point **photon_list, int num_photons, int *num_recorded_photons
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i > 0) return;

  num_recorded_photons[0] = 0;
  for (int idx = 0; idx < num_photons; idx++) {
    if (!(photon_list[idx] -> location.vector_is_inf())) {
      (num_recorded_photons[0])++;
      photon_list[(num_recorded_photons[0]) - 1] = photon_list[idx];
    }
  }
}

__global__
void create_photon_list(Point **photon_list, int num_photons) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  photon_list[i] = new Point(
    vec3(INFINITY, INFINITY, INFINITY), vec3(1, 1, 1), vec3(0, 0, 1), 0
  ); 
}

__global__
void compute_accummulated_light_source_area(
  Primitive** light_source_geom_list, int num_light_source_geom,
  float *accummulated_light_source_area
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0) return;

  float acc = 0;
  float new_area = 0;
  for (int i = 0; i < num_light_source_geom; i++) {
    new_area = de_nan(light_source_geom_list[i] -> get_area());
    acc += new_area;
    accummulated_light_source_area[i] = acc;
  }
}

__device__ int pick_primitive_idx_for_sampling(
  int num_light_source_geom, float *accummulated_light_source_area, 
  curandState *rand_state
) {
  float random_number = curand_uniform(&rand_state[0]);
  int idx = 0;
  float accummulated_area = accummulated_light_source_area[idx];
  float random_number_2 = random_number * accummulated_light_source_area[
    num_light_source_geom - 1];

  while(
    random_number_2 > accummulated_area && idx < (num_light_source_geom - 1)
  ) {
    idx++;
    accummulated_area = accummulated_light_source_area[idx];
  }

  return idx;
}

__global__
void photon_pass(
  Primitive **target_geom_list, Node **geom_node_list,
  Point **photon_list, 
  int num_light_source_geom, float *accummulated_light_source_area,
  int num_photons, int max_bounce, curandState *rand_state
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  photon_list[i] -> assign_location(vec3(INFINITY, INFINITY, INFINITY));

  vec3 filter, light_source_color, tex_specular, tex_diffuse;
  hit_record rec;
  reflection_record ref;
  Material* material_list[400];
  int num_bounce = 0, material_list_length = 0;
  bool hit = false, sss = false;
  curandState local_rand_state = rand_state[i];
  int light_source_idx = pick_primitive_idx_for_sampling(
    num_light_source_geom, accummulated_light_source_area, &local_rand_state
  );
  float random_number, reflection_prob, light_source_color_original_length;

  rec = target_geom_list[light_source_idx] -> get_random_point_on_surface(
    &local_rand_state
  );
  light_source_color = target_geom_list[light_source_idx] -> get_material() ->
    get_texture_emission(rec.uv_vector);
  light_source_color_original_length = light_source_color.length();
  Ray ray = generate_ray(
    rec.point, vec3(0, 0, 0), rec.normal, 2, 1, &local_rand_state);
  hit = traverse_bvh(geom_node_list[0], ray, rec);

  if (hit) {
    while (hit && num_bounce < max_bounce) {
      num_bounce += 1;

      rec.object -> get_material() -> check_next_path(
        rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
        sss, material_list, material_list_length,
        ref, rand_state
      );

      if (ref.false_hit && ref.entering)
        add_new_material(
          material_list, material_list_length, rec.object -> get_material()
        );

      if (ref.false_hit && !(ref.entering))
        remove_a_material(
          material_list, material_list_length, rec.object -> get_material()
        );

      if (!(ref.false_hit) && ref.refracted && ref.entering)
        add_new_material(
          material_list, material_list_length, rec.object -> get_material()
        );

      if (!(ref.false_hit) && ref.refracted && !(ref.entering))
        remove_a_material(
          material_list, material_list_length, rec.object -> get_material()
        );
     
      if (!(ref.false_hit)) {
        random_number = curand_uniform(&local_rand_state);
	reflection_prob = max(ref.k);
        if (random_number > reflection_prob) {
	  if (ref.diffuse) {
	    photon_list[i] -> assign_location(rec.point);
	    photon_list[i] -> assign_color(light_source_color);
	    photon_list[i] -> assign_direction(rec.coming_ray.dir);
	  }  
	} else {
	  light_source_color = ref.k * light_source_color;
	  light_source_color = light_source_color * (
	    light_source_color_original_length / light_source_color.length());
	}	
      } 

      ray = ref.ray;
      hit = traverse_bvh(geom_node_list[0], ray, rec);
    }
  }
}

#endif
