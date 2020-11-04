//File: photon_pass.h
#ifndef PHOTON_PASS_H
#define PHOTON_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/bvh/bvh_traversal.h"
#include "../../model/cartesian_system.h"
#include "../../model/geometry/primitive.h"
#include "../../model/material/material.h"
#include "../../model/point/point.h"
#include "../../model/ray/ray.h"
#include "../../util/vector_util.h"
#include "../material_list_operations.h"

__device__ vec3 _get_new_scattering_direction(
  vec3 current_dir, float g, curandState *rand_state
) {
  float cos_theta = henyey_greenstein_cos_theta(g, rand_state);
  float sin_theta = powf(1 - powf(cos_theta, 2), .5);
  float cot_theta = cos_theta / sin_theta;
  CartesianSystem cart_sys = CartesianSystem(current_dir);
  vec3 new_dir = get_random_unit_vector_disk(rand_state);
  float new_dir_z = cot_theta * powf(
    new_dir.x() * new_dir.x() + new_dir.y() * new_dir.y(), .5);
  new_dir = vec3(new_dir.x(), new_dir.y(), new_dir_z);
  new_dir.make_unit_vector();
  return cart_sys.to_world_system(new_dir);
}	

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
      photon_list[(num_recorded_photons[0]) - 1] -> assign_location(
         photon_list[idx] -> location);
      photon_list[(num_recorded_photons[0]) - 1] -> assign_color(
         photon_list[idx] -> color);
      photon_list[(num_recorded_photons[0]) - 1] -> assign_direction(
         photon_list[idx] -> direction);
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
void compute_accummulated_light_source_energy(
  Primitive** light_source_geom_list, int num_light_source_geom,
  float *accummulated_light_source_energy
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > 0) return;

  float acc = 0;
  float new_energy = 0;
  for (int i = 0; i < num_light_source_geom; i++) {
    new_energy = de_nan(light_source_geom_list[i] -> get_energy().mean());
    acc += new_energy;
    accummulated_light_source_energy[i] = acc;
  }
}

__device__ int pick_primitive_idx_for_sampling(
  int num_light_source_geom, float *accummulated_light_source_energy, 
  curandState *rand_state
) {
  float random_number = curand_uniform(&rand_state[0]);
  int idx = 0;
  float accummulated_energy = accummulated_light_source_energy[idx];
  float random_number_2 = random_number * accummulated_light_source_energy[
    num_light_source_geom - 1];

  while(
    random_number_2 > accummulated_energy && idx < (num_light_source_geom - 1)
  ) {
    idx++;
    accummulated_energy = accummulated_light_source_energy[idx];
  }

  return idx;
}

__global__
void photon_pass(
  Primitive **target_geom_list, Node **geom_node_list,
  Point **photon_list, 
  int num_light_source_geom, float *accummulated_light_source_energy,
  int num_photons, int max_bounce,
  int pass_iteration, curandState *rand_state
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  photon_list[i] -> assign_location(vec3(INFINITY, INFINITY, INFINITY));

  vec3 filter, light_source_color, tex_specular, tex_diffuse, prev_location;
  hit_record rec;
  reflection_record ref;
  Material* material_list[400];
  int num_bounce = 0, material_list_length = 0, light_source_idx;
  bool hit = false, sss = false;
  curandState local_rand_state = rand_state[i];
  float random_number, reflection_prob, mean_color, mean_color_tmp;
  float max_energy = accummulated_light_source_energy[num_light_source_geom - 1];

  for (int idx = 0; idx < pass_iteration; idx++) {
    curand_uniform(&local_rand_state);
  }

  light_source_idx = pick_primitive_idx_for_sampling(
    num_light_source_geom, accummulated_light_source_energy, &local_rand_state
  );

  rec = target_geom_list[light_source_idx] -> get_random_point_on_surface(
    &local_rand_state
  );
  light_source_color = target_geom_list[light_source_idx] -> get_material() ->
    get_texture_emission(rec.uv_vector);
  light_source_color *= max_energy / light_source_color.mean();
  mean_color = light_source_color.mean();
  Ray ray = generate_ray(
    rec.point, vec3(0, 0, 0), rec.normal, 2, 1, &local_rand_state);
  
  prev_location = rec.point;
  photon_list[i] -> assign_prev_location(prev_location);
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
	  if (ref.diffuse && num_bounce > 1) {
	    photon_list[i] -> assign_prev_location(prev_location);
	    photon_list[i] -> assign_location(rec.point);
	    photon_list[i] -> assign_color(light_source_color);
	    photon_list[i] -> assign_direction(rec.coming_ray.dir);
	  }  
	} else {
	  light_source_color = ref.k * light_source_color;
	  //light_source_color = light_source_color * (
	  //  max_energy / light_source_color.length());
	  mean_color_tmp = (
	    light_source_color.r() + light_source_color.g() + 
	    light_source_color.b()
	  ) / 3;
	  light_source_color *= (mean_color / mean_color_tmp);
	}	
      } 

      ray = ref.ray;
      prev_location = rec.point;
      hit = traverse_bvh(geom_node_list[0], ray, rec);
      
    }
  }
}

#endif
