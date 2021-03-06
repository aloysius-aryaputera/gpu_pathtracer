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
#include "../../param.h"
#include "../../util/vector_util.h"
#include "../material_list_operations.h"
#include "common.h"

__global__
void gather_recorded_photons(
  Point **photon_list, Point **surface_photon_list,
  Point **volume_photon_list, int num_photons, 
  int *num_surface_photons, int *num_volume_photons
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i > 0) return;

  int new_idx;

  num_surface_photons[0] = 0;
  num_volume_photons[0] = 0;

  for (int idx = 0; idx < num_photons; idx++) {
    if (!(photon_list[idx] -> location.vector_is_inf())) {
      if (photon_list[idx] -> on_surface) {
        (num_surface_photons[0])++;
	new_idx = num_surface_photons[0] - 1;
	surface_photon_list[new_idx] = photon_list[idx];
      } else {
	(num_volume_photons[0])++;
        new_idx = num_volume_photons[0] - 1;
	volume_photon_list[new_idx] = photon_list[idx];
      }
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
  Node **transparent_node_list, Point **photon_list, 
  int num_light_source_geom, float *accummulated_light_source_energy,
  int num_photons, int max_bounce,
  int pass_iteration, curandState *rand_state
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  photon_list[i] -> assign_location(vec3(INFINITY, INFINITY, INFINITY));

  vec3 filter, light_source_color, tex_specular, tex_diffuse, prev_location;
  vec3 new_scattering_dir;
  hit_record rec, rec_medium;
  reflection_record ref;
  Material* material_list[400], *medium;
  int num_bounce = -1, material_list_length = 0, light_source_idx;
  bool hit = false, sss = false, in_medium = false, scattered_in_medium = false;
  bool scattered_in_medium_now = false, direct_check_surface = false;
  curandState local_rand_state = rand_state[i];
  float random_number, reflection_prob, mean_color, mean_color_tmp, d;
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
  //photon_list[i] -> assign_prev_location(prev_location);

  while ((hit && num_bounce < max_bounce) || num_bounce < 0) {
    scattered_in_medium_now = false;

    num_bounce += 1;

    if (num_bounce <= 0) {
      init_material_list(
        material_list, material_list_length, transparent_node_list, rec.point, 
	rec.normal, &local_rand_state
      );
    }

    rec.object -> get_material() -> check_next_path(
      rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
      sss, material_list, material_list_length,
      ref, rand_state
    );

    if (num_bounce > 0) {
      rearrange_material_list(
        material_list, material_list_length, rec.object -> get_material(),
        ref.false_hit, ref.entering, ref.refracted 
      );
    }

    in_medium = check_if_entering_medium(ref, in_medium, medium);
 
    //if (!(ref.false_hit) && num_bounce > 0) {
    if (!(ref.false_hit)) {

      if (in_medium && !direct_check_surface) {
        ray = ref.ray;
        d = medium -> get_propagation_distance(&local_rand_state); 
        hit = traverse_bvh(geom_node_list[0], ray, rec_medium);
        
        while (d - rec_medium.t < SMALL_DOUBLE) {
          scattered_in_medium = true;
          scattered_in_medium_now = true;
          random_number = curand_uniform(&local_rand_state);
          if (random_number < medium -> scattering_prob) {
	    photon_list[i] -> assign_prev_location(prev_location);
            photon_list[i] -> assign_location(ray.get_vector(d));
            photon_list[i] -> assign_color(light_source_color);
            photon_list[i] -> assign_direction(rec_medium.coming_ray.dir);
            photon_list[i] -> undeclare_on_surface();
            return;
          }
          new_scattering_dir = medium -> get_new_scattering_direction(
            ray.dir, &local_rand_state);
          d = medium -> get_propagation_distance(&local_rand_state);
          ray = Ray(ray.get_vector(d), new_scattering_dir);
          prev_location = rec_medium.point;
          hit = traverse_bvh(geom_node_list[0], ray, rec_medium);
        }

	direct_check_surface = true;

        if (scattered_in_medium_now) {
          rec = rec_medium;
        }
      }
      
      if(!scattered_in_medium_now && num_bounce > 0) {
        random_number = curand_uniform(&local_rand_state);
        reflection_prob = max(ref.k);
        if (random_number > reflection_prob) {
          if (ref.diffuse && (num_bounce > 1 || scattered_in_medium)) {
            photon_list[i] -> assign_prev_location(prev_location);
            photon_list[i] -> assign_location(rec.point);
            photon_list[i] -> assign_color(light_source_color);
            photon_list[i] -> assign_direction(rec.coming_ray.dir);
            photon_list[i] -> declare_on_surface();
          }
          return;  
        } else {
          light_source_color = ref.k * light_source_color;
          mean_color_tmp = (
            light_source_color.r() + light_source_color.g() + 
            light_source_color.b()
          ) / 3;
          light_source_color *= (mean_color / mean_color_tmp);
	  direct_check_surface = false;
        }	
      }
    } 

    if (!scattered_in_medium_now) {
      ray = ref.ray;
      prev_location = rec.point;
      hit = traverse_bvh(geom_node_list[0], ray, rec);
    }
    
  }
}

#endif
