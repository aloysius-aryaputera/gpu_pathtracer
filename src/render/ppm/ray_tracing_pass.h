//File: ray_tracing_pass.h
#ifndef RAY_TRACING_PASS_H
#define RAY_TRACING_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/bvh/bvh_traversal.h"
#include "../../model/camera.h"
#include "../../model/material/material.h"
#include "../../model/point/ppm_hit_point.h"
#include "../../model/ray/ray.h"
#include "../../model/vector_and_matrix/vec3.h"
#include "../material_list_operations.h"

__device__
void _get_hit_point_details(
  reflection_record &ref, hit_record &rec, vec3 &filter,
  vec3 &direct_radiance,
  Camera **camera, int pixel_width_index, int pixel_height_index,
  Node **geom_node_list, int max_bounce,
  float camera_width_offset, float camera_height_offset, 
  bool& hit, 
  Primitive** target_geom_array,
  int num_target_geom,
  Node **target_node_list,
  Node **target_leaf_list,
  int num_light_source_sampling,
  curandState *rand_state,
  int pixel_idx 
) {
  
  hit_record rec_2, rec_3;
  reflection_record ref_2, ref_3;
  bool sss = false;
  Ray ray;
  Material* material_list[400];
  int material_list_length = 0, num_bounce = 0;
  float factor;
  int pixel_index = pixel_height_index * (camera[0] -> width) + pixel_width_index;
  vec3 emittance = vec3(0.0, 0.0, 0.0), filter_lag = vec3(1.0, 1.0, 1.0);
  direct_radiance = vec3(0.0, 0.0, 0.0);
  vec3 add_direct_radiance;
  bool write;

  if (pixel_index == 8864)
    write = true;
  else
    write = false;

  max_bounce = 64;
  add_new_material(material_list, material_list_length, nullptr);
  ray = camera[0] -> compute_ray(
    pixel_height_index + camera_height_offset, 
    pixel_width_index + camera_width_offset, rand_state);
  hit = false;
  rec.object = nullptr;
  ref.diffuse = false;
  filter = vec3(1.0, 1.0, 1.0);

  //if (pixel_index == 0) {
  //  for (int idx_2 = 0; idx_2 < num_target_geom; idx_2++) {
  //    printf("light source %d = (%f, %f, %f), (%f, %f, %f), (%f, %f, %f).\n",
  //      idx_2, 
  //      target_leaf_list[idx_2] -> object -> get_material() -> get_texture_emission(vec3(0, 0, 0)).r(), 
  //      target_leaf_list[idx_2] -> object -> get_material() -> get_texture_emission(vec3(0, 0, 0)).g(), 
  //      target_leaf_list[idx_2] -> object -> get_material() -> get_texture_emission(vec3(0, 0, 0)).b(),
  //     target_geom_array[idx_2] -> get_material() -> get_texture_emission(vec3(0, 0, 0)).r(),
  //     target_geom_array[idx_2] -> get_material() -> get_texture_emission(vec3(0, 0, 0)).g(),
  //     target_geom_array[idx_2] -> get_material() -> get_texture_emission(vec3(0, 0, 0)).b(),
  //     target_geom_array[idx_2] -> get_fixed_normal().x(),
  //     target_geom_array[idx_2] -> get_fixed_normal().y(),
  //     target_geom_array[idx_2] -> get_fixed_normal().z()
  //      );
  //  }
  //}

  hit = traverse_bvh(geom_node_list[0], ray, rec);
  
  if (hit) {
    while (!(ref.diffuse) && hit && num_bounce < max_bounce) {
      num_bounce += 1;

      rec.object -> get_material() -> check_next_path(
        rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
        sss, material_list, material_list_length,
        ref, rand_state, write
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
	if (pixel_index == 8864) {
	  printf("pixel %d has filter = (%f, %f, %f) and filter_2 = (%f, %f, %f),\npoint = (%f, %f, %f), coming_dir = (%f, %f, %f), normal = (%f, %f, %f),\ndiffuse = %d, reflected = %d, refracted = %d, random_number = %f\n\n",
			  pixel_index,
			  filter.r(), filter.g(), filter.b(),
			  ref.filter_2.r(), ref.filter_2.g(), ref.filter_2.b(),
			  rec.point.x(), rec.point.y(), rec.point.z(),
			  rec.coming_ray.dir.x(), rec.coming_ray.dir.y(), rec.coming_ray.dir.z(),
			  rec.normal.x(), rec.normal.y(), rec.normal.z(),
			  ref.diffuse, ref.reflected, ref.refracted,
			  curand_uniform(&rand_state[0])
			  );
	}
	filter_lag = filter;
        filter *= ref.filter_2;
      }

      if (ref.diffuse) {
        emittance = filter_lag * rec.object -> get_material(
        ) -> get_texture_emission(rec.uv_vector);
	
        rec.object -> get_material() -> check_next_path(
          rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
          sss, material_list, material_list_length,
          ref_2, rand_state
        );
        for (int idx = 0; idx < num_light_source_sampling; idx++) {
	  factor = 1;

          //if (pixel_index == 27785) {
	  //if (pixel_index == 25310) {
	  //  write = true;
	  //} else {
	  //  write = false;
	  //}

	  change_ref_ray(
	    rec, 
	    ref_2, 
	    target_geom_array, 
	    num_target_geom,
            factor,
            target_node_list,
	    target_leaf_list,
	    1,
	    rand_state,
	    false
	  );
	  ray = ref_2.ray;
	  hit = traverse_bvh(geom_node_list[0], ray, rec_2);

          //if (pixel_index == 27785) {
          //if (pixel_index == 25310) {
	  //  printf("ray dir for pixel_index %d = (%f, %f, %f), hit = %d, factor = %f.\n", 
	  //		    pixel_idx, 
	  //		    ray.dir.x(), ray.dir.y(), ray.dir.z(),
	  //		    hit, factor); 
	  //}

	  if (hit) {
	    rec_2.object -> get_material() -> check_next_path(
	      rec_2.coming_ray, rec_2.point, rec_2.normal, rec_2.uv_vector,
	      sss, material_list, material_list_length, ref_3, rand_state
	    );
	    add_direct_radiance = (filter_lag * ref_2.filter * clamp(0, .9999, factor)
	    ) * rec_2.object -> get_material() -> get_texture_emission(
	      rec_2.uv_vector
            );
	    direct_radiance += add_direct_radiance;
	  }
	}
        direct_radiance /= max(1.0, float(num_light_source_sampling));	
	direct_radiance += emittance;

      }

      if (!(ref.diffuse)) {
        ray = ref.ray;
        hit = traverse_bvh(geom_node_list[0], ray, rec);
      }
    }
  }
}

__global__
void compute_average_radius(
  PPMHitPoint** hit_point_list, int num_hit_points, float *average_radius
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i > 0) return;

  float current_radius;
  int num_valid = 0;
  average_radius[0] = 0;
  for (int idx = 0; idx < num_hit_points; idx++) {
    current_radius = hit_point_list[idx] -> current_photon_radius;
    if (!isinf(current_radius)) {
      num_valid++;
      average_radius[0] += current_radius;
    }
  }
  average_radius[0] /= num_valid; 
}

__global__
void compute_max_radius(
  PPMHitPoint** hit_point_list, int num_hit_points, float *max_radius
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i > 0) return;

  float current_radius;
  max_radius[0] = 0;
  for (int idx = 0; idx < num_hit_points; idx++) {
    current_radius = hit_point_list[idx] -> current_photon_radius;
    if (!isinf(current_radius) && max_radius[0] < current_radius) {
      max_radius[0] = current_radius;
    }
  } 
}

__global__
void assign_radius_to_invalid_hit_points(
  PPMHitPoint** hit_point_list, int num_hit_points, float new_radius
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_hit_points) return;

  float current_radius;
  current_radius = hit_point_list[i] -> current_photon_radius;
  if(isinf(current_radius)) {
    hit_point_list[i] -> update_radius(new_radius);
  }
  //hit_point_list[i] -> update_radius(new_radius);
}

__global__
void ray_tracing_pass(
  PPMHitPoint** hit_point_list, Camera **camera, curandState *rand_state,
  Node **geom_node_list, bool init, int max_bounce, float ppm_alpha,
  int pass_iteration, int num_target_geom, Primitive** target_geom_list,
  Node** target_node_list, Node** target_leaf_list, int sample_size,
  float radius_multiplier
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  bool hit = false, hit_2 = false, write = false;
  hit_record rec, rec_2;
  reflection_record ref, ref_2;
  Ray ray;
  vec3 filter = vec3(1.0, 1.0, 1.0), hit_loc[4];
  vec3 direct_radiance = vec3(0.0, 0.0, 0.0), direct_radiance_dummy;
  int pixel_index = i * (camera[0] -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];
  float radius, radius_tmp, camera_width_offset[4] = {0, 0, 1, 1}, \
    camera_height_offset[4] = {0, 1, 0, 1};

  for (int idx = 0; idx < pass_iteration; idx++) {
    curand_uniform(&local_rand_state);
  }

  if (init) {
    radius = INFINITY;
    hit_point_list[pixel_index] = new PPMHitPoint(
     vec3(INFINITY, INFINITY, INFINITY), radius, filter, vec3(0, 0, 1),
     ppm_alpha  
    ); 
  }

  radius = hit_point_list[pixel_index] -> current_photon_radius;
  _get_hit_point_details(
    ref, rec, filter, direct_radiance,
    camera, j, i, geom_node_list, max_bounce, 0.5, 0.5, hit, 
    target_geom_list, num_target_geom, target_node_list, target_leaf_list,
    sample_size, &local_rand_state, pixel_index
  );

  if (init) {
    for (int idx = 0; idx < 4; idx++) {
      _get_hit_point_details(
        ref_2, rec_2, filter, direct_radiance_dummy,
	camera, j, i, geom_node_list, max_bounce, 
	camera_width_offset[idx], camera_height_offset[idx], hit_2,
        target_geom_list, num_target_geom, target_node_list, target_leaf_list,
        0, &local_rand_state, pixel_index
      );
      if (hit_2 && ref_2.diffuse) {
        hit_loc[idx] = rec_2.point;
      } else {
        hit_loc[idx] = vec3(INFINITY, INFINITY, INFINITY);
      }
    }

    for (int idx = 0; idx < 4; idx++) {
      if (
        !(hit_loc[idx].vector_is_inf()) && !(rec.point.vector_is_inf()) &&
	hit && ref.diffuse
      ) {
        radius_tmp = compute_distance(hit_loc[idx], rec.point);
	if (radius > radius_tmp) {
	  radius = radius_tmp;
	}
      }
    }
    radius *= radius_multiplier;
  }

  if (hit && ref.diffuse) {
    hit_point_list[pixel_index] -> update_parameters(
      rec.point, radius, filter, rec.normal 
    );
    hit_point_list[pixel_index] -> update_direct_radiance(direct_radiance);
  } else {
    hit_point_list[pixel_index] -> update_parameters(
      vec3(INFINITY, INFINITY, INFINITY), radius, filter, vec3(0, 0, 1)
    );
  }

}

#endif
