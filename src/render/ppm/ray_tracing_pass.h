//File: ray_tracing_pass.h
#ifndef RAY_TRACING_PASS_H
#define RAY_TRACING_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/bvh/bvh_traversal.h"
#include "../../model/bvh/bvh_traversal_photon.h"
#include "../../model/camera.h"
#include "../../model/grid/bounding_cylinder.h"
#include "../../model/material/material.h"
#include "../../model/point/ppm_hit_point.h"
#include "../../model/ray/ray.h"
#include "../../model/vector_and_matrix/vec3.h"
#include "../material_list_operations.h"
#include "common.h"

__device__ vec3 _compute_direct_radiance(
  vec3 filter,
  hit_record rec,
  reflection_record ref,
  float &pdf,
  Primitive** target_geom_array,
  int num_target_geom,
  Node** geom_node_list,
  Node** target_node_list,
  Node** target_leaf_list,
  bool in_medium,
  Material* medium,
  Material** material_list,
  int &material_list_length,
  curandState *rand_state
) {

  float factor = 1;
  vec3 emittance = filter * rec.object -> get_material(
  ) -> get_texture_emission(rec.uv_vector);
  vec3 direct_radiance, prev_hit_point;
  Ray ray;
  bool sss, hit, prev_in_medium = in_medium, flag = true, force_refract = true;
  Material *prev_medium = medium;

  change_ref_ray(
    rec, 
    ref, 
    target_geom_array, 
    num_target_geom,
    factor,
    target_node_list,
    target_leaf_list,
    1,
    rand_state,
    false
  );

  //ray = ref.ray;
  //hit = traverse_bvh(geom_node_list[0], ray, rec);
  //
  //while (
  //  hit && rec.object -> get_material() -> n_i < SMALL_DOUBLE &&
  //  rec.object -> get_material() -> extinction_coef >= SMALL_DOUBLE
  //) {
  //  if (dot(rec.normal, ray.dir) >= 0) {
  //    transmittance *= rec.object -> get_material(
  //    ) -> get_transmittance(rec.t);
  //  }
  //  ray = Ray(rec.point, ray.dir);
  //  hit = traverse_bvh(geom_node_list[0], ray, rec);
  //}
  //
  //if (hit) {
  //  direct_radiance += (
  //    filter * ref.filter * transmittance * clamp(0, .999999, factor)
  //  ) * rec.object -> get_material() -> get_texture_emission(
  //    rec.uv_vector
  //  );
  //}

  //direct_radiance += emittance;
  //return direct_radiance;
 
  filter *= ref.filter;
  ray = ref.ray;
  prev_hit_point = rec.point;

  hit = traverse_bvh(geom_node_list[0], ray, rec);

  while(flag && hit) {
     rec.object -> get_material() -> check_next_path(
       rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
       sss, material_list, material_list_length, ref, rand_state, false,
       force_refract
     );

     rearrange_material_list(
       material_list, material_list_length, rec.object -> get_material(),
       ref.false_hit, ref.entering, ref.refracted 
     );
  
     in_medium = check_if_entering_medium(ref, in_medium, medium);

     if (!ref.false_hit && prev_in_medium) {
       vec3 dir = rec.point - prev_hit_point;
       float l = dir.length();
       float transmittance = prev_medium -> get_transmittance(l);
       filter *= transmittance;
     }

     filter *= ref.filter_2;

     if (!ref.false_hit && ref.diffuse) {
       direct_radiance += (
         filter * clamp(0, .999999, factor)
       ) * rec.object -> get_material() -> get_texture_emission(
         rec.uv_vector
       );
     }

     if (!ref.refracted || ref.bent) {
       flag = false;
     }

     if (!ref.false_hit) {
       prev_hit_point = rec.point;
       prev_in_medium = in_medium;
       prev_medium = medium;
     }

     ray = ref.ray;
     hit = traverse_bvh(geom_node_list[0], ray, rec);
  }

  direct_radiance += emittance;
  direct_radiance = vec3(
    min(1.0, direct_radiance.r()),
    min(1.0, direct_radiance.g()),
    min(1.0, direct_radiance.b())
  );
  return direct_radiance; 
  //return vec3(0.9, 0.3, 0.3);
}

__device__ reflection_record _set_init_ref(
  Material** material_list, int material_list_length, Ray init_ray
) {
  Material* highest_prioritised_material = nullptr;
  Material* second_highest_prioritised_material = nullptr;

  find_highest_prioritised_materials(
    material_list, material_list_length,
    highest_prioritised_material,
    second_highest_prioritised_material
  );

  reflection_record ref;
  ref.false_hit = false;
  ref.reflected = true;
  ref.refracted = false;
  ref.ray = init_ray;
  ref.filter = vec3(1.0, 1.0, 1.0);
  ref.filter_2 = vec3(1.0, 1.0, 1.0);
  ref.pdf = 1;
  ref.diffuse = false;
  ref.bent = false;
  ref.entering = true;
  ref.next_material = highest_prioritised_material;
  return ref;
}

__device__
void _get_hit_point_details(
  PPMHitPoint* hit_point,
  Node **volume_photon_node_list,
  reflection_record &ref, 
  hit_record &rec, 
  vec3 &filter, 
  float &pdf,
  vec3 &direct_radiance,
  Camera **camera, 
  int pixel_width_index, 
  int pixel_height_index,
  Node **geom_node_list, 
  int max_bounce,
  float camera_width_offset, 
  float camera_height_offset, 
  bool& hit, 
  Primitive** target_geom_array,
  int num_target_geom,
  Node **target_node_list,
  Node **target_leaf_list,
  Node **transparent_node_list,
  curandState *rand_state,
  int pixel_idx,
  bool init
) {
 
  hit_record rec_2, rec_3;
  reflection_record ref_2;
  bool sss = false, in_medium = false, prev_in_medium = false;
  Ray ray;
  Material* material_list[400], *medium = nullptr, *prev_medium = nullptr;
  int material_list_length = 0, num_bounce = 0;
  float factor, pdf_lag = 1, transmittance;
  vec3 emittance = vec3(0.0, 0.0, 0.0), filter_lag = vec3(1.0, 1.0, 1.0);
  direct_radiance = vec3(0.0, 0.0, 0.0);
  vec3 add_direct_radiance, prev_hit_point, start_point;
  bool write = false;

  max_bounce = 64;
  add_new_material(material_list, material_list_length, nullptr);

  ray = camera[0] -> compute_ray(
    pixel_height_index + camera_height_offset, 
    pixel_width_index + camera_width_offset, rand_state);
  hit = false;
  rec.object = nullptr;
  ref.diffuse = false;
  filter = vec3(1.0, 1.0, 1.0);
  pdf = 1.0;

  init_material_list(
    material_list, material_list_length, transparent_node_list, ray.p0,
    vec3(0, 0, 1), rand_state
  );
  ref = _set_init_ref(material_list, material_list_length, ray);
  in_medium = check_if_entering_medium(ref, in_medium, medium);

  if (!in_medium) {
    //printf("in_medium = %d; material_list_length = %d, ray.p0 = (%5.2f, %5.2f, %5.2f), ray.dir = (%5.2f, %5.2f, %5.2f)\n", 
      //in_medium, material_list_length, ray.p0.x(), ray.p0.y(), ray.p0.z(), ray.dir.x(), ray.dir.y(), ray.dir.z());
  }

  prev_in_medium = in_medium;
  prev_medium = medium;
  prev_hit_point = ray.p0;

  hit = traverse_bvh(geom_node_list[0], ray, rec);

  if (hit) {
    while (!(ref.diffuse) && hit && num_bounce < max_bounce) {
      num_bounce += 1;

      rec.object -> get_material() -> check_next_path(
        rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
        sss, material_list, material_list_length,
        ref, rand_state, write
      );

      rearrange_material_list(
        material_list, material_list_length, rec.object -> get_material(),
        ref.false_hit, ref.entering, ref.refracted 
      );
  
      in_medium = check_if_entering_medium(ref, in_medium, medium);

      //if (num_bounce == 1) {
      //  prev_in_medium = in_medium;
      // prev_medium = medium;
      //}

      if (!(ref.false_hit) && prev_in_medium && !init) {

	vec3 dir = rec.point - prev_hit_point;
	float l = dir.length();
        hit_point -> update_bounding_cylinder_parameters(
	  prev_hit_point, dir, l
	);
        traverse_bvh_volume_photon(
	  volume_photon_node_list[0], hit_point, medium, filter
	);

	if (hit_point -> tmp_accummulated_lm.vector_is_inf() || 
	  hit_point -> tmp_accummulated_lm.r() < 0 ||
	  hit_point -> tmp_accummulated_lm.g() < 0 ||
	  hit_point -> tmp_accummulated_lm.b() < 0
	) {
	  printf("wi = %d, hi = %d, false hit = %d, entering = %d, refracted = %d, reflected = %d, diffuse = %d, material list length = %d\n",
	    pixel_width_index, pixel_height_index,
	    ref.false_hit, ref.entering, ref.refracted, ref.reflected, ref.diffuse,
	    material_list_length
	  );
	} 

	transmittance = prev_medium -> get_transmittance(l);
	filter *= transmittance;
      } else {
        //if (!(ref.false_hit) && num_bounce < 2 && !init)
	  //printf("Not in medium while num_bounce < 2!\n");
      }

      if (!(ref.false_hit)) {
	filter_lag = filter;
        filter *= ref.filter_2;
	pdf_lag = pdf;
	pdf *= ref.pdf;
      }

      if (ref.diffuse) {
	pdf = pdf_lag;
        direct_radiance += _compute_direct_radiance(
	  filter_lag,
	  rec,
	  ref,
	  pdf,
          target_geom_array,
	  num_target_geom,
	  geom_node_list,
	  target_node_list,
	  target_leaf_list,
	  in_medium,
	  medium,
	  material_list, 
	  material_list_length,
	  rand_state
	);

	if (direct_radiance.vector_is_inf() || 
	  direct_radiance.r() < 0 ||
	  direct_radiance.g() < 0 ||
	  direct_radiance.b() < 0
	) {
	  printf("wi = %d, hi = %d, false hit = %d, entering = %d, refracted = %d, reflected = %d, diffuse = %d, material list length = %d\n",
	    pixel_width_index, pixel_height_index,
	    ref.false_hit, ref.entering, ref.refracted, ref.reflected, ref.diffuse,
	    material_list_length
	  );
	}

        //emittance = filter_lag * rec.object -> get_material(
        //) -> get_texture_emission(rec.uv_vector);
	//
	//ref_2 = ref;
	//pdf = pdf_lag;
        ////for (int idx = 0; idx < num_light_source_sampling; idx++) {
	//factor = 1;
	//transmittance = 1;

	//change_ref_ray(
	//  rec, 
	//  ref_2, 
	//  target_geom_array, 
	//  num_target_geom,
        //  factor,
        //  target_node_list,
	//  target_leaf_list,
	//  1,
	//  rand_state,
	//  false
	//);
	//ray = ref_2.ray;
	//hit = traverse_bvh(geom_node_list[0], ray, rec_2);

        //while (
	//  hit && rec_2.object -> get_material() -> n_i < SMALL_DOUBLE &&
	//  rec_2.object -> get_material() -> extinction_coef >= SMALL_DOUBLE
	//) {
	//  if (dot(rec_2.normal, ray.dir) >= 0) {
	//    transmittance *= rec_2.object -> get_material(
	//    ) -> get_transmittance(rec_2.t);
	//  }
	//  ray = Ray(rec_2.point, ray.dir);
	//  hit = traverse_bvh(geom_node_list[0], ray, rec_2);
	//}

	//if (hit) {
	//  add_direct_radiance = (
	//    filter_lag * ref_2.filter * transmittance * clamp(0, .999999, factor)
	//  ) * rec_2.object -> get_material() -> get_texture_emission(
	//    rec_2.uv_vector
        //  );
	//  direct_radiance += add_direct_radiance;
	//}
	////}
        ////direct_radiance /= max(1.0, float(num_light_source_sampling));	
	//direct_radiance += emittance;

	return;
      }

      if (!(ref.false_hit)) {
        prev_in_medium = in_medium;
	prev_hit_point = rec.point;
	prev_medium = medium;
      }

      ray = ref.ray;
      hit = traverse_bvh(geom_node_list[0], ray, rec);

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
    current_radius = hit_point_list[idx] -> surface_radius;
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
    current_radius = hit_point_list[idx] -> surface_radius;
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

  //float current_radius;
  //current_radius = hit_point_list[i] -> current_photon_radius;
  //if(
  //  isinf(current_radius) || current_radius > new_radius ||
  //  isnan(current_radius)
  //) {
  //  hit_point_list[i] -> update_radius(new_radius);
  //}
  hit_point_list[i] -> update_radius(new_radius);
}

//__global__
//void compute_radius(
//  PPMHitPoint** hit_point_list, Camera **camera, float radius_scaling_factor
//) {
//  int j = threadIdx.x + blockIdx.x * blockDim.x;
//  int i = threadIdx.y + blockIdx.y * blockDim.y;
//
//  if (
//    (j >= camera[0] -> width - 1) || 
//    (i >= camera[0] -> height - 1) || 
//    (i == 0) || (j == 0)
//  ) return;
//
//  int pixel_index = i * (camera[0] -> width) + j;
//  int pixel_index_2 = (i - 1) * (camera[0] -> width) + j;
//  int pixel_index_3 = (i + 1) * (camera[0] -> width) + j;
//  int pixel_index_4 = i * (camera[0] -> width) + j + 1;
//  int pixel_index_5 = i * (camera[0] -> width) + j - 1;
//
//  float dist_1 = compute_distance(
//    hit_point_list[pixel_index] -> location, 
//    hit_point_list[pixel_index_2] -> location);
//  float dist_2 = compute_distance(
//    hit_point_list[pixel_index] -> location, 
//    hit_point_list[pixel_index_3] -> location);
//  float dist_3 = compute_distance(
//    hit_point_list[pixel_index] -> location, 
//    hit_point_list[pixel_index_4] -> location);
//  float dist_4 = compute_distance(
//    hit_point_list[pixel_index] -> location, 
//    hit_point_list[pixel_index_5] -> location);
//
//  float radius = radius_scaling_factor * (
//    dist_1 + dist_2 + dist_3 + dist_4) / 4;
//  hit_point_list[pixel_index] -> update_radius(radius);
//  
//}

__global__
void ray_tracing_pass(
  PPMHitPoint** hit_point_list, Camera **camera, curandState *rand_state,
  Node **volume_photon_node_list,
  Node **geom_node_list, bool init, int max_bounce, float ppm_alpha,
  int pass_iteration, int num_target_geom, Primitive** target_geom_list,
  Node** target_node_list, Node** target_leaf_list, 
  Node** transparent_node_list, int sample_size, float radius_multiplier
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  bool hit = false, hit_2 = false;
  hit_record rec, rec_2;
  reflection_record ref, ref_2;
  Ray ray;
  vec3 filter = vec3(1.0, 1.0, 1.0), hit_loc[4];
  vec3 direct_radiance = vec3(0.0, 0.0, 0.0), direct_radiance_dummy;
  int pixel_index = i * (camera[0] -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];
  float radius, radius_tmp, camera_width_offset[4] = {0, 0, 1, 1}, \
    camera_height_offset[4] = {0, 1, 0, 1}, pdf = 1.0;
  float main_camera_width_offset = .5, main_camera_height_offset = .5;

  for (int idx = 0; idx < pass_iteration; idx++) {
    curand_uniform(&local_rand_state);
  }

  if (init) {
    hit_point_list[pixel_index] = new PPMHitPoint(ppm_alpha); 
  } else {
    hit_point_list[pixel_index] -> reset_tmp_accummulated_lm();
    main_camera_height_offset = curand_uniform(&local_rand_state);
    main_camera_width_offset = curand_uniform(&local_rand_state);
  }

  PPMHitPoint *hit_point = hit_point_list[pixel_index];
  radius = hit_point -> surface_radius;

  _get_hit_point_details(
    hit_point, volume_photon_node_list,
    ref, rec, filter, pdf, direct_radiance,
    camera, j, i, geom_node_list, max_bounce, main_camera_width_offset, 
    main_camera_height_offset, hit, target_geom_list, num_target_geom, 
    target_node_list, target_leaf_list, transparent_node_list,
    &local_rand_state, pixel_index, init
  );

  if (init) {
    for (int idx = 0; idx < 4; idx++) {
      _get_hit_point_details(
        hit_point, volume_photon_node_list,
	ref_2, rec_2, filter, pdf, 
	direct_radiance_dummy,
        camera, j, i, geom_node_list, max_bounce, 
        camera_width_offset[idx], camera_height_offset[idx], hit_2,
        target_geom_list, num_target_geom, target_node_list, target_leaf_list,
        transparent_node_list, &local_rand_state, pixel_index, init
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
        if (radius > radius_tmp && radius_tmp > 0) {
          radius = radius_tmp;
        }
      }

      for (int idx_2 = idx; idx_2 < 4; idx_2++) {
        if(!(hit_loc[idx_2].vector_is_inf()) && !(hit_loc[idx].vector_is_inf())
        ) {
          radius_tmp = compute_distance(hit_loc[idx], hit_loc[idx_2]);
          if (radius > radius_tmp && radius_tmp > 0) {
            radius = radius_tmp;
          }
        }
      }

    }
    radius *= radius_multiplier;
  }

  if (hit && ref.diffuse) {
    hit_point -> update_parameters(rec.point, radius, filter, rec.normal, pdf);
    hit_point -> update_direct_radiance(direct_radiance);
  } else {
    hit_point -> update_parameters(
      vec3(INFINITY, INFINITY, INFINITY), radius, filter, vec3(0, 0, 1), pdf
    );
  }

}

#endif
