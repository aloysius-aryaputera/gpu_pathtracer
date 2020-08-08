//File: ray_tracing_pass.h
#ifndef RAY_TRACING_PASS_H
#define RAY_TRACING_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/camera.h"
#include "../../model/material/material.h"
#include "../../model/point/ppm_hit_point.h"
#include "../../model/ray/ray.h"
#include "../../model/vector_and_matrix/vec3.h"
#include "../material_list_operations.h"

__device__
void _get_hit_point_details(
  reflection_record &ref, hit_record &rec, vec3 &filter,
  Camera **camera, int pixel_width_index, int pixel_height_index,
  Node **geom_node_list, int max_bounce,
  float camera_width_offset, float camera_height_offset, 
  bool& hit, curandState *rand_state 
) {
  
  bool sss = false;
  Ray ray;
  Material* material_list[400];
  int material_list_length = 0, num_bounce = 0;

  add_new_material(material_list, material_list_length, nullptr);
  ray = camera[0] -> compute_ray(
    pixel_width_index + camera_width_offset, 
    pixel_height_index + camera_height_offset, rand_state);
  hit = false;
  rec.object = nullptr;
  ref.diffuse = false;
  filter = vec3(1.0, 1.0, 1.0);

  hit = traverse_bvh(geom_node_list[0], ray, rec);
  
  if (hit) {
    while (!(ref.diffuse) && hit && num_bounce < max_bounce) {
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
        filter *= ref.filter_2;
      } 

      ray = ref.ray;
      hit = traverse_bvh(geom_node_list[0], ray, rec);
    }
  }
}

__global__
void ray_tracing_pass(
  PPMHitPoint** hit_point_list, Camera **camera, curandState *rand_state,
  Node **geom_node_list, bool init, int max_bounce
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
  int pixel_index = i * (camera[0] -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];
  float radius, radius_tmp, camera_width_offset[4] = {0, 0, 1, 1}, \
    camera_height_offset[4] = {0, 1, 0, 1};

  if (init) {
    radius = INFINITY;
    hit_point_list[pixel_index] = new PPMHitPoint(
     vec3(INFINITY, INFINITY, INFINITY), radius, filter, vec3(0, 0, 1)  
    ); 
  }

  radius = hit_point_list[pixel_index] -> current_photon_radius;
  _get_hit_point_details(
    ref, rec, filter, camera, i, j, geom_node_list, max_bounce, 0.5, 0.5, hit, 
    &local_rand_state
  );

  if (init) {
    for (int idx = 0; idx < 4; idx++) {
      _get_hit_point_details(
        ref_2, rec_2, filter, camera, i, j, geom_node_list, max_bounce, 
	camera_width_offset[idx], camera_height_offset[idx], hit_2, 
        &local_rand_state
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
  }

  if (hit && ref.diffuse) {
    hit_point_list[pixel_index] -> update_parameters(
      rec.point, radius, filter, vec3(0, 0, 1) 
    );
  } else {
    hit_point_list[pixel_index] -> update_parameters(
      vec3(INFINITY, INFINITY, INFINITY), radius, filter, vec3(0, 0, 1)
    );
  }

}

#endif
