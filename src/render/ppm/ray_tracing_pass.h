//File: ray_tracing_pass.h
#ifndef RAY_TRACING_PASS_H
#define RAY_TRACING_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh/bvh.h"
#include "../../model/camera.h"
#include "../../model/material/material.h"
#include "../../model/point/point.h"
#include "../../model/ray/ray.h"
#include "../material_list_operations.h"

__global__
void ray_tracing_pass(
  Point** hit_point_list, Camera **camera, curandState *rand_state,
  Node **geom_node_list
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  bool hit = false, sss = false;
  hit_record rec;
  reflection_record ref;
  Ray ray;
  Material* material_list[400];
  int pixel_index = i * (camera[0] -> width) + j, material_list_length = 0;
  curandState local_rand_state = rand_state[pixel_index];

  add_new_material(material_list, material_list_length, nullptr);
  ray = camera[0] -> compute_ray(i + .5, j + .5, &local_rand_state);
  rec.object = nullptr;

  hit = traverse_bvh(geom_node_list[0], ray, rec);
  
  if (hit) {
    rec.object -> get_material() -> check_next_path(
      rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
      sss, material_list, material_list_length,
      ref, &local_rand_state
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
  }

}

#endif
