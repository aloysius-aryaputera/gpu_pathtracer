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
#include "../material_list_operations.h"

__global__
void create_photon_list(Point **photon_list, int num_photons) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  photon_list[i] = new Point(
    vec3(INFINITY, INFINITY, INFINITY), vec3(1, 1, 1), vec3(0, 0, 1), 0
  ); 
}

__global__
void photon_pass(
  Primitive **target_geom_list, Node **geom_node_list, int num_photons,
  int max_bounce, curandState *rand_state
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_photons) return;

  hit_record rec;
  reflection_record ref;
  Material* material_list[400];
  int num_bounce = 0, material_list_length = 0;
  Ray ray;
  bool hit = false, sss = false;
  curandState local_rand_state = rand_state[i];

  //if (hit) {
  //  while (!(ref.diffuse) && hit && num_bounce < max_bounce) {
  //    num_bounce += 1;

  //    rec.object -> get_material() -> check_next_path(
  //      rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
  //      sss, material_list, material_list_length,
  //      ref, rand_state
  //    );

  //    if (ref.false_hit && ref.entering)
  //      add_new_material(
  //        material_list, material_list_length, rec.object -> get_material()
  //      );

  //    if (ref.false_hit && !(ref.entering))
  //      remove_a_material(
  //        material_list, material_list_length, rec.object -> get_material()
  //      );

  //    if (!(ref.false_hit) && ref.refracted && ref.entering)
  //      add_new_material(
  //        material_list, material_list_length, rec.object -> get_material()
  //      );

  //    if (!(ref.false_hit) && ref.refracted && !(ref.entering))
  //      remove_a_material(
  //        material_list, material_list_length, rec.object -> get_material()
  //      );
  //   
  //    if (!(ref.false_hit)) {
  //      filter *= ref.filter_2;
  //    } 

  //    ray = ref.ray;
  //    hit = traverse_bvh(geom_node_list[0], ray, rec);
  //  }
  //}
}

#endif
