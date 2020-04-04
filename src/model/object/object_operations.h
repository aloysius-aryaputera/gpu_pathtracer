//File: object_operations.h
#ifndef OBJECT_OPERATIONS_H
#define OBJECT_OPERATIONS_H

#include "object.h"

__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int *pt_offset_array,
  int num_objects, int num_pts_per_object
);

__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int *pt_offset_array,
  int num_objects, int num_pts_per_object
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  int offset = 0;
  num_sss_objects[0] = 0;
  for (int i = 0; i < num_objects; i++) {

    if (object_array[i] -> sub_surface_scattering)
      (num_sss_objects[0])++;

    if (i == 0) {
      pt_offset_array[i] = offset;
    } else {
      if (object_array[i - 1] -> sub_surface_scattering) {
        offset += num_pts_per_object;
      }
      pt_offset_array[i] = offset;
    }

  }
  printf("We have %d SSS objects.\n", num_sss_objects[0]);
}

__global__ void allocate_pts_sss(
  Object ** object_array, Point** point_array, int *pt_offset_array,
  int num_objects
) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  for (int i = 0; i < num_objects; i++) {
    object_array[i] -> allocate_point_array(point_array + pt_offset_array[i]);
  }
}

__global__ void create_sss_pts(
  Object** object, Primitive** geom_array, Point** point_array, int *pt_offset,
  curandState *rand_state, int object_idx, int num_pts_per_object
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (!object[object_idx] -> sub_surface_scattering) return;
  if (idx >= num_pts_per_object) return;

  int primitive_idx = (object[object_idx]) -> pick_primitive_idx_for_sampling(
    rand_state, idx);
  vec3 pts_loc = geom_array[primitive_idx] -> get_random_point_on_surface(
    rand_state);
  // vec3 pts_loc = vec3(0, 0, 0);
  point_array[pt_offset[object_idx] + idx] = new Point(pts_loc);
}

#endif
