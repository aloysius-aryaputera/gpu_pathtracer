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

#endif
