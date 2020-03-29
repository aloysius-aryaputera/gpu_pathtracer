//File: object_operations.h
#ifndef OBJECT_OPERATIONS_H
#define OBJECT_OPERATIONS_H

#include "object.h"

__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int num_objects
);

__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  num_sss_objects[0] = 0;
  for (int i = 0; i < num_objects; i++) {
    if (object_array[i] -> sub_surface_scattering)
      (num_sss_objects[0])++;
  }
  printf("We have %d SSS objects.\n", num_sss_objects[0]);
}

#endif
