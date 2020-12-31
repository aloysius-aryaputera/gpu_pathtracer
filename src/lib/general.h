//File: material_list_operations.h
#ifndef MATERIAL_LIST_OPERATIONS_H
#define MATERIAL_LIST_OPERATIONS_H

#include "../model/geometry/primitive.h"

__device__ void compute_num_transparent_geom(
  Primitive** primitive_array, int num_primitives,
  int &num_transparent_primitives
) {
  num_transparent_primitives = 0;

  for (int i = 0; i < num_primitives; i++) {
    
  }
}

#endif
