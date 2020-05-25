//File: bounding_box_operations.h
#ifndef BOUNDING_BOX_OPERATIONS_H
#define BOUNDING_BOX_OPERATIONS_H

#include <cuda_fp16.h>

#include "../../param.h"
#include "../geometry/primitive.h"
#include "bounding_box.h"

__global__ void compute_world_bounding_box(
  BoundingBox **world_bounding_box, Primitive **geom_array, int num_objects
) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {

    float x_min = INFINITY;
    float x_max = -INFINITY;
    float y_min = INFINITY;
    float y_max = -INFINITY;
    float z_min = INFINITY;
    float z_max = -INFINITY;

    for (int i = 0; i < num_objects; i++) {
      x_min = min(x_min, geom_array[i] -> get_bounding_box() -> x_min);
      x_max = max(x_max, geom_array[i] -> get_bounding_box() -> x_max);
      y_min = min(y_min, geom_array[i] -> get_bounding_box() -> y_min);
      y_max = max(y_max, geom_array[i] -> get_bounding_box() -> y_max);
      z_min = min(z_min, geom_array[i] -> get_bounding_box() -> z_min);
      z_max = max(z_max, geom_array[i] -> get_bounding_box() -> z_max);
    }

    x_min -= SMALL_DOUBLE;
    x_max += SMALL_DOUBLE;
    y_min -= SMALL_DOUBLE;
    y_max += SMALL_DOUBLE;
    z_min -= SMALL_DOUBLE;
    z_max += SMALL_DOUBLE;

    world_bounding_box[0] = new BoundingBox(
      x_min, x_max, y_min, y_max, z_min, z_max);
  }

}

#endif
