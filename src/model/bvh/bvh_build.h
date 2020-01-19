//File: bvh_build.h
#ifndef BVH_BUILD_H
#define BVH_BUILD_H

#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/grid.h"

__global__ void compute_normalized_center(
  Primitive **object_array, Grid **grid, int num_triangles
);

__global__ void compute_normalized_center(
  Primitive **object_array, Grid **grid, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles) return;

  object_array[idx] -> get_bounding_box() -> compute_normalized_center(
    grid[0] -> world_bounding_box
  );
}

#endif
