//File: bvh_build.h
#ifndef BVH_BUILD_H
#define BVH_BUILD_H

#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/grid.h"

__global__ void compute_morton_code_batch(
  Primitive **object_array, Grid **grid, int num_triangles
);

__device__ bool morton_code_smaller(Primitive* obj_1, Primitive* obj_2);

__global__ void compute_morton_code_batch(
  Primitive **object_array, Grid **grid, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles) return;

  object_array[idx] -> get_bounding_box() -> compute_normalized_center(
    grid[0] -> world_bounding_box
  );
  object_array[idx] -> get_bounding_box() -> compute_bb_morton_3d();
}

__device__ bool morton_code_smaller(Primitive* obj_1, Primitive* obj_2) {
  return (
    obj_1 -> get_bounding_box() -> morton_code <
      obj_2 -> get_bounding_box() -> morton_code
  );
}

#endif
