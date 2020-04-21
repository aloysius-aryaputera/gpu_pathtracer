//File: point_operations.h
#ifndef POINT_OPERATIONS_H
#define POINT_OPERATIONS_H

#include "../../util/image_util.h"
#include "../camera.h"
#include "../vector_and_matrix/vec3.h"
#include "point.h"

__global__ void create_point_image(
  vec3 *fb, Camera **camera, Point** point_array, int num_pts
);

__global__ void create_point_image(
  vec3 *fb, Camera **camera, Point** point_array, int num_pts
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // printf("Step 1\n");
  if (idx >= num_pts) return;

  // printf("Step 3\n");
  vec3 direction = unit_vector(
    point_array[idx] -> location - camera[0] -> eye);

  // printf("Step 4\n");
  int i = camera[0] -> compute_i(direction);
  int j = camera[0] -> compute_j(direction);

  // printf("Step 5\n");
  int pixel_index = i * (camera[0] -> width) + j;

  // printf("Step 6\n");
  if (
    pixel_index < (camera[0] -> width * camera[0] -> height) &&
    i >= 0 && j >= 0
  ) {
    fb[pixel_index] = point_array[idx] -> color;
  }

}

#endif
