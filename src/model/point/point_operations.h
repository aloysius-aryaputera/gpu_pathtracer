//File: point_operations.h
#ifndef POINT_OPERATIONS_H
#define POINT_OPERATIONS_H

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

  if (idx >= num_pts) return;

  vec3 direction = unit_vector(point_array[idx] -> location - camera[0] -> eye);
  int i = camera[0] -> compute_i(direction);
  int j = camera[0] -> compute_j(direction);
  int pixel_index = i * (camera[0] -> width) + j;

  if (pixel_index < (camera[0] -> width * camera[0] -> height))
    fb[pixel_index] = vec3(1, 1, 1);

}

#endif
