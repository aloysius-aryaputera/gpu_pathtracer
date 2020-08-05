//File: ray_tracing_pass.h
#ifndef RAY_TRACING_PASS_H
#define RAY_TRACING_PASS_H

#include <curand_kernel.h>

#include "../../model/bvh.h"
#include "../../model/camera.h"
#include "../../model/material/material.h"
#include "../../model/point/point.h"
#include "../../model/ray/ray.h"

__global
void ray_tracing_pass(
  Point** hit_point_list, Camera **camera, curandState *rand_state,
  Node **geom_node_list
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
 
  if ((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  hit_record rec;
  Ray camera_ray;

  camera_ray = camera[0] -> compute_ray(i + .5, j + .5, &local_rand_state);
}

#endif
