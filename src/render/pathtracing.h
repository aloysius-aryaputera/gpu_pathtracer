#ifndef PATHTRACING_H
#define PATHTRACING_H

#include <math.h>

#include "../model/camera.h"
#include "../model/data_structure/local_vector.h"
#include "../model/geometry/triangle.h"
#include "../model/ray.h"

__global__ void render(float *fb, int max_x, int max_y);
__global__ void render(
  float* fb, Camera* camera, Triangle** geom_array, int num_triangles
);

__global__
void render(float *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x * 3 + i * 3;
  fb[pixel_index + 0] = float(i) / max_x;
  fb[pixel_index + 1] = float(j) / max_y;
  fb[pixel_index + 2] = 0.2;
}

__global__
void render(
  float *fb, Camera **camera, Triangle **geom_array, int num_triangles
) {

  vec3 color;
  hit_record rec, best_rec;
  bool hit = false;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if((i >= camera[0] -> width) || (j >= camera[0] -> height)) {
    return;
  }

  best_rec.t = INFINITY;

  Ray camera_ray = camera[0] -> compute_ray(i, j);
  for (int idx = 0; idx < num_triangles; idx++) {
    hit = (geom_array[idx]) -> hit(camera_ray, 0, INFINITY, rec);
    if (hit && rec.t < best_rec.t) {
      best_rec.t = rec.t;
      best_rec.point = vec3(rec.point.x(), rec.point.y(), rec.point.z());
      best_rec.normal = vec3(rec.normal.x(), rec.normal.y(), rec.normal.z());
    }
  }

  if (best_rec.t < INFINITY) {
    color = vec3(best_rec.t / 10.0, best_rec.t / 10.0, best_rec.t / 10.0);
  } else {
    color = vec3(0, 0, 0);
  }

  int pixel_index = j * (camera[0] -> width) * 3 + i * 3;
  fb[pixel_index + 0] = color.r();
  fb[pixel_index + 1] = color.g();
  fb[pixel_index + 2] = color.b();
}

#endif
