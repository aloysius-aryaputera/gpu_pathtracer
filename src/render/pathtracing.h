#ifndef PATHTRACING_H
#define PATHTRACING_H

#include <math.h>

#include "../model/camera.h"
#include "../model/data_structure/local_vector.h"
#include "../model/geometry/triangle.h"
#include "../model/ray.h"

__global__ void render(float *fb, int max_x, int max_y);
__global__ void render(
  float* fb, Camera* camera, Triangle** geom_array
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
  float *fb, Camera *camera, Triangle **geom_array
) {
  if (
    threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0
  ) {
    print_vec3((*geom_array) -> normal);
  }
  camera = new Camera(
    vec3(0, -5, 0), vec3(0, 0, 0), vec3(0, 0, 1), 45, 100, 100
  );
  vec3 point_1 = vec3(0, 0, 0), point_2 = vec3(1, 1, 0), point_3 = vec3(1, 1, 1);
  Triangle* my_triangle = new Triangle(point_1, point_2, point_3);
  *geom_array = my_triangle;

  vec3 color;
  hit_record rec, best_rec;
  bool hit = false;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= camera -> width) || (j >= camera -> height)) return;

  best_rec.t = INFINITY;

  Ray camera_ray = camera -> compute_ray(i, j);
  for (int idx = 0; idx < 1; idx++) {
    hit = (*geom_array)[idx].hit(camera_ray, 0, INFINITY, rec);
    if (hit && rec.t < best_rec.t) {
      best_rec.t = rec.t;
      best_rec.point = vec3(rec.point.x(), rec.point.y(), rec.point.z());
      best_rec.normal = vec3(rec.normal.x(), rec.normal.y(), rec.normal.z());
    }
  }

  if (best_rec.t < INFINITY) {
    color = vec3(1, 1, 1);
  } else {
    color = vec3(0, 0, 0);
  }

  int pixel_index = j * camera -> width * 3 + i * 3;
  fb[pixel_index + 0] = color.r();
  fb[pixel_index + 1] = color.g();
  fb[pixel_index + 2] = color.b();
}

#endif
