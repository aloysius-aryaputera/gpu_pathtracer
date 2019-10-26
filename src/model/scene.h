//File: scene.h
#ifndef SCENE_H
#define SCENE_H

#include <cuda_fp16.h>
#include <math.h>

#include "../param.h"
#include "geometry/primitive.h"
#include "camera.h"
#include "grid/cell.h"
#include "grid/grid.h"

class Scene {

  private:
    __device__ void _compute_scene_boundaries();
    __device__ void _compute_grid_resolutions();

    float x_min, x_max, y_min, y_max, z_min, z_max, d_x, d_y, d_z, volume;

  public:
    __host__ __device__ Scene() {}
    __device__ Scene(
      Camera* camera_, Primitive** object_array_, int num_objects_);

    Camera *camera;
    Primitive** object_array;
    int num_objects, n_cell_x, n_cell_y, n_cell_z;
    Grid grid;

};

__device__ void Scene::_compute_grid_resolutions() {
  n_cell_x = d_x * powf(LAMBDA * num_objects / volume, 1 / 3);
  n_cell_y = d_y * powf(LAMBDA * num_objects / volume, 1 / 3);
  n_cell_z = d_z * powf(LAMBDA * num_objects / volume, 1 / 3);
}

__device__ void Scene::_compute_scene_boundaries() {
  x_min = camera -> eye.x();
  x_max = camera -> eye.x();
  y_min = camera -> eye.y();
  y_max = camera -> eye.y();
  z_min = camera -> eye.z();
  z_max = camera -> eye.z();

  for (int i = 0; i < num_objects; i++) {
    x_min = min(x_min, object_array[i] -> get_bounding_box() -> x_min);
    x_max = min(x_max, object_array[i] -> get_bounding_box() -> x_max);
    y_min = min(y_min, object_array[i] -> get_bounding_box() -> y_min);
    y_max = min(y_max, object_array[i] -> get_bounding_box() -> y_max);
    z_min = min(z_min, object_array[i] -> get_bounding_box() -> z_min);
    z_max = min(z_max, object_array[i] -> get_bounding_box() -> z_max);
  }

  d_x = x_max - x_min;
  d_y = y_max - y_min;
  d_z = z_max - z_min;
  volume = d_x * d_y * d_z;
}

__device__ Scene::Scene(
  Camera* camera_, Primitive** object_array_, int num_objects_
) {
  camera = camera_;
  object_array = object_array_;
}

#endif
