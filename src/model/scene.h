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

    float x_min, x_max, y_min, y_max, z_min, z_max, d_x, d_y, d_z, volume;

  public:
    __host__ __device__ Scene() {}
    __device__ Scene(Camera* camera_, Grid* grid_, int num_objects_);

    Camera *camera;
    int num_objects, n_cell_x, n_cell_y, n_cell_z;
    Grid *grid;

};

__device__ Scene::Scene(Camera* camera_, Grid *grid_, int num_objects_) {
  camera = camera_;
  grid = grid_;
  num_objects = num_objects_;
}

#endif
