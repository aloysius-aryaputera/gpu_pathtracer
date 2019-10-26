//File: grid.h
#ifndef GRID_H
#define GRID_H

#include <cuda_fp16.h>
#include <math.h>

#include "../../param.h"
#include "../geometry/primitive.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
#include "bounding_box.h"
#include "cell.h"

class Grid {

  private:
    __device__ void _build_cell_array();
    __device__ void _insert_objects();

    float x_min, x_max, y_min, y_max, z_min, z_max, cell_size_x, cell_size_y, \
      cell_size_z;
    int n_cell_x, n_cell_y, n_cell_z;
    BoundingBox world_bounding_box;
    Primitive** object_array;
    int num_objects;
    Cell cell_array[9999][9999][9999];

  public:
    __host__ __device__ Grid() {}
    __device__ Grid(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_, int n_cell_x_, int n_cell_y_, int n_cell_z_,
      Primitive** object_array_, int num_objects_
    );

};

__device__ void Grid::_insert_objects() {
  printf("================================================================\n");
  printf("Inserting objects into the grid\n");
  printf("================================================================\n");
  for (int i = 0; i < n_cell_x; i++) {
    for (int j = 0; j < n_cell_y; j++) {
      for (int k = 0; k < n_cell_z; k++) {
        for (int l = 0; l < num_objects; l++) {
          bool intersecting = \
            cell_array[i][j][k].are_intersecting(
              object_array[l] -> get_bounding_box()
            );
            if (intersecting) {
              cell_array[i][j][k].add_object(object_array[l]);
            }
        }
      }
    }
  }
}

__device__ void Grid::_build_cell_array() {
  float cell_x_min, cell_x_max, cell_y_min, cell_y_max, cell_z_min, cell_z_max;

  for (int i = 0; i < n_cell_x; i++) {
    for (int j = 0; j < n_cell_y; j++) {
      for (int k = 0; k < n_cell_z; k++) {

        cell_x_min = x_min + i * cell_size_x;
        cell_x_max = cell_x_min + cell_size_x;

        cell_y_min = y_min + j * cell_size_y;
        cell_y_max = cell_y_min + cell_size_y;

        cell_z_min = z_min + k * cell_size_z;
        cell_z_max = cell_z_min + cell_size_z;

        cell_array[i][j][k] = \
          Cell(
            cell_x_min, cell_x_max, cell_y_min, cell_y_max, cell_z_min,
            cell_z_max, i, j, k
          );
      }
    }
  }
}

__device__ Grid::Grid(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_, int n_cell_x_, int n_cell_y_, int n_cell_z_,
  Primitive** object_array_, int num_objects_
) {
  x_min = x_min_;
  x_max = x_max_;
  y_min = y_min_;
  y_max = y_max_;
  z_min = z_min_;
  z_max = z_max_;
  object_array = object_array_;
  num_objects = num_objects_;

  n_cell_x = n_cell_x_;
  n_cell_y = n_cell_y_;
  n_cell_z = n_cell_z_;

  cell_size_x = (x_max - x_min) / n_cell_x;
  cell_size_y = (y_max - y_min) / n_cell_y;
  cell_size_z = (z_max - z_min) / n_cell_z;

  _build_cell_array();
  _insert_objects();

  world_bounding_box = BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);
}

#endif
