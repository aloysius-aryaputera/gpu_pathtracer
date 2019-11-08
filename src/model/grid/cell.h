//File: cell.h
#ifndef CELL_H
#define CELL_H

#include <math.h>

#include "../../param.h"
#include "../geometry/primitive.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
#include "bounding_box.h"

class Cell {

  private:
    BoundingBox* bounding_box;
    int max_num_objects;

  public:
    __host__ __device__ Cell() {}
    __device__ Cell(
      float x_min, float x_max, float y_min, float y_max, float z_min,
      float z_max, int i_address_, int j_address_, int k_address_,
      Primitive** object_array_, int max_num_objects_,
      BoundingBox* bounding_box_
    );
    __device__ bool are_intersecting(BoundingBox* another_bounding_box);
    __device__ void add_object(Primitive* object);
    __device__ BoundingBox* get_bounding_box();

    int i_address, j_address, k_address, num_objects;
    Primitive** object_array;

};

__device__ Cell::Cell(
  float x_min, float x_max, float y_min, float y_max, float z_min,
  float z_max, int i_address_, int j_address_, int k_address_,
  Primitive** object_array_, int max_num_objects_, BoundingBox* bounding_box_
) {
  bounding_box_ = new BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);
  bounding_box = bounding_box_;
  i_address = i_address_;
  j_address = j_address_;
  k_address = k_address_;
  num_objects = 0;
  max_num_objects = max_num_objects_;
  object_array = object_array_;
}

__device__ BoundingBox* Cell::get_bounding_box() {
  return bounding_box;
}

__device__ bool Cell::are_intersecting(BoundingBox* another_bounding_box) {
  
  return (
    this -> bounding_box -> x_min <= another_bounding_box -> x_max &&
    another_bounding_box -> x_min <= this -> bounding_box -> x_max &&
    this -> bounding_box -> y_min <= another_bounding_box -> y_max &&
    another_bounding_box -> y_min <= this -> bounding_box -> y_max &&
    this -> bounding_box -> z_min <= another_bounding_box -> z_max &&
    another_bounding_box -> z_min <= this -> bounding_box -> z_max
  );
}

__device__ void Cell::add_object(Primitive *object) {
  if (num_objects < max_num_objects) {
    *(object_array + num_objects) = object;
    num_objects += 1;
  }
}

#endif
