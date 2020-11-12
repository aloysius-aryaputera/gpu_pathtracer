//File: point.h
#ifndef POINT_H
#define POINT_H

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../vector_and_matrix/vec3.h"

class Point {
  private:
    __device__ void _create_bounding_box();

  public:
    __host__ __device__ Point();
    __device__ Point(
      vec3 location_, vec3 filter_, vec3 normal_, int object_idx_
    );
    __device__ void assign_color(vec3 color_);
    __device__ void assign_location(vec3 location_);
    __device__ void assign_prev_location(vec3 location_);
    __device__ void assign_direction(vec3 direction_);
    __device__ void declare_on_surface();

    vec3 location, filter, normal, color, direction, prev_location;
    int object_idx;
    BoundingBox *bounding_box;
    bool on_surface;
};

__device__ void Point::declare_on_surface() {
  this -> on_surface = true;
}

__device__ Point::Point(
  vec3 location_, vec3 filter_, vec3 normal_, int object_idx_
) {
  this -> object_idx = object_idx_;
  this -> location = location_;
  this -> filter = filter_;
  this -> normal = normal_;
  this -> on_surface = false;
  this -> _create_bounding_box();
}

__device__ void Point::assign_prev_location(vec3 location_) {
  this -> prev_location = location_;
}

__device__ void Point::assign_location(vec3 location_) {
  this -> location = location_;
  this -> bounding_box -> initialize(
    this -> location.x() - SMALL_DOUBLE, this -> location.x() + SMALL_DOUBLE,
    this -> location.y() - SMALL_DOUBLE, this -> location.y() + SMALL_DOUBLE,
    this -> location.z() - SMALL_DOUBLE, this -> location.z() + SMALL_DOUBLE
  );
}

__device__ void Point::_create_bounding_box() {
  this -> bounding_box = new BoundingBox(
    this -> location.x() - SMALL_DOUBLE, this -> location.x() + SMALL_DOUBLE,
    this -> location.y() - SMALL_DOUBLE, this -> location.y() + SMALL_DOUBLE,
    this -> location.z() - SMALL_DOUBLE, this -> location.z() + SMALL_DOUBLE
  );
}

__device__ void Point::assign_color(vec3 color_) {
  this -> color = color_;
}

__device__ void Point::assign_direction(vec3 direction_) {
  this -> direction = direction_;
}

#endif
