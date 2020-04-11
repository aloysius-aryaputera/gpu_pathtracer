//File: point.h
#ifndef POINT_H
#define POINT_H

#include "../grid/bounding_box.h"
#include "../vector_and_matrix/vec3.h"

class Point {
  public:
    __host__ __device__ Point();
    __device__ Point(vec3 location_, vec3 filter_, vec3 normal_);
    __device__ void assign_color(vec3 color_);

    vec3 location;
    vec3 filter;
    vec3 normal;
    vec3 color;
    BoundingBox *bounding_box;
};

__device__ Point::Point(vec3 location_, vec3 filter_, vec3 normal_) {
  this -> location = location_;
  this -> filter = filter_;
  this -> normal = normal_;
  this -> bounding_box = new BoundingBox(
    this -> location.x() - SMALL_DOUBLE, this -> location.y() + SMALL_DOUBLE,
    this -> location.y() - SMALL_DOUBLE, this -> location.y() + SMALL_DOUBLE,
    this -> location.z() - SMALL_DOUBLE, this -> location.z() + SMALL_DOUBLE
  );
}

__device__ void Point::assign_color(vec3 color_) {
  this -> color = color_;
}

#endif
