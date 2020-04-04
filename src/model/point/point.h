//File: point.h
#ifndef POINT_H
#define POINT_H

#include "../vector_and_matrix/vec3.h"

class Point {
  public:
    __host__ __device__ Point();
    __device__ Point(vec3 location_);
    vec3 location;
};

__device__ Point::Point(vec3 location_) {
  this -> location = location_;
}

#endif
