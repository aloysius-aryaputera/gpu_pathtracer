//File: point.h
#ifndef POINT_H
#define POINT_H

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../grid/bounding_sphere.h"
#include "../vector_and_matrix/vec3.h"

class Point {
  private:
    __device__ void _create_bounding_box();
    __device__ void _create_bounding_sphere();

  public:
    __host__ __device__ Point();
    __device__ Point(
      vec3 location_, vec3 filter_, vec3 normal_, int object_idx_,
      bool bounding_box_required, bool bounding_sphere_required
    );
    __device__ void assign_color(vec3 color_);

    vec3 location, filter, normal, color;
    int object_idx, accummulated_photon_count;
    BoundingBox *bounding_box;
    BoundingSphere *bounding_sphere;
    float current_photon_radius;
};

__device__ Point::Point(
  vec3 location_, vec3 filter_, vec3 normal_, int object_idx_,
  bool bounding_box_required, bool bounding_sphere_required
) {
  this -> object_idx = object_idx_;
  this -> location = location_;
  this -> filter = filter_;
  this -> normal = normal_;
  if (bounding_box_required)
    this -> _create_bounding_box();
  if (bounding_sphere_required)
    this -> _create_bounding_sphere();
}

__device__ void Point::_create_bounding_sphere() {
  this -> bounding_sphere = new BoundingSphere(this -> location, 0);
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

#endif
