//File: bounding_sphere.h
#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

#include "../vector_and_matrix/vec3.h"

class BoundingSphere {
  public:
    __device__ BoundingSphere(vec3 center, float r);
    __device__ bool is_inside(vec3 coordinate);
    __device__ void assign_new_radius(float r_);
    __device__ void assign_new_center(vec3 center_);

    float r;
    vec3 center;
};

__device__ BoundingSphere::BoundingSphere(vec3 center_, float r_) {
  this -> r = r_;
  this -> center = center_;
}

__device__ void BoundingSphere::assign_new_center(vec3 center_) {
  this -> center = center_;
}

__device__ void BoundingSphere::assign_new_radius(float r_) {
  this -> r = r_;
}

__device__ bool BoundingSphere::is_inside(vec3 coordinate) {
  float distance = compute_distance(this -> center, coordinate);
  if (distance <= this -> r) {
    return true;
  } else {
    return false;
  }
}

#endif
