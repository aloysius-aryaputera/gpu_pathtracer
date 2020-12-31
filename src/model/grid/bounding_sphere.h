//File: bounding_sphere.h
#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

#include "../vector_and_matrix/vec3.h"

class BoundingSphere {
  public:
    __device__ BoundingSphere();
    __device__ BoundingSphere(vec3 center_, float r_);
    __device__ void reset();
    __device__ void initialize(vec3 center_, float r_);
    __device__ bool is_inside(vec3 coordinate);
    __device__ bool is_inside(vec3 coordinate, vec3 normal);
    __device__ void assign_new_radius(float r_);
    __device__ void assign_new_center(vec3 center_);

    float r;
    vec3 center;
    bool initialized;
};

__device__ BoundingSphere::BoundingSphere() {
  this -> initialized = false;
}

__device__ BoundingSphere::BoundingSphere(vec3 center_, float r_) {
  this -> initialize(center_, r_);
}

__device__ void BoundingSphere::initialize(vec3 center_, float r_) {
  this -> r = r_;
  this -> center = center_;
  this -> initialized = true;
}

__device__ void BoundingSphere::reset() {
  this -> initialized = false;
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

__device__ bool BoundingSphere::is_inside(vec3 coordinate, vec3 normal) {
  float distance = compute_distance(this -> center, coordinate);
  if (distance > this -> r) {
    return false;
  }
  vec3 r_dir = coordinate - this -> center;
  float parallel_dir = dot(r_dir, normal);
  if (abs(parallel_dir) <= 0.1 * this -> r) {
    return true;
  } else {
    return false;
  }
}

__device__ void compute_bs_union(
  BoundingSphere *bs_1, BoundingSphere *bs_2, vec3 &center, float &r
) {
  vec3 center_dir = bs_2 -> center - bs_1 -> center;
  center_dir.make_unit_vector();

  vec3 point_1 = bs_1 -> center - bs_1 -> r * center_dir;
  vec3 point_2 = bs_1 -> center + bs_1 -> r * center_dir;
  vec3 point_3 = bs_2 -> center - bs_2 -> r * center_dir;
  vec3 point_4 = bs_2 -> center + bs_2 -> r * center_dir;

  float t_min = -min(
    compute_distance(point_1, bs_1 -> center),
    compute_distance(point_3, bs_1 -> center)
  );
  float t_max = max(
    compute_distance(point_2, bs_1 -> center),
    compute_distance(point_4, bs_1 -> center)
  );

  r = (-t_min + t_max) / 2;
  center = (
    bs_1 -> center + t_min * center_dir + bs_1 -> center + t_max * center_dir
  ) / 2;
}

#endif
