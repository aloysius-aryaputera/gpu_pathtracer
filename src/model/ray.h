//File: ray.h
#ifndef RAY_H
#define RAY_H

#include "vector_and_matrix/vec3.h"

class Ray {
  public:
    __host__ __device__ Ray(vec3 p0_, vec3 dir_);
    __host__ __device__ vec3 get_vector(float t);

    vec3 p0, dir;
};

__host__ __device__ Ray::Ray(vec3 p0_, vec3 dir_) {
  p0 = p0_;
  dir = unit_vector(dir_);
}

__host__ __device__ vec3 Ray::get_vector(float t) {
  return p0 + t * dir;
}

#endif
