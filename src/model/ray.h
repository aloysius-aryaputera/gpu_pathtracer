//File: ray.h
#ifndef RAY_H
#define RAY_H

#include <cuda_fp16.h>

#include "vector_and_matrix/vec3.h"

class Ray {
  public:
    __host__ __device__ Ray();
    __host__ __device__ Ray(vec3 p0_, vec3 dir_);
    __host__ __device__ vec3 get_vector(float t);

    vec3 p0, dir;
};

__host__ __device__ Ray::Ray() {
  this -> p0 = vec3(0, 0, 0);
  this -> dir = vec3(1, 0, 0);
}

__host__ __device__ Ray::Ray(vec3 p0_, vec3 dir_) {
  this -> p0 = p0_;
  this -> dir = unit_vector(dir_);
}

__host__ __device__ vec3 Ray::get_vector(float t) {
  return this -> p0 + t * this -> dir;
}

#endif
