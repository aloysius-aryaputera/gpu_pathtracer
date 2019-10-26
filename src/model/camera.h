//File: camera.h
#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>

#include "../param.h"
#include "ray.h"
#include "vector_and_matrix/vec3.h"

class Camera {
  private:
    vec3 center, up, u, v, w;
    float fovy, fovx;

  public:
    __host__ __device__ Camera(
      vec3 eye_, vec3 center_, vec3 up_, float fovy_, int width_, int height_
    );
    __host__ __device__ Ray compute_ray(float i, float j);

    int width, height;
    vec3 eye;
};

__host__ __device__ Camera::Camera(
  vec3 eye_, vec3 center_, vec3 up_, float fovy_, int width_, int height_
) {
  eye = eye_;
  center = center_;
  up = up_;
  width = width_;
  height = height_;
  fovy = fovy_;
  fovx = 2.0f * atan(((float)width / (float)height) * tan(M_PI * fovy / 180 / 2)) * \
    180 / M_PI;
  w = unit_vector(eye - center);
  u = unit_vector(cross(up, w));
  v = cross(w, u);
}

__host__ __device__ Ray Camera::compute_ray(float i, float j) {
  vec3 dir;
  float alpha, beta;
  alpha = tan(fovx * M_PI / 180.0 / 2) * (j - ((float)width / 2)) / (
    (float)width / 2);
  beta = tan(fovy * M_PI / 180.0 / 2) * (((float)height / 2) - i) / (
    (float)height / 2);
  dir = unit_vector(alpha * u + beta * v - w);
  Ray out = Ray(eye, dir);
  return out;
}

#endif
