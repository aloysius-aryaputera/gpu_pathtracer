//File: camera.h
#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>

#include "../param.h"
#include "../util/vector_util.h"
#include "ray.h"
#include "vector_and_matrix/vec3.h"

class Camera {
  private:
    vec3 center, up, u, v, w;
    float fovy, fovx, aperture, lens_radius, focus_dist;

  public:
    __host__ __device__ Camera(
      vec3 eye_, vec3 center_, vec3 up_, float fovy_, int width_, int height_,
      float aperture_, float focus_dist_
    );
    __device__ Ray compute_ray(float i, float j, curandState *rand_state);

    int width, height;
    vec3 eye;
};

__host__ __device__ Camera::Camera(
  vec3 eye_, vec3 center_, vec3 up_, float fovy_, int width_, int height_,
  float aperture_, float focus_dist_
) {
  this -> eye = eye_;
  this -> center = center_;
  this -> up = up_;
  this -> width = width_;
  this -> height = height_;
  this -> aperture = aperture_;
  this -> lens_radius = aperture_ / 2.0;
  this -> focus_dist = focus_dist_;
  this -> fovy = fovy_;
  this -> fovx = 2.0f * atan(((float)width / (float)height) * tan(M_PI * fovy / 180 / 2)) * \
    180 / M_PI;
  this -> w = unit_vector(this -> eye - this -> center);
  this -> u = unit_vector(cross(this -> up, this -> w));
  this -> v = cross(this -> w, this -> u);
}

__device__ Ray Camera::compute_ray(
  float i, float j, curandState *rand_state
) {
  vec3 dir;
  float alpha, beta;
  alpha = tan(this -> fovx * M_PI / 180.0 / 2) * (j - ((float)this -> width / 2)) / (
    (float)this -> width / 2);
  beta = tan(this -> fovy * M_PI / 180.0 / 2) * (((float)this -> height / 2) - i) / (
    (float)this -> height / 2);
  dir = unit_vector(alpha * this -> u + beta * this -> v - this -> w);

  vec3 point = this -> eye + this -> focus_dist * dir;
  vec3 rd = this -> lens_radius * get_random_unit_vector_disk(rand_state);
  vec3 offset = rd.x() * this -> u + rd.y() * this -> v;

  Ray out = Ray(this -> eye + offset, point - this -> eye - offset);
  return out;
}

#endif
