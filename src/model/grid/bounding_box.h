//File: bounding_box.h
#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <math.h>

#include "../../param.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"

class BoundingBox {
  private:
    __device__ void _compute_t_x_range(Ray ray, float *t_range);
    __device__ void _compute_t_y_range(Ray ray, float *t_range);
    __device__ void _compute_t_z_range(Ray ray, float *t_range);

    float tolerance_x, tolerance_y, tolerance_z;

  public:
    __host__ __device__ BoundingBox() {}
    __device__ BoundingBox(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_
    );
    __device__ bool is_intersection(Ray ray);
    __device__ bool is_inside(vec3 position);

    float x_min, x_max, y_min, y_max, z_min, z_max;
};

__device__ bool _are_intersecting(float *t_range_1, float *t_range_2);

__device__ bool _are_intersecting(float *t_range_1, float *t_range_2) {
  float t_1_min = t_range_1[0];
  float t_1_max = t_range_1[1];
  float t_2_min = t_range_2[0];
  float t_2_max = t_range_2[1];
  return t_1_min <= t_2_max && t_2_min <= t_1_max;
}

__device__ BoundingBox::BoundingBox(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_
) {
  x_min = x_min_;
  x_max = x_max_;
  y_min = y_min_;
  y_max = y_max_;
  z_min = z_min_;
  z_max = z_max_;

  tolerance_x = (x_max - x_min) / 100;
  tolerance_y = (y_max - y_min) / 100;
  tolerance_z = (z_max - z_min) / 100;
}

__device__ void BoundingBox::_compute_t_x_range(Ray ray, float *t_range) {
  float t1, t2, t_x_min, t_x_max;
  t1 = (x_min - ray.p0.x()) /  ray.dir.x();
  t2 = (x_max - ray.p0.x()) /  ray.dir.x();
  if (t1 > t2) {
    t_x_min = t1;
    t_x_max = t2;
  } else {
    t_x_min = t2;
    t_x_max = t1;
  }
  t_range[0] = t_x_min;
  t_range[1] = t_x_max;
}

__device__ void BoundingBox::_compute_t_y_range(Ray ray, float *t_range) {
  float t1, t2, t_y_min, t_y_max;
  t1 = (y_min - ray.p0.y()) /  ray.dir.y();
  t2 = (y_max - ray.p0.y()) /  ray.dir.y();
  if (t1 > t2) {
    t_y_min = t1;
    t_y_max = t2;
  } else {
    t_y_min = t2;
    t_y_max = t1;
  }
  t_range[0] = t_y_min;
  t_range[1] = t_y_max;
}

__device__ void BoundingBox::_compute_t_z_range(Ray ray, float *t_range) {
  float t1, t2, t_z_min, t_z_max;
  t1 = (z_min - ray.p0.z()) /  ray.dir.z();
  t2 = (z_max - ray.p0.z()) /  ray.dir.z();
  if (t1 > t2) {
    t_z_min = t1;
    t_z_max = t2;
  } else {
    t_z_min = t2;
    t_z_max = t1;
  }
  t_range[0] = t_z_min;
  t_range[1] = t_z_max;
}

__device__ bool BoundingBox::is_intersection(Ray ray) {
  float t_range_x[2], t_range_y[2], t_range_z[2];
  _compute_t_x_range(ray, t_range_x);
  _compute_t_y_range(ray, t_range_y);
  _compute_t_z_range(ray, t_range_z);
  if (_are_intersecting(t_range_x, t_range_y)) {
    float t_range_xy[2];
    t_range_xy[0] = max(t_range_x[0], t_range_y[0]);
    t_range_xy[1] = min(t_range_x[1], t_range_y[1]);
    if (_are_intersecting(t_range_xy, t_range_z)) {
      return true;
    }
  }
  return false;
}

__device__ bool BoundingBox::is_inside(vec3 position) {
  if (
    position.x() >= (x_min - tolerance_x) &&
    position.x() <= (x_max + tolerance_x) &&
    position.y() >= (y_min - tolerance_y) &&
    position.y() <= (y_max + tolerance_y) &&
    position.z() >= (z_min - tolerance_z) &&
    position.z() <= (z_max + tolerance_z)
  ) {
    return true;
  } else {
    return false;
  }
}

#endif
