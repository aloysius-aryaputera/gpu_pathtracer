//File: bounding_box.h
#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <math.h>

#include "../../param.h"
#include "../../util/bvh_util.h"
#include "../ray/ray.h"
#include "../vector_and_matrix/vec3.h"

class BoundingBox {
  private:
    __device__ void _compute_t_x_range(
      Ray ray, float &t_min, float &t_max);
    __device__ void _compute_t_y_range(
      Ray ray, float &t_min, float &t_max);
    __device__ void _compute_t_z_range(
      Ray ray, float &t_min, float &t_max);

    float tolerance_x, tolerance_y, tolerance_z;

  public:
    __host__ __device__ BoundingBox();
    __device__ BoundingBox(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_
    );
    __device__ void initialize(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_
    );
    __device__ bool is_intersection(Ray ray, float &t);
    __device__ bool is_inside(vec3 position);
    __device__ void compute_normalized_center(BoundingBox *world_bounding_box);
    __device__ void compute_bb_morton_3d();
    __device__ void print_bounding_box();

    float x_min, x_max, y_min, y_max, z_min, z_max;
    float x_center, y_center, z_center;
    float length_x, length_y, length_z;
    float norm_x_center, norm_y_center, norm_z_center;
    unsigned int morton_code;
    bool initialized;
};

__device__ bool _are_intersecting(
  float t_1_min, float t_1_max, float t_2_min, float t_2_max
);

__device__ bool _are_intersecting(
  float t_1_min, float t_1_max, float t_2_min, float t_2_max
) {
  return t_1_min <= t_2_max && t_2_min <= t_1_max;
}

__device__ void BoundingBox::print_bounding_box() {
  printf("================================================================\n");
  printf("x_min = %5.5f, x_max = %5.5f\n", this -> x_min, this -> x_max);
  printf("y_min = %5.5f, y_max = %5.5f\n", this -> y_min, this -> y_max);
  printf("z_min = %5.5f, z_max = %5.5f\n", this -> z_min, this -> z_max);
  printf("================================================================\n");
}

__device__ void BoundingBox::compute_bb_morton_3d() {
  this -> morton_code = compute_morton_3d(
    this -> norm_x_center, this -> norm_y_center, this -> norm_z_center
  );
}

__device__ void BoundingBox::compute_normalized_center(
  BoundingBox *world_bounding_box
) {
  this -> norm_x_center = (this -> x_center - world_bounding_box -> x_min) / \
    world_bounding_box -> length_x;
  this -> norm_y_center = (this -> y_center - world_bounding_box -> y_min) / \
    world_bounding_box -> length_y;
  this -> norm_z_center = (this -> z_center - world_bounding_box -> z_min) / \
    world_bounding_box -> length_z;
}

__device__ BoundingBox::BoundingBox() {
  this -> initialized = false;
}

__device__ void BoundingBox::initialize(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_
) {
  this -> x_min = x_min_;
  this -> x_max = x_max_;
  this -> y_min = y_min_;
  this -> y_max = y_max_;
  this -> z_min = z_min_;
  this -> z_max = z_max_;

  this -> x_center = 0.5 * (this -> x_min + this -> x_max);
  this -> y_center = 0.5 * (this -> y_min + this -> y_max);
  this -> z_center = 0.5 * (this -> z_min + this -> z_max);

  this -> length_x = this -> x_max - this -> x_min;
  this -> length_y = this -> y_max - this -> y_min;
  this -> length_z = this -> z_max - this -> z_min;

  this -> tolerance_x = max(this -> length_x / 100, SMALL_DOUBLE);
  this -> tolerance_y = max(this -> length_y / 100, SMALL_DOUBLE);
  this -> tolerance_z = max(this -> length_z / 100, SMALL_DOUBLE);

  this -> initialized = true;
}

__device__ BoundingBox::BoundingBox(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_
) {
  this -> x_min = x_min_;
  this -> x_max = x_max_;
  this -> y_min = y_min_;
  this -> y_max = y_max_;
  this -> z_min = z_min_;
  this -> z_max = z_max_;

  this -> x_center = 0.5 * (this -> x_min + this -> x_max);
  this -> y_center = 0.5 * (this -> y_min + this -> y_max);
  this -> z_center = 0.5 * (this -> z_min + this -> z_max);

  this -> length_x = this -> x_max - this -> x_min;
  this -> length_y = this -> y_max - this -> y_min;
  this -> length_z = this -> z_max - this -> z_min;

  this -> tolerance_x = max(this -> length_x / 100, SMALL_DOUBLE);
  this -> tolerance_y = max(this -> length_y / 100, SMALL_DOUBLE);
  this -> tolerance_z = max(this -> length_z / 100, SMALL_DOUBLE);

  this -> initialized = true;
}

__device__ void BoundingBox::_compute_t_x_range(
  Ray ray, float &t_min, float &t_max
) {
  float t1, t2, inv_dir_x = 1.0 / ray.dir.x();
  t1 = (x_min - ray.p0.x()) * inv_dir_x;
  t2 = (x_max - ray.p0.x()) * inv_dir_x;
  if (t1 < t2) {
    t_min = t1;
    t_max = t2;
  } else {
    t_min = t2;
    t_max = t1;
  }
}

__device__ void BoundingBox::_compute_t_y_range(
  Ray ray, float &t_min, float &t_max
) {
  float t1, t2, inv_dir_y = 1.0 / ray.dir.y();
  t1 = (y_min - ray.p0.y()) * inv_dir_y;
  t2 = (y_max - ray.p0.y()) * inv_dir_y;
  if (t1 < t2) {
    t_min = t1;
    t_max = t2;
  } else {
    t_min = t2;
    t_max = t1;
  }
}

__device__ void BoundingBox::_compute_t_z_range(
  Ray ray, float &t_min, float &t_max
) {
  float t1, t2, inv_dir_z = 1.0 / ray.dir.z();
  t1 = (z_min - ray.p0.z()) * inv_dir_z;
  t2 = (z_max - ray.p0.z()) * inv_dir_z;
  if (t1 < t2) {
    t_min = t1;
    t_max = t2;
  } else {
    t_min = t2;
    t_max = t1;
  }
}

__device__ bool BoundingBox::is_intersection(Ray ray, float &t) {
  float t_x_min, t_x_max, t_y_min, t_y_max, t_z_min, t_z_max;
  _compute_t_x_range(ray, t_x_min, t_x_max);
  _compute_t_y_range(ray, t_y_min, t_y_max);
  _compute_t_z_range(ray, t_z_min, t_z_max);
  if (_are_intersecting(t_x_min, t_x_max, t_y_min, t_y_max)) {
    float t_xy_min = max(t_x_min, t_y_min);
    float t_xy_max = min(t_x_max, t_y_max);
    if (_are_intersecting(t_xy_min, t_xy_max, t_z_min, t_z_max)) {
      t = max(t_xy_min, t_z_min);
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

__device__ void compute_bb_union(
  BoundingBox *bb_1, BoundingBox *bb_2, float &x_min, float &x_max,
  float &y_min, float &y_max, float &z_min, float &z_max
) {
  x_min = min(bb_1 -> x_min, bb_2 -> x_min);
  x_max = max(bb_1 -> x_max, bb_2 -> x_max);
  y_min = min(bb_1 -> y_min, bb_2 -> y_min);
  y_max = max(bb_1 -> y_max, bb_2 -> y_max);
  z_min = min(bb_1 -> z_min, bb_2 -> z_min);
  z_max = max(bb_1 -> z_max, bb_2 -> z_max);
}

#endif
