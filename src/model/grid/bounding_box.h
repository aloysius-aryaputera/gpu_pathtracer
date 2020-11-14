//File: bounding_box.h
#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <math.h>

#include "../../param.h"
#include "../../util/bvh_util.h"
#include "../ray/ray.h"
#include "../vector_and_matrix/vec3.h"
#include "bounding_sphere.h"

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
    __device__ void reset();
    __device__ void initialize(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_
    );
    __device__ bool is_intersection(Ray ray, float &t);
    __device__ bool is_intersection(BoundingSphere bounding_sphere);
    __device__ bool is_intersection(BoundingSphere *bounding_sphere);
    __device__ bool is_inside(vec3 position);
    __device__ void compute_normalized_center(BoundingBox *world_bounding_box);
    __device__ void compute_normalized_center(
      float x_min, float x_max, float y_min, float y_max,
      float z_min, float z_max
    );
    __device__ void compute_bb_morton_3d();
    __device__ void print_bounding_box();
    __device__ float compute_incident_angle(vec3 point, vec3 normal);
    __device__ float compute_covering_cone_angle(vec3 point);
    __device__ float compute_minimum_angle_to_shading_point(
      vec3 point, vec3 cone_axis, float cone_theta_0, float theta_u 
    );

    float x_min, x_max, y_min, y_max, z_min, z_max;
    float x_center, y_center, z_center;
    vec3 center;
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

__device__ float BoundingBox::compute_minimum_angle_to_shading_point(
  vec3 point, vec3 cone_axis, float cone_theta_0, float theta_u 
){
	vec3 dir = unit_vector(point - this -> center);
  float theta = acos(dot(cone_axis, dir));
  return fmaxf(theta - cone_theta_0 - theta_u, 0);
}

__device__ float BoundingBox::compute_covering_cone_angle(vec3 point) {
  vec3 v1 = unit_vector(this -> center - point);

	vec3 dir;
	float cos_value, min_cos_value = 1;

  for (int i = 0; i < 8; i++) {
	  if (i == 0) 
			dir = vec3(this -> x_min, this -> y_min, this -> z_min) - point;
	  else if (i == 1)
			dir = vec3(this -> x_min, this -> y_min, this -> z_max) - point;
		else if (i == 2)
			dir = vec3(this -> x_min, this -> y_max, this -> z_min) - point;
		else if (i == 3)
			dir = vec3(this -> x_min, this -> y_max, this -> z_max) - point;
		else if (i == 4)
			dir = vec3(this -> x_max, this -> y_min, this -> z_min) - point;
		else if (i == 5)
			dir = vec3(this -> x_max, this -> y_min, this -> z_max) - point;
		else if (i == 6)
			dir = vec3(this -> x_max, this -> y_max, this -> z_min) - point;
		else
			dir = vec3(this -> x_max, this -> y_max, this -> z_max) - point;

		dir = unit_vector(dir);
    cos_value = dot(v1, dir);
		if (cos_value < min_cos_value) min_cos_value = cos_value;
	}
	
	return acos(min_cos_value);
}

__device__ float BoundingBox::compute_incident_angle(
  vec3 point, vec3 normal
) {
  vec3 v1 = unit_vector(this -> center - point);
  return acos(dot(v1, normal));
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

__device__ void BoundingBox::compute_normalized_center(
  float x_min, float x_max, float y_min, float y_max,
  float z_min, float z_max
) {
  this -> norm_x_center = (this -> x_center - x_min) / (x_max - x_min);
  this -> norm_y_center = (this -> y_center - y_min) / (y_max - y_min);
  this -> norm_z_center = (this -> z_center - z_min) / (z_max - z_min);
}

__device__ BoundingBox::BoundingBox() {
  this -> initialized = false;
}

__device__ void BoundingBox::reset() {
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
  this -> center = vec3(x_center, y_center, z_center);

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

  this -> initialize(x_min_, x_max_, y_min_, y_max_, z_min_, z_max_);

  //this -> x_min = x_min_;
  //this -> x_max = x_max_;
  //this -> y_min = y_min_;
  //this -> y_max = y_max_;
  //this -> z_min = z_min_;
  //this -> z_max = z_max_;

  //this -> x_center = 0.5 * (this -> x_min + this -> x_max);
  //this -> y_center = 0.5 * (this -> y_min + this -> y_max);
  //this -> z_center = 0.5 * (this -> z_min + this -> z_max);
  //this -> center = vec3(x_center, y_center, z_center);

  //this -> length_x = this -> x_max - this -> x_min;
  //this -> length_y = this -> y_max - this -> y_min;
  //this -> length_z = this -> z_max - this -> z_min;

  //this -> tolerance_x = max(this -> length_x / 100, SMALL_DOUBLE);
  //this -> tolerance_y = max(this -> length_y / 100, SMALL_DOUBLE);
  //this -> tolerance_z = max(this -> length_z / 100, SMALL_DOUBLE);

  //this -> initialized = true;
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

__device__ bool BoundingBox::is_intersection(BoundingSphere *bounding_sphere) {
  float x_dist = abs(this -> x_center - bounding_sphere -> center.x());
  float y_dist = abs(this -> y_center - bounding_sphere -> center.y());
  float z_dist = abs(this -> z_center - bounding_sphere -> center.z());
  float half_x_ext = this -> x_max - this -> x_center;
  float half_y_ext = this -> y_max - this -> y_center;
  float half_z_ext = this -> z_max - this -> z_center;
  if (
    x_dist <= (half_x_ext + bounding_sphere -> r) &&
    y_dist <= (half_y_ext + bounding_sphere -> r) &&
    z_dist <= (half_z_ext + bounding_sphere -> r)
  ) {
    return true;
  } else {
    return false;
  }
}

__device__ bool BoundingBox::is_intersection(BoundingSphere bounding_sphere) {
  float x_dist = abs(this -> x_center - bounding_sphere.center.x());
  float y_dist = abs(this -> y_center - bounding_sphere.center.y());
  float z_dist = abs(this -> z_center - bounding_sphere.center.z());
  float half_x_ext = this -> x_max - this -> x_center;
  float half_y_ext = this -> y_max - this -> y_center;
  float half_z_ext = this -> z_max - this -> z_center;
  if (
    x_dist <= (half_x_ext + bounding_sphere.r) &&
    y_dist <= (half_y_ext + bounding_sphere.r) &&
    z_dist <= (half_z_ext + bounding_sphere.r)
  ) {
    return true;
  } else {
    return false;
  }
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
