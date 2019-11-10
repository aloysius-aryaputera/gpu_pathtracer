//File: triangle.h
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
#include "primitive.h"

class Triangle: public Primitive {
  private:
    __host__ __device__ float _compute_tolerance();
    __device__ void _compute_bounding_box();

    float area, tolerance;
    vec3 point_1, point_2, point_3, norm_1, norm_2, norm_3, normal;
    Material *material;
    BoundingBox *bounding_box;

  public:
    __host__ __device__ Triangle() {};
    __device__ Triangle(
      vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_,
      vec3 norm_1_, vec3 norm_2_, vec3 norm_3_
    );
    __device__ bool hit(Ray ray, float t_max, hit_record& rec);
    __device__ Material* get_material();
    __device__ BoundingBox* get_bounding_box();

};

__host__ __device__ float _compute_triangle_area(
  vec3 point_1, vec3 point_2, vec3 point_3);

__device__ void Triangle::_compute_bounding_box() {
  float x_min, x_max, y_min, y_max, z_min, z_max;

  x_min = min(point_1.x(), point_2.x());
  x_min = min(x_min, point_3.x());

  x_max = max(point_1.x(), point_2.x());
  x_max = max(x_max, point_3.x());

  y_min = min(point_1.y(), point_2.y());
  y_min = min(y_min, point_3.y());

  y_max = max(point_1.y(), point_2.y());
  y_max = max(y_max, point_3.y());

  z_min = min(point_1.z(), point_2.z());
  z_min = min(z_min, point_3.z());

  z_max = max(point_1.z(), point_2.z());
  z_max = max(z_max, point_3.z());

  this -> bounding_box = new BoundingBox(
    x_min - min(SMALL_DOUBLE, tolerance),
    x_max + min(SMALL_DOUBLE, tolerance),
    y_min - min(SMALL_DOUBLE, tolerance),
    y_max + min(SMALL_DOUBLE, tolerance),
    z_min - min(SMALL_DOUBLE, tolerance),
    z_max + min(SMALL_DOUBLE, tolerance)
  );
}

__device__ Triangle::Triangle(
  vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_,
  vec3 norm_1_=vec3(0, 0, 0), vec3 norm_2_=vec3(0, 0, 0),
  vec3 norm_3_=vec3(0, 0, 0)
) {
  this -> point_1 = vec3(point_1_.x(), point_1_.y(), point_1_.z());
  this -> point_2 = vec3(point_2_.x(), point_2_.y(), point_2_.z());
  this -> point_3 = vec3(point_3_.x(), point_3_.y(), point_3_.z());
  this -> material = material_;
  this -> tolerance = this -> _compute_tolerance();
  this -> area = _compute_triangle_area(
    this -> point_1, this -> point_2, this -> point_3);
  this -> normal = unit_vector(
    cross(this -> point_2 - this -> point_1, this -> point_3 - this -> point_1));
  this -> norm_1 = unit_vector(norm_1_);
  this -> norm_2 = unit_vector(norm_2_);
  this -> norm_3 = unit_vector(norm_3_);

  if (
    compute_distance(norm_1_, vec3(0, 0, 0)) < SMALL_DOUBLE ||
    compute_distance(norm_2_, vec3(0, 0, 0)) < SMALL_DOUBLE ||
    compute_distance(norm_3_, vec3(0, 0, 0)) < SMALL_DOUBLE
  ) {
    this -> norm_1 = this -> normal;
    this -> norm_2 = this -> normal;
    this -> norm_3 = this -> normal;
  }
  this -> _compute_bounding_box();
}

__host__ __device__ float Triangle::_compute_tolerance() {
  float dist_1 = compute_distance(this -> point_1, this -> point_2);
  float dist_2 = compute_distance(this -> point_1, this -> point_3);
  float dist_3 = compute_distance(this -> point_2, this -> point_3);
  float tolerance_;
  if (dist_1 < dist_2) {
    tolerance_ = dist_1;
  } else {
    tolerance_ = dist_2;
  }
  if (dist_3 < tolerance_) {
    tolerance_ = dist_3;
  }
  return tolerance_ / 100;
}

__device__ Material* Triangle::get_material() {
  return material;
}

__device__ BoundingBox* Triangle::get_bounding_box() {
  return bounding_box;
}

__device__ __device__ bool Triangle::hit(
  Ray ray, float t_max, hit_record& rec) {
  float t = (
    dot(this -> point_1, this -> normal) - dot(ray.p0, this -> normal)) / \
      dot(ray.dir, this -> normal);

  if (t > t_max) {
    return false;
  }

  vec3 point_4 = ray.get_vector(t);

  float dist_1 = compute_distance(point_4, point_1);
  float dist_2 = compute_distance(point_4, point_2);
  float dist_3 = compute_distance(point_4, point_3);

  if (
      (dist_1 < tolerance ||
       dist_2 < tolerance ||
       dist_3 < tolerance) &&
      t > min(SMALL_DOUBLE, tolerance) &&
      t < (1 / min(SMALL_DOUBLE, tolerance))
  ) {
    rec.t = t;
    rec.point = ray.get_vector(t);

    if (dist_1 < tolerance) {
      rec.normal = this -> norm_1;
    } else if (dist_2 < tolerance) {
      rec.normal = this -> norm_2;
    } else {
      rec.normal = this -> norm_3;
    }

    rec.object = this;
    return true;
  }

  float area_1 = _compute_triangle_area(point_1, point_2, point_4);
  float area_2 = _compute_triangle_area(point_1, point_3, point_4);
  float area_3 = _compute_triangle_area(point_2, point_3, point_4);

  if (
    (area_1 + area_2 + area_3 - area) < tolerance &&
    t > min(SMALL_DOUBLE, tolerance) &&
    t < (1 / min(SMALL_DOUBLE, tolerance))
  ) {
    rec.t = t;
    rec.point = ray.get_vector(t);

    float alpha = area_3 / this -> area;
    float beta = area_2 / this -> area;
    float gamma = 1 - alpha - beta;

    vec3 new_normal = alpha * this -> norm_1 + beta * this -> norm_2 + \
      gamma * this -> norm_3;

    rec.normal = unit_vector(new_normal);

    rec.object = this;
    return true;
  }

  return false;
}

__host__ __device__ float _compute_triangle_area(
  vec3 point_1, vec3 point_2, vec3 point_3
) {

  float s1 = compute_distance(point_1, point_2);
  float s2 = compute_distance(point_1, point_3);
  float s3 = compute_distance(point_2, point_3);
  float s = (s1 + s2 + s3) / 2.0f;
  return abs(sqrt(s * (s - s1) * (s - s2) * (s - s3)));
}

#endif
