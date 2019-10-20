//File: triangle.h
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <math.h>

#include "../../param.h"
#include "../material.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
// #include "hitable.h"

struct hit_record;

class Triangle {
  private:
    __host__ __device__ float _compute_tolerance();

    float area, tolerance;

  public:
    __host__ __device__ Triangle() {};
    __host__ __device__ Triangle(
      vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_);
    __host__ __device__ bool hit(Ray ray, float t_max, hit_record& rec, bool print);
    __host__ __device__ vec3 get_normal(vec3 point_on_surface);

    vec3 point_1, point_2, point_3, normal;
    Material *material;
};

struct hit_record
{
    float t;
    vec3 point;
    vec3 normal;
    Triangle* object;
};

__host__ __device__ float _compute_triangle_area(vec3 point_1, vec3 point_2, vec3 point_3);

__host__ __device__ Triangle::Triangle(
  vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_
) {
  point_1 = vec3(point_1_.x(), point_1_.y(), point_1_.z());
  point_2 = vec3(point_2_.x(), point_2_.y(), point_2_.z());
  point_3 = vec3(point_3_.x(), point_3_.y(), point_3_.z());
  material = material_;
  tolerance = _compute_tolerance();
  area = _compute_triangle_area(point_1, point_2, point_3);
  normal = get_normal(point_1);
}

__host__ __device__ float Triangle::_compute_tolerance() {
  float dist_1 = compute_distance(point_1, point_2);
  float dist_2 = compute_distance(point_1, point_3);
  float dist_3 = compute_distance(point_2, point_3);
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

__host__ __device__ vec3 Triangle::get_normal(vec3 point_on_surface) {
  vec3 cross_product = cross(point_2 - point_1, point_3 - point_1);
  return unit_vector(cross_product);
}

__host__ __device__ bool Triangle::hit(
  Ray ray, float t_max, hit_record& rec, bool print=false) {
  float t = (dot(point_1, normal) - dot(ray.p0, normal)) / dot(ray.dir, normal);

  if (t > t_max) {
    return false;
  }

  vec3 point_4 = ray.get_vector(t);

  if (
      (compute_distance(point_4, point_1) < tolerance ||
       compute_distance(point_4, point_2) < tolerance ||
       compute_distance(point_4, point_3) < tolerance) &&
      t > min(SMALL_DOUBLE, tolerance) &&
      t < (1 / min(SMALL_DOUBLE, tolerance))
  ) {
    rec.t = t;
    rec.point = ray.get_vector(t);
    rec.normal = get_normal(rec.point);
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
    rec.normal = get_normal(rec.point);
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
