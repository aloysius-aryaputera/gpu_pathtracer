//File: triangle.h
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <math.h>

#include "../../param.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
// #include "hitable.h"

struct hit_record
{
    float t;
    vec3 point;
    vec3 normal;
};

class Triangle {
  private:
    float area;

  public:
    __host__ __device__ Triangle() {};
    __host__ __device__ Triangle(vec3 point_1_, vec3 point_2_, vec3 point_3_);
    __device__ virtual bool hit(Ray ray, float t_min, float t_max, hit_record& rec);
    __host__ __device__ vec3 get_normal(vec3 point_on_surface);

    vec3 point_1, point_2, point_3, normal;
};

__host__ __device__ float _compute_triangle_area(vec3 point_1, vec3 point_2, vec3 point_3);

__host__ __device__ Triangle::Triangle(vec3 point_1_, vec3 point_2_, vec3 point_3_) {
  point_1 = point_1_;
  point_2 = point_2_;
  point_3 = point_3_;
  area = _compute_triangle_area(point_1, point_2, point_3);
  normal = get_normal(point_1);
}

__host__ __device__ vec3 Triangle::get_normal(vec3 point_on_surface) {
  vec3 cross_product = cross(point_2 - point_1, point_3 - point_1);
  return unit_vector(cross_product);
}

__device__ bool Triangle::hit(Ray ray, float t_min, float t_max, hit_record& rec) {
  float t = (dot(point_1, normal) - dot(ray.p0, normal)) / dot(ray.dir, normal);
  vec3 point_4 = ray.get_vector(t);

  if (
    (
      compute_distance(point_4, point_1) < SMALL_DOUBLE ||
      compute_distance(point_4, point_2) < SMALL_DOUBLE ||
      compute_distance(point_4, point_3) < SMALL_DOUBLE &&
      t > SMALL_DOUBLE
    )
  ) {
    rec.t = t;
    rec.point = ray.get_vector(t);
    rec.normal = get_normal(rec.point);
    return true;
  }

  float area_1 = _compute_triangle_area(point_1, point_2, point_4);
  float area_2 = _compute_triangle_area(point_1, point_3, point_4);
  float area_3 = _compute_triangle_area(point_2, point_3, point_4);

  if ((area_1 + area_2 + area_3 - area) < SMALL_DOUBLE && t > SMALL_DOUBLE) {
    rec.t = t;
    rec.point = ray.get_vector(t);
    rec.normal = get_normal(rec.point);
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
  return sqrt(s * (s - s1) * (s - s2) * (s - s3));
}

#endif
