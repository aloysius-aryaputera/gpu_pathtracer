//File: triangle.h
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <math.h>

#include "../../param.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
};

class Triangle {
  private:
    vec3 point_1, point_2, point_3, normal;
    float area;

  public:
    __host__ Triangle(vec3 point_1_, vec3 point_2_, vec3 point_3_);
    __device__ bool hit(Ray ray, hit_record& rec);
    __device__ vec3 get_normal(vec3 point_on_surface);
};

__host__ __device__ float _compute_triangle_area(vec3 point_1, vec3 point_2, vec3 point_3);

__host__ Triangle::Triangle(vec3 point_1_, vec3 point_2_, vec3 point_3_) {
  point_1 = point_1_;
  point_2 = point_2_;
  point_3 = point_3_;
  area = _compute_triangle_area(point_1, point_2, point_3);
}

__device__ vec3 Triangle::get_normal(vec3 point_on_surface) {
  vec3 cross_product = cross(point_2 - point_1, point_3 - point_1);
  return unit_vector(cross_product);
}

__device__ bool Triangle::hit(Ray ray, hit_record& rec) {
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
    point = new vec3(point_4.x, point_4.y, point_4.z);
  }
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
