//File: sphere.h
#ifndef SPHERE_H
#define SPHERE_H

#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
#include "primitive.h"

class Sphere: public Primitive {
  private:
    __host__ __device__ float _compute_tolerance();
    __device__ void _compute_bounding_box();

    vec3 center;
    float r, tolerance;
    Material* material;
    BoundingBox *bounding_box;

  public:
    __host__ __device__ Sphere() {};
    __device__ Sphere(vec3 center_, float r_, Material* material_);
    __device__ bool hit(Ray ray, float t_max, hit_record& rec);
    __device__ vec3 get_normal(vec3 point_on_surface);
    __device__ Material* get_material();
    __device__ BoundingBox* get_bounding_box();
};

__device__ int _get_num_roots(float a, float b, float c, float* root_array);
__device__ float _get_discriminant(float a, float b, float c);

__device__ int _get_num_roots(float a, float b, float c, float* root_array) {
  float discriminant, root;
  int iter = 0;
  discriminant = _get_discriminant(a, b, c);
  if (discriminant == 0) {
    root = -b / (2 * a);
    // if (root > SMALL_DOUBLE) {
      *(root_array + iter++) = root;
    // }
  }
  if (discriminant > 0) {
    root = (-b - sqrt(discriminant)) / (2 * a);
    // if (root > SMALL_DOUBLE) {
      *(root_array + iter++) = root;
    // }

    root = (-b + sqrt(discriminant)) / (2 * a);
    // if (root > SMALL_DOUBLE) {
      *(root_array + iter++) = root;
    // }
  }
  return iter;
}

__device__ float _get_discriminant(float a, float b, float c) {
  return b * b - 4 * a * c;
}

__device__ void Sphere::_compute_bounding_box() {
  float x_min, x_max, y_min, y_max, z_min, z_max;
  int iter = 0;
  vec3 point[8];

  point[iter++] = vec3(center.x() - r, center.y() - r, center.z() - r);
  point[iter++] = vec3(center.x() - r, center.y() - r, center.z() + r);
  point[iter++] = vec3(center.x() - r, center.y() + r, center.z() - r);
  point[iter++] = vec3(center.x() - r, center.y() + r, center.z() + r);
  point[iter++] = vec3(center.x() + r, center.y() - r, center.z() - r);
  point[iter++] = vec3(center.x() + r, center.y() - r, center.z() + r);
  point[iter++] = vec3(center.x() + r, center.y() + r, center.z() - r);
  point[iter++] = vec3(center.x() + r, center.y() + r, center.z() + r);

  x_min = INFINITY;
  y_min = INFINITY;
  z_min = INFINITY;
  x_max = -INFINITY;
  y_max = -INFINITY;
  z_max = -INFINITY;
  for (iter = 0; iter < 8; iter++) {
    x_min = min(x_min, point[iter].x());
    x_max = min(x_max, point[iter].x());
    y_min = min(y_min, point[iter].y());
    y_max = min(y_max, point[iter].y());
    z_min = min(z_min, point[iter].z());
    z_max = min(z_max, point[iter].z());
  }

  bounding_box = new BoundingBox(
    x_min - tolerance, x_max + tolerance, y_min - tolerance,
    y_max + tolerance, z_min - tolerance, z_max + tolerance
  );
}

__device__ bool Sphere::hit(Ray ray, float t_max, hit_record& rec) {
  float a, b, c, t;
  float root_array[2];
  int num_roots;
  vec3 vec1;
  a = dot(ray.dir, ray.dir);
  vec1 = ray.p0 - center;
  b = 2 * dot(ray.dir, vec1);
  c = (vec1.x() * vec1.x() + vec1.y() * vec1.y() + vec1.z() * vec1.z()) - \
    (r * r);
  num_roots = _get_num_roots(a, b, c, root_array);
  if (
    num_roots > 0 && root_array[0] > tolerance && root_array[0] < (1 / tolerance)
  ) {
    t = root_array[0];
    rec.t = t;
    rec.point = ray.get_vector(t);
    rec.normal = get_normal(rec.point);
    rec.object = this;
    return true;
  }
  return false;
}

__device__ Sphere::Sphere(vec3 center_, float r_, Material *material_) {
  center = center_;
  r = r_;
  material = material_;
  tolerance = _compute_tolerance();
  _compute_bounding_box();
}

__device__ Material* Sphere::get_material() {
  return material;
}

__device__ vec3 Sphere::get_normal(vec3 point_on_surface) {
  return unit_vector(point_on_surface - center);
}

__device__ float Sphere::_compute_tolerance() {
  return min(SMALL_DOUBLE, r / 100.0);
}

__device__ BoundingBox* Sphere::get_bounding_box() {
  return bounding_box;
}

#endif
