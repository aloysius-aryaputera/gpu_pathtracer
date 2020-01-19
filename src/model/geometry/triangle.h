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

    float area, inv_tolerance, tolerance;
    vec3 point_1, point_2, point_3, norm_1, norm_2, norm_3, normal;
    vec3 tex_1, tex_2, tex_3;
    Material *material;
    BoundingBox *bounding_box;

  public:
    __host__ __device__ Triangle() {};
    __device__ Triangle(
      vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_,
      vec3 norm_1_, vec3 norm_2_, vec3 norm_3_, vec3 tex_1_, vec3 tex_2_,
      vec3 tex_3_
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
    x_min - this -> tolerance, x_max + this -> tolerance,
    y_min - this -> tolerance, y_max + this -> tolerance,
    z_min - this -> tolerance, z_max + this -> tolerance
  );
}

__device__ Triangle::Triangle(
  vec3 point_1_, vec3 point_2_, vec3 point_3_, Material* material_,
  vec3 norm_1_=vec3(0, 0, 0), vec3 norm_2_=vec3(0, 0, 0),
  vec3 norm_3_=vec3(0, 0, 0), vec3 tex_1_=vec3(0, 0, 0),
  vec3 tex_2_=vec3(0, 0, 0), vec3 tex_3_=vec3(0, 0, 0)
) {
  this -> point_1 = vec3(point_1_.x(), point_1_.y(), point_1_.z());
  this -> point_2 = vec3(point_2_.x(), point_2_.y(), point_2_.z());
  this -> point_3 = vec3(point_3_.x(), point_3_.y(), point_3_.z());
  this -> material = material_;
  this -> tolerance = this -> _compute_tolerance();
  this -> inv_tolerance = 1.0f / this -> tolerance;
  this -> area = _compute_triangle_area(
    this -> point_1, this -> point_2, this -> point_3);
  this -> normal = unit_vector(
    cross(this -> point_2 - this -> point_1, this -> point_3 - this -> point_1));

  this -> norm_1 = unit_vector(norm_1_);
  this -> norm_2 = unit_vector(norm_2_);
  this -> norm_3 = unit_vector(norm_3_);

  this -> tex_1 = tex_1_;
  this -> tex_2 = tex_2_;
  this -> tex_3 = tex_3_;

  if (
    compute_distance(norm_1_, vec3(0, 0, 0)) < this -> tolerance ||
    compute_distance(norm_2_, vec3(0, 0, 0)) < this -> tolerance ||
    compute_distance(norm_3_, vec3(0, 0, 0)) < this -> tolerance
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
  return min(SMALL_DOUBLE, tolerance_ / 100);
}

__device__ Material* Triangle::get_material() {
  return material;
}

__device__ BoundingBox* Triangle::get_bounding_box() {
  return bounding_box;
}

__device__ bool Triangle::hit(Ray ray, float t_max, hit_record& rec) {

  vec3 p1t = this -> point_1 - ray.p0;
  vec3 p2t = this -> point_2 - ray.p0;
  vec3 p3t = this -> point_3 - ray.p0;

  int kz = max_dimension(abs(ray.dir));
  int kx = kz + 1; if (kx == 3) kx = 0;
  int ky = kx + 1; if (ky == 3) ky = 0;

  vec3 d = permute(ray.dir, kx, ky, kz);
  p1t = permute(p1t, kx, ky, kz);
  p2t = permute(p2t, kx, ky, kz);
  p3t = permute(p3t, kx, ky, kz);

  float sx = -d.x() / d.z();
  float sy = -d.y() / d.z();
  float sz = 1.0 / d.z();
  p1t = vec3(p1t.x() + sx * p1t.z(), p1t.y() + sy * p1t.z(), p1t.z());
  p2t = vec3(p2t.x() + sx * p2t.z(), p2t.y() + sy * p2t.z(), p2t.z());
  p3t = vec3(p3t.x() + sx * p3t.z(), p3t.y() + sy * p3t.z(), p3t.z());

  float e1 = p2t.x() * p3t.y() - p2t.y() * p3t.x();
  float e2 = p3t.x() * p1t.y() - p3t.y() * p1t.x();
  float e3 = p1t.x() * p2t.y() - p1t.y() * p2t.x();

  if (e1 == 0 || e2 == 0 || e3 == 0) {
    double p1t_x_double = (double)p1t.x();
    double p1t_y_double = (double)p1t.y();
    double p2t_x_double = (double)p2t.x();
    double p2t_y_double = (double)p2t.y();
    double p3t_x_double = (double)p3t.x();
    double p3t_y_double = (double)p3t.y();
    e1 = (float)(p2t_x_double * p3t_y_double - p2t_y_double * p3t_x_double);
    e2 = (float)(p3t_x_double * p1t_y_double - p3t_y_double * p1t_x_double);
    e3 = (float)(p1t_x_double * p2t_y_double - p1t_y_double * p2t_x_double);
  }

  if ((e1 < 0 || e2 < 0 || e3 < 0) && (e1 > 0 || e2 > 0 || e3 > 0))
    return false;
  float det = e1 + e2 + e3;
  if (det == 0)
    return false;

  p1t = vec3(p1t.x(), p1t.y(), p1t.z() * sz);
  p2t = vec3(p2t.x(), p2t.y(), p2t.z() * sz);
  p3t = vec3(p3t.x(), p3t.y(), p3t.z() * sz);
  float t_scaled = e1 * p1t.z() + e2 * p2t.z() + e3 * p3t.z();

  float inv_det = 1.0 / det;
  float b1 = e1 * inv_det, b2 = e2 * inv_det, b3 = e3 * inv_det;
  float t = t_scaled * inv_det;

  if (t > t_max || t < this -> tolerance)
    return false;

  rec.t = t;
  rec.object = this;
  rec.coming_ray = ray;
  rec.point = b1 * this -> point_1 + b2 * this -> point_2 + \
    b3 * this -> point_3;
  rec.normal = unit_vector(
    b1 * this -> norm_1 + b2 * this -> norm_2 + b3 * this -> norm_3);
  rec.uv_vector = b1 * this -> tex_1 + b2 * this -> tex_2 + b3 * this -> tex_3;

  return true;
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
