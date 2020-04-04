//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

#include "../point/point.h"

class Object {
  private:
    __device__ void _compute_accummulated_triangle_area();

    float *triangle_area;
    float *accumulated_triangle_area;

  public:
    __host__ __device__ Object() {}
    __device__ Object(
      int primitives_offset_idx_, int num_primitives_, float *triangle_area_
    );
    __device__ void set_as_sub_surface_scattering();
    __device__ void allocate_point_array(Point** sss_pt_array_);

    int num_primitives, primitives_offset_idx;
    bool sub_surface_scattering;
    Point** sss_pt_array;
};

__device__ Object::Object(
  int primitives_offset_idx_, int num_primitives_, float *triangle_area_
) {
  this -> primitives_offset_idx = primitives_offset_idx_;
  this -> num_primitives = num_primitives_;
  this -> triangle_area = triangle_area_;
  this -> sub_surface_scattering = false;

  this -> _compute_accummulated_triangle_area();
}

__device__ void Object::allocate_point_array(Point** sss_pt_array_) {
  this -> sss_pt_array = sss_pt_array_;
}

__device__ void Object::set_as_sub_surface_scattering() {
  this -> sub_surface_scattering = true;
}

__device__ void Object::_compute_accummulated_triangle_area() {
  float acc = 0;
  for (int i = 0; i < this -> num_primitives; i++) {
    acc += this -> triangle_area[i];
    this -> accumulated_triangle_area[i] = acc;
  }
}

#endif
