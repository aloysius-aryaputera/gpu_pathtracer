//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

#include <curand_kernel.h>

#include "../point/point.h"
#include "../vector_and_matrix/vec3.h"

class Object {
  private:

    float *triangle_area;
    float *accumulated_triangle_area;

  public:
    __host__ __device__ Object() {}
    __device__ Object(
      int primitives_offset_idx_, int num_primitives_, float *triangle_area_,
      float *accumulated_triangle_area_
    );
    __device__ void set_as_sub_surface_scattering();
    __device__ void allocate_point_array(Point** sss_pt_array_);
    __device__ int pick_primitive_idx_for_sampling(
      curandState *rand_state, int sampling_idx
    );
    __device__ void compute_accummulated_triangle_area();

    int num_primitives, primitives_offset_idx;
    bool sub_surface_scattering;
    Point** sss_pt_array;
};

__device__ Object::Object(
  int primitives_offset_idx_, int num_primitives_, float *triangle_area_,
  float *accumulated_triangle_area_
) {
  this -> primitives_offset_idx = primitives_offset_idx_;
  this -> num_primitives = num_primitives_;
  this -> triangle_area = triangle_area_;
  this -> accumulated_triangle_area = accumulated_triangle_area_;
  this -> sub_surface_scattering = false;
}

__device__ int Object::pick_primitive_idx_for_sampling(
  curandState *rand_state, int sampling_idx
) {
  curandState local_rand_state = rand_state[sampling_idx];
  float random_number = curand_uniform(&rand_state[sampling_idx]);
  float accumulated_area = 0;
  int idx = 0;

  // printf("random_number = %f\n", random_number);

  random_number *= this -> accumulated_triangle_area[
    (this -> num_primitives) - 1];

  // printf("max acc triangle area = %f\n",
  //   this -> accumulated_triangle_area[(this -> num_primitives) - 1]);

  while(
    random_number > accumulated_area && idx < (this -> num_primitives)
  ) {
    accumulated_area = this -> accumulated_triangle_area[idx];
    idx++;
  }

  // printf("idx = %d\n", idx);

  return idx;
}

__device__ void Object::allocate_point_array(Point** sss_pt_array_) {
  this -> sss_pt_array = sss_pt_array_;
}

__device__ void Object::set_as_sub_surface_scattering() {
  this -> sub_surface_scattering = true;
}

__device__ void Object::compute_accummulated_triangle_area() {
  float acc = 0;
  for (int i = 0; i < this -> num_primitives; i++) {
    acc += (this -> triangle_area)[i];
    // printf("triangle_area[%d] = %f\n", i, this -> triangle_area[i]);
    (this -> accumulated_triangle_area)[i] = acc;
  }
}

#endif
