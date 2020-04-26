//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

#include <curand_kernel.h>

#include "../bvh/bvh.h"
#include "../grid/bounding_box.h"
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
    __device__ void set_as_sub_surface_scattering(int num_pts_);
    __device__ void allocate_point_array(Point** sss_pt_array_);
    __device__ int pick_primitive_idx_for_sampling(
      curandState *rand_state, int sampling_idx
    );
    __device__ void compute_accummulated_triangle_area();
    __device__ void compute_boundaries();
    __device__ void assign_bvh_root_node_idx(int idx);
    __device__ void assign_bvh_leaf_zero_idx(int idx);

    int num_primitives, primitives_offset_idx, num_pts;
    bool sub_surface_scattering;
    Point** sss_pt_array;
    int bvh_root_node_idx, bvh_leaf_zero_idx;
    float x_min, x_max, y_min, y_max, z_min, z_max;
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

__device__ void Object::assign_bvh_root_node_idx(int idx) {
  this -> bvh_root_node_idx = idx;
}

__device__ void Object::assign_bvh_leaf_zero_idx(int idx) {
  this -> bvh_leaf_zero_idx = idx;
}

__device__ int Object::pick_primitive_idx_for_sampling(
  curandState *rand_state, int sampling_idx
) {
  float random_number = curand_uniform(&rand_state[sampling_idx]);
  int idx = 0;
  float accumulated_area = this -> accumulated_triangle_area[idx];

  random_number *= this -> accumulated_triangle_area[
    (this -> num_primitives) - 1];

  while(
    random_number > accumulated_area && idx < (this -> num_primitives) - 1
  ) {
    idx++;
    accumulated_area = this -> accumulated_triangle_area[idx];
  }

  return idx + this -> primitives_offset_idx;
}

__device__ void Object::allocate_point_array(Point** sss_pt_array_) {
  this -> sss_pt_array = sss_pt_array_;
}

__device__ void Object::set_as_sub_surface_scattering(int num_pts_) {
  this -> sub_surface_scattering = true;
  this -> num_pts = num_pts_;
}

__device__ void Object::compute_accummulated_triangle_area() {
  float acc = 0;
  for (int i = 0; i < this -> num_primitives; i++) {
    acc += (this -> triangle_area)[i];
    (this -> accumulated_triangle_area)[i] = acc;
  }
}

__device__ void Object::compute_boundaries() {
  float x_min = INFINITY, x_max = -INFINITY;
  float y_min = INFINITY, y_max = -INFINITY;
  float z_min = INFINITY, z_max = -INFINITY;

  for (int i = 0; i < this -> num_pts; i++) {
    this -> x_min = min(x_min, this -> sss_pt_array[i] -> bounding_box -> x_min);
    this -> x_max = max(x_max, this -> sss_pt_array[i] -> bounding_box -> x_max);
    this -> y_min = min(y_min, this -> sss_pt_array[i] -> bounding_box -> y_min);
    this -> y_max = max(y_max, this -> sss_pt_array[i] -> bounding_box -> y_max);
    this -> z_min = min(z_min, this -> sss_pt_array[i] -> bounding_box -> z_min);
    this -> z_max = max(z_max, this -> sss_pt_array[i] -> bounding_box -> z_max);
  }

}

#endif
