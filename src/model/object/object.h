//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

class Object {
  private:
    float *triangle_area;

  public:
    __host__ __device__ Object() {}
    __host__ __device__ Object(
      int primitives_offset_idx_, int num_primitives_, float *triangle_area_
    );
    __device__ void set_as_sub_surface_scattering();

    int num_primitives, primitives_offset_idx;
    bool sub_surface_scattering;
};

__host__ __device__ Object::Object(
  int primitives_offset_idx_, int num_primitives_, float *triangle_area_
) {
  this -> primitives_offset_idx = primitives_offset_idx_;
  this -> num_primitives = num_primitives_;
  this -> triangle_area = triangle_area_;
  this -> sub_surface_scattering = false;
}

__device__ void Object::set_as_sub_surface_scattering() {
  this -> sub_surface_scattering = true;
}

#endif
