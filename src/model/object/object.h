//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

class Object {
  public:
    __host__ __device__ Object() {}
    __host__ __device__ Object(
      int primitives_offset_idx_, int num_primitives_);

    int num_primitives, primitives_offset_idx;
};

__host__ __device__ Object::Object(
  int primitives_offset_idx_, int num_primitives_
) {
  this -> primitives_offset_idx = primitives_offset_idx_;
  this -> num_primitives = num_primitives_;
}

#endif
