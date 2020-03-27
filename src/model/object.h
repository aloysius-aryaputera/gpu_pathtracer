//File: object.h
#ifndef OBJECT_H
#define OBJECT_H

#include "geometry/primitive.h"

class Object {
  public:
    __host__ __device__ Object() {}
    __host__ __device__ Object(
      Primitive **primitive_list_, int num_primitives_
    );
    // __device__ assign_primitive_list(Primitive **primitive_list_);
    // __device__ assign_num_primitives(int num_primitives_);

    Primitive **primitive_list;
    int num_primitives;
};

__host__ __device__ Object::Object(
  Primitive **primitive_list_, int num_primitives_
) {
  this -> primitive_list = primitive_list_;
  this -> num_primitives = num_primitives_;
}

// __device__ Object::assign_primitive_list(Primitive **primitive_list_) {
//   this -> primitive_list = primitive_list_;
// }
//
// __device__ Object::assign_num_primitives(int num_primitives_) {
//   this -> num_primitives = num_primitives_;
// }

#endif
