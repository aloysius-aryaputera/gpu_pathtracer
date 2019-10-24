//File: primitive.h
#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"

struct hit_record;

class Primitive {
  private:
    Material *material;
    BoundingBox *bounding_box;

  public:
    __host__ __device__ Primitive() {};
    __device__ virtual bool hit(Ray ray, float t_max, hit_record& rec) {
      return false;
    }
    __device__ virtual vec3 get_normal(vec3 point_on_surface) {
      return vec3(0, 0, 0);
    }
    __device__ virtual Material* get_material() {
      return material;
    }
    __device__ virtual BoundingBox* get_bounding_box() {
      return bounding_box;
    }
};

struct hit_record
{
    float t;
    vec3 point;
    vec3 normal;
    Primitive* object;
};

#endif
