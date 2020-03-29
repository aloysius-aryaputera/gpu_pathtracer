//File: primitive.h
#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material.h"
#include "../object/object.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"

struct hit_record;

class Primitive {
  private:
    Material *material;

  public:
    __host__ __device__ Primitive() {}
    __device__ virtual bool hit(Ray ray, float t_max, hit_record& rec) {
      return false;
    }
    __device__ virtual Material* get_material() {
      return this -> material;
    }
    __device__ virtual BoundingBox* get_bounding_box() {
      return this -> bounding_box;
    }

    BoundingBox *bounding_box;
    Object *object;
    float area;
    bool sub_surface_scattering;

};

struct hit_record
{
    float t;
    vec3 point;
    vec3 normal;
    vec3 uv_vector;
    Primitive* object;
    Ray coming_ray;
};

#endif
