//File: primitive.h
#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <curand_kernel.h>
#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material.h"
#include "../object/object.h"
#include "../ray/ray.h"
#include "../vector_and_matrix/vec3.h"

struct hit_record;

class Primitive {
  private:
    Material *material;
    bool sub_surface_scattering;
    float area;

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
    __device__ virtual bool is_sub_surface_scattering() {
      return false;
    }

    __device__ virtual hit_record get_random_point_on_surface(
      curandState *rand_state);

    __device__ virtual float get_area() {
      return this -> area;
    }

    BoundingBox *bounding_box;
    // Object *object;
    int object_idx;

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

__device__ hit_record Primitive::get_random_point_on_surface(
  curandState *rand_state
) {
  hit_record new_hit_record;
  return new_hit_record;
}

#endif
