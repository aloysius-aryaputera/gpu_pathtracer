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
    vec3 t, b;

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

    __device__ virtual int get_object_idx() {
      return this -> object_idx;
    }

    __device__ virtual vec3 get_t() {
      return this -> t;
    }

    __device__ virtual vec3 get_b() {
      return this -> b;
    }

    __device__ virtual int get_point_1_idx() {
      return -1;
    }

    __device__ virtual int get_point_2_idx() {
      return -1;
    }

    __device__ virtual int get_point_3_idx() {
      return -1;
    }

    __device__ virtual void assign_tangent(vec3 tangent_, int idx) {

    }

    BoundingBox *bounding_box;
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
