//File: primitive.h
#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include <curand_kernel.h>
#include <math.h>

#include "../../param.h"
#include "../grid/bounding_box.h"
#include "../material/material.h"
#include "../object/object.h"
#include "../ray/ray.h"
#include "../vector_and_matrix/vec3.h"

struct hit_record;

class Primitive {
  private:
    Material *material;
    bool sub_surface_scattering;
    bool light_source;
    bool transparent_geom;
    float area;
    vec3 t, b, normal;

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

    __device__ virtual bool is_light_source() {
      return false;
    }

    __device__ virtual bool is_transparent_geom() {
      return false;
    }

    __device__ virtual hit_record get_random_point_on_surface(
      curandState *rand_state
    );

    __device__ virtual float get_area() {
      return this -> area;
    }

    __device__ virtual int get_object_idx() {
      return this -> object_idx;
    }

    __device__ virtual vec3 get_fixed_normal() {
      return this -> normal;
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

    __device__ virtual float get_hittable_pdf(vec3 origin, vec3 dir) {
      return 0;
    }

    __device__ virtual vec3 compute_directed_energy(
      vec3 point, vec3 point_normal
    ) {
      return vec3(0.0, 0.0, 0.0);
    }

    __device__ virtual vec3 get_energy() {
      return this -> energy;
    }

    BoundingBox *bounding_box;
    int object_idx;
    vec3 energy;
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
