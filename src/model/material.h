//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector_and_matrix/vec3.h"

class Material {
  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_);

    vec3 ambient, diffuse, emission, albedo;
};

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_
) {
  ambient = ambient_;
  diffuse = diffuse_;
  albedo = albedo_;
  emission = emission_;
}

#endif
