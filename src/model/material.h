//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector_and_matrix/vec3.h"

class Material {
  private:
    vec3 *texture;
    int texture_width, texture_height;

  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_,
      int material_image_height, int material_image_width, vec3 *texture_
    );

    vec3 ambient, diffuse, emission, albedo;
};

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_,
  int material_image_height, int material_image_width, vec3 *texture_
) {
  this -> ambient = ambient_;
  this -> diffuse = diffuse_;
  this -> albedo = albedo_;
  this -> emission = emission_;
  this -> texture_height = material_image_height;
  this -> texture_width = material_image_width;
  this -> texture = texture_;
}

#endif
