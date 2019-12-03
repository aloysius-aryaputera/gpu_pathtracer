//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector_and_matrix/vec3.h"

class Material {
  // private:


  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_,
      int material_image_height, int material_image_width, vec3 **texture_
    );
    __device__ vec3 get_texture(vec3 uv_vector);

    vec3 ambient, diffuse, emission, albedo;
    vec3 **texture;
    int texture_width, texture_height;
};

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 emission_, vec3 albedo_,
  int material_image_height, int material_image_width, vec3 **texture_
) {
  this -> ambient = ambient_;
  this -> diffuse = diffuse_;
  this -> albedo = albedo_;
  this -> emission = emission_;
  this -> texture_height = material_image_height;
  this -> texture_width = material_image_width;
  this -> texture = texture_;
}

__device__ vec3 Material::get_texture(vec3 uv_vector) {
  if (this -> texture_width * this -> texture_height > 0) {
    int idx_u = floorf((uv_vector.u() - floorf(uv_vector.u())) * (this -> texture_width - 1));
    int idx_v = floorf((uv_vector.v() - floorf(uv_vector.v())) * (this -> texture_height - 1));

    // int idx = idx_u + idx_v * (this -> texture_width - 1);
    int idx = this -> texture_width * idx_v + idx_u;

    vec3 selected_texture = vec3(
      this -> texture[idx] -> r(),
      this -> texture[idx] -> g(),
      this -> texture[idx] -> b()
    );

    // if (compute_distance(this -> texture[idx][0], vec3(0, 0, 0)) < SMALL_DOUBLE) {
      // printf(
      //   "u = %5.5f, v = %5.5f, idx_u = %d, idx_v = %d\n",
      //   uv_vector.u(), uv_vector.v(), idx_u, idx_v
      // );
    // }

    if (
      isnan(this -> texture[idx] -> r()) |
      isnan(this -> texture[idx] -> g()) |
      isnan(this -> texture[idx] -> b())
    ) {
      printf(
        "u = %5.5f, v = %5.5f, idx_u = %d, idx_v = %d\n",
        uv_vector.u(), uv_vector.v(), idx_u, idx_v
      );
    }

    return selected_texture;
  } else {
    return this -> diffuse;
  }
}

#endif
