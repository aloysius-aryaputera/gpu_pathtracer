//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>

#include "../util/vector_util.h"
#include "cartesian_system.h"
#include "ray.h"
#include "vector_and_matrix/vec3.h"

__device__ vec3 reflect(vec3 v, vec3 normal);

__device__ vec3 reflect(vec3 v, vec3 normal) {
  return v - 2 * dot(v, normal) * normal;
}

struct reflection_record
{
  Ray ray;
  float cos_theta;
};

class Material {
  private:
    float diffuse_mag, specular_mag;

  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
      vec3 albedo_, float n_s_, int material_image_height,
      int material_image_width, vec3 **texture_
    );
    __device__ vec3 get_texture(vec3 uv_vector);
    __device__ reflection_record get_reflection_ray(
      Ray coming_ray, vec3 hit_point, vec3 normal,
      curandState *rand_state
    );

    vec3 ambient, diffuse, specular, emission, albedo;
    vec3 **texture;
    int texture_width, texture_height;
    float n_s;
};

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_, vec3 albedo_,
  float n_s_, int material_image_height, int material_image_width,
  vec3 **texture_
) {
  this -> ambient = ambient_;
  this -> diffuse = diffuse_;
  this -> specular = specular_;
  this -> albedo = albedo_;
  this -> emission = emission_;
  this -> n_s = n_s_;
  this -> texture_height = material_image_height;
  this -> texture_width = material_image_width;
  this -> texture = texture_;

  this -> diffuse_mag = diffuse_.length();
  this -> specular_mag = specular_.length();
}

__device__ reflection_record Material::get_reflection_ray(
  Ray coming_ray, vec3 hit_point, vec3 normal, curandState *rand_state
) {
  CartesianSystem new_xyz_system;
  vec3 v3_rand = get_random_unit_vector_hemisphere(rand_state), v3_rand_world;
  vec3 reflected_ray_dir;
  float cos_theta, random_number = curand_uniform(&rand_state[0]);
  reflection_record new_reflection_record;
  float factor = \
    this -> diffuse_mag / (this -> diffuse_mag + this -> specular_mag);

  if (random_number <= factor) {
    new_xyz_system = CartesianSystem(normal);
    cos_theta = v3_rand.z();
    v3_rand_world = new_xyz_system.to_world_system(v3_rand);
    new_reflection_record.ray = Ray(hit_point, v3_rand_world);
    new_reflection_record.cos_theta = cos_theta;
  } else {
    reflected_ray_dir = reflect(coming_ray.dir, normal);
    new_xyz_system = CartesianSystem(reflected_ray_dir);
    v3_rand_world = new_xyz_system.to_world_system(v3_rand);
    reflected_ray_dir = unit_vector(
      (1 - this -> n_s / 1000) * reflected_ray_dir +
      (this -> n_s / 1000) * v3_rand_world
    );
    new_reflection_record.ray = Ray(hit_point, reflected_ray_dir);
    new_reflection_record.cos_theta = dot(reflected_ray_dir, normal);
  }

  return new_reflection_record;
}

__device__ vec3 Material::get_texture(vec3 uv_vector) {
  if (this -> texture_width * this -> texture_height > 0) {
    int idx_u = floorf((uv_vector.u() - floorf(uv_vector.u())) * (this -> texture_width - 1));
    int idx_v = floorf((uv_vector.v() - floorf(uv_vector.v())) * (this -> texture_height - 1));

    int idx = this -> texture_width * idx_v + idx_u;

    vec3 selected_texture = vec3(
      this -> texture[idx] -> r(),
      this -> texture[idx] -> g(),
      this -> texture[idx] -> b()
    );

    return selected_texture * this -> diffuse;
  } else {
    return this -> diffuse;
  }
}

#endif
