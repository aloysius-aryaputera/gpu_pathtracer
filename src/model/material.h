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
  vec3 color;
  char material_type;
};

class Material {
  private:
    __device__ vec3 _get_texture(
      vec3 uv_vector, vec3 filter, float* texture_r, float* texture_g,
      float* texture_b, int texture_height, int texture_width
    );
    __device__ vec3 _get_texture_diffuse(vec3 uv_vector);
    __device__ vec3 _get_texture_specular(vec3 uv_vector);
    __device__ float _get_texture_n_s(vec3 uv_vector);

    float diffuse_mag, specular_mag;
    vec3 ambient, diffuse, specular, transmission;
    int texture_width_diffuse, texture_height_diffuse;
    int texture_width_specular, texture_height_specular;
    int texture_width_n_s, texture_height_n_s;
    float t_r, n_s, n_i;
    float *texture_r_diffuse, *texture_g_diffuse, *texture_b_diffuse;
    float *texture_r_specular, *texture_g_specular, *texture_b_specular;
    float *texture_r_n_s, *texture_g_n_s, *texture_b_n_s;

  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
      vec3 transmission_,
      float t_r_, float n_s_, float n_i_,
      int texture_height_diffuse_,
      int texture_width_diffuse_,
      float *texture_r_diffuse_,
      float *texture_g_diffuse_,
      float *texture_b_diffuse_,
      int texture_height_specular_,
      int texture_width_specular_,
      float *texture_r_specular_,
      float *texture_g_specular_,
      float *texture_b_specular_,
      int texture_height_n_s_,
      int texture_width_n_s_,
      float *texture_r_n_s_,
      float *texture_g_n_s_,
      float *texture_b_n_s_
    );
    __device__ bool is_reflected_or_refracted(
      Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
      reflection_record &ref, curandState *rand_state
    );

    vec3 emission;
};

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
  vec3 transmission_,
  float t_r_,
  float n_s_,
  float n_i_,
  int texture_height_diffuse_,
  int texture_width_diffuse_,
  float *texture_r_diffuse_,
  float *texture_g_diffuse_,
  float *texture_b_diffuse_,
  int texture_height_specular_,
  int texture_width_specular_,
  float *texture_r_specular_,
  float *texture_g_specular_,
  float *texture_b_specular_,
  int texture_height_n_s_,
  int texture_width_n_s_,
  float *texture_r_n_s_,
  float *texture_g_n_s_,
  float *texture_b_n_s_
) {
  this -> ambient = ambient_;
  this -> diffuse = diffuse_;
  this -> specular = specular_;
  this -> emission = emission_;
  this -> transmission = transmission_;
  this -> n_s = n_s_;
  this -> n_i = n_i_;
  this -> t_r = t_r_;

  this -> texture_height_diffuse = texture_height_diffuse_;
  this -> texture_width_diffuse = texture_width_diffuse_;
  this -> texture_r_diffuse = texture_r_diffuse_;
  this -> texture_g_diffuse = texture_g_diffuse_;
  this -> texture_b_diffuse = texture_b_diffuse_;

  this -> texture_height_specular = texture_height_specular_;
  this -> texture_width_specular = texture_width_specular_;
  this -> texture_r_specular = texture_r_specular_;
  this -> texture_g_specular = texture_g_specular_;
  this -> texture_b_specular = texture_b_specular_;

  this -> texture_height_n_s = texture_height_n_s_;
  this -> texture_width_n_s = texture_width_n_s_;
  this -> texture_r_n_s = texture_r_n_s_;
  this -> texture_g_n_s = texture_g_n_s_;
  this -> texture_b_n_s = texture_b_n_s_;

  this -> diffuse_mag = diffuse_.length();
  this -> specular_mag = specular_.length();
}

__device__ bool Material::is_reflected_or_refracted(
  Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
  reflection_record &ref, curandState *rand_state
) {
  CartesianSystem new_xyz_system = CartesianSystem(normal);
  vec3 v3_rand = get_random_unit_vector_hemisphere(rand_state);
  vec3 v3_rand_world = new_xyz_system.to_world_system(v3_rand);
  vec3 reflected_ray_dir;
  float cos_theta, random_number = curand_uniform(&rand_state[0]);
  float factor = \
    this -> diffuse_mag / (this -> diffuse_mag + this -> specular_mag);
  float fuziness, local_n_s = this -> _get_texture_n_s(uv_vector);

  if (local_n_s == 0) {
    fuziness = 99999;
  } else {
    fuziness = 1 / local_n_s;
  }

  if (random_number <= factor) {
    cos_theta = v3_rand.z();
    ref.ray = Ray(hit_point, v3_rand_world);
    ref.cos_theta = cos_theta;
    ref.color = this -> _get_texture_diffuse(uv_vector);
    ref.material_type = 'd';
    return true;
  } else {
    reflected_ray_dir = reflect(coming_ray.dir, normal);
    reflected_ray_dir = unit_vector(
      reflected_ray_dir + fuziness * v3_rand_world
    );
    cos_theta = dot(reflected_ray_dir, normal);
    ref.ray = Ray(hit_point, reflected_ray_dir);
    ref.cos_theta = cos_theta;
    ref.color = this -> _get_texture_specular(uv_vector);
    ref.material_type = 's';
    if (cos_theta <= 0) {
      return false;
    } else {
      return true;
    }
  }

  return false;
}

__device__ vec3 Material::_get_texture(
  vec3 uv_vector, vec3 filter, float* texture_r, float* texture_g,
  float* texture_b, int texture_height, int texture_width
) {
  int idx_u = floorf(
    (uv_vector.u() - floorf(uv_vector.u())) * (texture_width - 1));
  int idx_v = floorf(
    (uv_vector.v() - floorf(uv_vector.v())) * (texture_height - 1));

  int idx = texture_width * idx_v + idx_u;

  vec3 selected_texture = vec3(
    texture_r[idx],
    texture_g[idx],
    texture_b[idx]
  );

  return selected_texture * filter;
}

__device__ vec3 Material::_get_texture_diffuse(vec3 uv_vector) {
  return this -> _get_texture(
    uv_vector, this -> diffuse, this -> texture_r_diffuse,
    this -> texture_g_diffuse, this -> texture_b_diffuse,
    this -> texture_height_diffuse, this -> texture_width_diffuse
  );
}

__device__ vec3 Material::_get_texture_specular(vec3 uv_vector) {
  return this -> _get_texture(
    uv_vector, this -> specular, this -> texture_r_specular,
    this -> texture_g_specular, this -> texture_b_specular,
    this -> texture_height_specular, this -> texture_width_specular
  );
}

__device__ float Material::_get_texture_n_s(vec3 uv_vector) {
  vec3 filter = vec3(
    this -> n_s / powf(3, .5), this -> n_s / powf(3, .5),
    this -> n_s / powf(3, .5));
  vec3 n_s_vector =  this -> _get_texture(
    uv_vector, filter,
    this -> texture_r_n_s,
    this -> texture_g_n_s,
    this -> texture_b_n_s,
    this -> texture_height_n_s,
    this -> texture_width_n_s
  );
  return n_s_vector.length();
}

#endif
