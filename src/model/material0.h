//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>

#include "../util/vector_util.h"
#include "cartesian_system.h"
#include "ray/ray.h"
#include "ray/ray_operations.h"
#include "vector_and_matrix/vec3.h"

struct reflection_record
{
  Ray ray;
  vec3 filter;
};

__device__ vec3 reflect(vec3 v, vec3 normal);

__device__ float _compute_schlick_specular(float cos_theta);

__device__ vec3 reflect(vec3 v, vec3 normal) {
  return v - 2 * dot(v, normal) * normal;
}

__device__ float _compute_schlick_specular(
  float cos_theta, float n_1, float n_2
) {
  float r_0 = powf((n_1 - n_2) / (n_1 + n_2), 2);
  return r_0 + (1 - r_0) * powf(1 - cos_theta, 5);
}

class Material {
  private:
    __device__ vec3 _get_texture(
      vec3 uv_vector, vec3 filter, float* texture_r, float* texture_g,
      float* texture_b, int texture_height, int texture_width
    );
    __device__ vec3 _get_texture_specular(vec3 uv_vector);
    __device__ float _get_texture_n_s(vec3 uv_vector);
    __device__ reflection_record _refract(
      vec3 hit_point, vec3 v_in, vec3 normal,
      bool &reflected, bool &false_hit, bool &refracted,
      bool &entering, bool &sss,
      Material** material_list, int material_list_length,
      curandState *rand_state
    );

    float diffuse_mag, specular_mag;
    vec3 ambient, diffuse, specular, transmission;
    int texture_width_diffuse, texture_height_diffuse;
    int texture_width_specular, texture_height_specular;
    int texture_width_emission, texture_height_emission;
    int texture_width_n_s, texture_height_n_s;
    float t_r, n_s;
    float *texture_r_diffuse, *texture_g_diffuse, *texture_b_diffuse;
    float *texture_r_specular, *texture_g_specular, *texture_b_specular;
    float *texture_r_emission, *texture_g_emission, *texture_b_emission;
    float *texture_r_n_s, *texture_g_n_s, *texture_b_n_s;

  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
      vec3 transmission_,
      float path_length_,
      float t_r_, float n_s_, float n_i_,
      int priority_,
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
      int texture_height_emission_,
      int texture_width_emission_,
      float *texture_r_emission_,
      float *texture_g_emission_,
      float *texture_b_emission_,
      int texture_height_n_s_,
      int texture_width_n_s_,
      float *texture_r_n_s_,
      float *texture_g_n_s_,
      float *texture_b_n_s_
    );
    __device__ void check_next_path(
      Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
      bool &reflected, bool &false_hit, bool &refracted,
      bool &entering, bool &sss,
      Material **material_list, int material_list_length,
      reflection_record &ref, curandState *rand_state
    );
    __device__ vec3 get_texture_emission(vec3 uv_vector);
    __device__ vec3 get_texture_diffuse(vec3 uv_vector);

    vec3 emission;
    int priority;
    float n_i, path_length;
    bool sub_surface_scattering;
};

__device__ int get_material_priority(Material* material) {
  if (material == nullptr) {
    return 9999999;
  } else {
    return material -> priority;
  }
}

__device__ float get_material_refraction_index(Material* material) {
  if (material == nullptr) {
    return 1.0;
  } else {
    return material -> n_i;
  }
}

__device__ void find_highest_prioritised_materials(
  Material** material_list, int material_list_length,
  Material* highest_prioritised_material,
  Material* second_highest_prioritised_material
) {
  highest_prioritised_material = nullptr;
  second_highest_prioritised_material = nullptr;
  for (int idx = material_list_length - 1; idx >= 0; idx--) {
    if (
      get_material_priority(material_list[idx]) <
        get_material_priority(highest_prioritised_material)
    ) {
      highest_prioritised_material = material_list[idx];
    }

    if (
      (get_material_priority(material_list[idx]) <
        get_material_priority(second_highest_prioritised_material)) &&
      (get_material_priority(material_list[idx]) >=
        get_material_priority(highest_prioritised_material))
    ) {
      second_highest_prioritised_material = material_list[idx];
    }
  }
}

__device__ reflection_record Material::_refract(
  vec3 hit_point, vec3 v_in, vec3 normal,
  bool &reflected, bool &false_hit, bool &refracted,
  bool &entering, bool &sss,
  Material** material_list, int material_list_length,
  curandState *rand_state
) {
  reflection_record ref;
  float random_number = curand_uniform(&rand_state[0]);

  Material *highest_prioritised_material = nullptr;
  Material *second_highest_prioritised_material = nullptr;
  float highest_prioritised_material_ref_idx = 1.0;
  float second_highest_prioritised_material_ref_idx = 1.0;
  int highest_prioritised_material_priority = 99999;
  // int second_highest_prioritised_material_priority = 99999;

  find_highest_prioritised_materials(
    material_list, material_list_length,
    highest_prioritised_material,
    second_highest_prioritised_material
  );

  highest_prioritised_material_ref_idx = get_material_refraction_index(
    highest_prioritised_material);
  second_highest_prioritised_material_ref_idx = \
    get_material_refraction_index(
      second_highest_prioritised_material);

  highest_prioritised_material_priority = get_material_priority(
    highest_prioritised_material);
  // second_highest_prioritised_material_priority = get_material_priority(
  //   second_highest_prioritised_material);

  if (this -> priority > highest_prioritised_material_priority) {
    false_hit = true;
    reflected = false;
    refracted = true;
    sss = false;
    ref.ray = Ray(hit_point, v_in);
    ref.filter = vec3(1.0, 1.0, 1.0);

    if (dot(v_in, normal) <= 0) {
      entering = true;
    } else {
      entering = false;
    }

    return ref;
  }

  if (dot(v_in, normal) <= 0) {
    float cos_theta_1 = dot(v_in, -normal);
    float reflection_probability = _compute_schlick_specular(
      cos_theta_1, highest_prioritised_material_ref_idx, this -> n_i
    );

    if (random_number >= reflection_probability) {
      float sin_theta_1 = powf(1 - powf(cos_theta_1, 2), .5);
      vec3 v_in_perpendicular = - cos_theta_1 * normal;
      vec3 v_in_parallel = v_in - v_in_perpendicular;
      float sin_theta_2 = \
        highest_prioritised_material_ref_idx / this -> n_i * sin_theta_1;
      float cos_theta_2 = powf(1 - powf(sin_theta_2, 2), .5);
      float tan_theta_2 = sin_theta_2 / cos_theta_2;
      vec3 v_out_perpendicular = \
        - 1 / tan_theta_2 * v_in_parallel.length() * normal;
      vec3 v_out = v_in_parallel + v_out_perpendicular;
      v_out.make_unit_vector();
      Ray ray_out = Ray(hit_point, v_out);
      ref.ray = ray_out;
      ref.filter = this -> transmission * this -> t_r;

      reflected = false;
      refracted = true;
      false_hit = false;
      sss = false;
      entering = true;

    } else {
      vec3 v_out = reflect(v_in, normal);
      ref.ray = Ray(hit_point, v_out);
      ref.filter = this -> transmission * this -> t_r;

      reflected = true;
      refracted = false;
      false_hit = false;
      sss = false;
      entering = false;

    }
  } else {

    float sin_theta_1_max = \
      second_highest_prioritised_material_ref_idx / this -> n_i;
    float cos_theta_1 = dot(v_in, normal);
    float sin_theta_1 = powf(1 - powf(cos_theta_1, 2), .5);
    float reflection_probability = _compute_schlick_specular(
      cos_theta_1, this -> n_i, second_highest_prioritised_material_ref_idx);
    if (
      sin_theta_1 >= sin_theta_1_max | random_number <= reflection_probability
    ) {
      vec3 v_out = reflect(v_in, -normal);
      ref.ray = Ray(hit_point, v_out);
      ref.filter = this -> transmission * this -> t_r;

      reflected = true;
      refracted = false;
      false_hit = false;
      entering = false;

    }else {
      vec3 v_in_perpendicular = cos_theta_1 * normal;
      vec3 v_in_parallel = v_in - v_in_perpendicular;
      float sin_theta_2 = \
        this -> n_i / second_highest_prioritised_material_ref_idx * sin_theta_1;
      float cos_theta_2 = powf(1 - powf(sin_theta_2, 2), .5);
      float tan_theta_2 = sin_theta_2 / cos_theta_2;
      vec3 v_out_perpendicular = \
        1 / tan_theta_2 * v_in_parallel.length() * normal;
      vec3 v_out = v_in_parallel + v_out_perpendicular;
      v_out.make_unit_vector();
      Ray ray_out = Ray(hit_point, v_out);
      ref.ray = ray_out;
      ref.filter = this -> transmission * this -> t_r;

      reflected = false;
      refracted = true;
      false_hit = false;
      entering = false;
    }
  }
  return ref;
}

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
  vec3 transmission_,
  float path_length_,
  float t_r_,
  float n_s_,
  float n_i_,
  int priority_,
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
  int texture_height_emission_,
  int texture_width_emission_,
  float *texture_r_emission_,
  float *texture_g_emission_,
  float *texture_b_emission_,
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
  this -> path_length = path_length_;
  this -> n_s = n_s_;
  this -> n_i = n_i_;
  this -> t_r = t_r_;
  this -> priority = priority_;

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

  this -> texture_height_emission = texture_height_emission_;
  this -> texture_width_emission = texture_width_emission_;
  this -> texture_r_emission = texture_r_emission_;
  this -> texture_g_emission = texture_g_emission_;
  this -> texture_b_emission = texture_b_emission_;

  this -> texture_height_n_s = texture_height_n_s_;
  this -> texture_width_n_s = texture_width_n_s_;
  this -> texture_r_n_s = texture_r_n_s_;
  this -> texture_g_n_s = texture_g_n_s_;
  this -> texture_b_n_s = texture_b_n_s_;

  this -> diffuse_mag = diffuse_.length();
  this -> specular_mag = specular_.length();

  this -> sub_surface_scattering = (this -> path_length > 0);

  if (this -> sub_surface_scattering)
    printf("The material is SSS\n");
}

__device__ void Material::check_next_path(
  Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
  bool &reflected, bool &false_hit, bool &refracted,
  bool &entering, bool &sss,
  Material** material_list, int material_list_length,
  reflection_record &ref, curandState *rand_state
) {

  if (this -> t_r > 0) {
    ref = this -> _refract(
      hit_point, coming_ray.dir, normal,
      reflected, false_hit, refracted,
      entering, sss,
      material_list, material_list_length,
      rand_state
    );
    return;
  }

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
    ref.ray = generate_ray(hit_point, vec3(0, 0, 0), normal, 1, rand_state);
    cos_theta = dot(ref.ray.dir, normal);
    ref.filter = this -> get_texture_diffuse(uv_vector) * cos_theta;
    refracted = false;
    reflected = true;
    false_hit = false;

    if (this -> sub_surface_scattering) {
      sss = true;
    } else {
      sss = false;
    }

    return;
  } else {
    reflected_ray_dir = reflect(coming_ray.dir, normal);
    ref.ray = generate_ray(
      hit_point, reflected_ray_dir, normal, fuziness, rand_state);
    cos_theta = dot(ref.ray.dir, normal);
    ref.filter = this -> _get_texture_specular(uv_vector) * cos_theta;
    if (cos_theta <= 0) {
      refracted = false;
      reflected = false;
      false_hit = false;
      sss = false;
      return;
    } else {
      refracted = false;
      reflected = true;
      false_hit = false;
      sss = false;
      return;
    }
  }

  refracted = false;
  reflected = false;
  false_hit = false;
  return;
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

__device__ vec3 Material::get_texture_emission(vec3 uv_vector) {
  return this -> _get_texture(
    uv_vector, this -> emission, this -> texture_r_emission,
    this -> texture_g_emission, this -> texture_b_emission,
    this -> texture_height_emission, this -> texture_width_emission
  );
}


__device__ vec3 Material::get_texture_diffuse(vec3 uv_vector) {
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
