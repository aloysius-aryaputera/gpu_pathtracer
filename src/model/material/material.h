//File: material.h
#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>

#include "../../util/vector_util.h"
#include "../cartesian_system.h"
#include "../ray/ray.h"
#include "../ray/ray_operations.h"
#include "../vector_and_matrix/vec3.h"

struct reflection_record;

__device__ reflection_record _get_false_hit_parameters(
  vec3 hit_point, vec3 v_in, vec3 normal
);

class Material {
  private:
    __device__ vec3 _get_texture(
      vec3 uv_vector, vec3 filter, float* texture_r, float* texture_g,
      float* texture_b, int texture_height, int texture_width
    );
    __device__ float _get_texture_n_s(vec3 uv_vector);
    __device__ void _refract(
      reflection_record &ref,
      vec3 hit_point, 
      vec3 v_in, 
      vec3 normal,
      vec3 uv_vector,
      bool &sss,
      Material *highest_prioritised_material,
      Material *second_highest_prioritised_material,
      curandState *rand_state,
      bool write
    );
    __device__ bool _check_if_false_hit(
      Material** material_list, int material_list_length,
      Material *highest_prioritised_material,
      Material *second_highest_prioritised_material
    );

    float diffuse_mag, specular_mag;
    vec3 ambient, diffuse, specular, transmission;
    int texture_width_diffuse, texture_height_diffuse;
    int texture_width_specular, texture_height_specular;
    int texture_width_emission, texture_height_emission;
    int texture_width_n_s, texture_height_n_s;
    int texture_width_bump, texture_height_bump;
    float n_s, bm;
    float *texture_r_diffuse, *texture_g_diffuse, *texture_b_diffuse;
    float *texture_r_specular, *texture_g_specular, *texture_b_specular;
    float *texture_r_emission, *texture_g_emission, *texture_b_emission;
    float *texture_r_n_s, *texture_g_n_s, *texture_b_n_s;
    float *texture_r_bump, *texture_g_bump, *texture_b_bump;

  public:
    __host__ __device__ Material() {};
    __host__ __device__ Material(
      vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
      vec3 transmission_,
      float path_length_,
      float t_r_, float n_s_, float n_i_, float bm_,
      float scattering_coef_, float absorption_coef_, float g_,
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
      float *texture_b_n_s_,
      int texture_height_bump_,
      int texture_width_bump_,
      float *texture_r_bump_,
      float *texture_g_bump_,
      float *texture_b_bump_
    );
    __device__ void check_next_path(
      Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
      bool &sss,
      Material** material_list, int material_list_length,
      reflection_record &ref, curandState *rand_state, bool write
    );
    __device__ vec3 get_texture_emission(vec3 uv_vector);
    __device__ vec3 get_texture_diffuse(vec3 uv_vector);
    __device__ vec3 get_texture_bump(vec3 uv_vector);
    __device__ vec3 get_texture_specular(vec3 uv_vector);
    __device__ float get_transmittance(float t);
    __device__ vec3 get_new_scattering_direction(
      vec3 current_dir, curandState *rand_state);
    __device__ float get_propagation_distance(curandState *rand_state);
    __device__ float get_phase_function_value(vec3 dir_1, vec3 dir_2);

    vec3 emission;
    int priority;
    float n_i, path_length, t_r;
    float scattering_coef, absorption_coef, extinction_coef, g;
    float scattering_prob;
    bool sub_surface_scattering;
};

struct reflection_record
{
  Ray ray;
  vec3 k;
  vec3 filter, filter_2;
  vec3 perfect_reflection_dir;
  float pdf, n;
  bool diffuse, reflected, refracted, false_hit, entering;
  Material *next_material;
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

__device__ float Material::get_phase_function_value(vec3 dir_1, vec3 dir_2) {
  return henyey_greenstein_pdf(this -> g, dir_1, dir_2);
}

__device__ float Material::get_propagation_distance(curandState *rand_state) {
  float random_number = curand_uniform(&rand_state[0]);
  return - logf(random_number) / this -> extinction_coef;
}

__device__ vec3 Material::get_new_scattering_direction(
  vec3 current_dir, curandState *rand_state
) {
  float cos_theta = henyey_greenstein_cos_theta(this -> g, rand_state);
  float sin_theta = powf(1 - powf(cos_theta, 2), .5);
  float cot_theta = cos_theta / sin_theta;
  CartesianSystem cart_sys = CartesianSystem(current_dir);
  vec3 new_dir = get_random_unit_vector_disk(rand_state);
  float new_dir_z = cot_theta * powf(
    new_dir.x() * new_dir.x() + new_dir.y() * new_dir.y(), .5);
  new_dir = vec3(new_dir.x(), new_dir.y(), new_dir_z);
  new_dir.make_unit_vector();
  return cart_sys.to_world_system(new_dir);
}	

__device__ float Material::get_transmittance(float t) {
  return exp(-t * this -> extinction_coef); 
}

__device__ bool Material::_check_if_false_hit(
  Material** material_list, int material_list_length,
  Material *highest_prioritised_material,
  Material *second_highest_prioritised_material
) {
  highest_prioritised_material = nullptr;
  second_highest_prioritised_material = nullptr;
  int highest_prioritised_material_priority = 99999;

  find_highest_prioritised_materials(
    material_list, material_list_length,
    highest_prioritised_material,
    second_highest_prioritised_material
  );

  highest_prioritised_material_priority = get_material_priority(
    highest_prioritised_material);

  if (this -> priority > highest_prioritised_material_priority) {
    return true;
  } else {
    return false;
  }
}

__device__ void Material::_refract(
  reflection_record &ref,
  vec3 hit_point, vec3 v_in, vec3 normal,
  vec3 uv_vector,
  bool &sss,
  Material *highest_prioritised_material,
  Material *second_highest_prioritised_material,
  curandState *rand_state,
  bool write=false
) {
  //reflection_record ref;
  float random_number = curand_uniform(&rand_state[0]);

  float highest_prioritised_material_ref_idx = get_material_refraction_index(
    highest_prioritised_material);
  float second_highest_prioritised_material_ref_idx = \
    get_material_refraction_index(second_highest_prioritised_material);

  vec3 k = this -> transmission * this -> t_r, v_out;
  float local_n_s = this -> _get_texture_n_s(uv_vector);
  ref.n = local_n_s;
  sss = false;

  if (dot(v_in, normal) <= 0) {
    float cos_theta_1 = dot(v_in, -normal);
    float reflection_probability = compute_schlick_specular(
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

      if (abs(tan_theta_2) > SMALL_DOUBLE) {
        vec3 v_out_perpendicular = \
          - 1 / tan_theta_2 * v_in_parallel.length() * normal;
        v_out = v_in_parallel + v_out_perpendicular;
        v_out.make_unit_vector();
      } else {
	v_out = -normal;
      }

      ref.perfect_reflection_dir = v_out;
      ref.diffuse = false;
      ref.reflected = false;
      ref.refracted = true;
      ref.false_hit = false;
      ref.entering = true;
      ref.next_material = this;
    } else {
      v_out = reflect(v_in, normal);
      v_out.make_unit_vector();
      ref.perfect_reflection_dir = v_out;

      ref.diffuse = false;
      ref.reflected = true;
      ref.refracted = false;
      ref.false_hit = false;
      ref.entering = false;
      ref.next_material = highest_prioritised_material;
    }
  } else {

    float sin_theta_1_max = \
      second_highest_prioritised_material_ref_idx / this -> n_i;
    float cos_theta_1 = dot(v_in, normal);
    float sin_theta_1 = powf(1 - powf(cos_theta_1, 2), .5);
    float reflection_probability = compute_schlick_specular(
      cos_theta_1, this -> n_i, second_highest_prioritised_material_ref_idx);

    if (
      sin_theta_1 >= sin_theta_1_max | random_number <= reflection_probability
    ) {
      v_out = reflect(v_in, -normal);
      v_out.make_unit_vector();
      ref.perfect_reflection_dir = v_out;
      ref.diffuse = false;
      ref.reflected = true;
      ref.refracted = false;
      ref.false_hit = false;
      ref.entering = false;
      ref.next_material = this;
    } else {
      vec3 v_in_perpendicular = cos_theta_1 * normal;
      vec3 v_in_parallel = v_in - v_in_perpendicular;
      float sin_theta_2 = \
        this -> n_i / second_highest_prioritised_material_ref_idx * sin_theta_1;
      float cos_theta_2 = powf(1 - powf(sin_theta_2, 2), .5);
      float tan_theta_2 = sin_theta_2 / cos_theta_2;

      if (abs(tan_theta_2) > SMALL_DOUBLE) {
      	vec3 v_out_perpendicular = \
          1 / tan_theta_2 * v_in_parallel.length() * normal;
      	v_out = v_in_parallel + v_out_perpendicular;
      	v_out.make_unit_vector();
      } else {
	v_out = normal;
      }

      ref.perfect_reflection_dir = v_out;
      ref.diffuse = false;
      ref.reflected = false;
      ref.refracted = true;
      ref.false_hit = false;
      ref.entering = false;
      ref.next_material = second_highest_prioritised_material;
    }
  }

  ref.ray = generate_ray(hit_point, v_out, normal, 1, local_n_s, rand_state);
  ref.k = k;
  ref.filter = compute_phong_filter(k, local_n_s, v_out, ref.ray.dir);
  ref.filter_2 = compute_phong_filter_2(k, local_n_s, v_out, ref.ray.dir);

  float sampling_pdf = compute_sampling_pdf_2(
    normal, ref.ray.dir, ref.diffuse, ref.n, v_in, ref.perfect_reflection_dir,
    ref.refracted
  );
  float scattering_pdf = compute_scattering_pdf(
    normal, ref.ray.dir, ref.diffuse, v_in, ref.refracted
  );
  ref.pdf = sampling_pdf * M_PI / scattering_pdf; 

  //return ref;
}

__host__ __device__ Material::Material(
  vec3 ambient_, vec3 diffuse_, vec3 specular_, vec3 emission_,
  vec3 transmission_,
  float path_length_,
  float t_r_,
  float n_s_,
  float n_i_,
  float bm_,
  float scattering_coef_,
  float absorption_coef_,
  float g_,
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
  float *texture_b_n_s_,
  int texture_height_bump_,
  int texture_width_bump_,
  float *texture_r_bump_,
  float *texture_g_bump_,
  float *texture_b_bump_
) {
  this -> ambient = ambient_;
  this -> diffuse = diffuse_;
  this -> specular = specular_;
  this -> emission = emission_;
  this -> transmission = transmission_;
  this -> path_length = path_length_;
  this -> n_i = n_i_;
  this -> t_r = t_r_;
  if (n_s_ >= MAX_PHONG_N_S && this -> t_r > 0) 
    this -> n_s = INFINITY;
  else
    this -> n_s = n_s_;
  this -> bm = bm_;
  this -> scattering_coef = scattering_coef_;
  this -> absorption_coef = absorption_coef_;
  this -> extinction_coef = scattering_coef_ + absorption_coef_;
  this -> g = g_;
  this -> scattering_prob = this -> scattering_coef / this -> extinction_coef;
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

  this -> texture_height_bump = texture_height_bump_;
  this -> texture_width_bump = texture_width_bump_;
  this -> texture_r_bump = texture_r_bump_;
  this -> texture_g_bump = texture_g_bump_;
  this -> texture_b_bump = texture_b_bump_;

  this -> diffuse_mag = diffuse_.length();
  this -> specular_mag = specular_.length();

  this -> sub_surface_scattering = (this -> path_length > 0);

  if (this -> sub_surface_scattering)
    printf("The material is SSS\n");
}


__device__ reflection_record _get_false_hit_parameters(
  vec3 hit_point, vec3 v_in, vec3 normal
) {
  reflection_record ref;
  ref.false_hit = true;
  ref.reflected = false;
  ref.refracted = true;
  ref.ray = Ray(hit_point, v_in);
  ref.filter = vec3(1.0, 1.0, 1.0);
  ref.filter_2 = vec3(1.0, 1.0, 1.0);
  ref.pdf = 1;
  ref.diffuse = false;
  if (dot(v_in, normal) <= 0) {
    ref.entering = true;
  } else {
    ref.entering = false;
  }
  return ref;
}

__device__ void Material::check_next_path(
  Ray coming_ray, vec3 hit_point, vec3 normal, vec3 uv_vector,
  bool &sss,
  Material** material_list, int material_list_length,
  reflection_record &ref, curandState *rand_state, bool write=false
) {

  Material *highest_prioritised_material = nullptr;
  Material *second_highest_prioritised_material = nullptr;
  vec3 v_in = coming_ray.dir;

  ref.false_hit = this -> _check_if_false_hit(
    material_list, material_list_length,
    highest_prioritised_material,
    second_highest_prioritised_material
  );

  if (ref.false_hit) {
    sss = false;
    ref = _get_false_hit_parameters(hit_point, v_in, normal);
    return;
  }

  if (
    this -> t_r > 0 && (
      dot(v_in, normal) <= 0 ||
      second_highest_prioritised_material == nullptr ||
      second_highest_prioritised_material -> t_r > 0
    )
  ) {
    this -> _refract(
      ref,
      hit_point, 
      coming_ray.dir, 
      normal,
      uv_vector,
      sss,
      highest_prioritised_material,
      second_highest_prioritised_material,
      rand_state,
      write
    );
    return;
  }

  Material* actual_mat;
  vec3 reflected_ray_dir;
  float random_number = curand_uniform(&rand_state[0]);

  if (this -> t_r > 0) {
    actual_mat = second_highest_prioritised_material;
  } else {
    actual_mat = this;
  }

  float kd_length = actual_mat -> get_texture_diffuse(uv_vector).length();
  float ks_length = actual_mat -> get_texture_specular(uv_vector).length();
  float factor = ks_length / (kd_length + ks_length);
  float local_n_s = actual_mat -> _get_texture_n_s(uv_vector);
  vec3 k;

  if (random_number > factor) {
    ref.ray = generate_ray(
      hit_point, vec3(0, 0, 0), normal, 0, 1, rand_state);
    ref.filter = actual_mat -> get_texture_diffuse(uv_vector);
    ref.filter_2 = ref.filter;
    ref.diffuse = true;
    ref.reflected = false;
    ref.refracted = false;
    ref.k = ref.filter;
    ref.n = 1;

    if (actual_mat -> sub_surface_scattering) {
      sss = true;
    } else {
      sss = false;
    }

    return;
  } else {
    reflected_ray_dir = reflect(v_in, normal);
    ref.ray = generate_ray(
      hit_point, reflected_ray_dir, normal, 1, local_n_s, rand_state);
    k = actual_mat -> get_texture_specular(uv_vector);
    ref.filter = compute_phong_filter(
      k, local_n_s, reflected_ray_dir, ref.ray.dir);
    ref.filter_2 = compute_phong_filter_2(
      k, local_n_s, reflected_ray_dir, ref.ray.dir);

    ref.diffuse = false;
    ref.reflected = true;
    ref.refracted = false;
    ref.perfect_reflection_dir = reflected_ray_dir;
    ref.n = local_n_s;
    ref.k = k;
    sss = false;
  }

  float sampling_pdf = compute_sampling_pdf_2(
    normal, ref.ray.dir, ref.diffuse, local_n_s, v_in, 
    ref.perfect_reflection_dir, ref.refracted
  );
  float scattering_pdf = compute_scattering_pdf(
    normal, ref.ray.dir, ref.diffuse, v_in, ref.refracted
  );
  ref.pdf = sampling_pdf * M_PI / scattering_pdf;

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

__device__ vec3 Material::get_texture_bump(vec3 uv_vector) {
  if (this -> texture_height_bump < 2 || this -> texture_width_bump < 2) {
    return vec3(0.0, 0.0, 0.0);
  } else {
    return 2 * this -> bm * (
      this -> _get_texture(
        uv_vector, vec3(1.0, 1.0, 1.0), this -> texture_r_bump,
        this -> texture_g_bump, this -> texture_b_bump,
        this -> texture_height_bump, this -> texture_width_bump
      ) - vec3(0.5, 0.5, 0.5)
    );
  }
}

__device__ vec3 Material::get_texture_specular(vec3 uv_vector) {
  return this -> _get_texture(
    uv_vector, this -> specular, this -> texture_r_specular,
    this -> texture_g_specular, this -> texture_b_specular,
    this -> texture_height_specular, this -> texture_width_specular
  );
}

__device__ float Material::_get_texture_n_s(vec3 uv_vector) {
  if (isinf(this -> n_s))
    return this -> n_s;
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
