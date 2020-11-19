#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/vector_and_matrix/vec3.h"
#include "../param.h"

__device__ float silverman_biweight_kernel(float x);
__device__ float henyey_greenstein_pdf(float g, vec3 dir_1, vec3 dir_2);
__device__ float henyey_greenstein_cos_theta(float g, curandState *rand_state);
__device__ vec3 get_random_unit_vector_hemisphere_cos_pdf(
  curandState *rand_state);
__device__ vec3 get_random_unit_vector_hemisphere(curandState *rand_state);
__device__ vec3 get_random_unit_vector_phong(curandState *rand_state);
__device__ vec3 get_random_unit_vector_disk(curandState *rand_state);
__device__ vec3 compute_phong_filter(
  vec3 k, float n, vec3 ideal_dir, vec3 dir
);
__device__ vec3 compute_phong_filter_2(
  vec3 k, float n, vec3 ideal_dir, vec3 dir
);
__device__ vec3 reflect(vec3 v, vec3 normal);
__device__ float compute_schlick_specular(float cos_theta);
__device__ float compute_diffuse_sampling_pdf(
  vec3 normal, vec3 reflected_dir
);
__device__ float compute_specular_sampling_pdf(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n, bool refracted
);
__device__ float _compute_reflection_sampling_pdf(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
);
__device__ float _compute_refraction_sampling_pdf(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
);
__device__ float compute_specular_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n, bool refracted
);
__device__ float _compute_reflection_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
);
__device__ float _compute_refraction_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
);

__device__ float silverman_biweight_kernel(float x) {
  return (3 / M_PI) * powf((1 - x * x), 2);
}

__device__ float henyey_greenstein_pdf(float g, vec3 dir_1, vec3 dir_2) {
  float cos_theta = dot(unit_vector(dir_1), unit_vector(dir_2));
  if (isnan(cos_theta)) {
    cos_theta = 0;
  }
  return (1.0 / (4 * M_PI)) * ((1 - g * g) / powf(
    1 + g * g - 2 * g * cos_theta, 1.5));
}

__device__ float henyey_greenstein_cos_theta(float g, curandState *rand_state) {
  float eta = curand_uniform(rand_state);
  if (abs(g) < SMALL_DOUBLE) {
    return 1 - (2 * eta);
  } else {
    return (-1 / abs(2 * g)) * (
      1 + g * g - powf((1 - g * g) / (1 - g + 2 * g * eta), 2));
  }
}

__device__ float _compute_reflection_sampling_pdf(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
) {
  float dot_prod_1 = dot(in, normal);
  float dot_prod_2 = dot(normal, out);
  if (
    (dot_prod_1 >= 0 && dot_prod_2 <= 0) ||
    (dot_prod_1 <= 0 && dot_prod_2 >= 0)
  ) {
    if (isinf(n)) {
      return MAX_PHONG_N_S / (2 * M_PI);
    } else {
      return (n + 1) * powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
    }
  } else {
    return 0;
  }
}

__device__ float _compute_refraction_sampling_pdf(
	vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
) {
  float dot_prod_1 = dot(in, normal);
  float dot_prod_2 = dot(normal, out);
  if (
    (dot_prod_1 >= 0 && dot_prod_2 >= 0) ||
    (dot_prod_1 <= 0 && dot_prod_2 <= 0)
  ) {
    if (isinf(n)) {
      return MAX_PHONG_N_S / (2 * M_PI);
    } else {
      return (n + 1) * powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
    }
  } else {
    return 0;
  }
}

__device__ float compute_specular_sampling_pdf(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n, bool refracted
) {
  if (refracted) {
    return _compute_refraction_sampling_pdf(in, out, normal, perfect_out, n);
  } else {
    return _compute_reflection_sampling_pdf(in, out, normal, perfect_out, n);
  }
}

__device__ float _compute_reflection_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
) {
  float dot_prod_1 = dot(in, normal);
  float dot_prod_2 = dot(normal, out);
  if (
    (dot_prod_1 >= 0 && dot_prod_2 <= 0) ||
    (dot_prod_1 <= 0 && dot_prod_2 >= 0)
  ) {
    if (isinf(n)) {
      return 1 / (2 * M_PI);
    } else {
      return powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
    }
  } else {
    return 0;
  }
}

__device__ float _compute_refraction_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
) {
  float dot_prod_1 = dot(in, normal);
  float dot_prod_2 = dot(normal, out);
  if (
    (dot_prod_1 >= 0 && dot_prod_2 >= 0) ||
    (dot_prod_1 <= 0 && dot_prod_2 <= 0)
  ) {
    if (isinf(n)) {
      return 1 / (2 * M_PI);
    } else {
      return powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
    }
  } else {
    return 0;
  }
}

__device__ float compute_scattering_pdf(
  vec3 normal, vec3 next_dir, bool diffuse = true, 
  vec3 coming_dir = vec3(1, 0, 0), bool refracted = false
) {
  if (diffuse) {
    return fmaxf(0.0, dot(normal, next_dir));
  } else {
    float dot_prod_1 = dot(coming_dir, normal);
    float dot_prod_2 = dot(next_dir, normal);
    return (dot_prod_1 >= 0 && dot_prod_2 <= 0 && !refracted) ||
      (dot_prod_1 <= 0 && dot_prod_2 >= 0 && !refracted) ||
      (dot_prod_1 >= 0 && dot_prod_2 >= 0 && refracted) ||
      (dot_prod_1 <= 0 && dot_prod_2 <= 0 && refracted);
  }
}

__device__ float compute_sampling_pdf_2(
  vec3 normal, vec3 next_dir, bool diffuse = true, float n = 1,
  vec3 coming_dir = vec3(1, 0, 0), vec3 perfect_next_dir = vec3(1, 0, 0),
  bool refracted = false
) {
  if (diffuse) {
    return compute_diffuse_sampling_pdf(normal, next_dir);
  } else {
    return compute_specular_sampling_pdf_2(
      coming_dir, next_dir, normal, perfect_next_dir, n, refracted
    );
  }
}

__device__ float compute_specular_sampling_pdf_2(
  vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n, bool refracted
) {
  if (refracted) {
    return _compute_refraction_sampling_pdf_2(in, out, normal, perfect_out, n);
  } else {
    return _compute_reflection_sampling_pdf_2(in, out, normal, perfect_out, n);
  }
}

__device__ float compute_diffuse_sampling_pdf(
  vec3 normal, vec3 reflected_dir
) {
  return fmaxf(0.0, dot(normal, reflected_dir) / M_PI);
}

__device__ float compute_schlick_specular(
  float cos_theta, float n_1, float n_2
) {
  float r_0 = powf((n_1 - n_2) / (n_1 + n_2), 2);
  return r_0 + (1 - r_0) * powf(1 - cos_theta, 5);
}

__device__ vec3 reflect(vec3 v, vec3 normal) {
  return v - 2 * dot(v, normal) * normal;
}

__device__ vec3 compute_phong_filter(
  vec3 k, float n, vec3 ideal_dir, vec3 dir
) {
  vec3 filter;
  if (isinf(n)) {
    filter = k * MAX_PHONG_N_S * vec3(1, 1, 1) / 2;
  } else {
    filter = k * (n + 2) * powf(fmaxf(0, dot(ideal_dir, dir)), n) / 2;
  } 
  return filter;
}

__device__ vec3 compute_phong_filter_2(
  vec3 k, float n, vec3 ideal_dir, vec3 dir
) {
  vec3 filter;
  if (isinf(n)) {
    filter = k;
  } else {
    filter = k * powf(fmaxf(0, dot(ideal_dir, dir)), n);
  } 
  return filter;
}

__device__ vec3 get_random_unit_vector_phong(float n, curandState *rand_state) {
  vec3 output_vector;
  if (isinf(n)) {
    output_vector = vec3(0, 0, 1);
  } else {
    float r1 = curand_uniform(&rand_state[0]);
    float r2 = curand_uniform(&rand_state[0]);
    float x = sqrt(1 - powf(r1, 2.0 / (n + 1))) * cos(2 * M_PI * r2);
    float y = sqrt(1 - powf(r1, 2.0 / (n + 1))) * sin(2 * M_PI * r2);
    float z = powf(r1, 1.0 / (n + 1));
    output_vector = vec3(x, y, z);
    output_vector.make_unit_vector();
  }
  return output_vector;
}

__device__ vec3 get_random_unit_vector_hemisphere(curandState *rand_state) {
  float sin_theta = curand_uniform(&rand_state[0]);
  float cos_theta = sqrt(1 - sin_theta * sin_theta);
  float phi = curand_uniform(&rand_state[0]) * 2 * M_PI;
  vec3 output_vector = vec3(
    sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
  output_vector.make_unit_vector();
  return output_vector;
}

__device__ vec3 get_random_unit_vector_hemisphere_cos_pdf(
  curandState *rand_state
) {
  float r1 = curand_uniform(&rand_state[0]);
  float r2 = curand_uniform(&rand_state[0]);
  float z = sqrt(1 - r2);
  float phi = 2 * M_PI * r1;
  float x = cos(phi) * sqrt(r2);
  float y = sin(phi) * sqrt(r2);

  vec3 output_vector = vec3(x, y, z);
  output_vector.make_unit_vector();

  return output_vector;
}

__device__ vec3 get_random_unit_vector_disk(curandState *rand_state) {
  float sin_theta = 2 * curand_uniform(&rand_state[0]) - 1;
  float cos_theta = sqrt(1 - sin_theta * sin_theta);
  float random_number = curand_uniform(&rand_state[0]);
  if (random_number <= .5) {
    cos_theta *= -1.0;
  }
  vec3 output_vector  = vec3(cos_theta, sin_theta, 0);
  output_vector.make_unit_vector();
  return output_vector;
}

#endif
