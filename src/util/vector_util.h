#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/vector_and_matrix/vec3.h"

__device__ vec3 get_random_unit_vector_hemisphere_cos_pdf(
	curandState *rand_state);
__device__ vec3 get_random_unit_vector_phong(curandState *rand_state);
__device__ vec3 get_random_unit_vector_disk(curandState *rand_state);
__device__ vec3 compute_phong_filter(
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

__device__ float _compute_reflection_sampling_pdf(
	vec3 in, vec3 out, vec3 normal, vec3 perfect_out, float n
) {
	float dot_prod_1 = dot(in, normal);
	float dot_prod_2 = dot(normal, out);
  if (
		(dot_prod_1 >= 0 && dot_prod_2 <= 0) ||
		(dot_prod_1 <= 0 && dot_prod_2 >= 0)
	) {
		return (n + 1) * powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
	}	else {
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
		return (n + 1) * powf(fmaxf(0.0, dot(perfect_out, out)), n) / (2 * M_PI);
	}	else {
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
	vec3 filter = k * (n + 2) * powf(fmaxf(0, dot(ideal_dir, dir)), n) / 2;
	//if (filter.vector_is_nan()) {
  //   printf("filter = [%f, %f, %f], dot_prod = %f, dot_prod_2 = %f, powf = %f, k = [%f, %f, %f], n = %f, ideal_dir = [%f, %f, %f], dir = [%f, %f, %f]\n",
	//			 filter.r(), filter.g(), filter.b(), 
	//			 dot(ideal_dir, dir),
	//			 fmaxf(0, dot(ideal_dir, dir)),
	//			 powf(fmaxf(0, dot(ideal_dir, dir)), n),
	//			 k.r(), k.g(), k.b(),
	//			 n,
	//			 ideal_dir.r(), ideal_dir.g(), ideal_dir.b(),
	//			 dir.r(), dir.g(), dir.b()
	//			 );
	//}
  return filter; 
}

__device__ vec3 get_random_unit_vector_phong(float n, curandState *rand_state) {
  float r1 = curand_uniform(&rand_state[0]);
	float r2 = curand_uniform(&rand_state[0]);
	float x = sqrt(1 - powf(r1, 2.0 / (n + 1))) * cos(2 * M_PI * r2);
	float y = sqrt(1 - powf(r1, 2.0 / (n + 1))) * sin(2 * M_PI * r2);
  float z = powf(r1, 1.0 / (n + 1));
	vec3 output_vector = vec3(x, y, z);
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
