#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/vector_and_matrix/vec3.h"

__device__ vec3 get_random_unit_vector_hemisphere(curandState *rand_state);

__device__ vec3 get_random_unit_vector_hemisphere_cos_pdf(curandState *rand_state);

__device__ vec3 get_random_unit_vector_phong(curandState *rand_state);

__device__ vec3 get_random_unit_vector_disk(curandState *rand_state);

__device__ vec3 get_random_unit_vector_phong(float n, curandState *rand_state) {
  float r1 = curand_uniform(&rand_state[0]);
	float r2 = curand_uniform(&rand_state[0]);
	float x = sqrt(1 - powf(r1, 2 / (n + 1))) * cos(2 * M_PI * r2);
	float y = sqrt(1 - powf(r1, 2 / (n + 1))) * sin(2 * M_PI * r2);
  float z = powf(r1, 1 / (n + 1));
	vec3 output_vector = vec3(x, y, z);
	output_vector.make_unit_vector();
	return output_vector;
}

__device__ vec3 get_random_unit_vector_hemisphere_cos_pdf(curandState *rand_state) {
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


__device__ vec3 get_random_unit_vector_hemisphere(curandState *rand_state) {
  float sin_theta = curand_uniform(&rand_state[0]);
  float cos_theta = sqrt(1 - sin_theta * sin_theta);
  float phi = curand_uniform(&rand_state[0]) * 2 * M_PI;
  vec3 output_vector = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
  output_vector.make_unit_vector();

  if (isnan(output_vector.x()) || isnan(output_vector.y()) || isnan(output_vector.z())) {
    printf(
      "sin_theta = %5.5f; cos_theta = %5.5f; phi = %5.5f\n",
      sin_theta, cos_theta, phi
    );
  }

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
