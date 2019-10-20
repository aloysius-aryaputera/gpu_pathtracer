#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/vector_and_matrix/vec3.h"

__device__ vec3 get_random_unit_vector_hemisphere(
  curandState *rand_state
);

__device__ vec3 get_random_unit_vector_hemisphere(
  curandState *rand_state
) {
  float sin_theta = curand_uniform(&rand_state[0]);
  float cos_theta = sqrt(1 - sin_theta * sin_theta);
  float phi = curand_uniform(&rand_state[0]) * 2 * M_PI;

  return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

#endif
