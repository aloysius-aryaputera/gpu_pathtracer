#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/vector_and_matrix/vec3.h"

__device__ vec3 get_random_unit_vector_hemisphere(curandState *rand_state);

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

#endif
