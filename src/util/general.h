#ifndef GENERAL_H
#define GENERAL_H

#include <curand_kernel.h>

__global__ void init_curand_state(
  int num_rand_state, curandState *rand_state
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_rand_state) return;

  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, i, 0, &rand_state[i]);
}

__device__ float de_nan(float number) {
  if (!(number == number)) { number = 0; }
  return number;
}

#endif
