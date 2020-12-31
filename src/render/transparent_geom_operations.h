//File: transparent_geom_operations.h
#ifndef TRANSPARENT_GEOM_OPERATIONS_H
#define TRANSPARENT_GEOM_OPERATIONS_H

__global__ void compute_num_transparent_geom(
  Primitive** geom_array, int num_geom, int *num_transparent_geom
);

__global__ void collect_transparent_geom(
  Primitive** geom_array, int num_geom, Primitive** transparent_geom_array
);

__global__ void compute_num_transparent_geom(
  Primitive** geom_array, int num_geom, int *num_transparent_geom
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > 0) return;

  num_transparent_geom[0] = 0;
  for (int i = 0; i < num_geom; i++) {
    if (geom_array[i] -> is_transparent_geom()) {
      num_transparent_geom[0]++;
    }
  }
}

__global__ void collect_transparent_geom(
  Primitive** geom_array, int num_geom, Primitive** transparent_geom_array
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > 0) return;

  int transparent_geom_idx = 0;

  for (int i = 0; i < num_geom; i++) {
    if (geom_array[i] -> is_transparent_geom()) {
      transparent_geom_array[transparent_geom_idx++] = geom_array[i];
    }
  }
}

#endif
