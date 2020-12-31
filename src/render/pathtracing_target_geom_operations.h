//File: pathtracing_target_geom_operations.h
#ifndef PATHTRACING_TARGET_GEOM_OPERATIONS_H
#define PATHTRACING_TARGET_GEOM_OPERATIONS_H

__global__ void compute_num_target_geom(
  Primitive** geom_array, int num_geom, int *num_target_geom
);

__global__ void collect_target_geom(
  Primitive** geom_array, int num_geom, Primitive** target_geom_array
);

__device__ bool _is_target_geom(Primitive *geom);

__device__ bool _is_target_geom(Primitive *geom) {
  return geom -> is_light_source();
}

__global__ void compute_num_target_geom(
  Primitive** geom_array, int num_geom, int *num_target_geom
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > 0) return;

  num_target_geom[0] = 0;
  for (int i = 0; i < num_geom; i++) {
    if (_is_target_geom(geom_array[i])) {
      num_target_geom[0]++;
    }
  }

  printf("There are %d target geometries.\n", num_target_geom[0]);
}

__global__ void collect_target_geom(
  Primitive** geom_array, int num_geom, Primitive** target_geom_array
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > 0) return;

  int target_geom_idx = 0;

  for (int i = 0; i < num_geom; i++) {
    if (_is_target_geom(geom_array[i])) {
      target_geom_array[target_geom_idx++] = geom_array[i];
    }
  }
}

#endif
