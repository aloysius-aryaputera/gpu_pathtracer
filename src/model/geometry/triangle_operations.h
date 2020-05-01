//File: triangle_operations.h
#ifndef TRIANGLE_OPERATIONS_H
#define TRIANGLE_OPERATIONS_H

#include "../vector_and_matrix/vec3.h"
#include "primitive.h"
#include "triangle.h"

__global__ void sum_up_tangent_and_bitangent(
  vec3 *tangent_array, vec3 *bitangent_array, Primitive** geom_array,
  int num_triangles
);

__global__ void assign_tangent(
  vec3 *tangent_array, Primitive** geom_array, int num_triangles
);

__global__ void assign_tangent(
  vec3 *tangent_array, Primitive** geom_array, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles) return;

  int idx_1 = geom_array[idx] -> get_point_1_idx();
  int idx_2 = geom_array[idx] -> get_point_2_idx();
  int idx_3 = geom_array[idx] -> get_point_3_idx();

  geom_array[idx] -> assign_tangent(tangent_array[idx_1], 1);
  geom_array[idx] -> assign_tangent(tangent_array[idx_2], 2);
  geom_array[idx] -> assign_tangent(tangent_array[idx_3], 3);
}

__global__ void sum_up_tangent_and_bitangent(
  vec3 *tangent_array, vec3 *bitangent_array, Primitive** geom_array,
  int num_triangles
) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  int idx_1, idx_2, idx_3;
  vec3 t, b;

  for (int i = 0; i < num_triangles; i++) {
    t = geom_array[i] -> get_t();
    b = geom_array[i] -> get_b();
    idx_1 = geom_array[i] -> get_point_1_idx();
    idx_2 = geom_array[i] -> get_point_2_idx();
    idx_3 = geom_array[i] -> get_point_3_idx();
    if (!t.vector_is_nan() && !b.vector_is_nan()) {
      tangent_array[idx_1] += t;
      tangent_array[idx_2] += t;
      tangent_array[idx_3] += t;
      bitangent_array[idx_1] += b;
      bitangent_array[idx_2] += b;
      bitangent_array[idx_3] += b;
    }
  }
}



#endif
