#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/sphere.h"
#include "model/geometry/triangle.h"
#include "model/grid/cell.h"
#include "model/grid/grid.h"
#include "model/material.h"
#include "model/ray.h"
#include "model/scene.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"
#include "util/read_file_util.h"

__global__ void create_world(
  Primitive** geom_array,
  Material** material_array,
  float *x, float *y, float *z,
  float *x_norm, float *y_norm, float *z_norm,
  int *point_1_idx, int *point_2_idx, int *point_3_idx,
  int *norm_1_idx, int *norm_2_idx, int *norm_3_idx,
  int *material_idx,
  int* num_triangles
);

__global__ void create_material(
  Material** material_array,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  int *num_materials
);

__global__ void create_camera(
  Camera** camera, float eye_x, float eye_y, float eye_z,
  float center_x, float center_y, float center_z,
  float up_x, float up_y, float up_z, float fovy,
  int image_width, int image_height
);

__global__ void create_camera(
  Camera** camera, float eye_x, float eye_y, float eye_z,
  float center_x, float center_y, float center_z,
  float up_x, float up_y, float up_z, float fovy,
  int image_width, int image_height
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(camera) = new Camera(
      vec3(eye_x, eye_y, eye_z), vec3(center_x, center_y, center_z),
      vec3(up_x, up_y, up_z), fovy, image_width, image_height
    );
  }
}

__global__ void create_material(
  Material** material_array,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  int *num_materials
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_materials[0]) return;

  *(material_array + i) = new Material(
    vec3(ka_x[i], ka_y[i], ka_z[i]),
    vec3(kd_x[i], kd_y[i], kd_z[i]),
    vec3(ke_x[i], ke_y[i], ke_z[i]),
    vec3(.49, .49, .49)
  );

}

__global__ void create_world(
  Primitive** geom_array,
  Material** material_array,
  float *x, float *y, float *z,
  float *x_norm, float *y_norm, float *z_norm,
  int *point_1_idx, int *point_2_idx, int *point_3_idx,
  int *norm_1_idx, int *norm_2_idx, int *norm_3_idx,
  int *material_idx,
  int* num_triangles
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = i;

  if (idx >= num_triangles[0]) return;

  *(geom_array + idx) = new Triangle(
    vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
    vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
    vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]),
    material_array[material_idx[idx]],
    vec3(x_norm[norm_1_idx[idx]], y_norm[norm_1_idx[idx]], z_norm[norm_1_idx[idx]]),
    vec3(x_norm[norm_2_idx[idx]], y_norm[norm_2_idx[idx]], z_norm[norm_2_idx[idx]]),
    vec3(x_norm[norm_3_idx[idx]], y_norm[norm_3_idx[idx]], z_norm[norm_3_idx[idx]])
  );

}
