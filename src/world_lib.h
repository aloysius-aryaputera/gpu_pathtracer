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
#include "model/object/object.h"
#include "model/ray.h"
#include "model/scene.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"
#include "util/read_file_util.h"

__global__ void create_world(
  Primitive** geom_array,
  float *triangle_area,
  Object** object_array,
  int *triangle_object_idx,
  Material** material_array,
  float *x, float *y, float *z,
  float *x_norm, float *y_norm, float *z_norm,
  float *x_tex, float *y_tex,
  int *point_1_idx, int *point_2_idx, int *point_3_idx,
  int *norm_1_idx, int *norm_2_idx, int *norm_3_idx,
  int *tex_1_idx, int *tex_2_idx, int *tex_3_idx,
  int *material_idx,
  int* num_triangles,
  bool* sss_object_marker_array
);

__global__ void create_objects(
  Object** object_array, int* object_num_primitives,
  int *object_primitive_offset_idx, float *triangle_area,  int num_objects
);

__global__ void create_material(
  Material** material_array,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  float *tf_x, float *tf_y, float *tf_z,
  float *path_length,
  float *t_r, float *n_s, float *n_i,
  int *material_priority,
  int *material_image_height_diffuse,
  int *material_image_width_diffuse,
  int *material_image_offset_diffuse,
  int *material_image_height_specular,
  int *material_image_width_specular,
  int *material_image_offset_specular,
  int *material_image_height_emission,
  int *material_image_width_emission,
  int *material_image_offset_emission,
  int *material_image_height_n_s,
  int *material_image_width_n_s,
  int *material_image_offset_n_s,
  float *material_image_r,
  float *material_image_g,
  float *material_image_b,
  int *num_materials
);

__global__ void create_camera(
  Camera** camera, float eye_x, float eye_y, float eye_z,
  float center_x, float center_y, float center_z,
  float up_x, float up_y, float up_z, float fovy,
  int image_width, int image_height,
  float aperture, float focus_dist
);

__global__ void create_camera(
  Camera** camera, float eye_x, float eye_y, float eye_z,
  float center_x, float center_y, float center_z,
  float up_x, float up_y, float up_z, float fovy,
  int image_width, int image_height,
  float aperture, float focus_dist
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(camera) = new Camera(
      vec3(eye_x, eye_y, eye_z), vec3(center_x, center_y, center_z),
      vec3(up_x, up_y, up_z), fovy, image_width, image_height,
      aperture, focus_dist
    );
  }
}

__global__ void create_material(
  Material** material_array,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  float *tf_x, float *tf_y, float *tf_z,
  float *path_length,
  float *t_r, float *n_s, float *n_i,
  int *material_priority,
  int *material_image_height_diffuse,
  int *material_image_width_diffuse,
  int *material_image_offset_diffuse,
  int *material_image_height_specular,
  int *material_image_width_specular,
  int *material_image_offset_specular,
  int *material_image_height_emission,
  int *material_image_width_emission,
  int *material_image_offset_emission,
  int *material_image_height_n_s,
  int *material_image_width_n_s,
  int *material_image_offset_n_s,
  float *material_image_r,
  float *material_image_g,
  float *material_image_b,
  int *num_materials
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_materials[0]) return;

  *(material_array + i) = new Material(
    vec3(ka_x[i], ka_y[i], ka_z[i]),
    vec3(kd_x[i], kd_y[i], kd_z[i]),
    vec3(ks_x[i], ks_y[i], ks_z[i]),
    vec3(ke_x[i], ke_y[i], ke_z[i]),
    vec3(tf_x[i], tf_y[i], tf_z[i]),
    path_length[i],
    t_r[i], n_s[i], n_i[i],
    material_priority[i],
    material_image_height_diffuse[i],
    material_image_width_diffuse[i],
    material_image_r + material_image_offset_diffuse[i],
    material_image_g + material_image_offset_diffuse[i],
    material_image_b + material_image_offset_diffuse[i],
    material_image_height_specular[i],
    material_image_width_specular[i],
    material_image_r + material_image_offset_specular[i],
    material_image_g + material_image_offset_specular[i],
    material_image_b + material_image_offset_specular[i],
    material_image_height_emission[i],
    material_image_width_emission[i],
    material_image_r + material_image_offset_emission[i],
    material_image_g + material_image_offset_emission[i],
    material_image_b + material_image_offset_emission[i],
    material_image_height_n_s[i],
    material_image_width_n_s[i],
    material_image_r + material_image_offset_n_s[i],
    material_image_g + material_image_offset_n_s[i],
    material_image_b + material_image_offset_n_s[i]
  );

}

__global__ void create_objects(
  Object** object_array, int* object_num_primitives,
  int *object_primitive_offset_idx, float* triangle_area, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_objects) return;

  *(object_array + idx) = new Object(
    object_primitive_offset_idx[idx], object_num_primitives[idx],
    triangle_area + object_primitive_offset_idx[idx]
  );

}

__global__ void create_world(
  Primitive** geom_array,
  float *triangle_area,
  Object** object_array,
  int *triangle_object_idx,
  Material** material_array,
  float *x, float *y, float *z,
  float *x_norm, float *y_norm, float *z_norm,
  float *x_tex, float *y_tex,
  int *point_1_idx, int *point_2_idx, int *point_3_idx,
  int *norm_1_idx, int *norm_2_idx, int *norm_3_idx,
  int *tex_1_idx, int *tex_2_idx, int *tex_3_idx,
  int *material_idx,
  int* num_triangles,
  bool* sss_object_marker_array
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles[0]) return;

  *(geom_array + idx) = new Triangle(
    vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
    vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
    vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]),

    material_array[material_idx[idx]],
    object_array[triangle_object_idx[idx]],

    vec3(x_norm[norm_1_idx[idx]], y_norm[norm_1_idx[idx]], z_norm[norm_1_idx[idx]]),
    vec3(x_norm[norm_2_idx[idx]], y_norm[norm_2_idx[idx]], z_norm[norm_2_idx[idx]]),
    vec3(x_norm[norm_3_idx[idx]], y_norm[norm_3_idx[idx]], z_norm[norm_3_idx[idx]]),

    vec3(x_tex[tex_1_idx[idx]], y_tex[tex_1_idx[idx]], 0),
    vec3(x_tex[tex_2_idx[idx]], y_tex[tex_2_idx[idx]], 0),
    vec3(x_tex[tex_3_idx[idx]], y_tex[tex_3_idx[idx]], 0)
  );

  triangle_area[idx] = (*(geom_array + idx)) -> area;

  if (geom_array[idx] -> is_sub_surface_scattering())
  {
    object_array[triangle_object_idx[idx]] -> set_as_sub_surface_scattering();
    sss_object_marker_array[triangle_object_idx[idx]] = true;
  } else {
    sss_object_marker_array[triangle_object_idx[idx]] = false;
  }

}
