#ifndef PATHTRACING_H
#define PATHTRACING_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/bvh/bvh_build.h"
#include "../model/camera.h"
#include "../model/cartesian_system.h"
#include "../model/geometry/triangle.h"
#include "../model/material.h"
#include "../model/ray.h"
#include "../param.h"
#include "../util/vector_util.h"
#include "material_list_operations.h"

__global__
void render(
  vec3 *fb, Camera **camera, curandState *rand_state, int sample_size, int level,
  vec3 sky_emission, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  Node **node_list
);

__device__ vec3 _compute_color(
  hit_record rec, int level, vec3 sky_emission,
  int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  curandState *rand_state,
  Node** node_list
);

__device__ vec3 _get_sky_color(
  vec3 sky_emission, vec3 look_dir, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b
);

__device__ vec3 _get_sky_color(
  vec3 sky_emission, vec3 look_dir, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b
) {
  // return sky_emission * (look_dir.y() + 1) / 2.0;
  float u = .5 + atan2(look_dir.z(), look_dir.x()) / (2.0 * M_PI);
  float v = .5 - asin(look_dir.y()) / M_PI;

  int idx_u = floorf((u - floorf(u)) * (bg_width - 1));
  int idx_v = floorf((v - floorf(v)) * (bg_height - 1));

  int idx = bg_width * idx_v + idx_u;

  return sky_emission * vec3(bg_r[idx], bg_g[idx], bg_b[idx]);
}

__device__ vec3 _compute_color(
  Ray ray_init, int level, vec3 sky_emission,
  int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  curandState *rand_state,
  Node **node_list
) {
  hit_record cur_rec;
  bool hit, reflected = false, refracted = false, false_hit = false;
  bool entering = false, exiting = false;
  vec3 mask = vec3(1, 1, 1), light = vec3(0, 0, 0), light_tmp = vec3(0, 0, 0);
  vec3 v3_rand, v3_rand_world;
  Ray ray = ray_init;
  reflection_record ref;
  Material* material_list[400];

  int material_list_length = 0;

  add_new_material(material_list, material_list_length, nullptr);

  cur_rec.object = nullptr;

  for (int i = 0; i < level; i++) {

    hit = traverse_bvh(node_list[0], ray, cur_rec);

    if (hit) {
      if (cur_rec.object == nullptr)
        printf("cur_rec.object = NULL\n");

      cur_rec.object -> get_material(
      ) -> check_if_reflected_or_refracted(
        cur_rec.coming_ray, cur_rec.point, cur_rec.normal, cur_rec.uv_vector,
        reflected, false_hit, refracted,
        entering, exiting,
        material_list, material_list_length,
        ref, rand_state
      );

      if (false_hit && entering)
        add_new_material(
          material_list, material_list_length, cur_rec.object -> get_material()
        );

      if (false_hit && exiting)
        remove_a_material(
          material_list, material_list_length, cur_rec.object -> get_material()
        );

      if (!false_hit && refracted && entering)
        add_new_material(
          material_list, material_list_length, cur_rec.object -> get_material()
        );

      if (!false_hit && refracted && exiting)
        remove_a_material(
          material_list, material_list_length, cur_rec.object -> get_material()
        );

      if (reflected || refracted) {

        ray = ref.ray;
        light_tmp = cur_rec.object -> get_material() -> emission;

        light += light_tmp;
        mask *= (1.0) * ref.filter;

        if (mask.r() < 0.005 && mask.g() < 0.005 && mask.b() < 0.005) {
          return vec3(0, 0, 0);
        }

        if (light.r() > 0 && light.g() > 0 && light.b() > 0) {
          return mask * light;
        }

      } else {
        return vec3(0, 0, 0);
      }

    } else {
      if (i < 1){
        return _get_sky_color(
          sky_emission, ray.dir, bg_height, bg_width, bg_r, bg_g, bg_b);
      } else {
        light += _get_sky_color(
          sky_emission, ray.dir, bg_height, bg_width, bg_r, bg_g, bg_b);
        return mask * light;
      }
    }
  }

  return mask * light;
}

__global__
void render(
  vec3 *fb, Camera **camera, curandState *rand_state, int sample_size, int level,
  vec3 sky_emission, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  Node **node_list
) {

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  hit_record init_rec, cur_rec;
  vec3 color = vec3(0, 0, 0), color_tmp;

  if(
    (j >= camera[0] -> width) || (i >= camera[0] -> height)
  ) {
    return;
  }

  int pixel_index = i * (camera[0] -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];
  Ray camera_ray = camera[0] -> compute_ray(
    i + .5, j + .5, &local_rand_state), ray;

  for(int idx = 0; idx < sample_size; idx++) {
    color_tmp = _compute_color(
      camera_ray, level, sky_emission, bg_height, bg_width, bg_r, bg_g,
      bg_b, &local_rand_state, node_list);
    color_tmp = de_nan(color_tmp);
    color += color_tmp;
  }
  color *= (1.0 / sample_size);

  rand_state[pixel_index] = local_rand_state;
  fb[pixel_index] = color;

  if (j == 0 && (i % (camera[0] -> height / 100) == 0)) {
    printf(
      "Progress = %5.5f %%\n",
      100.0 * i * camera[0] -> width / (
        camera[0] -> height * camera[0] -> width)
    );
  }

}

#endif
