#ifndef PATHTRACING_H
#define PATHTRACING_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/camera.h"
#include "../model/cartesian_system.h"
#include "../model/geometry/triangle.h"
#include "../model/ray.h"
#include "../model/scene.h"
#include "../util/vector_util.h"

__global__
void render(
  vec3 *fb, Scene **scene, curandState *rand_state, int sample_size, int level,
  vec3 sky_emission, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b
);

__device__ vec3 _compute_color(
  hit_record rec, int level, Scene **scene, vec3 sky_emission,
  int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  curandState *rand_state
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
  Ray ray_init, int level, Scene **scene, vec3 sky_emission,
  int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b,
  curandState *rand_state
) {
  hit_record cur_rec;
  bool hit, reflected_or_refracted;
  vec3 mask = vec3(1, 1, 1), light = vec3(0, 0, 0), light_tmp = vec3(0, 0, 0);
  vec3 v3_rand, v3_rand_world;
  Ray ray = ray_init;
  reflection_record ref;

  cur_rec.object = nullptr;

  for (int i = 0; i < level; i++) {
    hit = scene[0] -> grid -> do_traversal(ray, cur_rec);
    if (hit) {
      reflected_or_refracted = cur_rec.object -> get_material(
      ) -> is_reflected_or_refracted(
        cur_rec.coming_ray, cur_rec.point, cur_rec.normal, cur_rec.uv_vector,
        ref, rand_state
      );

      if (reflected_or_refracted) {

        ray = ref.ray;
        light_tmp = cur_rec.object -> get_material() -> emission;

        if (light_tmp.x() > 0 || light_tmp.y() > 0 || light_tmp.z() > 0) {
          light += light_tmp;
          return mask * light;
        } else {
          mask *= (1.0) * ref.filter;
        }

      } else {
        return vec3(0, 0, 0);
      }

    } else {
      if (i < 1){
        return _get_sky_color(
          vec3(1, 1, 1), ray.dir, bg_height, bg_width, bg_r, bg_g, bg_b);
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
  vec3 *fb, Scene **scene, curandState *rand_state, int sample_size, int level,
  vec3 sky_emission, int bg_height, int bg_width,
  float *bg_r, float *bg_g, float *bg_b
) {

  hit_record init_rec, cur_rec;
  vec3 color = vec3(0, 0, 0), color_tmp;
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if(
    (j >= scene[0] -> camera -> width) || (i >= scene[0] -> camera -> height)
  ) {
    return;
  }

  int pixel_index = i * (scene[0] -> camera -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];
  Ray camera_ray = scene[0] -> camera -> compute_ray(
    i + .5, j + .5, &local_rand_state), ray;

  for(int idx = 0; idx < sample_size; idx++) {
    color_tmp = _compute_color(
      camera_ray, level, scene, sky_emission, bg_height, bg_width, bg_r, bg_g,
      bg_b, &local_rand_state);
    color_tmp = de_nan(color_tmp);
    color += color_tmp;
  }
  color *= (1.0 / sample_size);

  rand_state[pixel_index] = local_rand_state;
  fb[pixel_index] = color;

}

#endif
