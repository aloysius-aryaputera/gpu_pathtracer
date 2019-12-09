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
  vec3 sky_emission
);

__device__ vec3 _compute_color(
  hit_record rec, int level, Scene **scene, vec3 sky_emission,
  curandState *rand_state
);

__device__ vec3 _compute_color(
  hit_record rec, int level, Scene **scene, vec3 sky_emission,
  curandState *rand_state
) {
  hit_record cur_rec = rec;
  bool hit;
  vec3 mask = vec3(1, 1, 1), light = vec3(0, 0, 0), v3_rand, v3_rand_world;
  float pdf = 1 / (2 * M_PI), cos_theta;
  Ray ray;
  reflection_record ref;

  ref = cur_rec.object -> get_material() -> get_reflection_ray(
    cur_rec.coming_ray, cur_rec.point, cur_rec.normal, cur_rec.uv_vector,
    rand_state
  );
  ray = ref.ray;
  cos_theta = ref.cos_theta;

  if (ref.material_type == 's' && cos_theta <= 0) {
    return vec3(0, 0, 0);
  }

  for (int i = 0; i < level; i++) {
    // CartesianSystem new_xyz_system = CartesianSystem(cur_rec.normal);
    // v3_rand = get_random_unit_vector_hemisphere(rand_state);
    // cos_theta = v3_rand.z();
    // v3_rand_world = new_xyz_system.to_world_system(v3_rand);
    // ray = Ray(cur_rec.point, v3_rand_world);

    hit = scene[0] -> grid -> do_traversal(ray, cur_rec);
    if (hit) {
      ref = cur_rec.object -> get_material() -> get_reflection_ray(
        cur_rec.coming_ray, cur_rec.point, cur_rec.normal, cur_rec.uv_vector,
        rand_state
      );
      cos_theta = ref.cos_theta;

      if (ref.material_type == 's' && cos_theta <= 0) {
        return vec3(0, 0, 0);
      }

      ray = ref.ray;
      light += cos_theta * cur_rec.object -> get_material() -> emission;

      if (light.x() > 0 || light.y() > 0 || light.z() > 0) {
        return mask * light;
      } else {
        mask *= cos_theta * \ // (1 / pdf) *
          (1 / M_PI) * \
          cur_rec.object -> get_material() -> albedo * ref.color;
          // cur_rec.object -> get_material() -> get_texture(cur_rec.uv_vector) *
      }

    } else {
      light += cos_theta * (sky_emission * (ray.dir.y() + 1) / 2.0);
      return mask * light;
    }
  }

  return mask * light;
}

__global__
void render(
  vec3 *fb, Scene **scene, curandState *rand_state, int sample_size, int level,
  vec3 sky_emission
) {

  hit_record init_rec, cur_rec;
  vec3 color = vec3(0, 0, 0);
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if(
    (j >= scene[0] -> camera -> width) || (i >= scene[0] -> camera -> height)
  ) {
    return;
  }

  int pixel_index = i * (scene[0] -> camera -> width) + j;
  curandState local_rand_state = rand_state[pixel_index];

  Ray camera_ray = scene[0] -> camera -> compute_ray(i + .5, j + .5), ray;
  bool hit = scene[0] -> grid -> do_traversal(camera_ray, init_rec);
  // CartesianSystem new_xyz_system = CartesianSystem(init_rec.normal);
  // vec3 v3_rand, v3_rand_world;
  float pdf = 1 / (2 * M_PI), cos_theta;
  // Ray ray;
  reflection_record ref;

  if (hit) {
    for(int idx = 0; idx < sample_size; idx++) {
      cur_rec = init_rec;

      // v3_rand = get_random_unit_vector_hemisphere(&local_rand_state);
      // cos_theta = v3_rand.z();
      // v3_rand_world = new_xyz_system.to_world_system(v3_rand);
      // ray = Ray(cur_rec.point, v3_rand_world);

      ref = cur_rec.object -> get_material() -> get_reflection_ray(
        cur_rec.coming_ray, cur_rec.point, cur_rec.normal, cur_rec.uv_vector,
        &local_rand_state
      );
      ray = ref.ray;
      cos_theta = ref.cos_theta;

      if (ref.material_type != 's' || cos_theta > 0) {

        hit = scene[0] -> grid -> do_traversal(ray, cur_rec);
        if (hit) {
          color += cos_theta * _compute_color(
            cur_rec, level, scene, sky_emission, &local_rand_state) * \
            ref.color;
        } else {
          color += cos_theta * (sky_emission * (ray.dir.y() + 1) / 2.0) * \
            ref.color;
        }

      }

    }
    color = init_rec.object -> get_material() -> emission + \
      (1.0f / sample_size) * \ //(1 / pdf) *
      (1 / M_PI) * color * \
      init_rec.object -> get_material() -> albedo;
      // init_rec.object -> get_material() -> get_texture(init_rec.uv_vector) * \

  } else {
    color = vec3(0, 0, 0);
  }
  rand_state[pixel_index] = local_rand_state;
  fb[pixel_index] = color;

}

#endif
