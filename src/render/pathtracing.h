#ifndef PATHTRACING_H
#define PATHTRACING_H

#include <curand_kernel.h>
#include <math.h>

#include "../model/camera.h"
#include "../model/cartesian_system.h"
#include "../model/data_structure/local_vector.h"
#include "../model/geometry/triangle.h"
#include "../model/ray.h"
#include "../util/vector_util.h"

__global__ void render(
  float* fb, Camera** camera, Triangle** geom_array, int *num_triangles,
  curandState *rand_state
);

__host__ __device__ bool _hit(
  Ray ray, Triangle **geom_array, int num_triangles, hit_record &rec);

__device__ vec3 _compute_color(
  hit_record rec, int level, Triangle **geom_array, int num_triangles,
  curandState *rand_state);

__device__ vec3 _compute_color(
  hit_record rec, int level, Triangle **geom_array, int num_triangles,
  curandState *rand_state
) {
  hit_record cur_rec = rec;
  bool hit;
  vec3 mask = vec3(1, 1, 1), light = vec3(0, 0, 0), v3_rand, v3_rand_world;
  float pdf = 1 / (2 * M_PI), cos_theta;
  Ray ray;

  for (int i = 0; i < level; i++) {
    CartesianSystem new_xyz_system = CartesianSystem(cur_rec.normal);
    v3_rand = get_random_unit_vector_hemisphere(rand_state);
    cos_theta = v3_rand.z();
    v3_rand_world = new_xyz_system.to_world_system(v3_rand);
    ray = Ray(cur_rec.point, v3_rand_world);
    hit = _hit(ray, geom_array, num_triangles, cur_rec);
    if (hit) {
      light += cos_theta * cur_rec.object -> material -> emission;
      if (light.x() > 0 || light.y() > 0 || light.z() > 0) {
        return mask * light;
      } else {
        mask *= (1 / pdf) * cur_rec.object -> material -> diffuse * cur_rec.object -> material -> albedo;
      }
    } else {
      light += cos_theta * vec3(0.1, 0.1, 0.1);
      return mask * light;
    }
  }

  return mask * light;
}

__host__ __device__ bool _hit(
  Ray ray, Triangle **geom_array, int num_triangles, hit_record &rec) {

  hit_record cur_rec;
  bool hit = false, intersection_found = false;
  vec3 color = vec3(0, 0, 0);

  rec.t = INFINITY;

  for (int idx = 0; idx < num_triangles; idx++) {
      hit = (geom_array[idx]) -> hit(ray, rec.t, cur_rec);
    if (hit) {
      intersection_found = true;
      rec = cur_rec;
    }
  }

  return intersection_found;
}

__global__
void render(
  vec3 *fb, Camera **camera, Triangle **geom_array, int *num_triangles,
  curandState *rand_state
) {

  hit_record init_rec, cur_rec;
  vec3 color = vec3(0, 0, 0);
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  int pixel_index = i * (camera[0] -> width) + j, sampling_size = 1024;
  // float max_dist = 40.0;

  if((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  curandState local_rand_state = rand_state[pixel_index];

  Ray camera_ray = camera[0] -> compute_ray(i + .5, j + .5);
  bool hit = _hit(camera_ray, geom_array, num_triangles[0], init_rec);
  CartesianSystem new_xyz_system = CartesianSystem(init_rec.normal);
  vec3 v3_rand, v3_rand_world;
  float pdf = 1 / (2 * M_PI), cos_theta;
  Ray ray;

  if (hit) {
    for(int idx = 0; idx < sampling_size; idx++) {
      cur_rec = init_rec;
      // printf("Processing sample = %d out of %d.\n", idx, sampling_size);
      v3_rand = get_random_unit_vector_hemisphere(&local_rand_state);
      cos_theta = v3_rand.z();
      v3_rand_world = new_xyz_system.to_world_system(v3_rand);
      ray = Ray(cur_rec.point, v3_rand_world);
      hit = _hit(ray, geom_array, num_triangles[0], cur_rec);
      if (hit) {
        color += cos_theta * _compute_color(
          cur_rec, 8, geom_array, num_triangles[0], &local_rand_state);
      }

    }
    color = init_rec.object -> material -> emission + init_rec.object -> material -> ambient + \
      (1.0f / sampling_size) * (1 / pdf) * color * init_rec.object -> material -> diffuse * init_rec.object -> material -> albedo;
  }
  rand_state[pixel_index] = local_rand_state;
  fb[pixel_index] = color;


}

#endif
