//File: object_operations.h
#ifndef OBJECT_OPERATIONS_H
#define OBJECT_OPERATIONS_H

#include "../grid/bounding_box.h"
#include "../point/point.h"
#include "object.h"

__global__ void compute_object_boundaries_batch(
  Object **object_array, int num_objects
);
__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int *pt_offset_array,
  int *num_pt_array, int num_objects, int num_pts_per_object
);
__global__ void allocate_pts_sss(
  Object ** object_array, Point** point_array, int *pt_offset_array,
  int num_objects
);
__global__ void create_sss_pts(
  Object** object, Primitive** geom_array, Point** point_array, int *pt_offset,
  curandState *rand_state, int object_idx, int num_pts_per_object
);
__global__ void compute_pts_morton_code_batch(
  Object** object_array, Point** point_array, int num_points
);
__global__ void compute_sss_pts_offset(Object **object_array, int num_objects);

__global__ void compute_sss_pts_offset(
  Object **object_array, int num_objects
) {
  int node_offset, leaf_offset;
  node_offset = 0;
  leaf_offset = 0;
  for (int i = 0; i < num_objects; i++) {
    object_array[i] -> assign_bvh_root_node_idx(node_offset);
    object_array[i] -> assign_bvh_leaf_zero_idx(leaf_offset);
    if (object_array[i] -> num_pts > 0) {
      node_offset += (object_array[i] -> num_pts - 1);
    }
    leaf_offset += object_array[i] -> num_pts;
  }
}

__global__ void compute_pts_morton_code_batch(
  Object** object_array, Point** point_array, int num_points
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_points) return;

  int object_idx = point_array[idx] -> object_idx;
  float x_min = object_array[object_idx] -> x_min;
  float x_max = object_array[object_idx] -> x_max;
  float y_min = object_array[object_idx] -> y_min;
  float y_max = object_array[object_idx] -> y_max;
  float z_min = object_array[object_idx] -> z_min;
  float z_max = object_array[object_idx] -> z_max;
  point_array[idx] -> bounding_box -> compute_normalized_center(
    x_min, x_max, y_min, y_max, z_min, z_max
  );
  point_array[idx] -> bounding_box -> compute_bb_morton_3d();
}

__global__ void compute_object_boundaries_batch(
  Object **object_array, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects) return;

  if (object_array[idx] -> sub_surface_scattering)
    object_array[idx] -> compute_boundaries();
}

__global__ void compute_num_sss_objects(
  int *num_sss_objects, Object** object_array, int *pt_offset_array,
  int *num_pt_array, int num_objects, int num_pts_per_object
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  int offset = 0;
  num_sss_objects[0] = 0;

  for (int i = 0; i < num_objects; i++) {

    object_array[i] -> compute_accummulated_triangle_area();

    if (object_array[i] -> sub_surface_scattering) {
      (num_sss_objects[0])++;
      num_pt_array[i] = num_pts_per_object;
    } else {
      num_pt_array[i] = 0;
    }

    if (i == 0) {
      pt_offset_array[i] = offset;
    } else {
      if (object_array[i - 1] -> sub_surface_scattering) {
        offset += num_pts_per_object;
      }
      pt_offset_array[i] = offset;
    }

  }
  printf("We have %d SSS objects.\n", num_sss_objects[0]);
}

__global__ void allocate_pts_sss(
  Object ** object_array, Point** point_array, int *pt_offset_array,
  int num_objects
) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx > 0) return;

  for (int i = 0; i < num_objects; i++) {
    object_array[i] -> allocate_point_array(point_array + pt_offset_array[i]);
  }
}

__global__ void create_sss_pts(
  Object** object, Primitive** geom_array, Point** point_array, int *pt_offset,
  curandState *rand_state, int object_idx, int num_pts_per_object
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (!object[object_idx] -> sub_surface_scattering) return;
  if (idx >= num_pts_per_object) return;

  curandState local_rand_state = rand_state[idx];
  int primitive_idx = (object[object_idx]) -> pick_primitive_idx_for_sampling(
    &local_rand_state);
  hit_record pts_record = geom_array[primitive_idx] -> get_random_point_on_surface(
    rand_state);
  vec3 filter = pts_record.object -> get_material() -> get_texture_diffuse(
    pts_record.uv_vector);
  point_array[pt_offset[object_idx] + idx] = new Point(
    pts_record.point, filter, pts_record.normal, object_idx, true, false);
}

#endif
