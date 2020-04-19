//File: pathtracing_sss.h
#ifndef PATHTRACING_SSS_H
#define PATHTRACING_SSS_H

#include "../model/geometry/primitive.h"
#include "../model/geometry/triangle.h"
#include "../model/grid/bounding_sphere.h"
#include "../model/object/object.h"
#include "../model/point/point.h"
#include "../model/bvh/bvh.h"
#include "../model/bvh/bvh_traversal_pts.h"

__device__ vec3 compute_color_sss(
  hit_record rec, Object** object_list, Node** node_list
);

__device__ vec3 compute_color_sss(
  hit_record rec, Object** object_list, Node** node_list
) {

  Object *object = object_list[rec.object -> get_object_idx()];
  Node** effective_node_list = node_list + object -> bvh_root_node_idx;
  Point* point_array[500];
  float bounding_sphere_r = rec.object -> get_material() -> path_length;
  BoundingSphere bounding_sphere = BoundingSphere(
    rec.point, bounding_sphere_r);
  bool pts_found = false;
  int num_pts;
  vec3 filter, color = vec3(0, 0, 0);

  pts_found = traverse_bvh_pts(
    effective_node_list[0], bounding_sphere, point_array, 500, num_pts
  );

  if (pts_found) {
    for (int i = 0; i < num_pts; i++) {
      color += point_array[i] -> color;
    }
    color /= num_pts;
    filter = rec.object -> get_material() -> get_texture_diffuse(
      rec.uv_vector);
    return color * filter;
  } else {
    return vec3(0.0, 0.0, 0.0);
  }

}

#endif
