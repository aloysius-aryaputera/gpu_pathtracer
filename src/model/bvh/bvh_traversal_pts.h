//File: bvh_traversal_pts.h
#ifndef BVH_TRAVERSAL_PTS_H
#define BVH_TRAVERSAL_PTS_H

#include "../grid/bounding_box.h"
#include "../grid/bounding_sphere.h"
#include "../point/point.h"
#include "bvh.h"

__device__ bool traverse_bvh_pts(
  Node* bvh_root, BoundingSphere bounding_sphere, //Point** point_array,
  vec3 &color
);

__device__ bool traverse_bvh_pts(
  Node* bvh_root, BoundingSphere bounding_sphere, //Point** point_array,
  vec3 &color
) {
  Node* stack[4000];
  Node *child_l, *child_r;
  Point *point;
  bool intersection_l, intersection_r, traverse_l, traverse_r;
  bool is_inside = false;
  bool pts_found = false;
  int idx_stack_top = 0;

  float weight, sum_weight = 0;
  color = vec3(0, 0, 0);

  stack[idx_stack_top] = nullptr;
  idx_stack_top++;

  Node *node = bvh_root;
  do {

    child_l = node -> left;
    child_r = node -> right;

    intersection_l = child_l -> bounding_box -> is_intersection(bounding_sphere);
    intersection_r = child_r -> bounding_box -> is_intersection(bounding_sphere);

    if (intersection_l && child_l -> is_leaf) {
      is_inside = bounding_sphere.is_inside(child_l -> point -> location);
      if (is_inside) {
        point = child_l -> point;
        pts_found = true;
        weight = 1.0 / compute_distance(
          point -> location, bounding_sphere.center);
        weight = min(9999.99, weight);
        sum_weight += weight;
        color += weight * point -> color;
      }
    }

    if (intersection_r && child_r -> is_leaf) {
      is_inside = bounding_sphere.is_inside(child_r -> point -> location);
      if (is_inside) {
        point = child_r -> point;
        pts_found = true;
        weight = 1.0 / compute_distance(
          point -> location, bounding_sphere.center);
        weight = min(9999.99, weight);
        sum_weight += weight;
        color += weight * point -> color;
      }
    }

    traverse_l = (intersection_l && !(child_l -> is_leaf));
    traverse_r = (intersection_r && !(child_r -> is_leaf));

    if (!traverse_l && !traverse_r) {
      idx_stack_top--;
      node = stack[idx_stack_top];

    } else {

      if (traverse_l)
        node = child_l;
      else {
        if (traverse_r)
          node = child_r;
      }

      if (traverse_l && traverse_r && !(child_r -> is_leaf)) {
        stack[idx_stack_top] = child_r;

        if (child_r -> left == nullptr && !(child_r -> is_leaf)){
          printf("child_r is a node, but it does not have left child.\n");
        }

        if (child_r -> right == nullptr && !(child_r -> is_leaf)){
          printf("child_r is a node, but it does not have right child.\n");
        }

        idx_stack_top++;
      }
    }

  } while(idx_stack_top > 0 && idx_stack_top < 4000 && node != nullptr);

  if (pts_found)
    color /= sum_weight;

  return pts_found;
}

#endif
