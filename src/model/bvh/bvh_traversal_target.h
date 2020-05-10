//File: bvh_traversal_target.h
#ifndef BVH_TRAVERSAL_TARGET_H
#define BVH_TRAVERSAL_TARGET_H

#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../ray/ray.h"
#include "bvh.h"

__device__ bool traverse_bvh_target(
  Node* bvh_root, Ray ray, int* potential_target_idx,
  int &num_potential_targets, int max_potential_targets
);

__device__ bool traverse_bvh_target(
  Node* bvh_root, Ray ray, int* potential_target_idx,
  int &num_potential_targets, int max_potential_targets
) {
  Node* stack[400];
  Node *child_l, *child_r;
  bool intersection_l, intersection_r, traverse_l, traverse_r;
  bool intersection_found = false;
  float t;
  int idx_stack_top = 0;

  num_potential_targets = 0;

  stack[idx_stack_top] = nullptr;
  idx_stack_top++;

  Node *node = bvh_root;
  do {

    child_l = node -> left;
    child_r = node -> right;

    intersection_l = child_l -> bounding_box -> is_intersection(ray, t);
    intersection_r = child_r -> bounding_box -> is_intersection(ray, t);

    if (intersection_l && child_l -> is_leaf) {
      potential_target_idx[num_potential_targets++] = child_l -> idx;
    }

    if (intersection_r && child_r -> is_leaf) {
      potential_target_idx[num_potential_targets++] = child_r -> idx;
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

  } while(
    idx_stack_top > 0 && idx_stack_top < 400 && node != nullptr &&
    num_potential_targets < (max_potential_targets - 1)
  );

  return intersection_found;
}

#endif
