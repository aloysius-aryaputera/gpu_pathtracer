//File: bvh_traversal_target.h
#ifndef BVH_TRAVERSAL_TARGET_H
#define BVH_TRAVERSAL_TARGET_H

#include <curand_kernel.h>

#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../ray/ray.h"
#include "bvh.h"

__device__ bool traverse_bvh_target(
  Node* bvh_root, Ray ray, int* potential_target_idx,
  int &num_potential_targets, int max_potential_targets
);

__device__ Primitive* traverse_bvh_to_pick_a_target(
  Node* bvh_root, vec3 shading_point, vec3 normal, vec3 kd, 
  curandState *rand_state, bool write
); 

__device__ float get_node_pdf(
  Node* selected_node, vec3 shading_point, vec3 normal, vec3 kd
);

__global__ void print_node_pdf(
  Node** node_list, int len_node_list, vec3 point, vec3 normal, vec3 kd
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= len_node_list) return;

  float left_importance = node_list[i] -> left -> compute_importance(
    point, normal, kd
  );
  float right_importance = node_list[i] -> right -> compute_importance(
    point, normal, kd
  );
  float prob_left = left_importance / (left_importance + right_importance);
  float prob_right = right_importance / (left_importance + right_importance);
  printf(
    "Node[%d]: left is leaf: %d, right is leaf: %d, importance_left = %f, importance_right = %f, prob_left = %f, prob_right = %f\n", i, 
    node_list[i] -> left -> is_leaf, node_list[i] -> right -> is_leaf, 
    left_importance, right_importance,
    prob_left, prob_right);
}

__device__ float get_node_pdf(
  Node* selected_node, vec3 shading_point, vec3 normal, vec3 kd
){
  Node* it_node = selected_node, *another_node;
  float pdf = 1.0, it_pdf, it_tot_pdf, importance_1, importance_2;
  while(it_node -> parent != nullptr) {
    importance_1 = it_node -> compute_importance(
      shading_point, normal, kd
    );
    if ((it_node -> parent -> left) == it_node) {
      another_node = it_node -> parent -> right;
    } else {
      another_node = it_node -> parent -> left;
    }
    importance_2 = another_node -> compute_importance(
      shading_point, normal, kd
    );

    it_tot_pdf = importance_1 + importance_2;
    it_pdf = importance_1 / it_tot_pdf;
    if (isnan(it_pdf) || isinf(it_pdf)) {
    //if (it_tot_pdf < 1E-10) {
      pdf *= .5;
    } else {
      pdf *= it_pdf;	
    }
    it_node = it_node -> parent;
  }
  return pdf;
}

__device__ Primitive* traverse_bvh_to_pick_a_target(
  Node* bvh_root, vec3 shading_point, vec3 normal, vec3 kd,
  curandState *rand_state, bool write=false
) {
  Node* selected_node = bvh_root;
  float left_importance, right_importance, random_number, factor;
  float total_importance;
  while(!(selected_node -> is_leaf)) {
    left_importance = selected_node -> left -> compute_importance(
      shading_point, normal, kd
    );
    right_importance = selected_node -> right -> compute_importance(
      shading_point, normal, kd
    );
    total_importance = left_importance + right_importance;
    
    if (total_importance < 1E-10)
      factor = .5;
    else
      factor = left_importance / total_importance;

    random_number = curand_uniform(&rand_state[0]);
    if (write) {
      printf("random_number = %f\n", random_number);
    }

    if (random_number < factor) {
      selected_node = selected_node -> left;
    } else {
      selected_node = selected_node -> right;
    }
  };
  return selected_node -> object;
}

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
