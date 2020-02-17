//File: bvh_build.h
#ifndef BVH_BUILD_H
#define BVH_BUILD_H

#include "../../util/bvh_util.h"
#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/grid.h"

class Node {
  public:
    __host__ __device__ Node() {}
    __device__ Node(Node* left_, Node* right_);

    Node *left, *right;
};

class Leaf: public Node {
  public:
    __device__ Leaf(Primitive* object_) {
      this -> object = object_;
    }

    Primitive* object;
};

__global__ void extract_morton_code_list(
  Primitive** object_list, unsigned int* morton_code_list, int num_objects
);

__global__ void build_node_list(
  Node** node_list, Leaf** leaf_list, unsigned int* morton_code_list,
  int num_objects
);

__global__ void build_leaf_list(
  Leaf** leaf_list, Primitive **object_list, int num_objects
);

__global__ void compute_morton_code_batch(
  Primitive **object_array, Grid **grid, int num_triangles
);

__device__ bool morton_code_smaller(Primitive* obj_1, Primitive* obj_2);

__device__ Node::Node(Node* left_, Node* right_) {
  this -> left = left_;
  this -> right = right_;
}

__global__ void compute_morton_code_batch(
  Primitive **object_array, Grid **grid, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles) return;

  object_array[idx] -> get_bounding_box() -> compute_normalized_center(
    grid[0] -> world_bounding_box
  );
  object_array[idx] -> get_bounding_box() -> compute_bb_morton_3d();
}

__device__ bool morton_code_smaller(Primitive* obj_1, Primitive* obj_2) {
  return (
    obj_1 -> get_bounding_box() -> morton_code <
      obj_2 -> get_bounding_box() -> morton_code
  );
}

__global__ void build_leaf_list(
  Leaf** leaf_list, Primitive **object_list, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_triangles) return;
  leaf_list[idx] = new Leaf(object_list[idx]);
}

__global__ void extract_morton_code_list(
  Primitive** object_list, unsigned int* morton_code_list, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects) return;
  morton_code_list[idx] = \
    object_list[idx] -> get_bounding_box() -> morton_code;
}

__global__ void build_node_list(
  Node** node_list, Leaf** leaf_list, unsigned int* morton_code_list,
  int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (num_objects - 1)) return;

  int d1 = length_longest_common_prefix(
    morton_code_list, idx, idx + 1, num_objects
  );
  int d2 = length_longest_common_prefix(
    morton_code_list, idx, idx - 1, num_objects
  );
  int d;
  if (d1 < d2) {
    d = -1;
  } else {
    d = 1;
  }
  
}

#endif
