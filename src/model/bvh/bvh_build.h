//File: bvh_build.h
#ifndef BVH_BUILD_H
#define BVH_BUILD_H

#include "../../util/bvh_util.h"
#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/grid.h"

class Node {
  public:
    __host__ __device__ Node() {
      this -> visited = false;
    }
    __device__ void set_left_child(Node* left_);
    __device__ void set_right_child(Node* right_);
    __device__ void set_parent(Node* parent_);
    __device__ void mark_visited();

    Node *left, *right, *parent;
    bool visited;
    BoundingBox *bounding_box;
};

class Leaf: public Node {
  public:
    __device__ Leaf(Primitive* object_) {
      this -> object = object_;
      this -> bounding_box = object_ -> bounding_box;
    }

    Primitive* object;
};

__global__ void extract_morton_code_list(
  Primitive** object_list, unsigned int* morton_code_list, int num_objects
);

__global__ void build_node_list(Node** node_list, int num_objects);

__global__ void set_node_relationship(
  Node** node_list, Leaf** leaf_list, unsigned int* morton_code_list,
  int num_objects
);

__global__ void build_node_hierarchy(
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

__device__ void Node::mark_visited() {
  this -> visited = true;
}

__device__ void Node::set_left_child(Node* left_) {
  this -> left = left_;
}

__device__ void Node::set_right_child(Node* right_) {
  this -> right = right_;
}

__device__ void Node::set_parent(Node* parent_) {
  this -> parent = parent_;
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

__global__ void build_node_list(Node** node_list, int num_objects) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects - 1) return;
  node_list[idx] = new Node();
}

__global__ void extract_morton_code_list(
  Primitive** object_list, unsigned int* morton_code_list, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects) return;
  morton_code_list[idx] = \
    object_list[idx] -> get_bounding_box() -> morton_code;
}

// __global__ void build_node_hierarchy(
//   Node** node_list, Leaf** leaf_list, unsigned int* morton_code_list,
//   int num_objects
// ) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= (num_objects - 1)) return;
//
//   int2
// }

__global__ void set_node_relationship(
  Node** node_list, Leaf** leaf_list, unsigned int* morton_code_list,
  int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= (num_objects - 1)) return;

  // Determine direction of the range (+1 or -1)
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

  // Compute upper bound for the length of range
  int d_min = length_longest_common_prefix(
    morton_code_list, idx, idx - d, num_objects
  );

  int l_max = 2;
  while (
    length_longest_common_prefix(
      morton_code_list, idx, idx + l_max * d, num_objects
    ) > d_min
  ) {
    l_max *= 2;
  }

  // Find the other end using binary search
  int l = 0;
  int divider = 2;
  int t = l_max / divider;
  while (t >= 1) {
    if (
      length_longest_common_prefix(
        morton_code_list, idx, idx + (l + t) * d, num_objects
      ) > d_min
    ) {
      l += t;
    }
    divider *= 2;
    t = l_max / divider;
  }
  int j = idx + l * d;

  // Find the split position using binary search
  int d_node = length_longest_common_prefix(
    morton_code_list, idx, j, num_objects
  );
  int s = 0;
  divider = 2;
  t = ceilf(l / divider);
  while (t >= 1) {
    if (length_longest_common_prefix(
      morton_code_list, idx, idx + (s + t) * d, num_objects
    ) > d_node) {
      s += t;
    }
    divider *= 2;
    t = ceilf(l / divider);
  }
  int gamma = idx + s * d + min(d, 0);
  Node *left, *right;

  if (min(idx, j) == gamma) {
    left = leaf_list[gamma];
  } else {
    left = node_list[gamma];
  }

  if (max(idx, j) == gamma + 1) {
    right = leaf_list[gamma + 1];
  } else {
    right = node_list[gamma + 1];
  }

  node_list[idx] -> set_left_child(left);
  node_list[idx] -> set_right_child(right);

  left -> set_parent(node_list[idx]);
  right -> set_parent(node_list[idx]);

}

__global__ void compute_node_bounding_boxes(
  Leaf** leaf_list, Node** node_list, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects) return;

  Node* current_node = leaf_list[idx];
  float bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max;

  if (current_node -> parent == NULL) {
    printf("Leaf %d has no parent.\n", idx);
    return;
  }

  // __syncthreads();
  // long long int my_time = clock64();
  // printf("my_time = %lu\n", my_time);

  // while(current_node -> parent -> visited && current_node != node_list[0]) {
  while(current_node != node_list[0]) {
    __syncthreads();
    printf("Inside\n");
    current_node = current_node -> parent;

    if (
      current_node -> left -> bounding_box == NULL ||
      current_node -> right -> bounding_box == NULL
    )
      return;

    compute_bb_union(
      current_node -> left -> bounding_box,
      current_node -> right -> bounding_box,
      bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max
    );
    current_node -> bounding_box = new BoundingBox(
      bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max
    );

    if (current_node == node_list[0]) {
      printf("Here!\n");
      current_node -> bounding_box -> print_bounding_box();
    }

  }

  current_node -> parent -> mark_visited();
}

#endif
