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

      if (object_ -> get_bounding_box() == NULL) {
        printf("object has no bounding_box!\n");
      }

      this -> bounding_box = object_ -> get_bounding_box();

      if (this -> bounding_box == NULL) {
        printf("leaf has no bounding_box!\n");
      }
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

__global__ void build_leaf_list(
  Leaf** leaf_list, Primitive **object_list, int num_objects
);

__global__ void compute_morton_code_batch(
  Primitive **object_array, Grid **grid, int num_triangles
);

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

  int start = idx;

  // Determine end
  int end = start + d;
  int min_delta = length_longest_common_prefix(
    morton_code_list, idx, idx - d, num_objects
  );
  int current_delta;
  bool flag = TRUE;
  while (end >= 0 && end <= num_objects - 1 && flag) {
    current_delta = length_longest_common_prefix(
      morton_code_list, start, end, num_objects
    );
    if (current_delta > min_delta) {
      end += d;
    } else {
      end -= d;
      flag = FALSE;
    }
  }
  if (end < 0) end = 0;
  if (end > num_objects - 1) end = num_objects - 1;

  // Determine split
  int split = start + d;
  int idx_1 = start, idx_2 = start + d;
  int min_delta_2 = length_longest_common_prefix(
    morton_code_list, idx_1, idx_2, num_objects
  );
  while((d > 0 && idx_2 <= end) || (d < 0 && idx_2 >= end)) {
    current_delta = length_longest_common_prefix(
      morton_code_list, idx_1, idx_2, num_objects
    );
    if (current_delta < min_delta_2) {
      split = idx_2;
      min_delta_2 = current_delta;
    }
    idx_1 += d;
    idx_2 += d;
  }

  if (
    start < 0 || end < 0 || start >= num_objects || end >= num_objects ||
    split < 0 || split >= num_objects
  ) {
    printf("start = %d; split = %d; end = %d; d = %d\n", start, split, end, d);
  }

  // Determine children
  Node *left, *right;
  if (d > 0) {
    if (split - start > 1)
      left = node_list[split - 1];
    else
      left = leaf_list[start];
    if (end - split > 0)
      right = node_list[split];
    else
      right = leaf_list[end];
  } else {
    if (start - split > 1)
      right = node_list[split + 1];
    else
      right = leaf_list[start];
    if (split - end > 0)
      left = node_list[split];
    else
      left = leaf_list[end];
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

  while(current_node != node_list[0]) {
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
      printf("We have reached the root!\n");
      current_node -> bounding_box -> print_bounding_box();
    }
  }
}

#endif
