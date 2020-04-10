//File: bvh_build.h
#ifndef BVH_BUILD_H
#define BVH_BUILD_H

#include "../../util/bvh_util.h"
#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../ray/ray.h"

class Node {
  public:
    __host__ __device__ Node() {
      this -> visited = false;
      this -> is_leaf = false;
      this -> idx = -30;
    }
    __device__ Node(int idx_) {
      this -> visited = false;
      this -> is_leaf = false;
      this -> idx = idx_;
    }
    __device__ Primitive* get_object() {
      return this -> object;
    }
    __device__ void assign_object(Primitive* object_) {
      this -> object = object_;
      this -> bounding_box = object_ -> get_bounding_box();
      this -> is_leaf = true;
    }
    __device__ void set_left_child(Node* left_);
    __device__ void set_right_child(Node* right_);
    __device__ void set_parent(Node* parent_);
    __device__ void mark_visited();

    Node *left, *right, *parent;
    bool visited, is_leaf;
    BoundingBox *bounding_box;
    Primitive *object;
    int idx;
};

__device__ bool traverse_bvh(Node* bvh_root, Ray ray, hit_record &rec);

__device__ bool traverse_bvh(Node* bvh_root, Ray ray, hit_record &rec) {
  Node* stack[400];
  Node *child_l, *child_r;
  bool intersection_l, intersection_r, traverse_l, traverse_r, hit = false;
  bool intersection_found = false;
  float t;
  hit_record cur_rec;
  int idx_stack_top = 0;

  rec.t = INFINITY;
  stack[idx_stack_top] = nullptr;
  idx_stack_top++;

  Node *node = bvh_root;
  do {

    child_l = node -> left;
    child_r = node -> right;

    intersection_l = child_l -> bounding_box -> is_intersection(ray, t);
    intersection_r = child_r -> bounding_box -> is_intersection(ray, t);

    if (intersection_l && child_l -> is_leaf) {
      hit = child_l -> get_object() -> hit(ray, rec.t, cur_rec);
      if (hit) {
        rec = cur_rec;
        intersection_found = true;
      }
    }

    if (intersection_r && child_r -> is_leaf) {
      hit = child_r -> get_object() -> hit(ray, rec.t, cur_rec);
      if (hit) {
        rec = cur_rec;
        intersection_found = true;
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

  } while(idx_stack_top > 0 && idx_stack_top < 400 && node != nullptr);

  return intersection_found;
}

__global__ void extract_morton_code_list(
  Primitive** object_list, unsigned int* morton_code_list, int num_objects
);

__global__ void build_node_list(Node** node_list, int num_objects);

__global__ void set_node_relationship(
  Node** node_list, Node** leaf_list, unsigned int* morton_code_list,
  int num_objects
);

__global__ void build_leaf_list(
  Node** leaf_list, Primitive **object_list, int num_objects
);

__global__ void compute_morton_code_batch(
  Primitive **object_array, BoundingBox **world_bounding_box, int num_triangles
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
  Primitive **object_array, BoundingBox **world_bounding_box, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_triangles) return;

  object_array[idx] -> get_bounding_box() -> compute_normalized_center(
    world_bounding_box[0]
  );
  object_array[idx] -> get_bounding_box() -> compute_bb_morton_3d();
}

__global__ void build_leaf_list(
  Node** leaf_list, Primitive **object_list, int num_triangles
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_triangles) return;
  leaf_list[idx] = new Node(idx);
  (leaf_list[idx]) -> assign_object(object_list[idx]);
}

__global__ void build_node_list(Node** node_list, int num_objects) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_objects - 1) return;
  node_list[idx] = new Node(idx);
  (node_list[idx]) -> bounding_box = new BoundingBox();
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
  Node** node_list, Node** leaf_list, unsigned int* morton_code_list,
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

__global__ void check(Node** leaf_list, Node** node_list, int num_objects) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < num_objects - 1) {
    if (leaf_list[idx] -> bounding_box == nullptr) {
      printf("leaf_list[%d] does not have bounding box\n", idx);
    }
    if (leaf_list[idx] -> object == nullptr) {
      printf("leaf_list[%d] does not have object\n", idx);
    }
    if (node_list[idx] -> is_leaf) {
      printf("node_list[%d] is a leaf\n", idx);
    }
    if (node_list[idx] -> bounding_box == nullptr) {
      printf("node_list[%d] does not have bounding box\n", idx);
    }
    if (node_list[idx] -> left == nullptr) {
      printf("node_list[%d] does not have left child\n", idx);
    }
    if (node_list[idx] -> right == nullptr) {
      printf("node_list[%d] does not have right child\n", idx);
    }
  }

  if (idx == num_objects - 1) {
    if (leaf_list[idx] -> bounding_box == nullptr) {
      printf("leaf_list[%d] does not have bounding box\n", idx);
    }
    if (leaf_list[idx] -> object == nullptr) {
      printf("leaf_list[%d] does not have object\n", idx);
    }
  }
}

__global__ void compute_node_bounding_boxes(
  Node** leaf_list, Node** node_list, int num_objects
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= num_objects) return;

  Node* current_node = leaf_list[idx];
  float bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max;

  if (current_node -> parent == nullptr) {
    printf("Leaf %d has no parent.\n", idx);
    return;
  }

  while(current_node != node_list[0]) {
    current_node = current_node -> parent;

    if (
      !(current_node -> left -> bounding_box -> initialized) ||
      !(current_node -> right -> bounding_box -> initialized)
    )
      return;

    compute_bb_union(
      current_node -> left -> bounding_box,
      current_node -> right -> bounding_box,
      bb_x_min, bb_x_max, bb_y_min, bb_y_max, bb_z_min, bb_z_max
    );

    if (!(current_node -> bounding_box -> initialized)) {
      current_node -> bounding_box -> initialize(
        bb_x_min, bb_x_max,
        bb_y_min, bb_y_max,
        bb_z_min, bb_z_max
      );
    }

    if (current_node == node_list[0]) {
      printf("We have reached the root!\n");
      current_node -> bounding_box -> print_bounding_box();
    }
  }
}

#endif
