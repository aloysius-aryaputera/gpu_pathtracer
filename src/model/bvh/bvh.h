//File: bvh.h
#ifndef BVH_H
#define BVH_H

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

#endif
