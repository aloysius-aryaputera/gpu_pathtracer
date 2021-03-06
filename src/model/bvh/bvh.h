//File: bvh.h
#ifndef BVH_H
#define BVH_H

#include "../../util/bvh_util.h"
#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/bounding_cone.h"
#include "../grid/bounding_sphere.h"
#include "../point/point.h"
#include "../ray/ray.h"

class Node {
  public:
    __host__ __device__ Node();
    __device__ Node(int idx_);
    __device__ Primitive* get_object();
    __device__ void assign_object(
      Primitive* object_, bool bounding_cone_required=false);
    __device__ void assign_point(Point* point_);
    __device__ void release_point(Point* point_);
    __device__ void set_left_child(Node* left_);
    __device__ void set_right_child(Node* right_);
    __device__ void set_parent(Node* parent_);
    __device__ void mark_visited();
    __device__ float compute_importance(
      vec3 point, vec3 normal, vec3 kd
    );
    __device__ void set_energy(vec3 energy_);

    Node *left, *right, *parent;
    bool visited, is_leaf;
    BoundingBox *bounding_box;
    BoundingCone *bounding_cone;
    BoundingSphere *bounding_sphere;
    Primitive *object;
    Point *point;
    int idx;
    vec3 energy;
};

__device__ Node::Node() {
  this -> visited = false;
  this -> is_leaf = false;
  this -> idx = -30;
  this -> energy = vec3(0.0, 0.0, 0.0);
}

__device__ Node::Node(int idx_) {
  this -> visited = false;
  this -> is_leaf = false;
  this -> idx = idx_;
  this -> energy = vec3(0.0, 0.0, 0.0);
}

__device__ void Node::set_energy(vec3 energy_) {
  this -> energy = energy_;
}

__device__ float Node::compute_importance(
  vec3 point, vec3 normal, vec3 kd	
) {
  vec3 dir = point - this -> bounding_box -> center;
  float cone_angle = this -> bounding_box -> compute_covering_cone_angle(
    point);
  float incident_angle = this -> bounding_box -> compute_incident_angle(
    point, normal);
  float min_incident_angle = fmaxf(incident_angle - cone_angle, 0);
  float min_angle_to_point = \
  this -> bounding_box -> compute_minimum_angle_to_shading_point(
    point, this -> bounding_cone -> axis, this -> bounding_cone -> theta_0, 
    cone_angle);
  //float multiplier;
  vec3 effective_energy;

  //if (min_angle_to_point < this -> bounding_cone -> theta_e) {
  //  //multiplier = fmaxf(cos(min_angle_to_point), 0);
  //  multiplier = cos(min_angle_to_point);
  //} else {
  //  multiplier = 0;
  //}

  if (this -> is_leaf) {
    effective_energy = this -> object -> compute_directed_energy(
      point, normal);
  } else {
    effective_energy = this -> energy;
  }

  //importance =  (
  //  kd * abs(cos(min_incident_angle)) * effective_energy * \
  //  multiplier / dir.squared_length()).length();
  //return importance;
  return effective_energy.length() / dir.squared_length();
  //return 1;
}

__device__ Primitive* Node::get_object() {
  return this -> object;
}

__device__ void Node::assign_object(
  Primitive* object_, bool bounding_cone_required
) {
  this -> object = object_;
  this -> bounding_box = object_ -> get_bounding_box();
  if (bounding_cone_required) {
    this -> bounding_cone = new BoundingCone(
      this -> object -> get_fixed_normal(), 0, M_PI / 2.0
    );
  }
  this -> is_leaf = true;
  this -> energy = object_ -> get_energy();
}

__device__ void Node::assign_point(Point* point_) {
  this -> point = point_;
  this -> bounding_box = point_ -> bounding_box;
  this -> bounding_sphere = point_ -> bounding_sphere;
  this -> is_leaf = true;
}

__device__ void Node::release_point(Point* point_) {
  this -> point = nullptr;
  this -> bounding_box = nullptr;
  this -> bounding_sphere = nullptr;
  this -> is_leaf = false;
}

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
