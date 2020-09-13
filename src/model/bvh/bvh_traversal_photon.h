//File: bvh_traversal_photon.h
#ifndef BVH_TRAVERSAL_PHOTON_H
#define BVH_TRAVERSAL_PHOTON_H

#include "../grid/bounding_box.h"
#include "../grid/bounding_sphere.h"
#include "../point/ppm_hit_point.h"
#include "../point/point.h"
#include "bvh.h"

__device__ bool _traverse_bvh_photon(
  Node* bvh_root, PPMHitPoint* hit_point, vec3 &iterative_flux, 
  int &num_photons
) {
  Node* stack[400];
  Node *child_l, *child_r;
  Point *point;
  bool intersection_l, intersection_r, traverse_l, traverse_r;
  bool is_inside = false;
  bool pts_found = false;
  int idx_stack_top = 0;

  float factor;
  iterative_flux = vec3(0.0, 0.0, 0.0);

  stack[idx_stack_top] = nullptr;
  idx_stack_top++;

  Node *node = bvh_root;
  do {

    child_l = node -> left;
    child_r = node -> right;

    intersection_l = child_l -> bounding_box -> is_intersection(
      hit_point -> bounding_sphere);
    intersection_r = child_r -> bounding_box -> is_intersection(
      hit_point -> bounding_sphere);

    if (intersection_l && child_l -> is_leaf) {
      is_inside = hit_point -> bounding_sphere -> is_inside(
        child_l -> point -> location);
      if (is_inside) {
        point = child_l -> point;
        pts_found = true;
	num_photons++;
        factor = max(0.0, dot(hit_point -> normal, -(point -> direction)));
        iterative_flux += factor * point -> color;
	//printf("factor = %f, point -> color = (%f, %f, %f)", 
	//       factor, point -> color.r(), point -> color.g(), point -> color.b());
      }
    }

    if (intersection_r && child_r -> is_leaf) {
      is_inside = hit_point -> bounding_sphere -> is_inside(
	child_r -> point -> location);
      if (is_inside) {
        point = child_r -> point;
        pts_found = true;
	num_photons++;
        factor = max(0.0, dot(hit_point -> normal, -(point -> direction)));
        iterative_flux += factor * point -> color;
	//printf("factor = %f, point -> color = (%f, %f, %f)", 
	//	factor, point -> color.r(), point -> color.g(), point -> color.b());
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

  return pts_found;
}

__global__
void update_hit_point_parameters(
  Node** photon_node_list, PPMHitPoint** hit_point_list, int num_hit_point
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_hit_point) return;
  if (hit_point_list[idx] -> location.vector_is_inf()) return;
  vec3 iterative_flux = vec3(0.0, 0.0, 0.0);
  int extra_photons = 0;
  bool photon_found = _traverse_bvh_photon(
    photon_node_list[0], hit_point_list[idx], iterative_flux, extra_photons
  );
  if (photon_found)
    hit_point_list[idx] -> update_accummulated_reflected_flux(
      iterative_flux, extra_photons
    );
//    printf("iterative_flux = (%f, %f, %f)\n", 
//		    iterative_flux.r(), iterative_flux.g(), iterative_flux.b());
}

#endif
