//File: bvh_traversal_photon.h
#ifndef BVH_TRAVERSAL_PHOTON_H
#define BVH_TRAVERSAL_PHOTON_H

#include "../../param.h"
#include "../geometry/primitive.h"
#include "../grid/bounding_box.h"
#include "../grid/bounding_sphere.h"
#include "../material/material.h"
#include "../point/ppm_hit_point.h"
#include "../point/point.h"
#include "bvh.h"
#include "bvh_traversal.h"

__device__ vec3 _compute_volume_photon_contribution(
  Point *photon, PPMHitPoint* hit_point, Material *medium
) {
  float dist_perpendicular, dist_parallel;
  float kernel_value = hit_point -> compute_ppm_volume_kernel(
    photon -> location, dist_perpendicular, dist_parallel
  );
  if (kernel_value > 0) {
    float transmittance = medium -> get_transmittance(dist_parallel);
    float phase_function_value = medium -> get_phase_function_value(
      hit_point -> bounding_cylinder -> axis.dir, photon -> direction 
    );
    //printf("kernel = %5.2f; transmittance = %5.2f; scattering_coef = %5.2f; phase_function_value = %5.2f; photon_color = (%5.2f, %5.2f, %5.2f)\n", kernel_value, transmittance, medium -> scattering_coef, phase_function_value, photon -> color.r(), photon -> color.g(), photon -> color.b());
    return kernel_value * transmittance * medium -> scattering_coef * 
      phase_function_value * photon -> color;  
  } else {
    return vec3(0.0, 0.0, 0.0);
  }
}

__device__ void traverse_bvh_volume_photon(
  Node* bvh_root, PPMHitPoint* hit_point, Material *medium, vec3 filter
) {
  Node* stack[400];
  Node *child_l, *child_r;
  Ray ray;
  vec3 ray_dir;
  bool intersection_l, intersection_r, traverse_l, traverse_r;
  int idx_stack_top = 0, num_intersections = 0;
  hit_record rec;
  vec3 photon_contribution = vec3(0.0, 0.0, 0.0);

  stack[idx_stack_top] = nullptr;
  idx_stack_top++;

  Node *node = bvh_root;
  do {

    child_l = node -> left;
    child_r = node -> right;

    intersection_l = hit_point -> bounding_cylinder -> is_intersection(
      child_l -> bounding_sphere);
    intersection_r = hit_point -> bounding_cylinder -> is_intersection(
      child_r -> bounding_sphere);

    if (intersection_l && child_l -> is_leaf) {
      num_intersections++;
      photon_contribution += _compute_volume_photon_contribution(
        child_l -> point, hit_point, medium
      ); 
    }

    if (intersection_r && child_r -> is_leaf) {
      num_intersections++;
      photon_contribution += _compute_volume_photon_contribution(
        child_r -> point, hit_point, medium
      ); 
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
        idx_stack_top++;
      }
    }

  } while(idx_stack_top > 0 && idx_stack_top < 400 && node != nullptr);

  hit_point -> add_tmp_accummulated_lm(filter * photon_contribution); 
  //if (num_intersections > 0) {
  //  print_vec3(filter * photon_contribution);
  //}
}

__device__ bool _traverse_bvh_surface_photon(
  Node* bvh_root, Node* geom_bvh_root, PPMHitPoint* hit_point, 
  vec3 &iterative_flux, int &num_photons
) {
  Node* stack[400];
  Node *child_l, *child_r;
  Point *point;
  Ray ray;
  vec3 ray_dir;
  bool intersection_l, intersection_r, traverse_l, traverse_r, geom_hit;
  bool is_inside = false;
  bool pts_found = false;
  int idx_stack_top = 0;
  hit_record rec;

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
	ray_dir = point -> prev_location - hit_point -> location;
	ray = Ray(hit_point -> location, ray_dir);
	geom_hit = traverse_bvh(geom_bvh_root, ray, rec);
	if (
	  geom_hit && abs(rec.t - ray_dir.length()) < SMALL_DOUBLE
	) {
          pts_found = true;
	  num_photons++;
          factor = max(0.0, dot(hit_point -> normal, -(point -> direction)));
          iterative_flux += factor * point -> color;
	}
      }
    }

    if (intersection_r && child_r -> is_leaf) {
      is_inside = hit_point -> bounding_sphere -> is_inside(
	child_r -> point -> location);
      if (is_inside) {
        point = child_r -> point;
	ray_dir = point -> prev_location - hit_point -> location;
	ray = Ray(hit_point -> location, ray_dir);
	geom_hit = traverse_bvh(geom_bvh_root, ray, rec);
	if (
	  geom_hit && abs(rec.t - ray_dir.length()) < SMALL_DOUBLE
	) {
          pts_found = true;
	  num_photons++;
          factor = max(0.0, dot(hit_point -> normal, -(point -> direction)));
          iterative_flux += factor * point -> color;
	}
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
        idx_stack_top++;
      }
    }

  } while(idx_stack_top > 0 && idx_stack_top < 400 && node != nullptr);

  return pts_found;
}

__global__
void update_hit_point_parameters(
  int iteration, Node** photon_node_list, Node** geom_node_list, 
  PPMHitPoint** hit_point_list, int num_hit_point, int emitted_photon_per_pass
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_hit_point) return;
  if (hit_point_list[idx] -> location.vector_is_inf()) return;
  vec3 iterative_flux = vec3(0.0, 0.0, 0.0);
  int extra_photons = 0;
  bool photon_found = _traverse_bvh_surface_photon(
    photon_node_list[0], geom_node_list[0], hit_point_list[idx], 
    iterative_flux, extra_photons
  );
  if (photon_found)
    hit_point_list[idx] -> update_accummulated_reflected_flux(
      iteration, iterative_flux, extra_photons, emitted_photon_per_pass
    );
}

#endif
