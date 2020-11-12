//File: bounding_sphere.h
#ifndef BOUNDING_SPHERE_H
#define BOUNDING_SPHERE_H

#include "../ray/ray.h"
#include "../vector_and_matrix/vec3.h"
#include "bounding_sphere.h"

class BoundingCylinder {
  public:
    __device__ BoundingCylinder(
      vec3 start, vec3 dir, float l_, float r_);
    __device__ bool intersect(BoundingSphere* sphere);
    __device__ bool point_inside_cylinder(
      vec3 loc, float &dist_perpendicular, float &dist_parallel, float buffer
    );

    Ray axis;
    float length, radius;
}

__device__ BoundingCylinder::BoundingCylinder(
  vec3 start, vec3 dir, float l_, float r_
) {
  this -> axis = Ray(start, dir);
  this -> l = l_;
  this -> r = r_;
}

__device__ bool point_inside_cylinder(
  vec3 loc, float &dist_perpendicular, float &dist_parallel, float buffer = 0
) {
  vec3 start_to_point = loc -  this -> axis.p0;
  float start_to_point_dist = start_to_point.length()
  float cos_theta = dot(
    this -> axis.dir, start_to_point / start_to_point_dist
  );
  float sin_theta = powf(1 - cos_theta * cos_theta, .5);
  dist_perpendicular = start_to_point_dist * sin_theta;
  if (dist_perpendicular <= (this -> r + buffer)) {
    dist_parallel = start_to_point_dist * cos_theta;
    if (
      (start_to_point_dist <= (this -> l + buffer)) &&
      (start_to_point_dist >= (-buffer))
    ) {
      return true;
    }  
  } 
  return false;
}

__device__ bool BoundingCylinder::intersect(BoundingSphere* sphere) {
  float dist_perpendicular, dist_parallel;
  return this -> point_inside_cylinder(
    sphere -> center, dist_perpendicular, dist_parallel, sphere -> r 
  );
  //vec3 start_to_sphere_center = sphere -> center -  this -> axis.p0;
  //float start_to_sphere_center_dist = start_to_sphere_center.length()
  //float cos_theta = dot(
  //  this -> axis.dir, start_to_sphere_center / start_to_sphere_center_dist
  //);
  //float sin_theta = powf(1 - cos_theta * cos_theta, .5);
  //float d = start_to_sphere_center_dist * sin_theta;
  //if (d <= (this -> r + sphere -> r)) {
  //  float start_to_projected_dist = start_to_sphere_center_dist * cos_theta;
  //  if (start_to_projected_dist <= this -> l) {
  //    return true;
  //  }  
  //} 
  //return false;
}

#endif
