//File: ppm_hit_point.h
#ifndef PPM_HIT_POINT_H
#define PPM_HIT_POINT_H

#include "../grid/bounding_sphere.h"
#include "../vector_and_matrix/vec3.h"

class PPMHitPoint {
  private:
    __device__ void _create_bounding_sphere();

  public:
    vec3 location, normal, accummulated_reflected_flux, filter;
    float current_photon_radius;
    int accummulated_photon_count;
    BoundingSphere *bounding_sphere;

    __host__ __device__ PPMHitPoint();
    __device__ PPMHitPoint(
      vec3 location_, float radius_, vec3 filter_, vec3 normal_
    ); 
    __device__ void update_parameters(
      vec3 location_, float radius_, vec3 filter_, vec3 normal_
    );
    __device__ void update_radius(float radius_);
    __device__ void add_accummuldated_photon_count(float extra_photons);
};

__device__ void PPMHitPoint::update_radius(float radius_) {
  this -> current_photon_radius = radius_;
  this -> bounding_sphere -> assign_new_radius(this -> current_photon_radius);
}

__device__ void PPMHitPoint::add_accummuldated_photon_count(
  float extra_photons
) {
  this -> accummulated_photon_count += extra_photons;
}

__device__ PPMHitPoint::PPMHitPoint(
  vec3 location_, float radius_, vec3 filter_, vec3 normal_
) {
  this -> location = location_;
  this -> current_photon_radius = radius_;
  this -> filter = filter_;
  this -> normal = normal_;

  this -> _create_bounding_sphere();
}

__device__ void PPMHitPoint::_create_bounding_sphere() {
  this -> bounding_sphere = new BoundingSphere(
    this -> location, this -> current_photon_radius
  );
}

__device__ void PPMHitPoint::update_parameters(
  vec3 location_, float radius_, vec3 filter_, vec3 normal_
) {
  this -> location = location_;
  this -> current_photon_radius = radius_;
  this -> filter = filter_;
  this -> normal = normal_;

  this -> bounding_sphere -> assign_new_center(this -> location);
  this -> bounding_sphere -> assign_new_radius(
    this -> current_photon_radius);
}

#endif
