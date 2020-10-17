//File: ppm_hit_point.h
#ifndef PPM_HIT_POINT_H
#define PPM_HIT_POINT_H

#include "../grid/bounding_sphere.h"
#include "../vector_and_matrix/vec3.h"

class PPMHitPoint {
  private:
    float ppm_alpha, pdf;

    __device__ void _create_bounding_sphere();

  public:
    vec3 location, normal, accummulated_reflected_flux, filter, direct_radiance;
    vec3 accummulated_indirect_radiance;
    float current_photon_radius;
    int accummulated_photon_count;
    BoundingSphere *bounding_sphere;

    __host__ __device__ PPMHitPoint();
    __device__ PPMHitPoint(
      vec3 location_, float radius_, vec3 filter_, vec3 normal_,
      float ppm_alpha_
    ); 
    __device__ void update_parameters(
      vec3 location_, float radius_, vec3 filter_, vec3 normal_,
      float pdf_
    );
    __device__ void update_radius(float radius_);
    __device__ void update_accummulated_reflected_flux(
      int iteration, vec3 iterative_total_photon_flux, int extra_photons,
      int emitted_photon_per_pass
    );
    __device__ void update_direct_radiance(vec3 extra_direct_radiance);
    __device__ vec3 compute_pixel_color(int num_passes, int type);
};

__device__ vec3 PPMHitPoint::compute_pixel_color(int num_passes, int type) {
  vec3 mean_radiance, mean_direct_radiance, mean_indirect_radiance;

  mean_radiance = (
    this -> direct_radiance + this -> accummulated_indirect_radiance
  ) / float(num_passes);
  mean_direct_radiance = this -> direct_radiance / float(num_passes);
  mean_indirect_radiance = this -> accummulated_indirect_radiance / float(
    num_passes);

  if (type == 0) {
    return de_nan(mean_direct_radiance);
  } else if (type == 1) {
    return de_nan(mean_indirect_radiance);
  } else {
    return de_nan(mean_radiance);
  }
}

__device__ void PPMHitPoint::update_radius(float radius_) {
  this -> current_photon_radius = radius_;
  this -> bounding_sphere -> assign_new_radius(this -> current_photon_radius);
}

__device__ void PPMHitPoint::update_direct_radiance(vec3 extra_direct_radiance) {
  this -> direct_radiance += de_nan(extra_direct_radiance);
}

__device__ void PPMHitPoint::update_accummulated_reflected_flux(
  int iteration, vec3 iterative_total_photon_flux, int extra_photons,
  int emitted_photon_per_pass
) {
  float new_radius;
  //if (extra_photons > 0) {
  //  new_radius = this -> current_photon_radius * powf(
  //    (this -> accummulated_photon_count + this -> ppm_alpha * extra_photons) /
  //    (this -> accummulated_photon_count + extra_photons),
  //    0.5
  //  );
  //} else {
  //  new_radius = this -> current_photon_radius;
  //}
  if (iteration >= 2) {
    new_radius = this -> current_photon_radius * powf(
      (iteration + this -> ppm_alpha) / (iteration + 1), 0.5
    );
  } else {
    new_radius = this -> current_photon_radius;
  }

  this -> accummulated_indirect_radiance += de_nan(
    this -> filter * iterative_total_photon_flux / 
    (emitted_photon_per_pass * M_PI * powf(this -> current_photon_radius, 2))
  );

  //this -> accummulated_photon_count += (this -> ppm_alpha * extra_photons);
  //this -> accummulated_reflected_flux = (
  //  this -> accummulated_reflected_flux +
  //  de_nan(this -> filter * iterative_total_photon_flux)
  //) * powf(new_radius / this -> current_photon_radius, 2);
  this -> current_photon_radius = new_radius;
  this -> bounding_sphere -> assign_new_radius(new_radius);
}

__device__ PPMHitPoint::PPMHitPoint(
  vec3 location_, float radius_, vec3 filter_, vec3 normal_, float ppm_alpha_
) {
  this -> location = location_;
  this -> current_photon_radius = radius_;
  this -> filter = filter_;
  this -> normal = normal_;
  this -> ppm_alpha = ppm_alpha_;
  this -> accummulated_reflected_flux = vec3(0.0, 0.0, 0.0);
  this -> accummulated_indirect_radiance = vec3(0.0, 0.0, 0.0);
  this -> direct_radiance = vec3(0.0, 0.0, 0.0);
  this -> accummulated_photon_count = 0;
  this -> pdf = 1;

  this -> _create_bounding_sphere();
}

__device__ void PPMHitPoint::_create_bounding_sphere() {
  this -> bounding_sphere = new BoundingSphere(
    this -> location, this -> current_photon_radius
  );
}

__device__ void PPMHitPoint::update_parameters(
  vec3 location_, float radius_, vec3 filter_, vec3 normal_, float pdf_
) {
  this -> location = location_;
  this -> current_photon_radius = radius_;
  this -> filter = filter_;
  this -> normal = normal_;
  this -> pdf = pdf_;

  this -> bounding_sphere -> assign_new_center(this -> location);
  this -> bounding_sphere -> assign_new_radius(
    this -> current_photon_radius);
}

#endif
