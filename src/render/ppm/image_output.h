//File: image_output.h
#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

#include "../../model/camera.h"
#include "../../model/point/ppm_hit_point.h"

__global__
void ppm_image_output(
  int ppm_pass, int num_photon_per_pass, vec3 *fb, 
  PPMHitPoint** hit_point_list, Camera **camera
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  int pixel_index = i * (camera[0] -> width) + j;

  fb[pixel_index] = hit_point_list[pixel_index] -> compute_pixel_color(
    ppm_pass, num_photon_per_pass
  );

  if (fb[pixel_index].vector_is_inf()) {
    printf("idx %d (%d, %d) has values (%f, %f, %f), filter = (%f, %f, %f), and  %d photons.\n", 
		    pixel_index, j, i, 
		    fb[pixel_index].r(), fb[pixel_index].g(), fb[pixel_index].b(), 
		    hit_point_list[pixel_index] -> filter.r(),
                    hit_point_list[pixel_index] -> filter.g(),
		    hit_point_list[pixel_index] -> filter.b(),
		    hit_point_list[pixel_index] -> accummulated_photon_count);
  }

  if (fb[pixel_index].vector_is_nan()) {
    printf("idx %d (%d, %d) has values (%f, %f, %f), filter = (%f, %f, %f), and  %d photons.\n", 
		    pixel_index, j, i, 
		    fb[pixel_index].r(), fb[pixel_index].g(), fb[pixel_index].b(), 
		    hit_point_list[pixel_index] -> filter.r(),
                    hit_point_list[pixel_index] -> filter.g(),
		    hit_point_list[pixel_index] -> filter.b(),
		    hit_point_list[pixel_index] -> accummulated_photon_count);
  }

  if (fb[pixel_index].r() < 1E-6 && fb[pixel_index].g() < 1E-6 && fb[pixel_index].b() < 1E-6 && hit_point_list[pixel_index] -> accummulated_photon_count > 0) {
    printf("idx %d (%d, %d) has values (%f, %f, %f), filter = (%f, %f, %f), and  %d photons.\n", 
		    pixel_index, j, i, 
		    fb[pixel_index].r(), fb[pixel_index].g(), fb[pixel_index].b(), 
		    hit_point_list[pixel_index] -> filter.r(),
                    hit_point_list[pixel_index] -> filter.g(),
		    hit_point_list[pixel_index] -> filter.b(),
		    hit_point_list[pixel_index] -> accummulated_photon_count);
  }

  //printf(
  //  "(%f, %f, %f), %d, pixel[%d, %d] = (%f, %f, %f)\n", 
  //  hit_point_list[pixel_index] -> accummulated_reflected_flux.r(),
  //  hit_point_list[pixel_index] -> accummulated_reflected_flux.g(),
  //  hit_point_list[pixel_index] -> accummulated_reflected_flux.b(),
  //  hit_point_list[pixel_index] -> accummulated_photon_count,
  //  i, j, 
  //  fb[pixel_index].r(), fb[pixel_index].g(), fb[pixel_index].b());
}

#endif
