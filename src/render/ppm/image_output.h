//File: image_output.h
#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

#include "../../model/camera.h"
#include "../../model/point/ppm_hit_point.h"

__global__
void get_ppm_image_output(
  int ppm_pass, vec3 *fb, PPMHitPoint** hit_point_list, Camera **camera, 
  int image_mode
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;

  if((j >= camera[0] -> width) || (i >= camera[0] -> height)) {
    return;
  }

  int pixel_index = i * (camera[0] -> width) + j;

  bool write = false;
  if (j == 208 && i == 179) {
    write = true;
  }

  fb[pixel_index] = hit_point_list[pixel_index] -> compute_pixel_color(
    ppm_pass, image_mode, write
  );

}

#endif
