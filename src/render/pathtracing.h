#ifndef PATHTRACING_H
#define PATHTRACING_H

__global__ void render(float *fb, int max_x, int max_y);

__global__
void render(float *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x * 3 + i * 3;
  fb[pixel_index + 0] = float(i) / max_x;
  fb[pixel_index + 1] = float(j) / max_y;
  fb[pixel_index + 2] = 0.2;
}

#endif
