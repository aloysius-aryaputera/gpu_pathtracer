#include <iostream>
#include <math.h>

#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

int main(int argc, char **argv) {
  int nx = 640, ny = 640, tx = 8, ty = 8;
  int num_pixels = nx*ny;
  size_t fb_size = 3 * num_pixels * sizeof(float);

  // allocate FB
  float *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  save_image(fb, nx, ny, argv[1]);

  checkCudaErrors(cudaFree(fb));

  return 0;
}
