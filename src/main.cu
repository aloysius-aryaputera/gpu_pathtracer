#include <iostream>
#include <math.h>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/triangle.h"
#include "model/ray.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"
#include "util/read_file_util.h"

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

__global__ void create_world(
  Triangle** geom_array, float *x, float *y, float *z, int *point_1_idx,
  int *point_2_idx, int *point_3_idx, int num_triangles
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int idx = 0; idx < num_triangles; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]])
        );
      }
    }
}

__global__ void set_camera(Camera** camera, int width, int height) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(camera) = new Camera(
      vec3(0, -8, 5), vec3(0, 0, 0), vec3(0, 0, 1), 45, width, height
    );
  }
}

__global__ void free_world(Triangle **geom_array, Camera **camera, int n) {
    for (int i = 0; i < n; i++){
      delete *(geom_array + i);
    }
    delete *camera;
}

int main(int argc, char **argv) {
  int im_width = 1000, im_height = 1000;
  int tx = 8, ty = 8;

  Triangle** my_geom_2;
  Camera **my_camera;
  float *fb;
  size_t fb_size = 3 * im_width * im_height * sizeof(float);

  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  checkCudaErrors(cudaMallocManaged((void **)&my_camera, sizeof(Camera *)));
  set_camera<<<1, 1>>>(my_camera, im_width, im_height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  float *x, *y, *z;
  int *point_1_idx, *point_2_idx, *point_3_idx;
  int num_triangles = 0;

  checkCudaErrors(cudaMallocManaged((void **)&x, 9999 * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y, 9999 * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&z, 9999 * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&point_1_idx, 9999 * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_2_idx, 9999 * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_3_idx, 9999 * sizeof(int)));

  extract_triangle_data(
    argv[2], x, y, z, point_1_idx, point_2_idx, point_3_idx, num_triangles
  );
  checkCudaErrors(cudaMallocManaged((void **)&my_geom_2, 999999 * sizeof(Triangle *)));
  create_world<<<1, 1>>>(my_geom_2, x, y, z, point_1_idx, point_2_idx, point_3_idx, num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("num_triangles = %d\n", num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());


  dim3 blocks(im_width / tx + 1, im_height / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, my_camera, my_geom_2, num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Saving image!\n");
  save_image(fb, im_width, im_height, argv[1]);
  printf("Image saved!\n");

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(my_geom_2, my_camera, 9999);
  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom_2));

  cudaDeviceReset();

  return 0;
}
