#include <iostream>
#include <math.h>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/triangle.h"
#include "model/ray.h"
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

__global__ void create_world(Triangle* geom_array, Camera* camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      vec3 point_1 = vec3(0, 0, 0), point_2 = vec3(1, 1, 0), point_3 = vec3(1, 1, 1);
      Triangle* my_triangle = new Triangle(point_1, point_2, point_3);
      camera = new Camera(
        vec3(0, -5, 0), vec3(0, 0, 0), vec3(0, 0, 1), 45, 100, 100
      );
      printf("Camera width = %d, height = %d\n", camera -> width, camera -> height);
      geom_array = my_triangle;
    }
}

__global__ void create_world(Triangle** geom_array) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      vec3 point_1 = vec3(0, 0, 0), point_2 = vec3(1, 1, 0), point_3 = vec3(1, 1, 1);
      *(geom_array) = new Triangle(point_1, point_2, point_3);
    }
}

int main(int argc, char **argv) {
  int tx = 8, ty = 8;

  Triangle** my_geom_2;
  Triangle* my_geom;
  vec3 point_1 = vec3(0, 0, 0), point_2 = vec3(1, 1, 0), point_3 = vec3(1, 1, 1);
  Triangle* my_triangle = new Triangle(point_1, point_2, point_3);
  Camera *my_camera = new Camera(
    vec3(0, -5, 0), vec3(0, 0, 0), vec3(0, 0, 1), 45, 100, 100
  );
  printf("Camera width = %d, height = %d\n", my_camera -> width, my_camera -> height);
  my_geom = my_triangle;

  size_t fb_size = 3 * 100 * 100 * sizeof(float);

  printf("fb_size = %lu\n", fb_size);

  // allocate FB
  float *fb;
  checkCudaErrors(cudaMallocManaged(&fb, fb_size));

  checkCudaErrors(cudaMalloc((void **)&my_geom_2, 1 * sizeof(Triangle *)));
  create_world<<<1, 1>>>(my_geom_2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // print_vec3((*my_geom_2) -> point_1);
  // print_vec3((*my_geom_2) -> point_2);
  // print_vec3((*my_geom_2) -> point_3);
  // print_vec3((*my_geom_2) -> normal);

  printf("Camera width = %d, height = %d\n", my_camera -> width, my_camera -> height);

  checkCudaErrors(cudaMallocManaged(&my_geom, 1 * sizeof(Triangle)));
  checkCudaErrors(cudaMallocManaged(&my_camera, sizeof(Camera)));

  dim3 blocks(100 / tx + 1, 100 / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, my_camera, my_geom_2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Camera width = %d, height = %d\n", my_camera -> width, my_camera -> height);

  save_image(fb, 100, 100, argv[1]);

  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom));

  return 0;
}
