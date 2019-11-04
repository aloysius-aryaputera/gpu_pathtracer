#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/sphere.h"
#include "model/geometry/triangle.h"
#include "model/grid/bounding_box.h"
#include "model/grid/cell.h"
#include "model/grid/grid.h"
#include "model/material.h"
#include "model/ray.h"
#include "model/scene.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"
#include "util/read_file_util.h"
#include "world_lib.h"

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

__global__ void create_scene(
  Scene** scene, Camera** camera, Grid** grid, int *num_objects
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(scene) = new Scene(camera[0], grid[0], num_objects[0]);
  }
}

__global__ void render_init(
  int im_width, int im_height, curandState *rand_state, int *progress
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
  if (i == 0 && j == 0) {
    progress[0] = 0;
  }
  if ((j >= im_width) || (i >= im_height)) {
    return;
  }
  int pixel_index = i * im_width + j;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void free_world(
  Scene** scene, Grid **grid, Primitive **geom_array, Camera **camera, int n
) {
    for (int i = 0; i < n; i++){
      delete *(geom_array + i);
    }
    delete *camera;
    delete *grid;
    delete *scene;
}

int main(int argc, char **argv) {
  int im_width = std::stoi(argv[3]), im_height = std::stoi(argv[4]);
  int tx = std::stoi(argv[5]), ty = std::stoi(argv[6]);
  int *n_cell_x, *n_cell_y, *n_cell_z;
  int max_n_cell_x = 60, max_n_cell_y = 60, max_n_cell_z = 60;
  int tx2 = 32, ty2 = 32, max_num_objects_per_cell = 500, *progress;

  printf("im_width = %d, im_height = %d\n", im_width, im_height);
  printf("tx = %d, ty = %d\n", tx, ty);

  Scene** my_scene;
  Grid** my_grid;
  Cell** my_cell;
  Primitive **my_geom, **my_cell_geom;
  Camera **my_camera;
  vec3 *fb;
  int num_pixels = im_width * im_height;
  int max_num_vertices = 60000, max_num_faces = 110000;
  size_t fb_size = num_pixels * sizeof(vec3);
  curandState *rand_state;
  size_t rand_state_size = num_pixels * sizeof(curandState);
  size_t cell_geom_size = max_num_objects_per_cell * (max_n_cell_x) * \
    (max_n_cell_y) * (max_n_cell_z) * sizeof(Primitive*);
  clock_t start, stop;

  start = clock();

  float *x, *y, *z;
  int *point_1_idx, *point_2_idx, *point_3_idx;
  int *num_triangles;

  checkCudaErrors(cudaMallocManaged((void **)&num_triangles, sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&x, max_num_vertices * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y, max_num_vertices * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&z, max_num_vertices * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&point_1_idx, max_num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_2_idx, max_num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_3_idx, max_num_faces * sizeof(int)));

  extract_triangle_data(
    argv[2], x, y, z, point_1_idx, point_2_idx, point_3_idx, num_triangles
  );

  checkCudaErrors(cudaMallocManaged((void **)&my_geom, max_num_faces * sizeof(Primitive *)));
  checkCudaErrors(cudaMallocManaged((void **)&my_camera, sizeof(Camera *)));

  printf("Creating the world!\n");
  create_world_2<<<1, 1>>>(
    my_camera, my_geom, x, y, z, point_1_idx, point_2_idx, point_3_idx,
    num_triangles, im_width, im_height
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  printf("World created!\n");
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(y));
  checkCudaErrors(cudaFree(z));
  checkCudaErrors(cudaFree(point_1_idx));
  checkCudaErrors(cudaFree(point_2_idx));
  checkCudaErrors(cudaFree(point_3_idx));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocManaged((void **)&my_grid, sizeof(Grid *)));
  checkCudaErrors(cudaMallocManaged((void **)&my_cell, max_n_cell_x * max_n_cell_y * max_n_cell_z * sizeof(Cell *)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_x, sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_y, sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_z, sizeof(int)));
  printf("Creating the grid!\n");
  create_grid<<<1, 1>>>(
    my_camera, my_grid, my_geom, num_triangles, my_cell, n_cell_x, n_cell_y,
    n_cell_z, max_n_cell_x, max_n_cell_y, max_n_cell_z, max_num_objects_per_cell
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  printf("Grid created!\n");

  dim3 blocks2(n_cell_x[0] / tx2 + 1, n_cell_y[0] / ty2 + 1);
  dim3 threads2(tx2, ty2);
  checkCudaErrors(cudaMallocManaged((void **)&my_cell_geom, cell_geom_size));
  build_cell_array<<<blocks2, threads2>>>(my_grid, my_cell_geom);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  insert_objects<<<blocks2, threads2>>>(my_grid);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocManaged((void **)&my_scene, sizeof(Scene *)));
  create_scene<<<1, 1>>>(my_scene, my_camera, my_grid, num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  dim3 blocks(im_width / tx + 1, im_height / ty + 1);
  dim3 threads(tx, ty);
  checkCudaErrors(cudaMallocManaged((void **)&rand_state, rand_state_size));
  checkCudaErrors(cudaMallocManaged((void **)&progress, sizeof(int)));
  render_init<<<blocks, threads>>>(im_width, im_height, rand_state, progress);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  vec3 sky_emission = vec3(1, 1, 1);
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
  render<<<blocks, threads>>>(
    fb, my_scene, rand_state, std::stoi(argv[7]), std::stoi(argv[8]),
    sky_emission, progress
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Saving image!\n");
  save_image(fb, im_width, im_height, argv[1]);
  printf("Image saved!\n");

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  printf("\nThe rendering took %5.5f seconds.\n", timer_seconds);

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(my_scene, my_grid, my_geom, my_camera, 9999);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(my_scene));
  checkCudaErrors(cudaFree(my_grid));
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom));
  checkCudaErrors(cudaFree(num_triangles));
  checkCudaErrors(cudaFree(n_cell_x));
  checkCudaErrors(cudaFree(n_cell_y));
  checkCudaErrors(cudaFree(n_cell_z));
  checkCudaErrors(cudaFree(rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();

  return 0;
}
