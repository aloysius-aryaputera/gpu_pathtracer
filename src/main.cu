#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/sphere.h"
#include "model/geometry/triangle.h"
#include "model/grid/cell.h"
#include "model/grid/grid.h"
#include "model/material.h"
#include "model/ray.h"
#include "model/scene.h"
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
  int *point_2_idx, int *point_3_idx, int* num_triangles
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      Material *triangle_material = new Material(
        vec3(.2, .2, .2), vec3(1, .2, .2), vec3(0, 0, 0), vec3(.3, .1, .1)
      );

      for (int idx = 0; idx < num_triangles[0]; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]),
          triangle_material
        );
      }

      triangle_material = new Material(
        vec3(.2, .2, .2), vec3(.2, 1, .2), vec3(0, 0, 0), vec3(.1, .3, .1)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-12, 0, 12), vec3(12, 0, 12), vec3(0, 0, -12),
        triangle_material
      );

    }
}

__global__ void create_world_2(
  Primitive** geom_array, float *x, float *y, float *z, int *point_1_idx,
  int *point_2_idx, int *point_3_idx, int* num_triangles
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      Material *triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .9, .2), vec3(0, 0, 0), vec3(.1, .3, .1)
      );

      for (int idx = 0; idx < num_triangles[0]; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]),
          triangle_material
        );
      }

      // triangle_material = new Material(
      //   vec3(0, 0, 0), vec3(.9, .9, .2), vec3(0, 0, 0), vec3(.3, .3, .1)
      // );
      // *(geom_array + num_triangles[0]++) = new Sphere(
      //   vec3(0, 4, 1), 1, triangle_material
      // );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .2, .9), vec3(40.0, 0.0, 40.0),
        vec3(.3, .1, .3)
      );
      *(geom_array + num_triangles[0]++) = new Sphere(
        vec3(4.1, 1.0, 4.1), 1, triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .9, .9), vec3(0, 0, 0), vec3(.1, .3, .3)
      );
      *(geom_array + num_triangles[0]++) = new Sphere(
        vec3(-4.1, 1.0, 4.1), 1, triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .9, .9), vec3(0, 0, 0), vec3(.3, .3, .3)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-12, 0, 12), vec3(12, 0, 12), vec3(0, 0, -12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .2, .2), vec3(0, 0, 0), vec3(.3, .1, .1)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-12, 12, 12), vec3(-12, 0, 12), vec3(0, 0, -12),
        triangle_material
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-12, 12, 12), vec3(0, 0, -12), vec3(0, 12, -12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .2, .9), vec3(0, 0, 0), vec3(.1, .1, .3)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(12, 12, 12), vec3(0, 0, -12), vec3(12, 0, 12),
        triangle_material
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(12, 12, 12), vec3(0, 12, -12), vec3(0, 0, -12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .9, .9), vec3(0, 0, 0), vec3(.3, .3, .3)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-12, 9, 12), vec3(0, 9, -12), vec3(12, 9, 12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(1, 1, 1), vec3(75.0, 75.0, 75.0), vec3(1, 1, 1)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-5, 8.8, 2), vec3(0, 8.8, 0), vec3(-1, 8.8, 2),
        triangle_material
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(1, 8.8, 2), vec3(0, 8.8, 0), vec3(5, 8.8, 2),
        triangle_material
      );

    }
}

__global__ void set_camera(Camera** camera, int width, int height) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(camera) = new Camera(
      vec3(0, 6, 10), vec3(0, 1, 0), vec3(0, 1, 0), 45, width, height
    );
  }
}

__global__ void create_scene(
  Scene** scene, Camera** camera, Grid** grid, int *num_objects
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(scene) = new Scene(camera[0], grid[0], num_objects[0]);
  }
}

__global__ void create_grid(
  Grid** grid, Primitive** geom_array, int *num_objects, Cell** cell_array,
  int n_cell_x, int n_cell_y, int n_cell_z, int max_num_objects_per_cell
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(grid) = new Grid(
      -20.0f, 20.0f, -20.0f, 20.0f, -20.0f, 20.0f, n_cell_x, n_cell_y,
      n_cell_z, geom_array, num_objects[0], cell_array, max_num_objects_per_cell
    );
  }
}

__global__ void render_init(int im_width, int im_height, curandState *rand_state) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if((j >= im_width) || (i >= im_height)) {
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
  int n_cell_x = 20, n_cell_y = 20, n_cell_z = 20;
  int tx2 = 32, ty2 = 32, max_num_objects_per_cell = 500;

  printf("im_width = %d, im_height = %d\n", im_width, im_height);
  printf("tx = %d, ty = %d\n", tx, ty);

  Scene** my_scene;
  Grid** my_grid;
  Cell** my_cell;
  Primitive **my_geom, **my_cell_geom;
  Camera **my_camera;
  vec3 *fb;
  int num_pixels = im_width * im_height;
  size_t fb_size = num_pixels * sizeof(vec3);
  curandState *rand_state;
  size_t rand_state_size = num_pixels * sizeof(curandState);
  size_t cell_geom_size = max_num_objects_per_cell * n_cell_x * n_cell_y * n_cell_z * sizeof(Primitive*);

  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
  checkCudaErrors(cudaMallocManaged((void **)&rand_state, rand_state_size));

  checkCudaErrors(cudaMallocManaged((void **)&my_camera, sizeof(Camera *)));
  set_camera<<<1, 1>>>(my_camera, im_width, im_height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  float *x, *y, *z;
  int *point_1_idx, *point_2_idx, *point_3_idx;
  int *num_triangles;

  checkCudaErrors(cudaMallocManaged((void **)&num_triangles, sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&x, 9999 * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y, 9999 * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&z, 9999 * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&point_1_idx, 9999 * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_2_idx, 9999 * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_3_idx, 9999 * sizeof(int)));

  extract_triangle_data(
    argv[2], x, y, z, point_1_idx, point_2_idx, point_3_idx, num_triangles
  );

  checkCudaErrors(cudaMallocManaged((void **)&my_geom, 9999 * sizeof(Primitive *)));

  create_world_2<<<1, 1>>>(
    my_geom, x, y, z, point_1_idx, point_2_idx, point_3_idx, num_triangles
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(y));
  checkCudaErrors(cudaFree(z));
  checkCudaErrors(cudaFree(point_1_idx));
  checkCudaErrors(cudaFree(point_2_idx));
  checkCudaErrors(cudaFree(point_3_idx));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocManaged((void **)&my_grid, sizeof(Grid *)));
  checkCudaErrors(cudaMallocManaged((void **)&my_cell, 999 * 999 * 999 * sizeof(Cell *)));
  checkCudaErrors(cudaMallocManaged((void **)&my_cell_geom, cell_geom_size));
  create_grid<<<1, 1>>>(
    my_grid, my_geom, num_triangles, my_cell, n_cell_x, n_cell_y, n_cell_z,
    max_num_objects_per_cell
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  dim3 blocks2(n_cell_x / tx2 + 1, n_cell_y / ty2 + 1);
  dim3 threads2(tx2, ty2);
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
  render_init<<<blocks, threads>>>(im_width, im_height, rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // render<<<blocks, threads>>>(
  //   fb, my_camera, my_geom, num_triangles, rand_state,
  //   std::stoi(argv[7]), std::stoi(argv[8])
  // );
  render<<<blocks, threads>>>(
    fb, my_scene, rand_state, std::stoi(argv[7]), std::stoi(argv[8])
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("Saving image!\n");
  save_image(fb, im_width, im_height, argv[1]);
  printf("Image saved!\n");

  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(my_scene, my_grid, my_geom, my_camera, 9999);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(my_scene));
  checkCudaErrors(cudaFree(my_grid));
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom));
  checkCudaErrors(cudaFree(num_triangles));
  checkCudaErrors(cudaFree(rand_state));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();

  return 0;
}
