#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <thrust/sort.h>

#include "external/libjpeg_cpp/jpeg.h"

#include "model/bvh/bvh_build.h"
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
#include "util/read_image_util.h"
#include "util/string_util.h"
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
  int im_width, int im_height, curandState *rand_state
) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i = threadIdx.y + blockIdx.y * blockDim.y;
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

__global__ void test(Primitive **object_array, int n) {
  printf("%u\n", object_array[0] -> get_bounding_box() -> morton_code);
  printf("%d\n", object_array[0] -> get_bounding_box() -> morton_code < object_array[1] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[1] -> get_bounding_box() -> morton_code);
  printf("%d\n", object_array[1] -> get_bounding_box() -> morton_code < object_array[2] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[2] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[n / 2] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[n - 5] -> get_bounding_box() -> morton_code);
  printf("%d\n", object_array[n - 5] -> get_bounding_box() -> morton_code < object_array[n - 3] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[n - 3] -> get_bounding_box() -> morton_code);
  printf("%d\n", object_array[n - 3] -> get_bounding_box() -> morton_code < object_array[n - 1] -> get_bounding_box() -> morton_code);
  printf("%u\n", object_array[n - 1] -> get_bounding_box() -> morton_code);
}

__global__ void my_clz(unsigned int x) {
  int result = __clz(x);
  printf("clz = %d\n", result);
  printf("clz_2 = %d\n", __clz(41));
  printf("******************************************\n");
}

int main(int argc, char **argv) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ULL*1024ULL*1024ULL*4ULL);

  std::string process;
  time_t my_time = time(NULL);
  clock_t first_start, start;

  process = "Rendering project";
  print_start_process(process, first_start);

  std::string input_folder_path = argv[1];
  std::string obj_filename = argv[2];
  std::string texture_bg_path = argv[3];
  std::string image_output_path = argv[4];

  int im_width = std::stoi(argv[5]);
  int im_height = std::stoi(argv[6]);
  int pathtracing_sample_size = std::stoi(argv[7]);
  int pathtracing_level = std::stoi(argv[8]);
  float eye_x = std::stof(argv[9]);
  float eye_y = std::stof(argv[10]);
  float eye_z = std::stof(argv[11]);
  float center_x = std::stof(argv[12]);
  float center_y = std::stof(argv[13]);
  float center_z = std::stof(argv[14]);
  float up_x = std::stof(argv[15]);
  float up_y = std::stof(argv[16]);
  float up_z = std::stof(argv[17]);
  float fovy = std::stof(argv[18]);
  float aperture = std::stof(argv[19]);
  float focus_dist = std::stof(argv[20]);

  float sky_emission_r = std::stof(argv[21]);
  float sky_emission_g = std::stof(argv[22]);
  float sky_emission_b = std::stof(argv[23]);

  int *n_cell_x, *n_cell_y, *n_cell_z;
  int max_n_cell_x = 120, max_n_cell_y = 120, max_n_cell_z = 120;
  int tx = 8, ty = 8, tx2 = 8, ty2 = 8, tz2 = 8, max_num_objects_per_cell = 1000;

  Scene** my_scene;
  Grid** my_grid;
  Cell** my_cell;
  Primitive **my_geom, **my_cell_geom;
  unsigned int *morton_code_list;
  Material **my_material;
  Camera **my_camera;
  vec3 *image_output;

  int num_pixels = im_width * im_height;
  int max_num_materials = 100;
  int num_vertices, num_faces, num_vt, num_vn;
  size_t image_size = num_pixels * sizeof(vec3);
  curandState *rand_state;
  size_t rand_state_size = num_pixels * sizeof(curandState);

  float *ka_x, *ka_y, *ka_z, *kd_x, *kd_y, *kd_z;
  float *ks_x, *ks_y, *ks_z, *ke_x, *ke_y, *ke_z, *n_s, *n_i, *t_r;
  float *tf_x, *tf_y, *tf_z;
  float *material_image_r, *material_image_g, *material_image_b;
  int *num_materials;
  int *material_image_height_diffuse, *material_image_width_diffuse, \
    *material_image_offset_diffuse;
  int *material_image_height_specular, *material_image_width_specular, \
    *material_image_offset_specular;
  int *material_image_height_n_s, *material_image_width_n_s, \
    *material_image_offset_n_s;

  float *bg_texture_r, *bg_texture_g, *bg_texture_b;
  int bg_height, bg_width;

  // int a1 = std::stoi(argv[24]);
  // int a2 = std::stoi(argv[25]);
  // printf("a1 = %d\n", a1);
  // printf("a2 = %d\n", a2);
  // my_clz<<<1, 1>>>((unsigned int)(a1) ^ (unsigned int)(a2));
  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaDeviceSynchronize());

  // printf("================================================================\n");

  ///////////////////////////////////////////////////////////////////////////
  // For offline testing
  ///////////////////////////////////////////////////////////////////////////
  // float ka_x[100], ka_y[100], ka_z[100], kd_x[100], kd_y[100], kd_z[100];
  // float ks_x[100], ks_y[100], ks_z[100], ke_x[100], ke_y[100], ke_z[100];
  // float material_image_r[1000], material_image_g[1000], material_image_b[1000];
  // int num_materials[1], material_image_height[100], material_image_width[100], material_image_offset[100];
  // int len_texture[1];
  ///////////////////////////////////////////////////////////////////////////

  start = clock();
  process = "Extracting background texture";
  print_start_process(process, start);
  extract_single_image_requirement(
    input_folder_path, texture_bg_path, bg_height, bg_width
  );

  checkCudaErrors(cudaMallocManaged(
    (void **)&bg_texture_r, bg_height * bg_width * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&bg_texture_g, bg_height * bg_width * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&bg_texture_b, bg_height * bg_width * sizeof(float)));

  int next_idx = 0;
  extract_single_image(
    input_folder_path, texture_bg_path, bg_texture_r, bg_texture_g,
    bg_texture_b, next_idx
  );
  print_end_process(process, start);

  std::vector <std::string> material_file_name_array, material_name;

  checkCudaErrors(cudaMallocManaged((void **)&num_materials, sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&ka_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ka_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ka_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&kd_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&kd_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&kd_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&ks_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ks_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ks_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&ke_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ke_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&ke_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&tf_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&tf_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&tf_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&t_r, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&n_s, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&n_i, max_num_materials * sizeof(float)));

  start = clock();
  process = "Extracting material file names";
  print_start_process(process, start);
  extract_material_file_names(
    input_folder_path,
    obj_filename,
    material_file_name_array
  );
  print_end_process(process, start);

  std::vector <std::string> texture_file_name_array;
  std::vector <int> texture_offset_array, texture_height_array, \
    texture_width_array;
  long int texture_length = 0;

  start = clock();
  process = "Extracting texture resource requirements";
  print_start_process(process, start);
  extract_image_resource_requirement(
    input_folder_path,
    material_file_name_array,
    texture_file_name_array,
    texture_offset_array,
    texture_height_array,
    texture_width_array,
    texture_length
  );
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged((void **)&material_image_r, texture_length * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&material_image_g, texture_length * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&material_image_b, texture_length * sizeof(float)));

  start = clock();
  process = "Extracting textures";
  print_start_process(process, start);
  extract_textures(
    input_folder_path,
    texture_file_name_array,
    material_image_r,
    material_image_g,
    material_image_b
  );
  print_end_process(process, start);

  start = clock();
  process = "Extracting the number of the elements";
  print_start_process(process, start);
  extract_num_elements(
    input_folder_path, obj_filename,
    num_vertices, num_vt, num_vn, num_faces
  );
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_diffuse, max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_diffuse, max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_diffuse, max_num_materials * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_specular,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_specular,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_specular,
    max_num_materials * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_n_s,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_n_s,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_n_s,
    max_num_materials * sizeof(int)));

  start = clock();
  process = "Extracting material data";
  print_start_process(process, start);
  extract_material_data(
    input_folder_path,
    material_file_name_array,
    texture_file_name_array,
    texture_offset_array,
    texture_height_array,
    texture_width_array,
    ka_x, ka_y, ka_z,
    kd_x, kd_y, kd_z,
    ks_x, ks_y, ks_z,
    ke_x, ke_y, ke_z,
    tf_x, tf_y, tf_z,
    t_r, n_s, n_i,
    material_image_height_diffuse, material_image_width_diffuse,
    material_image_offset_diffuse,
    material_image_height_specular, material_image_width_specular,
    material_image_offset_specular,
    material_image_height_n_s, material_image_width_n_s,
    material_image_offset_n_s,
    num_materials,
    material_name
  );
  print_end_process(process, start);

  float *x, *y, *z, *x_norm, *y_norm, *z_norm, *x_tex, *y_tex;
  int *point_1_idx, *point_2_idx, *point_3_idx, \
    *norm_1_idx, *norm_2_idx, *norm_3_idx, \
    *tex_1_idx, *tex_2_idx, *tex_3_idx;
  int *num_triangles, *material_idx;

  checkCudaErrors(cudaMallocManaged((void **)&num_triangles, sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&material_idx, num_faces * sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&x, max(1, num_vertices) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y, max(1, num_vertices) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&z, max(1, num_vertices) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&x_norm, max(1, num_vn) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y_norm, max(1, num_vn) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&z_norm, max(1, num_vn) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&x_tex, max(1, num_vt) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged((void **)&y_tex, max(1, num_vt) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged((void **)&point_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&point_3_idx, num_faces * sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&norm_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&norm_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&norm_3_idx, num_faces * sizeof(int)));

  checkCudaErrors(cudaMallocManaged((void **)&tex_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&tex_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&tex_3_idx, num_faces * sizeof(int)));

  start = clock();
  process = "Reading OBJ file";
  print_start_process(process, start);
  extract_triangle_data(
    input_folder_path,
    obj_filename,
    x, y, z,
    x_norm, y_norm, z_norm,
    x_tex, y_tex,
    point_1_idx, point_2_idx, point_3_idx,
    norm_1_idx, norm_2_idx, norm_3_idx,
    tex_1_idx, tex_2_idx, tex_3_idx,
    material_name,
    material_idx,
    num_triangles,
    num_materials
  );
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged((void **)&my_camera, sizeof(Camera *)));

  start = clock();
  process = "Creating the camera";
  print_start_process(process, start);
  create_camera<<<1, 1>>>(
    my_camera,
    eye_x, eye_y, eye_z,
    center_x, center_y, center_z,
    up_x, up_y, up_z, fovy,
    im_width, im_height,
    aperture, focus_dist
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&my_material, max_num_materials * sizeof(Material *)));

  start = clock();
  process = "Creating the materials";
  print_start_process(process, start);
  create_material<<<1, num_materials[0]>>>(
    my_material,
    ka_x, ka_y, ka_z,
    kd_x, kd_y, kd_z,
    ks_x, ks_y, ks_z,
    ke_x, ke_y, ke_z,
    tf_x, tf_y, tf_z,
    t_r, n_s, n_i,
    material_image_height_diffuse,
    material_image_width_diffuse,
    material_image_offset_diffuse,
    material_image_height_specular,
    material_image_width_specular,
    material_image_offset_specular,
    material_image_height_n_s,
    material_image_width_n_s,
    material_image_offset_n_s,
    material_image_r,
    material_image_g,
    material_image_b,
    num_materials
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaFree(ka_x));
  checkCudaErrors(cudaFree(ka_y));
  checkCudaErrors(cudaFree(ka_z));
  checkCudaErrors(cudaFree(kd_x));
  checkCudaErrors(cudaFree(kd_y));
  checkCudaErrors(cudaFree(kd_z));
  checkCudaErrors(cudaFree(ks_x));
  checkCudaErrors(cudaFree(ks_y));
  checkCudaErrors(cudaFree(ks_z));
  checkCudaErrors(cudaFree(ke_x));
  checkCudaErrors(cudaFree(ke_y));
  checkCudaErrors(cudaFree(ke_z));
  checkCudaErrors(cudaFree(tf_x));
  checkCudaErrors(cudaFree(tf_y));
  checkCudaErrors(cudaFree(tf_z));
  checkCudaErrors(cudaFree(n_s));
  checkCudaErrors(cudaFree(material_image_height_diffuse));
  checkCudaErrors(cudaFree(material_image_width_diffuse));
  checkCudaErrors(cudaFree(material_image_offset_diffuse));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocManaged(
    (void **)&my_geom, num_triangles[0] * sizeof(Primitive *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&morton_code_list, num_triangles[0] * sizeof(unsigned int)));

  start = clock();
  process = "Creating the world";
  print_start_process(process, start);
  dim3 blocks_world(num_triangles[0] / 256 + 1);
  dim3 threads_world(256);
  create_world<<<blocks_world, threads_world>>>(
    my_geom, my_material,
    x, y, z,
    x_norm, y_norm, z_norm,
    x_tex, y_tex,
    point_1_idx, point_2_idx, point_3_idx,
    norm_1_idx, norm_2_idx, norm_3_idx,
    tex_1_idx, tex_2_idx, tex_3_idx,
    material_idx,
    num_triangles
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaFree(x));
  checkCudaErrors(cudaFree(y));
  checkCudaErrors(cudaFree(z));
  checkCudaErrors(cudaFree(x_norm));
  checkCudaErrors(cudaFree(y_norm));
  checkCudaErrors(cudaFree(z_norm));
  checkCudaErrors(cudaFree(point_1_idx));
  checkCudaErrors(cudaFree(point_2_idx));
  checkCudaErrors(cudaFree(point_3_idx));
  checkCudaErrors(cudaFree(norm_1_idx));
  checkCudaErrors(cudaFree(norm_2_idx));
  checkCudaErrors(cudaFree(norm_3_idx));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMallocManaged((void **)&my_grid, sizeof(Grid *)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_x, sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_y, sizeof(int)));
  checkCudaErrors(cudaMallocManaged((void **)&n_cell_z, sizeof(int)));

  start = clock();
  process = "Preparing the grid";
  print_start_process(process, start);
  prepare_grid<<<1, 1>>>(
    my_camera, my_geom, num_triangles,
    n_cell_x, n_cell_y, n_cell_z,
    max_n_cell_x, max_n_cell_y, max_n_cell_z
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(
    cudaMallocManaged(
      (void **)&my_cell,
      n_cell_x[0] * n_cell_y[0] * n_cell_z[0] * sizeof(Cell*)));

  start = clock();
  process = "Creating the grid";
  print_start_process(process, start);
  create_grid<<<1, 1>>>(
    my_camera, my_grid, my_geom, num_triangles, my_cell, n_cell_x, n_cell_y,
    n_cell_z, max_n_cell_x, max_n_cell_y, max_n_cell_z, max_num_objects_per_cell
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Computing the morton code of every bounding box";
  print_start_process(process, start);
  compute_morton_code_batch<<<blocks_world, threads_world>>>(
    my_geom, my_grid, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  auto ff = []  __device__ (Primitive* obj_1, Primitive* obj_2) {
    return obj_1 -> get_bounding_box() -> morton_code < \
      obj_2 -> get_bounding_box() -> morton_code;
  };

  start = clock();
  process = "Sorting the objects based on morton code";
  print_start_process(process, start);
  thrust::stable_sort(thrust::device, my_geom, my_geom + num_triangles[0], ff);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  Node** node_list;
  Leaf** leaf_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&node_list, (num_triangles[0] - 1) * sizeof(Node *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&leaf_list, num_triangles[0] * sizeof(Leaf *)));

  start = clock();
  process = "Building leaves";
  print_start_process(process, start);
  build_leaf_list<<<blocks_world, threads_world>>>(
    leaf_list, my_geom, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Building nodes";
  print_start_process(process, start);
  build_node_list<<<blocks_world, threads_world>>>(
    node_list, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Extracting morton codes";
  print_start_process(process, start);
  extract_morton_code_list<<<blocks_world, threads_world>>>(
    my_geom, morton_code_list, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  // start = clock();
  // process = "Set node relationship";
  // print_start_process(process, start);
  // set_node_relationship<<<blocks_world, threads_world>>>(
  //   node_list, leaf_list, morton_code_list, num_triangles[0]
  // );
  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaDeviceSynchronize());
  // print_end_process(process, start);
  //
  // start = clock();
  // process = "Compute node bounding boxes";
  // print_start_process(process, start);
  // compute_node_bounding_boxes<<<blocks_world, threads_world>>>(
  //   leaf_list, node_list, num_triangles[0]
  // );
  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaDeviceSynchronize());
  // print_end_process(process, start);

  size_t cell_geom_size = max_num_objects_per_cell * \
    n_cell_x[0] * n_cell_y[0] * n_cell_z[0] * sizeof(Primitive*);
  checkCudaErrors(cudaMallocManaged((void **)&my_cell_geom, cell_geom_size));
  dim3 blocks2(n_cell_x[0] / tx2 + 1, n_cell_y[0] / ty2 + 1, n_cell_z[0] / tz2 + 1);
  dim3 threads2(tx2, ty2, tz2);

  start = clock();
  process = "Building cell array";
  print_start_process(process, start);
  build_cell_array<<<blocks2, threads2>>>(my_grid, my_cell_geom);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Inserting objects into the grid";
  print_start_process(process, start);
  insert_objects<<<blocks2, threads2>>>(my_grid);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged((void **)&my_scene, sizeof(Scene *)));

  start = clock();
  process = "Creating the scene";
  print_start_process(process, start);
  create_scene<<<1, 1>>>(my_scene, my_camera, my_grid, num_triangles);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  dim3 blocks(im_width / tx + 1, im_height / ty + 1);
  dim3 threads(tx, ty);
  checkCudaErrors(cudaMallocManaged((void **)&rand_state, rand_state_size));

  start = clock();
  process = "Preparing the rendering process";
  print_start_process(process, start);
  render_init<<<blocks, threads>>>(im_width, im_height, rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  vec3 sky_emission = vec3(sky_emission_r, sky_emission_g, sky_emission_b);
  checkCudaErrors(cudaMallocManaged((void **)&image_output, image_size));

  start = clock();
  process = "Rendering";
  print_start_process(process, start);
  render<<<blocks, threads>>>(
    image_output, my_scene, rand_state, pathtracing_sample_size,
    pathtracing_level, sky_emission, bg_height, bg_width,
    bg_texture_r, bg_texture_g, bg_texture_b
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Saving image";
  print_start_process(process, start);
  save_image(image_output, im_width, im_height, image_output_path);
  print_end_process(process, start);
  checkCudaErrors(cudaDeviceSynchronize());

  start = clock();
  process = "Cleaning";
  print_start_process(process, start);
  // free_world<<<1,1>>>(my_scene, my_grid, my_geom, my_camera, max_num_faces);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(my_scene));
  checkCudaErrors(cudaFree(my_grid));
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom));
  checkCudaErrors(cudaFree(my_material));
  checkCudaErrors(cudaFree(num_triangles));
  checkCudaErrors(cudaFree(n_cell_x));
  checkCudaErrors(cudaFree(n_cell_y));
  checkCudaErrors(cudaFree(n_cell_z));
  checkCudaErrors(cudaFree(rand_state));
  checkCudaErrors(cudaFree(material_image_r));
  checkCudaErrors(cudaFree(material_image_g));
  checkCudaErrors(cudaFree(material_image_b));
  checkCudaErrors(cudaFree(bg_texture_r));
  checkCudaErrors(cudaFree(bg_texture_g));
  checkCudaErrors(cudaFree(bg_texture_b));
  checkCudaErrors(cudaFree(image_output));
  print_end_process(process, start);

  print_end_process("Rendering project", first_start);

  cudaDeviceReset();

  return 0;
}
