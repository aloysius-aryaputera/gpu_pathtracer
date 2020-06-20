#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <thrust/sort.h>

#include "external/libjpeg_cpp/jpeg.h"

#include "input/input_param.h"
#include "input/read_file_util.h"
#include "input/read_image_util.h"
#include "lib/world.h"
#include "model/bvh/bvh.h"
#include "model/bvh/bvh_building.h"
#include "model/bvh/bvh_building_pts.h"
#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/primitive.h"
#include "model/geometry/triangle.h"
#include "model/geometry/triangle_operations.h"
#include "model/grid/bounding_box.h"
#include "model/grid/bounding_box_operations.h"
#include "model/material/material.h"
#include "model/object/object.h"
#include "model/object/object_operations.h"
#include "model/point/point.h"
#include "model/point/point_operations.h"
#include "model/ray/ray.h"
#include "model/vector_and_matrix/mat3.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "render/pathtracing_target_geom_operations.h"
#include "util/general.h"
#include "util/image_util.h"
#include "util/string_util.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(
  cudaError_t result, char const *const func, const char *const file, 
	int const line
) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
    file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

int main(int argc, char **argv) {
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024ULL*1024ULL*1024ULL*4ULL);

  std::string process;
  time_t my_time = time(NULL);
  clock_t first_start, start;

  process = "Rendering project";
  print_start_process(process, first_start);

  std::string master_file_path = argv[1];
  std::string image_output_path = argv[2];

  InputParam input_param = InputParam();
  input_param.extract_parameters(master_file_path);

  std::string input_folder_path = input_param.input_folder_path;
  std::string obj_filename = input_param.obj_filename;
  std::string texture_bg_path = input_param.texture_bg_path;

  int im_width = input_param.image_width;
  int im_height = input_param.image_height;
  int pathtracing_sample_size = input_param.pathtracing_sample_size;
  int pathtracing_level = input_param.pathtracing_level;
  float eye_x = input_param.eye_x;
  float eye_y = input_param.eye_y;
  float eye_z = input_param.eye_z;
  float center_x = input_param.center_x;
  float center_y = input_param.center_y;
  float center_z = input_param.center_z;
  float up_x = input_param.up_x;
  float up_y = input_param.up_y;
  float up_z = input_param.up_z;
  float fovy = input_param.fovy;
  float aperture = input_param.aperture;
  float focus_dist = input_param.focus_dist;

  float sky_emission_r = input_param.sky_emission_r;
  float sky_emission_g = input_param.sky_emission_g;
  float sky_emission_b = input_param.sky_emission_b;

  int sss_pts_per_object = input_param.sss_pts_per_object;
  float hittable_pdf_weight = input_param.hittable_pdf_weight;

  int tx = 8, ty = 8;

  BoundingBox **world_bounding_box;
  Primitive **my_geom;
  Primitive **target_geom_list;
  Object **my_objects;
  unsigned int *morton_code_list;
  Material **my_material;
  Camera **my_camera;
  Point **sss_pts;
  vec3 *image_output;

  int num_pixels = im_width * im_height;
  int max_num_materials = 100;
  int num_objects, num_vertices, num_faces, num_vt, num_vn;
  int *num_sss_objects, *num_target_geom;
  size_t image_size = num_pixels * sizeof(vec3);
  curandState *rand_state_sss, *rand_state_image;
  size_t rand_state_image_size = num_pixels * sizeof(curandState);

  bool *sss_object_marker_array;
  int *pt_offset_array, *num_pt_array;

  float *ka_x, *ka_y, *ka_z, *kd_x, *kd_y, *kd_z;
  float *ks_x, *ks_y, *ks_z, *ke_x, *ke_y, *ke_z, *n_s, *n_i, *t_r;
  float *tf_x, *tf_y, *tf_z;
  float *path_length;
  float *material_image_r, *material_image_g, *material_image_b;
	float *bm;
  int *num_materials;
  int *material_image_height_diffuse, *material_image_width_diffuse, \
    *material_image_offset_diffuse, *material_priority;
  int *material_image_height_specular, *material_image_width_specular, \
    *material_image_offset_specular;
  int *material_image_height_n_s, *material_image_width_n_s, \
    *material_image_offset_n_s;
  int *material_image_height_emission, *material_image_width_emission, \
    *material_image_offset_emission;
  int *material_image_height_bump, *material_image_width_bump, \
    *material_image_offset_bump;
  vec3 *tangent, *bitangent;

  float *bg_texture_r, *bg_texture_g, *bg_texture_b;
  int bg_height, bg_width;

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

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_r, texture_length * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_g, texture_length * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_b, texture_length * sizeof(float)));

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
    num_objects, num_vertices, num_vt, num_vn, num_faces
  );
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&ka_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ka_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ka_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&kd_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&kd_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&kd_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&ks_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ks_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ks_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&ke_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ke_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&ke_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&tf_x, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&tf_y, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&tf_z, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&path_length, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&bm, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&t_r, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&n_s, max_num_materials * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&n_i, max_num_materials * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_diffuse,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_diffuse,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_diffuse,
    max_num_materials * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_priority, max_num_materials * sizeof(int)));

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

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_emission,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_emission,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_emission,
    max_num_materials * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_height_bump,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_width_bump,
    max_num_materials * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_image_offset_bump,
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
    path_length,
    t_r, n_s, n_i, bm,
    material_priority,
    material_image_height_diffuse, material_image_width_diffuse,
    material_image_offset_diffuse,
    material_image_height_specular, material_image_width_specular,
    material_image_offset_specular,
    material_image_height_emission, material_image_width_emission,
    material_image_offset_emission,
    material_image_height_n_s, material_image_width_n_s,
    material_image_offset_n_s,
    material_image_height_bump, material_image_width_bump,
    material_image_offset_bump,
    num_materials,
    material_name
  );
  print_end_process(process, start);

  float *x, *y, *z, *x_norm, *y_norm, *z_norm, *x_tex, *y_tex;
  int *point_1_idx, *point_2_idx, *point_3_idx, \
    *norm_1_idx, *norm_2_idx, *norm_3_idx, \
    *tex_1_idx, *tex_2_idx, *tex_3_idx;
  int *num_triangles, *material_idx;
  int *object_num_primitives, *object_primitive_offset_idx;
  int *triangle_object_idx;
  float *triangle_area, *accumulated_triangle_area;

  checkCudaErrors(cudaMallocManaged(
    (void **)&num_triangles, sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&object_num_primitives, num_objects * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&object_primitive_offset_idx, num_objects * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&triangle_object_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&material_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&triangle_area, num_faces * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&accumulated_triangle_area, num_faces * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&x, max(1, num_vertices) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&y, max(1, num_vertices) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&z, max(1, num_vertices) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&x_norm, max(1, num_vn) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&y_norm, max(1, num_vn) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&z_norm, max(1, num_vn) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&x_tex, max(1, num_vt) * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&y_tex, max(1, num_vt) * sizeof(float)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&point_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&point_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&point_3_idx, num_faces * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&norm_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&norm_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&norm_3_idx, num_faces * sizeof(int)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&tex_1_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&tex_2_idx, num_faces * sizeof(int)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&tex_3_idx, num_faces * sizeof(int)));

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
    num_materials,
    triangle_object_idx,
    object_num_primitives,
    object_primitive_offset_idx
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
    path_length,
    t_r, n_s, n_i, bm,
    material_priority,
    material_image_height_diffuse,
    material_image_width_diffuse,
    material_image_offset_diffuse,
    material_image_height_specular,
    material_image_width_specular,
    material_image_offset_specular,
    material_image_height_emission,
    material_image_width_emission,
    material_image_offset_emission,
    material_image_height_n_s,
    material_image_width_n_s,
    material_image_offset_n_s,
    material_image_height_bump,
    material_image_width_bump,
    material_image_offset_bump,
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
    (void **)&my_objects, num_objects * sizeof(Object *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&my_geom, num_triangles[0] * sizeof(Primitive *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&morton_code_list, num_triangles[0] * sizeof(unsigned int)));

  start = clock();
  process = "Creating the objects";
  print_start_process(process, start);
  create_objects<<<1, num_objects>>>(
    my_objects, object_num_primitives, object_primitive_offset_idx,
    triangle_area, accumulated_triangle_area, num_objects
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&sss_object_marker_array, num_objects * sizeof(bool)));

  start = clock();
  process = "Creating the world";
  print_start_process(process, start);
  dim3 blocks_world(num_triangles[0] / 1 + 1);
  dim3 threads_world(1);
  create_world<<<blocks_world, threads_world>>>(
    my_geom,
    triangle_area,
    my_objects, triangle_object_idx,
    my_material,
    x, y, z,
    x_norm, y_norm, z_norm,
    x_tex, y_tex,
    point_1_idx, point_2_idx, point_3_idx,
    norm_1_idx, norm_2_idx, norm_3_idx,
    tex_1_idx, tex_2_idx, tex_3_idx,
    material_idx,
    num_triangles,
    sss_object_marker_array,
    sss_pts_per_object
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&tangent, num_vertices * sizeof(vec3)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&bitangent, num_vertices * sizeof(vec3)));

  start = clock();
  process = "Summing up tangents and bitangents";
  print_start_process(process, start);
  sum_up_tangent_and_bitangent<<<1, 1>>>(
    tangent, bitangent, my_geom, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Assigning tangents";
  print_start_process(process, start);
  assign_tangent<<<num_triangles[0], 1>>>(
    tangent, my_geom, num_triangles[0]
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

  checkCudaErrors(cudaMallocManaged((void **)&num_sss_objects, sizeof(int)));
  checkCudaErrors(
    cudaMallocManaged((void **)&pt_offset_array, num_objects * sizeof(int)));
  checkCudaErrors(
    cudaMallocManaged((void **)&num_pt_array, num_objects * sizeof(int)));

  start = clock();
  process = "Computing the number of SSS objects";
  print_start_process(process, start);
  compute_num_sss_objects<<<1, 1>>>(
    num_sss_objects, my_objects, pt_offset_array, num_pt_array,
    num_objects, sss_pts_per_object
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  int num_sss_points = sss_pts_per_object * num_sss_objects[0];

  checkCudaErrors(cudaMallocManaged
    ((void **)&sss_pts, max(1, num_sss_points) * sizeof(Point*)));

  start = clock();
  process = "Allocating " + std::to_string(num_sss_points) + \
    " points for SSS objects";
  print_start_process(process, start);
  allocate_pts_sss<<<1, num_objects>>>(
    my_objects, sss_pts, pt_offset_array, num_objects);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&world_bounding_box, sizeof(BoundingBox *)));

  checkCudaErrors(cudaMallocManaged(
    (void **)&rand_state_sss, max(1, num_sss_points) * sizeof(curandState)));

  start = clock();
  process = "Generating curand state for SSS points sampling";
  print_start_process(process, start);
  init_curand_state<<<max(1, num_sss_points), 1>>>(
    num_sss_points, rand_state_sss);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  process = "Creating SSS points samplings";
  start = clock();
  print_start_process(process, start);
  for (int i = 0; i < num_objects; i++) {
    create_sss_pts<<<sss_pts_per_object, 1>>>(
      my_objects, my_geom, sss_pts, pt_offset_array, rand_state_sss,
      i, sss_pts_per_object
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  print_end_process(process, start);

  start = clock();
  process = "Computing the object boundaries";
  print_start_process(process, start);
  compute_object_boundaries_batch<<<num_objects, 1>>>(
    my_objects, num_objects
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Computing the morton code of every point bounding box";
  print_start_process(process, start);
  compute_pts_morton_code_batch<<<blocks_world, threads_world>>>(
    my_objects, sss_pts, num_sss_points
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  auto sort_points = []  __device__ (Point* pt_1, Point* pt_2) {
    return pt_1 -> bounding_box -> morton_code < \
      pt_2 -> bounding_box -> morton_code;
  };

  process = "Sorting the points based on morton code";
  start = clock();
  print_start_process(process, start);
  for (int i = 0; i < num_objects; i++) {
    thrust::stable_sort(
      thrust::device, sss_pts + pt_offset_array[i],
      sss_pts + pt_offset_array[i] + num_pt_array[i],
      sort_points);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  print_end_process(process, start);

  start = clock();
  process = "Computing sss points offset list";
  print_start_process(process, start);
  compute_sss_pts_offset<<<1, 1>>>(my_objects, num_objects);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  Node** sss_pts_node_list, **sss_pts_leaf_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&sss_pts_node_list,
    max(1, (num_sss_points - num_sss_objects[0])) * sizeof(Node *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&sss_pts_leaf_list,
    max(1, num_sss_points) * sizeof(Node *)));

  process = "Building sss points leaves";
  start = clock();
  print_start_process(process, start);
  for (int i = 0; i < num_objects; i++) {
    build_sss_pts_leaf_list<<<max(1, num_sss_points), 1>>>(
      sss_pts_leaf_list, sss_pts, my_objects, i, pt_offset_array
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  print_end_process(process, start);

  checkCudaErrors(cudaFree(pt_offset_array));
  checkCudaErrors(cudaFree(num_pt_array));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  process = "Building sss points nodes";
  start = clock();
  print_start_process(process, start);
  for (int i = 0; i < num_objects; i++) {
    build_sss_pts_node_list<<<max(1, num_sss_points), 1>>>(
      sss_pts_node_list, my_objects, i
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  print_end_process(process, start);

  unsigned int *sss_morton_code_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&sss_morton_code_list,
    max(1, num_sss_points) * sizeof(unsigned int)));

  start = clock();
  process = "Extracting the morton codes of the SSS points";
  print_start_process(process, start);
  extract_sss_morton_code_list<<<max(1, num_sss_points), 1>>>(
    sss_pts, sss_morton_code_list, num_sss_points
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  process = "Setting the sss nodes relationship";
  start = clock();
  print_start_process(process, start);
  for (int i = 0; i < num_objects; i++) {
    set_pts_sss_node_relationship<<<max(1, num_sss_points), 1>>>(
      sss_pts_node_list, sss_pts_leaf_list, sss_morton_code_list, my_objects, i
    );
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }
  print_end_process(process, start);

  checkCudaErrors(cudaFree(sss_morton_code_list));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  start = clock();
  process = "Compute pts node bounding boxes";
  print_start_process(process, start);
  compute_node_bounding_boxes<<<max(1, num_sss_points), 1>>>(
    sss_pts_leaf_list, sss_pts_node_list, num_sss_points
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Computing the world bounding box";
  print_start_process(process, start);
  compute_world_bounding_box<<<1, 1>>>(
    world_bounding_box, my_geom, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Computing the morton code of every geometry bounding box";
  print_start_process(process, start);
  compute_morton_code_batch<<<blocks_world, threads_world>>>(
    my_geom, world_bounding_box, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  auto sort_geom = []  __device__ (Primitive* obj_1, Primitive* obj_2) {
    return obj_1 -> get_bounding_box() -> morton_code < \
      obj_2 -> get_bounding_box() -> morton_code;
  };

  start = clock();
  process = "Sorting the objects based on morton code";
  print_start_process(process, start);
  thrust::stable_sort(
    thrust::device, my_geom, my_geom + num_triangles[0], sort_geom);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  Node** node_list, **leaf_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&node_list, (num_triangles[0] - 1) * sizeof(Node *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&leaf_list, num_triangles[0] * sizeof(Node *)));

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

  start = clock();
  process = "Setting node relationship";
  print_start_process(process, start);
  set_node_relationship<<<blocks_world, threads_world>>>(
    node_list, leaf_list, morton_code_list, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaFree(morton_code_list));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  start = clock();
  process = "Compute node bounding boxes";
  print_start_process(process, start);
  compute_node_bounding_boxes<<<blocks_world, threads_world>>>(
    leaf_list, node_list, num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Check";
  print_start_process(process, start);
  check<<<blocks_world, threads_world>>>(
    leaf_list,  node_list,  num_triangles[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  vec3 sky_emission = vec3(sky_emission_r, sky_emission_g, sky_emission_b);

  checkCudaErrors(cudaMallocManaged((void **)&num_target_geom, sizeof(int)));

  start = clock();
  process = "Computing the number of target geometries";
  print_start_process(process, start);
  compute_num_target_geom<<<1, 1>>>(
    my_geom, num_triangles[0], num_target_geom
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaMallocManaged(
    (void **)&target_geom_list, num_target_geom[0] * sizeof(Primitive *)));

  start = clock();
  process = "Collecting target geometries";
  print_start_process(process, start);
  collect_target_geom<<<1, 1>>>(
    my_geom, num_triangles[0], target_geom_list
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Sorting the target geometries based on morton code";
  print_start_process(process, start);
  thrust::stable_sort(
    thrust::device, target_geom_list, target_geom_list + num_target_geom[0],
    sort_geom);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  Node** target_node_list, **target_leaf_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&target_node_list, (num_target_geom[0] - 1) * sizeof(Node *)));
  checkCudaErrors(cudaMallocManaged(
    (void **)&target_leaf_list, num_target_geom[0] * sizeof(Node *)));

  start = clock();
  process = "Building target leaves";
  print_start_process(process, start);
  build_leaf_list<<<max(1, num_target_geom[0]), 1>>>(
    target_leaf_list, target_geom_list, num_target_geom[0], true
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Building target nodes";
  print_start_process(process, start);
  build_node_list<<<max(1, num_target_geom[0]), 1>>>(
    target_node_list, num_target_geom[0], true
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  unsigned int *target_morton_code_list;
  checkCudaErrors(cudaMallocManaged(
    (void **)&target_morton_code_list,
    max(1, num_target_geom[0]) * sizeof(unsigned int)));

  start = clock();
  process = "Extracting target morton codes";
  print_start_process(process, start);
  extract_morton_code_list<<<max(1, num_target_geom[0]), 1>>>(
    target_geom_list, target_morton_code_list, num_target_geom[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Setting target node relationship";
  print_start_process(process, start);
  set_node_relationship<<<max(1, num_target_geom[0]), 1>>>(
    target_node_list, target_leaf_list, target_morton_code_list,
    num_target_geom[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  checkCudaErrors(cudaFree(target_morton_code_list));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  start = clock();
  process = "Compute target node bounding boxes";
  print_start_process(process, start);
  compute_node_bounding_boxes<<<max(1, num_target_geom[0]), 1>>>(
    target_leaf_list, target_node_list, num_target_geom[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Compute target node bounding cones";
  print_start_process(process, start);
  compute_node_bounding_cones<<<max(1, num_target_geom[0]), 1>>>(
    target_leaf_list, target_node_list, num_target_geom[0]
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Doing first pass for SSS objects";
  print_start_process(process, start);
  do_sss_first_pass<<<max(1, num_sss_points), 1>>>(
    sss_pts, num_sss_points,
    pathtracing_sample_size,
    pathtracing_level, sky_emission,
    bg_height, bg_width,
    bg_texture_r, bg_texture_g, bg_texture_b, node_list,
    rand_state_sss, target_geom_list,
    target_node_list,
    num_target_geom[0],
    hittable_pdf_weight
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

	checkCudaErrors(cudaMallocManaged((void **)&image_output, image_size));
  dim3 blocks(im_width / tx + 1, im_height / ty + 1);
  dim3 threads(tx, ty);

  start = clock();
  process = "Clearing image";
  print_start_process(process, start);
  clear_image<<<blocks, threads>>>(image_output, im_width, im_height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Creating point image";
  print_start_process(process, start);
  create_point_image<<<num_sss_points / tx + 1, tx>>>(
    image_output, my_camera, sss_pts, num_sss_points
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Saving pts image";
  print_start_process(process, start);
  save_image(
    image_output, im_width, im_height, image_output_path + "_pts.ppm");
  print_end_process(process, start);
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(
    cudaMallocManaged((void **)&rand_state_image, rand_state_image_size));

  start = clock();
  process = "Generating curand state for rendering";
  print_start_process(process, start);
  init_curand_state<<<num_pixels / 8 + 1, 8>>>(num_pixels, rand_state_image);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Rendering";
  print_start_process(process, start);
  render<<<blocks, threads>>>(
    image_output, my_camera, rand_state_image, pathtracing_sample_size,
    pathtracing_level, sky_emission, bg_height, bg_width,
    bg_texture_r, bg_texture_g, bg_texture_b, node_list, my_objects,
    sss_pts_node_list,
    target_node_list,
    target_geom_list, num_target_geom[0],
    hittable_pdf_weight
  );
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  print_end_process(process, start);

  start = clock();
  process = "Saving image";
  print_start_process(process, start);
  save_image(image_output, im_width, im_height, image_output_path + ".ppm");
  print_end_process(process, start);
  checkCudaErrors(cudaDeviceSynchronize());

  start = clock();
  process = "Cleaning";
  print_start_process(process, start);
  // free_world<<<1,1>>>(my_scene, my_grid, my_geom, my_camera, max_num_faces);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(my_camera));
  checkCudaErrors(cudaFree(my_geom));
  checkCudaErrors(cudaFree(my_material));
  checkCudaErrors(cudaFree(num_triangles));
  checkCudaErrors(cudaFree(rand_state_image));
  checkCudaErrors(cudaFree(rand_state_sss));
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
