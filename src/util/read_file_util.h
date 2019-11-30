#ifndef READ_FILE_UTIL_H
#define READ_FILE_UTIL_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <vector>

#include "../external/libjpeg_cpp/jpeg.h"

#include "../model/geometry/triangle.h"
#include "../model/vector_and_matrix/vec3.h"
#include "string_util.h"

void _extract_single_material_data(
  std::string folder_path, std::string material_filename,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  int *len_texture,
  int *num_materials,
  std::vector <std::string> &material_name
);
std::vector<std::string> split(const std::string& s, char delimiter);
void extract_triangle_data(
  std::string folder_path,
  std::string obj_filename,
  float* x, float* y, float* z,
  float* x_norm, float* y_norm, float* z_norm,
  float* x_tex, float* y_tex,
  int* point_1_idx, int* point_2_idx, int* point_3_idx,
  int* norm_1_idx, int* norm_2_idx, int* norm_3_idx,
  int* tex_1_idx, int* tex_2_idx, int* tex_3_idx,
  std::vector <std::string> material_name,
  int *material_idx,
  int* num_triangles,
  int* num_materials
);
void extract_material_data(
  std::string folder_path, std::string material_filename,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  int *len_texture,
  int *num_materials,
  std::vector <std::string> &material_name
);
void extract_material_file_names(
  std::string folder_path, std::string obj_filename,
  std::vector <std::string> &material_file_name_array
);
void extract_num_elements(
  std::string folder_path, std::string obj_filename,
  int &num_vertices, int &num_vt, int &num_vn, int &num_faces
);

// https://thispointer.com/c-how-to-find-an-element-in-vector-and-get-its-index/
template < typename T>
std::pair<bool, int > find_in_vector(const std::vector<T>  & vecOfElements, const T  & element)
{
	std::pair<bool, int > result;

	// Find given element in vector
	auto it = std::find(vecOfElements.begin(), vecOfElements.end(), element);

	if (it != vecOfElements.end())
	{
		result.second = std::distance(vecOfElements.begin(), it);
		result.first = true;
	}
	else
	{
		result.first = false;
		result.second = -1;
	}

	return result;
}

void extract_num_elements(
  std::string folder_path, std::string obj_filename,
  int &num_vertices, int &num_vt, int &num_vn, int &num_faces
) {
  std::string complete_obj_filename = folder_path + obj_filename;
  std::ifstream myfile (complete_obj_filename.c_str());
  std::string str;

  num_vertices = 0;
  num_vt = 0;
  num_vn = 0;
  num_faces = 0;

  if (myfile.is_open()){
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, ' ');
        if (chunks[0] == "v") {
          num_vertices += 1;
        } else if (chunks[0] == "vt") {
          num_vt += 1;
        } else if (chunks[0] == "vn") {
          num_vn += 1;
        } else if (chunks[0] == "f") {
          num_faces += chunks.size() - 3;
        }
      }
    }
    myfile.close();
  }
  printf("Number of vertices = %d\n", num_vertices);
  printf("Number of vt       = %d\n", num_vt);
  printf("Number of vn       = %d\n", num_vn);
  printf("Number of faces    = %d\n", num_faces);
}

void extract_material_file_names(
  std::string folder_path, std::string obj_filename,
  std::vector <std::string> &material_file_name_array
) {
  std::string complete_obj_filename = folder_path + obj_filename;
  std::ifstream myfile (complete_obj_filename.c_str());
  std::string str, material_file_name;

  if (myfile.is_open()){

    material_file_name_array.push_back("file_default_123.mtl");

    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, ' ');
        if (chunks[0] == "mtllib") {
          for (int i = 1; i < chunks.size(); i++) {
            material_file_name = chunks[i];
            std::pair<bool, int> result = find_in_vector<std::string>(
              material_file_name_array, material_file_name);
            if (!result.first) {
              printf("Material file name: %s\n", material_file_name.c_str());
              material_file_name_array.push_back(material_file_name);
            }
          }
        }
      }
    }
    myfile.close();
  }
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

void _extract_single_material_data(
  std::string folder_path,
  std::string material_filename,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  float *material_image_r, float *material_image_g, float *material_image_b,
  int *material_image_height, int *material_image_width,
  int *material_image_offset,
  int *len_texture,
  int *num_materials,
  std::vector <std::string> &material_name
) {
  std::string complete_material_filename = folder_path + material_filename;
  std::string str, complete_image_filename, single_material_name;
  int idx = material_name.size() - 1;

  // complete_material_filename = clean_string_end(complete_material_filename);

  if (material_name.size() == 0) {
    idx = 0;
    material_name.push_back("Default_123");

    *(ka_x + idx) = 0;
    *(ka_y + idx) = 0;
    *(ka_z + idx) = 0;

    *(kd_x + idx) = .9;
    *(kd_y + idx) = .9;
    *(kd_z + idx) = .9;

    *(ks_x + idx) = 0;
    *(ks_y + idx) = 0;
    *(ks_z + idx) = 0;

    *(ke_x + idx) = 0;
    *(ke_y + idx) = 0;
    *(ke_z + idx) = 0;

    *(material_image_height + idx) = 0;
    *(material_image_width + idx) = 0;

    *(material_image_offset + idx) = 0;
  }

  std::ifstream myfile (complete_material_filename);

  if (myfile.is_open()) {
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, ' ');
        if (chunks[0] == "newmtl") {
          single_material_name = chunks[1];
          printf("Extracting material %s...\n", single_material_name.c_str());
          material_name.push_back(single_material_name);
          idx++;

          *(ka_x + idx) = 0;
          *(ka_y + idx) = 0;
          *(ka_z + idx) = 0;

          *(kd_x + idx) = .9;
          *(kd_y + idx) = .9;
          *(kd_z + idx) = .9;

          *(ks_x + idx) = 0;
          *(ks_y + idx) = 0;
          *(ks_z + idx) = 0;

          *(ke_x + idx) = 0;
          *(ke_y + idx) = 0;
          *(ke_z + idx) = 0;

          *(material_image_height + idx) = 0;
          *(material_image_width + idx) = 0;

          *(material_image_offset + idx) = \
            *(material_image_offset + idx - 1) +
            (*(material_image_height + idx - 1)) * \
            (*(material_image_width + idx - 1));

        } else if (chunks[0] == "Ka") {
          *(ka_x + idx) = std::stof(chunks[1]);
          *(ka_y + idx) = std::stof(chunks[2]);
          *(ka_z + idx) = std::stof(chunks[3]);
        } else if (chunks[0] == "Kd") {
          *(kd_x + idx) = std::stof(chunks[1]);
          *(kd_y + idx) = std::stof(chunks[2]);
          *(kd_z + idx) = std::stof(chunks[3]);
        } else if (chunks[0] == "Ks") {
          *(ks_x + idx) = std::stof(chunks[1]);
          *(ks_y + idx) = std::stof(chunks[2]);
          *(ks_z + idx) = std::stof(chunks[3]);
        } else if (chunks[0] == "Ke") {
          *(ke_x + idx) = std::stof(chunks[1]);
          *(ke_y + idx) = std::stof(chunks[2]);
          *(ke_z + idx) = std::stof(chunks[3]);
        } else if (chunks[0] == "map_Kd") {
          complete_image_filename = folder_path + chunks[1];
          marengo::jpeg::Image img(complete_image_filename.c_str());
          *(material_image_height + idx) = img.getHeight();
          *(material_image_width + idx) = img.getWidth();
          int local_offset = 0;
          for ( std::size_t y = 0; y < *(material_image_height + idx); ++y )
          {
            for ( std::size_t x = 0; x < *(material_image_width + idx); ++x ) {
              float r = 1.0 * img.getLuminance(x, y, 0) / 255.0;
              float g = 1.0 * img.getLuminance(x, y, 1) / 255.0;
              float b = 1.0 * img.getLuminance(x, y, 2) / 255.0;

              *(material_image_r + local_offset) = r;
              *(material_image_g + local_offset) = g;
              *(material_image_b + local_offset) = b;

              local_offset++;
            }
          }
          printf(
            "Image %s (%d x %d) extracted.\n", complete_image_filename.c_str(),
            *(material_image_height + idx), *(material_image_width + idx)
          );
        }
      }
    }
    myfile.close();
  }
  num_materials[0] = material_name.size();
  len_texture[0] = \
    (*(material_image_offset + idx)) + (*(material_image_height + idx)) *
    (*(material_image_width + idx));
  printf("Texture length so far      = %d\n", len_texture[0]);
  printf("Number of materials so far = %d\n", num_materials[0]);
}

void extract_material_data(
  std::string folder_path,
  std::vector <std::string> material_file_name_array,
  float *ka_x, float *ka_y, float *ka_z,
  float *kd_x, float *kd_y, float *kd_z,
  float *ks_x, float *ks_y, float *ks_z,
  float *ke_x, float *ke_y, float *ke_z,
  float *material_image_r, float *material_image_g, float *material_image_b,
  int *material_image_height, int *material_image_width,
  int *material_image_offset,
  int *len_texture,
  int *num_materials,
  std::vector <std::string> &material_name
) {
  for (int i = 0; i < material_file_name_array.size(); i++) {
    _extract_single_material_data(
      folder_path,
      material_file_name_array[i],
      ka_x, ka_y, ka_z,
      kd_x, kd_y, kd_z,
      ks_x, ks_y, ks_z,
      ke_x, ke_y, ke_z,
      material_image_r, material_image_g, material_image_b,
      material_image_height, material_image_width, material_image_offset,
      len_texture,
      num_materials,
      material_name
    );
  }
}

void extract_triangle_data(
  std::string folder_path,
  std::string obj_filename,
  float* x, float* y, float* z,
  float* x_norm, float* y_norm, float* z_norm,
  float* x_tex, float* y_tex,
  int* point_1_idx, int* point_2_idx, int* point_3_idx,
  int* norm_1_idx, int* norm_2_idx, int* norm_3_idx,
  int* tex_1_idx, int* tex_2_idx, int* tex_3_idx,
  std::vector <std::string> material_name,
  int *material,
  int* num_triangles,
  int* num_materials
) {

  int point_idx = 0, triangle_idx = 0, norm_idx = 0, current_material_idx = 0;
  std::string str, single_material_name;
  std::string filename = folder_path + obj_filename;
  std::ifstream myfile (filename.c_str());
  std::vector <int> index;
  std::vector <std::string> sub_chunks_1, sub_chunks_2, sub_chunks_3;

  if (myfile.is_open()){
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, ' ');
        if (chunks[0] == "usemtl") {
          if (material_name.size() > 1) {
            single_material_name = chunks[1];
            std::pair<bool, int> result = find_in_vector<std::string>(
              material_name, single_material_name);
            current_material_idx = result.second;
            printf("Current material idx = %d\n", current_material_idx);
          } else {
            current_material_idx = 0;
          }
        } else if (chunks[0] == "v") {
          *(x + point_idx) = std::stof(chunks[1]);
          *(y + point_idx) = std::stof(chunks[2]);
          *(z + point_idx) = std::stof(chunks[3]);
          point_idx++;
        } else if (chunks[0] == "vn") {
          *(x_norm + norm_idx) = std::stof(chunks[1]);
          *(y_norm + norm_idx) = std::stof(chunks[2]);
          *(z_norm + norm_idx) = std::stof(chunks[3]);
          norm_idx++;
        } else if (chunks[0] == "f") {

          sub_chunks_1 = split(chunks[1], '/');

          for (unsigned int i = 0; i < chunks.size() - 3; i++) {
            sub_chunks_2 = split(chunks[2 + i], '/');
            sub_chunks_3 = split(chunks[3 + i], '/');

            *(point_1_idx + triangle_idx) = std::stoi(sub_chunks_1[0]) - 1;
            *(point_2_idx + triangle_idx) = std::stoi(sub_chunks_2[0]) - 1;
            *(point_3_idx + triangle_idx) = std::stoi(sub_chunks_3[0]) - 1;

            if (sub_chunks_1.size() > 2) {
              *(norm_1_idx + triangle_idx) = std::stoi(sub_chunks_1[2]) - 1;
              *(norm_2_idx + triangle_idx) = std::stoi(sub_chunks_2[2]) - 1;
              *(norm_3_idx + triangle_idx) = std::stoi(sub_chunks_3[2]) - 1;
            } else {
              *(norm_1_idx + triangle_idx) = 0;
              *(norm_2_idx + triangle_idx) = 0;
              *(norm_3_idx + triangle_idx) = 0;
            }

            *(material + triangle_idx) = current_material_idx;

            triangle_idx++;
          }
        }
      }
    }

    if (norm_idx == 0) {
      *(x_norm + norm_idx) = 0;
      *(y_norm + norm_idx) = 0;
      *(z_norm + norm_idx) = 0;
      norm_idx++;
    }

    myfile.close();
  }
  num_triangles[0] = triangle_idx;
}

#endif
