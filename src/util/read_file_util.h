#ifndef READ_FILE_UTIL_H
#define READ_FILE_UTIL_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "../model/geometry/triangle.h"
#include "../model/vector_and_matrix/vec3.h"

std::vector<std::string> split(const std::string& s, char delimiter);
void extract_triangle_data(
  char* filename,
  float* x, float* y, float* z,
  float* x_norm, float* y_norm, float* z_norm,
  int* point_1_idx, int* point_2_idx, int* point_3_idx,
  int* norm_1_idx, int* norm_2_idx, int* norm_3_idx,
  int* num_triangles
);

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

void extract_triangle_data(
  std::string filename,
  float* x, float* y, float* z,
  float* x_norm, float* y_norm, float* z_norm,
  int* point_1_idx, int* point_2_idx, int* point_3_idx,
  int* norm_1_idx, int* norm_2_idx, int* norm_3_idx,
  int* num_triangles
) {

  int point_idx = 0, triangle_idx = 0, norm_idx = 0;
  std::ifstream myfile (filename);
  std::string str;
  std::vector <int> index;
  std::vector <std::string> sub_chunks_1, sub_chunks_2, sub_chunks_3;

  if (myfile.is_open()){
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        std::vector <std::string> chunks = split(str, ' ');
        // for (unsigned int i = 0; i < chunks.size(); i++) {
        if (chunks[0] == "v") {
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
