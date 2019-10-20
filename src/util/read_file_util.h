#ifndef READ_FILE_UTIL_H
#define READ_FILE_UTIL_H

#include <string>
#include <fstream>

#include "../model/geometry/triangle.h"
#include "../model/vector_and_matrix/vec3.h"

void extract_triangle_data(
  const char* filename, float* x, float* y, float* z, int* point_1_idx,
  int* point_2_idx, int* point_3_idx, int* num_triangles
);

void extract_triangle_data(
  const char* filename, float* x, float* y, float* z, int* point_1_idx,
  int* point_2_idx, int* point_3_idx, int* num_triangles
) {
  float num[3];
  char letter;
  int point_idx = 0, triangle_idx = 0;
  std::ifstream myfile (filename);

  if (myfile.is_open()){
    while(myfile >> letter) {
      if (letter != 'v' && letter != 'f') return;
      myfile >> num[0] >> num[1] >> num[2];
      if (letter == 'v') {
        *(x + point_idx) = num[0];
        *(y + point_idx) = num[1];
        *(z + point_idx) = num[2];
        point_idx++;
      }
      if (letter == 'f') {
        *(point_1_idx + triangle_idx) = int(num[0]) - 1;
        *(point_2_idx + triangle_idx) = int(num[1]) - 1;
        *(point_3_idx + triangle_idx) = int(num[2]) - 1;
        triangle_idx++;
      }
    }
    myfile.close();
  }
  num_triangles[0] = triangle_idx;
}

#endif
