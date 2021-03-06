#ifndef READ_IMAGE_UTIL_H
#define READ_IMAGE_UTIL_H

#include <string>
#include <vector>

#include "../util/string_util.h"

void extract_image_resource_requirement(
  std::string folder_path,
  std::vector <std::string> material_filename_array,
  std::vector <std::string> &image_file_name,
  std::vector <int> &image_offset,
  std::vector <int> &image_height,
  std::vector <int> &image_width,
  long int &texture_length
);

void _extract_image_resource_requirement_single_mtl(
  std::string folder_path,
  std::string material_filename,
  std::vector <std::string> &image_file_name,
  std::vector <int> &image_offset,
  std::vector <int> &image_height,
  std::vector <int> &image_width,
  long int &texture_length
);

void extract_textures(
  std::string folder_path,
  std::vector <std::string> image_file_name_array,
  float *texture_r,
  float *texture_g,
  float *texture_b
);

void extract_single_image_requirement(
  std::string folder_path, std::string image_file_name, int &height, int &width
);

void extract_single_image(
  std::string folder_path, std::string image_file_name,
  float *texture_r, float *texture_g, float *texture_b,
  int &next_idx
);

void extract_single_image(
  std::string folder_path, std::string image_file_name,
  float *texture_r, float *texture_g, float *texture_b,
  int &next_idx
) {
  std::string complete_image_filename = folder_path + image_file_name;
  marengo::jpeg::Image img(complete_image_filename.c_str());
  printf(
    "Extracting image %s (%lu x %lu)...\n", complete_image_filename.c_str(),
    img.getHeight(), img.getWidth()
  );
  for (int y = img.getHeight() - 1; y >= 0; --y ) {
    for (int x = 0; x < img.getWidth(); ++x ) {

      *(texture_r + next_idx) = 1.0 * img.getLuminance(x, y, 0) / 255.0;
      *(texture_g + next_idx) = 1.0 * img.getLuminance(x, y, 1) / 255.0;
      *(texture_b + next_idx) = 1.0 * img.getLuminance(x, y, 2) / 255.0;

      next_idx++;
    }
  }
  printf(
    "Image %s (%lu x %lu) extracted.\n", complete_image_filename.c_str(),
    img.getHeight(), img.getWidth()
  );
}

void extract_textures(
  std::string folder_path,
  std::vector <std::string> image_file_name_array,
  float *texture_r,
  float *texture_g,
  float *texture_b
) {
  int next_idx = 0;

  for (std::string image_file_name : image_file_name_array) {
    if (image_file_name == "Default_texture") {
      texture_r[next_idx] = 1;
      texture_g[next_idx] = 1;
      texture_b[next_idx] = 1;
      next_idx++;
    } else {
      extract_single_image(
        folder_path, image_file_name,
        texture_r, texture_g, texture_b,
        next_idx
      );
    }
  }
}

void extract_image_resource_requirement(
  std::string folder_path,
  std::vector <std::string> material_filename_array,
  std::vector <std::string> &image_file_name,
  std::vector <int> &image_offset,
  std::vector <int> &image_height,
  std::vector <int> &image_width,
  long int &texture_length
) {
  for (int i = 0; i < material_filename_array.size(); i++) {
    _extract_image_resource_requirement_single_mtl(
      folder_path,
      material_filename_array[i],
      image_file_name,
      image_offset,
      image_height,
      image_width,
      texture_length
    );
  }
}

void _extract_image_resource_requirement_single_mtl(
  std::string folder_path,
  std::string material_filename,
  std::vector <std::string> &image_file_name,
  std::vector <int> &image_offset,
  std::vector <int> &image_height,
  std::vector <int> &image_width,
  long int &texture_length
) {
  std::string complete_material_filename = folder_path + material_filename;
  std::string str;
  long int next_offset = texture_length;
  int height, width;

  if (image_file_name.size() == 0) {
    image_file_name.push_back("Default_texture");
    height = 1;
    width = 1;
    image_offset.push_back(next_offset);
    image_height.push_back(height);
    image_width.push_back(width);
    next_offset += height * width;
    texture_length = 1;
  }

  std::ifstream myfile (complete_material_filename);

  if (myfile.is_open()) {
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, ' ');
        if (
          chunks[0] == "map_Kd" || chunks[0] == "map_Ks" ||
          chunks[0] == "map_Ns" || chunks[0] == "map_Ke" ||
          chunks[0] == "map_Bump"
        ) {
					int chunks_size = chunks.size();
          std::pair<bool, int> result = find_in_vector<std::string>(
            image_file_name, chunks[chunks_size - 1]);
          if (!result.first) {
            extract_single_image_requirement(
              folder_path, chunks[chunks_size - 1], height, width
            );
            image_file_name.push_back(chunks[chunks_size - 1]);
            image_offset.push_back(next_offset);
            image_height.push_back(height);
            image_width.push_back(width);
            next_offset += height * width;
            texture_length = next_offset;
          }
        }
      }
    }
    myfile.close();
  }
}

void extract_single_image_requirement(
  std::string folder_path, std::string image_file_name, int &height, int &width
) {
  std::string complete_image_filename = folder_path + image_file_name;
  marengo::jpeg::Image img(complete_image_filename.c_str());
  height = img.getHeight();
  width = img.getWidth();
}

#endif
