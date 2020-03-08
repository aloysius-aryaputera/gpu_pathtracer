#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <fstream>
#include <sstream>
#include <string>

__host__ float clamp(const float &lo, const float &hi, const float &v);
__host__ void save_image(
  vec3* image, int width, int height, std::string filename
);

__host__ float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

__host__ void save_image(
  vec3* image, int width, int height, std::string filename
) {
  printf("Saving an image of width %d and height %d.\n", width, height);
  std::ofstream ofs;
  size_t pixel_index;
  float r, g, b;

  ofs.open(filename);
  ofs << "P3\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pixel_index = i * width + j;

      r = clamp(0, 1, sqrt(image[pixel_index].r()));
      g = clamp(0, 1, sqrt(image[pixel_index].g()));
      b = clamp(0, 1, sqrt(image[pixel_index].b()));

      ofs << (int)(255 * r) << " " << (int)(255 * g) << " " << (int)(255 * b) << " ";
    }
  }
  ofs.close();
}

#endif
