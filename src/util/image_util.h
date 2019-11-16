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
  std::ofstream ofs;
  ofs.open(filename);
  ofs << "P6\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      size_t pixel_index = i * width + j;

      float r = clamp(0, 1, image[pixel_index].r());
      float g = clamp(0, 1, image[pixel_index].g());
      float b = clamp(0, 1, image[pixel_index].b());

      char ir = (char)(255 * r);
      char ig = (char)(255 * g);
      char ib = (char)(255 * b);

      ofs << ir << ig << ib;
    }
  }
  ofs.close();
}

#endif
