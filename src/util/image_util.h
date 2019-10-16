#ifndef IMAGE_UTIL_H
#define IMAGE_UTIL_H

#include <fstream>
#include <sstream>

__host__ float clamp(const float &lo, const float &hi, const float &v);
__host__ void save_image(
  float* image, int width, int height, const char * filename
);

__host__ float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

__host__ void save_image(
  float* image, int width, int height, const char * filename
) {
  std::ofstream ofs;
  ofs.open(filename);
  ofs << "P3\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      size_t pixel_index = j*3*width + i*3;

      float r = clamp(0, 1, image[pixel_index + 0]);
      float g = clamp(0, 1, image[pixel_index + 1]);
      float b = clamp(0, 1, image[pixel_index + 2]);

      int ir = int(255.99 * r);
      int ig = int(255.99 * g);
      int ib = int(255.99 * b);

      ofs << ir << " " << ig << " " << ib << "\n";
    }
  }
  ofs.close();
}

#endif
