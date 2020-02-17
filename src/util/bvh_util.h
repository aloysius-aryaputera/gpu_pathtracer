#ifndef BVH_UTIL_H
#define BVH_UTIL_H

__device__ unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ unsigned int compute_morton_3d(float x, float y, float z) {
  x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
  y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
  z = fminf(fmaxf(z * 1024.0f, 0.0f ), 1023.0f);
  unsigned int xx = expand_bits((unsigned int)x);
  unsigned int yy = expand_bits((unsigned int)y);
  unsigned int zz = expand_bits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

__device__ int length_longest_common_prefix(
  unsigned int* morton_code_list, int i, int j, int array_length
) {
  if (i < 0 || j < 0 || i >= array_length || j >= array_length) return -1;
  int additional_length = 0;
  if (morton_code_list[i] == morton_code_list[j]) {
    additional_length = __clz(i ^ j);
  }
  return __clz(morton_code_list[i] ^ morton_code_list[j]) + additional_length;
}

#endif
