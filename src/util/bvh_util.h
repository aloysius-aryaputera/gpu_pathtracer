#ifndef BVH_UTIL_H
#define BVH_UTIL_H

__device__ unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ unsigned int compute_morton_3d(float x, float y, float z) {
  x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
  y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
  z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
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
    additional_length = __clz((unsigned int)i ^ (unsigned int)j);
  }
  return __clz(morton_code_list[i] ^ morton_code_list[j]) + additional_length;
}

// __device__ int find_split(
//   unsigned int *morton_code_list, int first, int last
// ) {
//   // Ref: https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
//
//   unsigned int first_code = morton_code_list[first];
//   unsigned int last_code = morton_code_list[last];
//
//   // Identical Morton codes => split the range in the middle
//   if (first_code == last_code)
//     return (first + last) >> 1;
//
//   // Calculate the number of highest bits that are the same for all objects,
//   // using the count-leading-zeros intrinsic
//   int common_prefix = __clz(first_code ^ last_code);
//
//   // Use binary search to find where the next bit differs.
//   // Specifically, we are looking for the highest object that
//   // shares more than common_prefix bits with the first one.
//
//   int split = first;
//   int step = last - first;
//   int new_split, split_prefix;
//   unsigned int split_code;
//
//   do {
//     step = (step + 1) >> 1;
//     new_split = split + step;
//
//     if (new_split < last) {
//       split_code = morton_code_list[new_split];
//       split_prefix = __clz(first_code ^ last_code);
//       if (split_prefix > common_prefix) {
//         split = new_split;
//       }
//     }
//   }
//   while (step > 1);
//
//   return split;
//
// }

#endif
