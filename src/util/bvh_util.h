#ifndef BVH_UTIL_H
#define BVH_UTIL_H

unsigned int expand_bits(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

unsigned int morton_3d(float x, float y, float z) {
  x = min(max(x * 1024.0f), 1023.0f);
  y = min(max(y * 1024.0f), 1023.0f);
  z = min(max(z * 1024.0f), 1023.0f);
  unsigned int xx = expand_bits((unsigned int)x);
  unsigned int yy = expand_bits((unsigned int)y);
  unsigned int zz = expand_bits((unsigned int)z);
  return xx * 4 + yy * 2 + zz;
}

#endif
