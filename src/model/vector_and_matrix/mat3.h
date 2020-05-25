//File: mat3.h
#ifndef MAT3_H
#define MAT3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "vec3.h"

class Mat3 {
  public:
    __host__ __device__ Mat3();
    __host__ __device__ Mat3(
      float a, float b, float c,
      float d, float e, float f,
      float g, float h, float i
    );

    vec3 row[3];

};

inline vec3 operator * (const Mat3 &m, const vec3 &v);

__host__ __device__ Mat3::Mat3() {
  this -> row[0] = vec3();
  this -> row[1] = vec3();
  this -> row[2] = vec3();
}

__host__ __device__ Mat3::Mat3(
	float a, float b, float c,
	float d, float e, float f,
	float g, float h, float i
) {
  this -> row[0] = vec3(a, b, c);
  this -> row[1] = vec3(d, e, f);
  this -> row[2] = vec3(g, h, i);
}


inline vec3 operator * (const Mat3 &m, const vec3 &v) {
  vec3 result;
  result = vec3(
  	dot(m.row[0], v), dot(m.row[1], v), dot(m.row[2], v)
	);
  return result;
}

#endif
