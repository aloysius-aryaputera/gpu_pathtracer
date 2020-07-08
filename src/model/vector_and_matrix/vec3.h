//File: vec3.h
#ifndef VEC3_H
#define VEC3_H

// Taken from: https://github.com/rogerallen/raytracinginoneweekendincuda/blob/ch02_vec3_cuda/vec3.h

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {

  public:
    __host__ __device__ vec3() { e[0] = 0; e[1] = 0; e[2] = 0; }
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }
    __host__ __device__ inline float u() const { return e[0]; }
    __host__ __device__ inline float v() const { return e[1]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();
    __device__ inline bool vector_is_nan();


    float e[3];
};

__host__ __device__ inline vec3 rotate(vec3 vector, vec3 axis, float theta);
inline std::istream& operator>>(std::istream &is, vec3 &t);
inline std::ostream& operator<<(std::ostream &os, const vec3 &t);
__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 operator*(float t, const vec3 &v);
__host__ __device__ inline vec3 operator/(vec3 v, float t);
__host__ __device__ inline vec3 operator*(const vec3 &v, float t);
__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2);
__host__ __device__ inline float compute_distance(
  const vec3 &v1, const vec3 &v2
);
__host__ __device__ inline vec3 permute(vec3 v, int kx, int ky, int kz);
__device__ inline vec3 abs(vec3 v);
__device__ inline int max_dimension(vec3 v);
__device__ inline vec3 de_nan(const vec3& c);
__host__ __device__ inline vec3 unit_vector(vec3 v);
__host__ __device__ void print_vec3(vec3 v);

__host__ __device__ inline vec3 rotate(vec3 vector, vec3 axis, float theta) {
  return cos(theta) * vector + sin(theta) * cross(vector, axis) + 
    (1 - cos(theta)) * dot(axis, vector) * axis; 
}

inline std::istream& operator>>(std::istream &is, vec3 &t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
  float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
  e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
  return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
  return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
  return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
  return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
              (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
              (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

__host__ __device__ inline float compute_distance(
  const vec3 &v1, const vec3 &v2
) {
  vec3 temp = v1 - v2;
  return temp.length();
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v){
  e[0]  += v.e[0];
  e[1]  += v.e[1];
  e[2]  += v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
  e[0]  *= v.e[0];
  e[1]  *= v.e[1];
  e[2]  *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
  e[0]  /= v.e[0];
  e[1]  /= v.e[1];
  e[2]  /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
  e[0]  -= v.e[0];
  e[1]  -= v.e[1];
  e[2]  -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
  e[0]  *= t;
  e[1]  *= t;
  e[2]  *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
  float k = 1.0/t;

  e[0]  *= k;
  e[1]  *= k;
  e[2]  *= k;
  return *this;
}

__host__ __device__ inline vec3 permute(vec3 v, int kx, int ky, int kz) {
  return vec3(v.e[kx], v.e[ky], v.e[kz]);
}

__device__ inline bool vec3::vector_is_nan() {
  return isnan(e[0]) || isnan(e[1]) || isnan(e[2]);
}

__device__ inline vec3 abs(vec3 v) {
  return vec3(abs(v.x()), abs(v.y()), abs(v.z()));
}

__device__ inline int max_dimension(vec3 v) {
  if (v.x() > v.y() && v.x() > v.z()) {
    return 0;
  }
  if (v.y() > v.x() && v.y() > v.z()) {
    return 1;
  }
  return 2;
}

__device__ inline vec3 de_nan(const vec3& c) {
    vec3 temp = c;
    if (!(temp[0] == temp[0])) temp[0] = 0;
    if (!(temp[1] == temp[1])) temp[1] = 0;
    if (!(temp[2] == temp[2])) temp[2] = 0;
    return temp;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__host__ __device__ void print_vec3(vec3 v) {
  printf("\n");
  printf("x = %5.5f; y = %5.5f; z = %5.5f\n", v.x(), v.y(), v.z());
  printf("\n");
}

#endif
