#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>

#include "model/camera.h"
#include "model/data_structure/local_vector.h"
#include "model/geometry/sphere.h"
#include "model/geometry/triangle.h"
#include "model/grid/cell.h"
#include "model/grid/grid.h"
#include "model/material.h"
#include "model/ray.h"
#include "model/scene.h"
#include "model/vector_and_matrix/vec3.h"
#include "render/pathtracing.h"
#include "util/image_util.h"
#include "util/read_file_util.h"

__global__ void create_world(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
);

__global__ void create_world_2(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
);

__global__ void create_world_3(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
);

__global__ void create_world_3(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      *(camera) = new Camera(
        vec3(30, 30, 20), vec3(0, 1, 0), vec3(0, 0, 0), 45, image_width,
        image_height
      );

      Material *triangle_material = new Material(
        vec3(.2, .2, .2), vec3(.3, .3, .3), vec3(0, 0, 0), vec3(.3, .3, .3)
      );

      for (int idx = 0; idx < num_triangles[0]; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]),
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]),
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]),
          triangle_material
        );
      }

    }
}

__global__ void create_world(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      *(camera) = new Camera(
        vec3(0, 5, 9), vec3(0, 1, 0), vec3(0, 1, 0), 45, image_width,
        image_height
      );

      Material *triangle_material = new Material(
        vec3(0, 0, 0), vec3(1, 1, 1), vec3(0, 0, 0), vec3(.45, .45, .45)
      );

      float s_x = 1, s_y = 1, s_z = 1, t_x = 0, t_y = 0, t_z = 0;
      vec3 t_v = vec3(t_x, t_y, t_z);

      for (int idx = 0; idx < num_triangles[0]; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]) *\
            s_x + t_v,
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]) *\
            s_y + t_v,
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]) *\
            s_z + t_v,
          triangle_material
        );
      }

      triangle_material = new Material(
        vec3(.2, .2, .2), vec3(.2, 1, .2), vec3(0, 0, 0), vec3(.1, .3, .1)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-6, 0, 6), vec3(6, 0, 6), vec3(0, 0, -6),
        triangle_material
      );

    }
}

__global__ void create_world_2(
  Camera** camera, Primitive** geom_array, float *x, float *y, float *z,
  int *point_1_idx, int *point_2_idx, int *point_3_idx, int* num_triangles,
  int image_width, int image_height
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      *(camera) = new Camera(
        vec3(0, 5, 9), vec3(0, 1, 0), vec3(0, 1, 0), 45, image_width,
        image_height
      );

      Material *triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .9, .2), vec3(0, 0, 0), vec3(.5, .5, .5)
      );

      float s_x = .4, s_y = .4, s_z = .4, t_x = 0, t_y = 0, t_z = 0;
      vec3 t_v = vec3(t_x, t_y, t_z);

      for (int idx = 0; idx < num_triangles[0]; idx++) {
        *(geom_array + idx) = new Triangle(
          vec3(x[point_1_idx[idx]], y[point_1_idx[idx]], z[point_1_idx[idx]]) *\
            s_x + t_v,
          vec3(x[point_2_idx[idx]], y[point_2_idx[idx]], z[point_2_idx[idx]]) *\
            s_y + t_v,
          vec3(x[point_3_idx[idx]], y[point_3_idx[idx]], z[point_3_idx[idx]]) *\
            s_z + t_v,
          triangle_material
        );
      }

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .2, .9), vec3(40.0, .2, 40.0),
        vec3(.3, .1, .3)
      );
      *(geom_array + num_triangles[0]++) = new Sphere(
        vec3(2.5, 1.0, 2.5), 1, triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .9, .9), vec3(0, 0, 0), vec3(.5, .5, .5)
      );
      *(geom_array + num_triangles[0]++) = new Sphere(
        vec3(-2.5, 1.0, 2.5), 1, triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .9, .9), vec3(0, 0, 0), vec3(.3, .3, .3)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-6, 0, 6), vec3(6, 0, 6), vec3(0, 0, -12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .2, .2), vec3(0, 0, 0), vec3(.5, .5, .5)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-8, 8.8, 8), vec3(-6, 0, 8), vec3(0, 0, -12),
        triangle_material
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-8, 8.8, 8), vec3(0, 0, -12), vec3(0, 8.8, -12),
        triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.2, .2, .9), vec3(0, 0, 0), vec3(.5, .5, .5)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(8, 8.8, 8), vec3(0, 0, -12), vec3(6, 0, 8), triangle_material
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(8, 8.8, 8), vec3(0, 8.8, -12), vec3(0, 0, -12), triangle_material
      );

      triangle_material = new Material(
        vec3(0, 0, 0), vec3(.9, .9, .9), vec3(0, 0, 0), vec3(.3, .3, .3)
      );
      *(geom_array + num_triangles[0]++) = new Triangle(
        vec3(-8, 9, 8), vec3(0, 9, -12), vec3(8, 9, 8), triangle_material
      );

      // triangle_material = new Material(
      //   vec3(0, 0, 0), vec3(1, 1, 1), vec3(30.0, 30.0, 30.0), vec3(1, 1, 1)
      // );
      // *(geom_array + num_triangles[0]++) = new Sphere(
      //   vec3(0, 11, 0), 3, triangle_material);

    }
}
