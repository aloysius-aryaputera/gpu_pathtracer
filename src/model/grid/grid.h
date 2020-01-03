//File: grid.h
#ifndef GRID_H
#define GRID_H

#include <cuda_fp16.h>
#include <math.h>

#include "../../param.h"
#include "../geometry/primitive.h"
#include "../ray.h"
#include "../vector_and_matrix/vec3.h"
#include "bounding_box.h"
#include "cell.h"

class Grid {

  private:
    __device__ bool _is_inside(vec3 position);
    __device__ bool _grid_hit(
      Ray ray, Primitive **object_array_, int num_objects_, hit_record &rec
    );

    BoundingBox *world_bounding_box;

  public:
    __host__ __device__ Grid() {}
    __device__ Grid(
      float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
      float z_max_, int n_cell_x_, int n_cell_y_, int n_cell_z_,
      Primitive** object_array_, int num_objects_, Cell** cell_array_,
      int max_num_objects_
    );
    __device__ int convert_3d_to_1d_cell_address(int i, int j, int k);
    __device__ int convert_3d_to_1d_cell_address(vec3 address_3d);
    __device__ vec3 find_cell_address(vec3 position);
    __device__ bool do_traversal(Ray ray, hit_record &rec);

    float x_min, x_max, y_min, y_max, z_min, z_max, cell_size_x, cell_size_y, \
      cell_size_z;
    int n_cell_x, n_cell_y, n_cell_z, num_objects, max_num_objects_per_cell;
    Cell** cell_array;
    Primitive** object_array;

};

__global__ void build_cell_array(Grid** grid, Primitive** cell_object_array);
__global__ void insert_objects(Grid** grid);
__global__ void prepare_grid(
  Camera** camera, Primitive** geom_array, int *num_objects,
  int *n_cell_x, int *n_cell_y, int *n_cell_z,
  int max_n_cell_x, int max_n_cell_y, int max_n_cell_z
);
__global__ void create_grid(
  Camera** camera, Grid** grid, Primitive** geom_array, int *num_objects,
  Cell** cell_array, int *n_cell_x, int *n_cell_y, int *n_cell_z,
  int max_n_cell_x, int max_n_cell_y, int max_n_cell_z,
  int max_num_objects_per_cell
);
__device__ void _compute_scene_boundaries(
  float &x_min, float &x_max, float &y_min, float &y_max, float &z_min,
  float &z_max, float &volume, Primitive **geom_array, int num_objects,
  Camera **camera
);
__device__ void _print_grid_details(Grid* grid);

__device__ void _print_grid_details(Grid* grid) {
  float d_x = grid -> x_max - grid -> x_min;
  float d_y = grid -> y_max - grid -> y_min;
  float d_z = grid -> z_max - grid -> z_min;

  printf("================================================================\n");
  printf("Boundaries\n");
  printf("================================================================\n");
  printf("x_min = %5.5f; x_max = %5.5f, d_x = %5.5f\n",
    grid -> x_min, grid -> x_max, d_x);
  printf("y_min = %5.5f; y_max = %5.5f, d_y = %5.5f\n",
    grid -> y_min, grid -> y_max, d_y);
  printf("z_min = %5.5f; z_max = %5.5f, d_z = %5.5f\n",
    grid -> z_min, grid -> z_max, d_z);
  printf("\n");
  printf("================================================================\n");
  printf("Elements\n");
  printf("================================================================\n");
  printf("Number of objects = %d\n", grid -> num_objects);
  printf("\n");
  printf("================================================================\n");
  printf("Grid\n");
  printf("================================================================\n");
  printf("x resolution = %d, y resolution = %d, z resolution = %d\n",
         grid -> n_cell_x, grid -> n_cell_y, grid -> n_cell_z);
  printf("\n");
}

__device__ void _compute_scene_boundaries(
  float &x_min, float &x_max, float &y_min, float &y_max, float &z_min,
  float &z_max, Primitive **geom_array, int num_objects, Camera *camera
) {
  x_min = camera -> eye.x();
  x_max = camera -> eye.x();
  y_min = camera -> eye.y();
  y_max = camera -> eye.y();
  z_min = camera -> eye.z();
  z_max = camera -> eye.z();

  printf("num_objects = %d\n", num_objects);
  for (int i = 0; i < num_objects; i++) {
    x_min = min(x_min, geom_array[i] -> get_bounding_box() -> x_min);
    x_max = max(x_max, geom_array[i] -> get_bounding_box() -> x_max);
    y_min = min(y_min, geom_array[i] -> get_bounding_box() -> y_min);
    y_max = max(y_max, geom_array[i] -> get_bounding_box() -> y_max);
    z_min = min(z_min, geom_array[i] -> get_bounding_box() -> z_min);
    z_max = max(z_max, geom_array[i] -> get_bounding_box() -> z_max);
  }

  x_min -= 1;
  x_max += 1;
  y_min -= 1;
  y_max += 1;
  z_min -= 1;
  z_max += 1;

}

__global__ void prepare_grid(
  Camera** camera, Primitive** geom_array, int *num_objects,
  int *n_cell_x, int *n_cell_y, int *n_cell_z,
  int max_n_cell_x, int max_n_cell_y, int max_n_cell_z
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float x_min, x_max, y_min, y_max, z_min, z_max, d_x, d_y, d_z, volume;

    _compute_scene_boundaries(
      x_min, x_max, y_min, y_max, z_min, z_max, geom_array, num_objects[0],
      camera[0]
    );

    d_x = x_max - x_min;
    d_y = y_max - y_min;
    d_z = z_max - z_min;
    volume = d_x * d_y * d_z;

    n_cell_x[0] = min(
      max_n_cell_x, int(d_x * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));
    n_cell_y[0] = min(
      max_n_cell_y, int(d_y * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));
    n_cell_z[0] = min(
      max_n_cell_z, int(d_z * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));
  }
}

__global__ void create_grid(
  Camera** camera, Grid** grid, Primitive** geom_array, int *num_objects,
  Cell** cell_array, int *n_cell_x, int *n_cell_y, int *n_cell_z,
  int max_n_cell_x, int max_n_cell_y, int max_n_cell_z,
  int max_num_objects_per_cell
) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float x_min, x_max, y_min, y_max, z_min, z_max, d_x, d_y, d_z, volume;

    _compute_scene_boundaries(
      x_min, x_max, y_min, y_max, z_min, z_max, geom_array, num_objects[0],
      camera[0]
    );

    d_x = x_max - x_min;
    d_y = y_max - y_min;
    d_z = z_max - z_min;
    volume = d_x * d_y * d_z;

    n_cell_x[0] = min(
      max_n_cell_x, int(d_x * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));
    n_cell_y[0] = min(
      max_n_cell_y, int(d_y * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));
    n_cell_z[0] = min(
      max_n_cell_z, int(d_z * powf(LAMBDA * num_objects[0] / volume, 1.0f / 3)));

    *(grid) = new Grid(
      x_min, x_max, y_min, y_max, z_min, z_max, n_cell_x[0], n_cell_y[0],
      n_cell_z[0], geom_array, num_objects[0], cell_array,
      max_num_objects_per_cell
    );

    _print_grid_details(grid[0]);
  }
}

__global__ void insert_objects(Grid** grid) {
  int cell_address;
  bool intersecting = false;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if(
    (k >= grid[0] -> n_cell_z) || (j >= grid[0] -> n_cell_y) ||
    (i >= grid[0] -> n_cell_x)
  ) return;

  int counter = 0;
  for (int l = 0; l < grid[0] -> num_objects; l++) {
    BoundingBox *obj_bounding_box = grid[0] -> object_array[l] -> get_bounding_box();
    cell_address = grid[0] -> convert_3d_to_1d_cell_address(i, j, k);
    intersecting = \
      grid[0] -> cell_array[cell_address] -> are_intersecting(
        obj_bounding_box
      );
      if (intersecting && counter < grid[0] -> max_num_objects_per_cell) {
        grid[0] -> cell_array[cell_address] -> add_object(
          grid[0] -> object_array[l]);
        counter++;
      }

      if (counter >= grid[0] -> max_num_objects_per_cell) {
        printf(
          "Break! (num_objects = %d, max_objects/cell = %d, l = %d, \
            (%d, %d, %d), (%d, %d, %d))\n",
          grid[0] -> num_objects, grid[0] -> max_num_objects_per_cell,
          l, threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z
        );
        break;
      }
  }
}

__global__ void build_cell_array(Grid** grid, Primitive** cell_object_array) {
  float cell_x_min, cell_x_max, cell_y_min, cell_y_max, cell_z_min, cell_z_max;
  int cell_address;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  if(
    (k >= grid[0] -> n_cell_z) || (j >= grid[0] -> n_cell_y) ||
    (i >= grid[0] -> n_cell_x)
  ) return;

  cell_x_min = grid[0] -> x_min + i * grid[0] -> cell_size_x;
  cell_x_max = cell_x_min + grid[0] -> cell_size_x;

  cell_y_min = grid[0] -> y_min + j * grid[0] -> cell_size_y;
  cell_y_max = cell_y_min + grid[0] -> cell_size_y;

  cell_z_min = grid[0] -> z_min + k * grid[0] -> cell_size_z;
  cell_z_max = cell_z_min + grid[0] -> cell_size_z;

  cell_address = grid[0] -> convert_3d_to_1d_cell_address(i, j, k);
  *((grid[0] -> cell_array) + cell_address) = \
    new Cell(
      cell_x_min, cell_x_max, cell_y_min, cell_y_max, cell_z_min,
      cell_z_max, i, j, k,
      cell_object_array + (grid[0] -> max_num_objects_per_cell * cell_address),
      grid[0] -> max_num_objects_per_cell
    );
}

__device__ int Grid::convert_3d_to_1d_cell_address(int i, int j, int k) {
  return k + j * n_cell_z + i * n_cell_z * n_cell_y;
}

__device__ int Grid::convert_3d_to_1d_cell_address(vec3 address_3d) {
  int k = floorf(fminf(address_3d.z(), n_cell_z - 1));
  int j = floorf(fminf(address_3d.y(), n_cell_y - 1));
  int i = floorf(fminf(address_3d.x(), n_cell_x - 1));
  return k + j * n_cell_z + i * n_cell_z * n_cell_y;
}

__device__ Grid::Grid(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_, int n_cell_x_, int n_cell_y_, int n_cell_z_,
  Primitive** object_array_, int num_objects_, Cell** cell_array_,
  int max_num_objects_per_cell_
) {
  x_min = x_min_;
  x_max = x_max_;
  y_min = y_min_;
  y_max = y_max_;
  z_min = z_min_;
  z_max = z_max_;
  object_array = object_array_;
  num_objects = num_objects_;
  cell_array = cell_array_;
  max_num_objects_per_cell = max_num_objects_per_cell_;

  n_cell_x = n_cell_x_;
  n_cell_y = n_cell_y_;
  n_cell_z = n_cell_z_;

  cell_size_x = (x_max - x_min) / n_cell_x;
  cell_size_y = (y_max - y_min) / n_cell_y;
  cell_size_z = (z_max - z_min) / n_cell_z;

  world_bounding_box = new BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max);
}

__device__ bool Grid::_is_inside(vec3 position) {
  return world_bounding_box -> is_inside(position);
}

__device__ vec3 Grid::find_cell_address(vec3 position) {
  int cell_address_i, cell_address_j, cell_address_k;
  if (!_is_inside(position)) {
    printf("The position is not inside the grid!");
  }
  cell_address_i = floorf((position.x() - x_min) / cell_size_x);
  cell_address_j = floorf((position.y() - y_min) / cell_size_y);
  cell_address_k = floorf((position.z() - z_min) / cell_size_z);
  return vec3(cell_address_i, cell_address_j, cell_address_k);
}

__device__ bool Grid::_grid_hit(
  Ray ray, Primitive **object_array_, int num_objects_, hit_record &rec
) {
  hit_record cur_rec;
  bool hit = false, intersection_found = false;

  rec.t = INFINITY;

  for (int idx = 0; idx < num_objects_; idx++) {
      hit = (object_array_[idx]) -> hit(ray, rec.t, cur_rec);
    if (hit) {
      intersection_found = true;
      rec = cur_rec;
    }
  }

  return intersection_found;
}

__device__ bool Grid::do_traversal(Ray ray, hit_record &rec) {

  vec3 initial_address = find_cell_address(ray.p0);

  if (
    ray.p0.x() <= x_min || ray.p0.x() >= x_max || ray.p0.y() <= y_min ||
    ray.p0.y() >= y_max || ray.p0.z() <= z_min || ray.p0.z() >= z_max
  ) {
    print_vec3(ray.p0);
    print_vec3(initial_address);
    printf("threadIdx.x = %d, threadIdx.y = %d, blockIdx.x = %d, blockIdx.y = %d\n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
  }

  Cell* initial_cell = cell_array[convert_3d_to_1d_cell_address(initial_address)];
  Cell* current_cell;
  float o_x, o_y, o_z, t_x_0, t_y_0, t_z_0, d_t_x, d_t_y, d_t_z, t_x, t_y, t_z;
  int i, j, k;
  bool hit = false;
  hit_record cur_rec;

  o_x = fmaxf(fminf(ray.p0.x(), x_max), x_min) - initial_cell -> get_bounding_box() -> x_min;
  o_y = fmaxf(fminf(ray.p0.y(), y_max), y_min) - initial_cell -> get_bounding_box() -> y_min;
  o_z = fmaxf(fminf(ray.p0.z(), z_max), z_min) - initial_cell -> get_bounding_box() -> z_min;

  if (ray.dir.x() < 0) {
    t_x_0 = o_x / fabsf(ray.dir.x());
  } else {
    t_x_0 = (cell_size_x - o_x) / fabsf(ray.dir.x());
  }
  d_t_x = cell_size_x / fabsf(ray.dir.x());

  if (ray.dir.y() < 0) {
    t_y_0 = o_y / fabsf(ray.dir.y());
  } else {
    t_y_0 = (cell_size_y - o_y) / fabsf(ray.dir.y());
  }
  d_t_y = cell_size_y / fabsf(ray.dir.y());

  if (ray.dir.z() < 0) {
    t_z_0 = o_z / fabsf(ray.dir.z());
  } else {
    t_z_0 = (cell_size_z - o_z) / fabsf(ray.dir.z());
  }
  d_t_z = cell_size_z / fabsf(ray.dir.z());

  t_x = t_x_0;
  t_y = t_y_0;
  t_z = t_z_0;
  i = initial_cell -> i_address;
  j = initial_cell -> j_address;
  k = initial_cell -> k_address;

  float inc_i = round(9999 * ray.dir.x() / fabsf(9999 * ray.dir.x()));
  float inc_j = round(9999 * ray.dir.y() / fabsf(9999 * ray.dir.y()));
  float inc_k = round(9999 * ray.dir.z() / fabsf(9999 * ray.dir.z()));

  float t_x_prev, t_y_prev, t_z_prev;
  int count = 0;

  while(
    i < n_cell_x && j < n_cell_y && k < n_cell_z && i >= 0 && j >= 0 && k >= 0
  ) {
    count++;

    current_cell = cell_array[convert_3d_to_1d_cell_address(i, j, k)];
    hit = this -> _grid_hit(
      ray, current_cell -> object_array, current_cell -> num_objects, rec);

    if (t_x <= t_y && t_x <= t_z && fabsf(ray.dir.x()) > 0) {
      t_x_prev = t_x;
      t_x += d_t_x;
      i += inc_i;
    } else {
      if (t_y <= t_x && t_y <= t_z && fabsf(ray.dir.y()) > 0) {
        t_y_prev = t_y;
        t_y += d_t_y;
        j += inc_j;
      } else {
        if (t_z <= t_x && t_z <= t_y && fabsf(ray.dir.z()) > 0) {
          t_z_prev = t_z;
          t_z += d_t_z;
          k += inc_k;
        } else {
          // printf(
          //   "count = %d, i = %d, j = %d, k = %d, \
          //   t_x_prev = %5.5f, t_y_prev = %5.5f, t_z_prev = %5.5f, \
          //   t_x = %5.5f, t_y = %5.5f, t_z = %5.5f, \
          //   d_t_x = %5.5f, d_t_y = %5.5f, d_t_z = %5.5f, \
          //   ray_dir_x = %5.5f, ray_dir_y = %5.5f, ray_dir_z = %5.5f\n",
          //   count, i, j, k,
          //   t_x_prev, t_y_prev, t_z_prev,
          //   t_x, t_y, t_z,
          //   d_t_x, d_t_y, d_t_z,
          //   ray.dir.x(), ray.dir.y(), ray.dir.z()
          // );
          return false;
        }
      }
    }

    if (hit) {
      if (current_cell -> get_bounding_box() -> is_inside(rec.point)) {
          // num_ray_intersections++;
          return true;
      }
    }
  }

  // return intersection_tuple_ref;

  return false;

}


#endif
