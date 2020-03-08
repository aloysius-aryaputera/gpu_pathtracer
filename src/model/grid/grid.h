//File: grid.h
#ifndef GRID_H
#define GRID_H

#include <cuda_fp16.h>
#include <math.h>

#include "../../param.h"
#include "../camera.h"
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
    BoundingBox *world_bounding_box;

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

  printf("The grid details are:\n");
  printf("x min = %5.5f; x max = %5.5f, d x = %5.5f, number of cells = %d\n",
    grid -> x_min, grid -> x_max, d_x, grid -> n_cell_x);
  printf("y min = %5.5f; y max = %5.5f, d y = %5.5f, number of cells = %d\n",
    grid -> y_min, grid -> y_max, d_y, grid -> n_cell_y);
  printf("z min = %5.5f; z max = %5.5f, d z = %5.5f, number of cells = %d\n",
    grid -> z_min, grid -> z_max, d_z, grid -> n_cell_z);
  printf("\n");
}

__device__ void _compute_scene_boundaries(
  float &x_min, float &x_max, float &y_min, float &y_max, float &z_min,
  float &z_max, Primitive **geom_array, int num_objects, Camera *camera
) {

  x_min = INFINITY;
  x_max = -INFINITY;
  y_min = INFINITY;
  y_max = -INFINITY;
  z_min = INFINITY;
  z_max = -INFINITY;

  for (int i = 0; i < num_objects; i++) {
    x_min = min(x_min, geom_array[i] -> get_bounding_box() -> x_min);
    x_max = max(x_max, geom_array[i] -> get_bounding_box() -> x_max);
    y_min = min(y_min, geom_array[i] -> get_bounding_box() -> y_min);
    y_max = max(y_max, geom_array[i] -> get_bounding_box() -> y_max);
    z_min = min(z_min, geom_array[i] -> get_bounding_box() -> z_min);
    z_max = max(z_max, geom_array[i] -> get_bounding_box() -> z_max);
  }

  x_min -= SMALL_DOUBLE;
  x_max += SMALL_DOUBLE;
  y_min -= SMALL_DOUBLE;
  y_max += SMALL_DOUBLE;
  z_min -= SMALL_DOUBLE;
  z_max += SMALL_DOUBLE;

}

__global__ void compute_world_bounding_box(
  BoundingBox **world_bounding_box, Primitive **geom_array, int num_objects
) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {

    float x_min = INFINITY;
    float x_max = -INFINITY;
    float y_min = INFINITY;
    float y_max = -INFINITY;
    float z_min = INFINITY;
    float z_max = -INFINITY;

    for (int i = 0; i < num_objects; i++) {
      x_min = min(x_min, geom_array[i] -> get_bounding_box() -> x_min);
      x_max = max(x_max, geom_array[i] -> get_bounding_box() -> x_max);
      y_min = min(y_min, geom_array[i] -> get_bounding_box() -> y_min);
      y_max = max(y_max, geom_array[i] -> get_bounding_box() -> y_max);
      z_min = min(z_min, geom_array[i] -> get_bounding_box() -> z_min);
      z_max = max(z_max, geom_array[i] -> get_bounding_box() -> z_max);
    }

    x_min -= SMALL_DOUBLE;
    x_max += SMALL_DOUBLE;
    y_min -= SMALL_DOUBLE;
    y_max += SMALL_DOUBLE;
    z_min -= SMALL_DOUBLE;
    z_max += SMALL_DOUBLE;

    world_bounding_box[0] = new BoundingBox(
      x_min, x_max, y_min, y_max, z_min, z_max);
  }

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
  return k + j * this -> n_cell_z + i * this -> n_cell_z * this -> n_cell_y;
}

__device__ int Grid::convert_3d_to_1d_cell_address(vec3 address_3d) {
  int k = floorf(fminf(address_3d.z(), this -> n_cell_z - 1));
  int j = floorf(fminf(address_3d.y(), this -> n_cell_y - 1));
  int i = floorf(fminf(address_3d.x(), this -> n_cell_x - 1));
  return k + j * this -> n_cell_z + i * this -> n_cell_z * this -> n_cell_y;
}

__device__ Grid::Grid(
  float x_min_, float x_max_, float y_min_, float y_max_, float z_min_,
  float z_max_, int n_cell_x_, int n_cell_y_, int n_cell_z_,
  Primitive** object_array_, int num_objects_, Cell** cell_array_,
  int max_num_objects_per_cell_
) {
  this -> x_min = x_min_;
  this -> x_max = x_max_;
  this -> y_min = y_min_;
  this -> y_max = y_max_;
  this -> z_min = z_min_;
  this -> z_max = z_max_;
  this -> object_array = object_array_;
  this -> num_objects = num_objects_;
  this -> cell_array = cell_array_;
  this -> max_num_objects_per_cell = max_num_objects_per_cell_;

  this -> n_cell_x = n_cell_x_;
  this -> n_cell_y = n_cell_y_;
  this -> n_cell_z = n_cell_z_;

  this -> cell_size_x = (this -> x_max - this -> x_min) / this -> n_cell_x;
  this -> cell_size_y = (this -> y_max - this -> y_min) / this -> n_cell_y;
  this -> cell_size_z = (this -> z_max - this -> z_min) / this -> n_cell_z;

  this -> world_bounding_box = new BoundingBox(
    this -> x_min, this -> x_max, this -> y_min, this -> y_max,
    this -> z_min, this -> z_max);
}

__device__ bool Grid::_is_inside(vec3 position) {
  return this -> world_bounding_box -> is_inside(position);
}

__device__ vec3 Grid::find_cell_address(vec3 position) {
  int cell_address_i, cell_address_j, cell_address_k;
  if (!(this -> _is_inside(position))) {
    printf("The position is not inside the grid!");
  }
  cell_address_i = floorf(
    (position.x() - this -> x_min) / this -> cell_size_x);
  cell_address_j = floorf(
    (position.y() - this -> y_min) / this -> cell_size_y);
  cell_address_k = floorf(
    (position.z() - this -> z_min) / this -> cell_size_z);
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

  if (!(this -> _is_inside(ray.p0))) {
    float t;
    bool ray_intersect = this -> world_bounding_box -> is_intersection(ray, t);
    if (ray_intersect && t >= 0) {
      vec3 start_vec = ray.get_vector(t + SMALL_DOUBLE);
      float start_vec_x = min(max(this -> x_min, start_vec.x()), this -> x_max);
      float start_vec_y = min(max(this -> y_min, start_vec.y()), this -> y_max);
      float start_vec_z = min(max(this -> z_min, start_vec.z()), this -> z_max);
      start_vec = vec3(start_vec_x, start_vec_y, start_vec_z);
      ray = Ray(start_vec, ray.dir);
    } else {
      return false;
    }
  }

  vec3 initial_address = find_cell_address(ray.p0);
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
