//File: material_list_operations.h
#ifndef MATERIAL_LIST_OPERATIONS_H
#define MATERIAL_LIST_OPERATIONS_H

#include "../model/bvh/bvh.h"
#include "../model/material/material.h"
#include "../model/vector_and_matrix/vec3.h"

__device__ void add_new_material(
  Material** material_list, int &material_list_length, Material* material
) {
  if (is_material_inside(material_list, material_list_length, material)) {
    return;
  }
  material_list[material_list_length] = material;
  material_list_length++;
}

__device__ void remove_a_material(
  Material** material_list, int &material_list_length, Material* material
) {
  int selected_idx, idx = material_list_length - 1;
  bool material_found = false;

  while (!material_found && idx >= 0) {
    if (material_list[idx] == material) {
      material_found = true;
      selected_idx = idx;
    }
    idx--;
  }

  if (material_found) {
    for (idx = selected_idx; idx < material_list_length - 1; idx++) {
      material_list[idx] = material_list[idx + 1];
    }
    material_list[material_list_length - 1] = nullptr;
    material_list_length--;
  }

}

__device__ void rearrange_material_list(
  Material** material_list, int &material_list_length, Material* material,
  bool false_hit, bool entering, bool refracted 
) {
  if (false_hit && entering)
    add_new_material(material_list, material_list_length, material);
  
  if (false_hit && !entering)
    remove_a_material(material_list, material_list_length, material);
  
  if (!false_hit && refracted && entering)
    add_new_material(material_list, material_list_length, material);
  
  if (!false_hit && refracted && !entering)
    remove_a_material(material_list, material_list_length, material);
}

__device__ void init_material_list(
  Material** material_list, int& material_list_length,
  Node** transparent_geom_node_list, vec3 init_point, vec3 init_dir,
  curandState *rand_state
) {
  float t = 99999;
  Ray ray = Ray(init_point + t * init_dir, -init_dir);
  hit_record rec;
  reflection_record ref;
  bool force_refract = true, sss;
  bool hit = traverse_bvh(transparent_geom_node_list[0], ray, rec);

  while (hit && t - rec.t > SMALL_DOUBLE) {
    t -= rec.t;

    rec.object -> get_material() -> check_next_path(
      rec.coming_ray, rec.point, rec.normal, rec.uv_vector,
      sss, material_list, material_list_length, ref, rand_state, false,
      force_refract
    );

    rearrange_material_list(
      material_list, material_list_length, rec.object -> get_material(),
      ref.false_hit, ref.entering, ref.refracted 
    );

    ray = Ray(rec.point, -init_dir);

    hit = traverse_bvh(transparent_geom_node_list[0], ray, rec);

  }

}

#endif
