//File: material_list_operations.h
#ifndef MATERIAL_LIST_OPERATIONS_H
#define MATERIAL_LIST_OPERATIONS_H

#include "../model/material/material.h"

__device__ void add_new_material(
  Material** material_list, int &material_list_length, Material* material
) {
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

#endif
