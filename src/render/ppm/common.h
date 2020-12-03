//File: common.h
#ifndef COMMON_H
#define COMMON_H

#include "../../model/geometry/primitive.h"
#include "../../model/material/material.h"

__device__ bool check_if_entering_medium(
  reflection_record ref, bool in_medium, Material* &medium
) {
  bool entering_medium = (
    !(ref.false_hit) &&
    ref.next_material != nullptr && 
    ref.next_material -> extinction_coef > SMALL_DOUBLE
  ) || (
    ref.false_hit && in_medium
  );

  if (entering_medium)
    medium = ref.next_material;

  return entering_medium;
}

__device__ bool check_if_inside_medium(
  Material** material_list, int material_list_length, Material* &medium
) {
 
  Material *second_material;

  find_highest_prioritised_materials(
    material_list, material_list_length, medium, second_material
  );

  if (medium != nullptr && medium -> extinction_coef > SMALL_DOUBLE) {
    return true;
  }
  return false;
}

#endif
