//File: common.h
#ifndef COMMON_H
#define COMMON_H

#include "../../model/geometry/primitive.h"
#include "../../model/material/material.h"

__device__ bool check_if_entering_medium(
  hit_record rec, reflection_record ref, bool in_medium
) {
  return (
    !(ref.false_hit) &&
    ref.next_material != nullptr && 
    ref.next_material -> extinction_coef > 0
  ) || (
    ref.false_hit && in_medium
  );
}

#endif
