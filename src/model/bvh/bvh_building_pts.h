//File: bvh_building_pts.h
#ifndef BVH_BUILDING_PTS_H
#define BVH_BUILDING_PTS_H

__global__ void extract_sss_morton_code_list(
  Point** point_list, unsigned int* morton_code_list, int num_pts
);

__global__ void build_sss_pts_leaf_list(
  Node** leaf_list, Point** point_list, Object** object_list, int object_idx,
  int *pt_offset_array
);

__global__ void build_sss_pts_node_list(
  Node** node_list, Object** object_list, int object_idx
);

__global__ void set_pts_sss_node_relationship(
  Node** node_list, Node** leaf_list, unsigned int* morton_code_list,
  Object** object_list, int object_idx
);

__global__ void extract_sss_morton_code_list(
  Point** point_list, unsigned int* morton_code_list, int num_pts
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_pts) return;
  morton_code_list[idx] = point_list[idx] -> bounding_box -> morton_code;
}

__global__ void build_sss_pts_leaf_list(
  Node** leaf_list, Point** point_list, Object** object_list, int object_idx,
  int *pt_offset_array
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (object_list[object_idx] -> num_pts < 1) return;
  if (idx >= object_list[object_idx] -> num_pts) return;

  Node** effective_leaf_list = leaf_list + \
    object_list[object_idx] -> bvh_leaf_zero_idx;
  Point** effective_point_list = point_list + \
    pt_offset_array[object_idx];

  effective_leaf_list[idx] = new Node(idx);
  (effective_leaf_list[idx]) -> assign_point(effective_point_list[idx]);

}

__global__ void build_sss_pts_node_list(
  Node** node_list, Object** object_list, int object_idx
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  Node** effective_node_list = node_list + \
    object_list[object_idx] -> bvh_root_node_idx;
  if (object_list[object_idx] -> num_pts < 1) return;
  if (idx >= object_list[object_idx] -> num_pts - 1) return;
  effective_node_list[idx] = new Node(idx);
  (effective_node_list[idx]) -> bounding_box = new BoundingBox();
}

__global__ void set_pts_sss_node_relationship(
  Node** node_list, Node** leaf_list, unsigned int* morton_code_list,
  Object** object_list, int object_idx
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int num_pts = object_list[object_idx] -> num_pts;
  Node **effective_node_list = node_list + \
    object_list[object_idx] -> bvh_root_node_idx;
  Node **effective_leaf_list = leaf_list + \
    object_list[object_idx] -> bvh_leaf_zero_idx;

  if (idx >= (num_pts - 1)) return;

  // Determine direction of the range (+1 or -1)
  int d1 = length_longest_common_prefix(
    morton_code_list, idx, idx + 1, num_pts
  );
  int d2 = length_longest_common_prefix(
    morton_code_list, idx, idx - 1, num_pts
  );
  int d;
  if (d1 < d2) {
    d = -1;
  } else {
    d = 1;
  }

  int start = idx;

  // Determine end
  int end = start + d;
  int min_delta = length_longest_common_prefix(
    morton_code_list, idx, idx - d, num_pts
  );
  int current_delta;
  bool flag = TRUE;
  while (end >= 0 && end <= num_pts - 1 && flag) {
    current_delta = length_longest_common_prefix(
      morton_code_list, start, end, num_pts
    );
    if (current_delta > min_delta) {
      end += d;
    } else {
      end -= d;
      flag = FALSE;
    }
  }
  if (end < 0) end = 0;
  if (end > num_pts - 1) end = num_pts - 1;

  // Determine split
  int split = start + d;
  int idx_1 = start, idx_2 = start + d;
  int min_delta_2 = length_longest_common_prefix(
    morton_code_list, idx_1, idx_2, num_pts
  );
  while((d > 0 && idx_2 <= end) || (d < 0 && idx_2 >= end)) {
    current_delta = length_longest_common_prefix(
      morton_code_list, idx_1, idx_2, num_pts
    );
    if (current_delta < min_delta_2) {
      split = idx_2;
      min_delta_2 = current_delta;
    }
    idx_1 += d;
    idx_2 += d;
  }

  if (
    start < 0 || end < 0 || start >= num_pts || end >= num_pts ||
    split < 0 || split >= num_pts
  ) {
    printf("start = %d; split = %d; end = %d; d = %d\n", start, split, end, d);
  }

  // Determine children
  Node *left, *right;
  if (d > 0) {
    if (split - start > 1)
      left = effective_node_list[split - 1];
    else
      left = effective_leaf_list[start];
    if (end - split > 0)
      right = effective_node_list[split];
    else
      right = effective_leaf_list[end];
  } else {
    if (start - split > 1)
      right = effective_node_list[split + 1];
    else
      right = effective_leaf_list[start];
    if (split - end > 0)
      left = effective_node_list[split];
    else
      left = effective_leaf_list[end];
  }

  effective_node_list[idx] -> set_left_child(left);
  effective_node_list[idx] -> set_right_child(right);

  left -> set_parent(effective_node_list[idx]);
  right -> set_parent(effective_node_list[idx]);

}

#endif
