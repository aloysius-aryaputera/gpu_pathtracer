//File: bvh_building_photon.h
#ifndef BVH_BUILDING_PHOTON_H
#define BVH_BUILDING_PHOTON_H

__global__ void init_photon_leaves(Node** leaf_list, int num_photons) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_photons) return;
  leaf_list[idx] = new Node(idx);
}

__global__ void init_photon_nodes(Node** node_list, int num_photons) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_photons - 1) return;
  node_list[idx] = new Node(idx);
  node_list[idx] -> bounding_box = new BoundingBox();
}

__global__ void reset_photon_nodes(Node** node_list, int num_photons) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_photons - 1) return;
  node_list[idx] -> bounding_box -> reset();
}

__global__ void assign_photons(
  Node** leaf_list, Point** photon_list, int num_recorded_photons
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num_recorded_photons) return;
  leaf_list[idx] -> assign_point(photon_list[idx]);
}

__global__
void compute_photon_morton_code_batch(
  Point** photon_list, int num_recorded_photons,
  BoundingBox** world_bounding_box
) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i > num_recorded_photons) return;

  photon_list[i] -> bounding_box -> compute_normalized_center(
    world_bounding_box[0]
  );
  photon_list[i] -> bounding_box -> compute_bb_morton_3d();
}	

__global__ void set_photon_node_relationship(
  Node** node_list, Node** leaf_list, unsigned int* morton_code_list,
  int num_recorded_photons
) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= (num_recorded_photons - 1)) return;

  // Determine direction of the range (+1 or -1)
  int d1 = length_longest_common_prefix(
    morton_code_list, idx, idx + 1, num_recorded_photons
  );
  int d2 = length_longest_common_prefix(
    morton_code_list, idx, idx - 1, num_recorded_photons
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
    morton_code_list, idx, idx - d, num_recorded_photons
  );
  int current_delta;
  bool flag = TRUE;
  while (end >= 0 && end <= num_recorded_photons - 1 && flag) {
    current_delta = length_longest_common_prefix(
      morton_code_list, start, end, num_recorded_photons
    );
    if (current_delta > min_delta) {
      end += d;
    } else {
      end -= d;
      flag = FALSE;
    }
  }
  if (end < 0) end = 0;
  if (end > num_recorded_photons - 1) end = num_recorded_photons - 1;

  // Determine split
  int split = start + d;
  int idx_1 = start, idx_2 = start + d;
  int min_delta_2 = length_longest_common_prefix(
    morton_code_list, idx_1, idx_2, num_recorded_photons
  );
  while((d > 0 && idx_2 <= end) || (d < 0 && idx_2 >= end)) {
    current_delta = length_longest_common_prefix(
      morton_code_list, idx_1, idx_2, num_recorded_photons
    );
    if (current_delta < min_delta_2) {
      split = idx_2;
      min_delta_2 = current_delta;
    }
    idx_1 += d;
    idx_2 += d;
  }

  if (
    start < 0 || end < 0 || start >= num_recorded_photons || 
    end >= num_recorded_photons || split < 0 || 
    split >= num_recorded_photons
  ) {
    printf("start = %d; split = %d; end = %d; d = %d\n", start, split, end, d);
  }

  // Determine children
  Node *left, *right;
  if (d > 0) {
    if (split - start > 1)
      left = node_list[split - 1];
    else
      left = leaf_list[start];
    if (end - split > 0)
      right = node_list[split];
    else
      right = leaf_list[end];
  } else {
    if (start - split > 1)
      right = node_list[split + 1];
    else
      right = leaf_list[start];
    if (split - end > 0)
      left = node_list[split];
    else
      left = leaf_list[end];
  }

  node_list[idx] -> set_left_child(left);
  node_list[idx] -> set_right_child(right);

  left -> set_parent(node_list[idx]);
  right -> set_parent(node_list[idx]);

}

#endif
