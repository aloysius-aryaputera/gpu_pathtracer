//File: bounding_cone.h
#ifndef BOUNDING_CONE_H
#define BOUNDING_CONE_H

#include "../vector_and_matrix/vec3.h"

class BoundingCone {
  public:
    __host__ __device__ BoundingCone();
    __host__ __device__ BoundingCone(
			vec3 axis_, float theta_0_, float theta_e_
		);
    __host__ __device__ void initialize(
			vec3 axis_, float theta_0_, float theta_e_
		);

		float theta_0, theta_e;
		vec3 axis;
		bool initialized;
};

__host__ __device__ BoundingCone::BoundingCone() {
	this -> initialized = false;
}

__host__ __device__ BoundingCone::BoundingCone(
	vec3 axis_, float theta_0_, float theta_e_
) {
	this -> axis = axis_;
	this -> theta_0 = theta_0_;
	this -> theta_e = theta_e_;
	this -> initialized = true;
}

__host__ __device__ void BoundingCone::initialize(
	vec3 axis_, float theta_0_, float theta_e_
) {
	this -> axis = axis_;
	this -> theta_0 = theta_0_;
	this -> theta_e = theta_e_;
	this -> initialized = true;
}

__device__ void cone_union(
	BoundingCone* cone_1, BoundingCone* cone_2, vec3 &new_axis, 
	float &new_theta_0, float &new_theta_e
);

__device__ void cone_union(
	BoundingCone* cone_1, BoundingCone* cone_2, vec3 &new_axis,
	float &new_theta_0, float &new_theta_e
) {
	BoundingCone* cone_a = cone_1;
	BoundingCone* cone_b = cone_2;
	float theta_d;

	if (cone_b -> theta_0 > cone_a -> theta_0) {
		BoundingCone* temp;
		temp = cone_b;
		cone_b = cone_a;
		cone_a = temp;
	}

	theta_d = acos(dot(cone_a -> axis, cone_b -> axis));
	new_theta_e = fmaxf(cone_a -> theta_e, cone_b -> theta_e);

	if (fminf(theta_d + cone_b -> theta_0, M_PI) <= cone_a -> theta_0) {
		new_axis = cone_a -> axis;
		new_theta_0 = cone_a -> theta_0;
		return;
	} else {
		new_theta_0 = (cone_a -> theta_0 + theta_d + cone_b -> theta_0) / 2;
		if (M_PI <= new_theta_0) {
			new_axis = cone_a -> axis;
			new_theta_0 = M_PI;
			return;
		}

		float theta_r = new_theta_0 - cone_a -> theta_0;
		new_axis = rotate(
			cone_a -> axis, cross(cone_a -> axis, cone_b -> axis), theta_r
		);
		return;
	}
}

#endif
