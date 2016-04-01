#pragma once
#include "ray.cuh"

struct Mat;

struct hit_record{
	float t;
	vec3 p;
	vec3 normal;
	struct Mat *mat_ptr;
};

typedef struct Sph{
	vec3 center;
	float radius;
	struct Mat *mat;
	__host__ __device__ void set( vec3 cen, float r, struct Mat *m ) 
	{
		center = cen; radius = r; mat = m;
	}
	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
	{
		vec3 oc = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = dot(oc, r.direction());
		float c = dot(oc, oc) - radius*radius;
		float delta = b*b - a*c;
		if(delta > 0)
		{
			float temp = (-b-sqrt(delta))/a;
			if(temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center)/radius;
				rec.mat_ptr = mat;
				return true;
			}
			temp = (-b+sqrt(delta))/a;
			if(temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center)/radius;
				rec.mat_ptr = mat;
				return true;
			}
		}
		return false;
	}
}Sphere;
