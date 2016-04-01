#pragma once
#include "hitable.h"

typedef struct Hit_list{
	Sphere **sphere_list;
	int sphere_list_size;

	__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
	{
		hit_record temp_rec;
		bool hit_anything = false;
		double closest_so_far = t_max;
		for(int i = 0; i < sphere_list_size; i++)
		{
			if(sphere_list[i]->hit(r, t_min, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

}Hitable_list;
