#pragma once
#include "ray.h"

class material;

struct hit_record{
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class hitable{
public:
	virtual bool hit( const ray &r, float t_min, float t_max, hit_record &rec ) const = 0;
};


class sphere : public hitable{
public:
	vec3 center;
	float radius;
	material *mat;
	sphere() {}
	sphere( vec3 cen, float r, material *m ) : center(cen), radius(r), mat(m) {}
	virtual bool hit( const ray &r, float t_min, float t_max, hit_record &rec ) const;
};

bool sphere::hit( const ray &r, float t_min, float t_max, hit_record &rec ) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float delta = b*b - a*c;
	if(delta > 0.0)
	{
		float temp = (-b - sqrt(delta))/a;
		if(temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center)/radius;
			rec.mat_ptr = mat;
			return true;
		}
		temp = (-b + sqrt(delta))/a;
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