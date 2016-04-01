#pragma once
#include "cu_math.cuh"
#include "util.cuh"
class Ray{
private:
	vec3 or, dir;
public:
	__host__ __device__ Ray( vec3 a, vec3 b ) { set(a,b); }
	__host__ __device__ void set( vec3 a, vec3 b ) { or = a; dir = b; }
	__host__ __device__ vec3 origin() { return or; }
	__host__ __device__ vec3 direction() { return dir; }
	__host__ __device__ vec3 point_at( float t ) { return or + t * dir; }
};

class Camera : public Managed{
private:
	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
	float aspectR;
	bool simplify;
public:
	__host__ __device__ Camera( vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
		float aspect, float aperture, float focus_dist )
	{
		lens_radius = aperture/2.0f;
        float theta = vfov*M_PI/180.0f;
        float half_height = tan(theta/2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2*half_width*focus_dist*u;
        vertical = 2*half_height*focus_dist*v;
		aspectR = aspect;
		simplify = false;
	}

	__host__ __device__ Camera( vec3 lookfrom, vec3 lookat, vec3 vup, float aspect )
	{
		origin = lookfrom;
		w = unit_vector(lookat - origin);
		u = unit_vector(cross(w, vup));
		v = unit_vector(cross(u, w));
		aspectR = aspect;
		simplify = true;
	}
	
	__host__ __device__ Ray get_ray( float s, float t ) 
	{
		if(simplify)
		{
			float ss = 2.0f*s - 1.0f;
			float tt = 2.0f*t - 1.0f;
			ss *= aspectR;
			return Ray(origin, unit_vector(2.0f*w + ss*u + tt*v));
		}
		else
			return Ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
	}
};