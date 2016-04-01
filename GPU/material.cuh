#pragma once
#include "cuda_runtime.h"
#include "hitable.h"

/* 0 = lambertian, 1 = metal, 2 = dieletric*/
typedef struct Mat{
	int type;
	vec3 albedo;
	float fuzz;
	float ref_idx;

	__host__ __device__ void set( int tp, vec3 a, float f, float rf )
	{
		type = tp; albedo = a; fuzz = f; ref_idx = rf;
	}

	__device__ bool scatter_lambertian( const ray &r_in, const hit_record& rec,
										vec3 &attenuation, ray& scattered, curandState state )
	{
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(state);
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}

	__device__ bool scatter_metal( const ray &r_in, const hit_record& rec,
										vec3 &attenuation, ray& scattered, curandState state )
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	__device__ bool scatter_dieletric( const ray &r_in, const hit_record& rec,
										vec3 &attenuation, ray& scattered, curandState state )
	{
		vec3 outward_normal;
			vec3 reflected = reflect(r_in.direction(), rec.normal);
			float ni_over_nt;
			attenuation = vec3(1.0, 1.0, 1.0);
			vec3 refracted;
			float reflect_prob;
			float cosine;
			if(dot(r_in.direction(), rec.normal) > 0)
			{
				outward_normal = -rec.normal;
				ni_over_nt = ref_idx;
				cosine = ref_idx * dot(r_in.direction(), rec.normal)/(r_in.direction().length());
			}
			else
			{
				outward_normal = rec.normal;
				ni_over_nt = 1.0/ref_idx;
				cosine = -dot(r_in.direction(), rec.normal)/(r_in.direction().length());
			}
			if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
			{
				reflect_prob = schlick(cosine, ref_idx);
			}
			else{
				scattered = ray(rec.p, reflected);
				reflect_prob = 1.0;
			}
			if(curand_uniform(&state) < reflect_prob)
			{
				scattered =	ray(rec.p, reflected);
			}
			else
				scattered = ray(rec.p, refracted);
			return true;	
	}

}Material;

