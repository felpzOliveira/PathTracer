#pragma once
#include "util.cuh"
#include "cu_math.cuh"
#include "camera.cuh"
#include "noise.cuh"

class Material;

struct hit_record{
	float t;
	vec3 normal;
	vec3 point;
	Material *mat;
};

enum MATERIAL_TYPE{
	DIFFUSE, SPECULAR, REFRACTIVE, LIGHT
};

enum TEXTURE_TYPE{
	CONSTANT, CHECKER
};

class Texture : public Managed{
private:
	vec3 m_color;
	vec3 m_checker0;
	vec3 m_checker1;
	TEXTURE_TYPE m_type;
	NoiseEngine *n_engine;
public:
	__host__ Texture( TEXTURE_TYPE type )
	{ 
		m_type = type; m_checker0 = vec3(0.98,0.98,0.98);
		m_checker1 = vec3(0.01,0.98,0.01); m_color = vec3(0,0,0); n_engine = NULL;
	}

	__host__ Texture( TEXTURE_TYPE type, vec3 color ) 
	{ 
		m_type = type; m_color = color; 
		m_checker0 = vec3(0.98,0.98,0.98); m_checker1 = vec3(0.01,0.98,0.01); n_engine = NULL;
	}

	__host__ Texture( TEXTURE_TYPE type, NOISE_ENGINE n_type, vec3 color ) 
	{ 
		m_type = type; m_color = color; 
		m_checker0 = vec3(0.98,0.98,0.98); m_checker1 = vec3(0.01,0.98,0.01);
		n_engine = new NoiseEngine(n_type);
	}

	__host__ __device__ Texture( TEXTURE_TYPE type, vec3 checker0, vec3 checker1 )
	{ m_type = type; m_checker0 = checker0; m_checker1 = checker1; m_color = vec3(0,0,0); n_engine = NULL; }

	__host__ __device__ vec3 GetValue( hit_record rec )
	{
		vec3 cl(0,0,0);
		if(m_type == CONSTANT) cl = m_color;
		else if(m_type == CHECKER) cl = checker_pattern(rec);
		if(n_engine != NULL) cl  = turbulance_pattern(rec);
		return cl;
	}
private:

	__host__ __device__ vec3 turbulance_pattern( hit_record rec )
	{
		return m_color * (0.2 + n_engine->turbulence(10*rec.point));
	}

	__host__ __device__ vec3 checker_pattern( hit_record rec )
	{
		vec3 p = rec.point;
		float sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
		if(sines < 0) return m_checker0;
		else return m_checker1;
	}
};

class Material : public Managed{
private:
	Texture *m_texture;
	MATERIAL_TYPE m_type;
	float m_fuzz;
	float m_ref_index;
public:

	__host__ Material( float ref_indx, MATERIAL_TYPE type )
	{ m_texture = new Texture(CONSTANT, vec3(1,1,1)); m_ref_index = ref_indx; m_type = type; }

	__host__ Material( Texture *tex, float fuz, MATERIAL_TYPE type ) 
	{ m_texture = tex; m_type = type; m_fuzz = fuz; }

	__host__ Material( Texture *tex, MATERIAL_TYPE type )
	{ m_texture = tex; m_type = type; m_fuzz = 0; }

	__device__ vec3 Emitted( hit_record rec )
	{
		if(m_type != LIGHT) return vec3(0,0,0);
		return m_texture->GetValue(rec);
	}
	
	__device__ bool Scatter( Ray r_in, hit_record rec, vec3 &attenuation,
										Ray &scattered, curandState *state )
	{
		if(m_type == DIFFUSE)
			return Scatter_diffuse(r_in, rec, attenuation, scattered, state);
		else if(m_type == SPECULAR)
			return Scatter_specular(r_in, rec, attenuation, scattered, state);
		else if(m_type == REFRACTIVE)
			return Scatter_refractive(r_in, rec, attenuation, scattered, state);
		else //LIGHT
			return false;
	}
private:
	__device__ bool Scatter_diffuse( Ray r_in, hit_record rec, vec3 &attenuation,
												Ray &scattered, curandState *state )
	{
		vec3 target = rec.point + rec.normal + random_in_unit_sphere(state);
		scattered = Ray(rec.point, target - rec.point);
		attenuation = m_texture->GetValue(rec);
		return true;
	}

	__device__ bool Scatter_specular( Ray r_in, hit_record rec, vec3 &attenuation,
												Ray &scattered, curandState *state )
	{
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		scattered = Ray(rec.point, unit_vector(reflected) + m_fuzz*random_in_unit_sphere(state));
		attenuation = m_texture->GetValue(rec);
		return (dot(scattered.direction(), rec.normal) > 0);
	}

	__device__ bool Scatter_refractive( Ray r_in, hit_record rec, vec3 &attenuation,
												Ray &scattered, curandState *state )
	{
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		attenuation = m_texture->GetValue(rec);
		vec3 refracted;
		float reflect_prob;
		float cosine;
		if(dot(r_in.direction(), rec.normal) > 0)
		{
			outward_normal = -rec.normal;
			ni_over_nt = m_ref_index;
			cosine = m_ref_index * dot(r_in.direction(), rec.normal)/(r_in.direction().length());
		}
		else
		{
			outward_normal = rec.normal;
			ni_over_nt = 1.0/m_ref_index;
			cosine = -dot(r_in.direction(), rec.normal)/(r_in.direction().length());
		}
		if(refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
		{
			reflect_prob = schlick(cosine, m_ref_index);
		}
		else{
			scattered = Ray(rec.point, reflected);
			reflect_prob = 1.0;
		}
		if(getrandom(state) < reflect_prob)
		{
			scattered =	Ray(rec.point, reflected);
		}
		else
			scattered = Ray(rec.point, refracted);
		return true;	
	}
};