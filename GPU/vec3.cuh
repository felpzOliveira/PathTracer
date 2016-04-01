#pragma once
#define M_PI 3.141592f
#include "cuda_runtime.h"
#include <math.h>
#include "curand_kernel.h"

class vec3{
public:
	float e[3];
	__host__ __device__ vec3() { e[0] = 0; e[1] = 0; e[2] = 0; }
	__host__ __device__ vec3( float e0, float e1, float e2 ) { e[0] = e0; e[1] = e1; e[2] = e2; }
	
	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline const vec3& operator+() const { return *this; }
	__host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float& operator[](int i) { return e[i]; }

	__host__ __device__ inline vec3& operator+=( const vec3& v2);
	__host__ __device__ inline vec3& operator-=( const vec3& v2);
	__host__ __device__ inline vec3& operator*=( const vec3& v2);
	__host__ __device__ inline vec3& operator/=( const vec3& v2);
	__host__ __device__ inline vec3& operator*=( float t );
	__host__ __device__ inline vec3& operator/=( float t );

	__host__ __device__ inline float length() { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
	__host__ __device__ inline float squared_length() {return (e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
	__host__ __device__ inline void make_unit_vector();
};

__host__ __device__ inline void vec3::make_unit_vector()
{
	float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+( const vec3& v1, const vec3& v2 )
{
	 return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-( const vec3& v1, const vec3& v2 )
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
	 return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) 
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) 
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) 
{
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) 
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) 
{
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) 
{
    return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v)
{
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v)
{
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v)
{
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) 
{
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) 
{
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) 
{
    float k = 1.0/t;
    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 reflect( const vec3 &v, const vec3 &n )
{
	return v - 2.0*dot(n,v)*n;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) 
{
    return v / v.length();
}

__host__ __device__ inline bool refract( const vec3& v, const vec3 &n, float ni_over_nt, vec3 &refracted )
{
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float delta = 1.0 - ni_over_nt*ni_over_nt*(1.0 - dt*dt);
	if(delta > 0.0)
	{
		refracted = ni_over_nt*(uv - n*dt)-n*sqrt(delta);
		return true;
	}
	return false;
}

__host__ __device__ inline float schlick( float cosine, float ref_idx )
{
	float r0 = (1.0 - ref_idx)/(1.0 + ref_idx);
	r0 = r0*r0;
	return r0 + (1.0 - r0)*pow((1.0 - cosine), 5);
}

__device__ inline vec3 random_in_unit_sphere( curandState state )
{
	vec3 p;
	do{
		p = 2.0*vec3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state)) - vec3(1,1,1);
	}while(dot(p,p) >= 1.0);
	return p;
}

__device__ inline vec3 random_in_unit_disk( curandState state )
{
	vec3 p;
	do{
		p = 2.0*vec3(curand_uniform(&state), curand_uniform(&state), 0) - vec3(1,1,0);
	}while(dot(p,p) >= 1.0);
	return p;
}
