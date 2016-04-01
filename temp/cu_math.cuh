#pragma once
#define M_PI 3.141592f
#include "cuda_runtime.h"
#include <math.h>
#include "curand_kernel.h"

class vec3{
public:
	float e[3];
	__host__ __device__ vec3() { e[0] = 0; e[1] = 0; e[2] = 0; }
	__host__ __device__ vec3( int e0, int e1, int e2 ) {e[0] = float(e0); e[1] = float(e1); e[2] = float(e2); }
	__host__ __device__ vec3( float e0, float e1, float e2 ) { e[0] = e0; e[1] = e1; e[2] = e2; }
	__host__ __device__ vec3( double e0, double e1, double e2 ){e[0] = float(e0); e[1] = float(e1); e[2] = float(e2); }
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
	float k = 1.0f / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
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
    float k = 1.0f/t;
    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 reflect( const vec3 &v, const vec3 &n )
{
	return v - 2.0f*dot(n,v)*n;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) 
{
    return v / v.length();
}

__host__ __device__ vec3 rotate_y( vec3 v, float angle )
{
	float rad = M_PI/180.0f * angle;
	float s = sin(rad);
	float c = cos(rad);
	return vec3(c*v[0] + s*v[2], v[1], -s*v[0] + c*v[2]);
}

__host__ __device__ inline float fract( float x ){ return x - floor(x); }

__host__ __device__ inline bool refract( const vec3& v, const vec3 &n, float ni_over_nt, vec3 &refracted )
{
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float delta = 1.0f - ni_over_nt*ni_over_nt*(1.0f - dt*dt);
	if(delta > 0.0f)
	{
		refracted = ni_over_nt*(uv - n*dt)-n*sqrt(delta);
		return true;
	}
	return false;
}

__host__ __device__ inline float schlick( float cosine, float ref_idx )
{
	float r0 = (1.0f - ref_idx)/(1.0f + ref_idx);
	r0 = r0*r0;
	return r0 + (1.0f - r0)*pow((1.0f - cosine), 5);
}

__device__ inline vec3 random_in_unit_sphere( curandState *state )
{
	vec3 p;
	do{
		p = 2.0*vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) - vec3(1,1,1);
	}while(dot(p,p) >= 1.0f);
	return p;
}

#define uint32 unsigned int
__host__ __device__ inline uint32 WangHash( uint32 a ) 
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

__device__ inline vec3 random_in_unit_disk( curandState *state )
{
	vec3 p;
	do{
		p = 2.0f*vec3(curand_uniform(state), curand_uniform(state), 0.0f) - vec3(1,1,0);
	}while(dot(p,p) >= 1.0f);
	return p;
}

__device__ inline vec3 random_cosine_direction( curandState *state )
{
	float r1 = curand_uniform(state);
	float r2 = curand_uniform(state);
	float z = sqrt(1 - r2);
	float phi = 2.0f*M_PI*r1;
	float x = cos(phi)*2*sqrt(r2);
	float y = sin(phi)*2*sqrt(r2);
	return vec3(x, y, z);
}

__device__ float getrandom( curandState *state )
{
	float rd = curand_uniform(state);
	if(rd == 1.0){
		float rd2 = curand_uniform(state);
		while(rd2 == 0)
			rd2 = curand_uniform(state);
		
		rd -= 0.001*rd2;
	}
	return rd;
}

__device__ inline vec3 random_to_sphere( float radius, float distance_squared, curandState *state )
{
	float r1 = getrandom(state);
	float r2 = getrandom(state);
	float z = 1.0f + r2*(sqrt(1.0f - radius*radius/distance_squared) - 1.0f);
	float phi = 2.0f*M_PI*r1;
	float x = cos(phi)*sqrt(1.0f - z*z);
	float y = sin(phi)*sqrt(1.0f - z*z);
	return vec3(x,y,z);
}

template<typename T>
__host__ __device__ inline T fmax( T a, T b ) { return (a > b ? a : b); }

template<typename T>
__host__ __device__ inline T fmin( T a, T b ) { return ( a < b ? a : b); }

class onb{
public:
	vec3 axis[3];
	__host__ __device__ onb(){}
	__host__ __device__ inline vec3 operator[](int i ) const{ return axis[i]; }
	__host__ __device__ vec3 u() const { return axis[0]; }
	__host__ __device__ vec3 v() const { return axis[1]; }
	__host__ __device__ vec3 w() const { return axis[2]; }
	__host__ __device__ vec3 local(float a, float b, float c) const { return a*u() + b*v() + c*w(); }
	__host__ __device__ vec3 local(const vec3& a) const { return a.x()*u() + a.y()*v() + a.z()*w(); }
	__host__ __device__ void build_from_w( const vec3&);
};

__host__ __device__ void onb::build_from_w( const vec3& n )
{
	axis[2] = unit_vector(n);
	vec3 a;
	if(fabs(w().x()) > 0.9)
		a = vec3(0, 1, 0);
	else
		a = vec3(1, 0, 0);
	axis[1] = unit_vector(cross(w(), a));
	axis[0] = cross(w(), v());
}
