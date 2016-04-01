#pragma once
#include "cu_math.cuh"
#include "util.cuh"
#include "rand.cuh"

class perlin : public Managed{
private:
		vec3 *ranvec;
		int *perm_x;
		int *perm_y;
		int *perm_z;
public:
		__host__ perlin( void )
		{
			vec3 *rvec = perlin_generate();
			int *p_x = perlin_generate_perm();
			int *p_y = perlin_generate_perm();
			int *p_z = perlin_generate_perm();

			CHECK(cudaMalloc((void**)&ranvec, sizeof(vec3)*256));
			CHECK(cudaMalloc((void**)&perm_x, sizeof(int)*256));
			CHECK(cudaMalloc((void**)&perm_y, sizeof(int)*256));
			CHECK(cudaMalloc((void**)&perm_z, sizeof(int)*256));

			CHECK(cudaMemcpy(ranvec, rvec, 256*sizeof(vec3), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(perm_x, p_x, 256*sizeof(int), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(perm_y, p_y, 256*sizeof(int), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(perm_z, p_z, 256*sizeof(int), cudaMemcpyHostToDevice));
			free(p_x); free(p_y); free(p_z); free(rvec);
		}

		__host__ __device__ float noise( vec3 p )
		{
			float u = fract(p.x());
			float v = fract(p.y());
			float w = fract(p.z());

			float i = floor(p.x());
			float j = floor(p.y());
			float k = floor(p.z());
			vec3 e[2][2][2];
			for(int di = 0; di < 2; di++)
				for(int dj = 0; dj < 2; dj++)
					for(int dk = 0; dk < 2; dk++)
						e[di][dj][dk] = ranvec[perm_x[int(i+di) & 255] ^
												 perm_y[int(j+dj) & 255] ^
												 perm_z[int(k+dk) & 255]];
			return trilinear_interp(e, u, v, w);
		}
private:
		
		__host__ __device__ float trilinear_interp( vec3 e[2][2][2], float u, float v, float w )
		{
			float uu = u*u*(3-2*u), vv = v*v*(3-2*v), ww = w*w*(3-2*w);
			float accum = 0;
			for(int i = 0; i < 2; i++)
				for(int j = 0; j < 2; j++)
					for(int k = 0; k < 2; k++)
					{
						vec3 weight_v(u-i, v-j, w-k);
						accum += (i*uu + (1-i)*(1-uu))*
								 (j*vv + (1-j)*(1-vv))*
								 (k*ww + (1-k)*(1-ww))*dot(e[i][j][k], weight_v);
					}
			return accum;
		}

		__host__ int * perlin_generate_perm( void )
		{
			int *p = new int[256];
			for(int i = 0; i < 256; i++)
				p[i] = i;
			permute(p, 256);
			return p;
		}
		
		__host__ vec3 * perlin_generate( void )
		{
			vec3 *p = new vec3[256];
			for(int i = 0; i < 256; i++)
				p[i] = unit_vector(vec3(1.0f - 2.0f*drand48(),
										-1.0f + 2.0f*drand48(),
										-1.0f + 2.0f*drand48()));
			return p;
		}

		__host__ void permute( int *p, int n )
		{
			for(int i = n-1; i > 0; i--)
			{
				int target = int(drand48() * (i+1));
				int tmp = p[i];
				p[i] = p[target];
				p[target] = tmp;
			}
		}
};

enum NOISE_ENGINE{
	NOISE_PERLIN
};

class NoiseEngine : public Managed{
private:
	perlin *per_n;
	NOISE_ENGINE type;
public:

	__host__ NoiseEngine( NOISE_ENGINE tp )
	{
		type = tp;
		if(type == NOISE_PERLIN) per_n = new perlin();
		else per_n = NULL;
	}

	__host__ __device__ float noise( vec3 p )
	{
		if(type == NOISE_PERLIN) return per_n->noise(p);
		else return 0;
	}

	__host__ __device__ float turbulence( vec3 p, int depth = 7 )
	{
		float accum = 0;
		vec3 temp_p = p;
		float weight = 1.0f;
		for(int i = 0; i < depth; i++){
			accum += weight * noise(temp_p);
			weight *= 0.5; temp_p *= 2;
		}
		return fabs(accum);
	}
};