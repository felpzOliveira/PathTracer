#pragma once
#include "cu_math.cuh"
#include "camera.cuh"
#include "util.cuh"
#include "Scene.cuh"
#include "curand_kernel.h"
#define BOUNCES 10

namespace INTEROP{
	enum PTRACER_GL_INTEROP{
		PTRACER_USE_GL_INTEROP, PTRACER_NO_GL_INTEROP
	};
};

static __device__ inline vec3 to_rgb( vec3 c )
{
	vec3 color = vec3(sqrt(c[0]), sqrt(c[1]), sqrt(c[2]));
	int ir = int(255.99*color[0]);
	int ig = int(255.99*color[1]);
	int ib = int(255.99*color[2]);
	return vec3(ir, ig, ib);
}

static __device__ vec3 get_sky( Ray r )
{
	return vec3(0,0,0);
	//vec3 unit_dir = unit_vector(r.direction());
	//float t = 0.5f*(unit_dir.y() + 1.0f);
	//return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

static __device__ vec3 radiance( Ray r, Scene *sc, curandState *state )
{
	vec3 color(0,0,0);
	vec3 mask(1,1,1);
	Ray traced = r;
	for(int i = 0; i < int(BOUNCES); i++)
	{
		hit_record rec;
		vec3 attenuation;
		if(!sc->hit(traced, 0.001, FLT_MAX, rec)){
			color += mask * get_sky(traced);
			break;
		}
		vec3 emit = rec.mat->Emitted(rec);
		color += emit * mask;
		bool scatter = rec.mat->Scatter(traced, rec, attenuation, traced, state);
		if(scatter){
			mask *= attenuation;
		}
		else
			break;
	}
	return color;
}

static __global__ void no_gl_trace( vec3 *buffer, Scene *sc, int samples, int batch,
								   int totalBatches, int nx, int ny )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int tid = x + y*nx;

	vec3 color(0,0,0);
	if(batch != 0) color = buffer[tid];

	float inv_s = 1.0f/float(samples*totalBatches);
	uint32 hash = WangHash(batch + samples);
	curandState state;
	curand_init(hash + tid, 0, 0, &state);
	Camera *camera = sc->get_camera();

	for(int i = 0; i < samples; i++)
	{
		float u = (float(x) + getrandom(&state))/float(nx);
		float v = (float(y) + getrandom(&state))/float(ny);
		Ray r = camera->get_ray(u,v);
		color += radiance(r, sc, &state)*inv_s;
	}

	if(batch == totalBatches - 1) color = to_rgb(color);

	buffer[tid] = color;
}

class PathTracer{
private:
	INTEROP::PTRACER_GL_INTEROP m_gl_interop;
	int m_resolutionx, m_resolutiony;
	int m_samples, m_batch_size;
	vec3 *m_buffer;
	uchar4 *m_ptr;
public:
	__host__ PathTracer( int dimx, int dimy, int samples_per_pixel,
		 int batch_size, INTEROP::PTRACER_GL_INTEROP interop )
	{
		m_gl_interop = interop;
		m_resolutionx = dimx;
		m_resolutiony = dimy;
		m_samples = samples_per_pixel;
		m_batch_size = batch_size;
		if(m_gl_interop == INTEROP::PTRACER_NO_GL_INTEROP)
			init_no_gl_tracer();
	}

	__host__ void Render( Scene *sc )
	{
		if(m_gl_interop == INTEROP::PTRACER_NO_GL_INTEROP)
			render_no_gl(sc);
	}

	__host__ ~PathTracer()
	{
		if(m_gl_interop == INTEROP::PTRACER_NO_GL_INTEROP){
			cudaFree(m_buffer);
		}
		else
			free(m_ptr);
	}

	__host__ void DisplayRenderWithGimp( void )
	{
		LaunchGimpWithImage();
	}
		
private:
	__host__ void render_no_gl( Scene *sc )
	{
		dim3 block(16,16,1);
		dim3 grid(m_resolutionx/block.x, m_resolutiony/block.y, 1);
		
		int total_batches = m_samples/m_batch_size;
		std::cout << "Rendering..." << std::endl;
		for(int i = 0; i < total_batches; i++){
			no_gl_trace<<<grid, block>>>(m_buffer, sc, m_batch_size, i, total_batches,
														m_resolutionx, m_resolutiony);
			cudaSynchronize();
			printPCT(i, total_batches);
		}
		std::cout << " Finished rendering." << std::endl;
		writeImage(m_buffer, m_resolutionx, m_resolutiony, "render.ppm");
	}

	__host__ void init_no_gl_tracer(void)
	{
		m_ptr = nullptr;
		CHECK(cudaMallocManaged(&m_buffer, sizeof(vec3)*m_resolutionx*m_resolutiony));
	}
};
