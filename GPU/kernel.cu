#include "Util.cuh"
#include "hitable_list.h"
#include "curand.h"
#include "camera.cuh"
#include "material.cuh"
#include "rand.cuh"
#include <fstream>
#include <ctime>
#define AA_SAMPLES_PER_BATCH 10

__global__ void init_random( curandState *states, int nx, int ny )
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int tid = i + j*nx;
	curand_init(1234, tid, 0, &states[tid]);
}

__device__ vec3 get_sky( const ray& r )
{
	vec3 unit_dir = unit_vector(r.direction());
	float t = 0.5*(unit_dir.y() + 1.0);
	return (1.0 - t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__device__ vec3 color( const ray &r, Hitable_list *world, int depth, curandState state )
{
	hit_record rec;
	if(world->hit(r, 0.001, FLT_MAX, rec))
	{
		ray scattered;
		vec3 attenuation;
		
		bool scat;
		if(rec.mat_ptr->type == 0)
			scat = rec.mat_ptr->scatter_lambertian(r, rec, attenuation, scattered, state);
		else if(rec.mat_ptr->type == 1)
			scat = rec.mat_ptr->scatter_metal(r, rec, attenuation, scattered, state);
		else if(rec.mat_ptr->type == 2)
			scat = rec.mat_ptr->scatter_dieletric(r, rec, attenuation, scattered, state);
		else 
			scat = false;

		if(depth < 10 && scat)
		{
			return attenuation*color(scattered, world, depth + 1, state);
		}
		else
			return vec3(0.0, 0.0, 0.0);
	}
	return get_sky(r);
}

/*
batch = 0 -> begin
batch = 2 -> processing
batch = 1 -> end
*/
__global__ void RenderBatch( vec3 *buffer, int nx, int ny, curandState *states,
					   Hitable_list *list, int batch, int totalBatchs )
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    float dist_to_focus = 10.0;
    float aperture = 0.1;

    camera cam(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus);
	vec3 col;
	
	if(batch == 0)
		col = vec3(0.0, 0.0, 0.0);
	else
		col = buffer[i + j*nx];

	int ns = int(AA_SAMPLES_PER_BATCH);
	for(int k = 0; k < ns; k++)
	{
		float u = float(i + curand_uniform(&states[i + j*nx]))/float(nx);
		float v = float(j + curand_uniform(&states[i + j*nx]))/float(ny);
		ray r = cam.get_ray(u, v, states[i + j*nx]);
		col += color(r, list, 0, states[i + j*nx]);
	}

	if(batch == 1)
	{
		int samples = ns*totalBatchs;
		col /= samples;
		col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
		
		int ir = int(255.99*col[0]);
		int ig = int(255.99*col[1]);
		int ib = int(255.99*col[2]);
		col = vec3(ir, ig, ib);
	}
	
	buffer[i + j*nx] = col;
}

Hitable_list * init_scene( void )
{
	Hitable_list *list;
	Material **dev_mat;
	Sphere **dev_sph;
	int n = 500;
	int c = 8;
	CHECK(cudaMallocManaged(&list, sizeof(Hitable_list)));
	CHECK(cudaMallocManaged(&list->sphere_list, sizeof(Sphere)*n));
	CHECK(cudaMallocManaged(&dev_mat, sizeof(Material*)*n));
	CHECK(cudaMallocManaged(&dev_sph, sizeof(Sphere*)*n));

	for(int i = 0; i < n; i++)
	{
		CHECK(cudaMallocManaged(&dev_mat[i], sizeof(Material)));
		CHECK(cudaMallocManaged(&dev_sph[i], sizeof(Sphere)));
	}

	dev_mat[0]->set(0, vec3(0.5, 0.5, 0.5), 0.0, 1.0);
	dev_sph[0]->set(vec3(0, -1000, 0), 1000, dev_mat[0]);

	int k = 1;
	
	for(int a = -c; a < c; a++)
	{
		for(int b = -c; b < c; b++)
		{
			float choose_mat = drand48();
			vec3 center(a + 0.9*drand48(), 0.2, b + 0.9*drand48());
			if((center - vec3(4, 0.2, 0)).length() > 0.9)
			{
				if(choose_mat < 0.8)
				{
					dev_mat[k]->set(0, vec3(drand48()*drand48(), drand48()*drand48(), drand48()*drand48()), 0.0, 1.0);
					dev_sph[k]->set(center, 0.2, dev_mat[k]);
					k++;
				}
				else if(choose_mat < 0.9)
				{
					dev_mat[k]->set(1, vec3(0.5*(1 + drand48()), 0.5*(1 + drand48()), 0.5*(1 + drand48())), 0.5*drand48(), 1.0);
					dev_sph[k]->set(center, 0.2, dev_mat[k]);
					k++;
				}
				else
				{
					dev_mat[k]->set(2, vec3(1.0, 1.0, 1.0), 0.0, 1.5);
					dev_sph[k]->set(center, 0.2, dev_mat[k]);
					k++;
				}
			}
		}
	}

	dev_mat[k]->set(2, vec3(1.0, 1.0, 1.0), 0.0, 1.5);
	dev_sph[k]->set(vec3(0, 1, 0), 1.0, dev_mat[k]); k++;

	dev_mat[k]->set(0, vec3(0.4, 0.2, 0.1), 0.0, 1.0);
	dev_sph[k]->set(vec3(-4, 1, 0), 1.0, dev_mat[k]); k++;

	dev_mat[k]->set(1, vec3(0.7, 0.6, 0.5), 0.0, 1.0);
	dev_sph[k]->set(vec3(4, 1, 0), 1.0, dev_mat[k]); k++; 

	for(int i = 0; i < k; i++)
		list->sphere_list[i] = dev_sph[i];
	
	list->sphere_list_size = k;

	return list;
}

int main(int argc, char **argv)
{
	int device = cudaInit();
	CHECK(cudaSetDevice(device));

	int nx = 256;
	int ny = 256;

	vec3 *dev_buffer;
	vec3 *hst_buffer;
	
	curandState *dev_states;
	CHECK(cudaMalloc((void**)&dev_buffer, sizeof(vec3)*nx*ny));
	CHECK(cudaMalloc((void**)&dev_states, sizeof(curandState)*nx*ny));
	hst_buffer = new vec3[nx*ny];
	dim3 block(16, 16, 1);
	dim3 grid(nx/block.x, ny/block.y, 1);
	clock_t begin = clock();

	Hitable_list *d_list = init_scene();

	init_random<<<grid, block>>>(dev_states, nx, ny);
	std::cout << "Running path tracer algorithm..." << std::endl;

	int AA_SAMPLES = 20;
	int batches = AA_SAMPLES / (int(AA_SAMPLES_PER_BATCH));

	for(int batch = 0; batch < batches; batch++)
	{
		if(batch == 0)
			RenderBatch<<<grid, block>>>(dev_buffer, nx, ny, dev_states, d_list, 0, batches);
		else if(batch == batches - 1)
			RenderBatch<<<grid, block>>>(dev_buffer, nx, ny, dev_states, d_list, 1, batches);
		else
			RenderBatch<<<grid, block>>>(dev_buffer, nx, ny, dev_states, d_list, 2, batches);

		cudaSynchronize();
		float pct = 100.0f*(float(batch + 1)/float(batches));
		std::cout << pct << "%" << std::endl;
	}
	
	clock_t end = clock();
	CHECK(cudaMemcpy(hst_buffer, dev_buffer, nx*ny*sizeof(vec3), cudaMemcpyDeviceToHost));

	std::cout << "Path tracing finished. Took : ";
	printTime((end - begin)/CLOCKS_PER_SEC);

	std::cout << "Saving image to file..." << std::endl;
	std::ofstream file("imagem.ppm");
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for(int j = ny-1; j >= 0; j--)
	{
		for(int i = 0; i < nx; i++)
		{
			vec3 col = hst_buffer[i + j*nx];
			file << col[0] << " " << col[1] << " " << col[2] << "\n";
		}
	}
	file.close();
	std::cout << "Saving process finished..." << std::endl;
	getchar();
	return 0;
}

