#include "cuda_util.cuh"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

static void _memset(void *addr, unsigned char value, long size){
    unsigned char *uaddr = (unsigned char *)addr;
    for(long i = 0; i < size; i += 1){
        *uaddr++ = value;
    }
}

void cudaFailure( void ){
    printf("GPU error, cannot continue\n");
    exit(0);
}

void _check( cudaError_t r, int line ){
	if (r != cudaSuccess){
        std::cout << "\nCUDA error on line " << line << ": "<< cudaGetErrorString(r) << "\n";
		getchar();
		exit(0);
	}
}

int cudaSynchronize( void ){
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
    int rv = 0;
	if(errSync != cudaSuccess){
		printf("\nSync kernel error: %s. \n", cudaGetErrorString(errSync));
        rv = 1;
    }
	if(errAsync != cudaSuccess){
		printf("\nAsync kernel error: %s. \n", cudaGetErrorString(errAsync));
        rv = 1;
    }
    return rv;
}

int cudaInit( void ){
	cudaDeviceProp prop;
	int dev;
	_memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1; prop.minor = 0;
	CHECK(cudaChooseDevice(&dev, &prop));
	CHECK(cudaGetDeviceProperties(&prop, dev));
	std::cout << "Using device " << prop.name <<
		" which has compute capability " <<  prop.major << "." << prop.minor << std::endl; 
	return dev;
}

void cudaReportMemoryUsage( void ){
	size_t free_byte;
	size_t total_byte;
	cudaError_t status = cudaMemGetInfo(&free_byte, &total_byte);
	if(status != cudaSuccess)
		printf("Failed to check memory usage!\n%s", cudaGetErrorString(status));
	else{
		double dfree = double(free_byte);
		double dtotal = double(total_byte);
		double used = dtotal - dfree;
		printf("GPU Memory usage: used = %.2f, free = %.2f MB, total = %.2f MB\n",
               used/1024.0/1024.0, dfree/1024.0/1024.0, dtotal/1024.0/1024.0);
	}
}

void cudaReportMemoryUsageR( void ){
	size_t free_byte;
	size_t total_byte;
	cudaError_t status = cudaMemGetInfo(&free_byte, &total_byte);
	if(status != cudaSuccess)
		printf("Failed to check memory usage!\n%s", cudaGetErrorString(status));
	else{
		double dfree = double(free_byte);
		double dtotal = double(total_byte);
		double used = dtotal - dfree;
		printf("GPU Memory usage: used = %.2f, free = %.2f MB, total = %.2f MB\r",
               used/1024.0/1024.0, dfree/1024.0/1024.0, dtotal/1024.0/1024.0);
	}
}

int cudaCanAllocate( size_t bytes ){
    size_t free_bytes;
    size_t total_bytes;
    cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if(status != cudaSuccess){
		printf("Failed to check memory usage!\n%s", cudaGetErrorString(status));
        return 0;
    }else{
        return free_bytes > bytes;
    }
}

void *cudaAllocOrFail(size_t memory){
    void *ptr = nullptr;
    if(cudaCanAllocate(memory)){
        CHECK(cudaMallocManaged(&ptr, memory));
    }else{
        cudaFailure();
    }
    
    return ptr;
}