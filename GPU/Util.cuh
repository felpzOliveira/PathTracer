#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <string>
#define CHECK(r) {_check((r), __LINE__);}
inline void _check( cudaError_t r, int line )
{
  if (r != cudaSuccess)
  {
    printf("CUDA error on line %d: %s\n", line, cudaGetErrorString(r));
	getchar();
    exit(0);
  }
}

inline __host__ void cudaSynchronize( void )
{
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if(errSync != cudaSuccess)
		printf("\nSync kernel error: %s. \n", cudaGetErrorString(errSync));
	if(errAsync != cudaSuccess)
		printf("\nAsync kernel error: %s. \n", cudaGetErrorString(errAsync));
}

inline __host__ int cudaInit( void )
{
	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1; prop.minor = 0;
	CHECK(cudaChooseDevice(&dev, &prop));
	CHECK(cudaGetDeviceProperties(&prop, dev));
	std::cout << "Using device " << prop.name <<
		" which has compute capability " <<  prop.major << "." << prop.minor << std::endl; 
	return dev;
}

inline void printTime( long tseconds ){
	int minutes = tseconds/60;
	int hours = minutes/60;
	minutes = int(minutes%60);
	int seconds = int(tseconds%60);
	std::string disp(" seconds");
	disp = std::to_string(seconds) + disp;
	if(minutes == 1) disp = std::string("1 minute ") + disp;
	else if(minutes > 1) disp = std::to_string(minutes) + std::string(" minutes ") + disp;
	if(hours == 1) disp = std::string("1 hour ") + disp;
	else if(hours > 1) disp = std::to_string(hours) + std::string(" hours ") + disp;
	std::cout << disp << std::endl;
}