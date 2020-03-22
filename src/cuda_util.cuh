#if !defined(CUDA_UTIL_H)
#define CUDA_UTIL_H
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK(r) {_check((r), __LINE__);}

void _check( cudaError_t r, int line );

void cudaSynchronize( void );

int cudaInit( void );

void cudaReportMemoryUsage( void );

void cudaReportMemoryUsageR( void );

int cudaCanAllocate( size_t bytes );

void cudaFailure( void );

void *cudaAllocOrFail(size_t memory);

#endif