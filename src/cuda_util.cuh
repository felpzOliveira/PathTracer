#if !defined(CUDA_UTIL_H)
#define CUDA_UTIL_H
#include "cuda_runtime.h"

#define CHECK(r) {_check((r), __LINE__);}

void _check( cudaError_t r, int line );

int cudaSynchronize( void );

int cudaInit( void );

void cudaReportMemoryUsage( void );

void cudaReportMemoryUsageR( void );

int cudaCanAllocate( size_t bytes );

void cudaFailure( void );

void *cudaAllocOrFail(size_t memory);

class Managed{
    public:
    void *operator new(size_t len){
        void *ptr = 0;
        CHECK(cudaMallocManaged(&ptr, len));
        return ptr;
    }
    
    void operator delete(void *ptr){
        cudaFree(ptr);
    }
};

#endif