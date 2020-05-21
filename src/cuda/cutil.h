#if !defined(CUTIL_H)
#define CUTIL_H
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>

/*
* CUDA UTILITIES
*/

#define CUCHECK(r) _check((r), __LINE__, __FILE__)
#define cudaAllocate(bytes) _cudaAllocate(bytes, __LINE__, __FILE__, true)
#define cudaAllocateEx(bytes, abort) _cudaAllocate(bytes, __LINE__, __FILE__, abort)
#define cudaAllocateVx(type, n) (type *)_cudaAllocate(sizeof(type)*n, __LINE__, __FILE__, true)
#define cudaDeviceAssert() if(cudaSynchronize()){ cudaSafeExit(); }

#define __bidevice__ __host__ __device__ 

typedef struct{
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    int valid;
}DeviceMemoryStats;

typedef struct{
    size_t allocated;
}Memory;

extern Memory global_memory;

/*
 * Sanity function to check for cuda* operations.
*/
void _check(cudaError_t err, int line, const char *filename);

/*
* Initialize a cuda capable device to start kernel launches.
*/
int  cudaInit(void);
void cudaInitEx(void);

/*
* Synchronizes the device so host access is not asynchronous,
* also checks devices for errors in recent kernel launches.
*/
int cudaSynchronize(void);

/*
* Get information about memory usage from the device.
*/
DeviceMemoryStats cudaReportMemoryUsage(void);

/*
* Checks if _at_this_moment_ it is possible to alocate memory on the device.
*/
int cudaHasMemory(size_t bytes);

/*
* Attempts to allocate a block of memory in the device. The returned
* memory when valid (!=nullptr) is managed.
*/
void *_cudaAllocate(size_t bytes, int line, const char *filename, bool abort);

/*
* Prints current amount of allocated device memory.
*/
void cudaPrintMemoryTaken(void);

void cudaSafeExit(void);

#endif