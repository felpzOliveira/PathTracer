#include <cutil.h>
#include <stdio.h>
#include <iostream>

Memory global_memory = {0};

void _check(cudaError_t err, int line, const char *filename){
    if(err != cudaSuccess){
        std::cout << "CUDA error > " << filename << ": " << line << "[" << cudaGetErrorString(err) << "]" << std::endl;
        getchar();
        exit(0);
    }
}

int cudaInit(){
    cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1; prop.minor = 0;
	CUCHECK(cudaChooseDevice(&dev, &prop));
	CUCHECK(cudaGetDeviceProperties(&prop, dev));
    global_memory.allocated = 0;
	std::cout << "Using device " << prop.name << "[ " <<  prop.major << "." << prop.minor << " ]" << std::endl; 
	return dev;
}

void cudaPrintMemoryTaken(){
    std::string unity("b");
    float amount = (float)(global_memory.allocated);
    if(amount > 1024){
        amount /= 1024.f;
        unity = "KB";
    }
    
    if(amount > 1024){
        amount /= 1024.f;
        unity = "MB";
    }
    
    if(amount > 1024){
        amount /= 1024.f;
        unity = "GB";
    }
    
    std::cout << "Took " << amount << " " << unity << " of GPU memory" << std::endl;
}

int cudaSynchronize(){
    int rv = 0;
    cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if(errSync != cudaSuccess){
        std::cout << "Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
		rv = 1;
    }
	if(errAsync != cudaSuccess){
        std::cout << "Sync kernel error: " << cudaGetErrorString(errAsync) << std::endl;
		rv = 1;
    }
    
    return rv;
}

DeviceMemoryStats cudaReportMemoryUsage(){
    DeviceMemoryStats memStats;
	cudaError_t status = cudaMemGetInfo(&memStats.free_bytes, &memStats.total_bytes);
    if(status != cudaSuccess){
        std::cout << "Could not query device for memory!" << std::endl;
        memStats.valid = 0;
    }else{
        memStats.used_bytes = memStats.total_bytes - memStats.free_bytes;
        memStats.valid = 1;
    }
    
    return memStats;
}

int cudaHasMemory(size_t bytes){
    DeviceMemoryStats mem = cudaReportMemoryUsage();
    int ok = 0;
    if(mem.valid){
        ok = mem.free_bytes > bytes ? 1 : 0;
    }
    
    return ok;
}

void *_cudaAllocate(size_t bytes, int line, const char *filename, bool abort){
    void *ptr = nullptr;
    if(cudaHasMemory(bytes)){
        cudaError_t err = cudaMallocManaged(&ptr, bytes);
        if(err != cudaSuccess){
            std::cout << "Failed to allocate memory " << filename << ":" << line << "[" << bytes << " bytes]" << std::endl;
            ptr = nullptr;
        }else{
            global_memory.allocated += bytes;
        }
    }
    
    if(!ptr && abort){
        getchar();
        exit(0);
    }
    
    return ptr;
}