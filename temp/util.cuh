#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <atlstr.h>
#include <Windows.h>
#include "cu_math.cuh"
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

class Managed{
public:
	void *operator new( size_t len )
	{
		void *ptr;
		CHECK(cudaMallocManaged(&ptr, len));
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete( void *ptr )
	{
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

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
	prop.major = 3; prop.minor = 2;
	CHECK(cudaChooseDevice(&dev, &prop));
	CHECK(cudaGetDeviceProperties(&prop, dev));
	std::cout << "Using device " << prop.name << std::endl; 
	return dev;
}

inline void printTime( long tseconds )
{
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

inline void printPCT( int batch, int totalBatches )
{
	float pct = 100.0f*(float(batch + 1)/float(totalBatches));
	printf("\r%.2f%s", pct, "%");
}
	
inline void writeImage( vec3 *buffer, int nx, int ny, const char *filepath )
{
	std::cout << "Writing image to file...";
	std::ofstream file(filepath);
	file << "P3\n" << nx << " " << ny << "\n255\n";
	for(int j = ny-1; j >= 0; j--)
	{
		for(int i = 0; i < nx; i++)
		{
			vec3 col = buffer[i + j*nx];
			file << col[0] << " " << col[1] << " " << col[2] << "\n";
		}
	}
	file.close();
	std::cout << "process finished.";
}

inline void LaunchGimpWithImage( void )
{
	CString cmdLine = CString(_T("\"C:\\Program Files\\GIMP 2\\bin\\gimp-2.8.exe\" \"C:\\Users\\felip\\OneDrive\\Documentos\\Visual Studio 2012\\Projects\\Renderer\\Renderer\\render.ppm\"")); 
	PROCESS_INFORMATION processInformation = {0};
	STARTUPINFO startupInfo = {0};
	startupInfo.cb = sizeof(startupInfo);
	int nStrBuffer = cmdLine.GetLength() + 50;

	BOOL result = CreateProcess(NULL, cmdLine.GetBuffer(nStrBuffer), NULL, NULL, FALSE,
		NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW, NULL, NULL, &startupInfo, &processInformation);

	DWORD exitCode;
	cmdLine.ReleaseBuffer();
	WaitForSingleObject(processInformation.hProcess, INFINITE);
	result = GetExitCodeProcess(processInformation.hProcess, &exitCode);
	CloseHandle(processInformation.hProcess);
	CloseHandle(processInformation.hThread);

	if(!result){ std::cout << "Process error!" << std::endl; getchar(); exit(-1); }
}