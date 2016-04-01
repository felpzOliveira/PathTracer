# PathTracer
Path Tracer based on the mini-book series written by Peter Shirley

This repo contains 2 implementations, CPU/GPU version (C/C++ , Cuda C).

Currently only worked on the first volume of the mini-books series : "Ray Tracing in One Weekend"

The images bellow were rendered using my (very slow) notebook:

Intel i5-5200U @2.20GHz,
8.0GB RAM,
Geforce 920M 2GB

Image rendered using the CPU version - 120 samples per pixel ( 5 hours 20 minutes and 45 seconds )

![Alt text](cpu.png "CPU")

Image rendered using the GPU version - 150 samples per pixel ( 55 minutes and 24 seconds )

![Alt text](gpu.png "GPU")

*GPU version uses cudaMallocManaged and recursion on __device__ function. Be sure your GPU supports that if you wish to build the program.
