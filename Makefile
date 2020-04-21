all:
	nvcc src/bsdf.cpp src/main.cu src/cuda_util.cpp src/miniz.c src/parser_v2.cpp -g -O0 -I./src -I/home/felpz/Documents/glm -o main
