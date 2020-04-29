COMPILE_DEBUG= -g -O0
COMPILE_FAST= -O3

.PHONY: all debug

all:
	nvcc -x cu -gencode=arch=compute_75,code=compute_75 src/cuda/cutil.cpp src/main.cu src/miniz.cpp src/parser_v2.cpp $(COMPILE_FAST) -I./src -I./src/include -I./src/cuda -I/home/felpz/Documents/glm -o main

debug:
	nvcc -x cu -gencode=arch=compute_75,code=compute_75 src/cuda/cutil.cpp src/main.cu src/miniz.cpp src/parser_v2.cpp $(COMPILE_DEBUG) -I./src -I./src/include -I./src/cuda -I/home/felpz/Documents/glm -o main
