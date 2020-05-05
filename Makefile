objects = src/main.o src/cuda/cutil.o src/third/miniz.o src/core/bsdf.o src/core/microfacet.o src/core/material.o
compile_debug_opts=-g -G
compile_fast_opts=-Xptxas -O3,-v

compile_opts=$(compile_fast_opts) -gencode arch=compute_75,code=sm_75

includes = -I./src -I/home/felpz/Documents/glm -I./src/core -I./src/cuda -I./src/third -I./src/detail -I./src/include -I/home/felpz/Documents/Graphics/include

libs = -L/home/felpz/Documents/Graphics/build -lgraphy

all: $(objects)
	nvcc $(compile_opts) $(objects) -o main $(libs)

%.o: %.cpp
	nvcc -x cu $(compile_opts) $(includes) -dc $< -o $@

clean:
	rm -f src/*.o main src/cuda/*.o src/core/*.o src/third/*.o

rebuild: clean all
