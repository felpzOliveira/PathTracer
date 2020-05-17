objects = src/main.o src/cuda/cutil.o src/third/miniz.o src/materials/mirror.o  src/core/transform.o src/shapes/sphere.o src/core/interaction.o src/core/primitive.o src/core/camera.o src/third/graphy.o src/core/reflection.o src/materials/matte.o src/core/material.o src/materials/glass.o src/core/microfacet.o src/materials/metal.o src/materials/translucent.o src/materials/plastic.o src/shapes/mesh.o src/shapes/loadablemesh.o src/materials/uber.o src/core/scene.o src/shapes/rectangle.o

compile_debug_opts=-g -G
compile_fast_opts=-Xptxas -O3

compile_opts=$(compile_fast_opts) -gencode arch=compute_75,code=sm_75

includes = -I./src -I./src/cuda -I./src/third -I./src/core -I/home/felpz/Documents/Graphics/include

libs = -ldl

all: $(objects)
	nvcc $(compile_opts) $(objects) -o main $(libs)

%.o: %.cpp
	nvcc -x cu $(compile_opts) $(includes) -dc $< -o $@

clean:
	rm -f src/*.o main src/core/*.o src/cuda/*.o src/third/*.o src/shapes/*.o src/materials/*.o

rebuild: clean all
