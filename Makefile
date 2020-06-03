objects = src/main.o src/cuda/cutil.o src/third/miniz.o src/materials/mirror.o  src/core/transform.o src/shapes/sphere.o src/core/interaction.o src/core/primitive.o src/core/camera.o src/third/graphy.o src/core/reflection.o src/materials/matte.o src/core/material.o src/materials/glass.o src/core/microfacet.o src/materials/metal.o src/materials/translucent.o src/materials/plastic.o src/shapes/mesh.o src/shapes/loadablemesh.o src/materials/uber.o src/core/scene.o src/shapes/rectangle.o src/third/ppm.o src/shapes/box.o src/core/light.o src/shapes/disk.o src/third/image_util.o src/third/mtl.o src/third/obj_loader.o src/core/medium.o src/core/procedural.o src/lights/diffusearea.o src/lights/distant.o src/core/texture.o src/lights/infinite.o src/core/bssrdf.o src/materials/subsurface.o

compile_debug_opts=-g -G
compile_fast_opts=-Xptxas -O3

compile_opts=$(compile_fast_opts) -gencode arch=compute_75,code=sm_75

includes = -I./src -I./src/cuda -I./src/third -I./src/core -I./src/third/graphy

libs = -ldl

all: $(objects)
	nvcc $(compile_opts) $(objects) -o main $(libs)

%.o: %.cpp
	nvcc -x cu $(compile_opts) $(includes) -dc $< -o $@

clean:
	rm -f src/*.o main src/core/*.o src/cuda/*.o src/third/*.o src/shapes/*.o src/materials/*.o src/lights/*.o

rebuild: clean all
