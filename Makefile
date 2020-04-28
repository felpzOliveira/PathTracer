all:
	nvcc -x cu src/cuda/cutil.cpp src/main.cu src/miniz.cpp src/parser_v2.cpp -g -O0 -I./src -I./src/include -I./src/cuda -I/home/felpz/Documents/glm -o main
