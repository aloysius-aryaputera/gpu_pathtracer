CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
# NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64
GENCODE_FLAGS  = -gencode arch=compute_37,code=sm_37

SRCS = main.cu
INCS = \
	model/camera.h \
	model/cartesian_system.h \
	model/data_structure/local_vector.h \
	model/geometry/primitive.h \
	model/geometry/sphere.h \
	model/geometry/triangle.h \
	model/grid/bounding_box.h \
	model/grid/cell.h \
	model/grid/grid.h \
	model/material.h \
	model/ray.h \
	model/scene.h \
	model/vector_and_matrix/vec3.h \
	param.h \
	render/pathtracing.h \
	util/image_util.h \
	util/read_file_util.h

gpu_path_tracing: gpu_path_tracing.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing gpu_path_tracing.o

gpu_path_tracing.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing.o -c main.cu

# gpu_path_tracing: gpu_path_tracing.o
# 	$(NVCC) $(NVCCFLAGS) -o gpu_path_tracing gpu_path_tracing.o
#
# gpu_path_tracing.o: $(SRCS) $(INCS)
# 	$(NVCC) $(NVCCFLAGS) -o gpu_path_tracing.o -c main.cu

# gpu_path_tracing: gpu_path_tracing.o
# 	$(NVCC) -o gpu_path_tracing gpu_path_tracing.o
#
# gpu_path_tracing.o: $(SRCS) $(INCS)
# 	$(NVCC) -o gpu_path_tracing.o -c main.cu
