CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
SM            ?= 60

# select one of these for Debug vs. Release
NVCC_DBG       = -g -G

NVCCFLAGS      = -m64 -O3 -std=c++14 --expt-extended-lambda
NVCCFLAGS_DBG  = $(NVCC_DBG) -m64 --expt-extended-lambda
GENCODE_FLAGS  = -gencode arch=compute_$(SM),code=sm_${SM}

SRCS = main.cu
INCS = external/libjpeg_cpp/jpeg.h

all: gpu_path_tracing gpu_path_tracing_debug

gpu_path_tracing: gpu_path_tracing.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing_${SM} gpu_path_tracing.o -ljpeg

gpu_path_tracing.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing.o -c main.cu -ljpeg

gpu_path_tracing_debug: gpu_path_tracing_debug.o
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS) -o gpu_path_tracing_debug_${SM} gpu_path_tracing_debug.o -ljpeg

gpu_path_tracing_debug.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS) -o gpu_path_tracing_debug.o -c main.cu -ljpeg

