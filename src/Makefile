CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
NVCC_DBG       = -g -G
# NVCC_DBG       =

NVCCFLAGS      = -m64 -O3 -std=c++14 --expt-extended-lambda
NVCCFLAGS_DBG  = $(NVCC_DBG) -m64 --expt-extended-lambda
GENCODE_FLAGS  = -gencode arch=compute_60,code=sm_60
GENCODE_FLAGS_2  = -gencode arch=compute_37,code=sm_37
GENCODE_FLAGS_3  = -gencode arch=compute_61,code=sm_61
GENCODE_FLAGS_4  = -gencode arch=compute_75,code=sm_75

SRCS = main.cu
INCS = external/libjpeg_cpp/jpeg.h

# all: gpu_path_tracing gpu_path_tracing_debug gpu_path_tracing_2 gpu_path_tracing_debug_2 gpu_path_tracing_3 gpu_path_tracing_debug_3 gpu_path_tracing_4 gpu_path_tracing_debug_4
all: gpu_path_tracing_4 gpu_path_tracing_debug_4

gpu_path_tracing: gpu_path_tracing.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing_60 gpu_path_tracing.o -ljpeg

gpu_path_tracing.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o gpu_path_tracing.o -c main.cu -ljpeg

gpu_path_tracing_debug: gpu_path_tracing_debug.o
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS) -o gpu_path_tracing_debug_60 gpu_path_tracing_debug.o -ljpeg

gpu_path_tracing_debug.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS) -o gpu_path_tracing_debug.o -c main.cu -ljpeg

gpu_path_tracing_2: gpu_path_tracing_2.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_2) -o gpu_path_tracing_37 gpu_path_tracing_2.o -ljpeg

gpu_path_tracing_2.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_2) -o gpu_path_tracing_2.o -c main.cu -ljpeg

gpu_path_tracing_debug_2: gpu_path_tracing_debug_2.o
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_2) -o gpu_path_tracing_debug_37 gpu_path_tracing_debug_2.o -ljpeg

gpu_path_tracing_debug_2.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_2) -o gpu_path_tracing_debug_2.o -c main.cu -ljpeg

gpu_path_tracing_3: gpu_path_tracing_3.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_3) -o gpu_path_tracing_61 gpu_path_tracing_3.o -ljpeg

gpu_path_tracing_3.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_3) -o gpu_path_tracing_3.o -c main.cu -ljpeg

gpu_path_tracing_debug_3: gpu_path_tracing_debug_3.o
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_3) -o gpu_path_tracing_debug_61 gpu_path_tracing_debug_3.o -ljpeg

gpu_path_tracing_debug_3.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_3) -o gpu_path_tracing_debug_3.o -c main.cu -ljpeg

gpu_path_tracing_4: gpu_path_tracing_4.o
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_4) -o gpu_path_tracing_75 gpu_path_tracing_4.o -ljpeg

gpu_path_tracing_4.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS_4) -o gpu_path_tracing_4.o -c main.cu -ljpeg

gpu_path_tracing_debug_4: gpu_path_tracing_debug_4.o
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_4) -o gpu_path_tracing_debug_75 gpu_path_tracing_debug_4.o -ljpeg

gpu_path_tracing_debug_4.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS_DBG) $(GENCODE_FLAGS_4) -o gpu_path_tracing_debug_4.o -c main.cu -ljpeg
