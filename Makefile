CUDA_INCLUDE = -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.9/cuda/13.0/include

# For gcc: gcc 10 and above for -std=c++20, use --std=c++2a for older versions, use -fopenmp for OpenMP support
NVC++_FLAGS = -std=c++20 -stdpar=multicore -O3 -gopt -mp=ompt $(CUDA_INCLUDE)
NVC++ = nvc++ $(NVC++_FLAGS) -ldl

NVCCC_FLAGS = -w -Wno-deprecated-gpu-targets --std=c++20 -O3 -DNDEBUG $(CUDA_INCLUDE) -arch=compute_89
NVCC = nvcc $(NVCCC_FLAGS) $(NVCC_LDFLAGS) -lcuda -lineinfo

NCU = ncu --set full --import-source yes -f --page details
NSYS = nsys profile --trace=cuda,nvtx,osrt,openmp --sample=cpu --cpuctxsw=process-tree --force-overwrite true
VTUNE = vtune -collect hotspots

SRC_DIR = ./src
OUTPUT_DIR = ./build
OUTPUT_FILE = $(OUTPUT_DIR)/$@

REPORTS_OUTPUT_DIR = ~/reports
NCU_OUTPUT_FILE = $(REPORTS_OUTPUT_DIR)/$@
NSYS_OUTPUT_FILE = $(REPORTS_OUTPUT_DIR)/$@

matrix_mul_gpu: $(SRC_DIR)/matrix_mul_gpu.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

matrix_mul_gpu_profile: $(OUTPUT_DIR)/matrix_mul_gpu
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

matrix_mul_cpu: $(SRC_DIR)/matrix_mul_cpu.cpp
	$(NVC++) -o $(OUTPUT_FILE) $^

matrix_mul_cpu_vtune: $(OUTPUT_DIR)/matrix_mul_cpu
	$(VTUNE) -r $(REPORTS_OUTPUT_DIR)/$@ $^

matrix_mul_cpu_nsys: $(OUTPUT_DIR)/matrix_mul_cpu
	$(NSYS) -o $(REPORTS_OUTPUT_DIR)/$@ $^

scan: $(SRC_DIR)/scan.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

scan_profile: $(OUTPUT_DIR)/scan
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

stream_compaction: $(SRC_DIR)/stream_compaction.cu
	$(NVCC) --extended-lambda -o $(OUTPUT_FILE) $^

stream_compaction_cpu: $(SRC_DIR)/stream_compaction_cpu.cpp $(SRC_DIR)/stream_compaction_utils.cpp
	$(NVC++) -o $(OUTPUT_FILE) $^

stream_compaction_gpu: $(SRC_DIR)/stream_compaction_gpu.cu $(SRC_DIR)/stream_compaction_utils.cpp
	$(NVCC) --extended-lambda -o $(OUTPUT_FILE) $^

stream_compaction_cpu_nsys: $(OUTPUT_DIR)/stream_compaction_cpu
	$(NSYS) -o $(REPORTS_OUTPUT_DIR)/$@ $^

stream_compaction_cpu_vtune: $(OUTPUT_DIR)/stream_compaction_cpu
	$(VTUNE) -r $(REPORTS_OUTPUT_DIR)/$@ $^

stream_compaction_gpu_profile: $(OUTPUT_DIR)/stream_compaction_gpu
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

stream_compaction_gpu_nsys: $(OUTPUT_DIR)/stream_compaction_gpu
	$(NSYS) -o $(NSYS_OUTPUT_FILE) $^

coordinates: $(SRC_DIR)/coordinates.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

coordinates_profile: $(OUTPUT_DIR)/coordinates
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

all: matrix_mul histogram reduction scan stream_compaction_cpu stream_compaction_gpu coordinates

clean:
	rm -f $(OUTPUT_DIR)/matrix_mul
	rm -f $(OUTPUT_DIR)/histogram
	rm -f $(OUTPUT_DIR)/reduction
	rm -f $(OUTPUT_DIR)/scan
	rm -f $(OUTPUT_DIR)/stream_compaction_cpu
	rm -f $(OUTPUT_DIR)/stream_compaction_gpu
	rm -f $(OUTPUT_DIR)/coordinates