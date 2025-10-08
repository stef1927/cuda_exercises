NVCCC_FLAGS = -w -Wno-deprecated-gpu-targets --std=c++20 -O3 -DNDEBUG 
NVCC_LDFLAGS = -lcuda
NVCC = nvcc $(NVCCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo
NCU = ncu --set full --import-source yes -f --page details

SRC_DIR = ./src
OUTPUT_DIR = ./build
OUTPUT_FILE = $(OUTPUT_DIR)/$@

NCU_OUTPUT_DIR = ~/reports
NCU_OUTPUT_FILE = $(NCU_OUTPUT_DIR)/$@

vector_add: $(SRC_DIR)/vector_add.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

vector_add_profile: $(OUTPUT_DIR)/vector_add
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

matrix_mul: $(SRC_DIR)/matrix_mul.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

matrix_mul_profile: $(OUTPUT_DIR)/matrix_mul
	$(NCU) -o $(NCU_OUTPUT_FILE) $^ --num-runs=1

histogram: $(SRC_DIR)/histogram.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

histogram_profile: $(OUTPUT_DIR)/histogram
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

reduction: $(SRC_DIR)/reduction.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

reduction_profile: $(OUTPUT_DIR)/reduction
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

clean:
	rm -f $(OUTPUT_DIR)/vector_add
	rm -f $(OUTPUT_DIR)/matrix_mul
	rm -f $(OUTPUT_DIR)/histogram
	rm -f $(OUTPUT_DIR)/reduction