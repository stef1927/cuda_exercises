NVCCC_FLAGS = -w -Wno-deprecated-gpu-targets --std=c++20 -O3 -DNDEBUG 
NVCC_LDFLAGS = -lcuda
NVCC = nvcc $(NVCCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo
NCU = ncu --set full --import-source yes -f --page details
NVC++ = nvc++ -std=c++20 -stdpar=gpu -O3  # not yet in use

SRC_DIR = ./src
OUTPUT_DIR = ./build
OUTPUT_FILE = $(OUTPUT_DIR)/$@

NCU_OUTPUT_DIR = ~/reports
NCU_OUTPUT_FILE = $(NCU_OUTPUT_DIR)/$@

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

scan: $(SRC_DIR)/scan.cu
	$(NVCC) -o $(OUTPUT_FILE) $^

scan_profile: $(OUTPUT_DIR)/scan
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

stream_compaction: $(SRC_DIR)/stream_compaction.cu
	$(NVCC) --extended-lambda -o $(OUTPUT_FILE) $^

stream_compaction_profile: $(OUTPUT_DIR)/stream_compaction
	$(NCU) -o $(NCU_OUTPUT_FILE) $^

clean:
	rm -f $(OUTPUT_DIR)/matrix_mul
	rm -f $(OUTPUT_DIR)/histogram
	rm -f $(OUTPUT_DIR)/reduction
	rm -f $(OUTPUT_DIR)/scan
	rm -f $(OUTPUT_DIR)/stream_compaction