# Memory & Communication Benchmark Build System
# Tests: CPU mem BW/latency, GPU mem BW/latency, CPU-GPU comm BW/latency

CC       = cc
CXX      = nvcc
CFLAGS   = -O3 -march=native -fopenmp -Wall
CUFLAGS  = -O3 -arch=native -Xcompiler "-fopenmp"
LDFLAGS  = -lm

BIN_DIR  = ../bin
TARGETS  = $(BIN_DIR)/stream $(BIN_DIR)/latency $(BIN_DIR)/gpu_bench

.PHONY: all clean

# Disable implicit rules to prevent accidental in-tree builds
.SUFFIXES:

all: $(TARGETS)
	@echo "All benchmarks built in $(BIN_DIR)/"

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/stream: stream.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/latency: latency.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/gpu_bench: gpu_bench.cu | $(BIN_DIR)
	$(CXX) $(CUFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
