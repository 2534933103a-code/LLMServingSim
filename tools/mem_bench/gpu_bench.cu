/*
 * GPU Memory & CPU-GPU Communication Benchmark
 * Tests:
 *   1. GPU device-to-device bandwidth
 *   2. GPU device memory latency (pointer chasing)
 *   3. CPU->GPU (H2D) bandwidth
 *   4. GPU->CPU (D2H) bandwidth
 *   5. CPU->GPU latency (small transfer round-trip)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHECK(call) do {                                               \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e));                                 \
        exit(1);                                                        \
    }                                                                   \
} while (0)

/* Allocate with graceful OOM handling: returns 1 on success, 0 on OOM */
static int try_cuda_malloc(void **ptr, size_t bytes) {
    cudaError_t e = cudaMalloc(ptr, bytes);
    if (e == cudaSuccess) return 1;
    if (e == cudaErrorMemoryAllocation) {
        fprintf(stderr, "  (skipped: out of memory for %.0f MB)\n", bytes / 1e6);
        return 0;
    }
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
    exit(1);
}

#define WARMUP_ITERS 10
#define BENCH_ITERS   50

static double walltime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* ---- GPU Device-to-Device Bandwidth ---- */
__global__ void d2d_copy_kernel(float *dst, const float *src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

void test_gpu_d2d_bandwidth(void) {
    printf("\n==========================================\n");
    printf("GPU Device-to-Device Bandwidth\n");
    printf("==========================================\n");

    size_t sizes[] = {
        4 * 1024 * 1024,     /* 4M floats = 16 MB */
        16 * 1024 * 1024,    /* 16M floats = 64 MB */
        64 * 1024 * 1024,    /* 64M floats = 256 MB */
        128 * 1024 * 1024,   /* 128M floats = 512 MB */
        256 * 1024 * 1024,   /* 256M floats = 1 GB */
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int block = 256;

    for (int si = 0; si < n_sizes; si++) {
        size_t n = sizes[si];
        size_t bytes = n * sizeof(float);
        float *d_src, *d_dst;
        if (!try_cuda_malloc((void **)&d_src, bytes)) continue;
        if (!try_cuda_malloc((void **)&d_dst, bytes)) {
            CHECK(cudaFree(d_src));
            continue;
        }

        /* Warmup */
        for (int i = 0; i < WARMUP_ITERS; i++) {
            int grid = (n + block - 1) / block;
            d2d_copy_kernel<<<grid, block>>>(d_dst, d_src, n);
        }
        CHECK(cudaDeviceSynchronize());

        /* Benchmark */
        double best = 1e12;
        for (int i = 0; i < BENCH_ITERS; i++) {
            double t0 = walltime();
            int grid = (n + block - 1) / block;
            d2d_copy_kernel<<<grid, block>>>(d_dst, d_src, n);
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best) best = t;
        }

        double bw = (bytes * 2.0) / (best * 1e9);
        printf("  %7.0f MB:  %8.3f ms  %8.1f GB/s\n",
               bytes / 1e6, best * 1000.0, bw);

        CHECK(cudaFree(d_src));
        CHECK(cudaFree(d_dst));
    }
}

/* ---- GPU Device Memory Latency (Pointer Chasing) ---- */
/*
 * Build a pointer-chasing linked list in GPU memory.
 * n      = number of chain nodes (buf_size / sizeof(uint64_t))
 * stride = elements between chain nodes (for cache-line spacing)
 *
 * Each node at buf[i * stride] points to buf[((i+1) % n) * stride].
 */
__global__ void setup_chain(uint64_t *buf, size_t n, size_t stride) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    size_t pos      = i * stride;
    size_t next_pos = ((i + 1) % n) * stride;
    buf[pos] = (uint64_t)(buf + next_pos);
}

/*
 * Follow the pointer chain and measure GPU clock cycles.
 * Launched with <<<1, 1>>> so only thread 0 executes.
 */
__global__ void pointer_chase(uint64_t *buf, size_t iters, uint64_t *cycles_out) {
    volatile uint64_t *p = buf;
    uint64_t start = clock64();
    for (size_t i = 0; i < iters; i++) {
        p = (volatile uint64_t *)*p;
    }
    uint64_t end = clock64();
    cycles_out[0] = (end - start) / iters;
    (void)p;
}

void test_gpu_latency(void) {
    printf("\n==========================================\n");
    printf("GPU Device Memory Latency (Pointer Chasing)\n");
    printf("==========================================\n");

    /* Test at various buffer sizes to observe cache hierarchy */
    size_t buf_sizes[] = {
        32 * 1024,        /* 32 KB  - L1 cache */
        128 * 1024,       /* 128 KB */
        1 * 1024 * 1024,  /* 1 MB   - L2 cache */
        4 * 1024 * 1024,  /* 4 MB */
        16 * 1024 * 1024, /* 16 MB  - device memory */
        64 * 1024 * 1024, /* 64 MB */
        256 * 1024 * 1024,/* 256 MB */
    };
    int n_sizes = sizeof(buf_sizes) / sizeof(buf_sizes[0]);
    size_t stride = 16; /* 16 x 8B = 128B stride (two cache lines on Ampere) */
    size_t iters = 200000;

    for (int si = 0; si < n_sizes; si++) {
        size_t buf_size = buf_sizes[si];
        size_t n_elems = buf_size / sizeof(uint64_t);
        size_t total_elems = n_elems * stride;

        uint64_t *d_buf, *d_cycles, h_cycle;
        if (!try_cuda_malloc((void **)&d_buf, total_elems * sizeof(uint64_t))) continue;
        CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t)));

        int block = 256;
        int grid = (int)((n_elems + block - 1) / block);
        setup_chain<<<grid, block>>>(d_buf, n_elems, stride);
        CHECK(cudaDeviceSynchronize());

        /* Warmup */
        pointer_chase<<<1, 1>>>(d_buf, iters / 10, d_cycles);
        CHECK(cudaDeviceSynchronize());

        /* Measure */
        pointer_chase<<<1, 1>>>(d_buf, iters, d_cycles);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(&h_cycle, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        /* Ampere (RTX 3090) base clock ~1.4 GHz, boost ~1.7 GHz */
        double ns = h_cycle / 1.4; /* cycles / (1.4 cycles/ns) ~= ns */
        printf("  %7.0f KB:  %5lu cycles  %7.1f ns\n",
               buf_size / 1024.0, (unsigned long)h_cycle, ns);

        CHECK(cudaFree(d_buf));
        CHECK(cudaFree(d_cycles));
    }
}

/* ---- CPU-GPU (H2D / D2H) Bandwidth ---- */
void test_h2d_d2h_bandwidth(void) {
    printf("\n==========================================\n");
    printf("CPU-GPU Communication Bandwidth\n");
    printf("==========================================\n");

    size_t sizes[] = {
        1 * 1024 * 1024,    /* 1 MB */
        8 * 1024 * 1024,    /* 8 MB */
        64 * 1024 * 1024,   /* 64 MB */
        256 * 1024 * 1024,  /* 256 MB */
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n%-12s %12s %12s %12s %12s\n",
           "Size", "H2D ms", "H2D GB/s", "D2H ms", "D2H GB/s");
    printf("----------------------------------------------------------------\n");

    for (int si = 0; si < n_sizes; si++) {
        size_t bytes = sizes[si];
        float *h_buf = (float *)malloc(bytes);
        if (!h_buf) { fprintf(stderr, "  (skipped: host OOM for %.0f MB)\n", bytes/1e6); continue; }
        float *d_buf;
        memset(h_buf, 0xAB, bytes);
        if (!try_cuda_malloc((void **)&d_buf, bytes)) { free(h_buf); continue; }

        /* H2D */
        for (int i = 0; i < WARMUP_ITERS; i++)
            CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));

        double best_h2d = 1e12;
        for (int i = 0; i < BENCH_ITERS; i++) {
            CHECK(cudaDeviceSynchronize());
            double t0 = walltime();
            CHECK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best_h2d) best_h2d = t;
        }

        /* D2H */
        for (int i = 0; i < WARMUP_ITERS; i++)
            CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));

        double best_d2h = 1e12;
        for (int i = 0; i < BENCH_ITERS; i++) {
            CHECK(cudaDeviceSynchronize());
            double t0 = walltime();
            CHECK(cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best_d2h) best_d2h = t;
        }

        double bw_h2d = bytes / (best_h2d * 1e9);
        double bw_d2h = bytes / (best_d2h * 1e9);

        printf("  %6.0f MB   %10.3f %10.1f %10.3f %10.1f\n",
               bytes / 1e6, best_h2d * 1000.0, bw_h2d,
               best_d2h * 1000.0, bw_d2h);

        free(h_buf);
        CHECK(cudaFree(d_buf));
    }
}

/* ---- CPU-GPU Small Transfer Latency (Round-Trip) ---- */
void test_h2d_d2h_latency(void) {
    printf("\n==========================================\n");
    printf("CPU-GPU Communication Latency (small transfer)\n");
    printf("==========================================\n");

    size_t sizes[] = {4, 64, 1024, 4096, 16384, 65536, 262144, 1048576};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iters = 2000;

    printf("\n%-10s %12s %12s %12s\n", "Bytes", "H2D us", "D2H us", "RTT us");
    printf("-------------------------------------------------------\n");

    for (int si = 0; si < n_sizes; si++) {
        size_t bytes = sizes[si];
        char *h_buf = (char *)malloc(bytes);
        char *d_buf;
        memset(h_buf, 0, bytes);
        CHECK(cudaMalloc(&d_buf, bytes));

        /* H2D latency */
        double best_h2d = 1e12;
        for (int i = 0; i < iters; i++) {
            CHECK(cudaDeviceSynchronize());
            double t0 = walltime();
            CHECK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best_h2d) best_h2d = t;
        }

        /* D2H latency */
        double best_d2h = 1e12;
        for (int i = 0; i < iters; i++) {
            CHECK(cudaDeviceSynchronize());
            double t0 = walltime();
            CHECK(cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best_d2h) best_d2h = t;
        }

        /* RTT: H2D + D2H in sequence */
        double best_rtt = 1e12;
        for (int i = 0; i < iters; i++) {
            CHECK(cudaDeviceSynchronize());
            double t0 = walltime();
            CHECK(cudaMemcpyAsync(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpyAsync(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
            CHECK(cudaDeviceSynchronize());
            double t = walltime() - t0;
            if (t < best_rtt) best_rtt = t;
        }

        printf("  %-8zu  %10.2f %10.2f %10.2f\n",
               bytes, best_h2d * 1e6, best_d2h * 1e6, best_rtt * 1e6);

        free(h_buf);
        CHECK(cudaFree(d_buf));
    }
}

int main(void) {
    /* Print GPU info */
    int dev;
    CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("==========================================================\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Max Clock Rate: %.2f GHz\n", prop.clockRate / 1e6);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("L2 Cache: %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("PCIe Gen: %d, NumLink: %d\n", prop.pciDeviceID >> 8, prop.pciBusID);
    printf("==========================================================\n");

    test_gpu_d2d_bandwidth();
    test_gpu_latency();
    test_h2d_d2h_bandwidth();
    test_h2d_d2h_latency();

    printf("\nDone.\n");
    return 0;
}
