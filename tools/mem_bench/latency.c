/* CPU memory latency benchmark via pointer chasing */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#define GB (1UL << 30)
#define MB (1UL << 20)
#define CACHELINE 64

static uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static double get_tsc_freq(void) {
    struct timeval tv1, tv2;
    uint64_t t1, t2;
    gettimeofday(&tv1, NULL);
    t1 = rdtsc();
    usleep(100000);
    t2 = rdtsc();
    gettimeofday(&tv2, NULL);
    double elapsed = (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec) * 1e-6;
    return (t2 - t1) / elapsed;
}

/* Build a pointer-chasing linked list over the buffer, stride = stride bytes */
static void *build_chain(void *buf, size_t size, size_t stride) {
    size_t n = size / stride;
    size_t *p = (size_t *)buf;
    for (size_t i = 0; i < n - 1; i++) {
        p[(i * stride) / sizeof(size_t)] = (size_t)&p[((i + 1) * stride) / sizeof(size_t)];
    }
    p[((n - 1) * stride) / sizeof(size_t)] = (size_t)buf;
    return buf;
}

static double measure_latency(void *buf, size_t size, size_t stride, int iters) {
    volatile size_t *p = (volatile size_t *)buf;
    /* warmup */
    for (int i = 0; i < iters / 4; i++) p = (volatile size_t *)*p;
    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) p = (volatile size_t *)*p;
    uint64_t end = rdtsc();
    double cycles = (double)(end - start) / iters;
    (void)p;
    return cycles;
}

static double cycles_to_ns(double cycles, double freq_hz) {
    return cycles / freq_hz * 1e9;
}

static void test_stride(size_t stride, const char *label) {
    size_t sizes[] = {
        16 * 1024,       /* L1 */
        64 * 1024,       /* L1 end */
        256 * 1024,      /* L2 */
        512 * 1024,      /* L2 end */
        2 * MB,          /* L3 start */
        8 * MB,          /* L3 */
        16 * MB,         /* L3 end */
        32 * MB,         /* beyond L3 */
        64 * MB,
        128 * MB,
        256 * MB,
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    double freq = get_tsc_freq();
    printf("  %s (stride=%zu, freq=%.2f GHz):\n", label, stride, freq / 1e9);

    for (int i = 0; i < n_sizes; i++) {
        void *buf = malloc(sizes[i]);
        if (!buf) { printf("    alloc failed for %zu MB\n", sizes[i] / MB); continue; }
        memset(buf, 0, sizes[i]);
        build_chain(buf, sizes[i], stride);
        int iters = (sizes[i] >= 32 * MB) ? 2000000 : 5000000;
        double lat_cyc = measure_latency(buf, sizes[i], stride, iters);
        double lat_ns = cycles_to_ns(lat_cyc, freq);
        printf("    %6.0f KB\t%7.1f cycles\t%6.1f ns\n",
               sizes[i] / 1024.0, lat_cyc, lat_ns);
        free(buf);
    }
}

int main(void) {
    printf("============================================================\n");
    printf("CPU Memory Latency Benchmark (Pointer Chasing)\n");
    printf("============================================================\n\n");

    /* Sequential access (stride = cache line) shows cache hierarchy latency */
    test_stride(CACHELINE, "Sequential access (64B stride)");

    /* Random access within a page */
    test_stride(4096, "Page-level random (4KB stride)");

    return 0;
}
