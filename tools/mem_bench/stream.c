/*
 * DRAM Memory Bandwidth Benchmark
 *
 * Two complementary tests that measure real DRAM bandwidth:
 *
 *   1. STREAM (sequential): best-case bandwidth with prefetcher help.
 *      Uses large heap-allocated arrays (default: 1.6 GB each, 4.8 GB total)
 *      to ensure the working set exceeds L3 cache.
 *
 *   2. Random-access: bandwidth under pointer-chasing across a large buffer.
 *      Defeats hardware prefetchers and measures true random-access DRAM throughput.
 *
 * Compile: cc -O3 -march=native -fopenmp -o stream stream.c -lm
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>

/* ── tunables ─────────────────────────────────────────────────────────── */
#ifndef N
#define N  200000000   /* 200M doubles = 1.6 GB per array, 4.8 GB total */
#endif

#ifndef NTIMES
#define NTIMES  20
#endif

#ifndef RAND_N
#define RAND_N  ((size_t)4UL * 1024 * 1024 * 1024 / 8)  /* 4 GB of uint64_t */
#endif

#define CACHELINE  64

/* ── helpers ──────────────────────────────────────────────────────────── */

static double walltime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

/* ── STREAM (sequential, best-case) ───────────────────────────────────── */

static void run_stream(void) {
    printf("-----------------------------------------------------\n");
    printf("STREAM Bandwidth Benchmark (sequential, prefetcher-friendly)\n");

    /* Use heap to allow large sizes */
    double *a = (double *)aligned_alloc(64, N * sizeof(double));
    double *b = (double *)aligned_alloc(64, N * sizeof(double));
    double *c = (double *)aligned_alloc(64, N * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "STREAM: allocation failed (%zu MB per array)\n",
                N * sizeof(double) / (1024 * 1024));
        free(a); free(b); free(c);
        return;
    }

    size_t bytes = N * sizeof(double);
    printf("Array size = %zu elements (%.0f MB each, %.1f GB total)\n",
           (size_t)N, bytes / 1e6, (bytes * 3.0) / 1e9);
    printf("NTIMES = %d\n", NTIMES);
    printf("-----------------------------------------------------\n");

    double scalar = 3.0;

    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    double best[4] = {1e12, 1e12, 1e12, 1e12};

    for (int k = 0; k < NTIMES; k++) {
        double t;

        /* COPY: c = a */
        t = walltime();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) c[i] = a[i];
        t = walltime() - t;
        if (t < best[0]) best[0] = t;

        /* SCALE: c = scalar * b */
        t = walltime();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) c[i] = scalar * b[i];
        t = walltime() - t;
        if (t < best[1]) best[1] = t;

        /* ADD: c = a + b */
        t = walltime();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) c[i] = a[i] + b[i];
        t = walltime() - t;
        if (t < best[2]) best[2] = t;

        /* TRIAD: c = a + scalar * b */
        t = walltime();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) c[i] = a[i] + scalar * b[i];
        t = walltime() - t;
        if (t < best[3]) best[3] = t;
    }

    printf("%-12s %10s %12s %12s\n", "Operation", "Time(ms)", "MB/s", "GB/s");
    printf("-----------------------------------------------------\n");

    const char *names[4] = {"COPY", "SCALE", "ADD", "TRIAD"};
    int reads[4]  = {1, 1, 2, 2};
    int writes[4] = {1, 1, 1, 1};

    for (int i = 0; i < 4; i++) {
        double bw = (bytes * (reads[i] + writes[i])) / (best[i] * 1e9);
        printf("%-12s %10.3f %10.1f %10.2f\n",
               names[i], best[i] * 1000.0, bw * 1000.0, bw);
    }
    printf("-----------------------------------------------------\n");

    /* Use results to prevent dead-code elimination */
    volatile double sink = c[N - 1];
    (void)sink;

    free(a); free(b); free(c);
}

/* ── Random-access bandwidth (defeats prefetchers) ─────────────────────── */

/*
 * Fill buf with a random permutation at cacheline granularity.
 * Uses a simple LCG; each cacheline-sized element points to the next.
 */
static void fill_random_chain(uint64_t *buf, size_t n_elems) {
    /* n_elems = total bytes / 64 */
    /* Fisher-Yates style: assign each slot a random target */
    for (size_t i = 0; i < n_elems - 1; i++) {
        buf[i] = (uint64_t)(buf + i + 1);  /* sequential for now */
    }
    buf[n_elems - 1] = (uint64_t)buf;

    /* Shuffle using PRNG with fixed seed for reproducibility */
    unsigned long seed = 42;
    for (size_t i = n_elems - 1; i > 0; i--) {
        seed = seed * 1103515245 + 12345;
        size_t j = seed % (i + 1);
        uint64_t tmp = buf[i];
        buf[i] = buf[j];
        buf[j] = tmp;
    }

    /* Re-link after shuffle: each entry stores the pointer to the next */
    for (size_t i = 0; i < n_elems; i++) {
        uint64_t target_addr = buf[i];
        if (target_addr == 0) continue;
        /* Resolve back to index */
        size_t target_idx = (target_addr - (uint64_t)buf) / 8;
        if (target_idx < n_elems) {
            buf[i] = (uint64_t)(buf + target_idx);
        }
    }
}

__attribute__((noinline))
static double random_read_bw(void *buf, size_t total_bytes, int n_iters) {
    size_t n_elems = total_bytes / 64;  /* one pointer per cacheline */

    /* Warmup */
    volatile uint64_t *cursor = (volatile uint64_t *)buf;
    for (size_t i = 0; i < n_elems / 10; i++) {
        cursor = (volatile uint64_t *)*cursor;
    }

    double best = 1e12;
    for (int iter = 0; iter < n_iters; iter++) {
        cursor = (volatile uint64_t *)buf;
        double t0 = walltime();
        for (size_t i = 0; i < n_elems; i++) {
            cursor = (volatile uint64_t *)*cursor;
        }
        double t = walltime() - t0;
        if (t < best) best = t;
    }
    (void)cursor;

    /* bytes read = n_elems * 64 (one cacheline per hop) */
    double bytes_read = (double)n_elems * 64.0;
    return bytes_read / (best * 1e9);  /* GB/s */
}

static void run_random_bw(void) {
    printf("\n-----------------------------------------------------\n");
    printf("Random-Access DRAM Bandwidth (pointer chasing, prefetcher-proof)\n");

    size_t sizes[] = {
        256UL * 1024 * 1024,    /* 256 MB */
        512UL * 1024 * 1024,    /* 512 MB */
        1024UL * 1024 * 1024,   /* 1 GB   */
        2048UL * 1024 * 1024,   /* 2 GB   */
        4096UL * 1024 * 1024,   /* 4 GB   */
    };
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iters = 5;

    printf("%-10s %12s %12s %12s\n", "Size", "Lat(ns)", "GB/s", "M ops/s");
    printf("-----------------------------------------------------\n");

    for (int si = 0; si < n_sizes; si++) {
        size_t bytes = sizes[si];
        uint64_t *buf = (uint64_t *)aligned_alloc(64, bytes);
        if (!buf) {
            printf("  %5.0f MB  (alloc failed)\n", bytes / 1e6);
            continue;
        }

        size_t n_elems = bytes / 64;  /* one pointer per cacheline */
        fill_random_chain(buf, n_elems);

        double bw = random_read_bw(buf, bytes, iters);

        /* Effective latency per access = time / ops = 1 / (ops/s) */
        double ops_per_sec = bw * 1e9 / 64.0;
        double lat_ns = 1e9 / ops_per_sec;

        printf("  %5.0f MB  %10.1f %10.2f %10.1f\n",
               bytes / 1e6, lat_ns, bw, ops_per_sec / 1e6);

        free(buf);
    }
    printf("-----------------------------------------------------\n");
    printf("Note: Random-access bandwidth (no prefetcher help) is typically\n");
    printf("      5-10x lower than STREAM. This is the true DRAM throughput\n");
    printf("      for irregular access patterns (pointer chasing, hash lookups).\n");
}

/* ── main ─────────────────────────────────────────────────────────────── */

int main(void) {
    run_stream();
    run_random_bw();
    return 0;
}
