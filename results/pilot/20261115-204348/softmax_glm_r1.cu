#include "common.cuh"

#define TPB 128
#define NWARP (TPB / 32)

// Full warp reduction for max (data-independent control flow: only shuffle lanes).
__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Full warp reduction for sum.
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Block-wide (128 threads = 4 warps) reductions via shared memory.
__inline__ __device__ float block_reduce_max(float v, float* sred, float* bcast) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < nwarp) ? sred[threadIdx.x] : -INFINITY;
        t = warp_reduce_max(t);
        if (threadIdx.x == 0) bcast[0] = t;
    }
    __syncthreads();
    return bcast[0];
}

__inline__ __device__ float block_reduce_sum(float v, float* sred, float* bcast) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = (int)blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sred[warp] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < nwarp) ? sred[threadIdx.x] : 0.0f;
        t = warp_reduce_sum(t);
        if (threadIdx.x == 0) bcast[0] = t;
    }
    __syncthreads();
    return bcast[0];
}

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N) {
    // One CTA per row; blockDim.x == 128.
    const size_t base = (size_t)blockIdx.x * (size_t)N;
    const float* __restrict__ xr = x + base;
    float* __restrict__       yr = y + base;

    __shared__ float sred[NWARP];  // per-warp partials
    __shared__ float bmax[1];      // broadcast slots (separate to avoid races)
    __shared__ float bsum[1];

    // ---- Pass 1: row maximum (stable for large |x|) ----
    float m = -INFINITY;
    for (int i = threadIdx.x; i < N; i += TPB)
        m = fmaxf(m, xr[i]);
    m = block_reduce_max(m, sred, bmax);

    // ---- Pass 2: exp(x - max); store in y, accumulate sum ----
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += TPB) {
        float e = EXP(xr[i] - m);
        yr[i] = e;
        s += e;
    }
    s = block_reduce_sum(s, sred, bsum);

    // ---- Pass 3: normalize in place (each thread touches only its own elements) ----
    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < N; i += TPB)
        yr[i] *= inv;
}
