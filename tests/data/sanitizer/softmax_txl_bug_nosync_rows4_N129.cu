
#ifdef __CUDACC_RTC__
  #include <cuda/std/cstdint>
  using cuda::std::uint8_t;
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::int8_t;
  using cuda::std::int16_t;
  using cuda::std::int32_t;
  using cuda::std::int64_t;

  #include <cuda/std/type_traits>
  namespace std {
    using cuda::std::is_same;
    using cuda::std::is_same_v;
    using cuda::std::is_integral;
    using cuda::std::is_signed;
    using cuda::std::is_unsigned;
    using cuda::std::is_floating_point;
    using cuda::std::enable_if;
    using cuda::std::conditional;
  }

  // NVRTC uses asm/volatile instead of __asm__/__volatile__ (gcc extension).
  #ifndef __asm__
  #define __asm__ asm
  #endif
  #ifndef __volatile__
  #define __volatile__ volatile
  #endif
#else
  #include <cstdint>
  #include <type_traits>
  #include <cuda.h>
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
#endif
__forceinline__ __device__ void tvm_builtin_ptx_st_global_f32(const void* __addr, float __value) {
  asm volatile("st.global.f32 [%0], %1;" :  : "l"(__addr), "f"(__value) : "memory");
}

__forceinline__ __device__ void tvm_builtin_cuda_cta_sync() {
    __syncthreads();
}
__forceinline__ __device__ void tvm_builtin_ptx_ld_shared_f32(float& __d, uint32_t __addr) {
  asm volatile("ld.shared.f32 %0, [%1];" : "=f"(__d) : "r"(__addr) : "memory");
}
__forceinline__ __device__ void tvm_builtin_ptx_st_shared_f32(uint32_t __addr, float __value) {
  asm volatile("st.shared.f32 [%0], %1;" :  : "r"(__addr), "f"(__value) : "memory");
}

__forceinline__ __device__ unsigned int tvm_builtin_cuda_cvta_generic_to_shared(void* p) {
    return __cvta_generic_to_shared(p);
}
__forceinline__ __device__ void tvm_builtin_ptx_ld_global_f32(float& __d, const void* __addr) {
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(__d) : "l"(__addr) : "memory");
}
extern "C" __global__ void __launch_bounds__(128) softmax_txl_kernel(float* __restrict__ x_ptr, float* __restrict__ y_ptr);
extern "C" __global__ void __launch_bounds__(128) softmax_txl_kernel(float* __restrict__ x_ptr, float* __restrict__ y_ptr) {
  int warp_id_in_cta = __shfl_sync((uint)4294967295, (((int)threadIdx.x) >> 5), 0, 32);
  int bx = ((int)blockIdx.x);
  int warp_id = warp_id_in_cta;
  int lane_id = (((int)threadIdx.x) & 31);
  int thread_id = ((int)threadIdx.x);
  extern __shared__ __align__(64) uchar v_ptr[];
  float* v = ((float*)((float*)(((char*)v_ptr) + 0)));
  alignas(64) float v_ptr_1[1];
  alignas(64) float v_ptr_2[1];
  alignas(64) float v_ptr_3[1];
  alignas(64) float v_ptr_4[1];
  v_ptr_4[0] = -0x1.ff933c78cdfadp+127f/*-3.400000e+38*/;
  for (int v_1 = 0; v_1 < 2; ++v_1) {
    if (((v_1 * 128) + ((int)threadIdx.x)) < 129) {
      tvm_builtin_ptx_ld_global_f32(v_ptr_1[0], (&(x_ptr[(((((int)blockIdx.x) * 129) + (v_1 * 128)) + ((int)threadIdx.x))])));
        float v_ = v_ptr_4[0];
        float v__1 = v_ptr_1[0];
      v_ptr_4[0] = (((v_ > v__1) || (v_ != v_)) ? v_ : v__1);
    }
  }
  tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), v_ptr_4[0]);
  tvm_builtin_cuda_cta_sync();
  if (((int)threadIdx.x) < 64) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 64)]))));
      float v__2 = v_ptr_2[0];
      float v__3 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__2 > v__3) || (v__2 != v__2)) ? v__2 : v__3));
  }
  if (((int)threadIdx.x) < 32) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 32)]))));
      float v__4 = v_ptr_2[0];
      float v__5 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__4 > v__5) || (v__4 != v__4)) ? v__4 : v__5));
  }
  if (((int)threadIdx.x) < 16) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 16)]))));
      float v__6 = v_ptr_2[0];
      float v__7 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__6 > v__7) || (v__6 != v__6)) ? v__6 : v__7));
  }
  if (((int)threadIdx.x) < 8) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 8)]))));
      float v__8 = v_ptr_2[0];
      float v__9 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__8 > v__9) || (v__8 != v__8)) ? v__8 : v__9));
  }
  if (((int)threadIdx.x) < 4) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 4)]))));
      float v__10 = v_ptr_2[0];
      float v__11 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__10 > v__11) || (v__10 != v__10)) ? v__10 : v__11));
  }
  if (((int)threadIdx.x) < 2) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 2)]))));
      float v__12 = v_ptr_2[0];
      float v__13 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__12 > v__13) || (v__12 != v__12)) ? v__12 : v__13));
  }
  if (((int)threadIdx.x) < 1) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 1)]))));
      float v__14 = v_ptr_2[0];
      float v__15 = v_ptr_3[0];
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (((v__14 > v__15) || (v__14 != v__14)) ? v__14 : v__15));
  }
  alignas(64) float v_ptr_5[1];
  tvm_builtin_ptx_ld_shared_f32(v_ptr_5[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[0]))));
  tvm_builtin_cuda_cta_sync();
  alignas(64) float v_ptr_6[1];
  v_ptr_6[0] = 0x0p+0f/*0.000000e+00*/;
  for (int v_2 = 0; v_2 < 2; ++v_2) {
    if (((v_2 * 128) + ((int)threadIdx.x)) < 129) {
      tvm_builtin_ptx_ld_global_f32(v_ptr_1[0], (&(x_ptr[(((((int)blockIdx.x) * 129) + (v_2 * 128)) + ((int)threadIdx.x))])));
      v_ptr_6[0] = (v_ptr_6[0] + expf((v_ptr_1[0] - v_ptr_5[0])));
    }
  }
  tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), v_ptr_6[0]);
  tvm_builtin_cuda_cta_sync();
  if (((int)threadIdx.x) < 64) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 64)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 32) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 32)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 16) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 16)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 8) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 8)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 4) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 4)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 2) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 2)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  if (((int)threadIdx.x) < 1) {
    tvm_builtin_ptx_ld_shared_f32(v_ptr_2[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))));
    tvm_builtin_ptx_ld_shared_f32(v_ptr_3[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[(((int)threadIdx.x) + 1)]))));
    tvm_builtin_ptx_st_shared_f32(tvm_builtin_cuda_cvta_generic_to_shared((&(v[((int)threadIdx.x)]))), (v_ptr_2[0] + v_ptr_3[0]));
  }
  alignas(64) float v_ptr_7[1];
  tvm_builtin_ptx_ld_shared_f32(v_ptr_7[0], tvm_builtin_cuda_cvta_generic_to_shared((&(v[0]))));
  tvm_builtin_cuda_cta_sync();
  for (int v_3 = 0; v_3 < 2; ++v_3) {
    if (((v_3 * 128) + ((int)threadIdx.x)) < 129) {
      tvm_builtin_ptx_ld_global_f32(v_ptr_1[0], (&(x_ptr[(((((int)blockIdx.x) * 129) + (v_3 * 128)) + ((int)threadIdx.x))])));
      tvm_builtin_ptx_st_global_f32((&(y_ptr[(((((int)blockIdx.x) * 129) + (v_3 * 128)) + ((int)threadIdx.x))])), (expf((v_ptr_1[0] - v_ptr_5[0])) / v_ptr_7[0]));
    }
  }
}

