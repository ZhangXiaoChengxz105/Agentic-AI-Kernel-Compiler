// Shared helpers for all hand-written kernels.
//
// EXP(x): VOLTA only understands the natural exponential when it appears as a call to an
// external, body-less function named __symexpf (the "symbolic exp hook" from the VOLTA paper).
// A normal expf() is expanded by nvcc into ex2.approx + range reduction, which VOLTA cannot
// treat as exp. So every kernel writes EXP(x) and we compile it two ways:
//   nvcc -DVOLTA -ptx ...   -> PTX for VOLTA   (EXP(x) = __symexpf(x), left as `call __symexpf`)
//   nvcc ...                -> real GPU build  (EXP(x) = expf(x))
#pragma once
#include <math.h>

#ifdef VOLTA
extern "C" __device__ float __symexpf(float x);
#define EXP(x) __symexpf(x)
#else
#define EXP(x) expf(x)
#endif
