# cuda round 2

model=glm-5.3 in=1733 out=4768 cost=0.02255 latency=31.7s finish=stop effort=high usage={"completion_tokens": 4768, "completion_tokens_details": null, "prompt_tokens": 1733, "prompt_tokens_details": {"audio_tokens": null, "cache_write_tokens": null, "cached_tokens": 320, "image_tokens": null, "text_tokens": null}, "reasoning_tokens": 3565, "total_tokens": 6501}

## reasoning
The verifier does exact real arithmetic over rationals, checking equivalence. Rational coefficient overflow in i128 happens because of the "online" rescaling trick: EXP(m - mn) multiplicatively creates huge rational expression trees... Actually the verifier treats EXP as an uninterpreted function symbol presumably, so EXP(a)*EXP(b)... Actually with exact real arithmetic and EXP as an op, expressions like exp(v0-mn)*exp(m-m')... The issue: the online approach creates products of many exponentials, causing rational coefficient explosion (i128 overflow) in the symbolic computation.

Fix: use a simpler two-pass or three-pass approach where each output is exp(x[i]-M)/L with M computed by pure max reductions (fmaxf) and L = sum of exp(x[i]-M) — no exp-based rescaling multiplications. That avoids products of exponentials. Sum of exps should keep coefficients small (just sums, each exp term coefficient 1).

Also avoid FMA maybe? fmaf is fine probably. But to be safe, remove fmaf and just use multiplication.

Also avoid re-reading x? Second read of x is fine.

Potential issue: 1.0f/L multiplication vs division. Reference probably divides. Exact real arithmetic: y = e / L vs e * (1/L) differ in exact rational arithmetic (1/L not exact). The verifier with exact real arithmetic: division is exact in rationals, so e/L == e/L. But e*(1/L) with 1/L computed as exact rational division is also equal... but the verifier models floating point? "exact real arithmetic" — if it uses exact reals with IEEE operations modeled, 1.0f/L rounds, then e*(rounded inv) ≠ e/L. To be safe, compute yr[i] = EXP(...)/L directly using division, matching reference y = exp/L. The task statement's formula: exp(...)/sum. So use division.

Similarly, summation order: L computed as warp butterflies then warp sums — the exact arithmetic sums differently than reference's sequential sum? If verifier checks equivalence under exact real arithmetic, additions are exact so order doesn't matter (addition is associative/commutative in reals). But rounding of float ops: if verifier uses "exact real arithmetic" it means it treats each float op as exact? Then order doesn't matter. The failure was coefficient overflow, i.e., the expression representation blew up due to products of many exps.

So the fix: eliminate the online rescaling (products like l * EXP(m - mn)). Compute:
- Pass 1: reduce max M only (no exps).
- Pass 2: compute s = EXP(x[i]-M) per element, reduce sum L.
- Pass 3: y[i] = EXP(x[i]-M) / L.

Three passes, two reads of x. No multiplication of exponentials at all. Sums of exponentials only. That should keep rational coefficients tiny.

Also, division: use / L directly.

Also careful: N>=1; with tid >= N threads, they should contribute -FLT_MAX to max and 0 to sum. With butterfly shuffles all lanes participate — fine since inactive threads have m=-FLT_MAX, l=0.

Also fmaf: replace with plain multiplication? fmaf(a,b,c) exact-real = a*b+c, same as l*exp + s. But in float arithmetic it's different. With exact real arithmetic verifier, both equal. But to be safe against the verifier modeling IEEE semantics... The error was overflow of i128 rational coefficients — that's about expression size, not rounding. The verifier likely uses exact rationals where EXP is treated exactly (maybe as a symbolic value). Products of exps with different arguments create huge rational coefficients when exp values are modeled as... hmm, actually if EXP is uninterpreted/symbolic, a product like exp(a)*exp(b) is just a monomial — coefficient 1. Overflow of rational coefficients... maybe the verifier evaluates with rational values of exp? exp(100) as exact rational has enormous numerator — i128 overflow. Wait but x ~ 100, exp(100) is a huge number but representable as rational with big numerator? exp(100) isn't rational exactly. Unless verifier uses intervals or fixed bit-width rationals.

Hmm. "rational coefficient overflowed i128" — the verifier represents numbers as rationals with i128 numerator/denominator. Products of many exps of args in [-200, 0] create tiny rationals: exp(-200) ~ denominator 10^86, still fits i128 (max ~1.7e38). Hmm 10^86 > 1.7e38! So exp(-100) alone ~ 3.7e-44, numerator/denominator... exp(-100) as rational? Not rational. Maybe EXP is treated symbolically and multiplied terms combine? Or maybe the verifier computes exp exactly as rational? exp of rational isn't rational.

More likely: the verifier treats EXP as uninterpreted function, and expressions are polynomials/rationals in these uninterpreted terms. Coefficients are i128 rationals. Where would coefficients overflow? The online scheme: l = l*EXP(m-mn) + EXP(v-mn)... with uninterpreted EXP, m-mn involves fmaxf which is a max function — piecewise. The max function makes it piecewise-rational; case explosion: with fmaxf, the expression becomes piecewise over cases, and each branch has coefficients... The number of pieces grows, and coefficients could overflow when combining branches? Actually "rational coefficient overflowed i128" — maybe coefficients from binomial expansions in Horner-like rescaling: l_new = l*e1 + s where e1 = EXP(m-mn) is a product of max-conditional args. Hmm.

Simplest interpretation: coefficients blow up due to nested multiplicative rescaling; removing it fixes. Alternatively, coefficients like 2^-149 (denormal) are fine in i128.

Actually maybe simpler: the verifier models floats as exact rationals with IEEE values, and computes exp exactly as... no.

I'll just do the clean three-pass design: pure max reduction (fmaxf only, no exp), then sum of exps, then division. This minimizes expression complexity. Also avoid 1/L multiply; use division per element to match reference formula exactly.

Also avoid fmaf (use * and +) since fmaf might complicate verification; exact arithmetic makes them equal, but let's be safe and simple.

Also, the sum reduction order: with exact reals, sum order doesn't matter. Keep butterfly.

One more concern: -FLT_MAX initial m. fmaxf(-FLT_MAX, v) = v. Fine. If all threads idle? N>=1 so at least thread 0 has an element. Even if row max is -FLT_MAX... fine.

Edge: for N<128, threads with i>=N skip loop, contribute m=-FLT_MAX, l=0. EXP(m-M) never computed in the sum pass since we only compute for i<N. Good.

But wait — in pass 2, per-element sums: s accumulates EXP(x[i]-M) for this thread's elements. Then butterfly reduce. Then pass 3 recomputes EXP(x[i]-M) again. Coefficients: each is a sum of N uninterpreted-ish exp terms with coefficient 1. Fine.

Could the verifier object to reading x twice? No.

Also keep the requirement: control flow depends only on threadIdx/N etc. Yes.

Let me also consider whether the division vs multiplication matters: reference is y = exp/sum. Use `yr[i] = EXP(xr[i]-M) / L;`. Safe.

Also should I keep 4-wide unrolling in pass 2? Sums like ((e0+e1)+(e2+e3)) — with exact arithmetic associativity, order irrelevant. Keep for performance. But hmm — if the verifier models IEEE rounding, order matters and the check would fail regardless... it says "exact real arithmetic", so order fine.

Actually wait: another potential source of i128 overflow: FLT_MAX = 3.4e38 fits in i128. exp arguments like m - M where m=-FLT_MAX and M large: if EXP evaluated exactly... For inactive threads we never compute EXP. In pass 1 no exp at all. Good.

Hmm, but one more: in pass 2, could M - x[i] be huge, e.g., x has values ±100: EXP(-200) ≈ 1.38e-87 — if the verifier evaluates exp as a rational approximation with i128, 10^-87 denominator... i128 max ≈ 1.7e38. So denominators of e-87 overflow i128! But the reference also computes exp(-200)... reference presumably uses the same, so both overflow? The error is about the equivalence check's coefficient overflow — the original kernel may have produced products like exp(-100)*exp(-100) = e-200+ with denominator 10^173, definitely overflowing. The reference itself has just single exps, e-87 denominators which... would also overflow i128? Unless verifier normalizes or the reference passes fine (it said "equivalence check failed: rational coefficient overflowed" — maybe the kernel side overflowed).

With max-subtraction, exp args ∈ [x_min - M, 0]. If x range is 200, exp(-200)~1.4e-87. As a rational with i128 — if verifier represents floats as rationals from float values, exp(-200) as a float is fine (float value is a dyadic rational, small). Hmm, floats are dyadic rationals: exp(-200) float ≈ 2^-289, denominator 2^289 overflows i128? 2^127 max for i128... 2^-289 needs numerator/denominator with denominator 2^289 — doesn't fit. Unless verifier keeps floats as their exact float64 or uses scaling. But the reference softmax does the same exp(x-max), so the reference presumably verifies OK. The distinguishing feature of my kernel vs reference: multiplicative rescaling (products of exps) and possibly the fmaf. So removing those is the fix.

Actually one more subtlety: my kernel multiplied by invL = 1.0f/L — the reciprocal introduces an extra division → coefficient growth? e*(1/L): 1/L is a rational, product of two rationals — could overflow if L tiny. But L >= 1. Fine either way; use direct division.

Also, the online pass had `EXP(m - mn)` where m - mn could be up to 200 (positive? no, m - mn ≤ 0 since mn ≥ m). exp(negative) products → tiny values → denominator overflow in exact rational representation. E.g., l*EXP(m-mn) with EXP(-200) each iteration, multiplied across many iterations: for N=256, up to ~2 iterations per thread... exp(-200)*exp(-200) = 2^-578. Definitely overflow. That's the bug! The reference never multiplies exps. So the fix: never multiply exponentials together.

Design the fixed kernel:

```cuda
#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + (size_t)blockIdx.x * N;
    float* yr = y + (size_t)blockIdx.x * N;

    // Pass 1: max
    float m = -FLT_MAX;
    int i = tid;
    for (; i < N; i += 128) m = fmaxf(m, xr[i]);

    // warp reduce max
    #pragma unroll
    for (int off=16; off; off>>=1) m = fmaxf(m, __shfl_xor_sync(~0u, m, off));
    __shared__ float smax[4], ssum[4];
    if (lane==0) smax[warp]=m;
    __syncthreads();
    const float M = fmaxf(fmaxf(smax[0],smax[1]), fmaxf(smax[2],smax[3]));
    __syncthreads(); // before reusing? we use separate arrays, fine; but need sync before writing ssum? smax and ssum distinct, and all threads read smax before... they read smax into M; then pass 2 writes ssum — no hazard since different array. But lane==0 writes ssum after butterfly; other threads already read smax. No conflict. Actually need __syncthreads() between reading smax and... no, ssum write doesn't touch smax. Fine. But careful: warp reduce of sum then lane 0 writes ssum[warp]; must ensure all threads finished reading smax? Not related. OK.

    // Pass 2: sum of exps
    float l = 0.0f;
    i = tid;
    for (; i < N; i += 128) l += EXP(xr[i] - M);
    #pragma unroll
    for (int off=16; off; off>>=1) l += __shfl_xor_sync(~0u, l, off);
    if (lane==0) ssum[warp]=l;
    __syncthreads();
    const float L = ssum[0]+ssum[1]+ssum[2]+ssum[3];

    // Pass 3
    i = tid;
    for (; i < N; i += 128) yr[i] = EXP(xr[i] - M) / L;
}
```

Wait — potential sync hazard: after `if (lane==0) ssum[warp]=l; __syncthreads();` read L. Then pass 3. Fine.

But there's a subtle hazard: between pass1's `__syncthreads()` (after writing smax) and reading smax — I read M right after sync, fine. Then in pass 2, lane 0 writes ssum — no one reads smax after? All threads computed M already before pass 2. Fine, no need for extra sync before writing ssum since it's a different array.

Should I keep the 4x unrolled strided loads for performance? It adds complexity but performance requirement says "Aim for high performance". However, verifier errors were about arithmetic, unrolling is fine. But to minimize risk of verifier issues, keep it moderately simple but still coalesced and reasonably fast. I could keep the unrolled version. Strided i, i+128, i+256, i+384 loads are coalesced (consecutive tids access consecutive addresses). Let me keep 4x unrolling for passes. It doesn't introduce exp products.

Hmm, one concern with unrolled sum: l = e0+e1+e2+e3 combined... fine.

For pass 1 max unroll: m = fmaxf(fmaxf(fmaxf(m,v0),v1),fmaxf(v2,v3)).

Edge N=1..: while (N - i > 384) requires i≥0 and works: for tid=0, N=500: N-0=500>384 → process 0,128,256,384; i=512; N-512 negative... int: N-i negative, not >384, exit. Then remainder loop. But careful: "N - i > 384" — i can exceed N, N-i negative, fine. Also i starts as tid≥0, so N-i ≤ N. OK.

Also note: could i overflow int? i increments by 512 until N-i ≤ 384, so i ≤ N+something; N ≤ INT_MAX, i bounded by N+512 roughly. For N near INT_MAX, i+384 could overflow? i < N always in unrolled body (since N-i>384 ⇒ i < N-384), so i+384 < N ≤ INT_MAX. Fine. size_t indexing for row offset: blockIdx.x * N could exceed int — use size_t cast. Yes did.

Division: reference divides by sum. Good.

Should I worry that reference might compute y = exp(x-max)*(1/sum)? The problem statement says exp/sum. Use division.

Also must not use fmaf? fmaf(l, EXP(...), s) — replaced. Fine.

One more verifier consideration: EXP(xr[i] - M) computed in pass 2 and pass 3 — identical expressions, fine.

Also -FLT_MAX: fmaxf chains fine.

Write final answer with exactly one cuda code block.

## content
```cuda
// Row-wise softmax for sm_80 (A100).
// Launch: gridDim.x = rows, blockDim.x = 128 (one CTA per row).
//
// Verification fix: the previous version used the "online softmax" trick,
// rescaling the running sum with factors like EXP(m - m_new). That multiplies
// exponentials together, and products such as EXP(-200)*EXP(-200) blow up the
// exact-rational (i128) coefficients used by the formal verifier. This version
// performs three clean passes with NO products of exponentials:
//   1) reduce the row maximum M (fmaxf only, no EXP),
//   2) reduce L = sum_i EXP(x_i - M) (a plain sum of single exponentials),
//   3) write y_i = EXP(x_i - M) / L (direct division, matching the reference).
// All control flow depends only on threadIdx/blockIdx/blockDim/N.

#include "common.cuh"
#include <cfloat>

extern "C" __global__ void softmax_glm(const float* __restrict__ x, float* __restrict__ y, int N)
{
    const int tid  = threadIdx.x;          // 0..127
    const int warp = tid >> 5;             // 0..3  (blockDim.x == 128 -> 4 warps)
    const int lane = tid & 31;

    const float* __restrict__ xr = x + (size_t)blockIdx.x * (size_t)N;
    float*       __restrict__ yr = y + (size_t)blockIdx.x * (size_t)N;

    __shared__ float smax[4];
    __shared__ float ssum[4];

    // ------------------------------------------------------------------
    // Pass 1: block-wide maximum M of the row (pure fmaxf, no exponentials).
    // 4 strided, fully coalesced loads per iteration.
    // ------------------------------------------------------------------
    float m = -FLT_MAX;
    int i = tid;
    while (N - i > 384) {
        const float v0 = xr[i];
        const float v1 = xr[i + 128];
        const float v2 = xr[i + 256];
        const float v3 = xr[i + 384];
        m = fmaxf(fmaxf(fmaxf(m, v0), v1), fmaxf(v2, v3));
        i += 512;
    }
    for (; i < N; i += 128)
        m = fmaxf(m, xr[i]);

    // warp-level max reduction (xor butterfly: every lane gets the result)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));

    if (lane == 0) smax[warp] = m;
    __syncthreads();

    // Global row max (every thread computes it; identical value for all).
    const float M = fmaxf(fmaxf(smax[0], smax[1]), fmaxf(smax[2], smax[3]));

    // ------------------------------------------------------------------
    // Pass 2: block-wide sum L = sum_i EXP(x_i - M).
    // Only additions of single exponentials (arguments <= 0); no products.
    // ------------------------------------------------------------------
    float l = 0.0f;
    i = tid;
    while (N - i > 384) {
        const float e0 = EXP(xr[i]       - M);
        const float e1 = EXP(xr[i + 128] - M);
        const float e2 = EXP(xr[i + 256] - M);
        const float e3 = EXP(xr[i + 384] - M);
        l = (e0 + e1) + (e2 + l);
        i += 512;
    }
    for (; i < N; i += 128)
        l = l + EXP(xr[i] - M);

    // warp-level sum reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        l = l + __shfl_xor_sync(0xffffffffu, l, off);

    if (lane == 0) ssum[warp] = l;
    __syncthreads();

    // Total sum (L >= 1, since the max element contributes EXP(0) = 1).
    const float L = (ssum[0] + ssum[1]) + (ssum[2] + ssum[3]);

    // ------------------------------------------------------------------
    // Pass 3: y_i = EXP(x_i - M) / L  (second coalesced read of x).
    // Direct division matches the reference formula exactly; no reciprocal
    // is precomputed, so no extra rational operations are introduced.
    // ------------------------------------------------------------------
    i = tid;
    while (N - i > 384) {
        yr[i]       = EXP(xr[i]       - M) / L;
        yr[i + 128] = EXP(xr[i + 128] - M) / L;
        yr[i + 256] = EXP(xr[i + 256] - M) / L;
        yr[i + 384] = EXP(xr[i + 384] - M) / L;
        i += 512;
    }
    for (; i < N; i += 128)
        yr[i] = EXP(xr[i] - M) / L;
}
```
