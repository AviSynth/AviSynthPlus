// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://avisynth.nl

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

#include <avs/config.h>
#include "../core/internal.h"

#include <avs/alignment.h>
#include <avs/minmax.h>

#include "check_avx512.h" // compiler avx512 directives check
#include "resample_avx512.h"
#include <type_traits>

#include <immintrin.h> // Includes AVX-512 intrinsics

#include "resample_avx512.hpp"

// FIXME!!! Use constexpr, compiler BUG in v143/v145! if (!lessthan16bit) generates bad code.
// https://developercommunity.visualstudio.com/t/Silent-Bad-CodeGen:-Regression-in-Lambda/11030256
// Fixed In: 18.4.0

/**
 * Simulates _mm256_dpwssd_epi32 for CPUs without AVX512_VNNI (e.g., Xeon 613x).
 * Logic: For each 32-bit lane, it treats the inputs as pairs of 16-bit signed ints,
 * multiplies the pairs, adds them together, and adds the result to the accumulator.
 */
static AVS_FORCEINLINE __m256i _MM256_DPWSSD_EPI32_SIMUL(__m256i acc, __m256i a, __m256i b) {
#if defined(__AVX512VNNI__) || defined(__AVX_VNNI__)
  return _mm256_dpwssd_epi32(acc, a, b);
#else
  // vpmaddwd: (a_even * b_even) + (a_odd * b_odd) for each 32-bit slot
  __m256i product = _mm256_madd_epi16(a, b);
  // vpaddd: add the products to the existing accumulator
  return _mm256_add_epi32(acc, product);
#endif
}

/**
 * SIMD Optimization Strategy for Horizontal Resampling Kernel (Filter) Execution.
 *
 * This section details the performance optimization strategy based on the filter's kernel size,
 * ensuring high throughput across common resampling scenarios (upscaling/downscaling).
 *
 * I. Typical Kernel Sizes (Taps):
 * ----------------------------------------------------------------------------------------------------------------
 * Small (Best Case/Upscale): Taps = 4 (Bilinear/Spline36), 8 (Spline64/Lanczos 4).
 * Medium (Mild Downscale):   Taps = 12 (Lanczos 3, 2x downscale).
 * Large (Worst Case/Downscale): Taps = 16 to 32 (Spline64/Lanczos 4, 2x to 4x downscale).
 * ----------------------------------------------------------------------------------------------------------------
 *
 * II. Optimization Tiers (Template Specialization):
 *
 * 1. Dedicated, Fully Unrolled Paths (FixedFilterSize = 4, 8, 12):
 * - Purpose: Eliminate all loop overhead (setup/teardown, bounds checks).
 * - SIMD Choice: Uses __m128i/__m256i for efficiency with small loads, accumulating into __m512i.
 * - Performance Gain: Estimated 1.5x to 2.5x speedup over generic loops for small, common kernels.
 * - Micro-arch Benefit (i7-11700): Reduces uOps (instruction count) and avoids unnecessary register pressure.
 *
 * 2. Aligned VNNI Paths (FixedFilterSize = 16, 32, 48, 64...):
 * - Purpose: Maximize vector utilization for worst-case downscaling.
 * - SIMD Choice: Uses __m512i (AVX-512) processing 32 taps per iteration.
 * - VNNI Advantage: Uses _mm512_dpwssd_epi32 for fused 16-bit multiply + 32-bit accumulation.
 *
 * 3. Generic Path (FixedFilterSize = -1):
 * - Purpose: Handles all remaining unaligned or uncommon kernel sizes. Slower, but safe fallback.
 */

 // Helper to reduce a ZMM (16x int32) to a scalar int32 sum
// -----------------------------------------------------------------------------------------
// Helper: Reduce ZMM (16x int32) to scalar int32
// -----------------------------------------------------------------------------------------
AVS_FORCEINLINE static int32_t _mm512_reduce_add_epi32_compat(__m512i v) {
  /*
  __m256i v256 = _mm256_add_epi32(_mm512_extracti64x4_epi64(v, 0), _mm512_extracti64x4_epi64(v, 1));
  __m128i v128 = _mm_add_epi32(_mm256_castsi256_si128(v256), _mm256_extracti128_si256(v256, 1));
  v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
  v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(0, 3, 0, 1)));
  return _mm_cvtsi128_si32(v128);
  */
  return _mm512_reduce_add_epi32(v);
}

// These are the direct rewrite of the full-generic AVX2 h resamplers
// - resizer_h_avx512_generic_uint8_t and
// - resizer_h_avx512_generic_uint16_t<bool lessthan16bit>
// They are not any quicker than the AVX2 versions, but they serve as a base for further optimizations.
// The 512-bitness is not exploited, only have more registers, but they did not help. WIP.

// -----------------------------------------------------------------------------------------
// Core Processing: 4x16, 2x16 Taps
// -----------------------------------------------------------------------------------------
// taps16, 4 coeffs
template<typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_two_16pixels_core(const pixel_t * AVS_RESTRICT src, int begin1, int begin2, int i, const short* AVS_RESTRICT current_coeff, int filter_size, __m256i & result1, __m256i & result2, __m256i & shifttosigned) {
  __m256i data_1, data_2;

  if constexpr (sizeof(pixel_t) == 1) {
    data_1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i)));
    data_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i)));
  }
  else {
    data_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin1 + i));
    data_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin2 + i));
    if constexpr (!lessthan16bit) {
      data_1 = _mm256_add_epi16(data_1, shifttosigned);
      data_2 = _mm256_add_epi16(data_2, shifttosigned);
    }
  }

  // Aligned load is not OK for coeffs if filter size is only 4
  // Coeffs are aligned to 4 or 8 shorts, so alignment is 8 or 16 bytes, _m256i requires 32 bytes alignment.

  // assume 16 alignment

  __m256i coeff_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff));
  __m256i coeff_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff + filter_size));

  result1 = _MM256_DPWSSD_EPI32_SIMUL(result1, data_1, coeff_1); // vnni, not really a bottleneck here
  result2 = _MM256_DPWSSD_EPI32_SIMUL(result2, data_2, coeff_2);
}

// taps16, 4 coeffs
template<typename pixel_t, bool lessthan16bit>
AVS_FORCEINLINE static void process_four_16pixels_core(const pixel_t* AVS_RESTRICT src,
  int begin1, int begin2, int begin3, int begin4, int i, const short* AVS_RESTRICT current_coeff, int filter_size,
  __m256i& result1, __m256i& result2, __m256i& result3, __m256i& result4, __m256i& shifttosigned) {
  __m256i data_1, data_2, data_3, data_4;

  if constexpr (sizeof(pixel_t) == 1) {
    data_1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i)));
    data_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i)));
    data_3 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin3 + i)));
    data_4 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin4 + i)));
  }
  else {
    data_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin1 + i));
    data_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin2 + i));
    data_3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin3 + i));
    data_4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin4 + i));
    if constexpr (!lessthan16bit) {
      data_1 = _mm256_add_epi16(data_1, shifttosigned);
      data_2 = _mm256_add_epi16(data_2, shifttosigned);
      data_3 = _mm256_add_epi16(data_3, shifttosigned);
      data_4 = _mm256_add_epi16(data_4, shifttosigned);
    }
  }

  // Aligned load is not OK for coeffs if filter size is only 4
  // Coeffs are aligned to 4 or 8 shorts, so alignment is 8 or 16 bytes, _m256i requires 32 bytes alignment.

  // assume 16 alignment

  __m256i coeff_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff));
  __m256i coeff_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff + filter_size));
  __m256i coeff_3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff + 2 * filter_size));
  __m256i coeff_4 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff + 3 * filter_size));

  result1 = _MM256_DPWSSD_EPI32_SIMUL(result1, data_1, coeff_1);
  result2 = _MM256_DPWSSD_EPI32_SIMUL(result2, data_2, coeff_2);
  result3 = _MM256_DPWSSD_EPI32_SIMUL(result3, data_3, coeff_3);
  result4 = _MM256_DPWSSD_EPI32_SIMUL(result4, data_4, coeff_4);
}

// -----------------------------------------------------------------------------------------
// Helper: Unrolled Partial Core (4 or 8 Taps) XMM would be enough
// -----------------------------------------------------------------------------------------
// filter_size must be the aligned size, better named as filter_coeff_stride
template<typename pixel_t, bool lessthan16bit, int Taps>
AVS_FORCEINLINE static void process_two_partial_unrolled(const pixel_t* src, int begin1, int begin2, int offset, const short* coeff, int filter_size, __m256i& result1, __m256i& result2, __m256i& shifttosigned) {
  // Taps is 4 or 8.
  __m128i d1, d2;

  // Load Data
  if constexpr (sizeof(pixel_t) == 1) {
    if constexpr (Taps == 4) {
      d1 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin1 + offset)));
      d2 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin2 + offset)));
    }
    else { // 8
      d1 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + offset)));
      d2 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + offset)));
    }
  }
  else {
    if constexpr (Taps == 4) {
      d1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + offset));
      d2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + offset));
    }
    else { // 8
      d1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + offset));
      d2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + offset));
    }
    if constexpr (!lessthan16bit) {
      d1 = _mm_add_epi16(d1, _mm256_castsi256_si128(shifttosigned));
      d2 = _mm_add_epi16(d2, _mm256_castsi256_si128(shifttosigned));
    }
  }

  // Load Coeffs (Need to handle offset)
  __m128i c1, c2;
  if constexpr (Taps == 4) {
    c1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + offset));
    c2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + filter_size + offset));
  }
  else {
    c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + offset));
    c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + filter_size + offset));
  }

  // Calc
/*
  result1 = _mm512_zextsi128_si512(_mm_dpwssd_epi32(_mm512_castsi512_si128(result1), d1, c1));
  result2 = _mm512_zextsi128_si512(_mm_dpwssd_epi32(_mm512_castsi512_si128(result2), d2, c2));
*/
  __m256i c1_256 = _mm256_zextsi128_si256(c1);
  __m256i c2_256 = _mm256_zextsi128_si256(c2);
  __m256i d1_256 = _mm256_zextsi128_si256(d1);
  __m256i d2_256 = _mm256_zextsi128_si256(d2);

  result1 = _MM256_DPWSSD_EPI32_SIMUL(result1, d1_256, c1_256);
  result2 = _MM256_DPWSSD_EPI32_SIMUL(result2, d2_256, c2_256);

}

// -----------------------------------------------------------------------------------------
// Helper: Unrolled Partial Core (4 or 8 Taps) XMM would be enough
// -----------------------------------------------------------------------------------------
// filter_size must be the aligned size, better named as filter_coeff_stride
template<typename pixel_t, bool lessthan16bit, int Taps>
AVS_FORCEINLINE static void process_four_partial_unrolled(const pixel_t* src,
  int begin1, int begin2, int begin3, int begin4, int offset,
  const short* coeff, int filter_size,
  __m256i& result1, __m256i& result2, __m256i& result3, __m256i& result4, __m256i& shifttosigned) {
  // Taps is 4 or 8.
  __m128i d1, d2, d3, d4;

  // Load Data
  if constexpr (sizeof(pixel_t) == 1) {
    if constexpr (Taps == 4) {
      d1 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin1 + offset)));
      d2 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin2 + offset)));
      d3 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin3 + offset)));
      d4 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src + begin4 + offset)));
    }
    else { // 8
      d1 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + offset)));
      d2 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + offset)));
      d3 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3 + offset)));
      d4 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4 + offset)));
    }
  }
  else {
    if constexpr (Taps == 4) {
      d1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin1 + offset));
      d2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin2 + offset));
      d3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin3 + offset));
      d4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + begin4 + offset));
    }
    else { // 8
      d1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + offset));
      d2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + offset));
      d3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin3 + offset));
      d4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin4 + offset));
    }
    if constexpr (!lessthan16bit) {
      d1 = _mm_add_epi16(d1, _mm256_castsi256_si128(shifttosigned));
      d2 = _mm_add_epi16(d2, _mm256_castsi256_si128(shifttosigned));
      d3 = _mm_add_epi16(d3, _mm256_castsi256_si128(shifttosigned));
      d4 = _mm_add_epi16(d4, _mm256_castsi256_si128(shifttosigned));
    }
  }

  // Load Coeffs (Need to handle offset)
  __m128i c1, c2, c3, c4;
  if constexpr (Taps == 4) {
    c1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + offset));
    c2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + filter_size + offset));
    c3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + 2 * filter_size + offset));
    c4 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(coeff + 3 * filter_size + offset));
  }
  else {
    c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + offset));
    c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + filter_size + offset));
    c3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + 2 * filter_size + offset));
    c4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(coeff + 3 * filter_size + offset));
  }

  // Calc
/*
  result1 = _mm512_zextsi128_si512(_mm_dpwssd_epi32(_mm512_castsi512_si128(result1), d1, c1));
  result2 = _mm512_zextsi128_si512(_mm_dpwssd_epi32(_mm512_castsi512_si128(result2), d2, c2));
*/
  __m256i c1_256 = _mm256_zextsi128_si256(c1);
  __m256i c2_256 = _mm256_zextsi128_si256(c2);
  __m256i c3_256 = _mm256_zextsi128_si256(c3);
  __m256i c4_256 = _mm256_zextsi128_si256(c4);
  __m256i d1_256 = _mm256_zextsi128_si256(d1);
  __m256i d2_256 = _mm256_zextsi128_si256(d2);
  __m256i d3_256 = _mm256_zextsi128_si256(d3);
  __m256i d4_256 = _mm256_zextsi128_si256(d4);

  result1 = _MM256_DPWSSD_EPI32_SIMUL(result1, d1_256, c1_256);
  result2 = _MM256_DPWSSD_EPI32_SIMUL(result2, d2_256, c2_256);
  result3 = _MM256_DPWSSD_EPI32_SIMUL(result3, d3_256, c3_256);
  result4 = _MM256_DPWSSD_EPI32_SIMUL(result4, d4_256, c4_256);
}


// ---------------------------------------------------------------------------
// FULLY VECTORIZED TREE REDUCTION (AVX2/AVX-512VL)
// Input: 8x __m256i accumulators (r0 through r7)
// Output: __m256i with 8 final, rounded pixel sums (p0 through p7)
// ---------------------------------------------------------------------------

AVS_FORCEINLINE static __m256i reduce_8x256i_32i_tree(
  __m256i r0, __m256i r1, __m256i r2, __m256i r3,
  __m256i r4, __m256i r5, __m256i r6, __m256i r7,
  int rounder_scalar)
{
  // --- Round 1: Reduce pairs (8 elements -> 4 elements per vector) ---
  // VPHADDD on each pair. The results are still interleaved within the 128-bit blocks.
  __m256i sum01 = _mm256_hadd_epi32(r0, r1);
  __m256i sum23 = _mm256_hadd_epi32(r2, r3);
  __m256i sum45 = _mm256_hadd_epi32(r4, r5);
  __m256i sum67 = _mm256_hadd_epi32(r6, r7);

  // --- Round 2: Reduce quads (4 elements -> 2 elements per vector) ---
  // The final pixel sum is now split between the low 128-bit half and the high 128-bit half.
  __m256i sum0123 = _mm256_hadd_epi32(sum01, sum23);
  __m256i sum4567 = _mm256_hadd_epi32(sum45, sum67);

  // --- Round 3: Final Horizontal Reduction (Across 128-bit boundary) ---

  // 1. Add the low 128-bit half to the high 128-bit half to finalize the sum for P0-P7.
  // VPERM2I128 (0x01 swaps the 128-bit halves)
  __m256i hi_add0123 = _mm256_permute2f128_si256(sum0123, sum0123, 0x01);
  __m256i hi_add4567 = _mm256_permute2f128_si256(sum4567, sum4567, 0x01);

  // The final sums for P0-P3 are now consolidated into the low 4 lanes of final0123.
  __m256i final0123 = _mm256_add_epi32(sum0123, hi_add0123);
  __m256i final4567 = _mm256_add_epi32(sum4567, hi_add4567);

  // --- Round 4: Final Consolidation (P0-P7 into one __m256i) ---

  // 1. Extract the low 128 bits (which contain the P0-P3 sums)
  __m128i p0_p3 = _mm256_castsi256_si128(final0123);
  __m128i p4_p7 = _mm256_castsi256_si128(final4567);

  // 2. Assemble the two 128-bit blocks into the final 256-bit result (VINSERTI128).
  __m256i result_8x_32 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0_p3), p4_p7, 1);

  // --- Final Vectorized Rounding ---

  // Create the rounder vector
  __m256i rounder_v = _mm256_set1_epi32(rounder_scalar);

  // Apply rounding to all 8 pixels simultaneously (VPADDD)
  return _mm256_add_epi32(result_8x_32, rounder_v);
}


AVS_FORCEINLINE static __m256i reduce_8x128i_to_8x32i(
  __m128i r0, __m128i r1, __m128i r2, __m128i r3,
  __m128i r4, __m128i r5, __m128i r6, __m128i r7,
  int rounder_scalar)
{
  // --- Round 1: Reduce pairs (8 elements -> 4 elements per vector) ---
  // VPHADDD on each pair. The results are still interleaved within the 128-bit blocks.
  __m128i sum01 = _mm_hadd_epi32(r0, r1);
  __m128i sum23 = _mm_hadd_epi32(r2, r3);
  __m128i sum45 = _mm_hadd_epi32(r4, r5);
  __m128i sum67 = _mm_hadd_epi32(r6, r7);

  // --- Round 2: Reduce quads (4 elements -> 2 elements per vector) ---
  // The final pixel sum is now split between the low 128-bit half and the high 128-bit half.
  __m128i sum0123 = _mm_hadd_epi32(sum01, sum23);
  __m128i sum4567 = _mm_hadd_epi32(sum45, sum67);

  // 2. Assemble the two 128-bit blocks into the final 256-bit result (VINSERTI128).
  __m256i result_8x_32 = _mm256_inserti128_si256(_mm256_castsi128_si256(sum0123), sum4567, 1);

  // --- Final Vectorized Rounding ---

  // Create the rounder vector
  __m256i rounder_v = _mm256_set1_epi32(rounder_scalar);

  // Apply rounding to all 8 pixels simultaneously (VPADDD)
  return _mm256_add_epi32(result_8x_32, rounder_v);
}


// -----------------------------------------------------------------------------------------
// Wrapper for Two-Pixel Processing
// -----------------------------------------------------------------------------------------
template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit, int FixedFilterSize>
AVS_FORCEINLINE static void process_two_pixels_h_avx512(const pixel_t * AVS_RESTRICT src_ptr, int begin1, int begin2, const short* AVS_RESTRICT current_coeff,
  int filter_size, __m256i & result1, __m256i & result2, int kernel_size, __m256i & shifttosigned) {

  // filter_size here is the stride for coeffs, kernel_size is the actual number of taps to process.

  if constexpr (FixedFilterSize == 4) {
    process_two_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, 0, current_coeff, filter_size, result1, result2, shifttosigned);
    return;
  }
  if constexpr (FixedFilterSize == 8) {
    process_two_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, 0, current_coeff, filter_size, result1, result2, shifttosigned);
    return;
  }
  if constexpr (FixedFilterSize == 12) {
    process_two_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, 0, current_coeff, filter_size, result1, result2, shifttosigned);
    process_two_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, 8, current_coeff, filter_size, result1, result2, shifttosigned);
    return;
  }

  // 2. Large Kernel Loop (16-tap blocks)
  int i = 0;
  // We can use the FixedFilterSize to cap the loop if it's large (like 48, 64)
  int runtime_filter_size = (FixedFilterSize >= 1) ? FixedFilterSize : filter_size;
  int ksmod16 = (safe_aligned_mode && FixedFilterSize >= 16) ? (FixedFilterSize / 16 * 16) : (kernel_size / 16 * 16);

  for (; i < ksmod16; i += 16) {
    process_two_16pixels_core<pixel_t, lessthan16bit>(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2, shifttosigned);
  }

  // 3. Tail Handling
  // If we are in safe mode and FixedSize is a multiple of 32, we are done.
  if constexpr (safe_aligned_mode && (FixedFilterSize % 16 == 0) && FixedFilterSize > 0) return;

  int remaining = runtime_filter_size - i;
  if (remaining <= 0) return;

  // Process remaining using scalar fallbacks
  // Unrolled helpers for chunks of 8/4/scalar.

  // Chunk 8
  if (remaining >= 8) {
    process_two_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, i, current_coeff, filter_size, result1, result2, shifttosigned);
    i += 8;
    remaining -= 8;
  }
  if (remaining >= 4) {
    process_two_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, i, current_coeff, filter_size, result1, result2, shifttosigned);
    i += 4;
    remaining -= 4;
  }

  // Final scalar tail (1-3 pixels)
  while (remaining > 0) {
    int val1, val2;
    if constexpr (sizeof(pixel_t) == 1) {
      val1 = src_ptr[begin1 + i];
      val2 = src_ptr[begin2 + i];
    }
    else {
      val1 = src_ptr[begin1 + i];
      val2 = src_ptr[begin2 + i];
      if constexpr (!lessthan16bit) { val1 -= 32768; val2 -= 32768; }
    }
    int c1 = current_coeff[i];
    int c2 = current_coeff[filter_size + i];

    // Add to the first lane of the accumulator
    __m256i s1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val1 * c1));
    __m256i s2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val2 * c2));

    result1 = _mm256_add_epi32(result1, s1);
    result2 = _mm256_add_epi32(result2, s2);

    i++;
    remaining--;
  }
}

// -----------------------------------------------------------------------------------------
// Wrapper for Four-Pixel Processing
// -----------------------------------------------------------------------------------------
template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit, int FixedFilterSize>
AVS_FORCEINLINE static void process_four_pixels_h_avx512(const pixel_t* AVS_RESTRICT src_ptr,
  int begin1, int begin2, int begin3, int begin4, const short* AVS_RESTRICT current_coeff, int filter_size,
  __m256i& result1, __m256i& result2, __m256i& result3, __m256i& result4, int kernel_size, __m256i& shifttosigned) {

  // filter_size here is the stride for coeffs, kernel_size is the actual number of taps to process.

  if constexpr (FixedFilterSize == 4) {
    process_four_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, begin3, begin4, 0, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    return;
  }
  if constexpr (FixedFilterSize == 8) {
    process_four_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, begin3, begin4, 0, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    return;
  }
  if constexpr (FixedFilterSize == 12) {
    process_four_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, begin3, begin4, 0, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    process_four_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, begin3, begin4, 8, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    return;
  }

  // 2. Large Kernel Loop (16-tap blocks)
  int i = 0;
  // We can use the FixedFilterSize to cap the loop if it's large (like 48, 64)
  int runtime_filter_size = (FixedFilterSize >= 1) ? FixedFilterSize : filter_size;
  int ksmod16 = (safe_aligned_mode && FixedFilterSize >= 16) ? (FixedFilterSize / 16 * 16) : (kernel_size / 16 * 16);

  for (; i < ksmod16; i += 16) {
    process_four_16pixels_core<pixel_t, lessthan16bit>(src_ptr, begin1, begin2, begin3, begin4, i, current_coeff + i, filter_size, result1, result2, result3, result4, shifttosigned);
  }

  // 3. Tail Handling
  // If we are in safe mode and FixedSize is a multiple of 32, we are done.
  if constexpr (safe_aligned_mode && (FixedFilterSize % 16 == 0) && FixedFilterSize > 0) return;

  int remaining = runtime_filter_size - i;
  if (remaining <= 0) return;

  // Process remaining using scalar fallbacks
  // Unrolled helpers for chunks of 8/4/scalar.

  // Chunk 8
  if (remaining >= 8) {
    process_four_partial_unrolled<pixel_t, lessthan16bit, 8>(src_ptr, begin1, begin2, begin3, begin4, i, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    i += 8;
    remaining -= 8;
  }
  if (remaining >= 4) {
    process_four_partial_unrolled<pixel_t, lessthan16bit, 4>(src_ptr, begin1, begin2, begin3, begin4, i, current_coeff, filter_size, result1, result2, result3, result4, shifttosigned);
    i += 4;
    remaining -= 4;
  }

  // Final scalar tail (1-3 pixels)
  while (remaining > 0) {
    int val1, val2, val3, val4;
    if constexpr (sizeof(pixel_t) == 1) {
      val1 = src_ptr[begin1 + i];
      val2 = src_ptr[begin2 + i];
      val3 = src_ptr[begin3 + i];
      val4 = src_ptr[begin4 + i];
    }
    else {
      val1 = src_ptr[begin1 + i];
      val2 = src_ptr[begin2 + i];
      val3 = src_ptr[begin3 + i];
      val4 = src_ptr[begin4 + i];
      if constexpr (!lessthan16bit) { val1 -= 32768; val2 -= 32768; val3 -= 32768; val4 -= 32768; }
    }
    int c1 = current_coeff[i];
    int c2 = current_coeff[filter_size + i];
    int c3 = current_coeff[2 * filter_size + i];
    int c4 = current_coeff[3 * filter_size + i];

    // Add to the first lane of the accumulator
    __m256i s1 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val1 * c1));
    __m256i s2 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val2 * c2));
    __m256i s3 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val3 * c3));
    __m256i s4 = _mm256_zextsi128_si256(_mm_cvtsi32_si128(val4 * c4));

    result1 = _mm256_add_epi32(result1, s1);
    result2 = _mm256_add_epi32(result2, s2);
    result3 = _mm256_add_epi32(result3, s3);
    result4 = _mm256_add_epi32(result4, s4);

    i++;
    remaining--;
  }
}

// -----------------------------------------------------------------------------------------
// Main Block Processor, TWO_PIXELS_AT_ONCE for testing performance difference
// -----------------------------------------------------------------------------------------
template<bool is_safe, typename pixel_t, bool lessthan16bit, int FixedFilterSize>
AVS_FORCEINLINE static void process_sixteen_pixels_h_avx512(const pixel_t * src, int x, const short* current_coeff_base, int filter_size,
  int rounder_scalar, __m256i& shifttosigned, __m256i& clamp_limit,
  pixel_t* dst, ResamplingProgram* program)
{
  int run_filter_size_stride = (FixedFilterSize >= 1) ? FixedFilterSize : filter_size; // quasi constexpr if templated
  const short* AVS_RESTRICT current_coeff = current_coeff_base + x * run_filter_size_stride;
  const int unaligned_kernel_size = program->filter_size_real;

#ifdef TWO_PIXELS_AT_ONCE
  __m256i result0 = _mm256_setzero_si256();
  __m256i result1 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x], program->pixel_offset[x + 1], current_coeff, run_filter_size_stride, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result2 = _mm256_setzero_si256();
  __m256i result3 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 2], program->pixel_offset[x + 3], current_coeff, run_filter_size_stride, result2, result3, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result4 = _mm256_setzero_si256();
  __m256i result5 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 4], program->pixel_offset[x + 5], current_coeff, run_filter_size_stride, result4, result5, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result6 = _mm256_setzero_si256();
  __m256i result7 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 6], program->pixel_offset[x + 7], current_coeff, run_filter_size_stride, result6, result7, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;
#else
  __m256i result0 = _mm256_setzero_si256();
  __m256i result1 = _mm256_setzero_si256();
  __m256i result2 = _mm256_setzero_si256();
  __m256i result3 = _mm256_setzero_si256();
  process_four_pixels_h_avx512 < is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src,
    program->pixel_offset[x], program->pixel_offset[x + 1],
    program->pixel_offset[x + 2], program->pixel_offset[x + 3],
    current_coeff, run_filter_size_stride,
    result0, result1, result2, result3,
    unaligned_kernel_size, shifttosigned);
  current_coeff += 4 * run_filter_size_stride;
  __m256i result4 = _mm256_setzero_si256();
  __m256i result5 = _mm256_setzero_si256();
  __m256i result6 = _mm256_setzero_si256();
  __m256i result7 = _mm256_setzero_si256();
  process_four_pixels_h_avx512 < is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src,
    program->pixel_offset[x + 4], program->pixel_offset[x + 5],
    program->pixel_offset[x + 6], program->pixel_offset[x + 7],
    current_coeff, run_filter_size_stride,
    result4, result5, result6, result7,
    unaligned_kernel_size, shifttosigned);
  current_coeff += 4 * run_filter_size_stride;
#endif
  __m256i result_8x_32_lo = reduce_8x256i_32i_tree(
    result0, result1, result2, result3,
    result4, result5, result6, result7,
    rounder_scalar);

  // same for pixels 8..15
#ifdef TWO_PIXELS_AT_ONCE
  __m256i result8 = _mm256_setzero_si256();
  __m256i result9 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 8], program->pixel_offset[x + 9], current_coeff, run_filter_size_stride, result8, result9, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result10 = _mm256_setzero_si256();
  __m256i result11 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 10], program->pixel_offset[x + 11], current_coeff, run_filter_size_stride, result10, result11, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result12 = _mm256_setzero_si256();
  __m256i result13 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 12], program->pixel_offset[x + 13], current_coeff, run_filter_size_stride, result12, result13, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * run_filter_size_stride;

  __m256i result14 = _mm256_setzero_si256();
  __m256i result15 = _mm256_setzero_si256();
  process_two_pixels_h_avx512<is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src, program->pixel_offset[x + 14], program->pixel_offset[x + 15], current_coeff, run_filter_size_stride, result14, result15, unaligned_kernel_size, shifttosigned);
  // last one, no need:
  // current_coeff += 2 * run_filter_size_stride;
#else
  __m256i result8 = _mm256_setzero_si256();
  __m256i result9 = _mm256_setzero_si256();
  __m256i result10 = _mm256_setzero_si256();
  __m256i result11 = _mm256_setzero_si256();
  process_four_pixels_h_avx512 < is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src,
    program->pixel_offset[x + 8], program->pixel_offset[x + 9],
    program->pixel_offset[x + 10], program->pixel_offset[x + 11],
    current_coeff, run_filter_size_stride,
    result8, result9, result10, result11,
    unaligned_kernel_size, shifttosigned);
  current_coeff += 4 * run_filter_size_stride;
  __m256i result12 = _mm256_setzero_si256();
  __m256i result13 = _mm256_setzero_si256();
  __m256i result14 = _mm256_setzero_si256();
  __m256i result15 = _mm256_setzero_si256();
  process_four_pixels_h_avx512 < is_safe, pixel_t, lessthan16bit, FixedFilterSize>(src,
    program->pixel_offset[x + 12], program->pixel_offset[x + 13],
    program->pixel_offset[x + 14], program->pixel_offset[x + 15],
    current_coeff, run_filter_size_stride,
    result12, result13, result14, result15,
    unaligned_kernel_size, shifttosigned);
#endif

  __m256i result_8x_32_hi = reduce_8x256i_32i_tree(
    result8, result9, result10, result11,
    result12, result13, result14, result15,
    rounder_scalar);

  // 
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit) {
    const __m256i shiftfromsigned = _mm256_set1_epi32(+32768 << FPScale16bits);
    result_8x_32_lo = _mm256_add_epi32(result_8x_32_lo, shiftfromsigned);
    result_8x_32_hi = _mm256_add_epi32(result_8x_32_hi, shiftfromsigned);
  }

  // ... scale/pack ...
  const int shift = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  __m256i scaled_lo = _mm256_srai_epi32(result_8x_32_lo, shift);
  __m256i scaled_hi = _mm256_srai_epi32(result_8x_32_hi, shift);

  // integer 32->unsigned 16 bit, the usual and quick way
  __m256i result_16 = _mm256_packus_epi32(scaled_lo, scaled_hi);

  // we have 8x16 bit unsigned pixels now in result_16
  if constexpr (sizeof(pixel_t) == 2 && lessthan16bit) {
    result_16 = _mm256_min_epu16(result_16, clamp_limit);
  }

  result_16 =  _mm256_permute4x64_epi64(result_16, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));

  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result_8 = _mm_packus_epi16(_mm256_castsi256_si128(result_16), _mm256_extracti128_si256(result_16, 1));
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x), result_8); // 16x 8bit pixels
  }
  else {
    _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + x), result_16);
  }
}

// -----------------------------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------------------------
template<typename pixel_t, bool lessthan16bit, int FixedFilterSize>
static void internal_resizer_h_avx512_generic(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  int rounder_scalar = 1 << (current_fp_scale_bits - 1);

  __m256i shifttosigned = _mm256_set1_epi16(-32768);
  __m256i clamp_limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1));

  const pixel_t* src = reinterpret_cast<const pixel_t*>(src8);
  pixel_t* dst = reinterpret_cast<pixel_t*>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  const int PIXELS_AT_A_TIME = 16;

  const int filter_size = (FixedFilterSize >= 1) ? FixedFilterSize : program->filter_size; // aligned coeff stride
  const int w_safe = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < w_safe; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_avx512<true, pixel_t, lessthan16bit, FixedFilterSize>(src, x, program->pixel_coefficient, filter_size, rounder_scalar, shifttosigned, clamp_limit, dst, program);
    }
    for (int x = w_safe; x < width; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_avx512<false, pixel_t, lessthan16bit, FixedFilterSize>(src, x, program->pixel_coefficient, filter_size, rounder_scalar, shifttosigned, clamp_limit, dst, program);
    }
    dst += dst_pitch;
    src += src_pitch;
  }
}

// Entry Points: Map filter sizes to optimized templates
void resizer_h_avx512_generic_uint8_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int fs = program->filter_size; // aligned coeff stride
  // Dispatch to optimized templates based on filter size
  if (fs == 4)  internal_resizer_h_avx512_generic<uint8_t, true, 4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 8)  internal_resizer_h_avx512_generic<uint8_t, true, 8>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 12) internal_resizer_h_avx512_generic<uint8_t, true, 12>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 16) internal_resizer_h_avx512_generic<uint8_t, true, 16>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 32) internal_resizer_h_avx512_generic<uint8_t, true, 32>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 48) internal_resizer_h_avx512_generic<uint8_t, true, 48>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else internal_resizer_h_avx512_generic<uint8_t, true, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// 16 bit Horizontal Dispatcher
// Handles both full 16-bit (lessthan16bit=false) and 10-14 bit (lessthan16bit=true)
template<bool lessthan16bit>
void resizer_h_avx512_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int fs = program->filter_size; // aligned coeff stride
  // Dispatch to optimized templates based on filter size
  if (fs == 4)       internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 8)  internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 8>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 12) internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 12>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 16) internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 16>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 32) internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 32>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (fs == 48) internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, 48>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else               internal_resizer_h_avx512_generic<uint16_t, lessthan16bit, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// Explicit template instantiation
// AVX512 16-bit (unsigned, full range)
template void resizer_h_avx512_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// AVX512 10-14 bit (requires clamping)
template void resizer_h_avx512_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


//------- 512 bit float Horizontals

// Safe quad lane partial load with AVX512 using masks.
// Replaces scalar set_ps sequences with hardware-masked loads.
// Requires AVX-512VL (standard on Rocket Lake, Zen 4/5).
AVS_FORCEINLINE static __m512 _mm512_load_partial_safe_4_m128(const float* src_ptr_offsetted1, const float* src_ptr_offsetted2, const float* src_ptr_offsetted3, const float* src_ptr_offsetted4, int floats_to_load) {
  // Example: N=3 -> (1<<3)-1 = 7 (binary 0111) -> Loads floats 0, 1, 2.
  const __mmask8 k = (1 << floats_to_load) - 1;

  // perform masked loads.
  // _mm_maskz_loadu_ps(k, ptr):
  // - Loads 'valid_pixels' floats from memory.
  // - Zeros out the remaining upper floats in the XMM register (z-masking).
  // - FAULT SUPPRESSION: Hardware guarantees no page fault for masked-off elements.
  //   Thought Avisynth frame buffers are overallocated to avoid OOB reads, this adds extra safety.
  __m128 s1 = _mm_maskz_loadu_ps(k, src_ptr_offsetted1);
  __m128 s2 = _mm_maskz_loadu_ps(k, src_ptr_offsetted2);
  __m128 s3 = _mm_maskz_loadu_ps(k, src_ptr_offsetted3);
  __m128 s4 = _mm_maskz_loadu_ps(k, src_ptr_offsetted4);

  // Combine into ZMM
  __m512 result = _mm512_castps128_ps512(s1);      // Free (register aliasing)
  result = _mm512_insertf32x4(result, s2, 1);      // vinsertf32x4
  result = _mm512_insertf32x4(result, s3, 2);
  result = _mm512_insertf32x4(result, s4, 3);

  return result;
}

AVS_FORCEINLINE static __m512 _mm512_load_partial_safe_2_m256(const float* src_ptr_offsetted1, const float* src_ptr_offsetted2, int floats_to_load) {
  // Calculate the mask. 
  const __mmask8 k = (1U << floats_to_load) - 1;

  // _mm256_maskz_loadu_ps provides fault suppression for masked-off elements,
  // ensuring no page faults occur even if the pointer is near a boundary.
  __m256 s1 = _mm256_maskz_loadu_ps(k, src_ptr_offsetted1);
  __m256 s2 = _mm256_maskz_loadu_ps(k, src_ptr_offsetted2);

  // Combine into ZMM.
  __m512 result = _mm512_castps256_ps512(s1);
  result = _mm512_insertf32x8(result, s2, 1);

  return result;
}

#if 0
// Safe quad lane partial load with AVX512
// Read exactly N pixels (where N mod 4 is the template parameter), avoiding
// - reading beyond the end of the source buffer.
// - avoid NaN contamination by padding with zeros.
template <int Nmod4>
AVS_FORCEINLINE static __m512 _mm512_load_partial_safe_4_m128_avx2like(const float* src_ptr_offsetted1, const float* src_ptr_offsetted2, const float* src_ptr_offsetted3, const float* src_ptr_offsetted4) {
  __m128 s1, s2, s3, s4;
  switch (Nmod4) {
  case 1:
    s1 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted4[0]);
    // ideally: movss
    break;
  case 2:
    s1 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movsd
    break;
  case 3:
    s1 = _mm_set_ps(0.0f, src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, src_ptr_offsetted3[2], src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, src_ptr_offsetted4[2], src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movss + movsd + shuffle or movsd + insert
    break;
  case 0:
    s1 = _mm_set_ps(src_ptr_offsetted1[3], src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(src_ptr_offsetted2[3], src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(src_ptr_offsetted3[3], src_ptr_offsetted3[2], src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(src_ptr_offsetted4[3], src_ptr_offsetted4[2], src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movups
    break;
  default:
    s1 = _mm_setzero_ps(); // n/a cannot happen
    s2 = _mm_setzero_ps();
    s3 = _mm_setzero_ps();
    s4 = _mm_setzero_ps();
  }
  __m512 result = _mm512_castps128_ps512(s1); // Cast the first __m128 to __m512
  result = _mm512_insertf32x4(result, s2, 1); // Insert the second __m128 at position 1
  result = _mm512_insertf32x4(result, s3, 2); // Insert the third __m128 at position 2
  result = _mm512_insertf32x4(result, s4, 3); // Insert the fourth __m128 at position 3
  return result;
}
#endif

// Processes a horizontal resampling kernel of up to four coefficients for float pixel types.
// Supports BilinearResize, BicubicResize, or sinc with up to 2 taps (filter size <= 4).
// AVX512 optimization loads and processes four float coefficients and sixteen pixels simultaneously.
// This AVX512 requires only filter_size_alignment of 4.
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride
  // needed for partial load
  const int Nmod4 = program->filter_size_real % 4;
  const int floats_to_load = (Nmod4 == 0) ? 4 : Nmod4;

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once conceptually with our safe load.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4);
  assert(program->target_size_alignment >= 16); // Must align for 16 pixel offsets
  assert(program->filter_size_alignment >= 4);
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Vertical stripe alignment
  constexpr int STRIPE_ALIGN = 16;

  int max_scanlines = program->max_scanlines / STRIPE_ALIGN * STRIPE_ALIGN;
  if (max_scanlines < STRIPE_ALIGN) max_scanlines = STRIPE_ALIGN;

  // --- outer loop: vertical stripes ---
  for (auto y_from = 0; y_from < height; y_from += max_scanlines) {
    size_t y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

    size_t x = 0;

    // Lambda for 512-bit Core
    auto do_h_float_core = [&](auto partial_load) {

      // load 4x4 sets of coefficients (16 pixels total)
      // at once before the height loop.
      // Coefficients for the source pixel offset (for src_ptr + begin1 [0..3], begin5 [0..3], begin9 [0..3], begin13 [0..3])
      __m512 coef_1_5_9_13 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_2_6_10_14 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_3_7_11_15 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_4_8_12_16 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_1_5_9_13, coef_2_6_10_14, coef_3_7_11_15, coef_4_8_12_16);

      // Pixel offsets for the current target x-positions.
      // Even for x >= width, these offsets are guaranteed to be within the allocated 'target_size_alignment'.
      const int begin1 = program->pixel_offset[x + 0];
      const int begin2 = program->pixel_offset[x + 1];
      const int begin3 = program->pixel_offset[x + 2];
      const int begin4 = program->pixel_offset[x + 3];
      const int begin5 = program->pixel_offset[x + 4];
      const int begin6 = program->pixel_offset[x + 5];
      const int begin7 = program->pixel_offset[x + 6];
      const int begin8 = program->pixel_offset[x + 7];
      const int begin9 = program->pixel_offset[x + 8];
      const int begin10 = program->pixel_offset[x + 9];
      const int begin11 = program->pixel_offset[x + 10];
      const int begin12 = program->pixel_offset[x + 11];
      const int begin13 = program->pixel_offset[x + 12];
      const int begin14 = program->pixel_offset[x + 13];
      const int begin15 = program->pixel_offset[x + 14];
      const int begin16 = program->pixel_offset[x + 15];

      int y = y_from;

      // Calculate pointers ONCE before the inner loop (Optimization from AVX2 version)
      float* AVS_RESTRICT dst_ptr = dst + y * dst_pitch + x;
      const float* src_ptr = src + y * src_pitch;

      // Inner loop: vertical processing. unroll 2 tested, no benefit
      for (; y < y_to; ++y) {

        __m512 data_1_5_9_13;
        __m512 data_2_6_10_14;
        __m512 data_3_7_11_15;
        __m512 data_4_8_12_16;

        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline.
          data_1_5_9_13 = _mm512_load_partial_safe_4_m128(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13, floats_to_load);
          data_2_6_10_14 = _mm512_load_partial_safe_4_m128(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14, floats_to_load);
          data_3_7_11_15 = _mm512_load_partial_safe_4_m128(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15, floats_to_load);
          data_4_8_12_16 = _mm512_load_partial_safe_4_m128(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16, floats_to_load);
        }
        else {
          // In the safe zone, we can directly load 4 source pixels at a time for each of the 16 lanes.
          data_1_5_9_13 = _mm512_loadu_4_m128(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13);
          data_2_6_10_14 = _mm512_loadu_4_m128(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14);
          data_3_7_11_15 = _mm512_loadu_4_m128(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15);
          data_4_8_12_16 = _mm512_loadu_4_m128(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16);
        }

        _MM_TRANSPOSE16_LANE4_PS(data_1_5_9_13, data_2_6_10_14, data_3_7_11_15, data_4_8_12_16);

        // two sets, hint for the compiler to allow parallel fma's
        __m512 result_0 = _mm512_mul_ps(data_1_5_9_13, coef_1_5_9_13);
        __m512 result_1 = _mm512_mul_ps(data_2_6_10_14, coef_2_6_10_14);
        result_0 = _mm512_fmadd_ps(data_3_7_11_15, coef_3_7_11_15, result_0);
        result_1 = _mm512_fmadd_ps(data_4_8_12_16, coef_4_8_12_16, result_1);
        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result_0, result_1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      } // y

        // Move to the next set of coefficients for the next 16 output pixels
      current_coeff += filter_size * 16;
      };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});  // partial_load == false, use direct _mm512_loadu_ps
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer _mm512_load_partial_safe_4_m128
    }
  }
}


void resize_h_planar_float_avx512_transpose_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once conceptually with our safe load.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 8);
  assert(program->target_size_alignment >= 16); // Must align for 16 pixel offsets
  assert(program->filter_size_alignment >= 8);
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Vertical stripe alignment
  constexpr int STRIPE_ALIGN = 16; // this must be multiple of PIXELS_AT_A_TIME 

  int max_scanlines = program->max_scanlines / STRIPE_ALIGN * STRIPE_ALIGN;

  if (max_scanlines < STRIPE_ALIGN) max_scanlines = STRIPE_ALIGN;

  // --- outer loop: vertical stripes ---
  for (auto y_from = 0; y_from < height; y_from += max_scanlines) {
    size_t y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

    size_t x = 0;

    // Lambda for 512-bit Core
    auto do_h_float_core = [&](auto partial_load) {

      // Load to 8 coefficients per source pixel at once before the height loop.
      // Pre-loading and transposing coefficients keeps register usage efficient.
      // Assumes 'filter_size_aligned' is at least 8.

      __m256 coef0 = _mm256_loadu_ps(current_coeff + filter_size * 0);
      __m256 coef1 = _mm256_loadu_ps(current_coeff + filter_size * 1);
      __m512 coef01 = _mm512_insert_2_m256(coef0, coef1);

      __m256 coef2 = _mm256_loadu_ps(current_coeff + filter_size * 2);
      __m256 coef3 = _mm256_loadu_ps(current_coeff + filter_size * 3);
      __m512 coef23 = _mm512_insert_2_m256(coef2, coef3);

      __m256 coef4 = _mm256_loadu_ps(current_coeff + filter_size * 4);
      __m256 coef5 = _mm256_loadu_ps(current_coeff + filter_size * 5);
      __m512 coef45 = _mm512_insert_2_m256(coef4, coef5);

      __m256 coef6 = _mm256_loadu_ps(current_coeff + filter_size * 6);
      __m256 coef7 = _mm256_loadu_ps(current_coeff + filter_size * 7);
      __m512 coef67 = _mm512_insert_2_m256(coef6, coef7);

      __m256 coef8 = _mm256_loadu_ps(current_coeff + filter_size * 8);
      __m256 coef9 = _mm256_loadu_ps(current_coeff + filter_size * 9);
      __m512 coef89 = _mm512_insert_2_m256(coef8, coef9);

      __m256 coef10 = _mm256_loadu_ps(current_coeff + filter_size * 10);
      __m256 coef11 = _mm256_loadu_ps(current_coeff + filter_size * 11);
      __m512 coef1011 = _mm512_insert_2_m256(coef10, coef11);

      __m256 coef12 = _mm256_loadu_ps(current_coeff + filter_size * 12);
      __m256 coef13 = _mm256_loadu_ps(current_coeff + filter_size * 13);
      __m512 coef1213 = _mm512_insert_2_m256(coef12, coef13);

      __m256 coef14 = _mm256_loadu_ps(current_coeff + filter_size * 14);
      __m256 coef15 = _mm256_loadu_ps(current_coeff + filter_size * 15);
      __m512 coef1415 = _mm512_insert_2_m256(coef14, coef15);

      // Before: 8x_m512 as 8x2x_m256 holding 16*8 coefficients each for 16 pixels
      // coef01:   0.0 ... 0.7 | 1.0 ... 1.7
      // coef23:   2.0 ... 2.7 | 3.0 ... 3.7
      // coef45:   4.0 ... 4.7 | 5.0 ... 5.7
      // coef67:   6.0 ... 6.7 | 7.0 ... 7.7
      // coef89:   8.0 ... 8.7 | 9.0 ... 9.7
      // coef1011: A.0 ... A.7 | B.0 ... B.7
      // coef1213: C.0 ... C.7 | D.0 ... D.7
      // coef1415: E.0 ... E.7 | F.0 ... F.7

      _MM_TRANSPOSE8x16_PS(coef01, coef23, coef45, coef67, coef89, coef1011, coef1213, coef1415);

      // After: 8x _m512 holding 16 coefficients, each for 16 pixels
      // result0, as old_coef01 : 0.0 .. 7.0 | 8.0 .. F.0
      // result1, as old_coef23 : 0.1 .. 7.1 | 8.1 .. F.1
      // result2, as old_coef45 : 0.2 .. 7.2 | 8.2 .. F.2
      // result3, as old_coef67 : 0.3 .. 7.3 | 8.3 .. F.3
      // result4, as old_coef89 : 0.4 .. 7.4 | 8.4 .. F.4
      // result5, as old_coef1011: 0.5 .. 7.5 | 8.5 .. F.5
      // result6, as old_coef1213: 0.6 .. 7.6 | 8.6 .. F.6
      // result7, as old_coef1415: 0.7 .. 7.7 | 8.7 .. F.7

      // Pixel offsets for the current target x-positions.
      // Even for x >= width, these offsets are guaranteed to be within the allocated 'target_size_alignment'.
      const int begin1 = program->pixel_offset[x + 0];
      const int begin2 = program->pixel_offset[x + 1];
      const int begin3 = program->pixel_offset[x + 2];
      const int begin4 = program->pixel_offset[x + 3];
      const int begin5 = program->pixel_offset[x + 4];
      const int begin6 = program->pixel_offset[x + 5];
      const int begin7 = program->pixel_offset[x + 6];
      const int begin8 = program->pixel_offset[x + 7];
      const int begin9 = program->pixel_offset[x + 8];
      const int begin10 = program->pixel_offset[x + 9];
      const int begin11 = program->pixel_offset[x + 10];
      const int begin12 = program->pixel_offset[x + 11];
      const int begin13 = program->pixel_offset[x + 12];
      const int begin14 = program->pixel_offset[x + 13];
      const int begin15 = program->pixel_offset[x + 14];
      const int begin16 = program->pixel_offset[x + 15];

      int y = y_from;

      // Calculate pointers ONCE before the inner loop (Optimization from AVX2 version)
      float* AVS_RESTRICT dst_ptr = dst + y * dst_pitch + x;
      const float* src_ptr = src + y * src_pitch;

      // only needed for partial load
      const int Nmod8 = program->filter_size_real % 8;
      const int floats_to_load = Nmod8 == 0 ? 8 : Nmod8;

      for (; y < y_to; ++y) {

        __m512 data01, data23, data45, data67, data89, data1011, data1213, data1415;

        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline.
          data01 = _mm512_load_partial_safe_2_m256(src_ptr + begin1, src_ptr + begin2, floats_to_load);
          data23 = _mm512_load_partial_safe_2_m256(src_ptr + begin3, src_ptr + begin4, floats_to_load);
          data45 = _mm512_load_partial_safe_2_m256(src_ptr + begin5, src_ptr + begin6, floats_to_load);
          data67 = _mm512_load_partial_safe_2_m256(src_ptr + begin7, src_ptr + begin8, floats_to_load);
          data89 = _mm512_load_partial_safe_2_m256(src_ptr + begin9, src_ptr + begin10, floats_to_load);
          data1011 = _mm512_load_partial_safe_2_m256(src_ptr + begin11, src_ptr + begin12, floats_to_load);
          data1213 = _mm512_load_partial_safe_2_m256(src_ptr + begin13, src_ptr + begin14, floats_to_load);
          data1415 = _mm512_load_partial_safe_2_m256(src_ptr + begin15, src_ptr + begin16, floats_to_load);
        }
        else {
          // In the safe zone, we can directly load 8 source pixels at a time for each of the 16 lanes.
          data01 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin1), _mm256_loadu_ps(src_ptr + begin2));
          data23 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin3), _mm256_loadu_ps(src_ptr + begin4));
          data45 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin5), _mm256_loadu_ps(src_ptr + begin6));
          data67 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin7), _mm256_loadu_ps(src_ptr + begin8));
          data89 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin9), _mm256_loadu_ps(src_ptr + begin10));
          data1011 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin11), _mm256_loadu_ps(src_ptr + begin12));
          data1213 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin13), _mm256_loadu_ps(src_ptr + begin14));
          data1415 = _mm512_insert_2_m256(_mm256_loadu_ps(src_ptr + begin15), _mm256_loadu_ps(src_ptr + begin16));
        }

        _MM_TRANSPOSE8x16_PS(data01, data23, data45, data67, data89, data1011, data1213, data1415);

        // two sets, hint for the compiler to allow parallel fma's
        __m512 result_0 = _mm512_mul_ps(data01, coef01);
        __m512 result_1 = _mm512_mul_ps(data23, coef23);
        result_0 = _mm512_fmadd_ps(data45, coef45, result_0);
        result_1 = _mm512_fmadd_ps(data67, coef67, result_1);
        result_0 = _mm512_fmadd_ps(data89, coef89, result_0);
        result_1 = _mm512_fmadd_ps(data1011, coef1011, result_1);
        result_0 = _mm512_fmadd_ps(data1213, coef1213, result_0);
        result_1 = _mm512_fmadd_ps(data1415, coef1415, result_1);

        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result_0, result_1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      } // y

      // Move to the next set of coefficients for the next 16 output pixels
      current_coeff += filter_size * 16;
      }; // lambda

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});  // partial_load == false, use direct _mm512_loadu_ps
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer _mm512_load_partial_safe_2_m256
    }
  }
}

// Similar to AVX2 resize_h_planar_float_avx512_permutex_vstripe_ks4
// but doing 16 pixels at a time with AVX512 permutex instructions.
void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  assert(program->filter_size_alignment >= 4);

  const int max_scanlines = program->max_scanlines;

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe (points to start of row's coeffs)
    const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_float_core = [&](auto partial_load) {

      // prepare coefs in transposed V-form
      // We load 4 coefficients sets (for 4 pixels) into 4 lanes of a zmm register
      __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      __m512i perm_0 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      int iStart = program->pixel_offset[x];
      perm_0 = _mm512_sub_epi32(perm_0, _mm512_set1_epi32(iStart));
      /* like this:
      __m512i perm_0 = _mm512_set_epi32(
        program->pixel_offset[x + 15] - iStart,
        ...
        program->pixel_offset[x + 0] - iStart);
      */

      // Taps are contiguous (0, 1, 2, 3), so we increment perm indexes by 1.
      __m512i one_epi32 = _mm512_set1_epi32(1);
      __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
      __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
      __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + iStart + y_from * src_pitch; // all permute offsets relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode
      const int remaining = program->source_size - iStart;

      for (int y = y_from; y < y_to; y++)
      {
        __m512 data_src, data_src2;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          // Load first 16 floats
          int rem1 = std::max(0, std::min(16, remaining));
          __mmask16 k1 = (1U << rem1) - 1;
          data_src = _mm512_maskz_loadu_ps(k1, src_ptr);

          // Load next 16 floats (offset by 16)
          int rem2 = std::max(0, std::min(16, remaining - 16));
          __mmask16 k2 = (1U << rem2) - 1;
          data_src2 = _mm512_maskz_loadu_ps(k2, src_ptr + 16);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src = _mm512_loadu_ps(src_ptr);
          data_src2 = _mm512_loadu_ps(src_ptr + 16);
        }

        __m512 data_0 = _mm512_permutex2var_ps(data_src, perm_0, data_src2);
        __m512 data_1 = _mm512_permutex2var_ps(data_src, perm_1, data_src2);
        __m512 data_2 = _mm512_permutex2var_ps(data_src, perm_2, data_src2);
        __m512 data_3 = _mm512_permutex2var_ps(data_src, perm_3, data_src2);

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

        result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff += filter_size * 16;
      };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{});
    }
  }
}

// Similar to resize_h_planar_float_avx512_permutex_vstripe_ks4 but for kernel size up to 8
// 16 target pixels at a time with AVX512 permutex instructions.
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  const int width_safe_mod = (program->safelimit_8_pixels.overread_possible ? program->safelimit_8_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 8); // We preload all relevant coefficients (up to 8) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x8 float loads.
  // n/a: load once 8. align 4 is the same as 8 for the purpose.
  assert(program->filter_size_alignment >= 8);

  const int max_scanlines = program->max_scanlines;

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe (points to start of row's coeffs)
    const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_float_core = [&](auto partial_load) {

      // prepare coefs in transposed V-form
      // 4 coefficients sets (for 4 pixels) into 4 lanes of a zmm register
      __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      const float* AVS_RESTRICT current_coeff_47 = current_coeff + 4;

      __m512 coef_r4 = _mm512_load_4_m128(current_coeff_47 + filter_size * 0, current_coeff_47 + filter_size * 4, current_coeff_47 + filter_size * 8, current_coeff_47 + filter_size * 12);
      __m512 coef_r5 = _mm512_load_4_m128(current_coeff_47 + filter_size * 1, current_coeff_47 + filter_size * 5, current_coeff_47 + filter_size * 9, current_coeff_47 + filter_size * 13);
      __m512 coef_r6 = _mm512_load_4_m128(current_coeff_47 + filter_size * 2, current_coeff_47 + filter_size * 6, current_coeff_47 + filter_size * 10, current_coeff_47 + filter_size * 14);
      __m512 coef_r7 = _mm512_load_4_m128(current_coeff_47 + filter_size * 3, current_coeff_47 + filter_size * 7, current_coeff_47 + filter_size * 11, current_coeff_47 + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);
      _MM_TRANSPOSE16_LANE4_PS(coef_r4, coef_r5, coef_r6, coef_r7);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      __m512i perm_0 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      int iStart = program->pixel_offset[x + 0];
      perm_0 = _mm512_sub_epi32(perm_0, _mm512_set1_epi32(iStart));
      /* like this:
      __m512i perm_0 = _mm512_set_epi32(
        program->pixel_offset[x + 15] - iStart,
        ...
        program->pixel_offset[x + 0] - iStart);
        */

      // Taps are contiguous (0, 1, 2, 3 .. 7), so we increment perm indexes by 1.
      __m512i one_epi32 = _mm512_set1_epi32(1);
      __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
      __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
      __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
      __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
      __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
      __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
      __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + iStart + y_from * src_pitch; // all permute offsets relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode
      const int remaining = program->source_size - iStart;

      for (int y = y_from; y < y_to; y++)
      {
        __m512 data_src, data_src2;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          // Load first 16 floats
          int rem1 = std::max(0, std::min(16, remaining));
          __mmask16 k1 = (1U << rem1) - 1;
          data_src = _mm512_maskz_loadu_ps(k1, src_ptr);

          // Load next 16 floats (offset by 16)
          int rem2 = std::max(0, std::min(16, remaining - 16));
          __mmask16 k2 = (1U << rem2) - 1;
          data_src2 = _mm512_maskz_loadu_ps(k2, src_ptr + 16);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src = _mm512_loadu_ps(src_ptr);
          data_src2 = _mm512_loadu_ps(src_ptr + 16);
        }

        __m512 data_0 = _mm512_permutex2var_ps(data_src, perm_0, data_src2);
        __m512 data_1 = _mm512_permutex2var_ps(data_src, perm_1, data_src2);
        __m512 data_2 = _mm512_permutex2var_ps(data_src, perm_2, data_src2);
        __m512 data_3 = _mm512_permutex2var_ps(data_src, perm_3, data_src2);
        __m512 data_4 = _mm512_permutex2var_ps(data_src, perm_4, data_src2);
        __m512 data_5 = _mm512_permutex2var_ps(data_src, perm_5, data_src2);
        __m512 data_6 = _mm512_permutex2var_ps(data_src, perm_6, data_src2);
        __m512 data_7 = _mm512_permutex2var_ps(data_src, perm_7, data_src2);

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_2, coef_r2);
        __m512 result2 = _mm512_mul_ps(data_4, coef_r4);
        __m512 result3 = _mm512_mul_ps(data_6, coef_r6);

        result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);
        result2 = _mm512_fmadd_ps(data_5, coef_r5, result2);
        result3 = _mm512_fmadd_ps(data_7, coef_r7, result3);

        __m512 result01 = _mm512_add_ps(result0, result1);
        __m512 result23 = _mm512_add_ps(result2, result3);
        __m512 result0123 = _mm512_add_ps(result01, result23);
        _mm512_stream_ps(dst_ptr, result0123);

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff += filter_size * 16;
      };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{});
    }
  }
}


// Similar to resize_h_planar_float_avx512_permutex_vstripe_ks4 but for kernel size up to 
// 16 target pixels at a time with AVX512 permutex instructions.
// Uses 2 groups of 8 output samples processing by independednd gathering 2x32 contigous groups of sources to support more downscale ratios
void resize_h_planar_float_avx512_permutex_vstripe_2s8_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process 2 groups of 8 pixels in parallel using AVX512 with wider source gathering for downsample

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  const int width_safe_mod = (program->safelimit_8_pixels.overread_possible ? program->safelimit_8_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 8); // We preload all relevant coefficients (up to 8) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x8 float loads.
  assert(program->filter_size_alignment >= 8);
  // n/a: load once 8. align 4 is the same as 8 for the purpose.

  const int max_scanlines = program->max_scanlines;

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe (points to start of row's coeffs)
    const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_float_core = [&](auto partial_load) {

      // prepare coefs in transposed V-form
      // 4 coefficients sets (for 4 pixels) into 4 lanes of a zmm register
      __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      const float* AVS_RESTRICT current_coeff_47 = current_coeff + 4;

      __m512 coef_r4 = _mm512_load_4_m128(current_coeff_47 + filter_size * 0, current_coeff_47 + filter_size * 4, current_coeff_47 + filter_size * 8, current_coeff_47 + filter_size * 12);
      __m512 coef_r5 = _mm512_load_4_m128(current_coeff_47 + filter_size * 1, current_coeff_47 + filter_size * 5, current_coeff_47 + filter_size * 9, current_coeff_47 + filter_size * 13);
      __m512 coef_r6 = _mm512_load_4_m128(current_coeff_47 + filter_size * 2, current_coeff_47 + filter_size * 6, current_coeff_47 + filter_size * 10, current_coeff_47 + filter_size * 14);
      __m512 coef_r7 = _mm512_load_4_m128(current_coeff_47 + filter_size * 3, current_coeff_47 + filter_size * 7, current_coeff_47 + filter_size * 11, current_coeff_47 + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);
      _MM_TRANSPOSE16_LANE4_PS(coef_r4, coef_r5, coef_r6, coef_r7);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      // shorter SIMD-way - single memory load (hacky SIMD load from int vector ?)
      __m512i perm_0_low8 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      int iStart_low8 = program->pixel_offset[x];
      perm_0_low8 = _mm512_sub_epi32(perm_0_low8, _mm512_set1_epi32(iStart_low8)); // vpbroadcastd zmm, r32

      __m512i perm_0_high8 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 8]));
      int iStart_high8 = program->pixel_offset[x + 8];
      perm_0_high8 = _mm512_sub_epi32(perm_0_high8, _mm512_set1_epi32(iStart_high8)); // vpbroadcastd zmm, r32
      perm_0_high8 = _mm512_inserti64x4(perm_0_high8, _mm512_castsi512_si256(perm_0_high8), 1);// shift low 8 epi32 to high 8
      
      const __mmask16 k_high8 = 0xFF00;
      __m512i perm_0 = _mm512_mask_blend_epi32(k_high8, perm_0_low8, perm_0_high8); 

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr_low8 = src + iStart_low8 + y_from * src_pitch; // all permute offsets relative to this start offset
      const float* src_ptr_high8 = src + iStart_high8 + y_from * src_pitch; // all permute offsets relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode
      const int remaining_low8 = program->source_size - iStart_low8;
      const int remaining_high8 = program->source_size - iStart_high8;

      int rem1_low8 = std::max(0, std::min(16, remaining_low8));
      __mmask16 k1_low8 = (1U << rem1_low8) - 1;
      int rem2 = std::max(0, std::min(16, remaining_low8 - 16));
      __mmask16 k2_low8 = (1U << rem2) - 1;
      int rem1_high8 = std::max(0, std::min(16, remaining_high8));
      __mmask16 k1_high8 = (1U << rem1_high8) - 1;
      int rem2_high8 = std::max(0, std::min(16, remaining_high8 - 16));
      __mmask16 k2_high8 = (1U << rem2_high8) - 1;

      // Taps are contiguous (0, 1, 2, 3 .. 7), so we increment perm indexes by 1.
      const __m512i one_epi32 = _mm512_set1_epi32(1);
      const __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
      const __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
      const __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
      const __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
      const __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
      const __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
      const __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);

      for (int y = y_from; y < y_to; y++)
      {
        __m512 data_src_low8, data_src2_low8;
        __m512 data_src_high8, data_src2_high8;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          // Load first 16 floats
          data_src_low8 = _mm512_maskz_loadu_ps(k1_low8, src_ptr_low8);
          // Load next 16 floats (offset by 16)
          data_src2_low8 = _mm512_maskz_loadu_ps(k2_low8, src_ptr_low8 + 16);

          // high8
          // Safe masked loads for the image edge
          // Load first 16 floats
          data_src_high8 = _mm512_maskz_loadu_ps(k1_high8, src_ptr_high8);
          // Load next 16 floats (offset by 16)
          data_src2_high8 = _mm512_maskz_loadu_ps(k2_high8, src_ptr_high8 + 16);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src_low8 = _mm512_loadu_ps(src_ptr_low8);
          data_src2_low8 = _mm512_loadu_ps(src_ptr_low8 + 16);
          data_src_high8 = _mm512_loadu_ps(src_ptr_high8);
          data_src2_high8 = _mm512_loadu_ps(src_ptr_high8 + 16);
        }

       /* __m512 data_0 = _mm512_permutex2var_ps(data_src_low8, perm_0, data_src2_low8);
        __m512 data_0_high8 = _mm512_permutex2var_ps(data_src_high8, perm_0, data_src2_high8);
        data_0 = _mm512_mask_blend_ps(k_high8, data_0, data_0_high8);*/
        __m512 data_0 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0, data_src2_high8));
        __m512 data_1 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1, data_src2_high8));
        __m512 data_2 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_2, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_2, data_src2_high8));
        __m512 data_3 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_3, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_3, data_src2_high8));
        __m512 data_4 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_4, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_4, data_src2_high8));
        __m512 data_5 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_5, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_5, data_src2_high8));
        __m512 data_6 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_6, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_6, data_src2_high8));
        __m512 data_7 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_7, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_7, data_src2_high8));

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_2, coef_r2);
        __m512 result2 = _mm512_mul_ps(data_4, coef_r4);
        __m512 result3 = _mm512_mul_ps(data_6, coef_r6);

        result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);
        result2 = _mm512_fmadd_ps(data_5, coef_r5, result2);
        result3 = _mm512_fmadd_ps(data_7, coef_r7, result3);

        __m512 result01 = _mm512_add_ps(result0, result1);
        __m512 result23 = _mm512_add_ps(result2, result3);
        __m512 result0123 = _mm512_add_ps(result01, result23);
        _mm512_stream_ps(dst_ptr, result0123); 

        dst_ptr += dst_pitch;
        src_ptr_low8 += src_pitch;
        src_ptr_high8 += src_pitch;
      }

      current_coeff += filter_size * PIXELS_AT_A_TIME;
    };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{});
    }
  }
}



// Similar to resize_h_planar_float_avx512_permutex_vstripe_ks4 but for kernel size up to 16
// 16 target pixels at a time with AVX512 permutex instructions.
void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  const int width_safe_mod = (program->safelimit_8_pixels.overread_possible ? program->safelimit_8_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 16); // We preload all relevant coefficients (up to 8) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x16 float loads.
  // We gather_load 'short' coeffs, from 0..15. 
  // Load is unaligned, but we treat 16 coeffs.
  // - For kernel_size 9..16 we have valid entries until 16 when filter_size_alignment==8:
  //   alignment 8 is enough for gathering with filter_size*15 offset.
  // - But for kernel_size 1..8 we need alignment 16 to be able to gather up to
  //   filter_size*15 offset.
  // This is just an extra safety; this case is called in practice between kernel sizes 9..16.
  assert(
    (program->filter_size_real < 8 && program->filter_size_alignment >= 16)
    ||
    (program->filter_size_real >= 8 && program->filter_size_alignment >= 8)
  );

  const int max_scanlines = program->max_scanlines;

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff_t to pretransposed coefficient buffer (tap-major layout)
    const float* AVS_RESTRICT current_coeff_t = (const float* AVS_RESTRICT)program->pixel_coefficient_AVX512_float_H;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_float_core = [&](auto partial_load) {

      // Load pretransposed coefficients: 16 sequential aligned loads instead of 16 gathers.
      // Layout per x-group: [coef_r0[px0..15], coef_r1[px0..15], ..., coef_r15[px0..15]]
      const __m512 coef_r0  = _mm512_load_ps(current_coeff_t +  0 * 16);
      const __m512 coef_r1  = _mm512_load_ps(current_coeff_t +  1 * 16);
      const __m512 coef_r2  = _mm512_load_ps(current_coeff_t +  2 * 16);
      const __m512 coef_r3  = _mm512_load_ps(current_coeff_t +  3 * 16);
      const __m512 coef_r4  = _mm512_load_ps(current_coeff_t +  4 * 16);
      const __m512 coef_r5  = _mm512_load_ps(current_coeff_t +  5 * 16);
      const __m512 coef_r6  = _mm512_load_ps(current_coeff_t +  6 * 16);
      const __m512 coef_r7  = _mm512_load_ps(current_coeff_t +  7 * 16);
      const __m512 coef_r8  = _mm512_load_ps(current_coeff_t +  8 * 16);
      const __m512 coef_r9  = _mm512_load_ps(current_coeff_t +  9 * 16);
      const __m512 coef_r10 = _mm512_load_ps(current_coeff_t + 10 * 16);
      const __m512 coef_r11 = _mm512_load_ps(current_coeff_t + 11 * 16);
      const __m512 coef_r12 = _mm512_load_ps(current_coeff_t + 12 * 16);
      const __m512 coef_r13 = _mm512_load_ps(current_coeff_t + 13 * 16);
      const __m512 coef_r14 = _mm512_load_ps(current_coeff_t + 14 * 16);
      const __m512 coef_r15 = _mm512_load_ps(current_coeff_t + 15 * 16);

      const __m512i one_epi32 = _mm512_set1_epi32(1);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      // shorter SIMD-way - single memory load (hacky SIMD load from int vector ?)
      __m512i perm_0 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      int iStart = _mm256_extract_epi32(_mm512_castsi512_si256(perm_0), 0);
      perm_0 = _mm512_sub_epi32(perm_0, _mm512_set1_epi32(iStart)); // vpbroadcastd zmm, r32

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + iStart + y_from * src_pitch; // all permute offsets relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode
      const int remaining = program->source_size - iStart;

      int rem1 = std::max(0, std::min(16, remaining));
      __mmask16 k1 = (1U << rem1) - 1;
      int rem2 = std::max(0, std::min(16, remaining - 16));
      __mmask16 k2 = (1U << rem2) - 1;

      for (int y = y_from; y < y_to; y++)
      {
        // Taps are contiguous (0, 1, 2, 3 .. 7), so we increment perm indexes by 1.
        // To save register usage in ks16 version - calculate offsets at runtime
        // working indexes, reloaded from constant perm_0
        __m512i perm_0w = perm_0;
        __m512i perm_1w = _mm512_add_epi32(perm_0, one_epi32);

        const __m512i two_epi32 = _mm512_set1_epi32(2);

        __m512 data_src, data_src2;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          // Load first 16 floats
          data_src = _mm512_maskz_loadu_ps(k1, src_ptr);
          // Load next 16 floats (offset by 16)
          data_src2 = _mm512_maskz_loadu_ps(k2, src_ptr + 16);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src = _mm512_loadu_ps(src_ptr);
          data_src2 = _mm512_loadu_ps(src_ptr + 16);
        }

        __m512 data_0 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2); // TODO: replace with shorter _mm512_mul_ps(_mm512_permutex2var_ps(data_src, perm_0w, data_src2), coef_r0);
        __m512 data_1 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_1, coef_r1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_2 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2); // TODO: replace with shorter result0 = _mm512_fmadd_ps(_mm512_permutex2var_ps(data_src, perm_0w, data_src2), coef_r2, result0);
        __m512 data_3 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_4 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_5 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
        result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_6 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_7 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
        result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_8 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_9 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_8, coef_r8, result0);
        result1 = _mm512_fmadd_ps(data_9, coef_r9, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_10 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_11 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_10, coef_r10, result0);
        result1 = _mm512_fmadd_ps(data_11, coef_r11, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_12 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_13 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_12, coef_r12, result0);
        result1 = _mm512_fmadd_ps(data_13, coef_r13, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_14 = _mm512_permutex2var_ps(data_src, perm_0w, data_src2);
        __m512 data_15 = _mm512_permutex2var_ps(data_src, perm_1w, data_src2);

        result0 = _mm512_fmadd_ps(data_14, coef_r14, result0);
        result1 = _mm512_fmadd_ps(data_15, coef_r15, result1);

        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff_t += 16 * PIXELS_AT_A_TIME; // 256 floats per x-group (16 taps × 16 pixels)
    };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{});
    }
  }
}


// Similar to resize_h_planar_float_avx512_permutex_vstripe_ks4 but for kernel size up to 16
// 16 target pixels at a time with AVX512 permutex instructions.
// Uses 2 groups of 8 output samples processing by independent gathering 2x32 contigous groups of sources to support more downscale ratios
void resize_h_planar_float_avx512_permutex_vstripe_2s8_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16; // Process 16 pixels in parallel using AVX512 for partial downsampling works

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  const int width_safe_mod = (program->safelimit_8_pixels.overread_possible ? program->safelimit_8_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 16); // We preload all relevant coefficients (up to 8) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x16 float loads.
  // n/a: load once 16. align 8 is the same as 16 for the purpose.
  assert(program->filter_size_alignment >= 8);

  const int max_scanlines = program->max_scanlines;

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff_t to pretransposed coefficient buffer (tap-major layout)
    const float* AVS_RESTRICT current_coeff_t = (const float* AVS_RESTRICT)program->pixel_coefficient_AVX512_float_H;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_float_core = [&](auto partial_load) {

      // Load pretransposed coefficients: 16 sequential aligned loads instead of 16 gathers.
      // Layout per x-group: [coef_r0[px0..15], coef_r1[px0..15], ..., coef_r15[px0..15]]
      const __m512 coef_r0  = _mm512_load_ps(current_coeff_t +  0 * 16);
      const __m512 coef_r1  = _mm512_load_ps(current_coeff_t +  1 * 16);
      const __m512 coef_r2  = _mm512_load_ps(current_coeff_t +  2 * 16);
      const __m512 coef_r3  = _mm512_load_ps(current_coeff_t +  3 * 16);
      const __m512 coef_r4  = _mm512_load_ps(current_coeff_t +  4 * 16);
      const __m512 coef_r5  = _mm512_load_ps(current_coeff_t +  5 * 16);
      const __m512 coef_r6  = _mm512_load_ps(current_coeff_t +  6 * 16);
      const __m512 coef_r7  = _mm512_load_ps(current_coeff_t +  7 * 16);
      const __m512 coef_r8  = _mm512_load_ps(current_coeff_t +  8 * 16);
      const __m512 coef_r9  = _mm512_load_ps(current_coeff_t +  9 * 16);
      const __m512 coef_r10 = _mm512_load_ps(current_coeff_t + 10 * 16);
      const __m512 coef_r11 = _mm512_load_ps(current_coeff_t + 11 * 16);
      const __m512 coef_r12 = _mm512_load_ps(current_coeff_t + 12 * 16);
      const __m512 coef_r13 = _mm512_load_ps(current_coeff_t + 13 * 16);
      const __m512 coef_r14 = _mm512_load_ps(current_coeff_t + 14 * 16);
      const __m512 coef_r15 = _mm512_load_ps(current_coeff_t + 15 * 16);

      const __m512i one_epi32 = _mm512_set1_epi32(1);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      // shorter SIMD-way - single memory load (hacky SIMD load from int vector ?)
      __m512i perm_0_low8 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      int iStart_low8 = program->pixel_offset[x];
      perm_0_low8 = _mm512_sub_epi32(perm_0_low8, _mm512_set1_epi32(iStart_low8)); // vpbroadcastd zmm, r32

      __m512i perm_0_high8 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 8]));
      int iStart_high8 = program->pixel_offset[x + 8];
      perm_0_high8 = _mm512_sub_epi32(perm_0_high8, _mm512_set1_epi32(iStart_high8)); // vpbroadcastd zmm, r32
      perm_0_high8 = _mm512_inserti64x4(perm_0_high8, _mm512_castsi512_si256(perm_0_high8), 1);// shift low 8 epi32 to high 8

      const __mmask16 k_high8 = 0xFF00;
      const __m512i perm_0 = _mm512_mask_blend_epi32(k_high8, perm_0_low8, perm_0_high8);

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr_low8 = src + iStart_low8 + y_from * src_pitch; // all permute offsets in a first group relative to this start offset
      const float* src_ptr_high8 = src + iStart_high8 + y_from * src_pitch; // all permute offsets in a second group relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode
      const int remaining_low8 = program->source_size - iStart_low8;
      const int remaining_high8 = program->source_size - iStart_high8;

      int rem1_low8 = std::max(0, std::min(16, remaining_low8));
      __mmask16 k1_low8 = (1U << rem1_low8) - 1;
      int rem1_high8 = std::max(0, std::min(16, remaining_high8));
      __mmask16 k1_high8 = (1U << rem1_high8) - 1;

      for (int y = y_from; y < y_to; y++)
      {
        // Taps are contiguous (0, 1, 2, 3 .. 7), so we increment perm indexes by 1.
        // To save register usage in ks16 version - calculate offsets at runtime
        // working indexes, reloaded from constant perm_0
        __m512i perm_0w = perm_0;
        __m512i perm_1w = _mm512_add_epi32(perm_0, one_epi32);

        const __m512i two_epi32 = _mm512_set1_epi32(2);

        __m512 data_src_low8, data_src2_low8;
        __m512 data_src_high8, data_src2_high8;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          // Load first 16 floats
          data_src_low8 = _mm512_maskz_loadu_ps(k1_low8, src_ptr_low8);

          // Load next 16 floats (offset by 16)
          int rem2 = std::max(0, std::min(16, remaining_low8 - 16));
          __mmask16 k2_low8 = (1U << rem2) - 1;
          data_src2_low8 = _mm512_maskz_loadu_ps(k2_low8, src_ptr_low8 + 16);

          // high8
          // Safe masked loads for the image edge
          // Load first 16 floats
          data_src_high8 = _mm512_maskz_loadu_ps(k1_high8, src_ptr_high8);

          // Load next 16 floats (offset by 16)
          int rem2_high8 = std::max(0, std::min(16, remaining_high8 - 16));
          __mmask16 k2_high8 = (1U << rem2_high8) - 1;
          data_src2_high8 = _mm512_maskz_loadu_ps(k2_high8, src_ptr_high8 + 16);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src_low8 = _mm512_loadu_ps(src_ptr_low8);
          data_src2_low8 = _mm512_loadu_ps(src_ptr_low8 + 16);
          data_src_high8 = _mm512_loadu_ps(src_ptr_high8);
          data_src2_high8 = _mm512_loadu_ps(src_ptr_high8 + 16);
        }

        __m512 data_0 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_1 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_1, coef_r1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_2 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_3 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_4 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_5 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
        result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_6 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_7 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
        result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_8 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_9 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_8, coef_r8, result0);
        result1 = _mm512_fmadd_ps(data_9, coef_r9, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_10 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_11 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_10, coef_r10, result0);
        result1 = _mm512_fmadd_ps(data_11, coef_r11, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_12 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_13 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_12, coef_r12, result0);
        result1 = _mm512_fmadd_ps(data_13, coef_r13, result1);

        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        __m512 data_14 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_0w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_0w, data_src2_high8));
        __m512 data_15 = _mm512_mask_blend_ps(k_high8, _mm512_permutex2var_ps(data_src_low8, perm_1w, data_src2_low8), _mm512_permutex2var_ps(data_src_high8, perm_1w, data_src2_high8));

        result0 = _mm512_fmadd_ps(data_14, coef_r14, result0);
        result1 = _mm512_fmadd_ps(data_15, coef_r15, result1);

        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result0, result1)); 

        dst_ptr += dst_pitch;
        src_ptr_low8 += src_pitch;
        src_ptr_high8 += src_pitch;
      }

      current_coeff_t += 16 * PIXELS_AT_A_TIME; // 256 floats per x-group (16 taps × 16 pixels)
    };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{});
    }
  }
}


// 16 target pixels at a time from 4 independent source groups of 4 output pixels each.
// Handles heavy downscaling where 2s8_ks16 is not feasible (each 8-px group needs >32 source floats)
// but each 4-px group still fits within a 32-float (2-ZMM) permutex window.
void resize_h_planar_float_avx512_permutex_vstripe_4s4_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 16;

  const int width_safe_mod = (program->safelimit_8_pixels.overread_possible ? program->safelimit_8_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 16);
  assert(program->target_size_alignment >= 16);
  assert(FRAME_ALIGN >= 64);
  assert(program->filter_size_alignment >= 8);

  const int max_scanlines = program->max_scanlines;

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);
    const float* AVS_RESTRICT current_coeff_t = (const float* AVS_RESTRICT)program->pixel_coefficient_AVX512_float_H;
    int x = 0;

    auto do_h_float_core = [&](auto partial_load) {
      const __m512 coef_r0  = _mm512_load_ps(current_coeff_t +  0 * 16);
      const __m512 coef_r1  = _mm512_load_ps(current_coeff_t +  1 * 16);
      const __m512 coef_r2  = _mm512_load_ps(current_coeff_t +  2 * 16);
      const __m512 coef_r3  = _mm512_load_ps(current_coeff_t +  3 * 16);
      const __m512 coef_r4  = _mm512_load_ps(current_coeff_t +  4 * 16);
      const __m512 coef_r5  = _mm512_load_ps(current_coeff_t +  5 * 16);
      const __m512 coef_r6  = _mm512_load_ps(current_coeff_t +  6 * 16);
      const __m512 coef_r7  = _mm512_load_ps(current_coeff_t +  7 * 16);
      const __m512 coef_r8  = _mm512_load_ps(current_coeff_t +  8 * 16);
      const __m512 coef_r9  = _mm512_load_ps(current_coeff_t +  9 * 16);
      const __m512 coef_r10 = _mm512_load_ps(current_coeff_t + 10 * 16);
      const __m512 coef_r11 = _mm512_load_ps(current_coeff_t + 11 * 16);
      const __m512 coef_r12 = _mm512_load_ps(current_coeff_t + 12 * 16);
      const __m512 coef_r13 = _mm512_load_ps(current_coeff_t + 13 * 16);
      const __m512 coef_r14 = _mm512_load_ps(current_coeff_t + 14 * 16);
      const __m512 coef_r15 = _mm512_load_ps(current_coeff_t + 15 * 16);

      const __m512i one_epi32 = _mm512_set1_epi32(1);

      // 4 source groups, each covering 4 consecutive output pixels (x+0..3, x+4..7, x+8..11, x+12..15).
      // Each group's perm indices are stored in the corresponding quarter of perm_0 (128-bit lane).
      const int iStart_g0 = program->pixel_offset[x];
      const int iStart_g1 = program->pixel_offset[x + 4];
      const int iStart_g2 = program->pixel_offset[x + 8];
      const int iStart_g3 = program->pixel_offset[x + 12];

      // Build combined perm vector: subtract each group's start from its own 4 offsets.
      // Positions 0..3 = g0-relative, 4..7 = g1-relative, 8..11 = g2-relative, 12..15 = g3-relative.
      const __m512i perm_0 = _mm512_sub_epi32(
        _mm512_loadu_si512((__m512i*)&program->pixel_offset[x]),
        _mm512_set_epi32(
          iStart_g3, iStart_g3, iStart_g3, iStart_g3,
          iStart_g2, iStart_g2, iStart_g2, iStart_g2,
          iStart_g1, iStart_g1, iStart_g1, iStart_g1,
          iStart_g0, iStart_g0, iStart_g0, iStart_g0
        )
      );

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_g0 = src + iStart_g0 + y_from * src_pitch;
      const float* src_g1 = src + iStart_g1 + y_from * src_pitch;
      const float* src_g2 = src + iStart_g2 + y_from * src_pitch;
      const float* src_g3 = src + iStart_g3 + y_from * src_pitch;

      const int rem_g0 = program->source_size - iStart_g0;
      const int rem_g1 = program->source_size - iStart_g1;
      const int rem_g2 = program->source_size - iStart_g2;
      const int rem_g3 = program->source_size - iStart_g3;

      const __mmask16 k1_g0 = (1U << std::max(0, std::min(16, rem_g0))) - 1;
      const __mmask16 k1_g1 = (1U << std::max(0, std::min(16, rem_g1))) - 1;
      const __mmask16 k1_g2 = (1U << std::max(0, std::min(16, rem_g2))) - 1;
      const __mmask16 k1_g3 = (1U << std::max(0, std::min(16, rem_g3))) - 1;

      for (int y = y_from; y < y_to; y++)
      {
        __m512 sg0, sg0b, sg1, sg1b, sg2, sg2b, sg3, sg3b;

        if constexpr (partial_load) {
          sg0  = _mm512_maskz_loadu_ps(k1_g0, src_g0);
          sg0b = _mm512_maskz_loadu_ps((1U << std::max(0, std::min(16, rem_g0 - 16))) - 1, src_g0 + 16);
          sg1  = _mm512_maskz_loadu_ps(k1_g1, src_g1);
          sg1b = _mm512_maskz_loadu_ps((1U << std::max(0, std::min(16, rem_g1 - 16))) - 1, src_g1 + 16);
          sg2  = _mm512_maskz_loadu_ps(k1_g2, src_g2);
          sg2b = _mm512_maskz_loadu_ps((1U << std::max(0, std::min(16, rem_g2 - 16))) - 1, src_g2 + 16);
          sg3  = _mm512_maskz_loadu_ps(k1_g3, src_g3);
          sg3b = _mm512_maskz_loadu_ps((1U << std::max(0, std::min(16, rem_g3 - 16))) - 1, src_g3 + 16);
        } else {
          sg0  = _mm512_loadu_ps(src_g0);       sg0b = _mm512_loadu_ps(src_g0 + 16);
          sg1  = _mm512_loadu_ps(src_g1);       sg1b = _mm512_loadu_ps(src_g1 + 16);
          sg2  = _mm512_loadu_ps(src_g2);       sg2b = _mm512_loadu_ps(src_g2 + 16);
          sg3  = _mm512_loadu_ps(src_g3);       sg3b = _mm512_loadu_ps(src_g3 + 16);
        }

        __m512i perm_0w = perm_0;
        __m512i perm_1w = _mm512_add_epi32(perm_0, one_epi32);
        const __m512i two_epi32 = _mm512_set1_epi32(2);

        // Gather from all 4 source groups using combined perm, then blend quarters.
        // Positions 0..3 from g0, 4..7 from g1, 8..11 from g2, 12..15 from g3.
        // Each permutex2var uses the same perm_w: the "wrong" quarters produce garbage but are discarded by blend.
#define G4(p) _mm512_mask_blend_ps(0xF000, \
    _mm512_mask_blend_ps(0x0F00, \
      _mm512_mask_blend_ps(0x00F0, \
        _mm512_permutex2var_ps(sg0, (p), sg0b), \
        _mm512_permutex2var_ps(sg1, (p), sg1b)), \
      _mm512_permutex2var_ps(sg2, (p), sg2b)), \
    _mm512_permutex2var_ps(sg3, (p), sg3b))

        __m512 result0 = _mm512_mul_ps(G4(perm_0w), coef_r0);
        __m512 result1 = _mm512_mul_ps(G4(perm_1w), coef_r1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r2, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r3, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r4, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r5, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r6, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r7, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r8, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r9, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r10, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r11, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r12, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r13, result1);
        perm_0w = _mm512_add_epi32(perm_0w, two_epi32);
        perm_1w = _mm512_add_epi32(perm_1w, two_epi32);

        result0 = _mm512_fmadd_ps(G4(perm_0w), coef_r14, result0);
        result1 = _mm512_fmadd_ps(G4(perm_1w), coef_r15, result1);

#undef G4

        _mm512_stream_ps(dst_ptr, _mm512_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_g0 += src_pitch; src_g1 += src_pitch;
        src_g2 += src_pitch; src_g3 += src_pitch;
      }

      current_coeff_t += 16 * PIXELS_AT_A_TIME;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
      do_h_float_core(std::false_type{});
    for (; x < width; x += PIXELS_AT_A_TIME)
      do_h_float_core(std::true_type{});
  }
}


//-------- 512 bit float Verticals

// base version, no horizontal unrolling
void resize_v_avx512_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

  const float* src = (const float*)src8;
  float* AVS_RESTRICT dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2; // Process pairs of rows for better efficiency
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;

    // 64 byte 16 floats (AVX512 register holds 16 floats)
    // no need for wmod8, alignment is safe 32 bytes at least - is it safe for 64 bytes ?
    for (int x = 0; x < width; x += 16) {
      __m512 result_single = _mm512_setzero_ps();
      __m512 result_single_2 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      // two result variables for potential parallel operation
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m512 coeff_even = _mm512_set1_ps(current_coeff[i]);
        __m512 coeff_odd = _mm512_set1_ps(current_coeff[i + 1]);

        __m512 src_even = _mm512_load_ps(src2_ptr);
        __m512 src_odd = _mm512_load_ps(src2_ptr + src_pitch);

        result_single = _mm512_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm512_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += 2 * src_pitch;
      }

      result_single = _mm512_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);
        __m512 src_val = _mm512_load_ps(src2_ptr);
        result_single = _mm512_fmadd_ps(src_val, coeff, result_single);
      }

      _mm512_stream_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// memory-optimized version of resize_v_avx512_planar_float
void resize_v_avx512_planar_float_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  const float* src = (const float*)src8;
  float* AVS_RESTRICT dst = (float*)dst8;

  const int dst_stride_float = dst_pitch / sizeof(float);
  const int src_stride_float = src_pitch / sizeof(float);

  const int kernel_size = program->filter_size_real;
  // Pre-calculate Mod2 size for the remainder loop
  const int kernel_size_mod2 = (kernel_size / 2) * 2;
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_row_start = src + offset * src_stride_float;

    int x = 0;

    // -----------------------------------------------------------------------
    // 128 pixels (512 bytes) per iteration
    // Uses ~17 ZMM registers. Safe for x64 (32 regs available).
    // Provides 8 independent dependency chains to hide FMA latency.
    // -----------------------------------------------------------------------
    const int width_mod128 = (width / 128) * 128;
    for (; x < width_mod128; x += 128) {
      __m512 result_1 = _mm512_setzero_ps();
      __m512 result_2 = _mm512_setzero_ps();
      __m512 result_3 = _mm512_setzero_ps();
      __m512 result_4 = _mm512_setzero_ps();
      __m512 result_5 = _mm512_setzero_ps();
      __m512 result_6 = _mm512_setzero_ps();
      __m512 result_7 = _mm512_setzero_ps();
      __m512 result_8 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_row_start + x;

      for (int i = 0; i < kernel_size; i++) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);

        // Loading 512 bytes contiguous memory (8 cache lines)
        __m512 src_1 = _mm512_load_ps(src2_ptr);       // 0..15
        __m512 src_2 = _mm512_load_ps(src2_ptr + 16);  // 16..31 (offset in floats)
        __m512 src_3 = _mm512_load_ps(src2_ptr + 32);
        __m512 src_4 = _mm512_load_ps(src2_ptr + 48);
        __m512 src_5 = _mm512_load_ps(src2_ptr + 64);
        __m512 src_6 = _mm512_load_ps(src2_ptr + 80);
        __m512 src_7 = _mm512_load_ps(src2_ptr + 96);
        __m512 src_8 = _mm512_load_ps(src2_ptr + 112);

        result_1 = _mm512_fmadd_ps(src_1, coeff, result_1);
        result_2 = _mm512_fmadd_ps(src_2, coeff, result_2);
        result_3 = _mm512_fmadd_ps(src_3, coeff, result_3);
        result_4 = _mm512_fmadd_ps(src_4, coeff, result_4);
        result_5 = _mm512_fmadd_ps(src_5, coeff, result_5);
        result_6 = _mm512_fmadd_ps(src_6, coeff, result_6);
        result_7 = _mm512_fmadd_ps(src_7, coeff, result_7);
        result_8 = _mm512_fmadd_ps(src_8, coeff, result_8);

        src2_ptr += src_stride_float;
      }

      _mm512_stream_ps(dst + x, result_1);
      _mm512_stream_ps(dst + x + 16, result_2);
      _mm512_stream_ps(dst + x + 32, result_3);
      _mm512_stream_ps(dst + x + 48, result_4);
      _mm512_stream_ps(dst + x + 64, result_5);
      _mm512_stream_ps(dst + x + 80, result_6);
      _mm512_stream_ps(dst + x + 96, result_7);
      _mm512_stream_ps(dst + x + 112, result_8);
    }

    // -----------------------------------------------------------------------
    // 64 pixels per iteration
    // -----------------------------------------------------------------------
    const int width_mod64 = (width / 64) * 64;
    for (; x < width_mod64; x += 64) {
      __m512 result_1 = _mm512_setzero_ps();
      __m512 result_2 = _mm512_setzero_ps();
      __m512 result_3 = _mm512_setzero_ps();
      __m512 result_4 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_row_start + x;

      for (int i = 0; i < kernel_size; i++) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);

        __m512 src_1 = _mm512_load_ps(src2_ptr);
        __m512 src_2 = _mm512_load_ps(src2_ptr + 16);
        __m512 src_3 = _mm512_load_ps(src2_ptr + 32);
        __m512 src_4 = _mm512_load_ps(src2_ptr + 48);

        result_1 = _mm512_fmadd_ps(src_1, coeff, result_1);
        result_2 = _mm512_fmadd_ps(src_2, coeff, result_2);
        result_3 = _mm512_fmadd_ps(src_3, coeff, result_3);
        result_4 = _mm512_fmadd_ps(src_4, coeff, result_4);

        src2_ptr += src_stride_float;
      }

      _mm512_stream_ps(dst + x, result_1);
      _mm512_stream_ps(dst + x + 16, result_2);
      _mm512_stream_ps(dst + x + 32, result_3);
      _mm512_stream_ps(dst + x + 48, result_4);
    }

    // -----------------------------------------------------------------------
    // 32 pixels per iteration
    // -----------------------------------------------------------------------
    const int width_mod32 = (width / 32) * 32;
    for (; x < width_mod32; x += 32) {
      __m512 result_1 = _mm512_setzero_ps();
      __m512 result_2 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_row_start + x;

      for (int i = 0; i < kernel_size; i++) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);

        __m512 src_1 = _mm512_load_ps(src2_ptr);
        __m512 src_2 = _mm512_load_ps(src2_ptr + 16);

        result_1 = _mm512_fmadd_ps(src_1, coeff, result_1);
        result_2 = _mm512_fmadd_ps(src_2, coeff, result_2);

        src2_ptr += src_stride_float;
      }

      _mm512_stream_ps(dst + x, result_1);
      _mm512_stream_ps(dst + x + 16, result_2);
    }

    // -----------------------------------------------------------------------
    // Remainder loop (16 pixels)
    // Uses vertical loop unrolling (pairs of taps) to hide FMA latency
    // because we don't have enough horizontal data to do it spatially.
    // -----------------------------------------------------------------------
    const int src_stride_2 = src_stride_float * 2;

    for (; x < width; x += 16) {
      __m512 result_single = _mm512_setzero_ps();
      __m512 result_single_2 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_row_start + x;
      int i = 0;

      // Process pairs of rows
      for (; i < kernel_size_mod2; i += 2) {
        __m512 coeff_even = _mm512_set1_ps(current_coeff[i]);
        __m512 coeff_odd = _mm512_set1_ps(current_coeff[i + 1]);

        __m512 src_even = _mm512_load_ps(src2_ptr);
        __m512 src_odd = _mm512_load_ps(src2_ptr + src_stride_float);

        result_single = _mm512_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm512_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += src_stride_2;
      }

      result_single = _mm512_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);
        __m512 src_val = _mm512_load_ps(src2_ptr);
        result_single = _mm512_fmadd_ps(src_val, coeff, result_single);
      }

      _mm512_stream_ps(dst + x, result_single);
    }

    dst += dst_stride_float;
    current_coeff += filter_size;
  }
}

// uint8_t
void resize_v_avx512_planar_uint8_t_w_sr(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);
  int filter_size = program->filter_size;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;
  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));
  __m512i zero = _mm512_setzero_si512();

  const int kernel_size = program->filter_size_real; // not the aligned

  const int width_mod128 = (width / 128) * 128;

  const __m512i perm_idx1 = _mm512_set_epi64(8 + 5, 8 + 4, 8 + 1, 8 + 0, 5, 4, 1, 0);
  const __m512i perm_idx2 = _mm512_set_epi64(8 + 7, 8 + 6, 8 + 3, 8 + 2, 7, 6, 3, 2);

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* AVS_RESTRICT src_ptr = src + offset * src_pitch;

    for (int x = 0; x < width_mod128; x += 128) {

      __m512i result_lo = rounder;
      __m512i result_hi = rounder;
      __m512i result_lo2 = rounder;
      __m512i result_hi2 = rounder;

      __m512i result_lo_2 = rounder;
      __m512i result_hi_2 = rounder;
      __m512i result_lo2_2 = rounder;
      __m512i result_hi2_2 = rounder;

      const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      int i = 0;
      // 128 byte 128 pixel
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m512i coeff = _mm512_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m512i src_1_1 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr))); // 32x 8->16bit pixels
        __m512i src_1_2 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + 32))); // 32x 8->16bit pixels
        __m512i src_2_1 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + 64))); // 32x 8->16bit pixels
        __m512i src_2_2 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + 96))); // 32x 8->16bit pixels

        __m512i src_lo = _mm512_unpacklo_epi16(src_1_1, zero);
        __m512i src_hi = _mm512_unpackhi_epi16(src_1_1, zero);
        __m512i src_lo2 = _mm512_unpacklo_epi16(src_1_2, zero);
        __m512i src_hi2 = _mm512_unpackhi_epi16(src_1_2, zero);

        __m512i src_lo_2 = _mm512_unpacklo_epi16(src_2_1, zero);
        __m512i src_hi_2 = _mm512_unpackhi_epi16(src_2_1, zero);
        __m512i src_lo2_2 = _mm512_unpacklo_epi16(src_2_2, zero);
        __m512i src_hi2_2 = _mm512_unpackhi_epi16(src_2_2, zero);

        result_lo = _mm512_add_epi32(result_lo, _mm512_madd_epi16(src_lo, coeff)); // a*b + c
        result_hi = _mm512_add_epi32(result_hi, _mm512_madd_epi16(src_hi, coeff)); // a*b + c
        result_lo2 = _mm512_add_epi32(result_lo2, _mm512_madd_epi16(src_lo2, coeff)); // a*b + c
        result_hi2 = _mm512_add_epi32(result_hi2, _mm512_madd_epi16(src_hi2, coeff)); // a*b + c

        result_lo_2 = _mm512_add_epi32(result_lo_2, _mm512_madd_epi16(src_lo_2, coeff)); // a*b + c
        result_hi_2 = _mm512_add_epi32(result_hi_2, _mm512_madd_epi16(src_hi_2, coeff)); // a*b + c
        result_lo2_2 = _mm512_add_epi32(result_lo2_2, _mm512_madd_epi16(src_lo2_2, coeff)); // a*b + c
        result_hi2_2 = _mm512_add_epi32(result_hi2_2, _mm512_madd_epi16(src_hi2_2, coeff)); // a*b + c

        src2_ptr += src_pitch;

      }

      // scale back, store
      // shift back integer arithmetic 14 bits precision
      result_lo = _mm512_srai_epi32(result_lo, FPScale8bits);
      result_hi = _mm512_srai_epi32(result_hi, FPScale8bits);
      result_lo2 = _mm512_srai_epi32(result_lo2, FPScale8bits);
      result_hi2 = _mm512_srai_epi32(result_hi2, FPScale8bits);

      result_lo_2 = _mm512_srai_epi32(result_lo_2, FPScale8bits);
      result_hi_2 = _mm512_srai_epi32(result_hi_2, FPScale8bits);
      result_lo2_2 = _mm512_srai_epi32(result_lo2_2, FPScale8bits);
      result_hi2_2 = _mm512_srai_epi32(result_hi2_2, FPScale8bits);

      __m512i result_2x8x_uint16 = _mm512_packus_epi32(result_lo, result_hi);
      __m512i result2_2x8x_uint16 = _mm512_packus_epi32(result_lo2, result_hi2);

      __m512i result_2x8x_uint16_2 = _mm512_packus_epi32(result_lo_2, result_hi_2);
      __m512i result2_2x8x_uint16_2 = _mm512_packus_epi32(result_lo2_2, result_hi2_2);

      __m512i pack_1 = _mm512_permutex2var_epi64(result_2x8x_uint16, perm_idx1, result2_2x8x_uint16);
      __m512i pack_2 = _mm512_permutex2var_epi64(result_2x8x_uint16, perm_idx2, result2_2x8x_uint16);

      __m512i pack_1_2 = _mm512_permutex2var_epi64(result_2x8x_uint16_2, perm_idx1, result2_2x8x_uint16_2);
      __m512i pack_2_2 = _mm512_permutex2var_epi64(result_2x8x_uint16_2, perm_idx2, result2_2x8x_uint16_2);

      __m512i res = _mm512_packus_epi16(pack_1, pack_2);
      __m512i res_2 = _mm512_packus_epi16(pack_1_2, pack_2_2);

      _mm512_store_si512(reinterpret_cast<__m512i*>(dst + x), res);
      _mm512_store_si512(reinterpret_cast<__m512i*>(dst + x + 64), res_2);

    }

    // 64 byte 64 pixel
    // no need wmod16, alignment is safe at least 32
    for (int x = width_mod128; x < width; x += 64) {

      __m512i result_lo = rounder;
      __m512i result_hi = rounder;

      __m512i result_lo2 = rounder;
      __m512i result_hi2 = rounder;

      const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      int i = 0;
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m512i coeff = _mm512_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m512i src_1_1 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr))); // 32x 8->16bit pixels
        __m512i src_1_2 = _mm512_cvtepu8_epi16(_mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + 32))); // 32x 8->16bit pixels

        __m512i src_lo = _mm512_unpacklo_epi16(src_1_1, zero);
        __m512i src_hi = _mm512_unpackhi_epi16(src_1_1, zero);

        __m512i src_lo2 = _mm512_unpacklo_epi16(src_1_2, zero);
        __m512i src_hi2 = _mm512_unpackhi_epi16(src_1_2, zero);

        result_lo = _mm512_add_epi32(result_lo, _mm512_madd_epi16(src_lo, coeff)); // a*b + c
        result_hi = _mm512_add_epi32(result_hi, _mm512_madd_epi16(src_hi, coeff)); // a*b + c

        result_lo2 = _mm512_add_epi32(result_lo2, _mm512_madd_epi16(src_lo2, coeff)); // a*b + c
        result_hi2 = _mm512_add_epi32(result_hi2, _mm512_madd_epi16(src_hi2, coeff)); // a*b + c

        src2_ptr += src_pitch;

      }

      // scale back, store
      // shift back integer arithmetic 14 bits precision
      result_lo = _mm512_srai_epi32(result_lo, FPScale8bits);
      result_hi = _mm512_srai_epi32(result_hi, FPScale8bits);

      result_lo2 = _mm512_srai_epi32(result_lo2, FPScale8bits);
      result_hi2 = _mm512_srai_epi32(result_hi2, FPScale8bits);

      __m512i result_2x8x_uint16 = _mm512_packus_epi32(result_lo, result_hi);
      __m512i result_2x8x_uint16_2 = _mm512_packus_epi32(result_lo2, result_hi2);

      __m512i pack_1 = _mm512_permutex2var_epi64(result_2x8x_uint16, perm_idx1, result_2x8x_uint16_2);
      __m512i pack_2 = _mm512_permutex2var_epi64(result_2x8x_uint16, perm_idx2, result_2x8x_uint16_2);

      __m512i res = _mm512_packus_epi16(pack_1, pack_2);

      _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + x), res);

    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

//uint16_t
template<bool lessthan16bit>
void resize_v_avx512_planar_uint16_t_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  int filter_size = program->filter_size;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

  const __m512i zero = _mm512_setzero_si512();

  const int width_mod64 = (width / 64) * 64;

  // for 16 bits only
  const __m512i shifttosigned = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const __m512i rounder = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t * AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int kernel_size = program->filter_size_real; // not the aligned

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit); // clamp limit for <16 bits

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + offset * src_pitch;

    // 128 byte 32 word
    for (int x = 0; x < width_mod64; x += 64) {

      __m512i result_lo = rounder;
      __m512i result_hi = rounder;

      __m512i result_lo_2 = rounder;
      __m512i result_hi_2 = rounder;

      const uint16_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      int i = 0;
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m512i coeff = _mm512_set1_epi16(current_coeff[i]); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m512i src = _mm512_load_si512(reinterpret_cast<const __m512i*>(src2_ptr)); // 32x 16bit pixels
        __m512i src_2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(src2_ptr + 32)); // 32x 16bit pixels

        if constexpr (!lessthan16bit) {
          src = _mm512_add_epi16(src, shifttosigned);
          src_2 = _mm512_add_epi16(src_2, shifttosigned);
        }

        __m512i src_lo = _mm512_unpacklo_epi16(src, zero);
        __m512i src_hi = _mm512_unpackhi_epi16(src, zero);

        __m512i src_lo_2 = _mm512_unpacklo_epi16(src_2, zero);
        __m512i src_hi_2 = _mm512_unpackhi_epi16(src_2, zero);

        result_lo = _mm512_add_epi32(result_lo, _mm512_madd_epi16(src_lo, coeff)); // a*b + c
        result_hi = _mm512_add_epi32(result_hi, _mm512_madd_epi16(src_hi, coeff)); // a*b + c

        result_lo_2 = _mm512_add_epi32(result_lo_2, _mm512_madd_epi16(src_lo_2, coeff)); // a*b + c
        result_hi_2 = _mm512_add_epi32(result_hi_2, _mm512_madd_epi16(src_hi_2, coeff)); // a*b + c

        src2_ptr += src_pitch;
      }

      if constexpr (!lessthan16bit) {
        result_lo = _mm512_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm512_add_epi32(result_hi, shiftfromsigned);

        result_lo_2 = _mm512_add_epi32(result_lo_2, shiftfromsigned);
        result_hi_2 = _mm512_add_epi32(result_hi_2, shiftfromsigned);

      }
      // shift back integer arithmetic 13 bits precision
      result_lo = _mm512_srai_epi32(result_lo, FPScale16bits);
      result_hi = _mm512_srai_epi32(result_hi, FPScale16bits);

      result_lo_2 = _mm512_srai_epi32(result_lo_2, FPScale16bits);
      result_hi_2 = _mm512_srai_epi32(result_hi_2, FPScale16bits);

      __m512i result_2x8x_uint16 = _mm512_packus_epi32(result_lo, result_hi);
      __m512i result_2x8x_uint16_2 = _mm512_packus_epi32(result_lo_2, result_hi_2);
      if constexpr (lessthan16bit) {
        result_2x8x_uint16 = _mm512_min_epu16(result_2x8x_uint16, clamp_limit); // extra clamp for 10-14 bit
        result_2x8x_uint16_2 = _mm512_min_epu16(result_2x8x_uint16_2, clamp_limit); // extra clamp for 10-14 bit
      }
      _mm512_store_si512(reinterpret_cast<__m512i*>(dst + x), result_2x8x_uint16);
      _mm512_store_si512(reinterpret_cast<__m512i*>(dst + x + 32), result_2x8x_uint16_2);
    }

    // last 32
    // 64 byte 32 word
    for (int x = width_mod64; x < width; x += 32) {

      __m512i result_lo = rounder;
      __m512i result_hi = rounder;

      const uint16_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      int i = 0;
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m512i coeff = _mm512_set1_epi16(current_coeff[i]); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m512i src = _mm512_load_si512(reinterpret_cast<const __m512i*>(src2_ptr)); // 32x 16bit pixels
        if constexpr (!lessthan16bit) {
          src = _mm512_add_epi16(src, shifttosigned);
        }
        __m512i src_lo = _mm512_unpacklo_epi16(src, zero);
        __m512i src_hi = _mm512_unpackhi_epi16(src, zero);
        result_lo = _mm512_add_epi32(result_lo, _mm512_madd_epi16(src_lo, coeff)); // a*b + c
        result_hi = _mm512_add_epi32(result_hi, _mm512_madd_epi16(src_hi, coeff)); // a*b + c

        src2_ptr += src_pitch;
      }

      if constexpr (!lessthan16bit) {
        result_lo = _mm512_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm512_add_epi32(result_hi, shiftfromsigned);
      }
      // shift back integer arithmetic 13 bits precision
      result_lo = _mm512_srai_epi32(result_lo, FPScale16bits);
      result_hi = _mm512_srai_epi32(result_hi, FPScale16bits);

      __m512i result_2x8x_uint16 = _mm512_packus_epi32(result_lo, result_hi);
      if constexpr (lessthan16bit) {
        result_2x8x_uint16 = _mm512_min_epu16(result_2x8x_uint16, clamp_limit); // extra clamp for 10-14 bit
      }
      _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + x), result_2x8x_uint16);

    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// avx512 16
template void resize_v_avx512_planar_uint16_t_w_sr<false>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
// avx512 10-14bit
template void resize_v_avx512_planar_uint16_t_w_sr<true>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

//----------------------- generic horizontal avx512 float

// AVX512 Horizontal float

// Three helpers, each for processing 4 target pixels from 16, 8 and 4 source pixel/coeff pairs.

// Helper, _mm256_zextps128_ps256 exists only in AVX512 VL
// zero-extend 128-bit float vector to 256-bit float vector
AVS_FORCEINLINE static __m256 _mm256_zextps128_ps256_simul_avx(__m128 a)
{
  // Flags defines by MSVC at /AVX512 mode
  // other flags: __AVX512F__, __AVX512CD__, __AVX512VL__, __AVX512BW__, __AVX512DQ__
#ifdef __AVX512VL__
  return _mm256_zextps128_ps256(a);
#else
  __m256 zero_v = _mm256_setzero_ps();
  return _mm256_insertf128_ps(zero_v, a, 0);
#endif
}

// 4 target pixels, each from 16 source pixel/coeff pair
// Called only when accessing 16 source pixels and coefficients at a time is safe
AVS_FORCEINLINE static void process_pix4_coeff16_h_float_core_512(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m512& result1, __m512& result2, __m512& result3, __m512& result4)
{
  // 16 source floats for each of the four beginning source offsets
  __m512 data_1 = _mm512_loadu_ps(src + begin1);
  __m512 data_2 = _mm512_loadu_ps(src + begin2);
  __m512 data_3 = _mm512_loadu_ps(src + begin3);
  __m512 data_4 = _mm512_loadu_ps(src + begin4);

  // 16 coefficients for each of the four output pixels
  __m512 coeff_1 = _mm512_loadu_ps(current_coeff);               // 16 coeffs for pixel 1
  __m512 coeff_2 = _mm512_loadu_ps(current_coeff + 1 * filter_size); // 16 coeffs for pixel 2
  __m512 coeff_3 = _mm512_loadu_ps(current_coeff + 2 * filter_size); // 16 coeffs for pixel 3
  __m512 coeff_4 = _mm512_loadu_ps(current_coeff + 3 * filter_size); // 16 coeffs for pixel 4

  // multiply and accumulate
  result1 = _mm512_fmadd_ps(data_1, coeff_1, result1);
  result2 = _mm512_fmadd_ps(data_2, coeff_2, result2);
  result3 = _mm512_fmadd_ps(data_3, coeff_3, result3);
  result4 = _mm512_fmadd_ps(data_4, coeff_4, result4);
}

// 4 target pixels, each from 8 source pixel/coeff pair
// Called only when accessing 8 source pixels and coefficients at a time is safe
AVS_FORCEINLINE static void process_pix4_coeff8_h_float_core(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4)
{
  // Load 8 source floats for each of the four beginning source offsets
  // Load 8 coefficients for each of the four output pixels
  __m256 data_1 = _mm256_loadu_ps(src + begin1);
  __m256 coeff_1 = _mm256_load_ps(current_coeff);                    // 8 coeffs for pixel 1
  result1 = _mm256_fmadd_ps(data_1, coeff_1, result1);

  __m256 data_2 = _mm256_loadu_ps(src + begin2);
  __m256 coeff_2 = _mm256_load_ps(current_coeff + 1 * filter_size); // 8 coeffs for pixel 2
  result2 = _mm256_fmadd_ps(data_2, coeff_2, result2);

  __m256 data_3 = _mm256_loadu_ps(src + begin3);
  __m256 coeff_3 = _mm256_load_ps(current_coeff + 2 * filter_size); // 8 coeffs for pixel 3
  result3 = _mm256_fmadd_ps(data_3, coeff_3, result3);

  __m256 data_4 = _mm256_loadu_ps(src + begin4);
  __m256 coeff_4 = _mm256_load_ps(current_coeff + 3 * filter_size); // 8 coeffs for pixel 4
  result4 = _mm256_fmadd_ps(data_4, coeff_4, result4);
}

// 4 target pixels, each from 4 source pixel/coeff pair.
// Called only for first iteration when results are not initialized.
// Otherwise same as process_pix4_coeff8_h_float_core.
AVS_FORCEINLINE static void process_pix4_coeff4_h_float_core_first(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4)
{
  // Pixel 1: Load, Multiply, and Zero-Extend to __m256
  __m128 data_1 = _mm_loadu_ps(src + begin1);
  __m128 coeff_1 = _mm_load_ps(current_coeff);
  __m128 mul_result1 = _mm_mul_ps(data_1, coeff_1);
  result1 = _mm256_zextps128_ps256(mul_result1); // Sets upper 128 bits to zero

  // Pixel 2: Load, Multiply, and Zero-Extend to __m256
  __m128 data_2 = _mm_loadu_ps(src + begin2);
  __m128 coeff_2 = _mm_load_ps(current_coeff + 1 * filter_size);
  __m128 mul_result2 = _mm_mul_ps(data_2, coeff_2);
  result2 = _mm256_zextps128_ps256(mul_result2); // Sets upper 128 bits to zero

  // Pixel 3: Load, Multiply, and Zero-Extend to __m256
  __m128 data_3 = _mm_loadu_ps(src + begin3);
  __m128 coeff_3 = _mm_load_ps(current_coeff + 2 * filter_size);
  __m128 mul_result3 = _mm_mul_ps(data_3, coeff_3);
  result3 = _mm256_zextps128_ps256(mul_result3); // Sets upper 128 bits to zero

  // Pixel 4: Load, Multiply, and Zero-Extend to __m256
  __m128 data_4 = _mm_loadu_ps(src + begin4);
  __m128 coeff_4 = _mm_load_ps(current_coeff + 3 * filter_size);
  __m128 mul_result4 = _mm_mul_ps(data_4, coeff_4);
  result4 = _mm256_zextps128_ps256(mul_result4); // Sets upper 128 bits to zero
}

// filtersize_hint: special: 0..4 for 4,8,16,24,32. Generic: -1
// filter_size is an aligned value and always multiple of 8 (prerequisite)
// Processing rules:
// if filtersize_hint==0: filter size <=4, do one coeff4 step only
// if filtersize_hint>=2: do 1 or 2 coeff16 steps
// if filtersize_hint==1 or 3: do 1 coeff8 step (0*16 or 1*16 step done already)
// if filtersize_hint==-1: unknown filter size, do 16,8 steps as possible
template<bool safe_aligned_mode, int filtersize_hint>
AVS_FORCEINLINE static void process_four_pixels_h_float_pix4of16_ks_4_8_16(
  const float* src_ptr,
  int begin1, int begin2, int begin3, int begin4,
  float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4,
  int kernel_size)
{

  // very special case: filter size <= 4
  if constexpr (safe_aligned_mode) {
    if (filtersize_hint == 0) {
      // Process 4 target pixels and 4 source pixels/coefficients at a time
      // XMM-based loop internally, but returns __m256 with upper 128 cleared
      // Do not assume initialized zeros in result1..4, they will be set here.
      process_pix4_coeff4_h_float_core_first(
        src_ptr + 0, begin1, begin2, begin3, begin4,
        current_coeff + 0,
        filter_size,
        result1, result2, result3, result4);
      return;
    }
  }

  int i = 0;

  // do by 16 coeffs until possible
  if (filtersize_hint == -1 || filtersize_hint >= 2) {
    __m512 result1_512 = _mm512_setzero_ps();
    __m512 result2_512 = _mm512_setzero_ps();
    __m512 result3_512 = _mm512_setzero_ps();
    __m512 result4_512 = _mm512_setzero_ps();
    const int ksmod16 = safe_aligned_mode ? (filter_size / 16 * 16) : (kernel_size / 16 * 16);
    // Process 4 target pixels and 16 source pixels/coefficients at a time (ZMM-based loop)
    for (; i < ksmod16; i += 16) {
      process_pix4_coeff16_h_float_core_512(
        src_ptr + i, begin1, begin2, begin3, begin4,
        current_coeff + i,
        filter_size,
        result1_512, result2_512, result3_512, result4_512);
    }
    // Horizontal sum reduction from __m512 to __m256
    result1 = _mm256_add_ps(_mm512_castps512_ps256(result1_512), _mm512_extractf32x8_ps(result1_512, 1));
    result2 = _mm256_add_ps(_mm512_castps512_ps256(result2_512), _mm512_extractf32x8_ps(result2_512, 1));
    result3 = _mm256_add_ps(_mm512_castps512_ps256(result3_512), _mm512_extractf32x8_ps(result3_512, 1));
    result4 = _mm256_add_ps(_mm512_castps512_ps256(result4_512), _mm512_extractf32x8_ps(result4_512, 1));
  }

  // filter sizes 16 or 32 can return here
  if constexpr (safe_aligned_mode && (filtersize_hint == 2 || filtersize_hint == 4)) {
    return;
  }

  if constexpr (!safe_aligned_mode) {
    if (i == kernel_size) return; // kernel_size is not known compile time
  }

  // When to do the coeff8 step:
  // not safe-aligned mode: always. E.g. kernel_size == 28 -> 16 done, now 10 rest, do 8 next
  // filtersize_hint == -1: not-compile-time known filtersize (kernel_size / 16 * 16 done, rest follows)
  // filtersize_hint == 1 or 3: 0*16 or 1*16 done, now do 1*8
  if (!safe_aligned_mode || filtersize_hint == -1 || filtersize_hint == 1 || filtersize_hint == 3) {
    // 32 bytes contain 8 floats. We will use 256-bit registers (YMM).
    const int ksmod8 = safe_aligned_mode ? (filter_size / 8 * 8) : (kernel_size / 8 * 8);

    // Process 4 target pixels and 8 source pixels/coefficients at a time (YMM-based loop)
    for (; i < ksmod8; i += 8) {
      process_pix4_coeff8_h_float_core(
        src_ptr + i, begin1, begin2, begin3, begin4,
        current_coeff + i,
        filter_size,
        result1, result2, result3, result4);
    }
  }

  if constexpr (!safe_aligned_mode) {
    // Right edge case.
    // Coeffs are zero padded, reading them is no problem.
    // But if we read past the end of source then we can get possible NaN contamination.
    // Handle the remainder: 1 to 7 source/coefficient elements.
    // unaligned_kernel_size is used here, it's guaranteed that reading unaligned_kernel_size elements
    // from any pixel_offset[] is safe and ends within the source buffer.
    // Optional 4-2-1 processing loop.

    if (i == kernel_size) return;

    // --- Define Base Pointers for Source and Coefficients ---
    const float* src_ptr1 = src_ptr + begin1;
    const float* src_ptr2 = src_ptr + begin2;
    const float* src_ptr3 = src_ptr + begin3;
    const float* src_ptr4 = src_ptr + begin4;

    float* current_coeff2 = current_coeff + 1 * filter_size;
    float* current_coeff3 = current_coeff + 2 * filter_size;
    float* current_coeff4 = current_coeff + 3 * filter_size;

    const int ksmod4 = kernel_size / 4 * 4;

    // -------------------------------------------------------------------
    // Mod 4 Block (4 elements for four pixels using __m128)
    // -------------------------------------------------------------------
    if (i < ksmod4) {
      // Load 4 source floats and 4 coefficients for each of the four output pixels
      __m128 data_1 = _mm_loadu_ps(src_ptr1 + i);
      __m128 coeff_1 = _mm_loadu_ps(current_coeff + i);
      __m128 temp_result1 = _mm_mul_ps(data_1, coeff_1);

      __m128 data_2 = _mm_loadu_ps(src_ptr2 + i);
      __m128 coeff_2 = _mm_loadu_ps(current_coeff2 + i);
      __m128 temp_result2 = _mm_mul_ps(data_2, coeff_2);

      __m128 data_3 = _mm_loadu_ps(src_ptr3 + i);
      __m128 coeff_3 = _mm_loadu_ps(current_coeff3 + i);
      __m128 temp_result3 = _mm_mul_ps(data_3, coeff_3);

      __m128 data_4 = _mm_loadu_ps(src_ptr4 + i);
      __m128 coeff_4 = _mm_loadu_ps(current_coeff4 + i);
      __m128 temp_result4 = _mm_mul_ps(data_4, coeff_4);

      // --- Accumulate 128-bit results into 256-bit registers ---
      // Note: Since we are using __m256, we must zero the high 128-bits before insertion/addition.

      result1 = _mm256_add_ps(result1, _mm256_zextps128_ps256(temp_result1));
      result2 = _mm256_add_ps(result2, _mm256_zextps128_ps256(temp_result2));
      result3 = _mm256_add_ps(result3, _mm256_zextps128_ps256(temp_result3));
      result4 = _mm256_add_ps(result4, _mm256_zextps128_ps256(temp_result4));

      i += 4;
      if (i == kernel_size) return;
    }

    const int ksmod2 = kernel_size / 2 * 2;

    // -------------------------------------------------------------------
    // New Mod 2 Block (2 elements for four pixels using __m128)
    // -------------------------------------------------------------------
    if (i < ksmod2) {
      // We only need to load 2 elements (4 floats) for the __m128 load, 
      // but the low 2 elements of the __m128 register are used.
      // Since we use the scalar accumulation method, we load 4, but only the 
      // first 2 elements will hold non-zero data (or load 2, and rely on 
      // the two __m128 registers to contain the result).

      // Let's stick to using the low 2 elements of __m128 for 2 elements.

      // Load 2 source floats and 2 coefficients for each of the four output pixels
      __m128 data_1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(src_ptr1 + i))); // Load 2 floats (double)
      __m128 coeff_1 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(current_coeff + i)));
      __m128 temp_result1 = _mm_mul_ps(data_1, coeff_1);

      __m128 data_2 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(src_ptr2 + i)));
      __m128 coeff_2 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(current_coeff2 + i)));
      __m128 temp_result2 = _mm_mul_ps(data_2, coeff_2);

      __m128 data_3 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(src_ptr3 + i)));
      __m128 coeff_3 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(current_coeff3 + i)));
      __m128 temp_result3 = _mm_mul_ps(data_3, coeff_3);

      __m128 data_4 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(src_ptr4 + i)));
      __m128 coeff_4 = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(current_coeff4 + i)));
      __m128 temp_result4 = _mm_mul_ps(data_4, coeff_4);

      result1 = _mm256_add_ps(result1, _mm256_zextps128_ps256(temp_result1));
      result2 = _mm256_add_ps(result2, _mm256_zextps128_ps256(temp_result2));
      result3 = _mm256_add_ps(result3, _mm256_zextps128_ps256(temp_result3));
      result4 = _mm256_add_ps(result4, _mm256_zextps128_ps256(temp_result4));

      i += 2;
      if (i == kernel_size) return;
    }

    // -------------------------------------------------------------------
    // Fallback Scalar Operation (1 element remaining)
    // -------------------------------------------------------------------
    if (i < kernel_size) {

      // Optimized scalar loop for the single remaining element
      float final_scalar1 = src_ptr1[i] * current_coeff[i];
      float final_scalar2 = src_ptr2[i] * current_coeff2[i];
      float final_scalar3 = src_ptr3[i] * current_coeff3[i];
      float final_scalar4 = src_ptr4[i] * current_coeff4[i];

      __m128 s1_128 = _mm_set_ss(final_scalar1);
      __m128 s2_128 = _mm_set_ss(final_scalar2);
      __m128 s3_128 = _mm_set_ss(final_scalar3);
      __m128 s4_128 = _mm_set_ss(final_scalar4);

      result1 = _mm256_add_ps(result1, _mm256_zextps128_ps256(s1_128));
      result2 = _mm256_add_ps(result2, _mm256_zextps128_ps256(s2_128));
      result3 = _mm256_add_ps(result3, _mm256_zextps128_ps256(s3_128));
      result4 = _mm256_add_ps(result4, _mm256_zextps128_ps256(s4_128));

      // i is now equal to kernel_size (i++)
    }
  }
}


template<bool is_safe, int filtersize_hint>
AVS_FORCEINLINE static void process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16(
  const float* src, int x, float* current_coeff_base,
  int filter_size, // 8, 16, 24, 32 are quasi-constexpr here, others not compile-time known but still aligned to 8
  float* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 8);

  float* current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;
  const __m256 zero256 = _mm256_setzero_ps();

  // --- Block 1: Pixels 0, 1, 2, 3 ---
  __m256 result0 = zero256;
  __m256 result1 = zero256;
  __m256 result2 = zero256;
  __m256 result3 = zero256;

  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  int begin2 = program->pixel_offset[x + 2];
  int begin3 = program->pixel_offset[x + 3];

  process_four_pixels_h_float_pix4of16_ks_4_8_16<is_safe, filtersize_hint>(
    src, begin0, begin1, begin2, begin3, current_coeff, filter_size,
    result0, result1, result2, result3, unaligned_kernel_size);
  current_coeff += 4 * filter_size;

  // --- Block 2: Pixels 4, 5, 6, 7 ---
  __m256 result4 = zero256;
  __m256 result5 = zero256;
  __m256 result6 = zero256;
  __m256 result7 = zero256;

  int begin4 = program->pixel_offset[x + 4];
  int begin5 = program->pixel_offset[x + 5];
  int begin6 = program->pixel_offset[x + 6];
  int begin7 = program->pixel_offset[x + 7];

  process_four_pixels_h_float_pix4of16_ks_4_8_16<is_safe, filtersize_hint>(
    src, begin4, begin5, begin6, begin7, current_coeff, filter_size,
    result4, result5, result6, result7, unaligned_kernel_size);
  current_coeff += 4 * filter_size;

  // ---------------------------------------------------------------------------
  // REDUCTION FOR PIXELS 0-7 (Result256_low)
  // ---------------------------------------------------------------------------

  // Round 1: Reduce pairs (8 vectors -> 4 vectors)
  __m256 sum01 = _mm256_hadd_ps(result0, result1);
  __m256 sum23 = _mm256_hadd_ps(result2, result3);
  __m256 sum45 = _mm256_hadd_ps(result4, result5);
  __m256 sum67 = _mm256_hadd_ps(result6, result7);

  // Round 2: Reduce quads (4 vectors -> 2 vectors)
  __m256 sum0123 = _mm256_hadd_ps(sum01, sum23);
  __m256 sum4567 = _mm256_hadd_ps(sum45, sum67);

  // Round 3: Final Merge (Add Lower 128-bit to Upper 128-bit)
  __m128 lo_0123 = _mm256_castps256_ps128(sum0123);
  __m128 lo_4567 = _mm256_castps256_ps128(sum4567);
  __m256 result_lo = _mm256_insertf128_ps(_mm256_castps128_ps256(lo_0123), lo_4567, 1);

  __m128 hi_0123 = _mm256_extractf128_ps(sum0123, 1);
  __m128 hi_4567 = _mm256_extractf128_ps(sum4567, 1);
  __m256 result_hi = _mm256_insertf128_ps(_mm256_castps128_ps256(hi_0123), hi_4567, 1);

  // Assemble the Low 256-bit result (Pixels 0-7)
  __m256 result256_low = _mm256_add_ps(result_lo, result_hi);


  // --- Block 3: Pixels 8, 9, 10, 11 ---
  __m256 result8 = zero256;
  __m256 result9 = zero256;
  __m256 result10 = zero256;
  __m256 result11 = zero256;

  int begin8 = program->pixel_offset[x + 8];
  int begin9 = program->pixel_offset[x + 9];
  int begin10 = program->pixel_offset[x + 10];
  int begin11 = program->pixel_offset[x + 11];

  process_four_pixels_h_float_pix4of16_ks_4_8_16<is_safe, filtersize_hint>(
    src, begin8, begin9, begin10, begin11, current_coeff, filter_size,
    result8, result9, result10, result11, unaligned_kernel_size);
  current_coeff += 4 * filter_size;

  // --- Block 4: Pixels 12, 13, 14, 15 ---
  __m256 result12 = zero256;
  __m256 result13 = zero256;
  __m256 result14 = zero256;
  __m256 result15 = zero256;

  int begin12 = program->pixel_offset[x + 12];
  int begin13 = program->pixel_offset[x + 13];
  int begin14 = program->pixel_offset[x + 14];
  int begin15 = program->pixel_offset[x + 15];

  process_four_pixels_h_float_pix4of16_ks_4_8_16<is_safe, filtersize_hint>(
    src, begin12, begin13, begin14, begin15, current_coeff, filter_size,
    result12, result13, result14, result15, unaligned_kernel_size);


  // ---------------------------------------------------------------------------
  // REDUCTION FOR PIXELS 8-15 (Result256_high)
  // ---------------------------------------------------------------------------

  // Round 1: Reduce pairs (8 vectors -> 4 vectors)
  __m256 sum89 = _mm256_hadd_ps(result8, result9);
  __m256 sum1011 = _mm256_hadd_ps(result10, result11);
  __m256 sum1213 = _mm256_hadd_ps(result12, result13);
  __m256 sum1415 = _mm256_hadd_ps(result14, result15);

  // Round 2: Reduce quads (4 vectors -> 2 vectors)
  __m256 sum8_11 = _mm256_hadd_ps(sum89, sum1011);
  __m256 sum12_15 = _mm256_hadd_ps(sum1213, sum1415);

  // Round 3: Final Merge (Add Lower 128-bit to Upper 128-bit)
  __m128 lo_8_11 = _mm256_castps256_ps128(sum8_11);
  __m128 lo_12_15 = _mm256_castps256_ps128(sum12_15);
  __m256 result_lo_high = _mm256_insertf128_ps(_mm256_castps128_ps256(lo_8_11), lo_12_15, 1);

  __m128 hi_8_11 = _mm256_extractf128_ps(sum8_11, 1);
  __m128 hi_12_15 = _mm256_extractf128_ps(sum12_15, 1);
  __m256 result_hi_high = _mm256_insertf128_ps(_mm256_castps128_ps256(hi_8_11), hi_12_15, 1);

  // Assemble the High 256-bit result (Pixels 8-15)
  __m256 result256_high = _mm256_add_ps(result_lo_high, result_hi_high);

  // ---------------------------------------------------------------------------
  // Stream the two 256-bit results
  // ---------------------------------------------------------------------------
  _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result256_low);
  _mm256_stream_ps(reinterpret_cast<float*>(dst + x + 8), result256_high);
}

// filtersizealigned8: special: 0, 1..4, Generic : -1
template<int filtersize_hint>
static void internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  // filter_size is aligned to 8 (prerequisite), contrary that we have a special case for filter size <=4

  // We note that when template is used, filter_size is quasi-constexpr if filtersize_hint != -1.
  // When filtersize_hint == -1, then program->filter_size is aligned to 8 anyway, but not known at compile time.
  const int filter_size =
    filtersize_hint == 0 ? 8 : // though we'll optimize for 4 internally, coeff buffer is still allocated for 8
    (filtersize_hint >= 1) ? filtersize_hint * 8 : program->filter_size; // this latter is always aligned to 8 as well

  const float* src = (float*)src8;
  float* dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  constexpr int PIXELS_AT_A_TIME = 16;
  // Align safe zone to 16 pixels
  const int w_safe_mod16 = (program->safelimit_16_pixels.overread_possible ? program->safelimit_16_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  for (int y = 0; y < height; y++) {
    float* current_coeff_base = program->pixel_coefficient_float;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod16; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16<true, filtersize_hint>(src, x, current_coeff_base, filter_size, dst, program);
    }

    // Process up to the actual kernel size (unsafe zone)
    for (int x = w_safe_mod16; x < width; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16<false, filtersize_hint>(src, x, current_coeff_base, filter_size, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// Winner implementation: resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16;
// Other variants kept for reference, speed tested.
// Main test dimensions: pixels per cycle: 8,16,32 (pixX); sub-loops: 2,4,8 (subX); aligned filter sizes (ksX): 4, 8,16
// resizer_h_avx512_generic_float_pix8_sub8_ks16;
// resizer_h_avx512_generic_float_pix16_sub16_ks8;
// resizer_h_avx512_generic_float_pix32_sub8_ks8;
// resizer_h_avx2_generic_float_pix8_sub2_ks8; // like AVX2 version resizer_h_avx2_generic_float
// resizer_h_avx512_generic_float_pix8_sub2_ks8; // like AVX2 version with minor differences
// resizer_h_avx512_generic_float_pix8_sub4_ks8;
// resizer_h_avx512_generic_float_pix16_sub4_ks4;
// resizer_h_avx512_generic_float_pix16_sub4_ks8;

// Features of the chosen implementation:
// - 16 pixels per cycle
// - sub-loop 4 pixels per loop
// - filter size is aligned to 8 (prerequisite)
// - Special cases for aligned filter sizes 4,8,16,24,32
// - Depending on the filter size, calculates in chunks of 16, then 8, then 4 source pixels and coeffs at a time.
void resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = program->filter_size;
  // Expected alignment
  assert(program->filter_size_alignment >= 8);

  // Dispatcher template now supports filter_size aligned to 8 (8, 16, 24, 32) and a special case for <=4
  // Larger filter sizes will use the generic method (-1) which still benefit from 16-8-4 coeff processing blocks.
  if (filter_size == 1 * 8)
    if (program->filter_size_real <= 4)
      internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<0>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel); // Internally optimized for 4
    else
      internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel); // Internally optimized for 8
  else if (filter_size == 2 * 8) // Internally optimized for 16
    internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 3 * 8) // Internally optimized for 16+8
    internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 4 * 8) // Internally optimized for 2*16
    internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size, internally optimized for calculating coeffs in N*16 + 8 + 4 + 2 + 1 blocks
    internal_resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16<-1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// Horizontals uint8

// uint8_t h pretransposed_coeffs _base wrappers

void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks4_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  // template parameter false: no VNNI, base AVX512 madd
  resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks4_pretransposed_coeffs_internal<false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks8_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks8_pretransposed_coeffs_internal<false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
void resize_h_planar_uint8_avx512_permutex_vstripe_2s32_ks8_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint8_avx512_permutex_vstripe_2s32_ks8_pretransposed_coeffs_internal<false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks16_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks16_pretransposed_coeffs_internal<false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_2s32_ks64_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint8_avx512_permutex_vstripe_mpz_2s32_ks64_pretransposed_coeffs_internal<false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// uint16_t h pretransposed_coeffs _base wrappers

template<bool lessthan16bit>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4_pretransposed_coeffs_internal<lessthan16bit, false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
template<bool lessthan16bit>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8_pretransposed_coeffs_internal<lessthan16bit, false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
template<bool lessthan16bit>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8_pretransposed_coeffs_internal<lessthan16bit, false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
template<bool lessthan16bit>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16_pretransposed_coeffs_internal<lessthan16bit, false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}
template<bool lessthan16bit>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48_pretransposed_coeffs_base(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48_pretransposed_coeffs_internal<lessthan16bit, false>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// Explicit template instantiations for pretransposed uint16 _base variants
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4_pretransposed_coeffs_base<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4_pretransposed_coeffs_base<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8_pretransposed_coeffs_base<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8_pretransposed_coeffs_base<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8_pretransposed_coeffs_base<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8_pretransposed_coeffs_base<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16_pretransposed_coeffs_base<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16_pretransposed_coeffs_base<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48_pretransposed_coeffs_base<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48_pretransposed_coeffs_base<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// fixed_kernel_size == 0 marks: "not fixed".
void resize_prepare_coeffs_AVX512_H(ResamplingProgram* p, IScriptEnvironment* env, int iSamplesInTheGroup, int iGroupsCount, int fixed_kernel_size) {
  // note: filter_size_real was the max(kernel_sizes[])
  int filter_size_aligned = AlignNumber(p->filter_size_real, p->filter_size_alignment);
  // FIXME: really this needs to be dynamic based on SIMD used in resizer

  int target_size_aligned = AlignNumber(p->target_size, ALIGN_RESIZER_TARGET_SIZE);

  // align target_size to X units to allow safe, up to X pixels/cycle in H resizers.
  // also, this is the coeff table Y-size.
  // e.g. ALIGN_RESIZER_TARGET_SIZE = 64 allows to access coefficient table elements at
  // current_coeff + filter_size * 63, if we step current_coeff by 64 * filter_size
  p->target_size_alignment = ALIGN_RESIZER_TARGET_SIZE;

  // Common variables for both float and integer paths
  void* SIMD_coeff = nullptr;
  size_t element_size = 0;

  // allocate for a larger target_size area and nullify the coeffs.
  // Even between target_size and target_size_aligned.
  if (p->bits_per_pixel == 32) {
    element_size = sizeof(float);
    SIMD_coeff = env->Allocate(element_size * target_size_aligned * filter_size_aligned, 64, AVS_NORMAL_ALLOC);
    if (!SIMD_coeff) {
      env->Free(SIMD_coeff);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    std::fill_n((float*)SIMD_coeff, target_size_aligned * filter_size_aligned, 0.0f);
  }
  else {
    element_size = sizeof(short);
    SIMD_coeff = env->Allocate(element_size * target_size_aligned * filter_size_aligned, 64, AVS_NORMAL_ALLOC);
    if (!SIMD_coeff) {
      env->Free(SIMD_coeff);
      env->ThrowError("Could not reserve memory in a resampler.");
    }
    memset(SIMD_coeff, 0, element_size * target_size_aligned * filter_size_aligned);
  }

  int iSamplesAtATime = iSamplesInTheGroup * iGroupsCount;

  // Reset current_coeff for the start of the stripe (points to start of row's coeffs)
  const short* AVS_RESTRICT current_coeff = p->pixel_coefficient;

  const int filter_size_padded = p->filter_size; // aligned, practically the coeff table stride, always mod2 ?
  const int filter_size_real = p->filter_size_real;
  int filter_size_to_process = filter_size_real;
  if ((filter_size_real / 2 * 2) != filter_size_real) filter_size_to_process++; // add last zero coeffs to unpack hi/lo, they must present in zero-padded resampling program

  short* dst = (short*)SIMD_coeff;

  // Process coefficients - common code for both types
  for (int x = 0; x < p->target_size; x += iSamplesAtATime)
  {
    // clamp j so the last partial group doesn't overread pixel_coefficient
    const int avail = std::min(iSamplesAtATime, p->target_size - x);
    auto gc = [&](int j, int ki) -> short {
      return *(current_coeff + filter_size_padded * std::min(j, avail - 1) + ki);
    };
    // process by 2 rows because madd/dp can only make FMA from 2 unpacked uint16 pairs
    // Specific unrolled kernels (ks4/ks8/ks16) read a fixed number of taps per group of x.
    // They advance current_coeff_SIMD by an appropriate fixed amount, regardless of
    // filter_size_real. The coefficient table must be padded to that same fixed width
    // (fixed_kernel_size) or else discrapancy happens after the first x group.
    // Extra iterations access zero-padded taps and produce zero coefficient pairs — harmless.
    // For variable loop kernels like ks48/ks64 (signalled as fixed_kernel_size==0) advance
    // current_coeff_SIMD by filter_size_real*2 (pairs), so they must store exactly filter_size_to_process
    // pairs, contrary to the previous case, padding to a fixed width would cause coeff desynchronization
    // after the first x group.
    const int i_limit = (fixed_kernel_size > 0) ? fixed_kernel_size : filter_size_to_process;
    for (int i = 0; i < i_limit; i += 2)
    {
      if (iSamplesAtATime == 64) // 2 groups of 32 coeffs for columns 0..31 and 32..63
      {
        // use slow C-gathering, it is only once per filter init
        // first row
        const __m512i coef_rN_0_31 = _mm512_set_epi16(
          gc(31,i+0),gc(30,i+0),gc(29,i+0),gc(28,i+0),gc(27,i+0),gc(26,i+0),gc(25,i+0),gc(24,i+0),
          gc(23,i+0),gc(22,i+0),gc(21,i+0),gc(20,i+0),gc(19,i+0),gc(18,i+0),gc(17,i+0),gc(16,i+0),
          gc(15,i+0),gc(14,i+0),gc(13,i+0),gc(12,i+0),gc(11,i+0),gc(10,i+0),gc( 9,i+0),gc( 8,i+0),
          gc( 7,i+0),gc( 6,i+0),gc( 5,i+0),gc( 4,i+0),gc( 3,i+0),gc( 2,i+0),gc( 1,i+0),gc( 0,i+0)
        );
        const __m512i coef_rN_32_63 = _mm512_set_epi16(
          gc(63,i+0),gc(62,i+0),gc(61,i+0),gc(60,i+0),gc(59,i+0),gc(58,i+0),gc(57,i+0),gc(56,i+0),
          gc(55,i+0),gc(54,i+0),gc(53,i+0),gc(52,i+0),gc(51,i+0),gc(50,i+0),gc(49,i+0),gc(48,i+0),
          gc(47,i+0),gc(46,i+0),gc(45,i+0),gc(44,i+0),gc(43,i+0),gc(42,i+0),gc(41,i+0),gc(40,i+0),
          gc(39,i+0),gc(38,i+0),gc(37,i+0),gc(36,i+0),gc(35,i+0),gc(34,i+0),gc(33,i+0),gc(32,i+0)
        );

        // second row
        const __m512i coef_rNp1_0_31 = _mm512_set_epi16(
          gc(31,i+1),gc(30,i+1),gc(29,i+1),gc(28,i+1),gc(27,i+1),gc(26,i+1),gc(25,i+1),gc(24,i+1),
          gc(23,i+1),gc(22,i+1),gc(21,i+1),gc(20,i+1),gc(19,i+1),gc(18,i+1),gc(17,i+1),gc(16,i+1),
          gc(15,i+1),gc(14,i+1),gc(13,i+1),gc(12,i+1),gc(11,i+1),gc(10,i+1),gc( 9,i+1),gc( 8,i+1),
          gc( 7,i+1),gc( 6,i+1),gc( 5,i+1),gc( 4,i+1),gc( 3,i+1),gc( 2,i+1),gc( 1,i+1),gc( 0,i+1)
        );
        const __m512i coef_rNp1_32_63 = _mm512_set_epi16(
          gc(63,i+1),gc(62,i+1),gc(61,i+1),gc(60,i+1),gc(59,i+1),gc(58,i+1),gc(57,i+1),gc(56,i+1),
          gc(55,i+1),gc(54,i+1),gc(53,i+1),gc(52,i+1),gc(51,i+1),gc(50,i+1),gc(49,i+1),gc(48,i+1),
          gc(47,i+1),gc(46,i+1),gc(45,i+1),gc(44,i+1),gc(43,i+1),gc(42,i+1),gc(41,i+1),gc(40,i+1),
          gc(39,i+1),gc(38,i+1),gc(37,i+1),gc(36,i+1),gc(35,i+1),gc(34,i+1),gc(33,i+1),gc(32,i+1)
        );

        // unpack hi/lo
        const __m512i coef_rNrNp1_0_31lo = _mm512_unpacklo_epi16(coef_rN_0_31, coef_rNp1_0_31);
        const __m512i coef_rNrNp1_0_31hi = _mm512_unpackhi_epi16(coef_rN_0_31, coef_rNp1_0_31);

        const __m512i coef_rNrNp1_32_63lo = _mm512_unpacklo_epi16(coef_rN_32_63, coef_rNp1_32_63);
        const __m512i coef_rNrNp1_32_63hi = _mm512_unpackhi_epi16(coef_rN_32_63, coef_rNp1_32_63);

        // store (as pairs of rows for 0..31 and 32..63). May be better make array of 4-members structure ?
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst), coef_rNrNp1_0_31lo);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + 32), coef_rNrNp1_0_31hi); // in count of shorts
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + 64), coef_rNrNp1_32_63lo);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + 96), coef_rNrNp1_32_63hi);

        dst += 128;
      }
      else if (iSamplesAtATime == 32) // 1 group of 32 coeffs for columns 0..31 
      {
        // use slow C-gathering, it is only once per filter init
        // first row
        const __m512i coef_rN_0_31 = _mm512_set_epi16(
          gc(31,i+0),gc(30,i+0),gc(29,i+0),gc(28,i+0),gc(27,i+0),gc(26,i+0),gc(25,i+0),gc(24,i+0),
          gc(23,i+0),gc(22,i+0),gc(21,i+0),gc(20,i+0),gc(19,i+0),gc(18,i+0),gc(17,i+0),gc(16,i+0),
          gc(15,i+0),gc(14,i+0),gc(13,i+0),gc(12,i+0),gc(11,i+0),gc(10,i+0),gc( 9,i+0),gc( 8,i+0),
          gc( 7,i+0),gc( 6,i+0),gc( 5,i+0),gc( 4,i+0),gc( 3,i+0),gc( 2,i+0),gc( 1,i+0),gc( 0,i+0)
        );

        // second row
        const __m512i coef_rNp1_0_31 = _mm512_set_epi16(
          gc(31,i+1),gc(30,i+1),gc(29,i+1),gc(28,i+1),gc(27,i+1),gc(26,i+1),gc(25,i+1),gc(24,i+1),
          gc(23,i+1),gc(22,i+1),gc(21,i+1),gc(20,i+1),gc(19,i+1),gc(18,i+1),gc(17,i+1),gc(16,i+1),
          gc(15,i+1),gc(14,i+1),gc(13,i+1),gc(12,i+1),gc(11,i+1),gc(10,i+1),gc( 9,i+1),gc( 8,i+1),
          gc( 7,i+1),gc( 6,i+1),gc( 5,i+1),gc( 4,i+1),gc( 3,i+1),gc( 2,i+1),gc( 1,i+1),gc( 0,i+1)
        );

        // unpack hi/lo
        const __m512i coef_rNrNp1_0_31lo = _mm512_unpacklo_epi16(coef_rN_0_31, coef_rNp1_0_31);
        const __m512i coef_rNrNp1_0_31hi = _mm512_unpackhi_epi16(coef_rN_0_31, coef_rNp1_0_31);

        // store (as pairs of rows for 0..31 and 32..63). May be better make array of 4-members structure ?
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst), coef_rNrNp1_0_31lo);
        _mm512_store_si512(reinterpret_cast<__m512i*>(dst + 32), coef_rNrNp1_0_31hi); // in count of shorts

        dst += 64;
      }
      else
        env->ThrowError("prepare for AVX512 permute-based H-resize: unsupported");

    }
    current_coeff += filter_size_padded * iSamplesAtATime;
  }

  // assign to ResamplingProgram
  p->pixel_coefficient_AVX512_H = (short*)SIMD_coeff;
}

void resize_prepare_coeffs_AVX512_float_H(ResamplingProgram* p, IScriptEnvironment* env) {
  // Transpose float coefficients from [px][tap] row-major layout into
  // [x_group][tap][px] for 16-pixel-at-a-time permutex ks16 H-resizers.
  // Enables 16x _mm512_load_ps instead of 16x _mm512_i32gather_ps per x-group.
  assert(p->bits_per_pixel == 32);
  constexpr int PIXELS_AT_A_TIME = 16;
  constexpr int TAPS = 16;

  const int target_size_aligned = AlignNumber(p->target_size, ALIGN_RESIZER_TARGET_SIZE);
  const int filter_size_padded = p->filter_size;

  const size_t buf_count = (size_t)target_size_aligned * TAPS;
  float* buf = (float*)env->Allocate(sizeof(float) * buf_count, 64, AVS_NORMAL_ALLOC);
  if (!buf)
    env->ThrowError("resize_prepare_coeffs_AVX512_float_H: Could not reserve memory.");
  std::fill_n(buf, buf_count, 0.0f);

  const float* src = p->pixel_coefficient_float;
  float* dst = buf;

  for (int x = 0; x < p->target_size; x += PIXELS_AT_A_TIME) {
    const int avail = std::min(PIXELS_AT_A_TIME, p->target_size - x);
    for (int k = 0; k < TAPS; k++) {
      for (int j = 0; j < PIXELS_AT_A_TIME; j++) {
        const int pj = std::min(j, avail - 1);
        dst[k * PIXELS_AT_A_TIME + j] = (k < filter_size_padded) ? src[filter_size_padded * pj + k] : 0.0f;
      }
    }
    dst += TAPS * PIXELS_AT_A_TIME;
    src += filter_size_padded * PIXELS_AT_A_TIME;
  }

  p->pixel_coefficient_AVX512_float_H = buf;
}


