// AviSynth+.  Copyright 2026- AviSynth+ Project
// https://avs-plus.net
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

/*

This is a common source cpp include file (not header) for multi-arch AVX512 functions.
Functions here are static, they will be compiled into each translation unit including this file.

*/

// Original function needed avx512vbmi feature flag, but we want to support also base AVX512 without VBMI.
// We use _mm512_permutex2var_epi8_SIMUL<UseVBMI> and _mm512_maskz_permutex2var_epi8_SIMUL<UseVBMI>
// Thus both Base AVX512 and ICL level arch is supported.
// We are using two separated source modules and include this hpp file templated
// with UseVBMI/UseVNNI

// Notes:
// As of January 2026, Visual Studio 2026 ships with clang-cl (LLVM 20.1.8).
// - This version typically avoids using VNNI vpdpwssd instructions, opting instead for separate madd and add operations.
// - Masked permute operations are not optimized: for example, instead of using masked permutex2var_epi8,
//   it performs a basic permutex2var_epi8 followed by an "and" with a pre-loaded zmm mask.
// These behaviors result in slower code compared to MSVC builds, which utilize these instructions more efficiently.
// These optimization issues are resolved in LLVM 21 (e.g., Intel C++ Compiler 2025.3).

// helper function for simulating _mm512_permutex2var_epi8 when VBMI is not available
// The MSB bit (128) zeroing effect is _not_ considered here, the indices must be all positive and within 0-127 range.
// Helper for _mm512_permutex2var_epi8_SIMUL: gather 32 bytes from [a,b] using precomputed word_idx and shift_amt.
// Extracted from lambda to ensure MSVC inlines it (lambdas are not reliably inlined by MSVC).
// Accepts precomputed word_idx (target_idx>>1) and shift_amt ((target_idx<<3)&8) so callers outside
// the y-loop can hoist these invariants, avoiding recomputation every row.
// Returns 32 16-bit words with gathered bytes in low 8 bits (high 8 bits cleared).
// Returning __m512i instead of __m256i lets callers feed unpacklo/hi_epi16 directly,
// avoiding the cvtepu8_epi16 expansion round-trip that MSVC emits as vpmovwb+vpmovzxbw.
AVS_FORCEINLINE static __m512i _permutex2var_epi8_sim_get32(__m512i word_idx, __m512i shift_amt, __m512i a, __m512i b)
{
  __m512i words = _mm512_permutex2var_epi16(a, word_idx, b);
  // vpsrlvw: avoids k-register RAW dependency; MSVC compiles test_epi16_mask+mask_srli as
  // vptestmw->k1->vpsrlw{k1}, creating a read-after-write through the mask unit (higher latency)
  words = _mm512_srlv_epi16(words, shift_amt);
  // clear high byte so the caller can use unpacklo/hi_epi16 without cvtepu8_epi16
  return _mm512_and_si512(words, _mm512_set1_epi16(0x00FF));
}

template<bool UseVBMI>
AVS_FORCEINLINE static __m512i _mm512_permutex2var_epi8_SIMUL(__m512i a, __m512i idx, __m512i b) {
  if constexpr (UseVBMI) {
    return _mm512_permutex2var_epi8(a, idx, b);
  }
  else {
    const __m512i c_8 = _mm512_set1_epi16(8);

    __m512i idx_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(idx));
    __m512i idx_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(idx, 1));

    __m512i wi_lo = _mm512_srli_epi16(idx_lo, 1);
    __m512i sa_lo = _mm512_and_si512(_mm512_slli_epi16(idx_lo, 3), c_8);
    __m512i wi_hi = _mm512_srli_epi16(idx_hi, 1);
    __m512i sa_hi = _mm512_and_si512(_mm512_slli_epi16(idx_hi, 3), c_8);

    __m256i res_0_31  = _mm512_cvtepi16_epi8(_permutex2var_epi8_sim_get32(wi_lo, sa_lo, a, b));
    __m256i res_32_63 = _mm512_cvtepi16_epi8(_permutex2var_epi8_sim_get32(wi_hi, sa_hi, a, b));

    return _mm512_inserti64x4(_mm512_castsi256_si512(res_0_31), res_32_63, 1);
  }
}

template<bool UseVBMI>
AVS_FORCEINLINE static __m512i _mm512_maskz_permutex2var_epi8_SIMUL(
  __mmask64 k,
  __m512i a,
  __m512i idx,
  __m512i b)
{
  if constexpr (UseVBMI) {
    return _mm512_maskz_permutex2var_epi8(k, a, idx, b);
  }
  else {
    // 1. Run the base simulation to get the permuted bytes
    // Note: Using your existing logic to get the full 512-bit permuted result
    __m512i res = _mm512_permutex2var_epi8_SIMUL<false>(a, idx, b);

    // 2. Apply the zero-mask using AVX-512BW
    // _mm512_maskz_mov_epi8(k, src, src) returns 'src' where k[i]==1, and 0 where k[i]==0
    return _mm512_maskz_mov_epi8(k, res);
  }
}

// H-Float-Resampler: 16 pixels, filter size 4, transpose 4x (4x_m128) to 4x_m512
// Transposes a 4x4 matrix of 4-float vectors (16x16 float matrix effectively).
// Input/Output: Four 512-bit vectors (16 floats each) passed by reference.
AVS_FORCEINLINE static void _MM_TRANSPOSE16_LANE4_PS(__m512& row0, __m512& row1, __m512& row2, __m512& row3) {
  // Stage 1: Interleave 32-bit (float) elements within 128-bit chunks (lanes)
  // t0 = (r0_lo, r1_lo) | t1 = (r0_hi, r1_hi)
  // t2 = (r2_lo, r3_lo) | t3 = (r2_hi, r3_hi)
  __m512 t0 = _mm512_unpacklo_ps(row0, row1);
  __m512 t1 = _mm512_unpackhi_ps(row0, row1);
  __m512 t2 = _mm512_unpacklo_ps(row2, row3);
  __m512 t3 = _mm512_unpackhi_ps(row2, row3);

  // Stage 2: Shuffle 128-bit chunks (lanes) to complete the transpose
  // We use _mm512_shuffle_ps which shuffles 64-bit blocks across the 512-bit register.
  // _MM_SHUFFLE(w, z, y, x) applies to the 64-bit pairs (4 floats) within each 128-bit lane.
  // Result: row0 = columns 0, 1, 2, 3
  row0 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
  // Result: row1 = columns 4, 5, 6, 7
  row1 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
  // Result: row2 = columns 8, 9, 10, 11
  row2 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
  // Result: row3 = columns 12, 13, 14, 15
  row3 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
}

// H-float-resampler: 16 pixels, filter size 8, transpose 8x (2x_m256) to 8x_m512
// Transposes an 8x8 matrix of 2-float vectors (16x16 float matrix).
// Input/Output: Eight 512-bit vectors (16 floats each) passed by reference.
AVS_FORCEINLINE static void _MM_TRANSPOSE8x16_PS(
  __m512& r0, __m512& r1, __m512& r2, __m512& r3,
  __m512& r4, __m512& r5, __m512& r6, __m512& r7)
{
  // --- Stage 1: Unpack 32-bit (Pairs of rows) ---
  __m512 t0 = _mm512_unpacklo_ps(r0, r1);
  __m512 t1 = _mm512_unpackhi_ps(r0, r1);
  __m512 t2 = _mm512_unpacklo_ps(r2, r3);
  __m512 t3 = _mm512_unpackhi_ps(r2, r3);
  __m512 t4 = _mm512_unpacklo_ps(r4, r5);
  __m512 t5 = _mm512_unpackhi_ps(r4, r5);
  __m512 t6 = _mm512_unpacklo_ps(r6, r7);
  __m512 t7 = _mm512_unpackhi_ps(r6, r7);

  // --- Stage 2: Unpack 64-bit (Quads of rows) ---
  // Uses _mm512_unpacklo/hi_pd for 64-bit (double) to interleave pairs of __m512 floats
  __m512 u0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
  __m512 u1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t2)));
  __m512 u2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
  __m512 u3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1), _mm512_castps_pd(t3)));
  __m512 u4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
  __m512 u5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t4), _mm512_castps_pd(t6)));
  __m512 u6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));
  __m512 u7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t5), _mm512_castps_pd(t7)));

  // --- Stage 3: Shuffle 128-bit lanes (Octets of rows) ---
  // _mm512_shuffle_f32x4 shuffles the 128-bit (f32x4) sub-vectors within and between two __m512 vectors.
  // 0x88 = (10001000)_2: selects lane 0 from first input and lane 0 from second input for lo/hi 256 bits.
  // 0xDD = (11011101)_2: selects lane 3 from first input and lane 3 from second input for lo/hi 256 bits.
  __m512 v0 = _mm512_shuffle_f32x4(u0, u4, 0x88); // Col 0, 4 (interleaved)
  __m512 v1 = _mm512_shuffle_f32x4(u0, u4, 0xDD); // Col 1, 5 (interleaved)
  __m512 v2 = _mm512_shuffle_f32x4(u1, u5, 0x88); // Col 2, 6 (interleaved)
  __m512 v3 = _mm512_shuffle_f32x4(u1, u5, 0xDD); // Col 3, 7 (interleaved)
  __m512 v4 = _mm512_shuffle_f32x4(u2, u6, 0x88); // Col 8, 12 (interleaved)
  __m512 v5 = _mm512_shuffle_f32x4(u2, u6, 0xDD); // Col 9, 13 (interleaved)
  __m512 v6 = _mm512_shuffle_f32x4(u3, u7, 0x88); // Col 10, 14 (interleaved)
  __m512 v7 = _mm512_shuffle_f32x4(u3, u7, 0xDD); // Col 11, 15 (interleaved)

  // --- Stage 4: Permute to Linearize Indices ---
  // Corrects the order of the 128-bit lanes to linearize the columns.
  // The columns are currently: (0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
  // The required order is: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
  __m512i idx = _mm512_setr_epi32(
    0, 4, 1, 5,    /* Lane 0: Rows 0, 1, 2, 3 */
    2, 6, 3, 7,    /* Lane 1: Rows 4, 5, 6, 7 */
    8, 12, 9, 13, /* Lane 2: Rows 8, 9, 10, 11 */
    10, 14, 11, 15 /* Lane 3: Rows 12, 13, 14, 15 */
  );

  // --- Final Assignment with Correct Mapping ---
  // Maps the permuted vector components back to the original row variables (now columns).
  r0 = _mm512_permutexvar_ps(idx, v0); // Col 0
  r1 = _mm512_permutexvar_ps(idx, v2); // Col 1
  r2 = _mm512_permutexvar_ps(idx, v4); // Col 2
  r3 = _mm512_permutexvar_ps(idx, v6); // Col 3
  r4 = _mm512_permutexvar_ps(idx, v1); // Col 4
  r5 = _mm512_permutexvar_ps(idx, v3); // Col 5
  r6 = _mm512_permutexvar_ps(idx, v5); // Col 6
  r7 = _mm512_permutexvar_ps(idx, v7); // Col 7
}

// Loads two 256-bit float vectors from registers (__m256) into a single 512-bit register.
// Equivalent to _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1)
AVS_FORCEINLINE static __m512 _mm512_insert_2_m256(__m256 lo, __m256 hi) {
  return _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
}

// Loads four 128-bit float vectors (unaligned) into a single 512-bit register.
AVS_FORCEINLINE static __m512 _mm512_loadu_4_m128(
  /* __m128 const* */ const float* addr1,
  /* __m128 const* */ const float* addr2,
  /* __m128 const* */ const float* addr3,
  /* __m128 const* */ const float* addr4)
{
  // The cast is needed for the first insertion to make the target a 512-bit register
  __m512 v = _mm512_castps128_ps512(_mm_loadu_ps(addr1));
  v = _mm512_insertf32x4(v, _mm_loadu_ps(addr2), 1);
  v = _mm512_insertf32x4(v, _mm_loadu_ps(addr3), 2);
  v = _mm512_insertf32x4(v, _mm_loadu_ps(addr4), 3);
  return v;
}

// Loads four 128-bit float vectors (aligned) into a single 512-bit register.
AVS_FORCEINLINE static __m512 _mm512_load_4_m128(
  /* __m128 const* */ const float* addr1,
  /* __m128 const* */ const float* addr2,
  /* __m128 const* */ const float* addr3,
  /* __m128 const* */ const float* addr4)
{
  // The cast is needed for the first insertion to make the target a 512-bit register
  __m512 v = _mm512_castps128_ps512(_mm_load_ps(addr1));
  v = _mm512_insertf32x4(v, _mm_load_ps(addr2), 1);
  v = _mm512_insertf32x4(v, _mm_load_ps(addr3), 2);
  v = _mm512_insertf32x4(v, _mm_load_ps(addr4), 3);
  return v;
}

// Loads two 256 - bit unaligned integer vectors from registers(__m256i) into a single 512i register.
AVS_FORCEINLINE static __m512i _mm512i_loadu_2_m256i(
  const __m256i* addr1,
  const __m256i* addr2)
{
  return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(addr1)), _mm256_loadu_si256(addr2), 1);
}

// Loads two 256 - bit aligned integer vectors from registers(__m256) into a single 512 - bit register.
AVS_FORCEINLINE static __m512i _mm512i_load_2_m256i(
  const __m256i* addr1,
  const __m256i* addr2)
{
  return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256(addr1)), _mm256_load_si256(addr2), 1);
}

// Integers
// Loads four 128-bit integer vectors (unaligned) into a single 512-bit integer register.
AVS_FORCEINLINE static __m512i _mm512i_loadu_4_m128i(
  const __m128i* addr1,
  const __m128i* addr2,
  const __m128i* addr3,
  const __m128i* addr4)
{
  // The cast is needed for the first insertion to make the target a 512-bit register
  __m512i v = _mm512_castsi128_si512(_mm_loadu_si128(addr1));
  v = _mm512_inserti32x4(v, _mm_loadu_si128(addr2), 1);
  v = _mm512_inserti32x4(v, _mm_loadu_si128(addr3), 2);
  v = _mm512_inserti32x4(v, _mm_loadu_si128(addr4), 3);
  return v;
}

// filter size up to 4
// 64 target uint8_t pixels at a time
// 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_permutex2var_epi8 (uint8_t)

// filter size up to 8
// 64 target uint8_t pixels at a time
// 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_permutex2var_epi8(uint8_t)

// filter size up to 8
// 64 target uint8_t pixels at a time in 2 groups of 32 to support longer source loading to each group to support lower downsample ratios
// support /2 downsample ratios for resizers with no-resize kernel size of 4 (or support of 2 ?) (Bicubic, Bilinear, and others, also SinPowResize (?))
// 2 groups of 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_permutex2var_epi8 (uint8_t)

// filter size up to 16
// 32 target uint8_t pixels at a time
// 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_permutex2var_epi8 (uint8_t)
// expect to support all upsampling ratios up to filter support of 8 (or 7..6 ?) and some downsampling ratios with filter support up to 3 (with downsample ratios from 0.5 or a bit lower)

// uint8_t "mp" versions

// filter size up to 4
// 64 target uint8_t pixels at a time
// 127-byte source loads (127 uint8_t pixels)
// maximum permute index is 127 for _mm512_maskz_permutex2var_epi8 (uint8_t) 
// more premutex version to create 8->16bit converted and low-hi unpacked registers in single permutex instruction

// filter size up to 4
// 64 target uint8_t pixels at a time
// 127-byte source loads (127 uint8_t pixels)
// maximum permute index is 127 for _mm512_maskz_permutex2var_epi8 (uint8_t) 
// more premutex version to create 8->16bit converted and low-hi unpacked registers in single permutex instruction
template<bool UseVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks4_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  constexpr int PIXELS_AT_A_TIME = 64;

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // we load 2x64 source bytes at a time, so ensure safe overread if needed.
  // Our main loop processes calculates for 64 target pixels at a time.
  // Inside that, we load 128 source bytes (2x64) to be able to permutex from that.
  // This we have to check at each mod-PIXELS_AT_A_TIME boundary, the allowance of 128-byte source load.
  const int width_safe_mod = (program->safelimit_128_pixels_each64th_target.overread_possible ? program->safelimit_128_pixels_each64th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 15' when processing 16 H pixels at a time
  // 'filter_size * 63' when processing 64 H pixels at a time
  assert(program->target_size_alignment >= 64); // Adjusted for 64 pixels (is it enough for uint8 ?)

  assert(FRAME_ALIGN >= 64); // Good for 64x8 bit pixels

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  // We load 4x16bit coeffs at a time
  assert(program->filter_size_alignment >= 4);

  const int max_scanlines = program->max_scanlines;

  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));

  // Vertical stripe loop for L2 cache optimization
  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe (points to start of row's coeffs)
    // const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;
    // Pre-transposed coefficients for AVX512, stored in the program structure, ready for direct use in the main loop without needing to transpose on the fly.
    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    // Lambda to handle both safe (fast) and unsafe (masked/partial) loading paths
    auto do_h_integer_core = [&](auto partial_load) {

      // prepare coefs in transposed V-form
      // TODO: make storage in transposed form, 64 x uint16 transposition looks too slow
      __m512i one_epi16 = _mm512_set1_epi16(1);

      // load coeffs from prepared
      const __m512i coef_r0r1_0_31lo = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi = _mm512_load_si512(current_coeff_SIMD + 1); // in count of __m512i

      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);

      const __m512i coef_r2r3_0_31lo = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi = _mm512_load_si512(current_coeff_SIMD + 5);

      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      __m512i perm_0_0_15 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x])); // 16 offsets
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16])); //  16 offsets
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32])); //  16 offsets
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48])); //  16 offsets

      int iStart = program->pixel_offset[x];
      __m512i m512i_Start = _mm512_set1_epi32(iStart);

      perm_0_0_15 = _mm512_sub_epi32(perm_0_0_15, m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start);

      __m256i m256i_perm_0_0_15 = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15), m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      // Taps are contiguous (0, 1, 2, 3), so we increment perm indexes by 1.
      __m512i perm_1_0_31 = _mm512_add_epi16(perm_0_0_31, one_epi16);
      __m512i perm_2_0_31 = _mm512_add_epi16(perm_1_0_31, one_epi16);
      __m512i perm_3_0_31 = _mm512_add_epi16(perm_2_0_31, one_epi16);

      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);
      __m512i perm_2_32_63 = _mm512_add_epi16(perm_1_32_63, one_epi16);
      __m512i perm_3_32_63 = _mm512_add_epi16(perm_2_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo = _mm512_unpacklo_epi16(perm_0_0_31, perm_1_0_31);
      const __m512i perm_r0r1_0_31hi = _mm512_unpackhi_epi16(perm_0_0_31, perm_1_0_31);

      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i perm_r2r3_0_31lo = _mm512_unpacklo_epi16(perm_2_0_31, perm_3_0_31);
      const __m512i perm_r2r3_0_31hi = _mm512_unpackhi_epi16(perm_2_0_31, perm_3_0_31);

      const __m512i perm_r2r3_32_63lo = _mm512_unpacklo_epi16(perm_2_32_63, perm_3_32_63);
      const __m512i perm_r2r3_32_63hi = _mm512_unpackhi_epi16(perm_2_32_63, perm_3_32_63);

      __m512i wi_r0r1_0_31lo={},sa_r0r1_0_31lo={},wi_r0r1_0_31hi={},sa_r0r1_0_31hi={};
      __m512i wi_r0r1_32_63lo={},sa_r0r1_32_63lo={},wi_r0r1_32_63hi={},sa_r0r1_32_63hi={};
      __m512i wi_r2r3_0_31lo={},sa_r2r3_0_31lo={},wi_r2r3_0_31hi={},sa_r2r3_0_31hi={};
      __m512i wi_r2r3_32_63lo={},sa_r2r3_32_63lo={},wi_r2r3_32_63hi={},sa_r2r3_32_63hi={};
      if constexpr (!UseVNNI) {
        const __m512i c_8w = _mm512_set1_epi16(8);
        auto make_wi_sa = [&](__m512i p, __m512i& wi, __m512i& sa) {
          wi = _mm512_srli_epi16(p, 1);
          sa = _mm512_and_si512(_mm512_slli_epi16(p, 3), c_8w);
        };
        make_wi_sa(perm_r0r1_0_31lo,  wi_r0r1_0_31lo,  sa_r0r1_0_31lo);
        make_wi_sa(perm_r0r1_0_31hi,  wi_r0r1_0_31hi,  sa_r0r1_0_31hi);
        make_wi_sa(perm_r0r1_32_63lo, wi_r0r1_32_63lo, sa_r0r1_32_63lo);
        make_wi_sa(perm_r0r1_32_63hi, wi_r0r1_32_63hi, sa_r0r1_32_63hi);
        make_wi_sa(perm_r2r3_0_31lo,  wi_r2r3_0_31lo,  sa_r2r3_0_31lo);
        make_wi_sa(perm_r2r3_0_31hi,  wi_r2r3_0_31hi,  sa_r2r3_0_31hi);
        make_wi_sa(perm_r2r3_32_63lo, wi_r2r3_32_63lo, sa_r2r3_32_63lo);
        make_wi_sa(perm_r2r3_32_63hi, wi_r2r3_32_63hi, sa_r2r3_32_63hi);
      }

      uint8_t* AVS_RESTRICT dst_ptr = dst8 + x + y_from * dst_pitch;
      const uint8_t* src_ptr = src8 + iStart + y_from * src_pitch; // all permute offsets relative to this start offset

      // Calculate remaining pixels for bounds checking in partial_load mode. 1..128 remaining pixels possible.
      const int remaining = program->source_size - iStart;
      __mmask64 k1 = _bzhi_u64(~0ULL, remaining); // _bzhi_u64 creates a mask with the lower N bits set. If N >= 64, it returns all ones (~0ULL).
      __mmask64 k2 = _bzhi_u64(~0ULL, std::max(0, remaining - 64));

      // mask is used to zero out every odd byte, so that the result of the permute is a
      // vector of zero-extended 8-bit values in 16-bit lanes, preparing 8-bit data for 16-bit FMA operations in AVX512.
      const __mmask64 k_zh8 = 0x5555555555555555ULL;

      for (int y = y_from; y < y_to; y++)
      {
        // 8 coeffs + 8 permute_idx + 2 src + 4 temporal + 1 rounder ~= 23 regs (permute2var overwrite first source - really may be more needed)
        __m512i data_src, data_src2;

        if constexpr (partial_load) {
          // Safe masked loads for the image edge
          data_src = _mm512_maskz_loadu_epi8(k1, src_ptr);
          data_src2 = _mm512_maskz_loadu_epi8(k2, src_ptr + 64);
        }
        else {
          // Fast unaligned loads for the safe zone
          data_src = _mm512_loadu_si512(src_ptr);
          data_src2 = _mm512_loadu_si512(src_ptr + 64);
        }

        __m512i src_r0r1_0_31lo, src_r0r1_0_31hi, src_r0r1_32_63lo, src_r0r1_32_63hi;
        __m512i src_r2r3_0_31lo, src_r2r3_0_31hi, src_r2r3_32_63lo, src_r2r3_32_63hi;
        if constexpr (UseVNNI) {
          src_r0r1_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r0r1_0_31lo,  data_src2);
          src_r0r1_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r0r1_0_31hi,  data_src2);
          src_r0r1_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r0r1_32_63lo, data_src2);
          src_r0r1_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r0r1_32_63hi, data_src2);
          src_r2r3_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r2r3_0_31lo,  data_src2);
          src_r2r3_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r2r3_0_31hi,  data_src2);
          src_r2r3_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r2r3_32_63lo, data_src2);
          src_r2r3_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_r2r3_32_63hi, data_src2);
        } else {
          src_r0r1_0_31lo  = _permutex2var_epi8_sim_get32(wi_r0r1_0_31lo,  sa_r0r1_0_31lo,  data_src, data_src2);
          src_r0r1_0_31hi  = _permutex2var_epi8_sim_get32(wi_r0r1_0_31hi,  sa_r0r1_0_31hi,  data_src, data_src2);
          src_r0r1_32_63lo = _permutex2var_epi8_sim_get32(wi_r0r1_32_63lo, sa_r0r1_32_63lo, data_src, data_src2);
          src_r0r1_32_63hi = _permutex2var_epi8_sim_get32(wi_r0r1_32_63hi, sa_r0r1_32_63hi, data_src, data_src2);
          src_r2r3_0_31lo  = _permutex2var_epi8_sim_get32(wi_r2r3_0_31lo,  sa_r2r3_0_31lo,  data_src, data_src2);
          src_r2r3_0_31hi  = _permutex2var_epi8_sim_get32(wi_r2r3_0_31hi,  sa_r2r3_0_31hi,  data_src, data_src2);
          src_r2r3_32_63lo = _permutex2var_epi8_sim_get32(wi_r2r3_32_63lo, sa_r2r3_32_63lo, data_src, data_src2);
          src_r2r3_32_63hi = _permutex2var_epi8_sim_get32(wi_r2r3_32_63hi, sa_r2r3_32_63hi, data_src, data_src2);
        }

        __m512i result_0_31lo, result_0_31hi;
        __m512i result_32_63lo, result_32_63hi;

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo, coef_r0r1_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r2r3_0_31lo, coef_r2r3_0_31lo);

          result_0_31hi = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi, coef_r0r1_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r2r3_0_31hi, coef_r2r3_0_31hi);

          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);

          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          // making FMA in 32bits accs as in AVX256 V-resize
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo, coef_r0r1_0_31lo), _mm512_madd_epi16(src_r2r3_0_31lo, coef_r2r3_0_31lo));
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi, coef_r0r1_0_31hi), _mm512_madd_epi16(src_r2r3_0_31hi, coef_r2r3_0_31hi));

          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));

          // rounding
          result_0_31lo = _mm512_add_epi32(result_0_31lo, rounder);
          result_0_31hi = _mm512_add_epi32(result_0_31hi, rounder);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, rounder);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, rounder);
        }

        // scaling down
        result_0_31lo = _mm512_srai_epi32(result_0_31lo, FPScale8bits);
        result_0_31hi = _mm512_srai_epi32(result_0_31hi, FPScale8bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale8bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale8bits);

        __m512i result_0_31_int16 = _mm512_packus_epi32(result_0_31lo, result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        // cast is enough, no need to _mm512_zextsi256_si512
        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr),
          _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtusepi16_epi8(result_0_31_int16)), _mm512_cvtusepi16_epi8(result_32_63_int16), 1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      // current_coeff += filter_size * PIXELS_AT_A_TIME;
      current_coeff_SIMD += 8; // in number of __m512i

      };

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe masked loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

// filter size up to 8
// 64 target uint8_t pixels at a time
// 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_maskz_permutex2var_epi8 (uint8_t)
// support VNNI and madd FMA

// filter size up to 16
// 32 target uint8_t pixels at a time
// 128-byte source loads (128 uint8_t pixels)
// maximum permute index is 128 for _mm512_maskz_permutex2var_epi8 (uint8_t)

// uint16_t

// filter size up to 4
// 32 target uint16_t pixels at a time
// 128-byte source loads (64 uint16_t pixels)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// making lo-hi unpacking with single permutex2var operation and optional VNNI FMA

// filter size up to 8
// 64 target uint16_t pixels at a time in 4 groups of 16
// 128-byte source loads (64 uint16_t pixels), 2 groups of 128 byte source loads for each 16 output samples,
// expect to support /2 downsize (used in ConvertTo42x with default bicubic too)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// support VNNI and madd FMA
// checker: (16/*iSamplesInTheGroup*/, 64/*permutex_index_diff_limit*/, 8/*kernel_size*/))

template <bool AdvancePerm, bool UseVNNI>
AVS_FORCEINLINE static void process_row_pair(
  // 1. Accumulators (Modified across calls)
  __m512i& res_lo, __m512i& res_hi,
  __m512i& res_32_63lo, __m512i& res_32_63hi,
  // 2. Permutation Vectors (Modified across calls)
  __m512i& p_lo, __m512i& p_hi,
  __m512i& p_32_lo, __m512i& p_32_hi,
  // 3. Coefficients (Read-only)
  const __m512i& c_lo, const __m512i& c_hi,
  const __m512i& c_32_63lo, const __m512i& c_32_63hi,
  // 4. Source Data and Constants
  const __m512i& d0_15, const __m512i& d16_31,
  const __m512i& d32_47, const __m512i& d48_63,
  const __m512i& d2_0_15, const __m512i& d2_16_31,
  const __m512i& d2_32_47, const __m512i& d2_48_63,
  const __mmask32 &k_hi, const __m512i &v_two)
{
  // Generate source vectors via permutations
  auto src_lo = _mm512_mask_blend_epi16(k_hi,
    _mm512_permutex2var_epi16(d0_15, p_lo, d2_0_15),
    _mm512_permutex2var_epi16(d16_31, p_lo, d2_16_31));

  auto src_hi = _mm512_mask_blend_epi16(k_hi,
    _mm512_permutex2var_epi16(d0_15, p_hi, d2_0_15),
    _mm512_permutex2var_epi16(d16_31, p_hi, d2_16_31));

  auto src_32_63lo = _mm512_mask_blend_epi16(k_hi,
    _mm512_permutex2var_epi16(d32_47, p_32_lo, d2_32_47),
    _mm512_permutex2var_epi16(d48_63, p_32_lo, d2_48_63));

  auto src_32_63hi = _mm512_mask_blend_epi16(k_hi,
    _mm512_permutex2var_epi16(d32_47, p_32_hi, d2_32_47),
    _mm512_permutex2var_epi16(d48_63, p_32_hi, d2_48_63));

  // Accumulate results
  if constexpr (UseVNNI) {
    res_lo = _mm512_dpwssd_epi32(res_lo, src_lo, c_lo);
    res_hi = _mm512_dpwssd_epi32(res_hi, src_hi, c_hi);
    res_32_63lo = _mm512_dpwssd_epi32(res_32_63lo, src_32_63lo, c_32_63lo);
    res_32_63hi = _mm512_dpwssd_epi32(res_32_63hi, src_32_63hi, c_32_63hi);
  }
  else {
    res_lo = _mm512_add_epi32(res_lo, _mm512_madd_epi16(src_lo, c_lo));
    res_hi = _mm512_add_epi32(res_hi, _mm512_madd_epi16(src_hi, c_hi));
    res_32_63lo = _mm512_add_epi32(res_32_63lo, _mm512_madd_epi16(src_32_63lo, c_32_63lo));
    res_32_63hi = _mm512_add_epi32(res_32_63hi, _mm512_madd_epi16(src_32_63hi, c_32_63hi));
  }

  if constexpr (AdvancePerm) {
    p_lo = _mm512_add_epi16(p_lo, v_two);
    p_hi = _mm512_add_epi16(p_hi, v_two);
    p_32_lo = _mm512_add_epi16(p_32_lo, v_two);
    p_32_hi = _mm512_add_epi16(p_32_hi, v_two);
  }
}

// filter size up to 4
// 64 target uint16_t pixels at a time in 2 groups of 32
// 128-byte source loads (64 uint16_t pixels)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// making lo-hi unpacking with single permutex2var operation and optional VNNI FMA

// filter size up to 8
// 64 target uint16_t pixels at a time in 2 groups of 32
// 128-byte source loads (64 uint16_t pixels)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// support VNNI and madd FMA

// filter size up to 8
// 32 target uint16_t pixels at a time
// 128-byte source loads (64 uint16_t pixels), 2 groups of 128 byte source loads for each 16 output samples, expect to support /2 downsize (used in ConvertTo42x with default bicubic too)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// support VNNI and madd FMA
// checker function is program->resize_h_planar_gather_permutex_vstripe_check(16/*iSamplesInTheGroup*/, 64/*permutex_index_diff_limit*/, 8/*kernel_size*/))

// filter size up to 8
// 32 target uint16_t pixels at a time
// 128-byte source loads (64 uint16_t pixels)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// support VNNI and madd FMA

// filter size up to 16
// 32 target uint16_t pixels at a time
// 128-byte source loads (64 uint16_t pixels)
// maximum permute index is 64 for _mm512_permutex2var_epi16 (uint16_t)
// support VNNI and madd FMA

// filter size up to 8, pretransposed coefficients
// 64 target uint8_t pixels at a time
// 128-byte source loads (2x64 uint8_t pixels)
template<bool UseVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks8_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  constexpr int PIXELS_AT_A_TIME = 64;

  const int width_safe_mod = (program->safelimit_128_pixels_each64th_target.overread_possible ? program->safelimit_128_pixels_each64th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 8);
  assert(program->target_size_alignment >= 64);
  assert(FRAME_ALIGN >= 64);
  assert(program->filter_size_alignment >= 8);

  const int max_scanlines = program->max_scanlines;

  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r2r3_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r4r5_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r4r5_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r4r5_32_63lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r4r5_32_63hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r6r7_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r6r7_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r6r7_32_63lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r6r7_32_63hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart = program->pixel_offset[x];
      __m512i m512i_Start = _mm512_set1_epi32(iStart);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo  = _mm512_unpacklo_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_0_31hi  = _mm512_unpackhi_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      __m512i wi_r0r1_0_31lo={},sa_r0r1_0_31lo={},wi_r0r1_0_31hi={},sa_r0r1_0_31hi={};
      __m512i wi_r0r1_32_63lo={},sa_r0r1_32_63lo={},wi_r0r1_32_63hi={},sa_r0r1_32_63hi={};
      __m512i wi_r2r3_0_31lo={},sa_r2r3_0_31lo={},wi_r2r3_0_31hi={},sa_r2r3_0_31hi={};
      __m512i wi_r2r3_32_63lo={},sa_r2r3_32_63lo={},wi_r2r3_32_63hi={},sa_r2r3_32_63hi={};
      __m512i wi_r4r5_0_31lo={},sa_r4r5_0_31lo={},wi_r4r5_0_31hi={},sa_r4r5_0_31hi={};
      __m512i wi_r4r5_32_63lo={},sa_r4r5_32_63lo={},wi_r4r5_32_63hi={},sa_r4r5_32_63hi={};
      __m512i wi_r6r7_0_31lo={},sa_r6r7_0_31lo={},wi_r6r7_0_31hi={},sa_r6r7_0_31hi={};
      __m512i wi_r6r7_32_63lo={},sa_r6r7_32_63lo={},wi_r6r7_32_63hi={},sa_r6r7_32_63hi={};
      if constexpr (!UseVNNI) {
        const __m512i c_8w = _mm512_set1_epi16(8);
        auto make_wi_sa = [&](__m512i p, __m512i& wi, __m512i& sa) {
          wi = _mm512_srli_epi16(p, 1);
          sa = _mm512_and_si512(_mm512_slli_epi16(p, 3), c_8w);
        };
        __m512i p_lo = perm_r0r1_0_31lo, p_hi = perm_r0r1_0_31hi;
        __m512i p_lo2 = perm_r0r1_32_63lo, p_hi2 = perm_r0r1_32_63hi;
        make_wi_sa(p_lo,  wi_r0r1_0_31lo,  sa_r0r1_0_31lo);  make_wi_sa(p_hi,  wi_r0r1_0_31hi,  sa_r0r1_0_31hi);
        make_wi_sa(p_lo2, wi_r0r1_32_63lo, sa_r0r1_32_63lo); make_wi_sa(p_hi2, wi_r0r1_32_63hi, sa_r0r1_32_63hi);
        p_lo  = _mm512_add_epi16(p_lo,  two_epi16); p_hi  = _mm512_add_epi16(p_hi,  two_epi16);
        p_lo2 = _mm512_add_epi16(p_lo2, two_epi16); p_hi2 = _mm512_add_epi16(p_hi2, two_epi16);
        make_wi_sa(p_lo,  wi_r2r3_0_31lo,  sa_r2r3_0_31lo);  make_wi_sa(p_hi,  wi_r2r3_0_31hi,  sa_r2r3_0_31hi);
        make_wi_sa(p_lo2, wi_r2r3_32_63lo, sa_r2r3_32_63lo); make_wi_sa(p_hi2, wi_r2r3_32_63hi, sa_r2r3_32_63hi);
        p_lo  = _mm512_add_epi16(p_lo,  two_epi16); p_hi  = _mm512_add_epi16(p_hi,  two_epi16);
        p_lo2 = _mm512_add_epi16(p_lo2, two_epi16); p_hi2 = _mm512_add_epi16(p_hi2, two_epi16);
        make_wi_sa(p_lo,  wi_r4r5_0_31lo,  sa_r4r5_0_31lo);  make_wi_sa(p_hi,  wi_r4r5_0_31hi,  sa_r4r5_0_31hi);
        make_wi_sa(p_lo2, wi_r4r5_32_63lo, sa_r4r5_32_63lo); make_wi_sa(p_hi2, wi_r4r5_32_63hi, sa_r4r5_32_63hi);
        p_lo  = _mm512_add_epi16(p_lo,  two_epi16); p_hi  = _mm512_add_epi16(p_hi,  two_epi16);
        p_lo2 = _mm512_add_epi16(p_lo2, two_epi16); p_hi2 = _mm512_add_epi16(p_hi2, two_epi16);
        make_wi_sa(p_lo,  wi_r6r7_0_31lo,  sa_r6r7_0_31lo);  make_wi_sa(p_hi,  wi_r6r7_0_31hi,  sa_r6r7_0_31hi);
        make_wi_sa(p_lo2, wi_r6r7_32_63lo, sa_r6r7_32_63lo); make_wi_sa(p_hi2, wi_r6r7_32_63hi, sa_r6r7_32_63hi);
      }

      uint8_t* AVS_RESTRICT dst_ptr = dst8 + x + y_from * dst_pitch;
      const uint8_t* src_ptr = src8 + iStart + y_from * src_pitch;

      const int remaining = program->source_size - iStart;
      __mmask64 k1 = _bzhi_u64(~0ULL, remaining);
      const __mmask64 k2 = _bzhi_u64(~0ULL, std::max(0, remaining - 64));
      const __mmask64 k_zh8 = 0x5555555555555555ULL;

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src, data_src2;

        __m512i perm_rNrNp1_0_31lo_w  = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi_w  = perm_r0r1_0_31hi;
        __m512i perm_rNrNp1_32_63lo_w = perm_r0r1_32_63lo;
        __m512i perm_rNrNp1_32_63hi_w = perm_r0r1_32_63hi;

        if constexpr (partial_load) {
          data_src  = _mm512_maskz_loadu_epi8(k1, src_ptr);
          data_src2 = _mm512_maskz_loadu_epi8(k2, src_ptr + 64);
        }
        else {
          data_src  = _mm512_loadu_si512(src_ptr);
          data_src2 = _mm512_loadu_si512(src_ptr + 64);
        }

        // rows 0..1
        __m512i src_r0r1_0_31lo, src_r0r1_0_31hi, src_r0r1_32_63lo, src_r0r1_32_63hi;
        if constexpr (UseVNNI) {
          src_r0r1_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w,  data_src2);
          src_r0r1_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w,  data_src2);
          src_r0r1_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63lo_w, data_src2);
          src_r0r1_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63hi_w, data_src2);
        } else {
          src_r0r1_0_31lo  = _permutex2var_epi8_sim_get32(wi_r0r1_0_31lo,  sa_r0r1_0_31lo,  data_src, data_src2);
          src_r0r1_0_31hi  = _permutex2var_epi8_sim_get32(wi_r0r1_0_31hi,  sa_r0r1_0_31hi,  data_src, data_src2);
          src_r0r1_32_63lo = _permutex2var_epi8_sim_get32(wi_r0r1_32_63lo, sa_r0r1_32_63lo, data_src, data_src2);
          src_r0r1_32_63hi = _permutex2var_epi8_sim_get32(wi_r0r1_32_63hi, sa_r0r1_32_63hi, data_src, data_src2);
        }

        // for r2r3
        perm_rNrNp1_0_31lo_w  = _mm512_add_epi16(perm_rNrNp1_0_31lo_w,  two_epi16);
        perm_rNrNp1_0_31hi_w  = _mm512_add_epi16(perm_rNrNp1_0_31hi_w,  two_epi16);
        perm_rNrNp1_32_63lo_w = _mm512_add_epi16(perm_rNrNp1_32_63lo_w, two_epi16);
        perm_rNrNp1_32_63hi_w = _mm512_add_epi16(perm_rNrNp1_32_63hi_w, two_epi16);

        __m512i src_r2r3_0_31lo, src_r2r3_0_31hi, src_r2r3_32_63lo, src_r2r3_32_63hi;
        if constexpr (UseVNNI) {
          src_r2r3_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w,  data_src2);
          src_r2r3_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w,  data_src2);
          src_r2r3_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63lo_w, data_src2);
          src_r2r3_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63hi_w, data_src2);
        } else {
          src_r2r3_0_31lo  = _permutex2var_epi8_sim_get32(wi_r2r3_0_31lo,  sa_r2r3_0_31lo,  data_src, data_src2);
          src_r2r3_0_31hi  = _permutex2var_epi8_sim_get32(wi_r2r3_0_31hi,  sa_r2r3_0_31hi,  data_src, data_src2);
          src_r2r3_32_63lo = _permutex2var_epi8_sim_get32(wi_r2r3_32_63lo, sa_r2r3_32_63lo, data_src, data_src2);
          src_r2r3_32_63hi = _permutex2var_epi8_sim_get32(wi_r2r3_32_63hi, sa_r2r3_32_63hi, data_src, data_src2);
        }

        // for r4r5
        perm_rNrNp1_0_31lo_w  = _mm512_add_epi16(perm_rNrNp1_0_31lo_w,  two_epi16);
        perm_rNrNp1_0_31hi_w  = _mm512_add_epi16(perm_rNrNp1_0_31hi_w,  two_epi16);
        perm_rNrNp1_32_63lo_w = _mm512_add_epi16(perm_rNrNp1_32_63lo_w, two_epi16);
        perm_rNrNp1_32_63hi_w = _mm512_add_epi16(perm_rNrNp1_32_63hi_w, two_epi16);

        __m512i result_0_31lo, result_0_31hi;
        __m512i result_32_63lo, result_32_63hi;

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo,  coef_r0r1_0_31lo);
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r2r3_0_31lo,  coef_r2r3_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi,  coef_r0r1_0_31hi);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r2r3_0_31hi,  coef_r2r3_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  coef_r0r1_0_31lo),  _mm512_madd_epi16(src_r2r3_0_31lo,  coef_r2r3_0_31lo));
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  coef_r0r1_0_31hi),  _mm512_madd_epi16(src_r2r3_0_31hi,  coef_r2r3_0_31hi));
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));
        }

        // rows 4..5
        __m512i src_r4r5_0_31lo, src_r4r5_0_31hi, src_r4r5_32_63lo, src_r4r5_32_63hi;
        if constexpr (UseVNNI) {
          src_r4r5_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w,  data_src2);
          src_r4r5_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w,  data_src2);
          src_r4r5_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63lo_w, data_src2);
          src_r4r5_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63hi_w, data_src2);
        } else {
          src_r4r5_0_31lo  = _permutex2var_epi8_sim_get32(wi_r4r5_0_31lo,  sa_r4r5_0_31lo,  data_src, data_src2);
          src_r4r5_0_31hi  = _permutex2var_epi8_sim_get32(wi_r4r5_0_31hi,  sa_r4r5_0_31hi,  data_src, data_src2);
          src_r4r5_32_63lo = _permutex2var_epi8_sim_get32(wi_r4r5_32_63lo, sa_r4r5_32_63lo, data_src, data_src2);
          src_r4r5_32_63hi = _permutex2var_epi8_sim_get32(wi_r4r5_32_63hi, sa_r4r5_32_63hi, data_src, data_src2);
        }

        // for r6r7
        perm_rNrNp1_0_31lo_w  = _mm512_add_epi16(perm_rNrNp1_0_31lo_w,  two_epi16);
        perm_rNrNp1_0_31hi_w  = _mm512_add_epi16(perm_rNrNp1_0_31hi_w,  two_epi16);
        perm_rNrNp1_32_63lo_w = _mm512_add_epi16(perm_rNrNp1_32_63lo_w, two_epi16);
        perm_rNrNp1_32_63hi_w = _mm512_add_epi16(perm_rNrNp1_32_63hi_w, two_epi16);

        __m512i src_r6r7_0_31lo, src_r6r7_0_31hi, src_r6r7_32_63lo, src_r6r7_32_63hi;
        if constexpr (UseVNNI) {
          src_r6r7_0_31lo  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w,  data_src2);
          src_r6r7_0_31hi  = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w,  data_src2);
          src_r6r7_32_63lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63lo_w, data_src2);
          src_r6r7_32_63hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_32_63hi_w, data_src2);
        } else {
          src_r6r7_0_31lo  = _permutex2var_epi8_sim_get32(wi_r6r7_0_31lo,  sa_r6r7_0_31lo,  data_src, data_src2);
          src_r6r7_0_31hi  = _permutex2var_epi8_sim_get32(wi_r6r7_0_31hi,  sa_r6r7_0_31hi,  data_src, data_src2);
          src_r6r7_32_63lo = _permutex2var_epi8_sim_get32(wi_r6r7_32_63lo, sa_r6r7_32_63lo, data_src, data_src2);
          src_r6r7_32_63hi = _permutex2var_epi8_sim_get32(wi_r6r7_32_63hi, sa_r6r7_32_63hi, data_src, data_src2);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r4r5_0_31lo,  coef_r4r5_0_31lo);
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r6r7_0_31lo,  coef_r6r7_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r4r5_0_31hi,  coef_r4r5_0_31hi);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r6r7_0_31hi,  coef_r6r7_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r4r5_32_63lo, coef_r4r5_32_63lo);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r6r7_32_63lo, coef_r6r7_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r4r5_32_63hi, coef_r4r5_32_63hi);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r6r7_32_63hi, coef_r6r7_32_63hi);
          // rounding VNNI in first FMA already summed
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31lo,  coef_r4r5_0_31lo),  result_0_31lo);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31hi,  coef_r4r5_0_31hi),  result_0_31hi);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_32_63lo, coef_r4r5_32_63lo), result_32_63lo);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_32_63hi, coef_r4r5_32_63hi), result_32_63hi);

          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31lo,  coef_r6r7_0_31lo),  result_0_31lo);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31hi,  coef_r6r7_0_31hi),  result_0_31hi);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_32_63lo, coef_r6r7_32_63lo), result_32_63lo);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_32_63hi, coef_r6r7_32_63hi), result_32_63hi);

          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  rounder);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  rounder);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, rounder);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, rounder);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale8bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale8bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale8bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale8bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        __m256i result_0_31_u8  = _mm512_cvtusepi16_epi8(result_0_31_int16);
        __m256i result_32_63_u8 = _mm512_cvtusepi16_epi8(result_32_63_int16);

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), _mm512_inserti64x4(_mm512_castsi256_si512(result_0_31_u8), result_32_63_u8, 1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

// filter size up to 16, pretransposed coefficients
// 32 target uint8_t pixels at a time
// 128-byte source loads (2x64 uint8_t pixels)
template<bool UseVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks16_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  constexpr int PIXELS_AT_A_TIME = 32;

  // Safe zone: 32 pixels per step: a 128 pixel window from each 32-pixel block start. Each window start must be checked

  // (safelimit_128_pixels_each64th_target checked only the step start: the other windows could read past the source buffer).

  const int width_safe_mod = (program->safelimit_128_pixels_each32th_target.overread_possible ? program->safelimit_128_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 16);
  assert(program->target_size_alignment >= 32);
  assert(FRAME_ALIGN >= 32);
  assert(program->filter_size_alignment >= 16);

  const int max_scanlines = program->max_scanlines;

  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r2r3_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r2r3_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r4r5_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r4r5_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r6r7_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r6r7_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r8r9_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r8r9_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r10r11_0_31lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r10r11_0_31hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r12r13_0_31lo = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r12r13_0_31hi = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r14r15_0_31lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r14r15_0_31hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));

      int iStart = program->pixel_offset[x];
      __m512i m512i_Start = _mm512_set1_epi32(iStart);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);

      __m512i perm_0_0_31 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15), m256i_perm_0_16_31, 1);
      __m512i perm_1_0_31 = _mm512_add_epi16(perm_0_0_31, one_epi16);

      const __m512i perm_r0r1_0_31lo = _mm512_unpacklo_epi16(perm_0_0_31, perm_1_0_31);
      const __m512i perm_r0r1_0_31hi = _mm512_unpackhi_epi16(perm_0_0_31, perm_1_0_31);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      // BASE path: precompute word-index (wi) and shift-amount (sa) for each tap-pair perm.
      // perm_rNrNp1 holds 32 x 16-bit byte addresses; wi = addr>>1 selects the 16-bit word in
      // {data_src, data_src2}, sa = (addr<<3)&8 selects high (1) or low (0) byte within that word.
      // Precomputing here eliminates ~12 instructions/gather (cvtepu8+extract+slli+srli+maskz_mov)
      // from the hot y-loop, leaving only the 3-instruction sim_get32 core per tap-pair.
      __m512i wi_r0r1_lo={},sa_r0r1_lo={},wi_r0r1_hi={},sa_r0r1_hi={};
      __m512i wi_r2r3_lo={},sa_r2r3_lo={},wi_r2r3_hi={},sa_r2r3_hi={};
      __m512i wi_r4r5_lo={},sa_r4r5_lo={},wi_r4r5_hi={},sa_r4r5_hi={};
      __m512i wi_r6r7_lo={},sa_r6r7_lo={},wi_r6r7_hi={},sa_r6r7_hi={};
      __m512i wi_r8r9_lo={},sa_r8r9_lo={},wi_r8r9_hi={},sa_r8r9_hi={};
      __m512i wi_r10r11_lo={},sa_r10r11_lo={},wi_r10r11_hi={},sa_r10r11_hi={};
      __m512i wi_r12r13_lo={},sa_r12r13_lo={},wi_r12r13_hi={},sa_r12r13_hi={};
      __m512i wi_r14r15_lo={},sa_r14r15_lo={},wi_r14r15_hi={},sa_r14r15_hi={};
      if constexpr (!UseVNNI) {
        const __m512i c_8w = _mm512_set1_epi16(8);
        auto make_wi_sa = [&](__m512i p, __m512i& wi, __m512i& sa) {
          wi = _mm512_srli_epi16(p, 1);
          sa = _mm512_and_si512(_mm512_slli_epi16(p, 3), c_8w);
        };
        __m512i p_lo = perm_r0r1_0_31lo, p_hi = perm_r0r1_0_31hi;
        make_wi_sa(p_lo, wi_r0r1_lo,   sa_r0r1_lo);   make_wi_sa(p_hi, wi_r0r1_hi,   sa_r0r1_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r2r3_lo,   sa_r2r3_lo);   make_wi_sa(p_hi, wi_r2r3_hi,   sa_r2r3_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r4r5_lo,   sa_r4r5_lo);   make_wi_sa(p_hi, wi_r4r5_hi,   sa_r4r5_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r6r7_lo,   sa_r6r7_lo);   make_wi_sa(p_hi, wi_r6r7_hi,   sa_r6r7_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r8r9_lo,   sa_r8r9_lo);   make_wi_sa(p_hi, wi_r8r9_hi,   sa_r8r9_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r10r11_lo, sa_r10r11_lo); make_wi_sa(p_hi, wi_r10r11_hi, sa_r10r11_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r12r13_lo, sa_r12r13_lo); make_wi_sa(p_hi, wi_r12r13_hi, sa_r12r13_hi);
        p_lo = _mm512_add_epi16(p_lo, two_epi16); p_hi = _mm512_add_epi16(p_hi, two_epi16);
        make_wi_sa(p_lo, wi_r14r15_lo, sa_r14r15_lo); make_wi_sa(p_hi, wi_r14r15_hi, sa_r14r15_hi);
      }

      uint8_t* AVS_RESTRICT dst_ptr = dst8 + x + y_from * dst_pitch;
      const uint8_t* src_ptr = src8 + iStart + y_from * src_pitch;

      const int remaining = program->source_size - iStart;
      __mmask64 k1 = _bzhi_u64(~0ULL, remaining);
      const __mmask64 k2 = _bzhi_u64(~0ULL, std::max(0, remaining - 64));
      const __mmask64 k_zh8 = 0x5555555555555555ULL;

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src, data_src2;

        __m512i perm_rNrNp1_0_31lo_w = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi_w = perm_r0r1_0_31hi;

        if constexpr (partial_load) {
          data_src  = _mm512_maskz_loadu_epi8(k1, src_ptr);
          data_src2 = _mm512_maskz_loadu_epi8(k2, src_ptr + 64);
        }
        else {
          data_src  = _mm512_loadu_si512(src_ptr);
          data_src2 = _mm512_loadu_si512(src_ptr + 64);
        }

        // rows 0..1
        __m512i src_r0r1_0_31lo, src_r0r1_0_31hi;
        if constexpr (UseVNNI) {
          src_r0r1_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r0r1_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r0r1_0_31lo = _permutex2var_epi8_sim_get32(wi_r0r1_lo, sa_r0r1_lo, data_src, data_src2);
          src_r0r1_0_31hi = _permutex2var_epi8_sim_get32(wi_r0r1_hi, sa_r0r1_hi, data_src, data_src2);
        }

        // for r2r3
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r2r3_0_31lo, src_r2r3_0_31hi;
        if constexpr (UseVNNI) {
          src_r2r3_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r2r3_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r2r3_0_31lo = _permutex2var_epi8_sim_get32(wi_r2r3_lo, sa_r2r3_lo, data_src, data_src2);
          src_r2r3_0_31hi = _permutex2var_epi8_sim_get32(wi_r2r3_hi, sa_r2r3_hi, data_src, data_src2);
        }

        // for r4r5
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i result_0_31lo, result_0_31hi;

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo, coef_r0r1_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r2r3_0_31lo, coef_r2r3_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi, coef_r0r1_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r2r3_0_31hi, coef_r2r3_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo, coef_r0r1_0_31lo), _mm512_madd_epi16(src_r2r3_0_31lo, coef_r2r3_0_31lo));
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi, coef_r0r1_0_31hi), _mm512_madd_epi16(src_r2r3_0_31hi, coef_r2r3_0_31hi));
        }

        // rows 4..5
        __m512i src_r4r5_0_31lo, src_r4r5_0_31hi;
        if constexpr (UseVNNI) {
          src_r4r5_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r4r5_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r4r5_0_31lo = _permutex2var_epi8_sim_get32(wi_r4r5_lo, sa_r4r5_lo, data_src, data_src2);
          src_r4r5_0_31hi = _permutex2var_epi8_sim_get32(wi_r4r5_hi, sa_r4r5_hi, data_src, data_src2);
        }

        // for r6r7
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r6r7_0_31lo, src_r6r7_0_31hi;
        if constexpr (UseVNNI) {
          src_r6r7_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r6r7_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r6r7_0_31lo = _permutex2var_epi8_sim_get32(wi_r6r7_lo, sa_r6r7_lo, data_src, data_src2);
          src_r6r7_0_31hi = _permutex2var_epi8_sim_get32(wi_r6r7_hi, sa_r6r7_hi, data_src, data_src2);
        }

        // for r8r9
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r4r5_0_31lo, coef_r4r5_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r6r7_0_31lo, coef_r6r7_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r4r5_0_31hi, coef_r4r5_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r6r7_0_31hi, coef_r6r7_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31lo, coef_r4r5_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31hi, coef_r4r5_0_31hi), result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31lo, coef_r6r7_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31hi, coef_r6r7_0_31hi), result_0_31hi);
        }

        // rows 8..9
        __m512i src_r8r9_0_31lo, src_r8r9_0_31hi;
        if constexpr (UseVNNI) {
          src_r8r9_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r8r9_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r8r9_0_31lo = _permutex2var_epi8_sim_get32(wi_r8r9_lo, sa_r8r9_lo, data_src, data_src2);
          src_r8r9_0_31hi = _permutex2var_epi8_sim_get32(wi_r8r9_hi, sa_r8r9_hi, data_src, data_src2);
        }

        // for r10r11
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r10r11_0_31lo, src_r10r11_0_31hi;
        if constexpr (UseVNNI) {
          src_r10r11_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r10r11_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r10r11_0_31lo = _permutex2var_epi8_sim_get32(wi_r10r11_lo, sa_r10r11_lo, data_src, data_src2);
          src_r10r11_0_31hi = _permutex2var_epi8_sim_get32(wi_r10r11_hi, sa_r10r11_hi, data_src, data_src2);
        }

        // for r12r13
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r8r9_0_31lo,   coef_r8r9_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r10r11_0_31lo, coef_r10r11_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r8r9_0_31hi,   coef_r8r9_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r10r11_0_31hi, coef_r10r11_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r8r9_0_31lo,   coef_r8r9_0_31lo),   result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r8r9_0_31hi,   coef_r8r9_0_31hi),   result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r10r11_0_31lo, coef_r10r11_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r10r11_0_31hi, coef_r10r11_0_31hi), result_0_31hi);
        }

        // rows 12..13
        __m512i src_r12r13_0_31lo, src_r12r13_0_31hi;
        if constexpr (UseVNNI) {
          src_r12r13_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r12r13_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r12r13_0_31lo = _permutex2var_epi8_sim_get32(wi_r12r13_lo, sa_r12r13_lo, data_src, data_src2);
          src_r12r13_0_31hi = _permutex2var_epi8_sim_get32(wi_r12r13_hi, sa_r12r13_hi, data_src, data_src2);
        }

        // for r14r15
        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r14r15_0_31lo, src_r14r15_0_31hi;
        if constexpr (UseVNNI) {
          src_r14r15_0_31lo = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31lo_w, data_src2);
          src_r14r15_0_31hi = _mm512_maskz_permutex2var_epi8(k_zh8, data_src, perm_rNrNp1_0_31hi_w, data_src2);
        } else {
          src_r14r15_0_31lo = _permutex2var_epi8_sim_get32(wi_r14r15_lo, sa_r14r15_lo, data_src, data_src2);
          src_r14r15_0_31hi = _permutex2var_epi8_sim_get32(wi_r14r15_hi, sa_r14r15_hi, data_src, data_src2);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r12r13_0_31lo, coef_r12r13_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r14r15_0_31lo, coef_r14r15_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r12r13_0_31hi, coef_r12r13_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r14r15_0_31hi, coef_r14r15_0_31hi);
          // rounding VNNI in first FMA already summed
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r12r13_0_31lo, coef_r12r13_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r12r13_0_31hi, coef_r12r13_0_31hi), result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r14r15_0_31lo, coef_r14r15_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r14r15_0_31hi, coef_r14r15_0_31hi), result_0_31hi);

          result_0_31lo = _mm512_add_epi32(result_0_31lo, rounder);
          result_0_31hi = _mm512_add_epi32(result_0_31hi, rounder);
        }

        result_0_31lo = _mm512_srai_epi32(result_0_31lo, FPScale8bits);
        result_0_31hi = _mm512_srai_epi32(result_0_31hi, FPScale8bits);

        __m512i result_0_31_int16 = _mm512_packus_epi32(result_0_31lo, result_0_31hi);

        __m256i result_0_31_u8 = _mm512_cvtusepi16_epi8(result_0_31_int16);

        _mm256_stream_si256(reinterpret_cast<__m256i*>(dst_ptr), result_0_31_u8);

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

// filter size up to 64, pretransposed coefficients
// 64 target uint8_t pixels at a time in 2 groups of 32
// 2 groups of 128-byte source loads
template<bool UseVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_2s32_ks64_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  int filter_size_real = program->filter_size_real;
  if ((filter_size_real / 2 * 2) != filter_size_real) filter_size_real++;

  constexpr int PIXELS_AT_A_TIME = 64;

  // Safe zone: two 128 pixel windows, from x and x+32. Each window start must be checked

  // (safelimit_128_pixels_each64th_target checked only the step start: the other windows could read past the source buffer).

  const int width_safe_mod = (program->safelimit_128_pixels_each32th_target.overread_possible ? program->safelimit_128_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 64);
  assert(program->target_size_alignment >= 64);
  assert(FRAME_ALIGN >= 64);
  assert(program->filter_size_alignment >= 8);

  const int max_scanlines = program->max_scanlines;

  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart   = program->pixel_offset[x];
      int iStart_2 = program->pixel_offset[x + 32];
      __m512i m512i_Start   = _mm512_set1_epi32(iStart);
      __m512i m512i_Start_2 = _mm512_set1_epi32(iStart_2);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start_2);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start_2);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i one_epi16 = _mm512_set1_epi16(1);
      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      // BASE: precompute wi/sa outside y-loop. perm vectors are epi16 (32 16-bit byte indices).
      // Each kr step the perm advances by +2 bytes = +1 word, so wi += 1 and sa is invariant.
      // perm_1 = perm_0 + 1: wi_p1[i] = wi_p0[i] + (perm_0[i] & 1), sa_p1 = sa_p0 ^ 8.
      __m512i wi_g1_p0_base = {}, wi_g2_p0_base = {}, wi_g1_p1_base = {}, wi_g2_p1_base = {};
      __m512i sa_g1_p0 = {}, sa_g2_p0 = {}, sa_g1_p1 = {}, sa_g2_p1 = {};
      if constexpr (!UseVNNI) {
        const __m512i c_8w = _mm512_set1_epi16(8);
        sa_g1_p0      = _mm512_and_si512(_mm512_slli_epi16(perm_0_0_31, 3), c_8w);
        wi_g1_p0_base = _mm512_srli_epi16(perm_0_0_31, 1);
        sa_g1_p1      = _mm512_xor_si512(sa_g1_p0, c_8w);
        wi_g1_p1_base = _mm512_add_epi16(wi_g1_p0_base, _mm512_srli_epi16(sa_g1_p0, 3));
        sa_g2_p0      = _mm512_and_si512(_mm512_slli_epi16(perm_0_32_63, 3), c_8w);
        wi_g2_p0_base = _mm512_srli_epi16(perm_0_32_63, 1);
        sa_g2_p1      = _mm512_xor_si512(sa_g2_p0, c_8w);
        wi_g2_p1_base = _mm512_add_epi16(wi_g2_p0_base, _mm512_srli_epi16(sa_g2_p0, 3));
      }

      uint8_t* AVS_RESTRICT dst_ptr  = dst8 + x + y_from * dst_pitch;
      const uint8_t* src_ptr   = src8 + iStart   + y_from * src_pitch;
      const uint8_t* src_ptr_2 = src8 + iStart_2 + y_from * src_pitch;

      const int remaining   = program->source_size - iStart;
      const int remaining_2 = program->source_size - iStart_2;
      const __mmask64 k1   = _bzhi_u64(~0ULL, remaining);
      const __mmask64 k2   = _bzhi_u64(~0ULL, std::max(0, remaining - 64));
      const __mmask64 k1_2 = _bzhi_u64(~0ULL, remaining_2);
      const __mmask64 k2_2 = _bzhi_u64(~0ULL, std::max(0, remaining_2 - 64));
      const __mmask64 k_zh8 = 0x5555555555555555ULL;

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src, data_src2, data_src_2, data_src2_2;

        // VBMI: reset working perm copies per row; BASE: reset running wi counters per row
        __m512i perm_0_0_31w = {}, perm_0_32_63w = {}, perm_1_0_31w = {}, perm_1_32_63w = {};
        __m512i wi_g1_p0 = {}, wi_g2_p0 = {}, wi_g1_p1 = {}, wi_g2_p1 = {};
        if constexpr (UseVNNI) {
          perm_0_0_31w  = perm_0_0_31;
          perm_0_32_63w = perm_0_32_63;
          perm_1_0_31w  = perm_1_0_31;
          perm_1_32_63w = perm_1_32_63;
        } else {
          wi_g1_p0 = wi_g1_p0_base;
          wi_g2_p0 = wi_g2_p0_base;
          wi_g1_p1 = wi_g1_p1_base;
          wi_g2_p1 = wi_g2_p1_base;
        }

        if constexpr (partial_load) {
          data_src    = _mm512_maskz_loadu_epi8(k1,   src_ptr);
          data_src2   = _mm512_maskz_loadu_epi8(k2,   src_ptr + 64);
          data_src_2  = _mm512_maskz_loadu_epi8(k1_2, src_ptr_2);
          data_src2_2 = _mm512_maskz_loadu_epi8(k2_2, src_ptr_2 + 64);
        }
        else {
          data_src    = _mm512_loadu_si512(src_ptr);
          data_src2   = _mm512_loadu_si512(src_ptr + 64);
          data_src_2  = _mm512_loadu_si512(src_ptr_2);
          data_src2_2 = _mm512_loadu_si512(src_ptr_2 + 64);
        }

        __m512i result_0_31lo  = rounder;
        __m512i result_0_31hi  = rounder;
        __m512i result_32_63lo = rounder;
        __m512i result_32_63hi = rounder;

        const __m512i* current_coeff_SIMDw = current_coeff_SIMD;

        for (int kr = 0; kr < filter_size_real; kr += 2)
        {
          __m512i src_r0_0_31, src_r0_32_63, src_r1_0_31, src_r1_32_63;
          if constexpr (UseVNNI) {
            src_r0_0_31  = _mm512_maskz_permutex2var_epi8_SIMUL<true>(k_zh8, data_src,   perm_0_0_31w,  data_src2);
            src_r0_32_63 = _mm512_maskz_permutex2var_epi8_SIMUL<true>(k_zh8, data_src_2, perm_0_32_63w, data_src2_2);
            src_r1_0_31  = _mm512_maskz_permutex2var_epi8_SIMUL<true>(k_zh8, data_src,   perm_1_0_31w,  data_src2);
            src_r1_32_63 = _mm512_maskz_permutex2var_epi8_SIMUL<true>(k_zh8, data_src_2, perm_1_32_63w, data_src2_2);
            perm_0_0_31w  = _mm512_add_epi16(perm_0_0_31w,  two_epi16);
            perm_0_32_63w = _mm512_add_epi16(perm_0_32_63w, two_epi16);
            perm_1_0_31w  = _mm512_add_epi16(perm_1_0_31w,  two_epi16);
            perm_1_32_63w = _mm512_add_epi16(perm_1_32_63w, two_epi16);
          } else {
            // BASE: sim_get32 reads word wi (already indexed into [data_src | data_src2])
            // and shifts by sa to extract the target byte; avoids full SIMUL recomputation
            src_r0_0_31  = _permutex2var_epi8_sim_get32(wi_g1_p0, sa_g1_p0, data_src,   data_src2);
            src_r0_32_63 = _permutex2var_epi8_sim_get32(wi_g2_p0, sa_g2_p0, data_src_2, data_src2_2);
            src_r1_0_31  = _permutex2var_epi8_sim_get32(wi_g1_p1, sa_g1_p1, data_src,   data_src2);
            src_r1_32_63 = _permutex2var_epi8_sim_get32(wi_g2_p1, sa_g2_p1, data_src_2, data_src2_2);
            wi_g1_p0 = _mm512_add_epi16(wi_g1_p0, one_epi16);
            wi_g2_p0 = _mm512_add_epi16(wi_g2_p0, one_epi16);
            wi_g1_p1 = _mm512_add_epi16(wi_g1_p1, one_epi16);
            wi_g2_p1 = _mm512_add_epi16(wi_g2_p1, one_epi16);
          }

          __m512i src_r0r1_0_31lo  = _mm512_unpacklo_epi16(src_r0_0_31,  src_r1_0_31);
          __m512i src_r0r1_0_31hi  = _mm512_unpackhi_epi16(src_r0_0_31,  src_r1_0_31);
          __m512i src_r0r1_32_63lo = _mm512_unpacklo_epi16(src_r0_32_63, src_r1_32_63);
          __m512i src_r0r1_32_63hi = _mm512_unpackhi_epi16(src_r0_32_63, src_r1_32_63);

          if constexpr (UseVNNI)
          {
            result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r0r1_0_31lo,  _mm512_load_si512(current_coeff_SIMDw + 0));
            result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r0r1_0_31hi,  _mm512_load_si512(current_coeff_SIMDw + 1));
            result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r0r1_32_63lo, _mm512_load_si512(current_coeff_SIMDw + 2));
            result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r0r1_32_63hi, _mm512_load_si512(current_coeff_SIMDw + 3));
          }
          else
          {
            result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  _mm512_load_si512(current_coeff_SIMDw + 0)), result_0_31lo);
            result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  _mm512_load_si512(current_coeff_SIMDw + 1)), result_0_31hi);
            result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, _mm512_load_si512(current_coeff_SIMDw + 2)), result_32_63lo);
            result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, _mm512_load_si512(current_coeff_SIMDw + 3)), result_32_63hi);
          }

          current_coeff_SIMDw += 4;
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale8bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale8bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale8bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale8bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        __m256i result_0_31_u8  = _mm512_cvtusepi16_epi8(result_0_31_int16);
        __m256i result_32_63_u8 = _mm512_cvtusepi16_epi8(result_32_63_int16);

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), _mm512_inserti64x4(_mm512_castsi256_si512(result_0_31_u8), result_32_63_u8, 1));

        dst_ptr   += dst_pitch;
        src_ptr   += src_pitch;
        src_ptr_2 += src_pitch;
      }

      current_coeff_SIMD += filter_size_real * 2;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

// filter size 8, pretransposed coefficients
// 64 target uint8_t pixels at a time in 2 groups of 32
// 2 groups of 128-byte source loads
template<bool UseVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_2s32_ks8_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  constexpr int PIXELS_AT_A_TIME = 64;

  // Safe zone: two 128 pixel windows, from x and x+32. Each window start must be checked

  // (safelimit_128_pixels_each64th_target checked only the step start: the other windows could read past the source buffer).

  const int width_safe_mod = (program->safelimit_128_pixels_each32th_target.overread_possible ? program->safelimit_128_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  const int max_scanlines = program->max_scanlines;

  __m512i rounder = _mm512_set1_epi32(1 << (FPScale8bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      const __m512i coef_r0r1_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r2r3_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r4r5_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r4r5_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r4r5_32_63lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r4r5_32_63hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r6r7_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r6r7_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r6r7_32_63lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r6r7_32_63hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart   = program->pixel_offset[x];
      int iStart_2 = program->pixel_offset[x + 32];
      __m512i m512i_Start   = _mm512_set1_epi32(iStart);
      __m512i m512i_Start_2 = _mm512_set1_epi32(iStart_2);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start_2);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start_2);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m128i mm128i_perm_0_0_15  = _mm256_cvtepi16_epi8(m256i_perm_0_0_15);
      __m128i mm128i_perm_0_16_31 = _mm256_cvtepi16_epi8(m256i_perm_0_16_31);
      __m128i mm128i_perm_0_32_47 = _mm256_cvtepi16_epi8(m256i_perm_0_32_47);
      __m128i mm128i_perm_0_48_63 = _mm256_cvtepi16_epi8(m256i_perm_0_48_63);

      __m512i perm_0 = _mm512_inserti32x4(_mm512_castsi128_si512(mm128i_perm_0_0_15), mm128i_perm_0_16_31, 1);
      perm_0 = _mm512_inserti32x4(perm_0, mm128i_perm_0_32_47, 2);
      perm_0 = _mm512_inserti32x4(perm_0, mm128i_perm_0_48_63, 3);

      uint8_t* AVS_RESTRICT dst_ptr = dst8 + x + y_from * dst_pitch;
      const uint8_t* src_ptr   = src8 + iStart   + y_from * src_pitch;
      const uint8_t* src_ptr_2 = src8 + iStart_2 + y_from * src_pitch;

      const int remaining   = program->source_size - iStart;
      const int remaining_2 = program->source_size - iStart_2;
      const __mmask64 k1   = _bzhi_u64(~0ULL, remaining);
      const __mmask64 k2   = _bzhi_u64(~0ULL, std::max(0, remaining   - 64));
      const __mmask64 k1_2 = _bzhi_u64(~0ULL, remaining_2);
      const __mmask64 k2_2 = _bzhi_u64(~0ULL, std::max(0, remaining_2 - 64));

      // perm_0+N for each kernel tap — loop-invariant, precomputed for both VBMI and BASE paths
      const __m512i pw_1 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(1));
      const __m512i pw_2 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(2));
      const __m512i pw_3 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(3));
      const __m512i pw_4 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(4));
      const __m512i pw_5 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(5));
      const __m512i pw_6 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(6));
      const __m512i pw_7 = _mm512_add_epi8(perm_0, _mm512_set1_epi8(7));

      // BASE path only: precompute word_idx and shift_amt from each perm vector.
      // MSVC fails to hoist these from the y-loop even though perm_0 is invariant.
      __m512i wi_lo_0 = {}, sa_lo_0 = {}, wi_hi_0 = {}, sa_hi_0 = {};
      __m512i wi_lo_1 = {}, sa_lo_1 = {}, wi_hi_1 = {}, sa_hi_1 = {};
      __m512i wi_lo_2 = {}, sa_lo_2 = {}, wi_hi_2 = {}, sa_hi_2 = {};
      __m512i wi_lo_3 = {}, sa_lo_3 = {}, wi_hi_3 = {}, sa_hi_3 = {};
      __m512i wi_lo_4 = {}, sa_lo_4 = {}, wi_hi_4 = {}, sa_hi_4 = {};
      __m512i wi_lo_5 = {}, sa_lo_5 = {}, wi_hi_5 = {}, sa_hi_5 = {};
      __m512i wi_lo_6 = {}, sa_lo_6 = {}, wi_hi_6 = {}, sa_hi_6 = {};
      __m512i wi_lo_7 = {}, sa_lo_7 = {}, wi_hi_7 = {}, sa_hi_7 = {};
      if constexpr (!UseVNNI) {
        const __m512i c_8 = _mm512_set1_epi16(8);
        auto make_wi_sa = [&](const __m512i pw, __m512i &wi_lo, __m512i &sa_lo, __m512i &wi_hi, __m512i &sa_hi) {
          __m512i lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(pw));
          __m512i hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(pw, 1));
          wi_lo = _mm512_srli_epi16(lo, 1);
          sa_lo = _mm512_and_si512(_mm512_slli_epi16(lo, 3), c_8);
          wi_hi = _mm512_srli_epi16(hi, 1);
          sa_hi = _mm512_and_si512(_mm512_slli_epi16(hi, 3), c_8);
        };
        make_wi_sa(perm_0, wi_lo_0, sa_lo_0, wi_hi_0, sa_hi_0);
        make_wi_sa(pw_1,   wi_lo_1, sa_lo_1, wi_hi_1, sa_hi_1);
        make_wi_sa(pw_2,   wi_lo_2, sa_lo_2, wi_hi_2, sa_hi_2);
        make_wi_sa(pw_3,   wi_lo_3, sa_lo_3, wi_hi_3, sa_hi_3);
        make_wi_sa(pw_4,   wi_lo_4, sa_lo_4, wi_hi_4, sa_hi_4);
        make_wi_sa(pw_5,   wi_lo_5, sa_lo_5, wi_hi_5, sa_hi_5);
        make_wi_sa(pw_6,   wi_lo_6, sa_lo_6, wi_hi_6, sa_hi_6);
        make_wi_sa(pw_7,   wi_lo_7, sa_lo_7, wi_hi_7, sa_hi_7);
      }

      // 512-bit vpermi2b + vextracti64x4 + vpunpcklw/hi saturated port 5 on Ice Lake
      // (56 port-5 uops/iter vs BASE's 32), costing ~8% vs the BASE simulation path.
      // Combined 256-bit indices eliminate extract+unpack, matching BASE at 32 port-5 uops.
      // Coefficient layout is unchanged; unpackbw+unpackqdq construction replicates the word
      // order of unpacklo/hi_epi16. Dual-AVX-512-pipe CPUs (Sapphire Rapids+) may not have
      // suffered this bottleneck due to higher port-5 throughput.
      __m256i comb_01_g1lo={}, comb_01_g1hi={}, comb_01_g2lo={}, comb_01_g2hi={};
      __m256i comb_23_g1lo={}, comb_23_g1hi={}, comb_23_g2lo={}, comb_23_g2hi={};
      __m256i comb_45_g1lo={}, comb_45_g1hi={}, comb_45_g2lo={}, comb_45_g2hi={};
      __m256i comb_67_g1lo={}, comb_67_g1hi={}, comb_67_g2lo={}, comb_67_g2hi={};
      if constexpr (UseVNNI) {
        auto make_combined = [&](__m256i pa, __m256i pb, __m256i& clo, __m256i& chi) {
          __m256i t0 = _mm256_unpacklo_epi8(pa, pb);
          __m256i t1 = _mm256_unpackhi_epi8(pa, pb);
          clo = _mm256_unpacklo_epi64(t0, t1);
          chi = _mm256_unpackhi_epi64(t0, t1);
        };
        __m256i p0g1 = _mm512_castsi512_si256(perm_0);
        __m256i p0g2 = _mm512_extracti64x4_epi64(perm_0, 1);
        // pw_N lower half is a free register alias; upper half = p0g2+N avoids 7 extra extracts
        make_combined(p0g1, _mm512_castsi512_si256(pw_1), comb_01_g1lo, comb_01_g1hi);
        make_combined(p0g2, _mm256_add_epi8(p0g2, _mm256_set1_epi8(1)), comb_01_g2lo, comb_01_g2hi);
        make_combined(_mm512_castsi512_si256(pw_2), _mm512_castsi512_si256(pw_3), comb_23_g1lo, comb_23_g1hi);
        make_combined(_mm256_add_epi8(p0g2, _mm256_set1_epi8(2)), _mm256_add_epi8(p0g2, _mm256_set1_epi8(3)), comb_23_g2lo, comb_23_g2hi);
        make_combined(_mm512_castsi512_si256(pw_4), _mm512_castsi512_si256(pw_5), comb_45_g1lo, comb_45_g1hi);
        make_combined(_mm256_add_epi8(p0g2, _mm256_set1_epi8(4)), _mm256_add_epi8(p0g2, _mm256_set1_epi8(5)), comb_45_g2lo, comb_45_g2hi);
        make_combined(_mm512_castsi512_si256(pw_6), _mm512_castsi512_si256(pw_7), comb_67_g1lo, comb_67_g1hi);
        make_combined(_mm256_add_epi8(p0g2, _mm256_set1_epi8(6)), _mm256_add_epi8(p0g2, _mm256_set1_epi8(7)), comb_67_g2lo, comb_67_g2hi);
      }

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src, data_src2, data_src_2, data_src2_2;

        if constexpr (partial_load) {
          data_src    = _mm512_maskz_loadu_epi8(k1,   src_ptr);
          data_src2   = _mm512_maskz_loadu_epi8(k2,   src_ptr   + 64);
          data_src_2  = _mm512_maskz_loadu_epi8(k1_2, src_ptr_2);
          data_src2_2 = _mm512_maskz_loadu_epi8(k2_2, src_ptr_2 + 64);
        }
        else {
          data_src    = _mm512_loadu_si512(src_ptr);
          data_src2   = _mm512_loadu_si512(src_ptr   + 64);
          data_src_2  = _mm512_loadu_si512(src_ptr_2);
          data_src2_2 = _mm512_loadu_si512(src_ptr_2 + 64);
        }

        __m512i src_r0r1_0_31lo, src_r0r1_0_31hi, src_r0r1_32_63lo, src_r0r1_32_63hi;
        __m512i src_r2r3_0_31lo, src_r2r3_0_31hi, src_r2r3_32_63lo, src_r2r3_32_63hi;

        if constexpr (UseVNNI) {
          // 512-bit vpermi2b keeps the full 128-byte source window (256-bit vpermi2b only
          // addresses 64 bytes, corrupting pixels whose index falls in data_src[32..63]).
          // castsi256_si512 on index = free (upper 32 index bytes are undefined but only
          // affect output bytes 32-63 which are discarded by castsi512_si256 = also free).
          src_r0r1_0_31lo  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_01_g1lo), data_src2)));
          src_r0r1_0_31hi  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_01_g1hi), data_src2)));
          src_r0r1_32_63lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_01_g2lo), data_src2_2)));
          src_r0r1_32_63hi = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_01_g2hi), data_src2_2)));
          src_r2r3_0_31lo  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_23_g1lo), data_src2)));
          src_r2r3_0_31hi  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_23_g1hi), data_src2)));
          src_r2r3_32_63lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_23_g2lo), data_src2_2)));
          src_r2r3_32_63hi = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_23_g2hi), data_src2_2)));
        } else {
          // BASE: sim_get32 returns 32 16-bit words with gathered bytes in low 8 bits;
          // feed unpacklo/hi directly — avoids cvtepu8_epi16 round-trip
          __m512i d0l = _permutex2var_epi8_sim_get32(wi_lo_0, sa_lo_0, data_src, data_src2);
          __m512i d0h = _permutex2var_epi8_sim_get32(wi_hi_0, sa_hi_0, data_src_2, data_src2_2);
          __m512i d1l = _permutex2var_epi8_sim_get32(wi_lo_1, sa_lo_1, data_src, data_src2);
          __m512i d1h = _permutex2var_epi8_sim_get32(wi_hi_1, sa_hi_1, data_src_2, data_src2_2);
          __m512i d2l = _permutex2var_epi8_sim_get32(wi_lo_2, sa_lo_2, data_src, data_src2);
          __m512i d2h = _permutex2var_epi8_sim_get32(wi_hi_2, sa_hi_2, data_src_2, data_src2_2);
          __m512i d3l = _permutex2var_epi8_sim_get32(wi_lo_3, sa_lo_3, data_src, data_src2);
          __m512i d3h = _permutex2var_epi8_sim_get32(wi_hi_3, sa_hi_3, data_src_2, data_src2_2);
          src_r0r1_0_31lo  = _mm512_unpacklo_epi16(d0l, d1l);
          src_r0r1_0_31hi  = _mm512_unpackhi_epi16(d0l, d1l);
          src_r0r1_32_63lo = _mm512_unpacklo_epi16(d0h, d1h);
          src_r0r1_32_63hi = _mm512_unpackhi_epi16(d0h, d1h);
          src_r2r3_0_31lo  = _mm512_unpacklo_epi16(d2l, d3l);
          src_r2r3_0_31hi  = _mm512_unpackhi_epi16(d2l, d3l);
          src_r2r3_32_63lo = _mm512_unpacklo_epi16(d2h, d3h);
          src_r2r3_32_63hi = _mm512_unpackhi_epi16(d2h, d3h);
        }

        __m512i result_0_31lo, result_0_31hi, result_32_63lo, result_32_63hi;

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo,  coef_r0r1_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi,  coef_r0r1_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r2r3_0_31lo,  coef_r2r3_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r2r3_0_31hi,  coef_r2r3_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  coef_r0r1_0_31lo),  _mm512_madd_epi16(src_r2r3_0_31lo,  coef_r2r3_0_31lo));
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  coef_r0r1_0_31hi),  _mm512_madd_epi16(src_r2r3_0_31hi,  coef_r2r3_0_31hi));
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));
        }

        __m512i src_r4r5_0_31lo, src_r4r5_0_31hi, src_r4r5_32_63lo, src_r4r5_32_63hi;
        __m512i src_r6r7_0_31lo, src_r6r7_0_31hi, src_r6r7_32_63lo, src_r6r7_32_63hi;

        if constexpr (UseVNNI) {
          src_r4r5_0_31lo  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_45_g1lo), data_src2)));
          src_r4r5_0_31hi  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_45_g1hi), data_src2)));
          src_r4r5_32_63lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_45_g2lo), data_src2_2)));
          src_r4r5_32_63hi = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_45_g2hi), data_src2_2)));
          src_r6r7_0_31lo  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_67_g1lo), data_src2)));
          src_r6r7_0_31hi  = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src,   _mm512_castsi256_si512(comb_67_g1hi), data_src2)));
          src_r6r7_32_63lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_67_g2lo), data_src2_2)));
          src_r6r7_32_63hi = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_permutex2var_epi8(data_src_2, _mm512_castsi256_si512(comb_67_g2hi), data_src2_2)));
        } else {
          __m512i d4l = _permutex2var_epi8_sim_get32(wi_lo_4, sa_lo_4, data_src, data_src2);
          __m512i d4h = _permutex2var_epi8_sim_get32(wi_hi_4, sa_hi_4, data_src_2, data_src2_2);
          __m512i d5l = _permutex2var_epi8_sim_get32(wi_lo_5, sa_lo_5, data_src, data_src2);
          __m512i d5h = _permutex2var_epi8_sim_get32(wi_hi_5, sa_hi_5, data_src_2, data_src2_2);
          __m512i d6l = _permutex2var_epi8_sim_get32(wi_lo_6, sa_lo_6, data_src, data_src2);
          __m512i d6h = _permutex2var_epi8_sim_get32(wi_hi_6, sa_hi_6, data_src_2, data_src2_2);
          __m512i d7l = _permutex2var_epi8_sim_get32(wi_lo_7, sa_lo_7, data_src, data_src2);
          __m512i d7h = _permutex2var_epi8_sim_get32(wi_hi_7, sa_hi_7, data_src_2, data_src2_2);
          src_r4r5_0_31lo  = _mm512_unpacklo_epi16(d4l, d5l);
          src_r4r5_0_31hi  = _mm512_unpackhi_epi16(d4l, d5l);
          src_r4r5_32_63lo = _mm512_unpacklo_epi16(d4h, d5h);
          src_r4r5_32_63hi = _mm512_unpackhi_epi16(d4h, d5h);
          src_r6r7_0_31lo  = _mm512_unpacklo_epi16(d6l, d7l);
          src_r6r7_0_31hi  = _mm512_unpackhi_epi16(d6l, d7l);
          src_r6r7_32_63lo = _mm512_unpacklo_epi16(d6h, d7h);
          src_r6r7_32_63hi = _mm512_unpackhi_epi16(d6h, d7h);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r4r5_0_31lo,  coef_r4r5_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r4r5_0_31hi,  coef_r4r5_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r4r5_32_63lo, coef_r4r5_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r4r5_32_63hi, coef_r4r5_32_63hi);
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r6r7_0_31lo,  coef_r6r7_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r6r7_0_31hi,  coef_r6r7_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r6r7_32_63lo, coef_r6r7_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r6r7_32_63hi, coef_r6r7_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31lo,  coef_r4r5_0_31lo),  result_0_31lo);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31hi,  coef_r4r5_0_31hi),  result_0_31hi);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_32_63lo, coef_r4r5_32_63lo), result_32_63lo);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_32_63hi, coef_r4r5_32_63hi), result_32_63hi);
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31lo,  coef_r6r7_0_31lo),  result_0_31lo);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31hi,  coef_r6r7_0_31hi),  result_0_31hi);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_32_63lo, coef_r6r7_32_63lo), result_32_63lo);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_32_63hi, coef_r6r7_32_63hi), result_32_63hi);

          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  rounder);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  rounder);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, rounder);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, rounder);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale8bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale8bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale8bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale8bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        __m256i result_0_31_u8  = _mm512_cvtusepi16_epi8(result_0_31_int16);
        __m256i result_32_63_u8 = _mm512_cvtusepi16_epi8(result_32_63_int16);

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), _mm512_inserti64x4(_mm512_castsi256_si512(result_0_31_u8), result_32_63_u8, 1));

        dst_ptr     += dst_pitch;
        src_ptr     += src_pitch;
        src_ptr_2   += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

// filter size up to 4, pretransposed coefficients
// 64 target uint16_t pixels at a time in 2 groups of 32
template<bool lessthan16bit, bool UseVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  constexpr int PIXELS_AT_A_TIME = 64;

  const int width_safe_mod = (program->safelimit_64_pixels_each32th_target.overread_possible ? program->safelimit_64_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 4);
  assert(program->target_size_alignment >= 64);
  assert(FRAME_ALIGN >= 64);
  assert(program->filter_size_alignment >= 4);

  const int max_scanlines = program->max_scanlines;

  const __m512i shifttosigned   = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit);
  __m512i rounder     = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r2r3_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart_0_31  = program->pixel_offset[x];
      int iStart_32_63 = program->pixel_offset[x + 32];
      __m512i m512i_Start_0_31  = _mm512_set1_epi32(iStart_0_31);
      __m512i m512i_Start_32_63 = _mm512_set1_epi32(iStart_32_63);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start_0_31);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start_0_31);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start_32_63);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start_32_63);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo  = _mm512_unpacklo_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_0_31hi  = _mm512_unpackhi_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      uint16_t* AVS_RESTRICT dst_ptr     = dst + x + y_from * dst_pitch;
      const uint16_t* src_ptr_0_31  = src + iStart_0_31  + y_from * src_pitch;
      const uint16_t* src_ptr_32_63 = src + iStart_32_63 + y_from * src_pitch;

      const int remaining_0_31  = program->source_size - iStart_0_31;
      const __mmask32 k1_0_31   = _bzhi_u32(~0U, remaining_0_31);
      const __mmask32 k2_0_31   = _bzhi_u32(~0U, std::max(0, remaining_0_31 - 32));
      const int remaining_32_63 = program->source_size - iStart_32_63;
      const __mmask32 k1_32_63  = _bzhi_u32(~0U, remaining_32_63);
      const __mmask32 k2_32_63  = _bzhi_u32(~0U, std::max(0, remaining_32_63 - 32));

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src_0_31, data_src2_0_31, data_src_32_63, data_src2_32_63;

        __m512i perm_rNrNp1_0_31lo  = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi  = perm_r0r1_0_31hi;
        __m512i perm_rNrNp1_32_63lo = perm_r0r1_32_63lo;
        __m512i perm_rNrNp1_32_63hi = perm_r0r1_32_63hi;

        if constexpr (partial_load) {
          data_src_0_31   = _mm512_maskz_loadu_epi16(k1_0_31,  src_ptr_0_31);
          data_src2_0_31  = _mm512_maskz_loadu_epi16(k2_0_31,  src_ptr_0_31  + 32);
          data_src_32_63  = _mm512_maskz_loadu_epi16(k1_32_63, src_ptr_32_63);
          data_src2_32_63 = _mm512_maskz_loadu_epi16(k2_32_63, src_ptr_32_63 + 32);
        }
        else {
          data_src_0_31   = _mm512_loadu_si512(src_ptr_0_31);
          data_src2_0_31  = _mm512_loadu_si512(src_ptr_0_31  + 32);
          data_src_32_63  = _mm512_loadu_si512(src_ptr_32_63);
          data_src2_32_63 = _mm512_loadu_si512(src_ptr_32_63 + 32);
        }

        __m512i src_r0r1_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r0r1_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r0r1_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r0r1_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        // for r2r3
        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r0r1_0_31lo  = _mm512_add_epi16(src_r0r1_0_31lo,  shifttosigned);
          src_r0r1_0_31hi  = _mm512_add_epi16(src_r0r1_0_31hi,  shifttosigned);
          src_r0r1_32_63lo = _mm512_add_epi16(src_r0r1_32_63lo, shifttosigned);
          src_r0r1_32_63hi = _mm512_add_epi16(src_r0r1_32_63hi, shifttosigned);
        }

        __m512i result_0_31lo, result_0_31hi, result_32_63lo, result_32_63hi;

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo,  coef_r0r1_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi,  coef_r0r1_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  coef_r0r1_0_31lo),  rounder);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  coef_r0r1_0_31hi),  rounder);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), rounder);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), rounder);
        }

        __m512i src_r2r3_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r2r3_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r2r3_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r2r3_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        if constexpr (!lessthan16bit) {
          src_r2r3_0_31lo  = _mm512_add_epi16(src_r2r3_0_31lo,  shifttosigned);
          src_r2r3_0_31hi  = _mm512_add_epi16(src_r2r3_0_31hi,  shifttosigned);
          src_r2r3_32_63lo = _mm512_add_epi16(src_r2r3_32_63lo, shifttosigned);
          src_r2r3_32_63hi = _mm512_add_epi16(src_r2r3_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r2r3_0_31lo,  coef_r2r3_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r2r3_0_31hi,  coef_r2r3_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r2r3_0_31lo,  coef_r2r3_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r2r3_0_31hi,  coef_r2r3_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));
        }

        if constexpr (!lessthan16bit) {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  shiftfromsigned);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  shiftfromsigned);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, shiftfromsigned);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, shiftfromsigned);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale16bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale16bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale16bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale16bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        if constexpr (lessthan16bit) {
          result_0_31_int16  = _mm512_min_epu16(result_0_31_int16,  clamp_limit);
          result_32_63_int16 = _mm512_min_epu16(result_32_63_int16, clamp_limit);
        }

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), result_0_31_int16);

        const int remaining_32_63_out = width - (x + 32);
        if (remaining_32_63_out >= 32)
          _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr + 32), result_32_63_int16);
        else if (remaining_32_63_out > 0)
          _mm512_mask_storeu_epi16(dst_ptr + 32, _bzhi_u32(~0u, remaining_32_63_out), result_32_63_int16);

        dst_ptr       += dst_pitch;
        src_ptr_0_31  += src_pitch;
        src_ptr_32_63 += src_pitch;
      }

      current_coeff_SIMD += 8;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

template<bool lessthan16bit, bool UseVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  constexpr int PIXELS_AT_A_TIME = 64;

  const int width_safe_mod = (program->safelimit_64_pixels_each32th_target.overread_possible ? program->safelimit_64_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  const int max_scanlines = program->max_scanlines;

  const __m512i shifttosigned   = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit);
  __m512i rounder     = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r2r3_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r4r5_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r4r5_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r4r5_32_63lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r4r5_32_63hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r6r7_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r6r7_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r6r7_32_63lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r6r7_32_63hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart_0_31  = program->pixel_offset[x];
      int iStart_32_63 = program->pixel_offset[x + 32];
      __m512i m512i_Start_0_31  = _mm512_set1_epi32(iStart_0_31);
      __m512i m512i_Start_32_63 = _mm512_set1_epi32(iStart_32_63);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start_0_31);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start_0_31);
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, m512i_Start_32_63);
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, m512i_Start_32_63);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo  = _mm512_unpacklo_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_0_31hi  = _mm512_unpackhi_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      uint16_t* AVS_RESTRICT dst_ptr     = dst + x + y_from * dst_pitch;
      const uint16_t* src_ptr_0_31  = src + iStart_0_31  + y_from * src_pitch;
      const uint16_t* src_ptr_32_63 = src + iStart_32_63 + y_from * src_pitch;

      const int remaining_0_31  = program->source_size - iStart_0_31;
      const __mmask32 k1_0_31   = _bzhi_u32(~0U, remaining_0_31);
      const __mmask32 k2_0_31   = _bzhi_u32(~0U, std::max(0, remaining_0_31 - 32));
      const int remaining_32_63 = program->source_size - iStart_32_63;
      const __mmask32 k1_32_63  = _bzhi_u32(~0U, remaining_32_63);
      const __mmask32 k2_32_63  = _bzhi_u32(~0U, std::max(0, remaining_32_63 - 32));

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src_0_31, data_src2_0_31, data_src_32_63, data_src2_32_63;

        __m512i perm_rNrNp1_0_31lo  = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi  = perm_r0r1_0_31hi;
        __m512i perm_rNrNp1_32_63lo = perm_r0r1_32_63lo;
        __m512i perm_rNrNp1_32_63hi = perm_r0r1_32_63hi;

        if constexpr (partial_load) {
          data_src_0_31   = _mm512_maskz_loadu_epi16(k1_0_31,  src_ptr_0_31);
          data_src2_0_31  = _mm512_maskz_loadu_epi16(k2_0_31,  src_ptr_0_31  + 32);
          data_src_32_63  = _mm512_maskz_loadu_epi16(k1_32_63, src_ptr_32_63);
          data_src2_32_63 = _mm512_maskz_loadu_epi16(k2_32_63, src_ptr_32_63 + 32);
        }
        else {
          data_src_0_31   = _mm512_loadu_si512(src_ptr_0_31);
          data_src2_0_31  = _mm512_loadu_si512(src_ptr_0_31  + 32);
          data_src_32_63  = _mm512_loadu_si512(src_ptr_32_63);
          data_src2_32_63 = _mm512_loadu_si512(src_ptr_32_63 + 32);
        }

        __m512i result_0_31lo, result_0_31hi, result_32_63lo, result_32_63hi;

        __m512i src_r0r1_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r0r1_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r0r1_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r0r1_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r0r1_0_31lo  = _mm512_add_epi16(src_r0r1_0_31lo,  shifttosigned);
          src_r0r1_0_31hi  = _mm512_add_epi16(src_r0r1_0_31hi,  shifttosigned);
          src_r0r1_32_63lo = _mm512_add_epi16(src_r0r1_32_63lo, shifttosigned);
          src_r0r1_32_63hi = _mm512_add_epi16(src_r0r1_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo,  coef_r0r1_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi,  coef_r0r1_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  coef_r0r1_0_31lo),  rounder);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  coef_r0r1_0_31hi),  rounder);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), rounder);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), rounder);
        }

        __m512i src_r2r3_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r2r3_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r2r3_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r2r3_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r2r3_0_31lo  = _mm512_add_epi16(src_r2r3_0_31lo,  shifttosigned);
          src_r2r3_0_31hi  = _mm512_add_epi16(src_r2r3_0_31hi,  shifttosigned);
          src_r2r3_32_63lo = _mm512_add_epi16(src_r2r3_32_63lo, shifttosigned);
          src_r2r3_32_63hi = _mm512_add_epi16(src_r2r3_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r2r3_0_31lo,  coef_r2r3_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r2r3_0_31hi,  coef_r2r3_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r2r3_0_31lo,  coef_r2r3_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r2r3_0_31hi,  coef_r2r3_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));
        }

        __m512i src_r4r5_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r4r5_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r4r5_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r4r5_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r4r5_0_31lo  = _mm512_add_epi16(src_r4r5_0_31lo,  shifttosigned);
          src_r4r5_0_31hi  = _mm512_add_epi16(src_r4r5_0_31hi,  shifttosigned);
          src_r4r5_32_63lo = _mm512_add_epi16(src_r4r5_32_63lo, shifttosigned);
          src_r4r5_32_63hi = _mm512_add_epi16(src_r4r5_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r4r5_0_31lo,  coef_r4r5_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r4r5_0_31hi,  coef_r4r5_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r4r5_32_63lo, coef_r4r5_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r4r5_32_63hi, coef_r4r5_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r4r5_0_31lo,  coef_r4r5_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r4r5_0_31hi,  coef_r4r5_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r4r5_32_63lo, coef_r4r5_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r4r5_32_63hi, coef_r4r5_32_63hi));
        }

        __m512i src_r6r7_0_31lo  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31lo,  data_src2_0_31);
        __m512i src_r6r7_0_31hi  = _mm512_permutex2var_epi16(data_src_0_31,  perm_rNrNp1_0_31hi,  data_src2_0_31);
        __m512i src_r6r7_32_63lo = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63lo, data_src2_32_63);
        __m512i src_r6r7_32_63hi = _mm512_permutex2var_epi16(data_src_32_63, perm_rNrNp1_32_63hi, data_src2_32_63);

        if constexpr (!lessthan16bit) {
          src_r6r7_0_31lo  = _mm512_add_epi16(src_r6r7_0_31lo,  shifttosigned);
          src_r6r7_0_31hi  = _mm512_add_epi16(src_r6r7_0_31hi,  shifttosigned);
          src_r6r7_32_63lo = _mm512_add_epi16(src_r6r7_32_63lo, shifttosigned);
          src_r6r7_32_63hi = _mm512_add_epi16(src_r6r7_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r6r7_0_31lo,  coef_r6r7_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r6r7_0_31hi,  coef_r6r7_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r6r7_32_63lo, coef_r6r7_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r6r7_32_63hi, coef_r6r7_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r6r7_0_31lo,  coef_r6r7_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r6r7_0_31hi,  coef_r6r7_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r6r7_32_63lo, coef_r6r7_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r6r7_32_63hi, coef_r6r7_32_63hi));
        }

        if constexpr (!lessthan16bit) {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  shiftfromsigned);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  shiftfromsigned);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, shiftfromsigned);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, shiftfromsigned);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale16bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale16bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale16bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale16bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        if constexpr (lessthan16bit) {
          result_0_31_int16  = _mm512_min_epu16(result_0_31_int16,  clamp_limit);
          result_32_63_int16 = _mm512_min_epu16(result_32_63_int16, clamp_limit);
        }

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), result_0_31_int16);

        const int remaining_32_63_out = width - (x + 32);
        if (remaining_32_63_out >= 32)
          _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr + 32), result_32_63_int16);
        else if (remaining_32_63_out > 0)
          _mm512_mask_storeu_epi16(dst_ptr + 32, _bzhi_u32(~0u, remaining_32_63_out), result_32_63_int16);

        dst_ptr       += dst_pitch;
        src_ptr_0_31  += src_pitch;
        src_ptr_32_63 += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

template<bool lessthan16bit, bool UseVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  constexpr int PIXELS_AT_A_TIME = 64;

  // Safe zone: four 64 pixel windows, from x, x+16, x+32, x+48. Each window start must be checked

  // (safelimit_64_pixels_each32th_target checked only the step start: the other windows could read past the source buffer).

  const int width_safe_mod = (program->safelimit_64_pixels_each16th_target.overread_possible ? program->safelimit_64_pixels_each16th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  const int max_scanlines = program->max_scanlines;

  const __m512i shifttosigned   = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit);
  __m512i rounder     = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r0r1_32_63lo = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r0r1_32_63hi = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r2r3_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r2r3_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r2r3_32_63lo = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r2r3_32_63hi = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r4r5_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r4r5_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r4r5_32_63lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r4r5_32_63hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r6r7_0_31lo  = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r6r7_0_31hi  = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r6r7_32_63lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r6r7_32_63hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart_0_15  = program->pixel_offset[x];
      int iStart_16_31 = program->pixel_offset[x + 16];
      int iStart_32_47 = program->pixel_offset[x + 32];
      int iStart_48_63 = program->pixel_offset[x + 48];

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  _mm512_set1_epi32(iStart_0_15));
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, _mm512_set1_epi32(iStart_16_31));
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, _mm512_set1_epi32(iStart_32_47));
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, _mm512_set1_epi32(iStart_48_63));

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo  = _mm512_unpacklo_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_0_31hi  = _mm512_unpackhi_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i two_epi16 = _mm512_set1_epi16(2);
      const __mmask32 k_high  = 0xFFFF0000;

      uint16_t* AVS_RESTRICT dst_ptr      = dst + x + y_from * dst_pitch;
      const uint16_t* src_ptr_0_15  = src + iStart_0_15  + y_from * src_pitch;
      const uint16_t* src_ptr_16_31 = src + iStart_16_31 + y_from * src_pitch;
      const uint16_t* src_ptr_32_47 = src + iStart_32_47 + y_from * src_pitch;
      const uint16_t* src_ptr_48_63 = src + iStart_48_63 + y_from * src_pitch;

      const int remaining_0_15  = program->source_size - iStart_0_15;
      const __mmask32 k1_0_15   = _bzhi_u32(~0U, remaining_0_15);
      const __mmask32 k2_0_15   = _bzhi_u32(~0U, std::max(0, remaining_0_15  - 32));
      const int remaining_16_31 = program->source_size - iStart_16_31;
      const __mmask32 k1_16_31  = _bzhi_u32(~0U, remaining_16_31);
      const __mmask32 k2_16_31  = _bzhi_u32(~0U, std::max(0, remaining_16_31 - 32));
      const int remaining_32_47 = program->source_size - iStart_32_47;
      const __mmask32 k1_32_47  = _bzhi_u32(~0U, remaining_32_47);
      const __mmask32 k2_32_47  = _bzhi_u32(~0U, std::max(0, remaining_32_47 - 32));
      const int remaining_48_63 = program->source_size - iStart_48_63;
      const __mmask32 k1_48_63  = _bzhi_u32(~0U, remaining_48_63);
      const __mmask32 k2_48_63  = _bzhi_u32(~0U, std::max(0, remaining_48_63 - 32));

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src_0_15,  data_src2_0_15;
        __m512i data_src_16_31, data_src2_16_31;
        __m512i data_src_32_47, data_src2_32_47;
        __m512i data_src_48_63, data_src2_48_63;

        __m512i perm_rNrNp1_0_31lo  = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi  = perm_r0r1_0_31hi;
        __m512i perm_rNrNp1_32_63lo = perm_r0r1_32_63lo;
        __m512i perm_rNrNp1_32_63hi = perm_r0r1_32_63hi;

        if constexpr (partial_load) {
          data_src_0_15   = _mm512_maskz_loadu_epi16(k1_0_15,  src_ptr_0_15);
          data_src_16_31  = _mm512_maskz_loadu_epi16(k1_16_31, src_ptr_16_31);
          data_src_32_47  = _mm512_maskz_loadu_epi16(k1_32_47, src_ptr_32_47);
          data_src_48_63  = _mm512_maskz_loadu_epi16(k1_48_63, src_ptr_48_63);
          data_src2_0_15  = _mm512_maskz_loadu_epi16(k2_0_15,  src_ptr_0_15  + 32);
          data_src2_16_31 = _mm512_maskz_loadu_epi16(k2_16_31, src_ptr_16_31 + 32);
          data_src2_32_47 = _mm512_maskz_loadu_epi16(k2_32_47, src_ptr_32_47 + 32);
          data_src2_48_63 = _mm512_maskz_loadu_epi16(k2_48_63, src_ptr_48_63 + 32);
        }
        else {
          data_src_0_15   = _mm512_loadu_si512(src_ptr_0_15);
          data_src_16_31  = _mm512_loadu_si512(src_ptr_16_31);
          data_src_32_47  = _mm512_loadu_si512(src_ptr_32_47);
          data_src_48_63  = _mm512_loadu_si512(src_ptr_48_63);
          data_src2_0_15  = _mm512_loadu_si512(src_ptr_0_15  + 32);
          data_src2_16_31 = _mm512_loadu_si512(src_ptr_16_31 + 32);
          data_src2_32_47 = _mm512_loadu_si512(src_ptr_32_47 + 32);
          data_src2_48_63 = _mm512_loadu_si512(src_ptr_48_63 + 32);
        }

        __m512i result_0_31lo, result_0_31hi, result_32_63lo, result_32_63hi;

        __m512i src_r0r1_0_31lo  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31lo,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31lo,  data_src2_16_31));
        __m512i src_r0r1_0_31hi  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31hi,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31hi,  data_src2_16_31));
        __m512i src_r0r1_32_63lo = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63lo, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63lo, data_src2_48_63));
        __m512i src_r0r1_32_63hi = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63hi, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63hi, data_src2_48_63));

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r0r1_0_31lo  = _mm512_add_epi16(src_r0r1_0_31lo,  shifttosigned);
          src_r0r1_0_31hi  = _mm512_add_epi16(src_r0r1_0_31hi,  shifttosigned);
          src_r0r1_32_63lo = _mm512_add_epi16(src_r0r1_32_63lo, shifttosigned);
          src_r0r1_32_63hi = _mm512_add_epi16(src_r0r1_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo,  coef_r0r1_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi,  coef_r0r1_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63lo, coef_r0r1_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(rounder, src_r0r1_32_63hi, coef_r0r1_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  coef_r0r1_0_31lo),  rounder);
          result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  coef_r0r1_0_31hi),  rounder);
          result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, coef_r0r1_32_63lo), rounder);
          result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, coef_r0r1_32_63hi), rounder);
        }

        __m512i src_r2r3_0_31lo  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31lo,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31lo,  data_src2_16_31));
        __m512i src_r2r3_0_31hi  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31hi,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31hi,  data_src2_16_31));
        __m512i src_r2r3_32_63lo = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63lo, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63lo, data_src2_48_63));
        __m512i src_r2r3_32_63hi = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63hi, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63hi, data_src2_48_63));

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r2r3_0_31lo  = _mm512_add_epi16(src_r2r3_0_31lo,  shifttosigned);
          src_r2r3_0_31hi  = _mm512_add_epi16(src_r2r3_0_31hi,  shifttosigned);
          src_r2r3_32_63lo = _mm512_add_epi16(src_r2r3_32_63lo, shifttosigned);
          src_r2r3_32_63hi = _mm512_add_epi16(src_r2r3_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r2r3_0_31lo,  coef_r2r3_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r2r3_0_31hi,  coef_r2r3_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r2r3_32_63lo, coef_r2r3_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r2r3_32_63hi, coef_r2r3_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r2r3_0_31lo,  coef_r2r3_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r2r3_0_31hi,  coef_r2r3_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r2r3_32_63lo, coef_r2r3_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r2r3_32_63hi, coef_r2r3_32_63hi));
        }

        __m512i src_r4r5_0_31lo  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31lo,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31lo,  data_src2_16_31));
        __m512i src_r4r5_0_31hi  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31hi,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31hi,  data_src2_16_31));
        __m512i src_r4r5_32_63lo = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63lo, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63lo, data_src2_48_63));
        __m512i src_r4r5_32_63hi = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63hi, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63hi, data_src2_48_63));

        perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
        perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
        perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
        perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r4r5_0_31lo  = _mm512_add_epi16(src_r4r5_0_31lo,  shifttosigned);
          src_r4r5_0_31hi  = _mm512_add_epi16(src_r4r5_0_31hi,  shifttosigned);
          src_r4r5_32_63lo = _mm512_add_epi16(src_r4r5_32_63lo, shifttosigned);
          src_r4r5_32_63hi = _mm512_add_epi16(src_r4r5_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r4r5_0_31lo,  coef_r4r5_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r4r5_0_31hi,  coef_r4r5_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r4r5_32_63lo, coef_r4r5_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r4r5_32_63hi, coef_r4r5_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r4r5_0_31lo,  coef_r4r5_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r4r5_0_31hi,  coef_r4r5_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r4r5_32_63lo, coef_r4r5_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r4r5_32_63hi, coef_r4r5_32_63hi));
        }

        __m512i src_r6r7_0_31lo  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31lo,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31lo,  data_src2_16_31));
        __m512i src_r6r7_0_31hi  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31hi,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31hi,  data_src2_16_31));
        __m512i src_r6r7_32_63lo = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63lo, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63lo, data_src2_48_63));
        __m512i src_r6r7_32_63hi = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63hi, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63hi, data_src2_48_63));

        if constexpr (!lessthan16bit) {
          src_r6r7_0_31lo  = _mm512_add_epi16(src_r6r7_0_31lo,  shifttosigned);
          src_r6r7_0_31hi  = _mm512_add_epi16(src_r6r7_0_31hi,  shifttosigned);
          src_r6r7_32_63lo = _mm512_add_epi16(src_r6r7_32_63lo, shifttosigned);
          src_r6r7_32_63hi = _mm512_add_epi16(src_r6r7_32_63hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r6r7_0_31lo,  coef_r6r7_0_31lo);
          result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r6r7_0_31hi,  coef_r6r7_0_31hi);
          result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r6r7_32_63lo, coef_r6r7_32_63lo);
          result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r6r7_32_63hi, coef_r6r7_32_63hi);
        }
        else
        {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  _mm512_madd_epi16(src_r6r7_0_31lo,  coef_r6r7_0_31lo));
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  _mm512_madd_epi16(src_r6r7_0_31hi,  coef_r6r7_0_31hi));
          result_32_63lo = _mm512_add_epi32(result_32_63lo, _mm512_madd_epi16(src_r6r7_32_63lo, coef_r6r7_32_63lo));
          result_32_63hi = _mm512_add_epi32(result_32_63hi, _mm512_madd_epi16(src_r6r7_32_63hi, coef_r6r7_32_63hi));
        }

        if constexpr (!lessthan16bit) {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  shiftfromsigned);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  shiftfromsigned);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, shiftfromsigned);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, shiftfromsigned);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale16bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale16bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale16bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale16bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        if constexpr (lessthan16bit) {
          result_0_31_int16  = _mm512_min_epu16(result_0_31_int16,  clamp_limit);
          result_32_63_int16 = _mm512_min_epu16(result_32_63_int16, clamp_limit);
        }

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), result_0_31_int16);

        const int remaining_32_63_out = width - (x + 32);
        if (remaining_32_63_out >= 32)
          _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr + 32), result_32_63_int16);
        else if (remaining_32_63_out > 0)
          _mm512_mask_storeu_epi16(dst_ptr + 32, _bzhi_u32(~0u, remaining_32_63_out), result_32_63_int16);

        dst_ptr       += dst_pitch;
        src_ptr_0_15  += src_pitch;
        src_ptr_16_31 += src_pitch;
        src_ptr_32_47 += src_pitch;
        src_ptr_48_63 += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

template<bool lessthan16bit, bool UseVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  constexpr int PIXELS_AT_A_TIME = 32;

  const int width_safe_mod = (program->safelimit_64_pixels_each32th_target.overread_possible ? program->safelimit_64_pixels_each32th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  const int max_scanlines = program->max_scanlines;

  const __m512i shifttosigned   = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit);
  __m512i rounder     = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      const __m512i coef_r0r1_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 0);
      const __m512i coef_r0r1_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 1);
      const __m512i coef_r2r3_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 2);
      const __m512i coef_r2r3_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 3);
      const __m512i coef_r4r5_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 4);
      const __m512i coef_r4r5_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 5);
      const __m512i coef_r6r7_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 6);
      const __m512i coef_r6r7_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 7);
      const __m512i coef_r8r9_0_31lo   = _mm512_load_si512(current_coeff_SIMD + 8);
      const __m512i coef_r8r9_0_31hi   = _mm512_load_si512(current_coeff_SIMD + 9);
      const __m512i coef_r10r11_0_31lo = _mm512_load_si512(current_coeff_SIMD + 10);
      const __m512i coef_r10r11_0_31hi = _mm512_load_si512(current_coeff_SIMD + 11);
      const __m512i coef_r12r13_0_31lo = _mm512_load_si512(current_coeff_SIMD + 12);
      const __m512i coef_r12r13_0_31hi = _mm512_load_si512(current_coeff_SIMD + 13);
      const __m512i coef_r14r15_0_31lo = _mm512_load_si512(current_coeff_SIMD + 14);
      const __m512i coef_r14r15_0_31hi = _mm512_load_si512(current_coeff_SIMD + 15);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));

      int iStart = program->pixel_offset[x];
      __m512i m512i_Start = _mm512_set1_epi32(iStart);

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  m512i_Start);
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, m512i_Start);

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);

      __m512i perm_0_0_31 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15), m256i_perm_0_16_31, 1);
      __m512i perm_1_0_31 = _mm512_add_epi16(perm_0_0_31, one_epi16);

      const __m512i perm_r0r1_0_31lo = _mm512_unpacklo_epi16(perm_0_0_31, perm_1_0_31);
      const __m512i perm_r0r1_0_31hi = _mm512_unpackhi_epi16(perm_0_0_31, perm_1_0_31);

      const __m512i two_epi16 = _mm512_set1_epi16(2);

      uint16_t* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const uint16_t* src_ptr = src + iStart + y_from * src_pitch;

      const int remaining = program->source_size - iStart;
      const __mmask32 k1 = _bzhi_u32(~0U, remaining);
      const __mmask32 k2 = _bzhi_u32(~0U, std::max(0, remaining - 32));

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src, data_src2;

        __m512i perm_rNrNp1_0_31lo_w = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi_w = perm_r0r1_0_31hi;

        if constexpr (partial_load) {
          data_src  = _mm512_maskz_loadu_epi16(k1, src_ptr);
          data_src2 = _mm512_maskz_loadu_epi16(k2, src_ptr + 32);
        }
        else {
          data_src  = _mm512_loadu_si512(src_ptr);
          data_src2 = _mm512_loadu_si512(src_ptr + 32);
        }

        __m512i src_r0r1_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r0r1_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r2r3_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r2r3_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i result_0_31lo, result_0_31hi;

        if constexpr (!lessthan16bit) {
          src_r0r1_0_31lo = _mm512_add_epi16(src_r0r1_0_31lo, shifttosigned);
          src_r0r1_0_31hi = _mm512_add_epi16(src_r0r1_0_31hi, shifttosigned);
          src_r2r3_0_31lo = _mm512_add_epi16(src_r2r3_0_31lo, shifttosigned);
          src_r2r3_0_31hi = _mm512_add_epi16(src_r2r3_0_31hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31lo, coef_r0r1_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r2r3_0_31lo, coef_r2r3_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(rounder, src_r0r1_0_31hi, coef_r0r1_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r2r3_0_31hi, coef_r2r3_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo, coef_r0r1_0_31lo), _mm512_madd_epi16(src_r2r3_0_31lo, coef_r2r3_0_31lo));
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi, coef_r0r1_0_31hi), _mm512_madd_epi16(src_r2r3_0_31hi, coef_r2r3_0_31hi));
        }

        __m512i src_r4r5_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r4r5_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r6r7_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r6r7_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r4r5_0_31lo = _mm512_add_epi16(src_r4r5_0_31lo, shifttosigned);
          src_r4r5_0_31hi = _mm512_add_epi16(src_r4r5_0_31hi, shifttosigned);
          src_r6r7_0_31lo = _mm512_add_epi16(src_r6r7_0_31lo, shifttosigned);
          src_r6r7_0_31hi = _mm512_add_epi16(src_r6r7_0_31hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r4r5_0_31lo, coef_r4r5_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r6r7_0_31lo, coef_r6r7_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r4r5_0_31hi, coef_r4r5_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r6r7_0_31hi, coef_r6r7_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31lo, coef_r4r5_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r4r5_0_31hi, coef_r4r5_0_31hi), result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31lo, coef_r6r7_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r6r7_0_31hi, coef_r6r7_0_31hi), result_0_31hi);
        }

        __m512i src_r8r9_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r8r9_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r10r11_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r10r11_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        if constexpr (!lessthan16bit) {
          src_r8r9_0_31lo   = _mm512_add_epi16(src_r8r9_0_31lo,   shifttosigned);
          src_r8r9_0_31hi   = _mm512_add_epi16(src_r8r9_0_31hi,   shifttosigned);
          src_r10r11_0_31lo = _mm512_add_epi16(src_r10r11_0_31lo, shifttosigned);
          src_r10r11_0_31hi = _mm512_add_epi16(src_r10r11_0_31hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r8r9_0_31lo,   coef_r8r9_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r10r11_0_31lo, coef_r10r11_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r8r9_0_31hi,   coef_r8r9_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r10r11_0_31hi, coef_r10r11_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r8r9_0_31lo,   coef_r8r9_0_31lo),   result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r8r9_0_31hi,   coef_r8r9_0_31hi),   result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r10r11_0_31lo, coef_r10r11_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r10r11_0_31hi, coef_r10r11_0_31hi), result_0_31hi);
        }

        __m512i src_r12r13_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r12r13_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        perm_rNrNp1_0_31lo_w = _mm512_add_epi16(perm_rNrNp1_0_31lo_w, two_epi16);
        perm_rNrNp1_0_31hi_w = _mm512_add_epi16(perm_rNrNp1_0_31hi_w, two_epi16);

        __m512i src_r14r15_0_31lo = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31lo_w, data_src2);
        __m512i src_r14r15_0_31hi = _mm512_permutex2var_epi16(data_src, perm_rNrNp1_0_31hi_w, data_src2);

        if constexpr (!lessthan16bit) {
          src_r12r13_0_31lo = _mm512_add_epi16(src_r12r13_0_31lo, shifttosigned);
          src_r12r13_0_31hi = _mm512_add_epi16(src_r12r13_0_31hi, shifttosigned);
          src_r14r15_0_31lo = _mm512_add_epi16(src_r14r15_0_31lo, shifttosigned);
          src_r14r15_0_31hi = _mm512_add_epi16(src_r14r15_0_31hi, shifttosigned);
        }

        if constexpr (UseVNNI)
        {
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r12r13_0_31lo, coef_r12r13_0_31lo);
          result_0_31lo = _mm512_dpwssd_epi32(result_0_31lo, src_r14r15_0_31lo, coef_r14r15_0_31lo);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r12r13_0_31hi, coef_r12r13_0_31hi);
          result_0_31hi = _mm512_dpwssd_epi32(result_0_31hi, src_r14r15_0_31hi, coef_r14r15_0_31hi);
        }
        else
        {
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r12r13_0_31lo, coef_r12r13_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r12r13_0_31hi, coef_r12r13_0_31hi), result_0_31hi);
          result_0_31lo = _mm512_add_epi32(_mm512_madd_epi16(src_r14r15_0_31lo, coef_r14r15_0_31lo), result_0_31lo);
          result_0_31hi = _mm512_add_epi32(_mm512_madd_epi16(src_r14r15_0_31hi, coef_r14r15_0_31hi), result_0_31hi);

          result_0_31lo = _mm512_add_epi32(result_0_31lo, rounder);
          result_0_31hi = _mm512_add_epi32(result_0_31hi, rounder);
        }

        if constexpr (!lessthan16bit) {
          result_0_31lo = _mm512_add_epi32(result_0_31lo, shiftfromsigned);
          result_0_31hi = _mm512_add_epi32(result_0_31hi, shiftfromsigned);
        }

        result_0_31lo = _mm512_srai_epi32(result_0_31lo, FPScale16bits);
        result_0_31hi = _mm512_srai_epi32(result_0_31hi, FPScale16bits);

        __m512i result_0_31_int16 = _mm512_packus_epi32(result_0_31lo, result_0_31hi);

        if constexpr (lessthan16bit) {
          result_0_31_int16 = _mm512_min_epu16(result_0_31_int16, clamp_limit);
        }

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), result_0_31_int16);

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff_SIMD += 16;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}

template<bool lessthan16bit, bool UseVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48_pretransposed_coeffs_internal(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  int filter_size_real = program->filter_size_real;
  if ((filter_size_real / 2 * 2) != filter_size_real) filter_size_real++;

  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  constexpr int PIXELS_AT_A_TIME = 64;

  // Safe zone: four 64 pixel windows, from x, x+16, x+32, x+48. Each window start must be checked

  // (safelimit_64_pixels_each32th_target checked only the step start: the other windows could read past the source buffer).

  const int width_safe_mod = (program->safelimit_64_pixels_each16th_target.overread_possible ? program->safelimit_64_pixels_each16th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  const int max_scanlines = program->max_scanlines;

  const __m512i shifttosigned   = _mm512_set1_epi16(-32768);
  const __m512i shiftfromsigned = _mm512_set1_epi32(32768 << FPScale16bits);

  const int limit = (1 << bits_per_pixel) - 1;
  __m512i clamp_limit = _mm512_set1_epi16((short)limit);
  __m512i rounder     = _mm512_set1_epi32(1 << (FPScale16bits - 1));

  for (int y_from = 0; y_from < height; y_from += max_scanlines)
  {
    int y_to = std::min(y_from + max_scanlines, height);

    const __m512i* AVS_RESTRICT current_coeff_SIMD = (__m512i*)program->pixel_coefficient_AVX512_H;

    int x = 0;

    auto do_h_integer_core = [&](auto partial_load) {
      __m512i one_epi16 = _mm512_set1_epi16(1);

      __m512i perm_0_0_15  = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x]));
      __m512i perm_0_16_31 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 16]));
      __m512i perm_0_32_47 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 32]));
      __m512i perm_0_48_63 = _mm512_loadu_si512((__m512i*)(&program->pixel_offset[x + 48]));

      int iStart_0_15  = program->pixel_offset[x];
      int iStart_16_31 = program->pixel_offset[x + 16];
      int iStart_32_47 = program->pixel_offset[x + 32];
      int iStart_48_63 = program->pixel_offset[x + 48];

      perm_0_0_15  = _mm512_sub_epi32(perm_0_0_15,  _mm512_set1_epi32(iStart_0_15));
      perm_0_16_31 = _mm512_sub_epi32(perm_0_16_31, _mm512_set1_epi32(iStart_16_31));
      perm_0_32_47 = _mm512_sub_epi32(perm_0_32_47, _mm512_set1_epi32(iStart_32_47));
      perm_0_48_63 = _mm512_sub_epi32(perm_0_48_63, _mm512_set1_epi32(iStart_48_63));

      __m256i m256i_perm_0_0_15  = _mm512_cvtepi32_epi16(perm_0_0_15);
      __m256i m256i_perm_0_16_31 = _mm512_cvtepi32_epi16(perm_0_16_31);
      __m256i m256i_perm_0_32_47 = _mm512_cvtepi32_epi16(perm_0_32_47);
      __m256i m256i_perm_0_48_63 = _mm512_cvtepi32_epi16(perm_0_48_63);

      __m512i perm_0_0_31  = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_0_15),  m256i_perm_0_16_31, 1);
      __m512i perm_0_32_63 = _mm512_inserti64x4(_mm512_castsi256_si512(m256i_perm_0_32_47), m256i_perm_0_48_63, 1);

      __m512i perm_1_0_31  = _mm512_add_epi16(perm_0_0_31,  one_epi16);
      __m512i perm_1_32_63 = _mm512_add_epi16(perm_0_32_63, one_epi16);

      const __m512i perm_r0r1_0_31lo  = _mm512_unpacklo_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_0_31hi  = _mm512_unpackhi_epi16(perm_0_0_31,  perm_1_0_31);
      const __m512i perm_r0r1_32_63lo = _mm512_unpacklo_epi16(perm_0_32_63, perm_1_32_63);
      const __m512i perm_r0r1_32_63hi = _mm512_unpackhi_epi16(perm_0_32_63, perm_1_32_63);

      const __m512i two_epi16 = _mm512_set1_epi16(2);
      const __mmask32 k_high  = 0xFFFF0000;

      uint16_t* AVS_RESTRICT dst_ptr      = dst + x + y_from * dst_pitch;
      const uint16_t* src_ptr_0_15  = src + iStart_0_15  + y_from * src_pitch;
      const uint16_t* src_ptr_16_31 = src + iStart_16_31 + y_from * src_pitch;
      const uint16_t* src_ptr_32_47 = src + iStart_32_47 + y_from * src_pitch;
      const uint16_t* src_ptr_48_63 = src + iStart_48_63 + y_from * src_pitch;

      const int remaining_0_15  = program->source_size - iStart_0_15;
      const __mmask32 k1_0_15   = _bzhi_u32(~0U, remaining_0_15);
      const __mmask32 k2_0_15   = _bzhi_u32(~0U, std::max(0, remaining_0_15  - 32));
      const int remaining_16_31 = program->source_size - iStart_16_31;
      const __mmask32 k1_16_31  = _bzhi_u32(~0U, remaining_16_31);
      const __mmask32 k2_16_31  = _bzhi_u32(~0U, std::max(0, remaining_16_31 - 32));
      const int remaining_32_47 = program->source_size - iStart_32_47;
      const __mmask32 k1_32_47  = _bzhi_u32(~0U, remaining_32_47);
      const __mmask32 k2_32_47  = _bzhi_u32(~0U, std::max(0, remaining_32_47 - 32));
      const int remaining_48_63 = program->source_size - iStart_48_63;
      const __mmask32 k1_48_63  = _bzhi_u32(~0U, remaining_48_63);
      const __mmask32 k2_48_63  = _bzhi_u32(~0U, std::max(0, remaining_48_63 - 32));

      for (int y = y_from; y < y_to; y++)
      {
        __m512i data_src_0_15,  data_src2_0_15;
        __m512i data_src_16_31, data_src2_16_31;
        __m512i data_src_32_47, data_src2_32_47;
        __m512i data_src_48_63, data_src2_48_63;

        __m512i perm_rNrNp1_0_31lo  = perm_r0r1_0_31lo;
        __m512i perm_rNrNp1_0_31hi  = perm_r0r1_0_31hi;
        __m512i perm_rNrNp1_32_63lo = perm_r0r1_32_63lo;
        __m512i perm_rNrNp1_32_63hi = perm_r0r1_32_63hi;

        if constexpr (partial_load) {
          data_src_0_15   = _mm512_maskz_loadu_epi16(k1_0_15,  src_ptr_0_15);
          data_src_16_31  = _mm512_maskz_loadu_epi16(k1_16_31, src_ptr_16_31);
          data_src_32_47  = _mm512_maskz_loadu_epi16(k1_32_47, src_ptr_32_47);
          data_src_48_63  = _mm512_maskz_loadu_epi16(k1_48_63, src_ptr_48_63);
          data_src2_0_15  = _mm512_maskz_loadu_epi16(k2_0_15,  src_ptr_0_15  + 32);
          data_src2_16_31 = _mm512_maskz_loadu_epi16(k2_16_31, src_ptr_16_31 + 32);
          data_src2_32_47 = _mm512_maskz_loadu_epi16(k2_32_47, src_ptr_32_47 + 32);
          data_src2_48_63 = _mm512_maskz_loadu_epi16(k2_48_63, src_ptr_48_63 + 32);
        }
        else {
          data_src_0_15   = _mm512_loadu_si512(src_ptr_0_15);
          data_src_16_31  = _mm512_loadu_si512(src_ptr_16_31);
          data_src_32_47  = _mm512_loadu_si512(src_ptr_32_47);
          data_src_48_63  = _mm512_loadu_si512(src_ptr_48_63);
          data_src2_0_15  = _mm512_loadu_si512(src_ptr_0_15  + 32);
          data_src2_16_31 = _mm512_loadu_si512(src_ptr_16_31 + 32);
          data_src2_32_47 = _mm512_loadu_si512(src_ptr_32_47 + 32);
          data_src2_48_63 = _mm512_loadu_si512(src_ptr_48_63 + 32);
        }

        __m512i result_0_31lo  = rounder;
        __m512i result_0_31hi  = rounder;
        __m512i result_32_63lo = rounder;
        __m512i result_32_63hi = rounder;

        const __m512i* current_coeff_SIMDw = current_coeff_SIMD;

        for (int kr = 0; kr < filter_size_real; kr += 2)
        {
          __m512i src_r0r1_0_31lo  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31lo,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31lo,  data_src2_16_31));
          __m512i src_r0r1_0_31hi  = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_0_15,  perm_rNrNp1_0_31hi,  data_src2_0_15),  _mm512_permutex2var_epi16(data_src_16_31, perm_rNrNp1_0_31hi,  data_src2_16_31));
          __m512i src_r0r1_32_63lo = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63lo, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63lo, data_src2_48_63));
          __m512i src_r0r1_32_63hi = _mm512_mask_blend_epi16(k_high, _mm512_permutex2var_epi16(data_src_32_47, perm_rNrNp1_32_63hi, data_src2_32_47), _mm512_permutex2var_epi16(data_src_48_63, perm_rNrNp1_32_63hi, data_src2_48_63));

          perm_rNrNp1_0_31lo  = _mm512_add_epi16(perm_rNrNp1_0_31lo,  two_epi16);
          perm_rNrNp1_0_31hi  = _mm512_add_epi16(perm_rNrNp1_0_31hi,  two_epi16);
          perm_rNrNp1_32_63lo = _mm512_add_epi16(perm_rNrNp1_32_63lo, two_epi16);
          perm_rNrNp1_32_63hi = _mm512_add_epi16(perm_rNrNp1_32_63hi, two_epi16);

          if constexpr (!lessthan16bit) {
            src_r0r1_0_31lo  = _mm512_add_epi16(src_r0r1_0_31lo,  shifttosigned);
            src_r0r1_0_31hi  = _mm512_add_epi16(src_r0r1_0_31hi,  shifttosigned);
            src_r0r1_32_63lo = _mm512_add_epi16(src_r0r1_32_63lo, shifttosigned);
            src_r0r1_32_63hi = _mm512_add_epi16(src_r0r1_32_63hi, shifttosigned);
          }

          if constexpr (UseVNNI)
          {
            result_0_31lo  = _mm512_dpwssd_epi32(result_0_31lo,  src_r0r1_0_31lo,  _mm512_load_si512(current_coeff_SIMDw + 0));
            result_0_31hi  = _mm512_dpwssd_epi32(result_0_31hi,  src_r0r1_0_31hi,  _mm512_load_si512(current_coeff_SIMDw + 1));
            result_32_63lo = _mm512_dpwssd_epi32(result_32_63lo, src_r0r1_32_63lo, _mm512_load_si512(current_coeff_SIMDw + 2));
            result_32_63hi = _mm512_dpwssd_epi32(result_32_63hi, src_r0r1_32_63hi, _mm512_load_si512(current_coeff_SIMDw + 3));
          }
          else
          {
            result_0_31lo  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31lo,  _mm512_load_si512(current_coeff_SIMDw + 0)), result_0_31lo);
            result_0_31hi  = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_0_31hi,  _mm512_load_si512(current_coeff_SIMDw + 1)), result_0_31hi);
            result_32_63lo = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63lo, _mm512_load_si512(current_coeff_SIMDw + 2)), result_32_63lo);
            result_32_63hi = _mm512_add_epi32(_mm512_madd_epi16(src_r0r1_32_63hi, _mm512_load_si512(current_coeff_SIMDw + 3)), result_32_63hi);
          }

          current_coeff_SIMDw += 4;
        }

        if constexpr (!lessthan16bit) {
          result_0_31lo  = _mm512_add_epi32(result_0_31lo,  shiftfromsigned);
          result_0_31hi  = _mm512_add_epi32(result_0_31hi,  shiftfromsigned);
          result_32_63lo = _mm512_add_epi32(result_32_63lo, shiftfromsigned);
          result_32_63hi = _mm512_add_epi32(result_32_63hi, shiftfromsigned);
        }

        result_0_31lo  = _mm512_srai_epi32(result_0_31lo,  FPScale16bits);
        result_0_31hi  = _mm512_srai_epi32(result_0_31hi,  FPScale16bits);
        result_32_63lo = _mm512_srai_epi32(result_32_63lo, FPScale16bits);
        result_32_63hi = _mm512_srai_epi32(result_32_63hi, FPScale16bits);

        __m512i result_0_31_int16  = _mm512_packus_epi32(result_0_31lo,  result_0_31hi);
        __m512i result_32_63_int16 = _mm512_packus_epi32(result_32_63lo, result_32_63hi);

        if constexpr (lessthan16bit) {
          result_0_31_int16  = _mm512_min_epu16(result_0_31_int16,  clamp_limit);
          result_32_63_int16 = _mm512_min_epu16(result_32_63_int16, clamp_limit);
        }

        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr), result_0_31_int16);

        const int remaining_32_63_out = width - (x + 32);
        if (remaining_32_63_out >= 32)
          _mm512_stream_si512(reinterpret_cast<__m512i*>(dst_ptr + 32), result_32_63_int16);
        else if (remaining_32_63_out > 0)
          _mm512_mask_storeu_epi16(dst_ptr + 32, _bzhi_u32(~0u, remaining_32_63_out), result_32_63_int16);

        dst_ptr       += dst_pitch;
        src_ptr_0_15  += src_pitch;
        src_ptr_16_31 += src_pitch;
        src_ptr_32_47 += src_pitch;
        src_ptr_48_63 += src_pitch;
      }

      current_coeff_SIMD += filter_size_real * 2;
    };

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::false_type{});
    }

    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_integer_core(std::true_type{});
    }
  }
}
