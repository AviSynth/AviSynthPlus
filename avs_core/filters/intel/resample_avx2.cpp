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

// Intrinsics base header + really required extension headers
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else 
#include <x86intrin.h> // GCC/MinGW/Clang/LLVM
#endif

#include <vector>
#include <avisynth.h>
#include <avs/config.h>
#include <avs/types.h>
#include <cstdint>
#include "../resample_functions.h"
#include <cassert>
#include <type_traits>
#include <algorithm>

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER) && ! defined (__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined (__FMA__) && (defined(__GNUC__) || defined(__clang__))  && ! defined (__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__


#include "resample_avx2.h"

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

// Useful dual-source permute helper functions for AVX2

// Dual source float-granlarity lookup, like _mm512_permutex2var_ps-
// _mm512_permutex2var_ps: two __m512 sources, index 0..31, zeroing bit handled
// avx2_permutex2var_ps: two __m256 sources, index 0..15, no zeroing bit simulated
static AVS_FORCEINLINE __m256 avx2_permutex2var_ps(__m256 a, __m256 b, __m256i idx) {
  // 1. Permute from both sources independently
  __m256 res_a = _mm256_permutevar8x32_ps(a, idx);
  __m256 res_b = _mm256_permutevar8x32_ps(b, idx);

  // 2. Use the 4th bit of the index (value 8) to select between A and B
  // In AVX2, we use _mm256_blendv_ps
  // blendv uses the sign bit (bit 31) of each float, so we shift bit 3 (value 8) to bit 31
  __m256i select_mask = _mm256_slli_epi32(idx, 28); // Move bit 3 (value 8) to sign bit
  return _mm256_blendv_ps(res_a, res_b, _mm256_castsi256_ps(select_mask));
}

// Dual source byte-granularity lookup, like _mm512_permutex2var_epi8.
// _mm512_permutex2var_epi8: two __m512i sources, index 0..127, zeroing bit handled
// avx2_permutex2var_epi8: two __m256i sources, index 0..63, no zeroing bit simulated
static AVS_FORCEINLINE __m256i avx2_permutex2var_epi8(__m256i a, __m256i b, __m256i idx) {
  //  idx(0-63) Source Lane(0/1) Byte offset
  //    0–15     a         0        0–15
  //    16–31    a         1        0–15
  //    32–47    b         0        0–15
  //    48–63    b         1        0–15

  // 1. Prepare swapped versions of A and B (Swap 128-bit lanes)
  __m256i a_swapped = _mm256_permute2x128_si256(a, a, 0x01);
  __m256i b_swapped = _mm256_permute2x128_si256(b, b, 0x01);

  // 2. Lookup from A (Lanes 0 and 1)
  // idx & 16: if 0, use original lane; if 1, use swapped lane
  __m256i shuf_a_norm = _mm256_shuffle_epi8(a, idx);
  __m256i shuf_a_swap = _mm256_shuffle_epi8(a_swapped, idx);
  __m256i res_a = _mm256_blendv_epi8(shuf_a_norm, shuf_a_swap, _mm256_slli_epi32(idx, 3));
  // Note: blendv uses the top bit, so we shift bit 4 (16) to bit 7 (128)

  // 3. Lookup from B (Lanes 2 and 3)
  __m256i idx_b = _mm256_sub_epi8(idx, _mm256_set1_epi8(32));
  __m256i shuf_b_norm = _mm256_shuffle_epi8(b, idx_b);
  __m256i shuf_b_swap = _mm256_shuffle_epi8(b_swapped, idx_b);
  __m256i res_b = _mm256_blendv_epi8(shuf_b_norm, shuf_b_swap, _mm256_slli_epi32(idx_b, 3));

  // 4. Final Blend: Decide between Result A or Result B based on Bit 5 (value 32)
  return _mm256_blendv_epi8(res_a, res_b, _mm256_slli_epi32(idx, 2));
}

//-------- AVX2 Horizontals

// Dual line processing (speed gain): 2x16 pixels of two consecutive offset entries.
// Use aligned filtersize template until alignment and end conditions allow.
// Aligned case uses full 16 pix/coeffs in one cycle.
// Unsafe part starts with 16 pix/coeffs until safe, then 8, 4, 1.
// Basically the only difference between 8 and 10-16 bit is the load and store.
// Processing 8 bit pixels even has overhead:
// - need upconverting to 16 bit short on load
// - extra step when narrowing end results further down to 8 bits.
// When processing uint16_t, the exact 16 bit size needs an unsigned -> signed 16 bit conversion
// because multiple and add (madd) works in the signed 16 bit domain.

template<typename pixel_t, bool lessthan16bit, int filtersizealigned16>
AVS_FORCEINLINE static void process_two_16pixels_h_uint8_16_core(const pixel_t* AVS_RESTRICT src, int begin1, int begin2, int i, const short* AVS_RESTRICT current_coeff, int filter_size, __m256i& result1, __m256i& result2,
  __m256i& shifttosigned) {
  filter_size = (filtersizealigned16 >= 1) ? filtersizealigned16 * 16 : filter_size;
  // knowing a quasi-constexpr filter_size from template for commonly used sizes
  // aligned_filter_size 16, 32, 48, 64, hugely helps compiler optimization

  __m256i data_1, data_2;

  if constexpr (sizeof(pixel_t) == 1) {
    // pixel_t is uint8_t
    data_1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i)));
    data_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i)));
  }
  else {
    // pixel_t is uint16_t, at exact 16 bit size an unsigned -> signed 16 bit conversion needed
    data_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin1 + i));
    if constexpr (!lessthan16bit)
      data_1 = _mm256_add_epi16(data_1, shifttosigned); // unsigned -> signed
    data_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + begin2 + i));
    if constexpr (!lessthan16bit)
      data_2 = _mm256_add_epi16(data_2, shifttosigned); // unsigned -> signed
  }
  __m256i coeff_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff)); // 16 coeffs
  __m256i coeff_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(current_coeff + 1 * filter_size)); // 16x second pixel's coefficients
  result1 = _mm256_add_epi32(result1, _mm256_madd_epi16(data_1, coeff_1));
  result2 = _mm256_add_epi32(result2, _mm256_madd_epi16(data_2, coeff_2));
}

// filtersizealigned16: special: 1..4. Generic: -1
template<bool safe_aligned_mode, typename pixel_t, bool lessthan16bit, int filtersizealigned16>
AVS_FORCEINLINE static void process_two_pixels_h_uint8_16(const pixel_t* AVS_RESTRICT src_ptr, int begin1, int begin2, const short* AVS_RESTRICT current_coeff, int filter_size, __m256i& result1, __m256i& result2, int kernel_size,
  __m256i& shifttosigned) {

  filter_size = (filtersizealigned16 >= 1) ? filtersizealigned16 * 16 : filter_size;
  // knowing a quasi-constexpr filter_size from template for commonly used sizes
  // aligned_filter_size 16, 32, 48, 64, hugely helps compiler optimization

  int ksmod16;
  if constexpr (safe_aligned_mode)
    ksmod16 = filter_size / 16 * 16;
  else
    ksmod16 = kernel_size / 16 * 16; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const pixel_t* src_ptr1 = src_ptr + begin1;
  const pixel_t* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 16 elements at a time
  for (; i < ksmod16; i += 16) {
    process_two_16pixels_h_uint8_16_core<pixel_t, lessthan16bit, filtersizealigned16>(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2, shifttosigned);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    const short* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod8 = kernel_size / 8 * 8;
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 8 elements if needed
    if (i < ksmod8) {
      // Process 8 elements for first pixel
      __m128i data_1;
      if constexpr (sizeof(pixel_t) == 1)
        data_1 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i)));
      else {
        // uint16_t
        data_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr1 + i));
        if constexpr (!lessthan16bit)
          data_1 = _mm_add_epi16(data_1, _mm256_castsi256_si128(shifttosigned)); // unsigned -> signed
      }

      __m128i coeff_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 8 elements for second pixel
      __m128i data_2;
      if constexpr (sizeof(pixel_t) == 1)
        data_2 = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i)));
      else {
        data_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr2 + i));
        if constexpr (!lessthan16bit)
          data_2 = _mm_add_epi16(data_2, _mm256_castsi256_si128(shifttosigned)); // unsigned -> signed
      }
      __m128i coeff_2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      __m256i temp1 = _mm256_setzero_si256();
      __m256i temp2 = _mm256_setzero_si256();
      temp1 = _mm256_insertf128_si256(temp1, temp_result1, 0);
      temp2 = _mm256_insertf128_si256(temp2, temp_result2, 0);
      result1 = _mm256_add_epi32(result1, temp1);
      result2 = _mm256_add_epi32(result2, temp2);

      i += 8;
      if (i == kernel_size) return;
    }

    // Process 4 elements if needed
    if (i < ksmod4) {
      // Process 4 elements for first pixel
      __m128i data_1;
      if constexpr (sizeof(pixel_t) == 1)
        data_1= _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr1 + i)));
      else {
        data_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr1 + i));
        if constexpr (!lessthan16bit)
          data_1 = _mm_add_epi16(data_1, _mm256_castsi256_si128(shifttosigned)); // unsigned -> signed
      }
      __m128i coeff_1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff + i));
      __m128i temp_result1 = _mm_madd_epi16(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128i data_2;
      if constexpr (sizeof(pixel_t) == 1)
        data_2 = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*reinterpret_cast<const int*>(src_ptr2 + i)));
      else {
        data_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr2 + i));
        if constexpr (!lessthan16bit)
          data_2 = _mm_add_epi16(data_2, _mm256_castsi256_si128(shifttosigned)); // unsigned -> signed
      }
      __m128i coeff_2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(current_coeff2 + i));
      __m128i temp_result2 = _mm_madd_epi16(data_2, coeff_2);

      // update result vectors
      __m256i temp1 = _mm256_setzero_si256();
      __m256i temp2 = _mm256_setzero_si256();
      temp1 = _mm256_insertf128_si256(temp1, temp_result1, 0);
      temp2 = _mm256_insertf128_si256(temp2, temp_result2, 0);
      result1 = _mm256_add_epi32(result1, temp1);
      result2 = _mm256_add_epi32(result2, temp2);

      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining elements with scalar operations
    if (i < kernel_size) {
      int scalar_sum1[4] = { 0, 0, 0, 0 }; // like an __m128i
      int scalar_sum2[4] = { 0, 0, 0, 0 };


      for (; i < kernel_size; i++) {
        if constexpr (sizeof(pixel_t) == 1) {
          scalar_sum1[i % 4] += src_ptr1[i] * current_coeff[i];
          scalar_sum2[i % 4] += src_ptr2[i] * current_coeff2[i];
        }
        else {
          uint16_t pix1 = src_ptr1[i];
          uint16_t pix2 = src_ptr2[i];

          if constexpr (!lessthan16bit) {
            pix1 -= 32768;
            pix2 -= 32768;
          }

          scalar_sum1[i % 4] += (short)pix1 * current_coeff[i];
          scalar_sum2[i % 4] += (short)pix2 * current_coeff2[i];
        }
      }

      // Convert scalar results to SIMD and add to result vectors
      __m128i temp_result1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum1));
      __m128i temp_result2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(scalar_sum2));

      __m256i temp1 = _mm256_setzero_si256();
      __m256i temp2 = _mm256_setzero_si256();
      temp1 = _mm256_insertf128_si256(temp1, temp_result1, 0);
      temp2 = _mm256_insertf128_si256(temp2, temp_result2, 0);
      result1 = _mm256_add_epi32(result1, temp1);
      result2 = _mm256_add_epi32(result2, temp2);
    }
  }
}

// filtersizealigned16: special: 1..4. Generic: -1
template<bool is_safe, typename pixel_t, bool lessthan16bit, int filtersizealigned16>
AVS_FORCEINLINE static void process_eight_pixels_h_uint8_16(const pixel_t* src, int x, const short* current_coeff_base, int filter_size,
  __m256i& rounder256, __m256i& shifttosigned, __m128i& clamp_limit,
  pixel_t* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 16); // code assumes this

  filter_size = (filtersizealigned16 >= 1) ? filtersizealigned16 * 16 : filter_size;
  // knowing a quasi-constexpr filter_size from template for commonly used sizes
  // aligned_filter_size 16, 32, 48, 64, hugely helps compiler optimization

  const short* AVS_RESTRICT current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m256i result0 = rounder256;
  __m256i result1 = rounder256;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit, filtersizealigned16>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;
  __m256i sumQuad12 = _mm256_hadd_epi32(result0, result1);

  // 2 & 3
  result0 = rounder256;
  result1 = rounder256;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit, filtersizealigned16>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;
  __m256i sumQuad1234 = _mm256_hadd_epi32(sumQuad12, _mm256_hadd_epi32(result0, result1));

  // 4 & 5
  result0 = rounder256;
  result1 = rounder256;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit, filtersizealigned16>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  current_coeff += 2 * filter_size;
  __m256i sumQuad56 = _mm256_hadd_epi32(result0, result1);

  // 6 & 7
  result0 = rounder256;
  result1 = rounder256;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_uint8_16<is_safe, pixel_t, lessthan16bit, filtersizealigned16>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size, shifttosigned);
  //current_coeff += 2 * filter_size;
  __m256i sumQuad5678 = _mm256_hadd_epi32(sumQuad56, _mm256_hadd_epi32(result0, result1));

  __m128i pix1234 = _mm_add_epi32(_mm256_extractf128_si256(sumQuad1234, 0), _mm256_extractf128_si256(sumQuad1234, 1));
  __m128i pix5678 = _mm_add_epi32(_mm256_extractf128_si256(sumQuad5678, 0), _mm256_extractf128_si256(sumQuad5678, 1));
  __m256i result_8x_uint32 = _mm256_set_m128i(pix5678, pix1234);

  // correct if signed, scale back, store
  if constexpr (sizeof(pixel_t) == 2 && !lessthan16bit) {
    const __m256i shiftfromsigned = _mm256_set1_epi32(+32768 << FPScale16bits); // yes, 32 bit data. for 16 bits only
    result_8x_uint32 = _mm256_add_epi32(result_8x_uint32, shiftfromsigned);
  }

  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;

  // scale back, shuffle, store
  __m256i result = _mm256_srai_epi32(result_8x_uint32, current_fp_scale_bits);
  __m256i result_2x4x_uint16 = _mm256_packus_epi32(result, result /* n/a */);
  __m128i result_2x4x_uint16_128 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(result_2x4x_uint16, (0 << 0) | (2 << 2) | (0 << 4) | (0 << 6)));
  if constexpr (sizeof(pixel_t) == 2 && lessthan16bit)
    result_2x4x_uint16_128 = _mm_min_epu16(result_2x4x_uint16_128, clamp_limit); // extra clamp for 10-14 bits

  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result_2x4x_uint8 = _mm_packus_epi16(result_2x4x_uint16_128, _mm_setzero_si128());
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint8);
  }
  else {
    _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x), result_2x4x_uint16_128);
  }
}

// filtersizealigned16: special: 1..4. Generic: -1
template<typename pixel_t, bool lessthan16bit, int filtersizealigned16>
static void internal_resizer_h_avx2_generic_uint8_16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = (filtersizealigned16 >= 1) ? filtersizealigned16 * 16 : program->filter_size;
  // knowing a quasi-constexpr filter_size from template for commonly used sizes
  // aligned_filter_size 16, 32, 48, 64, hugely helps compiler optimization

  __m256i shifttosigned = _mm256_set1_epi16(-32768); // for 16 bits only

  const int current_fp_scale_bits = (sizeof(pixel_t) == 1) ? FPScale8bits : FPScale16bits;
  __m256i rounder256 = _mm256_setr_epi32(1 << (current_fp_scale_bits - 1), 0, 0, 0, 0, 0, 0, 0);

  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for 8< <16 bits

  const pixel_t* AVS_RESTRICT src = reinterpret_cast<const pixel_t*>(src8);
  pixel_t* AVS_RESTRICT dst = reinterpret_cast<pixel_t*>(dst8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);

  const int w_safe_mod8 = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    const short* AVS_RESTRICT current_coeff_base = program->pixel_coefficient;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_uint8_16<true, pixel_t, lessthan16bit, filtersizealigned16>(src, x, current_coeff_base, filter_size, rounder256, shifttosigned, clamp_limit, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_uint8_16<false, pixel_t, lessthan16bit, filtersizealigned16>(src, x, current_coeff_base, filter_size, rounder256, shifttosigned, clamp_limit, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// coeffs are safely padded/aligned to 16

// 8 bit Horizontal

void resizer_h_avx2_generic_uint8_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  const int filter_size = program->filter_size;
  assert(program->filter_size_alignment == 16);

  if (filter_size == 16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint8_t, true, 1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 2*16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint8_t, true, 2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 3*16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint8_t, true, 3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 4*16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint8_t, true, 4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_avx2_generic_uint8_16_t<uint8_t, true, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// 16 bit Horizontal

template<bool lessthan16bit>
void resizer_h_avx2_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = program->filter_size;
  assert(program->filter_size_alignment == 16);

  if (filter_size== 16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint16_t, lessthan16bit, 1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 2 * 16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint16_t, lessthan16bit, 2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 3 * 16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint16_t, lessthan16bit, 3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 4 * 16)
    internal_resizer_h_avx2_generic_uint8_16_t<uint16_t, lessthan16bit, 4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_avx2_generic_uint8_16_t<uint16_t, lessthan16bit, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// AVX2 Horizontal float

// 2x8 pixels of two consecutive offset entries.
AVS_FORCEINLINE static void process_pix2_coeff8_h_float_core(const float* src, int begin1, int begin2, int i, float* current_coeff, int filter_size, __m256& result1, __m256& result2) {
  __m256 data_1 = _mm256_loadu_ps(src + begin1 + i);
  __m256 data_2 = _mm256_loadu_ps(src + begin2 + i);
  __m256 coeff_1 = _mm256_load_ps(current_coeff); // 8 coeffs
  __m256 coeff_2 = _mm256_load_ps(current_coeff + 1 * filter_size); // 8x second pixel's coefficients
  result1 = _mm256_fmadd_ps(data_1, coeff_1, result1); // a*b + c
  result2 = _mm256_fmadd_ps(data_2, coeff_2, result2);
}

template<bool safe_aligned_mode>
AVS_FORCEINLINE static void process_two_pixels_h_float(const float* src_ptr, int begin1, int begin2, float* current_coeff, int filter_size, __m256& result1, __m256& result2, int kernel_size) {
  int ksmod8;
  // 32 bytes contain 8 floats
  if constexpr (safe_aligned_mode)
    ksmod8 = filter_size / 8 * 8;
  else
    ksmod8 = kernel_size / 8 * 8; // danger zone, scanline overread possible. Use exact unaligned kernel_size
  const float* src_ptr1 = src_ptr + begin1;
  const float* src_ptr2 = src_ptr + begin2;
  int i = 0;

  // Process 8 elements at a time
  for (; i < ksmod8; i += 8) {
    process_pix2_coeff8_h_float_core(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2);
  }

  if constexpr (!safe_aligned_mode) {
    // working with the original, unaligned kernel_size
    if (i == kernel_size) return;

    float* current_coeff2 = current_coeff + filter_size; // Points to second pixel's coefficients
    const int ksmod4 = kernel_size / 4 * 4;

    // Process 4 elements if needed
    if (i < ksmod4) {
      // Process 4 elements for first pixel
      __m128 data_1 = _mm_loadu_ps(src_ptr1 + i);
      __m128 coeff_1 = _mm_load_ps(current_coeff + i);
      __m128 temp_result1 = _mm_mul_ps(data_1, coeff_1);

      // Process 4 elements for second pixel
      __m128 data_2 = _mm_loadu_ps(src_ptr2 + i);
      __m128 coeff_2 = _mm_load_ps(current_coeff2 + i);
      __m128 temp_result2 = _mm_mul_ps(data_2, coeff_2);

      // update result vectors
      __m256 temp1 = _mm256_setzero_ps();
      __m256 temp2 = _mm256_setzero_ps();
      temp1 = _mm256_insertf128_ps(temp1, temp_result1, 0);
      temp2 = _mm256_insertf128_ps(temp2, temp_result2, 0);
      result1 = _mm256_add_ps(result1, temp1);
      result2 = _mm256_add_ps(result2, temp2);

      i += 4;
      if (i == kernel_size) return;
    }

    // Process remaining elements with scalar operations
    if (i < kernel_size) {
      float scalar_sum1[4] = { 0, 0, 0, 0 }; // like an __m128
      float scalar_sum2[4] = { 0, 0, 0, 0 };

      for (; i < kernel_size; i++) {
        scalar_sum1[i % 4] += src_ptr1[i] * current_coeff[i];
        scalar_sum2[i % 4] += src_ptr2[i] * current_coeff2[i];
      }

      // Convert scalar results to SIMD and add to result vectors
      __m128 temp_result1 = _mm_loadu_ps(scalar_sum1);
      __m128 temp_result2 = _mm_loadu_ps(scalar_sum2);

      __m256 temp1 = _mm256_setzero_ps();
      __m256 temp2 = _mm256_setzero_ps();
      temp1 = _mm256_insertf128_ps(temp1, temp_result1, 0);
      temp2 = _mm256_insertf128_ps(temp2, temp_result2, 0);
      result1 = _mm256_add_ps(result1, temp1);
      result2 = _mm256_add_ps(result2, temp2);
    }
  }
}

template<bool is_safe>
AVS_FORCEINLINE static void process_eight_pixels_h_float(const float* src, int x, float* current_coeff_base, int filter_size,
  __m128& zero128, __m256& zero256,
  float* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment >= 8); // code assumes this

  float* current_coeff = current_coeff_base + x * filter_size;
  const int unaligned_kernel_size = program->filter_size_real;

  // Unrolled processing of all 8 pixels

  // 0 & 1
  __m256 result0 = zero256;
  __m256 result1 = zero256;
  int begin0 = program->pixel_offset[x + 0];
  int begin1 = program->pixel_offset[x + 1];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m256 sumQuad12 = _mm256_hadd_ps(result0, result1); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2

  // 2 & 3
  result0 = zero256;
  result1 = zero256;
  begin0 = program->pixel_offset[x + 2];
  begin1 = program->pixel_offset[x + 3];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m256 sumQuad1234 = _mm256_hadd_ps(sumQuad12, _mm256_hadd_ps(result0, result1));

  __m128 result_lo = _mm_add_ps(_mm256_castps256_ps128(sumQuad1234), _mm256_extractf128_ps(sumQuad1234, 1)); // L1 L2 L3 L4

  // 4 & 5
  result0 = zero256;
  result1 = zero256;
  begin0 = program->pixel_offset[x + 4];
  begin1 = program->pixel_offset[x + 5];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  current_coeff += 2 * filter_size;
  __m256 sumQuad56 = _mm256_hadd_ps(result0, result1); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2

  // 6 & 7
  result0 = zero256;
  result1 = zero256;
  begin0 = program->pixel_offset[x + 6];
  begin1 = program->pixel_offset[x + 7];
  process_two_pixels_h_float<is_safe>(src, begin0, begin1, current_coeff, filter_size, result0, result1, unaligned_kernel_size);
  //current_coeff += 2 * filter_size;
  __m256 sumQuad5678 = _mm256_hadd_ps(sumQuad56, _mm256_hadd_ps(result0, result1));

  __m128 result_hi = _mm_add_ps(_mm256_castps256_ps128(sumQuad5678), _mm256_extractf128_ps(sumQuad5678, 1)); // L1 L2 L3 L4

  __m256 result256 = _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1); // merge result, result_hi

  _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result256); // 8 results at a time

}

// filtersizealigned8: special: 1..4. Generic: -1
template<int filtersizealigned8>
static void internal_resizer_h_avx2_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  const int filter_size = (filtersizealigned8 >= 1) ? filtersizealigned8 * 8 : program->filter_size;
  // knowing a quasi-constexpr filter_size from template for commonly used sizes
  // aligned_filter_size 8, 16, 24, 32 hugely helps compiler optimization

  __m128 zero128 = _mm_setzero_ps();
  __m256 zero256 = _mm256_setzero_ps();
  
  const float* src = (float*)src8;
  float* dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  // The safe path reads filter_size (aligned) source floats for each target pixel: limit by
  // safelimit_filter_size_aligned, not safelimit_8_pixels (8 is the target pixels per step).
  const int w_safe_mod8 = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / 8 * 8;

  for (int y = 0; y < height; y++) {
    float* current_coeff_base = program->pixel_coefficient_float;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod8; x += 8) {
      process_eight_pixels_h_float<true>(src, x, current_coeff_base, filter_size, zero128, zero256, dst, program);
    }

    // Process up to the actual kernel size instead of the aligned filter_size to prevent overreading beyond the last source pixel.
    // We assume extra offset entries were added to the p->pixel_offset array (aligned to 8 during initialization).
    // This may store 1-7 false pixels, but they are ignored since Avisynth will not read beyond the width.
    for (int x = w_safe_mod8; x < width; x += 8) {
      process_eight_pixels_h_float<false>(src, x, current_coeff_base, filter_size, zero128, zero256, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// here we cover filter sizes up to 32 (4x8) efficiently through template specialization
void resizer_h_avx2_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = program->filter_size;
  assert(program->filter_size_alignment >= 8); // coeffs are float, 8x4 float can be aligned_loaded with AVX2

  if (filter_size == 8)
    internal_resizer_h_avx2_generic_float<1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 2 * 8)
    internal_resizer_h_avx2_generic_float<2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 3 * 8)
    internal_resizer_h_avx2_generic_float<3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 4 * 8)
    internal_resizer_h_avx2_generic_float<4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_avx2_generic_float< -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// end of H float

//-------- 256 bit Verticals
// On x86-32 keep the 1×16 (or 2-lane/16-pixel) kernel
// On x86-64 use the 2×16 (4-lane/32-pixel) kernel.
// On 32-bit fewer YMM registers are available, 2x16 kernel causes register pressure issues.
// 10% performance loss on x86-32 with 2x16 kernel.

static void resize_v_avx2_planar_uint8_pix16(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);
  int filter_size = program->filter_size;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;
  __m256i rounder = _mm256_set1_epi32(1 << (FPScale8bits - 1));
  __m256i zero = _mm256_setzero_si256();

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* AVS_RESTRICT src_ptr = src + offset * src_pitch;

    // 16 byte 16 pixel
    // no need wmod16, alignment is safe at least 32
    for (int x = 0; x < width; x += 16) {

      __m256i result_single_lo = rounder;
      __m256i result_single_hi = rounder;

      const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {

        // Load two coefficients as a single packed value and broadcast
        __m256i coeff = _mm256_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co   CO|co|CO|co|CO|co|CO|co

        __m256i src_even = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr))); // 16x 8->16bit pixels
        __m256i src_odd = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch)));  // 16x 8->16bit pixels
        __m256i src_lo = _mm256_unpacklo_epi16(src_even, src_odd);
        __m256i src_hi = _mm256_unpackhi_epi16(src_even, src_odd);

        result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c
        src2_ptr += 2 * src_pitch;
      }

      // Process the last odd row if needed
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m256i coeff = _mm256_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m256i src_even = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr))); // 16x 8->16bit pixels
        __m256i src_lo = _mm256_unpacklo_epi16(src_even, zero);
        __m256i src_hi = _mm256_unpackhi_epi16(src_even, zero);
        result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c
        src2_ptr += src_pitch;

      }

      // scale back, store
      __m256i result_lo = result_single_lo;
      __m256i result_hi = result_single_hi;
      // shift back integer arithmetic 14 bits precision
      result_lo = _mm256_srai_epi32(result_lo, FPScale8bits);
      result_hi = _mm256_srai_epi32(result_hi, FPScale8bits);

      __m256i result_2x8x_uint16 = _mm256_packus_epi32(result_lo, result_hi);

      __m128i result128_lo = _mm256_castsi256_si128(result_2x8x_uint16);
      __m128i result128_hi = _mm256_extractf128_si256(result_2x8x_uint16, 1);
      __m128i result128 = _mm_packus_epi16(result128_lo, result128_hi);
      _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x), result128);

    }
    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

static void resize_v_avx2_planar_uint8_pix32(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
    AVS_UNUSED(bits_per_pixel);
    int filter_size = program->filter_size;
    const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;
    __m256i rounder = _mm256_set1_epi32(1 << (FPScale8bits - 1));
    __m256i zero = _mm256_setzero_si256();

    const int kernel_size = program->filter_size_real; // not the aligned
    const int kernel_size_mod2 = (kernel_size / 2) * 2;

    for (int y = 0; y < target_height; y++) {
        int offset = program->pixel_offset[y];
        const BYTE* AVS_RESTRICT src_ptr = src + offset * src_pitch;

        // 32 byte 32 pixel
        // alignment is safe till 64
        for (int x = 0; x < width; x += 32) {

            __m256i result_single_lo = rounder;
            __m256i result_single_hi = rounder;

            __m256i result_single_lo2 = rounder;
            __m256i result_single_hi2 = rounder;

            const uint8_t* AVS_RESTRICT src2_ptr = src_ptr + x;

            // Process pairs of rows for better efficiency (2 coeffs/cycle)
            int i = 0;
            for (; i < kernel_size_mod2; i += 2) {

                // Load two coefficients as a single packed value and broadcast
                __m256i coeff = _mm256_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co   CO|co|CO|co|CO|co|CO|co

                __m256i src_even = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr))); // 16x 8->16bit pixels
                __m256i src_odd = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch)));  // 16x 8->16bit pixels

                __m256i src_even2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + 16))); // 16x 8->16bit pixels
                __m256i src_odd2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + src_pitch + 16)));  // 16x 8->16bit pixels


                __m256i src_lo = _mm256_unpacklo_epi16(src_even, src_odd);
                __m256i src_hi = _mm256_unpackhi_epi16(src_even, src_odd);

                __m256i src_lo2 = _mm256_unpacklo_epi16(src_even2, src_odd2);
                __m256i src_hi2 = _mm256_unpackhi_epi16(src_even2, src_odd2);


                result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c

                result_single_lo2 = _mm256_add_epi32(result_single_lo2, _mm256_madd_epi16(src_lo2, coeff)); // a*b + c
                result_single_hi2 = _mm256_add_epi32(result_single_hi2, _mm256_madd_epi16(src_hi2, coeff)); // a*b + c

                src2_ptr += 2 * src_pitch;
            }

            // Process the last odd row if needed
            for (; i < kernel_size; i++) {
                // Broadcast a single coefficients
                __m256i coeff = _mm256_set1_epi16(*reinterpret_cast<const short*>(current_coeff + i)); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

                __m256i src_even = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr))); // 16x 8->16bit pixels

                __m256i src_even2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src2_ptr + 16))); // 16x 8->16bit pixels

                __m256i src_lo = _mm256_unpacklo_epi16(src_even, zero);
                __m256i src_hi = _mm256_unpackhi_epi16(src_even, zero);

                __m256i src_lo2 = _mm256_unpacklo_epi16(src_even2, zero);
                __m256i src_hi2 = _mm256_unpackhi_epi16(src_even2, zero);

                result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
                result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c

                result_single_lo2 = _mm256_add_epi32(result_single_lo2, _mm256_madd_epi16(src_lo2, coeff)); // a*b + c
                result_single_hi2 = _mm256_add_epi32(result_single_hi2, _mm256_madd_epi16(src_hi2, coeff)); // a*b + c

                src2_ptr += src_pitch;

            }

            // scale back, store
            __m256i result_lo = result_single_lo;
            __m256i result_hi = result_single_hi;

            __m256i result_lo2 = result_single_lo2;
            __m256i result_hi2 = result_single_hi2;


            // shift back integer arithmetic 14 bits precision
            result_lo = _mm256_srai_epi32(result_lo, FPScale8bits);
            result_hi = _mm256_srai_epi32(result_hi, FPScale8bits);

            result_lo2 = _mm256_srai_epi32(result_lo2, FPScale8bits);
            result_hi2 = _mm256_srai_epi32(result_hi2, FPScale8bits);

            __m256i result_2x8x_uint16 = _mm256_packus_epi32(result_lo, result_hi);

            __m256i result_2x8x_uint16_2 = _mm256_packus_epi32(result_lo2, result_hi2);

            __m128i result128_lo = _mm256_castsi256_si128(result_2x8x_uint16);
            __m128i result128_hi = _mm256_extractf128_si256(result_2x8x_uint16, 1);
            __m128i result128 = _mm_packus_epi16(result128_lo, result128_hi);

            __m128i result128_lo2 = _mm256_castsi256_si128(result_2x8x_uint16_2);
            __m128i result128_hi2 = _mm256_extractf128_si256(result_2x8x_uint16_2, 1);
            __m128i result128_2 = _mm_packus_epi16(result128_lo2, result128_hi2);


            _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x), result128);
            _mm_stream_si128(reinterpret_cast<__m128i*>(dst + x + 16), result128_2);

        }
        dst += dst_pitch;
        current_coeff += filter_size;
    }
}

#if defined(X86_64)
static void resize_v_avx2_planar_impl(BYTE* dst8, const BYTE* src, int dst_pitch, int src_pitch,
  ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  resize_v_avx2_planar_uint8_pix32(dst8, src, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel);
}
#elif defined(X86_32)
static void resize_v_avx2_planar_impl(BYTE* dst8, const BYTE* src, int dst_pitch, int src_pitch,
  ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  resize_v_avx2_planar_uint8_pix16(dst8, src, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel);
}
#else
#error Unsupported target for resize_v_avx2_planar_uint8_t
#endif

void resize_v_avx2_planar_uint8_t(BYTE* dst8, const BYTE* src, int dst_pitch, int src_pitch,
  ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  resize_v_avx2_planar_impl(dst8, src, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel);
}


template<bool lessthan16bit>
void resize_v_avx2_planar_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  int filter_size = program->filter_size;
  const short* AVS_RESTRICT current_coeff = program->pixel_coefficient;

  const __m256i zero = _mm256_setzero_si256();

  // for 16 bits only
  const __m256i shifttosigned = _mm256_set1_epi16(-32768);
  const __m256i shiftfromsigned = _mm256_set1_epi32(32768 << FPScale16bits);

  const __m256i rounder = _mm256_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t*)src8;
  uint16_t* AVS_RESTRICT dst = (uint16_t* AVS_RESTRICT)dst8;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2;

  const int limit = (1 << bits_per_pixel) - 1;
  __m256i clamp_limit = _mm256_set1_epi16((short)limit); // clamp limit for <16 bits

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + offset * src_pitch;

    // 32 byte 16 word
    // no need wmod16, alignment is safe at least 32

    for (int x = 0; x < width; x += 16) {

      __m256i result_single_lo = rounder;
      __m256i result_single_hi = rounder;

      const uint16_t* AVS_RESTRICT src2_ptr = src_ptr + x;

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        // Load two coefficients as a single packed value and broadcast
        __m256i coeff = _mm256_set1_epi32(*reinterpret_cast<const int*>(current_coeff + i)); // CO|co|CO|co|CO|co|CO|co   CO|co|CO|co|CO|co|CO|co

        __m256i src_even = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src2_ptr)); // 16x 16bit pixels
        __m256i src_odd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src2_ptr + src_pitch));  // 16x 16bit pixels
        if constexpr (!lessthan16bit) {
          src_even = _mm256_add_epi16(src_even, shifttosigned);
          src_odd = _mm256_add_epi16(src_odd, shifttosigned);
        }
        __m256i src_lo = _mm256_unpacklo_epi16(src_even, src_odd);
        __m256i src_hi = _mm256_unpackhi_epi16(src_even, src_odd);

        result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c

        src2_ptr += 2 * src_pitch;
      }

      // Process the last odd row if needed
      for (; i < kernel_size; i++) {
        // Broadcast a single coefficients
        __m256i coeff = _mm256_set1_epi16(current_coeff[i]); // 0|co|0|co|0|co|0|co   0|co|0|co|0|co|0|co

        __m256i src_even = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src2_ptr)); // 16x 16bit pixels
        if constexpr (!lessthan16bit) {
          src_even = _mm256_add_epi16(src_even, shifttosigned);
        }
        __m256i src_lo = _mm256_unpacklo_epi16(src_even, zero);
        __m256i src_hi = _mm256_unpackhi_epi16(src_even, zero);
        result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c

        src2_ptr += src_pitch;
      }

      // correct if signed, scale back, store
      __m256i result_lo = result_single_lo;
      __m256i result_hi = result_single_hi;
      if constexpr (!lessthan16bit) {
        result_lo = _mm256_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm256_add_epi32(result_hi, shiftfromsigned);
      }
      // shift back integer arithmetic 13 bits precision
      result_lo = _mm256_srai_epi32(result_lo, FPScale16bits);
      result_hi = _mm256_srai_epi32(result_hi, FPScale16bits);

      __m256i result_2x8x_uint16 = _mm256_packus_epi32(result_lo, result_hi);
      if constexpr (lessthan16bit) {
        result_2x8x_uint16 = _mm256_min_epu16(result_2x8x_uint16, clamp_limit); // extra clamp for 10-14 bit
      }
      _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + x), result_2x8x_uint16);

    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

//-------- 256 bit float Verticals

void resize_v_avx2_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
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

    // 32 byte 8 floats (AVX2 register holds 8 floats)
    // no need for wmod8, alignment is safe 32 bytes at least
    for (int x = 0; x < width; x += 8) {
      __m256 result_single = _mm256_setzero_ps();
      __m256 result_single_2 = _mm256_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      // two result variables for potential parallel operation
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m256 coeff_even = _mm256_set1_ps(current_coeff[i]);
        __m256 coeff_odd = _mm256_set1_ps(current_coeff[i + 1]);

        __m256 src_even = _mm256_loadu_ps(src2_ptr);
        __m256 src_odd = _mm256_loadu_ps(src2_ptr + src_pitch);

        result_single = _mm256_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm256_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += 2 * src_pitch;
      }

      result_single = _mm256_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m256 coeff = _mm256_set1_ps(current_coeff[i]);
        __m256 src_val = _mm256_loadu_ps(src2_ptr);
        result_single = _mm256_fmadd_ps(src_val, coeff, result_single);
      }

      _mm256_stream_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// Memory-transfer optimized version of resize_v_avx2_planar_float
void resize_v_avx2_planar_float_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
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

  const int width_mod32 = (width / 32) * 32;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;
    // Part #1: process 32 floats at a time
    // Optimize for memory throughput: process 32 floats (4x256bit) in parallel
    // Process by 4x 256bit (8 x 8 floats) to make memory read/write linear streams
    // longer, 16x256 bit registers in 64bit mode should be enough
    for (int x = 0; x < width_mod32; x += 32) {
      __m256 result_1 = _mm256_setzero_ps();
      __m256 result_2 = _mm256_setzero_ps();
      __m256 result_3 = _mm256_setzero_ps();
      __m256 result_4 = _mm256_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here
      // single coeffs/cycle, but 32 floats processed in parallel
      for (int i = 0; i < kernel_size; i++) {
        // coefs are equal for all H-samples
        __m256 coeff = _mm256_set1_ps(current_coeff[i]);

        // source always aligned in V-resizers
        __m256 src_1 = _mm256_load_ps(src2_ptr);
        __m256 src_2 = _mm256_load_ps(src2_ptr + 8);
        __m256 src_3 = _mm256_load_ps(src2_ptr + 16);
        __m256 src_4 = _mm256_load_ps(src2_ptr + 24);

        result_1 = _mm256_fmadd_ps(src_1, coeff, result_1);
        result_2 = _mm256_fmadd_ps(src_2, coeff, result_2);
        result_3 = _mm256_fmadd_ps(src_3, coeff, result_3);
        result_4 = _mm256_fmadd_ps(src_4, coeff, result_4);

        src2_ptr += src_pitch;
      }
      // here we use stream instead of store; in multithreading stream is better;
      // consider two templated versions if needed depending on actual MT usage
      _mm256_stream_ps(dst + x, result_1);
      _mm256_stream_ps(dst + x + 8, result_2);
      _mm256_stream_ps(dst + x + 16, result_3);
      _mm256_stream_ps(dst + x + 24, result_4);
    } // width_mod32

    // Part #2: process remaining. 32 byte 8 floats (AVX2 register holds 8 floats)
    // From now on the old resize_v_avx2_planar_float, starting at width_mod32.

    // No need for wmod8, scanline alignment is safe 32 bytes at least (really 64)
    for (int x = width_mod32; x < width; x += 8) {
      __m256 result_single = _mm256_setzero_ps();
      __m256 result_single_2 = _mm256_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      // two result variables for potential parallel operation
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m256 coeff_even = _mm256_set1_ps(current_coeff[i]);
        __m256 coeff_odd = _mm256_set1_ps(current_coeff[i + 1]);

        __m256 src_even = _mm256_load_ps(src2_ptr);
        __m256 src_odd = _mm256_load_ps(src2_ptr + src_pitch);

        result_single = _mm256_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm256_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += 2 * src_pitch;
      }

      result_single = _mm256_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m256 coeff = _mm256_set1_ps(current_coeff[i]);
        __m256 src_val = _mm256_load_ps(src2_ptr);
        result_single = _mm256_fmadd_ps(src_val, coeff, result_single);
      }

      _mm256_stream_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

// avx2 16bit
template void resizer_h_avx2_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
// avx2 10-14bit
template void resizer_h_avx2_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);



// avx2 16
template void resize_v_avx2_planar_uint16_t<false>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
// avx2 10-14bit
template void resize_v_avx2_planar_uint16_t<true>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);



// Helper for horizontal resampling 32 bit float
// Safe dual lane partial load with AVX
// Read exactly N pixels, avoiding
// - reading beyond the end of the source buffer.
// - avoid NaN contamination, since event with zero coefficients NaN * 0 = NaN
template <int Nmod4>
AVS_FORCEINLINE static __m256 _mm256_load_partial_safe_2_m128(const float* src_ptr_offsetted1, const float* src_ptr_offsetted2) {
  __m128 s1;
  __m128 s2;
  switch (Nmod4) {
  case 1:
    s1 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted2[0]);
    // ideally: movss
    break;
  case 2:
    s1 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    // ideally: movsd
    break;
  case 3:
    s1 = _mm_set_ps(0.0f, src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    // ideally: movss + movsd + shuffle or movsd + insert
    break;
  case 0:
    s1 = _mm_set_ps(src_ptr_offsetted1[3], src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(src_ptr_offsetted2[3], src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    // ideally: movups
    break;
  default:
    s1 = _mm_setzero_ps(); // n/a cannot happen
    s2 = _mm_setzero_ps();
  }
  return _mm256_set_m128(s2, s1);
}


// Processes a horizontal resampling kernel of up to four coefficients for float pixel types.
// Supports BilinearResize, BicubicResize, or sinc with up to 2 taps (filter size <= 4).
// Loads and processes four float coefficients and eight pixels simultaneously.
// The 'filtersizemod4' template parameter (0-3) helps optimize for different filter sizes modulo 4.

// this is a generic varsion for small kernels up to 4 taps, regardless of up or down scaling
// Note: there is a further optimized version of ks4 resampler, which combines gather or permutex.
template<int filtersizemod4>
void resize_h_planar_float_avx_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  assert(filtersizemod4 >= 0 && filtersizemod4 <= 3);

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  constexpr int PIXELS_AT_A_TIME = 8; // Process eight pixels in parallel using AVX2 (2x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once with '_mm_loadu_ps'.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 7' when processing 8 H pixels at a time or
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 8);

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  assert(program->filter_size_alignment >= 4);

  int x = 0;

  // This 'auto' lambda construct replaces the need of templates
  auto do_h_float_core = [&](auto partial_load) {
    // Load up to 2x4 coefficients at once before the height loop.
    // Pre-loading and transposing coefficients keeps register usage efficient.
    // Assumes 'filter_size_aligned' is at least 4.

    // Coefficients for the source pixel offset (for src_ptr + begin1 [0..3] and for src_ptr + begin5 [0..3] )
    __m256 coef_1_coef_5 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
    __m256 coef_2_coef_6 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
    __m256 coef_3_coef_7 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
    __m256 coef_4_coef_8 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);

    _MM_TRANSPOSE8_LANE4_PS(coef_1_coef_5, coef_2_coef_6, coef_3_coef_7, coef_4_coef_8);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src;

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

    for (int y = 0; y < height; y++)
    {
      __m256 data_1_data_5;
      __m256 data_2_data_6;
      __m256 data_3_data_7;
      __m256 data_4_data_8;

      if constexpr (partial_load) {
        // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
        // to prevent reading beyond the allocated source scanline. This handles cases where loading 4 floats
        // starting from 'src_ptr + beginX' might exceed the source buffer.

        // Example of the unsafe scenario: If target width is 320, a naive load at src_ptr + 317
        // would attempt to read floats at indices 317, 318, 319, and 320, potentially going out of bounds.

        // Two main issues in the unsafe zone:
        // 1.) Out-of-bounds memory access: Reading beyond the allocated memory for the source scanline can
        //     lead to access violations and crashes. '_mm_loadu_ps' attempts to load 16 bytes, so even if
        //     the starting address is within bounds, subsequent reads might not be.
        // 2.) Garbage or NaN values: Even if a read doesn't cause a crash, accessing uninitialized or
        //     out-of-bounds memory (especially for float types) can result in garbage data, including NaN.
        //     Multiplying by a valid coefficient and accumulating this NaN can contaminate the final result.

        // '_mm256_load_partial_safe_2_m128' safely loads up to 'filter_size_real' pixels and pads with zeros if needed,
        // preventing out-of-bounds reads and ensuring predictable results even near the image edges.

        data_1_data_5 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin1, src_ptr + begin5);
        data_2_data_6 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin2, src_ptr + begin6);
        data_3_data_7 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin3, src_ptr + begin7);
        data_4_data_8 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin4, src_ptr + begin8);
      }
      else {
        // In the safe zone, we can directly load 4 pixels at a time using unaligned loads.
        data_1_data_5 = _mm256_loadu_2_m128(src_ptr + begin1, src_ptr + begin5);
        data_2_data_6 = _mm256_loadu_2_m128(src_ptr + begin2, src_ptr + begin6);
        data_3_data_7 = _mm256_loadu_2_m128(src_ptr + begin3, src_ptr + begin7);
        data_4_data_8 = _mm256_loadu_2_m128(src_ptr + begin4, src_ptr + begin8);
      }

      _MM_TRANSPOSE8_LANE4_PS(data_1_data_5, data_2_data_6, data_3_data_7, data_4_data_8);

      // two sets, hint for the compiler to allow parallel fma's
      __m256 result_0 = _mm256_mul_ps(data_1_data_5, coef_1_coef_5);
      __m256 result_1 = _mm256_mul_ps(data_2_data_6, coef_2_coef_6);
      result_0 = _mm256_fmadd_ps(data_3_data_7, coef_3_coef_7, result_0);
      result_1 = _mm256_fmadd_ps(data_4_data_8, coef_4_coef_8, result_1);

      _mm256_stream_ps(dst_ptr, _mm256_add_ps(result_0, result_1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    } // y
    current_coeff += filter_size * 8; // Move to the next set of coefficients for the next 8 output pixels
    }; // end of lambda

  // Process the 'safe zone' where direct full unaligned loads are acceptable.
  for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
  {
    do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm_loadu_ps
  }

  // Process the potentially 'unsafe zone' near the image edge, using safe loading.
  for (; x < width; x += PIXELS_AT_A_TIME)
  {
    do_h_float_core(std::true_type{}); // partial_load == true, use the safer '_mm256_load_partial_safe_2_m128'
  }
}

// Instantiate them
template void resize_h_planar_float_avx_transpose_vstripe_ks4<0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx_transpose_vstripe_ks4<1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx_transpose_vstripe_ks4<2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx_transpose_vstripe_ks4<3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


/**
 * resize_h_planar_float_avx2_gather_permutex_vstripe_ks4 with per-frame check or
 * distinct checker, gather, permutex methods
 *
 * AVX2-optimized horizontal resampler for float planar images with small kernel sizes (filter_size_real <= 4).
 * Supports both upsampling and downsampling scenarios, automatically selecting the most efficient SIMD strategy.
 * (For larger kernels, use resizer_h_avx2_generic_float or other specialized functions.)
 *
 * Algorithm:
 *   - Analyzes the resampling program's pixel offset pattern to choose between two SIMD strategies.
 *     The upsampling scenario is divided into sub-cases, and the decision is made by analyzing the pixel offset pattern in the resampling program.
 *     The code checks, for each group of 8 output pixels, how far apart the corresponding source pixel offsets are.
 *
 *     If the span of source pixels (end_off - start_off) plus the kernel size (max filter_size_real - 1, that is 3)
 *     exceeds 8, it goes to gather based method: the required source pixels are not all within a single 8-float block.
 *
 *     If the span is <= 8, the function can use a single 8-float block load and permute (which is faster).
 *
 *     For "high upsampling ratio" (output much larger than input, so output pixels are close together in input),
 *     the offsets are usually contiguous, and the permute-transpose path is used.
 *
 *     1. Gather-based: For downsampling (or no-resize convolution) or non-contiguous pixel offsets, uses AVX2
 *        gather instructions to fetch each required source pixel.
 *     2. Permutex-based: For upsampling or contiguous pixel offsets, loads a block of 8 source floats and uses
 *        AVX2 permute instructions for fast access.
 *
 *   - Handles edge cases and buffer boundaries safely, using partial loads to avoid out-of-bounds memory access.
 *   - Processes 8 output pixels in parallel for high throughput.
 *
 * Assumes that resampling program provides sufficient alignment and padding for safe SIMD loads.
 *
 * Typical dispatcher usage:
 *   switch (program->filter_size_real) {
 *     case 1: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<1>;
 *     case 2: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<2>;
 *     case 3: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<3>;
 *     case 4: return resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<0>;
 *     default: return resizer_h_avx2_generic_float;
 *   }
 *
 * See also:
 * - resize_h_planar_float_avx2_transpose_vstripe_ks4
 * - resize_h_planar_float_avx2_permutex_vstripe_ks4
 * - resize_h_planar_float_avx_transpose_vstripe_ks4
 * - resizer_h_avx2_generic_float
 
 */

// Test script for the ks<=4 gather/permutex horizontal, float resampler cases
/*
SetMaxCPU("AVX2")
BlankClip(width=640, height=480, pixel_type="YUV444PS")
#BlankClip(width=640-1, height=480, pixel_type="YUV444PS") # -1 to -7 to test partial loads
Expr("sx 2 % 1.0 * ", "0", "0") # vertical stripes
BicubicResize(width*2,height) # permute, H kernel size 4
or
LanczosResize(width*2, height, taps=1) # permute, H kernel size 2
or
LanczosResize(int(width*0.5), height, taps=1) # gather, H kernel size 4
or
BilinearResize(int(width*0.97), height) # gather, H kernel size 3
*/

/*
 * Analyse input resampling program to select method of processing.
 * 
 * This check determines whether the AVX2 permutex optimization is valid for a block of 8 output pixels.
 * In the permutex path, we load 8 consecutive source floats starting at program->pixel_offset[x + 0] ('begin1').
 * Each output pixel's convolution window is indexed using perm_0..perm_3, which are offsets relative to begin1.
 * These permutation indices span from begin1 (program->pixel_offset[x + 0]) up to begin8 + 3 (program->pixel_offset[x + 7] + 3).
 * For the permute to be safe, ALL indices accessed (from begin1 to begin8 + 3) must fit within the loaded 8-float block.
 * This is guaranteed if (program->pixel_offset[x + 7] + 3 - program->pixel_offset[x + 0]) < 8.
 * In order the check work for the right edge, pixel_offset entries padded till target_size_aligned must repeat the last
 * valid offset, and not 0 (see in resize_prepare_coeffs).
 * 
 * If the span is not in the 0-7 range, some required source pixels for the convolution will fall outside
 * the loaded block, and the permutex method cannot be used; we must fall back to gather.
 * This logic relies on the assumption that pixel_offset[] is strictly increasing (or non-decreasing).
 * We check the maximum index accessed by the permutation logic, and since we use a fixed 4 coefficients
 * per output pixel, not just the filter_size_real, we add 3 to the last offset.
 * 
 * It is ensured during the resampling program setup (resize_prepare_coeffs) that pixel_offsets will
 * not only contain valid source offsets, but so that (pixel_offsets[x] + filter_size_real - 1) still
 * indexes valid source pixels.
 * On the right side of the image, this means that the end-of-line coefficients are shifted leftwards
 * during the pre-calculation so that the filter kernel will never read beyond the coefficient array
 * nor past the source buffer.
 * Out of bounds target pixels coefficients are padded with zeros up to program->filter_size_alignment.
*/

// resize_h_planar_float_avx2_xxx_vstripe_ks4 method #1: gather-based
template<int filtersizemod4>
void resize_h_planar_float_avx2_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  assert(filtersizemod4 >= 0 && filtersizemod4 <= 3);

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 8; // Process eight pixels in parallel using AVX2 (2x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once with '_mm_loadu_ps'.
  // So contrary to the 8-pixel-at-a-time fact, we only require safety for 4 pixels at a time here.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.
  // this goes to filtersizemod4 template parameter from 0 to 3

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 7', and pixel_offsets[x to x + 7] when processing 8 H pixels at a time
  assert(program->target_size_alignment >= 8);

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  assert(program->filter_size_alignment >= 4);

  // Split to H-stripes to make better source data locality in L2 cache (if L2 present per core ?)
  constexpr int STRIPE_ALIGN = 8;  // this must be multiple of PIXELS_AT_A_TIME 

  //max_scanlines = program->target_size; // test
  int max_scanlines = program->max_scanlines / STRIPE_ALIGN * STRIPE_ALIGN;
  if (max_scanlines < STRIPE_ALIGN) max_scanlines = STRIPE_ALIGN;

  for (auto y_from = 0; y_from < height; y_from += max_scanlines) {
    size_t y_to = std::min(y_from + max_scanlines, height);

    // Reset current_coeff for the start of the stripe
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float; // +iYstart * filter_size;

    int x = 0;

    // This 'auto' lambda construct replaces the need of templates
    auto do_h_float_core = [&](auto partial_load) {
      // Load up to 2x4 coefficients at once before the height loop.
      // Pre-loading and transposing coefficients keeps register usage efficient.
      // Assumes 'filter_size_aligned' is at least 4.

      // Coefficients for the source pixel offset (for src_ptr + begin1 [0..3] and for src_ptr + begin5 [0..3] )
      __m256 coef_1_coef_5 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
      __m256 coef_2_coef_6 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
      __m256 coef_3_coef_7 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
      __m256 coef_4_coef_8 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);

      _MM_TRANSPOSE8_LANE4_PS(coef_1_coef_5, coef_2_coef_6, coef_3_coef_7, coef_4_coef_8);

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

      size_t y = y_from;

      float* AVS_RESTRICT dst_ptr = dst + y * dst_pitch + x;
      const float* src_ptr = src + y * src_pitch;
      for (; y < y_to; ++y) {
        //float* AVS_RESTRICT dst_ptr = dst + y * dst_pitch + x;
        //const float* src_ptr = src + y * src_pitch;

        __m256 data_1_data_5;
        __m256 data_2_data_6;
        __m256 data_3_data_7;
        __m256 data_4_data_8;

        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline. This handles cases where loading 4 floats
          // starting from 'src_ptr + beginX' might exceed the source buffer.

          // Example of the unsafe scenario: If target width is 320, a load at src_ptr + 317
          // would attempt to read floats at indices 317, 318, 319, and 320, potentially going out of bounds.

          // Two main issues in the unsafe zone:
          // 1.) Out-of-bounds memory access: Reading beyond the allocated memory for the source scanline can
          //     lead to access violations and crashes. '_mm_loadu_ps' attempts to load 16 bytes, so even if
          //     the starting address is within bounds, subsequent reads might not be.
          // 2.) Garbage or NaN values: Even if a read doesn't cause a crash, accessing uninitialized or
          //     out-of-bounds memory (especially for float types) can result in garbage data, including NaN.
          //     Multiplying by a valid coefficient and accumulating this NaN can contaminate the final result.

          // '_mm256_load_partial_safe_2_m128' safely loads up to 'filter_size_real' pixels and pads with zeros if needed,
          // preventing out-of-bounds reads and ensuring predictable results even near the image edges.

          data_1_data_5 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin1, src_ptr + begin5);
          data_2_data_6 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin2, src_ptr + begin6);
          data_3_data_7 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin3, src_ptr + begin7);
          data_4_data_8 = _mm256_load_partial_safe_2_m128<filtersizemod4>(src_ptr + begin4, src_ptr + begin8);
        }
        else {
          // In the safe zone, we can directly load 4 pixels at a time using unaligned loads.
          data_1_data_5 = _mm256_loadu_2_m128(src_ptr + begin1, src_ptr + begin5);
          data_2_data_6 = _mm256_loadu_2_m128(src_ptr + begin2, src_ptr + begin6);
          data_3_data_7 = _mm256_loadu_2_m128(src_ptr + begin3, src_ptr + begin7);
          data_4_data_8 = _mm256_loadu_2_m128(src_ptr + begin4, src_ptr + begin8);
        }

        _MM_TRANSPOSE8_LANE4_PS(data_1_data_5, data_2_data_6, data_3_data_7, data_4_data_8);

        __m256 result = _mm256_mul_ps(data_1_data_5, coef_1_coef_5);
        result = _mm256_fmadd_ps(data_2_data_6, coef_2_coef_6, result);
        result = _mm256_fmadd_ps(data_3_data_7, coef_3_coef_7, result);
        result = _mm256_fmadd_ps(data_4_data_8, coef_4_coef_8, result);

        _mm256_stream_ps(dst_ptr, result);
        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      } // y
      current_coeff += filter_size * 8; // Move to the next set of coefficients for the next 8 output pixels
      }; // end of lambda

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm256_loadu_ps
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer _mm256_load_partial_safe_2_m128
    }
  }
}

// Helper for permutex style horizontal resampling 32 bit float
// Safe partial load for 1-7 floats, padding with zeros to avoid NaN contamination
static inline __m256 _mm256_load_partial_safe(const float* src_ptr, int floats_to_load) {
  if (floats_to_load == 1)
    return _mm256_setr_ps(src_ptr[0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  if (floats_to_load == 2)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  if (floats_to_load == 3)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  if (floats_to_load == 4)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], 0.0f, 0.0f, 0.0f, 0.0f);
  if (floats_to_load == 5)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], 0.0f, 0.0f, 0.0f);
  if (floats_to_load == 6)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], 0.0f, 0.0f);
  if (floats_to_load == 7)
    return _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], 0.0f);
  if (floats_to_load == 8)
    return _mm256_loadu_ps(src_ptr); // n/a cannot happen
  else
    return _mm256_setzero_ps(); // n/a cannot happen
}


// resize_h_planar_float_avx2_xxx_vstripe_ks4 method #2: permutex-based
void resize_h_planar_float_avx2_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 8; // Process eight pixels in parallel in AVX2

  // Pre-checked for permutex-based upsampling: the source pixels will surely fit within single 8 float loads
  // The right edge handling will be done via safe partial loads when needed, loading 8 pixels at once
  // may not be safe there.

  // 'source_overread_beyond_targetx' marks the x position in the target (output) scanline where,
  // if we process N pixels at a time (e.g., 8 for AVX2), the filter kernel may overread the source
  // buffer near the right edge due to kernel size and pixel offsets. Beyond this value, it is no
  // longer safe to read N source pixels at once from pixel_offset[].

  // For x positions < source_overread_beyond_targetx, it is safe to load N source pixels at once.
  // For x positions >= source_overread_beyond_targetx, we must use a safer loading method (e.g.,
  // partial loads with padding) to avoid out-of-bounds memory access.

  // permutex is even more special: the safety analysis is performed only for the beginning of each
  // block of 8 pixels processed at a time, so only the source loads for the offset position of
  // every 8th target pixel are considered. This is 'safelimit_8_pixels_each8th_target'.
  // The program's safe limits are pre-calculated during program setup.

  const int width_safe_mod = (program->safelimit_8_pixels_each8th_target.overread_possible ? program->safelimit_8_pixels_each8th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // coeff + filter_size*0 to filter_size*7 when processing 8 H pixels at a time
  assert(program->target_size_alignment >= 8);

  // Ensure that coefficient loading is safe for 4 float loads,
  // if less than 4, padded with zeros till filter_size_alignment.
  assert(program->filter_size_alignment >= 4);

  const int max_scanlines =  program->max_scanlines;

  for (int y_from = 0; y_from < height; y_from += max_scanlines) {
    int y_to = std::min(y_from + max_scanlines, height);
    // Reset current_coeff for the start of the stripe
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float; // +iYstart * filter_size;

    int x = 0;

    // This 'auto' lambda construct replaces the need of templates
    auto do_h_float_core = [&](auto partial_load) {
      // Assumes 'filter_size_alignment' <= 4, 'target_size_alignment' >= 8
      // Prepare 4 coefs per pixel for 8 pixels in transposed V-form at once before the height loop.
      __m256 coef_0 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
      __m256 coef_1 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
      __m256 coef_2 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
      __m256 coef_3 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);

      _MM_TRANSPOSE8_LANE4_PS(coef_0, coef_1, coef_2, coef_3);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      __m256i perm_0 = _mm256_loadu_si256((__m256i*)(&program->pixel_offset[x]));
      int iStart = program->pixel_offset[x];
      perm_0 = _mm256_sub_epi32(perm_0, _mm256_set1_epi32(iStart));
      /* like this:
      __m256i perm_0 = _mm512_set_epi32(
        program->pixel_offset[x + 7] - iStart,
        ...
        program->pixel_offset[x + 0] - iStart);
      */

      __m256i one_epi32 = _mm256_set1_epi32(1);
      __m256i perm_1 = _mm256_add_epi32(perm_0, one_epi32); // begin8_rel+1, begin7_rel+1, ... begin2_rel+1, begin1_rel+1
      __m256i perm_2 = _mm256_add_epi32(perm_1, one_epi32); // begin8_rel+2, begin7_rel+2, ... begin2_rel+2, begin1_rel+2
      __m256i perm_3 = _mm256_add_epi32(perm_2, one_epi32); // begin8_rel+3, begin7_rel+3, ... begin2_rel+3, begin1_rel+3
      // These indexes are guaranteed to be 0..7 due to the earlier analysis,
      // and can be used for the indexing parameter in _mm256_permutevar8x32_ps
      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + iStart + y_from * src_pitch;

      // for partial_load only
      const int remaining = program->source_size - iStart;
      const int floats_to_load = remaining >= 8 ? 8 : remaining;

      for (int y = y_from; y < y_to; ++y) {

        // process scanline y
        __m256 data_src;
        // We'll need exactly 8 floats starting from src+iStart
        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline. This handles cases where loading 8 floats
          // starting from 'src_ptr + beginX' might exceed the source buffer.
          data_src = _mm256_load_partial_safe(src_ptr, floats_to_load);
        }
        else {
          data_src = _mm256_loadu_ps(src_ptr); // load 8 source pixels, can contain garbage beyond the right edge in the last loop
        }

        // After we load 8 source pixels starting from begin1, we can be sure, that pixel_offset[x+0] .. pixel_offset[x+7] + 3 is
        // within valid source range. Pre-check chooses permutex method only if all needed pixels fit within these 8 loaded pixels.

        // perm_0 .. perm_3 contain the indexes to permute data_src into the correct order
        // for each of the 8 output pixels so they index into 0..7 (guaranteed) range of the source data loaded above
        __m256 data_0 = _mm256_permutevar8x32_ps(data_src, perm_0);
        __m256 data_1 = _mm256_permutevar8x32_ps(data_src, perm_1);
        __m256 data_2 = _mm256_permutevar8x32_ps(data_src, perm_2);
        __m256 data_3 = _mm256_permutevar8x32_ps(data_src, perm_3);

        __m256 result0 = _mm256_mul_ps(data_0, coef_0);
        __m256 result1 = _mm256_mul_ps(data_2, coef_2);

        result0 = _mm256_fmadd_ps(data_1, coef_1, result0);
        result1 = _mm256_fmadd_ps(data_3, coef_3, result1);

        // this must be stream until partial tile interface done
        _mm256_stream_ps(dst_ptr, _mm256_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }
      current_coeff += filter_size * 8;
      }; // end of lambda

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm_loadu_ps
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer '_mm256_load_partial_safe'
    }
  }
}

#if 0
// resize_h_planar_float_avx2_permutex_vstripe_ks8:
// Like ks4 but handles kernel sizes 5-8 using avx2_permutex2var_ps (two YMMs = 16-float range).
// Dispatch condition: check(8, 16, 8) = false — 8 output pixels' source span <= 16 floats.
// Covers upscale through ~0.875x downscale with LanczosResize(taps=4..8) / Spline36 etc.
void resize_h_planar_float_avx2_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size;

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 8;

  // Safe limit: check only at every 8th target pixel start, loading 16 floats from src+pixel_offset[x].
  // Beyond source_overread_beyond_targetx, load 16 floats at once is unsafe — switch to partial loads.
  const int width_safe_mod = (program->safelimit_16_pixels_each8th_target.overread_possible
    ? program->safelimit_16_pixels_each8th_target.source_overread_beyond_targetx
    : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  assert(program->filter_size_real <= 8);
  assert(program->target_size_alignment >= 8);
  assert(program->filter_size_alignment >= 8); // need 8 floats/pixel so +4 offset into tap-row is valid

  const int max_scanlines = program->max_scanlines;

  for (int y_from = 0; y_from < height; y_from += max_scanlines) {
    int y_to = std::min(y_from + max_scanlines, height);
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

    int x = 0;

    auto do_h_float_core = [&](auto partial_load) {
      // Load 8 taps per pixel for 8 output pixels.
      // Two groups of 4 taps each, transposed so that each coef_k vector holds
      // tap k for all 8 output pixels simultaneously.
      //
      // Taps 0-3: load from current_coeff + filter_size*j + 0..3 for j=0,1,...,7
      __m256 coef_0 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
      __m256 coef_1 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
      __m256 coef_2 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
      __m256 coef_3 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);
      _MM_TRANSPOSE8_LANE4_PS(coef_0, coef_1, coef_2, coef_3);
      // Taps 4-7: load from current_coeff + filter_size*j + 4 for j=0,1,...,7
      // filter_size is multiple of 8, so +4 floats (+16 bytes) keeps 16-byte alignment for _mm_load_ps.
      __m256 coef_4 = _mm256_load_2_m128(current_coeff + filter_size * 0 + 4, current_coeff + filter_size * 4 + 4);
      __m256 coef_5 = _mm256_load_2_m128(current_coeff + filter_size * 1 + 4, current_coeff + filter_size * 5 + 4);
      __m256 coef_6 = _mm256_load_2_m128(current_coeff + filter_size * 2 + 4, current_coeff + filter_size * 6 + 4);
      __m256 coef_7 = _mm256_load_2_m128(current_coeff + filter_size * 3 + 4, current_coeff + filter_size * 7 + 4);
      _MM_TRANSPOSE8_LANE4_PS(coef_4, coef_5, coef_6, coef_7);
      // After transpose: coef_k[lane i] = tap k coefficient for output pixel i.

      // Permute indices: perm_k[i] = pixel_offset[x+i] - iStart + k.
      // Guaranteed 0..15 by check(8,16,8)=false, suitable for avx2_permutex2var_ps (16-float range).
      __m256i perm_0 = _mm256_loadu_si256((__m256i*)(&program->pixel_offset[x]));
      const int iStart = program->pixel_offset[x];
      perm_0 = _mm256_sub_epi32(perm_0, _mm256_set1_epi32(iStart));
      const __m256i one_epi32 = _mm256_set1_epi32(1);
      __m256i perm_1 = _mm256_add_epi32(perm_0, one_epi32);
      __m256i perm_2 = _mm256_add_epi32(perm_1, one_epi32);
      __m256i perm_3 = _mm256_add_epi32(perm_2, one_epi32);
      __m256i perm_4 = _mm256_add_epi32(perm_3, one_epi32);
      __m256i perm_5 = _mm256_add_epi32(perm_4, one_epi32);
      __m256i perm_6 = _mm256_add_epi32(perm_5, one_epi32);
      __m256i perm_7 = _mm256_add_epi32(perm_6, one_epi32);

      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + iStart + y_from * src_pitch;

      // For partial_load: how many valid floats from iStart (capped at 8 per YMM).
      const int remaining = program->source_size - iStart;
      const int floats_to_load_a = std::min(remaining, 8);
      const int floats_to_load_b = std::max(remaining - 8, 0); // 0 → setzero in _mm256_load_partial_safe

      for (int y = y_from; y < y_to; ++y) {
        __m256 data_src, data_src2;
        if constexpr (partial_load) {
          data_src  = _mm256_load_partial_safe(src_ptr, floats_to_load_a);
          data_src2 = _mm256_load_partial_safe(src_ptr + 8, floats_to_load_b);
        }
        else {
          data_src  = _mm256_loadu_ps(src_ptr);
          data_src2 = _mm256_loadu_ps(src_ptr + 8);
        }

        // Gather each tap for all 8 output pixels from the 16-float window.
        __m256 data_0 = avx2_permutex2var_ps(data_src, data_src2, perm_0);
        __m256 data_1 = avx2_permutex2var_ps(data_src, data_src2, perm_1);
        __m256 data_2 = avx2_permutex2var_ps(data_src, data_src2, perm_2);
        __m256 data_3 = avx2_permutex2var_ps(data_src, data_src2, perm_3);
        __m256 data_4 = avx2_permutex2var_ps(data_src, data_src2, perm_4);
        __m256 data_5 = avx2_permutex2var_ps(data_src, data_src2, perm_5);
        __m256 data_6 = avx2_permutex2var_ps(data_src, data_src2, perm_6);
        __m256 data_7 = avx2_permutex2var_ps(data_src, data_src2, perm_7);

        // Two parallel FMA chains to reduce dependency depth.
        __m256 result0 = _mm256_mul_ps(data_0, coef_0);
        __m256 result1 = _mm256_mul_ps(data_2, coef_2);
        result0 = _mm256_fmadd_ps(data_1, coef_1, result0);
        result1 = _mm256_fmadd_ps(data_3, coef_3, result1);
        result0 = _mm256_fmadd_ps(data_4, coef_4, result0);
        result1 = _mm256_fmadd_ps(data_5, coef_5, result1);
        result0 = _mm256_fmadd_ps(data_6, coef_6, result0);
        result1 = _mm256_fmadd_ps(data_7, coef_7, result1);

        _mm256_stream_ps(dst_ptr, _mm256_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }
      current_coeff += filter_size * 8;
    }; // end of lambda

    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
      do_h_float_core(std::false_type{});
    for (; x < width; x += PIXELS_AT_A_TIME)
      do_h_float_core(std::true_type{});
  }
}
#endif

// Simulating the AVX512 case, where 16-way permutes are possible.
// H, kernel size 4, 2x8 pix version, 16 output pixels.
// Same as plain ks4, but 2x8 pixels instead of 1x8 pixels at a time.
// Since AVX2 only supports 256 bit and 8xfloat permute, we have to simulate 16 pixel permute, with
// handling cross-lane indices 0..15, but since the intrinsic support is only for 0..7 we have to use
// two separate permutes, masks and then blend the results together.

// Structure to hold all precalculated vectors for ONE set of coefficients/taps (e.g., perm_0)
// Since we have 4 taps (perm_0 to perm_3), you would need 4 instances of this structure.
typedef struct {
  // Permutation Indices (4 __m256i vectors)
  __m256i PL_A; // Low output half, source A indices (0-7)
  __m256i PL_B; // Low output half, source B indices (0-7)
  __m256i PH_A; // High output half, source A indices (0-7)
  __m256i PH_B; // High output half, source B indices (0-7)

  // Mask Vectors (2 __m256 vectors for blendv_ps)
  __m256 ML_B; // Low output half mask (1s select B, 0s select A)
  __m256 MH_B; // High output half mask (1s select B, 0s select A)
} PermuteVectors_AVX2;

static void precalculate_cross_perm_avx2(
  const int* pixel_offset,
  int x,
  PermuteVectors_AVX2* tap_vectors[4])
{
  // The base offset for the first loaded register A is pixel_offset[x + 0]
  const int begin1 = pixel_offset[x + 0];

  // Broadcast the constant 8 and the base offset 'begin1'
  const __m256i v_8 = _mm256_set1_epi32(8);
  const __m256i v_begin1 = _mm256_set1_epi32(begin1);

  // A constant 0 for index in the ignored lane
  const __m256i v_zero = _mm256_setzero_si256();

  // Loop through all 4 taps independently
  for (int tap = 0; tap < 4; ++tap) {

    // 1. Prepare the 16 absolute indices (I_k) by adding 'tap'
    // This still requires a temporary array or two separate loads/SIMD additions

    // Use temporary arrays of 8 elements for simplicity, but could be done directly in SIMD
    int I_k_low[8];
    int I_k_high[8];
    for (int k = 0; k < 8; ++k) {
      I_k_low[k] = pixel_offset[x + k] + tap;
      I_k_high[k] = pixel_offset[x + k + 8] + tap;
    }

    // Load the 16 absolute indices, split into two __m256i vectors
    __m256i v_I_low = _mm256_loadu_si256((const __m256i*)I_k_low);
    __m256i v_I_high = _mm256_loadu_si256((const __m256i*)I_k_high);

    // --- Calculate Relative Indices (J_k = I_k - begin1) ---

    __m256i v_J_low = _mm256_sub_epi32(v_I_low, v_begin1);
    __m256i v_J_high = _mm256_sub_epi32(v_I_high, v_begin1);

    // --- Calculate Mask B (ML_B and MH_B) ---
    // Mask: 0xFFFFFFFF if J_k >= 8, 0x00000000 if J_k < 8
    // _mm256_cmpgt_epi32(a, b) computes a > b. We want J_k >= 8, so we use J_k > 7.
    const __m256i v_7 = _mm256_set1_epi32(7);

    __m256i v_Mask_low = _mm256_cmpgt_epi32(v_J_low, v_7);
    __m256i v_Mask_high = _mm256_cmpgt_epi32(v_J_high, v_7);

    // Store the float mask vectors (Mask B)
    tap_vectors[tap]->ML_B = _mm256_castsi256_ps(v_Mask_low);
    tap_vectors[tap]->MH_B = _mm256_castsi256_ps(v_Mask_high);

    // --- Calculate Permutation Indices for Source B (PH_B and PL_B) ---
    // Index B is J_k - 8 (only for elements where J_k >= 8)
    __m256i v_Jm8_low = _mm256_sub_epi32(v_J_low, v_8);
    __m256i v_Jm8_high = _mm256_sub_epi32(v_J_high, v_8);

    // Select: (Mask B) ? (J_k - 8) : 0
    // _mm256_blendv_epi8 can be used as a general purpose blend for 32-bit integers
    // Note: The index '0' for the ignored lane doesn't matter, as the corresponding
    // output element will be selected from Source A, not B.
    tap_vectors[tap]->PL_B = _mm256_blendv_epi8(v_zero, v_Jm8_low, v_Mask_low);
    tap_vectors[tap]->PH_B = _mm256_blendv_epi8(v_zero, v_Jm8_high, v_Mask_high);

    // --- Calculate Permutation Indices for Source A (PH_A and PL_A) ---
    // Index A is J_k (only for elements where J_k < 8)
    // Select: (Mask B) ? 0 : J_k
    // The inverse mask can be created by NOTting the mask (using XOR with all ones, or NOT equivalent)
    // Since we want NOT Mask B to select J_k, we use the original mask to select 0.

    // A simpler way: J_k already contains the correct index (0-7). We just need to zero it out
    // where it's NOT needed (i.e., where Mask B is set).

    // Inverse Mask: 0xFFFFFFFF if J_k < 8, 0x00000000 if J_k >= 8
    __m256i v_InvMask_low = _mm256_xor_si256(v_Mask_low, _mm256_set1_epi32(0xFFFFFFFF));
    __m256i v_InvMask_high = _mm256_xor_si256(v_Mask_high, _mm256_set1_epi32(0xFFFFFFFF));

    // Select: (Inv Mask) ? J_k : 0 
    tap_vectors[tap]->PL_A = _mm256_blendv_epi8(v_zero, v_J_low, v_InvMask_low);
    tap_vectors[tap]->PH_A = _mm256_blendv_epi8(v_zero, v_J_high, v_InvMask_high);
  }
}

// Helper for permutex style horizontal resampling 32 bit float
// Safe partial load for 1-15 floats, padding with zeros to avoid NaN contamination
// Using jump tables instead of multiple if-else, each case is extremely
// optimized looking at the generated assembly..
static void _mm256_load_512_partial_safe(__m256 &A, __m256 &B, const float* src_ptr, int floats_to_load) {
  if (floats_to_load == 1) {
    A = _mm256_setr_ps(src_ptr[0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 2) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 3) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 4) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], 0.0f, 0.0f, 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 5) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], 0.0f, 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 6) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], 0.0f, 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 7) {
    A = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[4], src_ptr[5], src_ptr[6], 0.0f);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 8) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setzero_ps();
  }
  else if (floats_to_load == 9) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (floats_to_load == 10) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (floats_to_load == 11) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], src_ptr[10], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (floats_to_load == 12) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], src_ptr[10], src_ptr[11], 0.0f, 0.0f, 0.0f, 0.0f);
  }
  else if (floats_to_load == 13) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], src_ptr[10], src_ptr[11], src_ptr[12], 0.0f, 0.0f, 0.0f);
  }
  else if (floats_to_load == 14) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], src_ptr[10], src_ptr[11], src_ptr[12], src_ptr[13], 0.0f, 0.0f);
  }
  else if (floats_to_load == 15) {
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_setr_ps(src_ptr[8], src_ptr[9], src_ptr[10], src_ptr[11], src_ptr[12], src_ptr[13], src_ptr[14], 0.0f);
  }
  else if (floats_to_load == 16) { // cannot happen
    A = _mm256_loadu_ps(src_ptr);
    B = _mm256_loadu_ps(src_ptr + 8);
  }
  else {
    A = _mm256_setzero_ps(); // n/a cannot happen
    B = _mm256_setzero_ps(); // n/a cannot happen
  }
}


// resize_h_planar_float_avx2_xxx_vstripe_ks4 method #2: permutex-based, 16 pixel test version
// like resize_h_planar_float_avx512_permutex_vstripe_ks4
void resize_h_planar_float_avx2_permutex_vstripe_ks4_pix16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  constexpr int PIXELS_AT_A_TIME = 2 * 8; // Process eight pixels in parallel in AVX2

  // Pre-checked for permutex-based upsampling: the source pixels will surely fit within single 2x8 float loads
  // The right edge handling will be done via safe partial loads when needed, loading 2x8 pixels at once
  // may not be safe there.

  // 'source_overread_beyond_targetx' marks the x position in the target (output) scanline where,
  // if we process N pixels at a time (e.g., 8 for AVX2), the filter kernel may overread the source
  // buffer near the right edge due to kernel size and pixel offsets. Beyond this value, it is no
  // longer safe to read N source pixels at once from pixel_offset[].

  // For x positions < source_overread_beyond_targetx, it is safe to load N source pixels at once.
  // For x positions >= source_overread_beyond_targetx, we must use a safer loading method (e.g.,
  // partial loads with padding) to avoid out-of-bounds memory access.

  // permutex is even more special: the safety analysis is performed only for the beginning of each
  // block of 16 pixels processed at a time, so only the source loads for the offset position of
  // every 16th target pixel are considered. This is 'safelimit_16_pixels_each16th_target'.
  // The program's safe limits are pre-calculated during program setup.

  const int width_safe_mod = (program->safelimit_16_pixels_each16th_target.overread_possible ? program->safelimit_16_pixels_each16th_target.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // coeff + filter_size*0 to filter_size*7 when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16);

  // Ensure that coefficient loading is safe for 4 float loads,
  // if less than 4, padded with zeros till filter_size_alignment.
  assert(program->filter_size_alignment >= 4);

  const int max_scanlines = program->max_scanlines;

  // Example setup before the main loop (assuming memory allocation for 4 instances)
  PermuteVectors_AVX2 taps[4];
  PermuteVectors_AVX2* tap_pointers[4] = { &taps[0], &taps[1], &taps[2], &taps[3] };

  for (int y_from = 0; y_from < height; y_from += max_scanlines) {
    int y_to = std::min((int)(y_from + max_scanlines), height);
    // Reset current_coeff for the start of the stripe
    const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float; // +iYstart * filter_size;

    int x = 0;

    // This 'auto' lambda construct replaces the need of templates
    auto do_h_float_core = [&](auto partial_load) {

      // Call precalculation once per x position
      precalculate_cross_perm_avx2(program->pixel_offset.data(), x, tap_pointers);

      // Assumes 'filter_size_alignment' <= 4, 'target_size_alignment' >= 16
      // Prepare 4 coefs per pixel for 16 pixels in transposed V-form at once before the height loop.

      // ---------------------------------------------------------------------------
      // 1. Load Coefficients
      // ---------------------------------------------------------------------------
      // We process 16 pixels total.
      // Low Group (Pixels 0-7): Loaded into coef_0..coef_3
      // High Group (Pixels 8-15): Loaded into coef_4..coef_7

      // filter_size is typically the stride in floats (e.g., 4)

      // -- Load Low Group (Pixels 0,1,2,3 and 4,5,6,7 interleaved for Transpose macro) --
      __m256 coef_0 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
      __m256 coef_1 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
      __m256 coef_2 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
      __m256 coef_3 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);

      // -- Load High Group (Pixels 8,9,10,11 and 12,13,14,15 interleaved) --
      // Note: Offsets are shifted by 8 relative to the Low Group
      __m256 coef_4 = _mm256_load_2_m128(current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m256 coef_5 = _mm256_load_2_m128(current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m256 coef_6 = _mm256_load_2_m128(current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m256 coef_7 = _mm256_load_2_m128(current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      // ---------------------------------------------------------------------------
      // 2. Transpose
      // ---------------------------------------------------------------------------
      // After transpose:
      // coef_0 -> Tap 0 for Pixels 0-7
      // coef_1 -> Tap 1 for Pixels 0-7 ... etc
      _MM_TRANSPOSE8_LANE4_PS(coef_0, coef_1, coef_2, coef_3);

      // coef_4 -> Tap 0 for Pixels 8-15
      // coef_5 -> Tap 1 for Pixels 8-15 ... etc
      _MM_TRANSPOSE8_LANE4_PS(coef_4, coef_5, coef_6, coef_7);

      const int begin1 = program->pixel_offset[x + 0];
      // These indexes are guaranteed to be 0..15 due to the earlier analysis,
      // and can be used for the indexing parameter in combiner blendm mask, _mm256_permutevar8x32_ps
      float* AVS_RESTRICT dst_ptr = dst + x + y_from * dst_pitch;
      const float* src_ptr = src + begin1 + y_from * src_pitch;

      // for partial_load only
      const int remaining = program->source_size - begin1;
      const int floats_to_load = remaining >= 16 ? 16 : remaining;

      for (int y = y_from; y < y_to; ++y) {

        // process scanline y
        __m256 A;
        __m256 B;

        //__m256 data_src;
        // We'll need exactly 2x8 floats starting from src+begin1
        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline. This handles cases where loading 8 floats
          // starting from 'src_ptr + beginX' might exceed the source buffer.
          _mm256_load_512_partial_safe(/*ref*/A, /*ref*/B, src_ptr, floats_to_load);
        }
        else {
          A = _mm256_loadu_ps(src_ptr);
          B = _mm256_loadu_ps(src_ptr + 8);
          // data_src = _mm256_loadu_ps(src_ptr); // load 8 source pixels, can contain garbage beyond the right edge in the last loop
        }

        // After we load 2x8 source pixels starting from begin1, we can be sure, that pixel_offset[x+0] .. pixel_offset[x+15] + 3 is
        // within valid source range. Pre-check chooses permutex method only if all needed pixels fit within these 16 loaded pixels.

        // Permute and Blend for Tap 0 (data_0)
        __m256 A_perm_0L = _mm256_permutevar8x32_ps(A, taps[0].PL_A);
        __m256 B_perm_0L = _mm256_permutevar8x32_ps(B, taps[0].PL_B);
        __m256 data_0L = _mm256_blendv_ps(A_perm_0L, B_perm_0L, taps[0].ML_B); // Result: dst[x+0]..dst[x+7]

        __m256 A_perm_0H = _mm256_permutevar8x32_ps(A, taps[0].PH_A);
        __m256 B_perm_0H = _mm256_permutevar8x32_ps(B, taps[0].PH_B);
        __m256 data_0H = _mm256_blendv_ps(A_perm_0H, B_perm_0H, taps[0].MH_B); // Result: dst[x+8]..dst[x+15]

        // Repeat for data_1, data_2, data_3...
        // Permute and Blend for Tap 1 (data_1)
        __m256 A_perm_1L = _mm256_permutevar8x32_ps(A, taps[1].PL_A);
        __m256 B_perm_1L = _mm256_permutevar8x32_ps(B, taps[1].PL_B);
        __m256 data_1L = _mm256_blendv_ps(A_perm_1L, B_perm_1L, taps[1].ML_B); // Result: dst[x+0]..dst[x+7]

        __m256 A_perm_1H = _mm256_permutevar8x32_ps(A, taps[1].PH_A);
        __m256 B_perm_1H = _mm256_permutevar8x32_ps(B, taps[1].PH_B);
        __m256 data_1H = _mm256_blendv_ps(A_perm_1H, B_perm_1H, taps[1].MH_B); // Result: dst[x+8]..dst[x+15]
        // Permute and Blend for Tap 2 (data_2)
        __m256 A_perm_2L = _mm256_permutevar8x32_ps(A, taps[2].PL_A);
        __m256 B_perm_2L = _mm256_permutevar8x32_ps(B, taps[2].PL_B);
        __m256 data_2L = _mm256_blendv_ps(A_perm_2L, B_perm_2L, taps[2].ML_B); // Result: dst[x+0]..dst[x+7]

        __m256 A_perm_2H = _mm256_permutevar8x32_ps(A, taps[2].PH_A);
        __m256 B_perm_2H = _mm256_permutevar8x32_ps(B, taps[2].PH_B);
        __m256 data_2H = _mm256_blendv_ps(A_perm_2H, B_perm_2H, taps[2].MH_B); // Result: dst[x+8]..dst[x+15]
        // Permute and Blend for Tap 3 (data_3)
        __m256 A_perm_3L = _mm256_permutevar8x32_ps(A, taps[3].PL_A);
        __m256 B_perm_3L = _mm256_permutevar8x32_ps(B, taps[3].PL_B);
        __m256 data_3L = _mm256_blendv_ps(A_perm_3L, B_perm_3L, taps[3].ML_B); // Result: dst[x+0]..dst[x+7]

        __m256 A_perm_3H = _mm256_permutevar8x32_ps(A, taps[3].PH_A);
        __m256 B_perm_3H = _mm256_permutevar8x32_ps(B, taps[3].PH_B);
        __m256 data_3H = _mm256_blendv_ps(A_perm_3H, B_perm_3H, taps[3].MH_B); // Result: dst[x+8]..dst[x+15]

        // perm_0 .. perm_3 lo and hi contain the indexes to permute data_src into the correct order
        // for each of the 16 output pixels so they index into 0..15 (guaranteed) range of the source data loaded above
        // Since we have only 8-wide permute, we have to do two separate permutes and then blend the results together.

        // Assuming these eight data vectors have been calculated via Permute+Blend:
        // __m256 data_0L, data_0H;
        // __m256 data_1L, data_1H;
        // __m256 data_2L, data_2H;
        // __m256 data_3L, data_3H;

        // ---------------------------------------------------------------------------
        // 3. Calculation
        // ---------------------------------------------------------------------------

        // --- Low Half (Pixels 0-7) ---
        // Uses: data_0L..3L and coef_0..3

        __m256 result0L = _mm256_mul_ps(data_0L, coef_0);       // Tap 0
        __m256 result1L = _mm256_mul_ps(data_2L, coef_2);       // Tap 2

        result0L = _mm256_fmadd_ps(data_1L, coef_1, result0L);  // Tap 1
        result1L = _mm256_fmadd_ps(data_3L, coef_3, result1L);  // Tap 3

        __m256 final_result_L = _mm256_add_ps(result0L, result1L);


        // --- High Half (Pixels 8-15) ---
        // Uses: data_0H..3H and coef_4..7
        // Note: coef_4 corresponds to Tap 0 for this group, coef_5 to Tap 1, etc.

        __m256 result0H = _mm256_mul_ps(data_0H, coef_4);       // Tap 0 (High)
        __m256 result1H = _mm256_mul_ps(data_2H, coef_6);       // Tap 2 (High)

        result0H = _mm256_fmadd_ps(data_1H, coef_5, result0H);  // Tap 1 (High)
        result1H = _mm256_fmadd_ps(data_3H, coef_7, result1H);  // Tap 3 (High)

        __m256 final_result_H = _mm256_add_ps(result0H, result1H);

        // ---------------------------------------------------------------------------
        // 4. Store
        // ---------------------------------------------------------------------------

        _mm256_stream_ps(dst_ptr, final_result_L);     // dst[0..7]
        _mm256_stream_ps(dst_ptr + 8, final_result_H); // dst[8..15]

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }
      current_coeff += filter_size * 16;
      }; // end of lambda

    // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm_loadu_ps
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer '_mm256_load_partial_safe'
    }
  }

}



template<int filtersizemod4>
void resize_h_planar_float_avx2_gather_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{

  // FIXME: this analysis could be done once in the dispatcher instead of per call, since the program is constant
  bool bDoGather = program->resize_h_planar_gather_permutex_vstripe_check(8/*iSamplesInThGroup*/, 8 /*permutex_index_diff_limit*/, 4 /*kernel_size*/);
  if (bDoGather)
  {
    resize_h_planar_float_avx2_transpose_vstripe_ks4<filtersizemod4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  } // if bDoGather
  else
  {
    resize_h_planar_float_avx2_permutex_vstripe_ks4(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  }

  // versus the original one, which is a generic upsample/downsample method
  //resize_h_planar_float_avx_transpose_vstripe_ks4<filtersizemod4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

// Instantiate them
template void resize_h_planar_float_avx2_transpose_vstripe_ks4<0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_transpose_vstripe_ks4<1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_transpose_vstripe_ks4<2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_transpose_vstripe_ks4<3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// Instantiate them
template void resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx2_gather_permutex_vstripe_ks4<3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

//--------------------------------------------------------------------------

// AVX2 Horizontal float

// Three helpers, each for processing 4 target pixels from 16, 8 and 4 source pixel/coeff pairs.

// Helper, implemented for AVX2 (simulating zext)
// zero-extend 128-bit float vector to 256-bit float vector
AVS_FORCEINLINE static __m256 _mm256_zextps128_ps256_avx2(__m128 a)
{
  __m256 zero_v = _mm256_setzero_ps();
  return _mm256_insertf128_ps(zero_v, a, 0);
}

// 4 target pixels, each from 16 source pixel/coeff pair
// Called only when accessing 16 source pixels and coefficients at a time is safe
// AVX2 OPTIMIZATION: Process 2x8 blocks to simulate the 16-block stride
AVS_FORCEINLINE static void process_pix4_coeff16_h_float_core_avx2(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4)
{
  // --- Block A: First 8 floats ---
  __m256 data_1_a = _mm256_loadu_ps(src + begin1);
  __m256 coeff_1_a = _mm256_loadu_ps(current_coeff);
  result1 = _mm256_fmadd_ps(data_1_a, coeff_1_a, result1);
  __m256 data_1_b = _mm256_loadu_ps(src + begin1 + 8);
  __m256 coeff_1_b = _mm256_loadu_ps(current_coeff + 8);
  result1 = _mm256_fmadd_ps(data_1_b, coeff_1_b, result1);

  __m256 data_2_a = _mm256_loadu_ps(src + begin2);
  __m256 coeff_2_a = _mm256_loadu_ps(current_coeff + 1 * filter_size);
  result2 = _mm256_fmadd_ps(data_2_a, coeff_2_a, result2);
  __m256 data_2_b = _mm256_loadu_ps(src + begin2 + 8);
  __m256 coeff_2_b = _mm256_loadu_ps(current_coeff + 1 * filter_size + 8);
  result2 = _mm256_fmadd_ps(data_2_b, coeff_2_b, result2);

  __m256 data_3_a = _mm256_loadu_ps(src + begin3);
  __m256 coeff_3_a = _mm256_loadu_ps(current_coeff + 2 * filter_size);
  result3 = _mm256_fmadd_ps(data_3_a, coeff_3_a, result3);
  __m256 data_3_b = _mm256_loadu_ps(src + begin3 + 8);
  __m256 coeff_3_b = _mm256_loadu_ps(current_coeff + 2 * filter_size + 8);
  result3 = _mm256_fmadd_ps(data_3_b, coeff_3_b, result3);

  __m256 data_4_a = _mm256_loadu_ps(src + begin4);
  __m256 coeff_4_a = _mm256_loadu_ps(current_coeff + 3 * filter_size);
  result4 = _mm256_fmadd_ps(data_4_a, coeff_4_a, result4);
  __m256 data_4_b = _mm256_loadu_ps(src + begin4 + 8);
  __m256 coeff_4_b = _mm256_loadu_ps(current_coeff + 3 * filter_size + 8);
  result4 = _mm256_fmadd_ps(data_4_b, coeff_4_b, result4);

}

// 4 target pixels, each from 8 source pixel/coeff pair
// Called only when accessing 8 source pixels and coefficients at a time is safe
AVS_FORCEINLINE static void process_pix4_coeff8_h_float_core_avx2(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4)
{
  // Load 8 source floats for each of the four beginning source offsets
  // Load 8 coefficients for each of the four output pixels
  __m256 data_1 = _mm256_loadu_ps(src + begin1);
  __m256 coeff_1 = _mm256_load_ps(current_coeff);                     // 8 coeffs for pixel 1
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
// Optimized: Uses 256-bit MUL directly on zero-extended loads.
AVS_FORCEINLINE static void process_pix4_coeff4_h_float_core_first_avx2(
  const float* src,
  int begin1, int begin2, int begin3, int begin4,
  const float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4)
{
  // Pixel 1
  __m128 data_1 = _mm_loadu_ps(src + begin1);
  __m128 coeff_1 = _mm_load_ps(current_coeff);
  __m128 temp_result_1 = _mm_mul_ps(data_1, coeff_1);
  result1 = _mm256_zextps128_ps256_avx2(temp_result_1);

  // Pixel 2
  __m128 data_2 = _mm_loadu_ps(src + begin2);
  __m128 coeff_2 = _mm_load_ps(current_coeff + 1 * filter_size);
  __m128 temp_result_2 = _mm_mul_ps(data_2, coeff_2);
  result2 = _mm256_zextps128_ps256_avx2(temp_result_2);

  // Pixel 3
  __m128 data_3 = _mm_loadu_ps(src + begin3);
  __m128 coeff_3 = _mm_load_ps(current_coeff + 2 * filter_size);
  __m128 temp_result_3 = _mm_mul_ps(data_3, coeff_3);
  result3 = _mm256_zextps128_ps256_avx2(temp_result_3);

  // Pixel 4
  __m128 data_4 = _mm_loadu_ps(src + begin4);
  __m128 coeff_4 = _mm_load_ps(current_coeff + 3 * filter_size);
  __m128 temp_result_4 = _mm_mul_ps(data_4, coeff_4);
  result4 = _mm256_zextps128_ps256_avx2(temp_result_4);
}

#define HAS_COEFF16_STEP

// filtersize_hint: special: 0..4 for 4,8,16,24,32. Generic: -1
// filter_size is an aligned value and always multiple of 8 (prerequisite)
template<bool safe_aligned_mode, int filtersize_hint>
AVS_FORCEINLINE static void process_four_pixels_h_float_pix4of16_ks_4_8_16_avx2(
  const float* src_ptr,
  int begin1, int begin2, int begin3, int begin4,
  float* current_coeff,
  int filter_size,
  __m256& result1, __m256& result2, __m256& result3, __m256& result4,
  int kernel_size)
{

  // very special case: filter size <= 4
  if constexpr (safe_aligned_mode) {
    if constexpr(filtersize_hint == 0) {
      // Process 4 target pixels and 4 source pixels/coefficients at a time
      // XMM-based loop internally, but returns __m256 with upper 128 cleared
      // Do not assume initialized zeros in result1..4, they will be set here.
      process_pix4_coeff4_h_float_core_first_avx2(
        src_ptr + 0, begin1, begin2, begin3, begin4,
        current_coeff + 0,
        filter_size,
        result1, result2, result3, result4);
      return;
    }
  }

  // not: when filtersize_hint == -1, it is not covered with filtersize_hint maximum 4 case, which means
  // that real filter size is over 32.
  // Thus we surely can have 2x16 coeff processing here.

  int i = 0;
#ifdef HAS_COEFF16_STEP
  if constexpr(filtersize_hint == -1) {
    // Handle 2x16 coeffs first, since we know that real filter size is over 32 here, because of filtersize_hint == -1
    const int ksmod16_sure = 32;
    // this will be unrolled probably
    for (; i < ksmod16_sure; i += 16) {
      // Direct AVX2 adaptation: The core function now updates resultX (YMM) directly
      process_pix4_coeff16_h_float_core_avx2(
        src_ptr + i, begin1, begin2, begin3, begin4,
        current_coeff + i,
        filter_size,
        result1, result2, result3, result4);
    }
    // processed 32 coeffs from the kernel, do the rest
    const int ksmod16 = safe_aligned_mode ? (filter_size / 16 * 16) : (kernel_size / 16 * 16);
    // Process 4 target pixels and 16 source pixels/coefficients at a time
    for (; i < ksmod16; i += 16) {
      // Direct AVX2 adaptation: The core function now updates resultX (YMM) directly
      process_pix4_coeff16_h_float_core_avx2(
        src_ptr + i, begin1, begin2, begin3, begin4,
        current_coeff + i,
        filter_size,
        result1, result2, result3, result4);
    }
  }
  else {
    // do by 16 coeffs till possible
    if (filtersize_hint >= 2) {
      const int ksmod16 = safe_aligned_mode ? (filter_size / 16 * 16) : (kernel_size / 16 * 16);
      // Process 4 target pixels and 16 source pixels/coefficients at a time
      for (; i < ksmod16; i += 16) {
        // Direct AVX2 adaptation: The core function now updates resultX (YMM) directly
        process_pix4_coeff16_h_float_core_avx2(
          src_ptr + i, begin1, begin2, begin3, begin4,
          current_coeff + i,
          filter_size,
          result1, result2, result3, result4);
      }
    }
  }

  // filter sizes 16 or 32 can return here
  if constexpr (safe_aligned_mode && (filtersize_hint == 2 || filtersize_hint == 4)) {
    return;
  }

  if constexpr (!safe_aligned_mode) {
    if (i == kernel_size) return; // kernel_size is not known compile time
  }
#else
  // no coeff16 step
  if constexpr (filtersize_hint == -1) {
    // Handle 4x8 coeffs first, since we know that real filter size is over 32 here, because of filtersize_hint == -1
    const int ksmod8_sure = 32;
    // this will be unrolled probably
    for (; i < ksmod8_sure; i += 8) {
      process_pix4_coeff8_h_float_core_avx2(
        src_ptr + i, begin1, begin2, begin3, begin4,
        current_coeff + i,
        filter_size,
        result1, result2, result3, result4);
    }
  }
#endif

  // When to do the coeff8 step if we had the coeff16 step enabled:
  // not safe-aligned mode: always. E.g. kernel_size == 28 -> 16 done, now 10 rest, do 8 next
  // filtersize_hint == -1: not-compile-time known filtersize (kernel_size / 16 * 16 done, rest follows)
  // filtersize_hint == 1 or 3: 0*16 or 1*16 done, now do 1*8
#ifdef HAS_COEFF16_STEP
  if (!safe_aligned_mode || filtersize_hint == -1 || filtersize_hint == 1 || filtersize_hint == 3) {
#else
  if (true) {
#endif
    // 32 bytes contain 8 floats. We will use 256-bit registers (YMM).
    const int ksmod8 = safe_aligned_mode ? (filter_size / 8 * 8) : (kernel_size / 8 * 8);

    // Process 4 target pixels and 8 source pixels/coefficients at a time (YMM-based loop)
    for (; i < ksmod8; i += 8) {
      process_pix4_coeff8_h_float_core_avx2(
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
    // real_kernel_size is used here, it's guaranteed that reading real_kernel_size elements
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

      // Pixel 1
      __m128 data_1 = _mm_loadu_ps(src_ptr1 + i);
      __m128 coeff_1 = _mm_load_ps(current_coeff + i);
      __m128 temp_result_1 = _mm_mul_ps(data_1, coeff_1);
      result1 = _mm256_add_ps(result1, _mm256_zextps128_ps256_avx2(temp_result_1));

      // Pixel 2
      __m128 data_2 = _mm_loadu_ps(src_ptr2 + i);
      __m128 coeff_2 = _mm_load_ps(current_coeff2 + i);
      __m128 temp_result_2 = _mm_mul_ps(data_2, coeff_2);
      result2 = _mm256_add_ps(result2, _mm256_zextps128_ps256_avx2(temp_result_2));

      // Pixel 3
      __m128 data_3 = _mm_loadu_ps(src_ptr3 + i);
      __m128 coeff_3 = _mm_load_ps(current_coeff3 + i);
      __m128 temp_result_3 = _mm_mul_ps(data_3, coeff_3);
      result3 = _mm256_add_ps(result3, _mm256_zextps128_ps256_avx2(temp_result_3));

      // Pixel 4
      __m128 data_4 = _mm_loadu_ps(src_ptr4 + i);
      __m128 coeff_4 = _mm_load_ps(current_coeff4 + i);
      __m128 temp_result_4 = _mm_mul_ps(data_4, coeff_4);
      result4 = _mm256_add_ps(result4, _mm256_zextps128_ps256_avx2(temp_result_4));

      i += 4;
      if (i == kernel_size) return;
    }

    const int ksmod2 = kernel_size / 2 * 2;

    // -------------------------------------------------------------------
    // Mod 2 Block (2 elements for four pixels using __m128)
    // -------------------------------------------------------------------
    if (i < ksmod2) {
      // _mm_load_sd (vmovsd) loads 64 bits and zeroes everything else in the register.
      // Vector looks like: [0, 0, 0, 0, 0, 0, val1, val0]
      // We can use the exact same FMA trick.

      auto load_2_floats_as_ymm = [](const float* p) {
        return _mm256_zextps128_ps256_avx2(_mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double*>(p))));
        };

      result1 = _mm256_fmadd_ps(load_2_floats_as_ymm(src_ptr1 + i), load_2_floats_as_ymm(current_coeff + i), result1);
      result2 = _mm256_fmadd_ps(load_2_floats_as_ymm(src_ptr2 + i), load_2_floats_as_ymm(current_coeff2 + i), result2);
      result3 = _mm256_fmadd_ps(load_2_floats_as_ymm(src_ptr3 + i), load_2_floats_as_ymm(current_coeff3 + i), result3);
      result4 = _mm256_fmadd_ps(load_2_floats_as_ymm(src_ptr4 + i), load_2_floats_as_ymm(current_coeff4 + i), result4);

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

      // Using the helper for the last one to be safe/consistent with scalar ops
      result1 = _mm256_add_ps(result1, _mm256_zextps128_ps256_avx2(_mm_set_ss(final_scalar1)));
      result2 = _mm256_add_ps(result2, _mm256_zextps128_ps256_avx2(_mm_set_ss(final_scalar2)));
      result3 = _mm256_add_ps(result3, _mm256_zextps128_ps256_avx2(_mm_set_ss(final_scalar3)));
      result4 = _mm256_add_ps(result4, _mm256_zextps128_ps256_avx2(_mm_set_ss(final_scalar4)));
      // i is now equal to kernel_size (i++)
    }
  }
}


template<bool is_safe, int filtersize_hint>
AVS_FORCEINLINE static void process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16_avx2(
  const float* src, int x, float* current_coeff_base,
  int filter_size, // 8, 16, 24, 32 are quasi-constexpr here, others not compile-time known but still aligned to 8
  float* dst,
  ResamplingProgram* program)
{
  assert(program->filter_size_alignment == 8);

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

  process_four_pixels_h_float_pix4of16_ks_4_8_16_avx2<is_safe, filtersize_hint>(
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

  process_four_pixels_h_float_pix4of16_ks_4_8_16_avx2<is_safe, filtersize_hint>(
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
  _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result256_low);


  // --- Block 3: Pixels 8, 9, 10, 11 ---
  __m256 result8 = zero256;
  __m256 result9 = zero256;
  __m256 result10 = zero256;
  __m256 result11 = zero256;

  int begin8 = program->pixel_offset[x + 8];
  int begin9 = program->pixel_offset[x + 9];
  int begin10 = program->pixel_offset[x + 10];
  int begin11 = program->pixel_offset[x + 11];

  process_four_pixels_h_float_pix4of16_ks_4_8_16_avx2<is_safe, filtersize_hint>(
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

  process_four_pixels_h_float_pix4of16_ks_4_8_16_avx2<is_safe, filtersize_hint>(
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
  _mm256_stream_ps(reinterpret_cast<float*>(dst + x + 8), result256_high);
}

// filtersizealigned8: special: 0, 1..4, Generic : -1
template<int filtersize_hint>
static void internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
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
  // Align safe zone to 16 pixels.
  // The safe path reads filter_size (aligned) source floats for each target pixel: the limit is
  // safelimit_filter_size_aligned, not safelimit_16_pixels (16 is the target pixels per step, not the read width).
  // A read past the source row would multiply garbage (possibly NaN) by the zero padding coefficients.
  const int w_safe_mod16 = (program->safelimit_filter_size_aligned.overread_possible ? program->safelimit_filter_size_aligned.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  for (int y = 0; y < height; y++) {
    float* current_coeff_base = program->pixel_coefficient_float;

    // Process safe aligned pixels
    for (int x = 0; x < w_safe_mod16; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16_avx2<true, filtersize_hint>(src, x, current_coeff_base, filter_size, dst, program);
    }

    // Process up to the actual kernel size (unsafe zone)
    for (int x = w_safe_mod16; x < width; x += PIXELS_AT_A_TIME) {
      process_sixteen_pixels_h_float_pix16_sub4_ks_4_8_16_avx2<false, filtersize_hint>(src, x, current_coeff_base, filter_size, dst, program);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

// Winner implementation: resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16;
void resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = program->filter_size;
  // Expected alignment
  assert(program->filter_size_alignment >= 8);

  // Dispatcher template now supports filter_size aligned to 8 (8, 16, 24, 32) and a special case for <=4
  // Larger filter sizes will use the generic method (-1) which still benefit from 16-8-4 coeff processing blocks.
  if (filter_size == 1 * 8)
    if (program->filter_size_real <= 4)
      internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<0>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel); // Internally optimized for 4
    else
      internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel); // Internally optimized for 8
  else if (filter_size == 2 * 8) // Internally optimized for 16
    internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 3 * 8) // Internally optimized for 16+8
    internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size == 4 * 8) // Internally optimized for 2*16
    internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<4>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size, internally optimized for calculating coeffs in N*16 + 8 + 4 + 2 + 1 blocks
    internal_resizer_h_avx2_generic_float_pix16_sub4_ks_4_8_16<-1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

