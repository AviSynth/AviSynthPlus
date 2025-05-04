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

#include "resample_sse.h"
#include <avs/config.h>
#include "../core/internal.h"

#include <avs/alignment.h>
#include <avs/minmax.h>

// experimental simd includes for avx2 compiled files
#if defined (__GNUC__) && ! defined (__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

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

  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

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
AVS_FORCEINLINE static void process_two_8pixels_h_float_core(const float* src, int begin1, int begin2, int i, float* current_coeff, int filter_size, __m256& result1, __m256& result2) {
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
    process_two_8pixels_h_float_core(src_ptr, begin1, begin2, i, current_coeff + i, filter_size, result1, result2);
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

  const int w_safe_mod8 = (program->overread_possible ? program->source_overread_beyond_targetx : width) / 8 * 8;

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

void resizer_h_avx2_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size = program->filter_size;
  assert(program->filter_size_alignment == 8);

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
/*
void resize_v_avx2_planar_uint8_t(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
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
      _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result128);

    }
    dst += dst_pitch;
    current_coeff += filter_size;
  }
}
*/

void resize_v_avx2_planar_uint8_t(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
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
        // no need wmod16, alignment is safe at least 32
        for (int x = 0; x < width; x += 32) { // was +=16

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


            _mm_store_si128(reinterpret_cast<__m128i*>(dst + x), result128);
            _mm_store_si128(reinterpret_cast<__m128i*>(dst + x + 16), result128_2);

        }
        dst += dst_pitch;
        current_coeff += filter_size;
    }
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
        if (!lessthan16bit) {
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
        if (!lessthan16bit) {
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
      if (!lessthan16bit) {
        result_lo = _mm256_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm256_add_epi32(result_hi, shiftfromsigned);
      }
      // shift back integer arithmetic 13 bits precision
      result_lo = _mm256_srai_epi32(result_lo, FPScale16bits);
      result_hi = _mm256_srai_epi32(result_hi, FPScale16bits);

      __m256i result_2x8x_uint16 = _mm256_packus_epi32(result_lo, result_hi);
      if (lessthan16bit) {
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

// avx2 16bit
template void resizer_h_avx2_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
// avx2 10-14bit
template void resizer_h_avx2_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);



// avx2 16
template void resize_v_avx2_planar_uint16_t<false>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
// avx2 10-14bit
template void resize_v_avx2_planar_uint16_t<true>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);


// Transpose-based SIMD
void resize_h_planar_float_avx_transpose(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
    int filter_size = program->filter_size;

    const float* AVS_RESTRICT current_coeff;

    src_pitch = src_pitch / sizeof(float);
    dst_pitch = dst_pitch / sizeof(float);

    float* src = (float*)src8;
    float* dst = (float*)dst8;

    const int kernel_size = program->filter_size_real;
    const int ksmod4 = kernel_size / 4 * 4;
//    const int ksmod8 = kernel_size / 8 * 8;
#if 0

    for (int y = 0; y < height; y ++) {
        current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

        float* AVS_RESTRICT dst2_ptr = dst + y * dst_pitch;
        const float* src_ptr = src + y * src_pitch;

        for (int x = 0; x < width; x += 8) {

            __m256 result = _mm256_setzero_ps();

            for (int i = 0; i < ksmod4; i += 4) {

                __m256 data_1_data_5 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 0] + i, src_ptr + program->pixel_offset[x + 4] + i);
                __m256 data_2_data_6 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 1] + i, src_ptr + program->pixel_offset[x + 5] + i);
                __m256 data_3_data_7 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 2] + i, src_ptr + program->pixel_offset[x + 6] + i);
                __m256 data_4_data_8 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 3] + i, src_ptr + program->pixel_offset[x + 7] + i);

                __m256 coef_1_coef_5 = _mm256_load_2_m128(current_coeff + i + filter_size * 0, current_coeff + i + filter_size * 4);
                __m256 coef_2_coef_6 = _mm256_load_2_m128(current_coeff + i + filter_size * 1, current_coeff + i + filter_size * 5);
                __m256 coef_3_coef_7 = _mm256_load_2_m128(current_coeff + i + filter_size * 2, current_coeff + i + filter_size * 6);
                __m256 coef_4_coef_8 = _mm256_load_2_m128(current_coeff + i + filter_size * 3, current_coeff + i + filter_size * 7);

                _MM_TRANSPOSE8_LANE4_PS(data_1_data_5, data_2_data_6, data_3_data_7, data_4_data_8);
                _MM_TRANSPOSE8_LANE4_PS(coef_1_coef_5, coef_2_coef_6, coef_3_coef_7, coef_4_coef_8);

                result = _mm256_fmadd_ps(data_1_data_5, coef_1_coef_5, result);
                result = _mm256_fmadd_ps(data_2_data_6, coef_2_coef_6, result);
                result = _mm256_fmadd_ps(data_3_data_7, coef_3_coef_7, result);
                result = _mm256_fmadd_ps(data_4_data_8, coef_4_coef_8, result);

            }

            _mm256_store_ps(dst2_ptr + x, result);
            current_coeff += filter_size * 8;
        }
    }
#endif

    for (int y = 0; y < height; y+=2) {
        current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

        float* AVS_RESTRICT dst2_ptr = dst + y * dst_pitch;
        float* AVS_RESTRICT dst2_ptr2 = dst + (y + 1) * dst_pitch;
        const float* src_ptr = src + y * src_pitch;
        const float* src_ptr2 = src + (y + 1) * src_pitch;

        for (int x = 0; x < width; x += 8) {

            __m256 result = _mm256_setzero_ps();
            __m256 result2 = _mm256_setzero_ps();

            for (int i = 0; i < kernel_size; i += 4) { // is it always mod4 ?
                __m256 data_1_data_5 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 0] + i, src_ptr + program->pixel_offset[x + 4] + i);
                __m256 data_2_data_6 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 1] + i, src_ptr + program->pixel_offset[x + 5] + i);
                __m256 data_3_data_7 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 2] + i, src_ptr + program->pixel_offset[x + 6] + i);
                __m256 data_4_data_8 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 3] + i, src_ptr + program->pixel_offset[x + 7] + i);

                __m256 data_1_data_5_2 = _mm256_loadu_2_m128(src_ptr2 + program->pixel_offset[x + 0] + i, src_ptr2 + program->pixel_offset[x + 4] + i);
                __m256 data_2_data_6_2 = _mm256_loadu_2_m128(src_ptr2 + program->pixel_offset[x + 1] + i, src_ptr2 + program->pixel_offset[x + 5] + i);
                __m256 data_3_data_7_2 = _mm256_loadu_2_m128(src_ptr2 + program->pixel_offset[x + 2] + i, src_ptr2 + program->pixel_offset[x + 6] + i);
                __m256 data_4_data_8_2 = _mm256_loadu_2_m128(src_ptr2 + program->pixel_offset[x + 3] + i, src_ptr2 + program->pixel_offset[x + 7] + i);

                __m256 coef_1_coef_5 = _mm256_load_2_m128(current_coeff + i + filter_size * 0, current_coeff + i + filter_size * 4);
                __m256 coef_2_coef_6 = _mm256_load_2_m128(current_coeff + i + filter_size * 1, current_coeff + i + filter_size * 5);
                __m256 coef_3_coef_7 = _mm256_load_2_m128(current_coeff + i + filter_size * 2, current_coeff + i + filter_size * 6);
                __m256 coef_4_coef_8 = _mm256_load_2_m128(current_coeff + i + filter_size * 3, current_coeff + i + filter_size * 7);

                _MM_TRANSPOSE8_LANE4_PS(data_1_data_5, data_2_data_6, data_3_data_7, data_4_data_8);
                _MM_TRANSPOSE8_LANE4_PS(data_1_data_5_2, data_2_data_6_2, data_3_data_7_2, data_4_data_8_2);
                _MM_TRANSPOSE8_LANE4_PS(coef_1_coef_5, coef_2_coef_6, coef_3_coef_7, coef_4_coef_8);

                result = _mm256_fmadd_ps(data_1_data_5, coef_1_coef_5, result);
                result = _mm256_fmadd_ps(data_2_data_6, coef_2_coef_6, result);
                result = _mm256_fmadd_ps(data_3_data_7, coef_3_coef_7, result);
                result = _mm256_fmadd_ps(data_4_data_8, coef_4_coef_8, result);

                result2 = _mm256_fmadd_ps(data_1_data_5_2, coef_1_coef_5, result2);
                result2 = _mm256_fmadd_ps(data_2_data_6_2, coef_2_coef_6, result2);
                result2 = _mm256_fmadd_ps(data_3_data_7_2, coef_3_coef_7, result2);
                result2 = _mm256_fmadd_ps(data_4_data_8_2, coef_4_coef_8, result2);

            }

            // need to process last non-mod4 kernel samples in scalar way. or can we do over-read up to 3 kernel and source samples safely with main 4-kernel_samples loop ?

            _mm256_store_ps(dst2_ptr + x, result);
            _mm256_store_ps(dst2_ptr2 + x, result2);
            current_coeff += filter_size * 8;
        }
    }

}

// process kernel size from up to 4 - BilinearResize, BicubicResize or sinc up to taps=2
void resize_h_planar_float_avx_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
    int filter_size = program->filter_size;

    const float* AVS_RESTRICT current_coeff;

    src_pitch = src_pitch / sizeof(float);
    dst_pitch = dst_pitch / sizeof(float);

    float* src = (float*)src8;
    float* dst = (float*)dst8;

    current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    for (int x = 0; x < width; x += 8) 
    {
        __m256 coef_1_coef_5 = _mm256_load_2_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4);
        __m256 coef_2_coef_6 = _mm256_load_2_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5);
        __m256 coef_3_coef_7 = _mm256_load_2_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6);
        __m256 coef_4_coef_8 = _mm256_load_2_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7);
        
        _MM_TRANSPOSE8_LANE4_PS(coef_1_coef_5, coef_2_coef_6, coef_3_coef_7, coef_4_coef_8);

        float* AVS_RESTRICT dst_ptr = dst + x;
        const float* src_ptr = src;

        for (int y = 0; y < height; y++) 
        {
            __m256 data_1_data_5 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 0], src_ptr + program->pixel_offset[x + 4]);
            __m256 data_2_data_6 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 1], src_ptr + program->pixel_offset[x + 5]);
            __m256 data_3_data_7 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 2], src_ptr + program->pixel_offset[x + 6]);
            __m256 data_4_data_8 = _mm256_loadu_2_m128(src_ptr + program->pixel_offset[x + 3], src_ptr + program->pixel_offset[x + 7]);

            _MM_TRANSPOSE8_LANE4_PS(data_1_data_5, data_2_data_6, data_3_data_7, data_4_data_8);

            __m256 result = _mm256_mul_ps(data_1_data_5, coef_1_coef_5);
            result = _mm256_fmadd_ps(data_2_data_6, coef_2_coef_6, result);
            result = _mm256_fmadd_ps(data_3_data_7, coef_3_coef_7, result);
            result = _mm256_fmadd_ps(data_4_data_8, coef_4_coef_8, result);

            _mm256_store_ps(dst_ptr, result);

            dst_ptr += dst_pitch;
            src_ptr += src_pitch;
        }
        current_coeff += filter_size * 8;
    }

}