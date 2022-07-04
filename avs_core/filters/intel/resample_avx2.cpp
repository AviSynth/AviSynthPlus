// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://www.avisynth.org

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

//-------- 256 bit uint8_t Horizontals

AVS_FORCEINLINE static void process_two_16pixels_h_uint8_t(const uint8_t *src, int begin1, int begin2, int i, short *&current_coeff, int filter_size_numOfBlk16, __m256i &result1, __m256i &result2) {
  __m256i data_1 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i * 16)));
  __m256i data_2 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i * 16)));
  __m256i coeff_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_coeff)); // 16 coeffs
  __m256i coeff_2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(current_coeff + 1 * filter_size_numOfBlk16 * 16)); // 16 coeffs
  result1 = _mm256_add_epi32(result1, _mm256_madd_epi16(data_1, coeff_1));
  result2 = _mm256_add_epi32(result2, _mm256_madd_epi16(data_2, coeff_2));
  current_coeff += 16;
}

// filtersizealigned16: special: 1, 2. Generic: -1
template<int filtersizealigned16>
static void internal_resizer_h_avx2_generic_uint8_t(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height) {
  const int filter_size_numOfBlk16 = (filtersizealigned16 >= 1) ? filtersizealigned16 : (AlignNumber(program->filter_size, 16) / 16);
  __m256i zero = _mm256_setzero_si256();

  __m256i rounder256_1 = _mm256_setr_epi32(1 << (FPScale8bits - 1), 0, 0, 0, 0, 0, 0, 0);
  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;

    // 8 pixels!
    for (int x = 0; x < width; x += 8) {
      __m256i result1 = rounder256_1;
      __m256i result2 = result1;
      __m256i result3 = result1;
      __m256i result4 = result1;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, begin2
      for (int i = 0; i < filter_size_numOfBlk16; i++)
        process_two_16pixels_h_uint8_t(src, begin1, begin2, i, current_coeff, filter_size_numOfBlk16, result1, result2);
      current_coeff += 16 * filter_size_numOfBlk16; // because of dual pixel processing

      // begin3, begin4
      for (int i = 0; i < filter_size_numOfBlk16; i++)
        process_two_16pixels_h_uint8_t(src, begin3, begin4, i, current_coeff, filter_size_numOfBlk16, result3, result4);
      current_coeff += 16 * filter_size_numOfBlk16; // because of dual pixel processing

      __m256i sumQuad1234 = _mm256_hadd_epi32(_mm256_hadd_epi32(result1, result2), _mm256_hadd_epi32(result3, result4));
      // L1L1L1L1 L1L1L1L1 + L2L2L2L2 L2L2L2L2 = L1L1 L2L2 L1L1 L2L2
      // L3L3L3L3 L3L3L3L3 + L4L4L4L4 L4L4L4L4 = L3L3 L4L4 L3L3 L4L4
      // L1L1 L2L2 L1L1 L2L2 + L3L3 L4L4 L3L3 L4L4 = L1L2 L3L4 L1L2 L3L4

       // 4-7
      result1 = rounder256_1;
      result2 = result1;
      result3 = result1;
      result4 = result1;

      begin1 = program->pixel_offset[x + 4];
      begin2 = program->pixel_offset[x + 5];
      begin3 = program->pixel_offset[x + 6];
      begin4 = program->pixel_offset[x + 7];

      // begin1, begin2
      for (int i = 0; i < filter_size_numOfBlk16; i++)
        process_two_16pixels_h_uint8_t(src, begin1, begin2, i, current_coeff, filter_size_numOfBlk16, result1, result2);
      current_coeff += 16 * filter_size_numOfBlk16; // because of dual pixel processing

      // begin3, begin4
      for (int i = 0; i < filter_size_numOfBlk16; i++)
        process_two_16pixels_h_uint8_t(src, begin3, begin4, i, current_coeff, filter_size_numOfBlk16, result3, result4);
      current_coeff += 16 * filter_size_numOfBlk16; // because of dual pixel processing

      __m256i sumQuad5678 = _mm256_hadd_epi32(_mm256_hadd_epi32(result1, result2), _mm256_hadd_epi32(result3, result4));
      // L5L6 L7L8 L5L6 L7L8

      // Lo128bit  Hi128bit
      // L1L2 L3L4 L1L2 L3L4
      // L5L6 L7L8 L5L6 L7L8
      __m128i pix1234 = _mm_add_epi32(_mm256_extractf128_si256(sumQuad1234, 0), _mm256_extractf128_si256(sumQuad1234, 1));
      __m128i pix5678 = _mm_add_epi32(_mm256_extractf128_si256(sumQuad5678, 0), _mm256_extractf128_si256(sumQuad5678, 1));
      __m256i result_8x_uint32 = _mm256_set_m128i(pix5678, pix1234);

      // scale back, shuffle, store
      __m256i result = _mm256_srai_epi32(result_8x_uint32, FPScale8bits); // shift back integer arithmetic 14 bits precision for 8 bit data

      __m256i result_2x4x_uint16 = _mm256_packus_epi32(result, zero); // 8*32+zeros = lower 4*16 in both 128bit lanes
      __m128i result_2x4x_uint16_128 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(result_2x4x_uint16, (0 << 0) | (2 << 2) | (0 << 4) | (0 << 6))); // low64 of 2nd 128bit lane to hi64 of 1st 128bit lane
      __m128i result_2x4x_uint8 = _mm_packus_epi16(result_2x4x_uint16_128, _mm256_castsi256_si128(zero)); // L1 L2 L3 L4 | L5 L6 L7 L8
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + x), result_2x4x_uint8);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}


//-------- 256 bit float Horizontals

AVS_FORCEINLINE static void process_one_pixel_h_float(const float *src, int begin, int i, float *&current_coeff, __m256 &result) {
  __m256 data_single = _mm256_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8)); // float 8*32=256 8 pixels at a time
  __m256 coeff = _mm256_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
  result = _mm256_fmadd_ps(data_single, coeff, result); // a*b+c
  current_coeff += 8;
}

template<int filtersizemod8>
AVS_FORCEINLINE static void process_one_pixel_h_float_mask(const float *src, int begin, int i, float *&current_coeff, __m256 &result, __m256 &zero) {
  __m256 data_single = _mm256_loadu_ps(reinterpret_cast<const float*>(src + begin + i * 8)); // float 8*32=256 8 pixels at a time
  data_single = _mm256_blend_ps(zero, data_single, (1 << filtersizemod8) - 1);
  __m256 coeff = _mm256_load_ps(reinterpret_cast<const float*>(current_coeff)); // always aligned
  result = _mm256_fmadd_ps(data_single, coeff, result); // a*b+c
  current_coeff += 8;
}

// filtersizealigned8: special: 1, 2. Generic: -1
template<int filtersizealigned8, int filtersizemod8>
void resizer_h_avx2_generic_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);

  const float *src = reinterpret_cast<const float *>(src8);
  float *dst = reinterpret_cast<float *>(dst8);
  dst_pitch /= sizeof(float);
  src_pitch /= sizeof(float);

  // OMG! 18.01.19
  // Protection against NaN
  // When reading the last 8 consecutive pixels from right side offsets, it would access beyond-last-pixel area.
  // One SIMD cycle reads 8 bytes from (src + begin + i * 8)
  // When program->filter_size mod 8 is 1..7 then some of the last pixels should be masked because there can be NaN garbage.
  // So it's not enough to mask the coefficients by zero. Theory: let's multiply offscreen elements by 0 which works for integer samples.
  // But we are using float, so since NaN * Zero is NaN which propagates further to NaN when hadd is summing up the pixel*coeff series

  const int pixels_per_cycle = 8; // doing 8 is faster than 4
  const int unsafe_limit = (program->overread_possible && filtersizemod8 != 0) ? (program->source_overread_beyond_targetx / pixels_per_cycle) * pixels_per_cycle : width;

  for (int y = 0; y < height; y++) {
    float* current_coeff = program->pixel_coefficient_float;

    // loop for clean, non-offscreen data
    for (int x = 0; x < unsafe_limit; x += pixels_per_cycle) {
      __m256 result1 = _mm256_set1_ps(0.0f);
      __m256 result2 = result1;
      __m256 result3 = result1;
      __m256 result4 = result1;

      // 1-4
      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

      __m256 sumQuad12 = _mm256_hadd_ps(result1, result2); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2
      __m256 sumQuad34 = _mm256_hadd_ps(result3, result4); // L3L3L3L3L3L3L3L3 + L4L4L4L4L4L4L4L4L4 = L3L3 L4L4 L3L3 L4L4
      __m256 sumQuad1234 = _mm256_hadd_ps(sumQuad12, sumQuad34); // L1L1 L2L2 L1L1 L2L2 + L3L3 L4L4 L3L3 L4L4 = L1 L2 L3 L4 L1 L2 L3 L4
      __m128 result_lo = _mm_add_ps(_mm256_castps256_ps128(sumQuad1234), _mm256_extractf128_ps(sumQuad1234, 1)); // L1 L2 L3 L4

      // 5-8
      result1 = _mm256_set1_ps(0.0f);
      result2 = result1;
      result3 = result1;
      result4 = result1;

      begin1 = program->pixel_offset[x + 4];
      begin2 = program->pixel_offset[x + 5];
      begin3 = program->pixel_offset[x + 6];
      begin4 = program->pixel_offset[x + 7];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);

      sumQuad12 = _mm256_hadd_ps(result1, result2); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2
      sumQuad34 = _mm256_hadd_ps(result3, result4); // L3L3L3L3L3L3L3L3 + L4L4L4L4L4L4L4L4L4 = L3L3 L4L4 L3L3 L4L4
      sumQuad1234 = _mm256_hadd_ps(sumQuad12, sumQuad34); // L1L1 L2L2 L1L1 L2L2 + L3L3 L4L4 L3L3 L4L4 = L1 L2 L3 L4 L1 L2 L3 L4
      __m128 result_hi = _mm_add_ps(_mm256_castps256_ps128(sumQuad1234), _mm256_extractf128_ps(sumQuad1234, 1)); // L1 L2 L3 L4

      __m256 result256 = _mm256_insertf128_ps(_mm256_castps128_ps256(result_lo), result_hi, 1); // merge result, result_hi

      _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result256); // 8 results at a time
    } // for x

      // possibly right-side offscreen
      // and the same for the rest with masking the last filtersize/8 chunk
    __m256 zero = _mm256_setzero_ps();
    for (int x = unsafe_limit; x < width; x += 4) {
      __m256 result1 = _mm256_set1_ps(0.0f);
      __m256 result2 = result1;
      __m256 result3 = result1;
      __m256 result4 = result1;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, result1
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin1, i, current_coeff, result1);
      if (begin1 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin1, filter_size_numOfBlk8 - 1, current_coeff, result1, zero);

      // begin2, result2
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin2, i, current_coeff, result2);
      if (begin2 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin2, filter_size_numOfBlk8 - 1, current_coeff, result2, zero);

      // begin3, result3
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin3, i, current_coeff, result3);
      if (begin3 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin3, filter_size_numOfBlk8 - 1, current_coeff, result3, zero);

      // begin4, result4
      for (int i = 0; i < filter_size_numOfBlk8 - 1; i++)
        process_one_pixel_h_float(src, begin4, i, current_coeff, result4);
      if (begin4 < program->source_overread_offset)
        process_one_pixel_h_float(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4);
      else
        process_one_pixel_h_float_mask<filtersizemod8>(src, begin4, filter_size_numOfBlk8 - 1, current_coeff, result4, zero);

      const __m256 sumQuad12 = _mm256_hadd_ps(result1, result2); // L1L1L1L1L1L1L1L1 + L2L2L2L2L2L2L2L2L2 = L1L1 L2L2 L1L1 L2L2
      const __m256 sumQuad34 = _mm256_hadd_ps(result3, result4); // L3L3L3L3L3L3L3L3 + L4L4L4L4L4L4L4L4L4 = L3L3 L4L4 L3L3 L4L4
      const __m256 sumQuad1234 = _mm256_hadd_ps(sumQuad12, sumQuad34); // L1L1 L2L2 L1L1 L2L2 + L3L3 L4L4 L3L3 L4L4 = L1 L2 L3 L4 L1 L2 L3 L4
      __m128 result = _mm_add_ps(_mm256_castps256_ps128(sumQuad1234), _mm256_extractf128_ps(sumQuad1234, 1)); // L1 L2 L3 L4

      _mm_stream_ps(reinterpret_cast<float*>(dst + x), result); // 4 results at a time

    } // for x

    dst += dst_pitch;
    src += src_pitch;
  }

  /*
  // check Nans - debug
  dst -= dst_pitch * height;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (std::isnan(dst[x]))
      {
        x = x;
      }
    }
    dst += dst_pitch;
  }
  */
}

//-------- 256 bit uint16_t Horizontals

template<bool lessthan16bit>
AVS_FORCEINLINE static void process_two_pixels_h_uint16_t(const uint16_t *src, int begin1, int begin2, int i, short *&current_coeff, int filter_size_numOfBlk8, __m256i &result, const __m256i &shifttosigned) {
  __m128i data_single_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin1 + i * 8));
  __m128i data_single_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + begin2 + i * 8));
  __m256i data_single = _mm256_set_m128i(data_single_hi, data_single_lo);
  if (!lessthan16bit)
    data_single = _mm256_add_epi16(data_single, shifttosigned); // unsigned -> signed
  __m128i coeff_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff)); // 8 coeffs
  __m128i coeff_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(current_coeff + 1 * filter_size_numOfBlk8 * 8)); // 8 coeffs
  __m256i coeff = _mm256_set_m128i(coeff_hi, coeff_lo);
  result = _mm256_add_epi32(result, _mm256_madd_epi16(data_single, coeff));
  current_coeff += 8;
}

// filter_size <= 8 -> filter_size_align8 == 1 -> no loop, hope it'll be optimized
// filter_size <= 16 -> filter_size_align8 == 2 -> loop 0..1 hope it'll be optimized
// filter_size > 16 -> use parameter AlignNumber(program->filter_size_numOfFullBlk8, 8) / 8;
template<bool lessthan16bit, int filtersizealigned8>
void internal_resizer_h_avx2_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  // 1 and 2: special case for compiler optimization
  const int filter_size_numOfBlk8 = (filtersizealigned8 >= 1) ? filtersizealigned8 : (AlignNumber(program->filter_size, 8) / 8);
  const int incFilterSize = 8 * filter_size_numOfBlk8; // skip every 2nd coeffs (two pixels)

  const __m256i zero = _mm256_setzero_si256();
  const __m256i shifttosigned = _mm256_set1_epi16(-32768); // for 16 bits only
  const __m256i shiftfromsigned = _mm256_set1_epi32(+32768 << FPScale16bits); // for 16 bits only
  const __m256i rounder256 = _mm256_set_epi32(0, 0, 0, 1 << (FPScale16bits - 1), 0, 0, 0, 1 << (FPScale16bits - 1)); // only once

  const uint16_t *src = reinterpret_cast<const uint16_t *>(src8);
  uint16_t *dst = reinterpret_cast<uint16_t *>(dst8);
  dst_pitch /= sizeof(uint16_t);
  src_pitch /= sizeof(uint16_t);

  __m128i clamp_limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp limit for <16 bits

  for (int y = 0; y < height; y++) {
    short* current_coeff = program->pixel_coefficient;

    // 8 pixels!
    for (int x = 0; x < width; x += 8) {
      __m256i result12 = rounder256;
      __m256i result34 = result12;

      int begin1 = program->pixel_offset[x + 0];
      int begin2 = program->pixel_offset[x + 1];
      int begin3 = program->pixel_offset[x + 2];
      int begin4 = program->pixel_offset[x + 3];

      // begin1, begin2, result12
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_two_pixels_h_uint16_t<lessthan16bit>(src, begin1, begin2, i, current_coeff, filter_size_numOfBlk8, result12, shifttosigned);
      current_coeff += incFilterSize; // skip begin2

                                      // begin3, begin4, result34
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_two_pixels_h_uint16_t<lessthan16bit>(src, begin3, begin4, i, current_coeff, filter_size_numOfBlk8, result34, shifttosigned);
      current_coeff += incFilterSize; // skip begin4

      __m256i sumQuad1234 = _mm256_hadd_epi32(result12, result34); // L1L1L1L1L2L2L2L2 + L3L3L3L3L4L4L4L4 = L1L1 L3L3 L2L2 L4L4

                                                                   // 4-7
      result12 = rounder256;
      result34 = result12;

      begin1 = program->pixel_offset[x + 4];
      begin2 = program->pixel_offset[x + 5];
      begin3 = program->pixel_offset[x + 6];
      begin4 = program->pixel_offset[x + 7];

      // begin1, begin2, result12
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_two_pixels_h_uint16_t<lessthan16bit>(src, begin1, begin2, i, current_coeff, filter_size_numOfBlk8, result12, shifttosigned);
      current_coeff += incFilterSize; // skip begin2

                                      // begin3, begin4, result34
      for (int i = 0; i < filter_size_numOfBlk8; i++)
        process_two_pixels_h_uint16_t<lessthan16bit>(src, begin3, begin4, i, current_coeff, filter_size_numOfBlk8, result34, shifttosigned);
      current_coeff += incFilterSize; // skip begin4

      __m256i sumQuad5678 = _mm256_hadd_epi32(result12, result34); // L5L5L5L5L6L6L6L6 + L7L7L7L7L8L8L8L8 = L5L5 L7L7 L6L6 L8L8

      __m256i result = _mm256_hadd_epi32(sumQuad1234, sumQuad5678); // L1L1 L3L3 L2L2 L4L4 + L5L5 L7L7 L6L6 L8L8 = L1 L3 L5 L7 | L2 L4 L6 L8
      // correct if signed, scale back, store
      if (!lessthan16bit)
        result = _mm256_add_epi32(result, shiftfromsigned);
      result = _mm256_srai_epi32(result, FPScale16bits); // shift back integer arithmetic 13 bits precision

      __m256i result_2x4x_uint16 = _mm256_packus_epi32(result, zero); // 8*32+zeros = lower 4*16 in both 128bit lanes
      __m128i result_2x4x_uint16_128 = _mm256_castsi256_si128(_mm256_permute4x64_epi64(result_2x4x_uint16, (0 << 0) | (2 << 2) | (0 << 4) | (0 << 6))); // low64 of 2nd 128bit lane to hi64 of 1st 128bit lane
      if (lessthan16bit)
        result_2x4x_uint16_128 = _mm_min_epu16(result_2x4x_uint16_128, clamp_limit); // extra clamp for 10-14 bit
      result_2x4x_uint16_128 = _mm_shuffle_epi8(result_2x4x_uint16_128, _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15)); // 0, 4, 1, 5, 2, 6, 3, 7)
      _mm_stream_si128(reinterpret_cast<__m128i *>(dst + x), result_2x4x_uint16_128);
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}

//-------- 256 bit uint8_t Horizontal Dispatcher

// coeffs are safely padded/aligned to 16
void resizer_h_avx2_generic_uint8_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);
  const int filter_size_numOfBlk16 = AlignNumber(program->filter_size, 16) / 16;

  if (filter_size_numOfBlk16 == 1)
    internal_resizer_h_avx2_generic_uint8_t<1>(dst8, src8, dst_pitch, src_pitch, program, width, height);
  else if (filter_size_numOfBlk16 == 2)
    internal_resizer_h_avx2_generic_uint8_t<2>(dst8, src8, dst_pitch, src_pitch, program, width, height);
  else if (filter_size_numOfBlk16 == 3)
    internal_resizer_h_avx2_generic_uint8_t<3>(dst8, src8, dst_pitch, src_pitch, program, width, height);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_avx2_generic_uint8_t<-1>(dst8, src8, dst_pitch, src_pitch, program, width, height);
}

//-------- 128(/256) bit uint16_t Horizontal Dispatcher

template<bool lessthan16bit>
void resizer_h_avx2_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  const int filter_size_numOfBlk8 = AlignNumber(program->filter_size, 8) / 8; // yes, 8 is used here

  if (filter_size_numOfBlk8 == 1)
    internal_resizer_h_avx2_generic_uint16_t<lessthan16bit, 1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size_numOfBlk8 == 2)
    internal_resizer_h_avx2_generic_uint16_t<lessthan16bit, 2>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else if (filter_size_numOfBlk8 == 3)
    internal_resizer_h_avx2_generic_uint16_t<lessthan16bit, 3>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
  else // -1: basic method, use program->filter_size
    internal_resizer_h_avx2_generic_uint16_t<lessthan16bit, -1>(dst8, src8, dst_pitch, src_pitch, program, width, height, bits_per_pixel);
}

//-------- 256 bit uint8_t Verticals

void resize_v_avx2_planar_uint8_t(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);
  int filter_size = program->filter_size;
  short* current_coeff = program->pixel_coefficient;

  __m256i zero = _mm256_setzero_si256();

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const BYTE* src_ptr = src + pitch_table[offset];

    // safe 32 byte alignment
    for (int x = 0; x < width; x += 32) {
      __m256i result_l = _mm256_set1_epi16(32); // Init. with rounder ((1 << 6)/2 = 32)
      __m256i result_h = result_l;

      const BYTE* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size; i++) {
        __m256i src_p = _mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr));

        __m256i src_l = _mm256_unpacklo_epi8(src_p, zero);
        __m256i src_h = _mm256_unpackhi_epi8(src_p, zero);

        src_l = _mm256_slli_epi16(src_l, 7);
        src_h = _mm256_slli_epi16(src_h, 7);

        __m256i coeff = _mm256_set1_epi16(*(current_coeff + i));

        __m256i dst_l = _mm256_mulhrs_epi16(src_l, coeff);
        __m256i dst_h = _mm256_mulhrs_epi16(src_h, coeff);

        result_l = _mm256_add_epi16(result_l, dst_l);
        result_h = _mm256_add_epi16(result_h, dst_h);

        src2_ptr += src_pitch;
      }

      // Divide by 64
      result_l = _mm256_srai_epi16(result_l, 6);
      result_h = _mm256_srai_epi16(result_h, 6);

      // Pack and store
      __m256i result = _mm256_packus_epi16(result_l, result_h);

      _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + x), result);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

//-------- 256 bit uint16_t Verticals

template<bool lessthan16bit, int index>
AVS_FORCEINLINE static void process_chunk_v_uint16_t_256(const uint16_t *src2_ptr, int src_pitch, __m256i &coeff01234567, __m256i &result_single_lo, __m256i &result_single_hi, const __m256i &shifttosigned) {
  // offset table generating is what preventing us from overaddressing
  __m256i src_even = _mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + index * src_pitch)); // 8x 16bit pixels
  __m256i src_odd = _mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + (index + 1) * src_pitch));  // 8x 16bit pixels
  __m256i src_lo = _mm256_unpacklo_epi16(src_even, src_odd);
  __m256i src_hi = _mm256_unpackhi_epi16(src_even, src_odd);
  if (!lessthan16bit) {
    src_lo = _mm256_add_epi16(src_lo, shifttosigned);
    src_hi = _mm256_add_epi16(src_hi, shifttosigned);
  }
  __m256i coeff = _mm256_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6)); // spread short pair (128bit lanes!)
  result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
  result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c
}

// program->filtersize: 1..16 special optimized, >8: normal
template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
void internal_resize_v_avx2_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(storage);
  const int filter_size_numOfFullBlk8 = (_filter_size_numOfFullBlk8 >= 0) ? _filter_size_numOfFullBlk8 : (program->filter_size / 8);
  short* current_coeff = program->pixel_coefficient;

// #define NON32_BYTES_ALIGNMENT
// in AVS+ 32 bytes alignment is guaranteed
#ifdef NON32_BYTES_ALIGNMENT
  int wMod16 = (width / 16) * 16; // uint16: 16 at a time (256bit)
#endif

  const __m256i zero = _mm256_setzero_si256();
  const __m256i shifttosigned = _mm256_set1_epi16(-32768); // for 16 bits only
  const __m256i shiftfromsigned = _mm256_set1_epi32(32768 << FPScale16bits); // for 16 bits only
  const __m256i rounder = _mm256_set1_epi32(1 << (FPScale16bits - 1));

  const uint16_t* src = (uint16_t *)src0;
  uint16_t* dst = (uint16_t *)dst0;
  dst_pitch = dst_pitch / sizeof(uint16_t);
  src_pitch = src_pitch / sizeof(uint16_t);

  const int limit = (1 << bits_per_pixel) - 1;
  __m256i clamp_limit = _mm256_set1_epi16((short)limit); // clamp limit for <16 bits

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const uint16_t* src_ptr = src + pitch_table[offset] / sizeof(uint16_t);
#ifdef NON32_BYTES_ALIGNMENT
    for (int x = 0; x < wMod16; x += 16) { // 32 byte alignment guaranteed in avs+ no need
#else
    for (int x = 0; x < width; x += 16) { // 2x16 words, safe to read/write anywhere
#endif
      __m256i result_single_lo = rounder;
      __m256i result_single_hi = rounder;

      const uint16_t* src2_ptr = src_ptr + x;

      for (int i = 0; i < filter_size_numOfFullBlk8; i++) {
        __m128i coeff01234567_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(current_coeff + i * 8));
        __m256i coeff01234567 = _mm256_broadcastsi128_si256(coeff01234567_128); // 2x 4x (2x16bit) shorts for even/odd

        // offset table generating is what preventing us from overaddressing
        // 0-1
        process_chunk_v_uint16_t_256<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 2-3
        process_chunk_v_uint16_t_256<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 4-5
        process_chunk_v_uint16_t_256<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        // 6-7
        process_chunk_v_uint16_t_256<lessthan16bit, 6>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
        src2_ptr += 8 * src_pitch;
      }

      // and the rest non-div8 chunk
      __m128i coeff01234567_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current_coeff + filter_size_numOfFullBlk8 * 8));
      __m256i coeff01234567 = _mm256_broadcastsi128_si256(coeff01234567_128); // 4x (2x16bit) shorts for even/odd
      if (filtersizemod8 >= 2)
        process_chunk_v_uint16_t_256<lessthan16bit, 0>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if (filtersizemod8 >= 4)
        process_chunk_v_uint16_t_256<lessthan16bit, 2>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if (filtersizemod8 >= 6)
        process_chunk_v_uint16_t_256<lessthan16bit, 4>(src2_ptr, src_pitch, coeff01234567, result_single_lo, result_single_hi, shifttosigned);
      if (filtersizemod8 % 2) { // remaining odd one
        const int index = filtersizemod8 - 1;
        __m256i src_even = _mm256_load_si256(reinterpret_cast<const __m256i*>(src2_ptr + index * src_pitch)); // 4x 16bit pixels
        if (!lessthan16bit)
          src_even = _mm256_add_epi16(src_even, shifttosigned);
        __m256i coeff = _mm256_shuffle_epi32(coeff01234567, ((index / 2) << 0) | ((index / 2) << 2) | ((index / 2) << 4) | ((index / 2) << 6));
        __m256i src_lo = _mm256_unpacklo_epi16(src_even, zero);
        __m256i src_hi = _mm256_unpackhi_epi16(src_even, zero); // insert zero after the unsigned->signed shift!
        result_single_lo = _mm256_add_epi32(result_single_lo, _mm256_madd_epi16(src_lo, coeff)); // a*b + c
        result_single_hi = _mm256_add_epi32(result_single_hi, _mm256_madd_epi16(src_hi, coeff)); // a*b + c
      }

      // correct if signed, scale back, store
      __m256i result_lo = result_single_lo;
      __m256i result_hi = result_single_hi;
      if (!lessthan16bit) {
        result_lo = _mm256_add_epi32(result_lo, shiftfromsigned);
        result_hi = _mm256_add_epi32(result_hi, shiftfromsigned);
      }
      result_lo = _mm256_srai_epi32(result_lo, FPScale16bits); // shift back integer arithmetic 13 bits precision
      result_hi = _mm256_srai_epi32(result_hi, FPScale16bits); // shift back integer arithmetic 13 bits precision

      __m256i result_2x8x_uint16 = _mm256_packus_epi32(result_lo, result_hi); // 8*32+zeros = lower 4*16 in both 128bit lanes
      if (lessthan16bit) {
        result_2x8x_uint16 = _mm256_min_epu16(result_2x8x_uint16, clamp_limit); // extra clamp for 10-14 bit
      }
      _mm256_stream_si256(reinterpret_cast<__m256i *>(dst + x), result_2x8x_uint16);
    }

#ifdef NON32_BYTES_ALIGNMENT
    // Leftover, slow C
    for (int x = wMod16; x < width; x++) {
      int64_t result64 = 1 << (FPScale16bits - 1); // rounder
      const uint16_t* src2_ptr = src_ptr + x;
      for (int i = 0; i < program->filter_size; i++) {
        //result64 += (src_ptr + pitch_table[i] / sizeof(uint16_t))[x] * (int64_t)current_coeff[i];
        result64 += (int)(*src2_ptr) * (int64_t)current_coeff[i];
        src2_ptr += src_pitch;
      }
      int result = (int)(result64 >> FPScale16bits); // scale back 13 bits
      result = result > limit ? limit : result < 0 ? 0 : result; // clamp 10..16 bits
      dst[x] = (uint16_t)result;
    }
#endif

    dst += dst_pitch;
    current_coeff += program->filter_size;
  }
}

//-------- 256 bit float Verticals

template<int _filtersize>
void internal_resize_v_avx2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);
  // 1..8: special case for compiler optimization
  const int filter_size = _filtersize >= 1 ? _filtersize : program->filter_size;
  float* current_coeff_float = program->pixel_coefficient_float;

  int wMod16 = (width / 16) * 16; // float: 16 at a time

  const float* src = (float *)src0;
  float* dst = (float *)dst0;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int fsmod4 = (filter_size / 4) * 4;
  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + pitch_table[offset] / sizeof(float);

    // 16 pixels/cycle (64 bytes)
    for (int x = 0; x < wMod16; x += 16) {
      __m256 result_single_lo = _mm256_set1_ps(0.0f);
      __m256 result_single_hi = _mm256_set1_ps(0.0f);

      const float* src2_ptr = src_ptr + x;

      for (int i = 0; i < fsmod4; i += 4) {
        __m256 src_single_lo;
        __m256 src_single_hi;
        __m256 coeff0123 = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(current_coeff_float + i)); // loads 4 floats into both hi and lo 128
        __m256 coeff;

        // unroll 4x
        // #1
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch)); // float  8*32=256 8 pixels at a time
        src_single_hi = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 8));
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (0 << 0) | (0 << 2) | (0 << 4) | (0 << 6))); // spread 0th
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c
        result_single_hi = _mm256_fmadd_ps(src_single_hi, coeff, result_single_hi); // a*b + c

        // #2
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch)); // float  8*32=256 8 pixels at a time
        src_single_hi = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch + 8));
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6))); // spread 1st
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c
        result_single_hi = _mm256_fmadd_ps(src_single_hi, coeff, result_single_hi); // a*b + c

        // #3
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch)); // float  8*32=256 8 pixels at a time
        src_single_hi = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch + 8));
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6))); // spread 2nd
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c
        result_single_hi = _mm256_fmadd_ps(src_single_hi, coeff, result_single_hi); // a*b + c

        // #4
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch)); // float  8*32=256 8 pixels at a time
        src_single_hi = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch + 8));
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (3 << 0) | (3 << 2) | (3 << 4) | (3 << 6))); // spread 3rd
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c
        result_single_hi = _mm256_fmadd_ps(src_single_hi, coeff, result_single_hi); // a*b + c

        src2_ptr += 4 * src_pitch;
      }

      // one-by-one
      for (int i = fsmod4; i < filter_size; i++) {
        __m256 src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch)); // float  8*32=256 8 pixels at a time
        __m256 src_single_hi = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch + 8));
        __m256 coeff = _mm256_broadcast_ss(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 1, fills all 8 floats
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c
        result_single_hi = _mm256_fmadd_ps(src_single_hi, coeff, result_single_hi); // a*b + c

        src2_ptr += src_pitch;
      }

      _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result_single_lo);
      _mm256_stream_ps(reinterpret_cast<float*>(dst + x + 8), result_single_hi);
    }

    // 8 pixels/cycle (32 bytes)
    for (int x = wMod16; x < width; x += 8) { // safe to width, avs+ has min. 32 byte alignment
      __m256 result_single_lo = _mm256_set1_ps(0.0f);

      const float* src2_ptr = src_ptr + x;

      for (int i = 0; i < fsmod4; i += 4) {
        __m256 src_single_lo;
        __m256 coeff0123 = _mm256_broadcast_ps(reinterpret_cast<const __m128*>(current_coeff_float + i)); // loads 4 floats into both hi and lo 128
        __m256 coeff;

        // unroll 4x
        // #1
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch)); // float  8*32=256 8 pixels at a time
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (0 << 0) | (0 << 2) | (0 << 4) | (0 << 6))); // spread 0th
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c

        // #2
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 1 * src_pitch)); // float  8*32=256 8 pixels at a time
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6))); // spread 1st
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c

        // #3
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 2 * src_pitch)); // float  8*32=256 8 pixels at a time
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (2 << 0) | (2 << 2) | (2 << 4) | (2 << 6))); // spread 2nd
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c

        // #4
        src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 3 * src_pitch)); // float  8*32=256 8 pixels at a time
        coeff = _mm256_castsi256_ps(_mm256_shuffle_epi32(_mm256_castps_si256(coeff0123), (3 << 0) | (3 << 2) | (3 << 4) | (3 << 6))); // spread 3rd
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c

        src2_ptr += 4 * src_pitch;
      }

      // one-by-one
      for (int i = fsmod4; i < filter_size; i++) {
        __m256 src_single_lo = _mm256_load_ps(reinterpret_cast<const float*>(src2_ptr + 0 * src_pitch)); // float  8*32=256 8 pixels at a time
        __m256 coeff = _mm256_broadcast_ss(reinterpret_cast<const float*>(current_coeff_float + i)); // loads 1, fills all 8 floats
        result_single_lo = _mm256_fmadd_ps(src_single_lo, coeff, result_single_lo); // a*b + c

        src2_ptr += src_pitch;
      }

      _mm256_stream_ps(reinterpret_cast<float*>(dst + x), result_single_lo);
    }

#if 0
    // no need
    // Leftover, Slow C
    for (int x = wMod16; x < width; x++) {
      float result = 0;
      const float* src2_ptr = src_ptr + x;
      for (int i = 0; i < filter_size; i++) {
        result += (*src2_ptr) * current_coeff_float[i];
        src2_ptr += src_pitch;
      }
      dst[x] = result;
    }
#endif
    dst += dst_pitch;
    current_coeff_float += filter_size;
  }
}


//-------- Float Vertical Dispatcher

void resize_v_avx2_planar_float(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  // 1..8: special case for compiler optimization
  switch (program->filter_size) {
  case 1:
    internal_resize_v_avx2_planar_float<1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 2:
    internal_resize_v_avx2_planar_float<2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 3:
    internal_resize_v_avx2_planar_float<3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 4:
    internal_resize_v_avx2_planar_float<4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 5:
    internal_resize_v_avx2_planar_float<5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 6:
    internal_resize_v_avx2_planar_float<6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 7:
    internal_resize_v_avx2_planar_float<7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 8:
    internal_resize_v_avx2_planar_float<8>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 9:
    internal_resize_v_avx2_planar_float<9>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 10:
    internal_resize_v_avx2_planar_float<10>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 11:
    internal_resize_v_avx2_planar_float<11>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 12:
    internal_resize_v_avx2_planar_float<12>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 13:
    internal_resize_v_avx2_planar_float<13>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 14:
    internal_resize_v_avx2_planar_float<14>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 15:
    internal_resize_v_avx2_planar_float<15>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 16:
    internal_resize_v_avx2_planar_float<16>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  default:
    internal_resize_v_avx2_planar_float<-1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  }
}

//-------- uint16_t Vertical Dispatcher


template<bool lessthan16bit>
void resize_v_avx2_planar_uint16_t(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  // template<bool lessthan16bit, int _filter_size_numOfFullBlk8, int filtersizemod8>
  // filtersize 1..16: to template for optimization
  switch (program->filter_size) {
  case 1:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 2:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 3:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 4:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 5:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 6:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 7:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 0, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 8:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 9:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 10:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 11:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 12:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 13:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 14:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  case 15:
    internal_resize_v_avx2_planar_uint16_t<lessthan16bit, 1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
    break;
  default:
    switch (program->filter_size & 7) {
    case 0:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 0>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 1:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 1>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 2:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 2>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 3:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 3>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 4:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 4>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 5:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 5>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 6:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 6>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    case 7:
      internal_resize_v_avx2_planar_uint16_t<lessthan16bit, -1, 7>(dst0, src0, dst_pitch, src_pitch, program, width, target_height, bits_per_pixel, pitch_table, storage);
      break;
    }
    break;
  }
}

// instantiate here
// avx2 32 bit
template void resizer_h_avx2_generic_float<1, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<1, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template void resizer_h_avx2_generic_float<2, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<2, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template void resizer_h_avx2_generic_float<-1, 0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 4>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 5>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 6>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resizer_h_avx2_generic_float<-1, 7>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// avx2 16bit
template void resizer_h_avx2_generic_uint16_t<false>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
// avx2 10-14bit
template void resizer_h_avx2_generic_uint16_t<true>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);



// avx2 16
template void resize_v_avx2_planar_uint16_t<false>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);
// avx2 10-14bit
template void resize_v_avx2_planar_uint16_t<true>(BYTE* dst0, const BYTE* src0, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage);
