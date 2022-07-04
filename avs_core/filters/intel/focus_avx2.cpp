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

#include "focus_sse.h"
#include <cmath>
#include <vector>
#include <avs/alignment.h>
#include <avs/minmax.h>
#include "../core/internal.h"
#include <stdint.h>

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


#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

#ifndef _mm256_cvtsi256_si32
// int _mm256_cvtsi256_si32 (__m256i a)
#define _mm256_cvtsi256_si32(a) (_mm_cvtsi128_si32(_mm256_castsi256_si128(a)))
#endif

static AVS_FORCEINLINE __m256i af_blend_avx2(__m256i &upper, __m256i &center, __m256i &lower, __m256i &center_weight, __m256i &outer_weight, __m256i &round_mask) {
  __m256i outer_tmp = _mm256_add_epi16(upper, lower);
  __m256i center_tmp = _mm256_mullo_epi16(center, center_weight);

  outer_tmp = _mm256_mullo_epi16(outer_tmp, outer_weight);

  __m256i result = _mm256_adds_epi16(center_tmp, outer_tmp);
  result = _mm256_adds_epi16(result, center_tmp);
  result = _mm256_adds_epi16(result, round_mask);
  return _mm256_srai_epi16(result, 7);
}

static AVS_FORCEINLINE __m256i af_blend_uint16_t_avx2(__m256i &upper, __m256i &center, __m256i &lower, __m256i &center_weight, __m256i &outer_weight, __m256i &round_mask) {
  __m256i outer_tmp = _mm256_add_epi32(upper, lower);
  __m256i center_tmp;
  center_tmp = _mm256_mullo_epi32(center, center_weight);
  outer_tmp = _mm256_mullo_epi32(outer_tmp, outer_weight);

  __m256i result = _mm256_add_epi32(center_tmp, outer_tmp);
  result = _mm256_add_epi32(result, center_tmp);
  result = _mm256_add_epi32(result, round_mask);
  return _mm256_srai_epi32(result, 7);
}

static AVS_FORCEINLINE __m256i af_unpack_blend_avx2(__m256i &left, __m256i &center, __m256i &right, __m256i &center_weight, __m256i &outer_weight, __m256i &round_mask, __m256i &zero) {
  __m256i left_lo = _mm256_unpacklo_epi8(left, zero);
  __m256i left_hi = _mm256_unpackhi_epi8(left, zero);
  __m256i center_lo = _mm256_unpacklo_epi8(center, zero);
  __m256i center_hi = _mm256_unpackhi_epi8(center, zero);
  __m256i right_lo = _mm256_unpacklo_epi8(right, zero);
  __m256i right_hi = _mm256_unpackhi_epi8(right, zero);

  __m256i result_lo = af_blend_avx2(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m256i result_hi = af_blend_avx2(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);

  return _mm256_packus_epi16(result_lo, result_hi);
}

static AVS_FORCEINLINE __m256i af_unpack_blend_uint16_t_avx2(__m256i &left, __m256i &center, __m256i &right, __m256i &center_weight, __m256i &outer_weight, __m256i &round_mask, __m256i &zero) {
  __m256i left_lo = _mm256_unpacklo_epi16(left, zero);
  __m256i left_hi = _mm256_unpackhi_epi16(left, zero);
  __m256i center_lo = _mm256_unpacklo_epi16(center, zero);
  __m256i center_hi = _mm256_unpackhi_epi16(center, zero);
  __m256i right_lo = _mm256_unpacklo_epi16(right, zero);
  __m256i right_hi = _mm256_unpackhi_epi16(right, zero);

  __m256i result_lo = af_blend_uint16_t_avx2(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m256i result_hi = af_blend_uint16_t_avx2(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);
  return _mm256_packus_epi32(result_lo, result_hi);
}

void af_vertical_uint16_t_avx2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount) {
  // amount was: half_amount (32768). Full: 65536 (2**16)
  // now it becomes 2**(16-9)=2**7 scale
  int t = (amount + 256) >> 9; // 16-9 = 7 -> shift in
  __m256i center_weight = _mm256_set1_epi32(t);
  __m256i outer_weight = _mm256_set1_epi32(64 - t);
  __m256i round_mask = _mm256_set1_epi32(0x40);
  __m256i zero = _mm256_setzero_si256();

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 32) {
      __m256i upper = _mm256_load_si256(reinterpret_cast<const __m256i*>(line_buf + x));
      __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));
      __m256i lower = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + pitch + x));
      _mm256_store_si256(reinterpret_cast<__m256i*>(line_buf + x), center);

      __m256i upper_lo = _mm256_unpacklo_epi16(upper, zero);
      __m256i upper_hi = _mm256_unpackhi_epi16(upper, zero);
      __m256i center_lo = _mm256_unpacklo_epi16(center, zero);
      __m256i center_hi = _mm256_unpackhi_epi16(center, zero);
      __m256i lower_lo = _mm256_unpacklo_epi16(lower, zero);
      __m256i lower_hi = _mm256_unpackhi_epi16(lower, zero);

      __m256i result_lo = af_blend_uint16_t_avx2(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m256i result_hi = af_blend_uint16_t_avx2(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m256i result = _mm256_packus_epi32(result_lo, result_hi);

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 32) {
    __m256i upper = _mm256_load_si256(reinterpret_cast<const __m256i*>(line_buf + x));
    __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));

    __m256i upper_lo = _mm256_unpacklo_epi16(upper, zero);
    __m256i upper_hi = _mm256_unpackhi_epi16(upper, zero);
    __m256i center_lo = _mm256_unpacklo_epi16(center, zero);
    __m256i center_hi = _mm256_unpackhi_epi16(center, zero);

    __m256i result_lo = af_blend_uint16_t_avx2(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m256i result_hi = af_blend_uint16_t_avx2(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m256i result;
    result = _mm256_packus_epi32(result_lo, result_hi);

    _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), result);
  }
}

void af_vertical_avx2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount) {
  short t = (amount + 256) >> 9;
  __m256i center_weight = _mm256_set1_epi16(t);
  __m256i outer_weight = _mm256_set1_epi16(64 - t);
  __m256i round_mask = _mm256_set1_epi16(0x40);
  __m256i zero = _mm256_setzero_si256();

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < width; x += 32) {
      __m256i upper = _mm256_load_si256(reinterpret_cast<const __m256i*>(line_buf + x));
      __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));
      __m256i lower = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + pitch + x));
      _mm256_store_si256(reinterpret_cast<__m256i*>(line_buf + x), center);

      __m256i upper_lo = _mm256_unpacklo_epi8(upper, zero);
      __m256i upper_hi = _mm256_unpackhi_epi8(upper, zero);
      __m256i center_lo = _mm256_unpacklo_epi8(center, zero);
      __m256i center_hi = _mm256_unpackhi_epi8(center, zero);
      __m256i lower_lo = _mm256_unpacklo_epi8(lower, zero);
      __m256i lower_hi = _mm256_unpackhi_epi8(lower, zero);

      __m256i result_lo = af_blend_avx2(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m256i result_hi = af_blend_avx2(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m256i result = _mm256_packus_epi16(result_lo, result_hi);

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < width; x += 32) {
    __m256i upper = _mm256_load_si256(reinterpret_cast<const __m256i*>(line_buf + x));
    __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));

    __m256i upper_lo = _mm256_unpacklo_epi8(upper, zero);
    __m256i upper_hi = _mm256_unpackhi_epi8(upper, zero);
    __m256i center_lo = _mm256_unpacklo_epi8(center, zero);
    __m256i center_hi = _mm256_unpackhi_epi8(center, zero);

    __m256i result_lo = af_blend_avx2(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m256i result_hi = af_blend_avx2(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m256i result = _mm256_packus_epi16(result_lo, result_hi);

    _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), result);
  }
}

// -------------------------------------
// Blur/Sharpen Horizontal YV12 C++ Code
// -------------------------------------

template<typename pixel_t>
static AVS_FORCEINLINE void af_horizontal_planar_process_line_c(pixel_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight) {
  size_t x;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight));
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight));
}

static AVS_FORCEINLINE void af_horizontal_planar_process_line_uint16_c(uint16_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight, int bits_per_pixel) {
  size_t x;
  typedef uint16_t pixel_t;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  const int max_pixel_value = (1 << bits_per_pixel) - 1; // clamping on 10-12-14-16 bitdepth
  typedef std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight), max_pixel_value);
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight), max_pixel_value);
}

#if 0
static AVS_FORCEINLINE void af_horizontal_planar_process_line_float_c(float left, float *dstp, size_t row_size, float center_weight, float outer_weight) {
    size_t x;
    size_t width = row_size / sizeof(float);
    for (x = 0; x < width-1; ++x) {
        float temp = dstp[x] * center_weight + (left + dstp[x+1]) * outer_weight;
        left = dstp[x];
        dstp[x] = temp;
    }
    dstp[x] = dstp[x] * center_weight + (left + dstp[x]) * outer_weight;
}

static void af_horizontal_planar_float_c(BYTE* dstp8, size_t height, size_t pitch8, size_t row_size, float amount)
{
    float* dstp = reinterpret_cast<float *>(dstp8);
    size_t pitch = pitch8 / sizeof(float);
    float center_weight = amount;
    float outer_weight = (1.0f - amount) / 2.0f;
    float left;
    for (size_t y = height; y>0; --y) {
        left = dstp[0];
        af_horizontal_planar_process_line_float_c(left, dstp, row_size, center_weight, outer_weight);
        dstp += pitch;
    }
}
#endif

void af_horizontal_planar_avx2(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount) {
  size_t mod32_width = (width / 32) * 32;
  size_t sse_loop_limit = width == mod32_width ? mod32_width - 32 : mod32_width;
  int center_weight_c = int(amount*2);
  int outer_weight_c = int(32768-amount);

  short t = short((amount + 256) >> 9);
  __m256i center_weight = _mm256_set1_epi16(t);
  __m256i outer_weight = _mm256_set1_epi16(64 - t);
  __m256i round_mask = _mm256_set1_epi16(0x40);
  __m256i zero = _mm256_setzero_si256();

  __m128i left_mask_128 = _mm_set_epi32(0, 0, 0, 0xFF);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i right_mask_128 = _mm_set_epi8(0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m256i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp));
    __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp+1));
    __m128i center_lo128 = _mm256_extractf128_si256(center, 0);
    __m128i left_lo128 = _mm_or_si128(_mm_and_si128(center_lo128, left_mask_128), _mm_slli_si128(center_lo128, 1));
    __m128i left_hi128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + 16 - 1));
    left = _mm256_set_m128i(left_hi128, left_lo128);

    __m256i result = af_unpack_blend_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp+32-1));
    _mm256_store_si256(reinterpret_cast<__m256i*>(dstp), result);

    //main processing loop
    for (size_t x = 32; x < sse_loop_limit; x+= 32) {
      center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp+x));
      right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp+x+1));

      result = af_unpack_blend_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp+x+32-1)); // read ahead to prevent overwrite

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp+x), result);
    }

    //right border
    if(mod32_width == width) { //width is mod32, process with simd
      center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + mod32_width - 32));

      __m128i right_lo128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + mod32_width - 32 + 1));
      __m128i center_hi128 = _mm256_extractf128_si256(center, 1); // get high 128bit, really right! ptr+16
      __m128i right_hi128 = _mm_or_si128(_mm_and_si128(center_hi128, right_mask_128), _mm_srli_si128(center_hi128, 1));
      right = _mm256_set_m128i(right_hi128, right_lo128);

      result = af_unpack_blend_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp+mod32_width-32), result);
    } else { //some stuff left
      BYTE l = _mm256_cvtsi256_si32(left) & 0xFF;
      af_horizontal_planar_process_line_c<uint8_t>(l, dstp+mod32_width, width-mod32_width, center_weight_c, outer_weight_c);

    }

    dstp += pitch;
  }
}

void af_horizontal_planar_uint16_t_avx2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel) {
  size_t mod32_width = (row_size / 32) * 32;
  size_t sse_loop_limit = row_size == mod32_width ? mod32_width - 32 : mod32_width;
  int center_weight_c = int(amount * 2);
  int outer_weight_c = int(32768 - amount);

  int t = int((amount + 256) >> 9);
  __m256i center_weight = _mm256_set1_epi32(t);
  __m256i outer_weight = _mm256_set1_epi32(64 - t);
  __m256i round_mask = _mm256_set1_epi32(0x40);
  __m256i zero = _mm256_setzero_si256();

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i left_mask_128 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0xFFFF); // 0, 0, 0, 0, 0, 0, 0, FFFF
  __m128i right_mask_128 = _mm_set_epi16(0xFFFF, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m256i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m256i center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp));
    __m256i right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + sizeof(uint16_t)));
    __m128i center_lo128 = _mm256_extractf128_si256(center, 0);
    __m128i left_lo128 = _mm_or_si128(_mm_and_si128(center_lo128, left_mask_128), _mm_slli_si128(center_lo128, sizeof(uint16_t)));
    __m128i left_hi128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + 16 - sizeof(uint16_t)));
    left = _mm256_set_m128i(left_hi128, left_lo128);

    __m256i result = af_unpack_blend_uint16_t_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + (32 - sizeof(uint16_t))));
    _mm256_store_si256(reinterpret_cast<__m256i*>(dstp), result);

    //main processing loop
    for (size_t x = 32; x < sse_loop_limit; x += 32) {
      center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));
      right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + x + sizeof(uint16_t)));

      result = af_unpack_blend_uint16_t_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + x + (32 - sizeof(uint16_t)))); // read ahead to prevent overwrite

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), result);
    }

    //right border
    if (mod32_width == row_size) { //width is mod32, process with simd
      center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + mod32_width - 32));
      __m128i right_lo128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + mod32_width - 32 + sizeof(uint16_t)));
      __m128i center_hi128 = _mm256_extractf128_si256(center, 1); // get high 128bit, really right! ptr+16
      __m128i right_hi128 = _mm_or_si128(_mm_and_si128(center_hi128, right_mask_128), _mm_srli_si128(center_hi128, sizeof(uint16_t)));
      right = _mm256_set_m128i(right_hi128, right_lo128);

      result = af_unpack_blend_uint16_t_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + mod32_width - 32), result);
    }
    else { //some stuff left
      uint16_t l = _mm256_cvtsi256_si32(left) & 0xFFFF;
      af_horizontal_planar_process_line_uint16_c(l, dstp + mod32_width, row_size - mod32_width, center_weight_c, outer_weight_c, bits_per_pixel);
    }

    dstp += pitch;
  }
}
