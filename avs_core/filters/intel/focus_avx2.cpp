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
#include <avs/types.h>
#include <cstdint>
#include <type_traits>
#include "../core/internal.h"

// Intrinsics base header + really required extension headers
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else 
#include <x86intrin.h> // GCC/MinGW/Clang/LLVM
#endif
#include <immintrin.h>

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

void af_vertical_uint16_t_avx2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount, int bits_per_pixel) {
  // amount was: half_amount (32768). Full: 65536 (2**16)
  // now it becomes 2**(16-9)=2**7 scale
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp for 10-14 bits
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
      result = _mm256_min_epu16(result, limit);

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
    result = _mm256_min_epu16(result, limit);

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
  __m128i right_mask_128 = _mm_set_epi8((char)0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

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
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp for 10-14 bits
  __m256i outer_weight = _mm256_set1_epi32(64 - t);
  __m256i round_mask = _mm256_set1_epi32(0x40);
  __m256i zero = _mm256_setzero_si256();

  __m128i left_mask_128 = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, (short)0xFFFF); // 0, 0, 0, 0, 0, 0, 0, FFFF
  __m128i right_mask_128 = _mm_set_epi16((short)0xFFFF, 0, 0, 0, 0, 0, 0, 0);

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
    result = _mm256_min_epu16(result, limit);
    left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + (32 - sizeof(uint16_t))));
    _mm256_store_si256(reinterpret_cast<__m256i*>(dstp), result);

    //main processing loop
    for (size_t x = 32; x < sse_loop_limit; x += 32) {
      center = _mm256_load_si256(reinterpret_cast<const __m256i*>(dstp + x));
      right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dstp + x + sizeof(uint16_t)));

      result = af_unpack_blend_uint16_t_avx2(left, center, right, center_weight, outer_weight, round_mask, zero);
      result = _mm256_min_epu16(result, limit);

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
      result = _mm256_min_epu16(result, limit);

      _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + mod32_width - 32), result);
    }
    else { //some stuff left
      uint16_t l = _mm256_cvtsi256_si32(left) & 0xFFFF;
      af_horizontal_planar_process_line_uint16_c(l, dstp + mod32_width, row_size - mod32_width, center_weight_c, outer_weight_c, bits_per_pixel);
    }

    dstp += pitch;
  }
}


// TemporalSoften

static AVS_FORCEINLINE __m256i _mm256_cmple_epu8(__m256i x, __m256i y)
{
  // Returns 0xFF where x <= y:
  return _mm256_cmpeq_epi8(_mm256_min_epu8(x, y), x);
}

static AVS_FORCEINLINE __m256i _mm256_cmple_epu16(__m256i x, __m256i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm256_cmpeq_epi16(_mm256_min_epu16(x, y), x);
}

// 8 bit, based on accumulate_line_ssse3
// maxThreshold (255): simple accumulate for average (speed)
// threshold: 2 bytes in a word
template<bool maxThreshold>
void accumulate_line_avx2(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int div)
{
  const __m256i div_vector = _mm256_set1_epi16((short)div); // div = 32768/(planes+1)
  const __m256i thresh = _mm256_set1_epi16((short)threshold);
  const __m256i zero = _mm256_setzero_si256();

  for (size_t x = 0; x < rowsize_mod32; x += 32) {
    const __m256i current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_plane + x));
    __m256i low = _mm256_unpacklo_epi8(current, zero);
    __m256i high = _mm256_unpackhi_epi8(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(planeP[plane] + x));

      __m256i add_low, add_high;
      if constexpr (maxThreshold) {
        add_low = _mm256_unpacklo_epi8(p, zero);
        add_high = _mm256_unpackhi_epi8(p, zero);
      }
      else {
        const __m256i abs_cp = _mm256_or_si256(_mm256_subs_epu8(p, current), _mm256_subs_epu8(current, p));
        const __m256i leq_thresh = _mm256_cmple_epu8(abs_cp, thresh);
        const __m256i blended = _mm256_blendv_epi8(current, p, leq_thresh); // abs(p-c) <= thresh ? p : c
        add_low = _mm256_unpacklo_epi8(blended, zero);
        add_high = _mm256_unpackhi_epi8(blended, zero);
      }

      low = _mm256_adds_epu16(low, add_low);
      high = _mm256_adds_epu16(high, add_high);
    }

    // _mm256_mulhrs_epi16: INT16(((a * b) + 0x4000) >> 15)
    low = _mm256_mulhrs_epi16(low, div_vector);
    high = _mm256_mulhrs_epi16(high, div_vector);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_plane + x), _mm256_packus_epi16(low, high));
  }
}

template void accumulate_line_avx2<false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int div);
template void accumulate_line_avx2<true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int div);

// 10-16 bit, based on accumulate_line_16_sse41
// threshold: orig threshold scaled by (bits_per_pixel-8)
template<bool maxThreshold, bool lessThan16bit>
void accumulate_line_16_avx2(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int bits_per_pixel)
{
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // clamp for 10-14 bits
  const __m256 div_vector = _mm256_set1_ps(1.0f / (planes + 1));
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256i thresh = _mm256_set1_epi16((short)threshold);
  const __m256i zero = _mm256_setzero_si256();

  for (size_t x = 0; x < rowsize_mod32; x += 32) {
    const __m256i current = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_plane + x));
    __m256i low = _mm256_unpacklo_epi16(current, zero);
    __m256i high = _mm256_unpackhi_epi16(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(planeP[plane] + x));

      __m256i add_low, add_high;
      if constexpr (maxThreshold) {
        add_low = _mm256_unpacklo_epi16(p, zero);
        add_high = _mm256_unpackhi_epi16(p, zero);
      }
      else {
        const __m256i abs_cp = _mm256_or_si256(_mm256_subs_epu16(p, current), _mm256_subs_epu16(current, p));
        const __m256i leq_thresh = _mm256_cmple_epu16(abs_cp, thresh);
        const __m256i blended = _mm256_blendv_epi8(current, p, leq_thresh); // abs(p-c) <= thresh ? p : c
        add_low = _mm256_unpacklo_epi16(blended, zero);
        add_high = _mm256_unpackhi_epi16(blended, zero);
      }
      low = _mm256_add_epi32(low, add_low);
      high = _mm256_add_epi32(high, add_high);
    }

    low = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_cvtepi32_ps(low), div_vector, half));
    high = _mm256_cvttps_epi32(_mm256_fmadd_ps(_mm256_cvtepi32_ps(high), div_vector, half));
    __m256i acc = _mm256_packus_epi32(low, high);
    if constexpr (lessThan16bit)
      acc = _mm256_min_epu16(acc, limit);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(c_plane + x), acc);
  }
}

template void accumulate_line_16_avx2<false, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int bits_per_pixel);
template void accumulate_line_16_avx2<false, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int bits_per_pixel);
template void accumulate_line_16_avx2<true, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int bits_per_pixel);
template void accumulate_line_16_avx2<true, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize_mod32, int threshold, int bits_per_pixel);
