// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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

#include "../convert_bits.h"
#include "convert_bits_sse.h"
#include "../convert_helper.h"

#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/config.h>
#include <tuple>
#include <map>
#include <algorithm>

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <emmintrin.h>
#include <smmintrin.h> // SSE4.1

// float to 8 bit, float to 10/12/14/16 bit

// sse4.1
template<typename pixel_t, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_32_to_uintN_sse41(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const float* srcp0 = reinterpret_cast<const float*>(srcp);
  pixel_t* dstp0 = reinterpret_cast<pixel_t*>(dstp);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  const int src_width = src_rowsize / sizeof(float);

  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto dst_pixel_min = 0;
  const auto dst_pixel_max = (1 << target_bitdepth) - 1;

  auto src_offset_ps = _mm_set1_ps(d.src_offset);
  auto factor_ps = _mm_set1_ps(d.mul_factor);
  auto dst_offset_plus_round_ps = _mm_set1_ps(dst_offset_plus_round);
  auto dst_pixel_min_ps = _mm_set1_ps((float)dst_pixel_min);
  auto dst_pixel_max_ps = _mm_set1_ps((float)dst_pixel_max);

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x += 8) // 8 pixels at a time
    {
      __m128i result;
      __m128i result_0, result_1;
      __m128 src_0 = _mm_load_ps(reinterpret_cast<const float*>(srcp0 + x));
      __m128 src_1 = _mm_load_ps(reinterpret_cast<const float*>(srcp0 + x + 4));
      if constexpr (!chroma && !fulls) {
        // when offset is different from 0
        src_0 = _mm_sub_ps(src_0, src_offset_ps);
        src_1 = _mm_sub_ps(src_1, src_offset_ps);
      }
      src_0 = _mm_add_ps(_mm_mul_ps(src_0, factor_ps), dst_offset_plus_round_ps);
      src_1 = _mm_add_ps(_mm_mul_ps(src_1, factor_ps), dst_offset_plus_round_ps);

      src_0 = _mm_max_ps(_mm_min_ps(src_0, dst_pixel_max_ps), dst_pixel_min_ps);
      src_1 = _mm_max_ps(_mm_min_ps(src_1, dst_pixel_max_ps), dst_pixel_min_ps);
      result_0 = _mm_cvttps_epi32(src_0); // truncate
      result_1 = _mm_cvttps_epi32(src_1);
      if constexpr (sizeof(pixel_t) == 2) {
        result = _mm_packus_epi32(result_0, result_1); // sse41
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), result);
      }
      else {
        result = _mm_packs_epi32(result_0, result_1);
        result = _mm_packus_epi16(result, result); // lo 8 byte
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp0 + x), result);
      }

      // c: 
      //const int pixel = (int)((srcp0[x] - src_offset) * mul_factor + dst_offset_plus_round);
      //dstp0[x] = pixel_t(clamp(pixel, dst_pixel_min, dst_pixel_max));
    }
    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

// instantiate them
//template<typename pixel_t, bool chroma, bool fulls, bool fulld>
#define DEF_convert_32_to_uintN_functions(uint_X_t) \
template void convert_32_to_uintN_sse41<uint_X_t, false, true, true>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, true, true, true>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, false, true, false>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, true, true, false>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, false, false, true>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, true, false, true>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, false, false, false>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_sse41<uint_X_t, true, false, false>(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

DEF_convert_32_to_uintN_functions(uint8_t)
DEF_convert_32_to_uintN_functions(uint16_t)

#undef DEF_convert_32_to_uintN_functions

// YUV: bit shift 8-16 <=> 8-16 bits
// shift right or left, depending on expandrange
template<typename pixel_t_s, typename pixel_t_d>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void convert_uint_limited_sse41(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const pixel_t_s* srcp0 = reinterpret_cast<const pixel_t_s*>(srcp);
  pixel_t_d* dstp0 = reinterpret_cast<pixel_t_d*>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  const int src_width = src_rowsize / sizeof(pixel_t_s);

  if (target_bitdepth > source_bitdepth) // expandrange. pixel_t_d is always uint16_t
  {
    const int shift_bits = target_bitdepth - source_bitdepth;
    __m128i shift = _mm_set_epi32(0, 0, 0, shift_bits);
    for (int y = 0; y < src_height; y++)
    {
      for (int x = 0; x < src_width; x += 16)
      {
        __m128i src_lo, src_hi;
        if constexpr (sizeof(pixel_t_s) == 1) {
          auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
          src_lo = _mm_cvtepu8_epi16(src);                       // 8* uint16
          src_hi = _mm_unpackhi_epi8(src, _mm_setzero_si128());  // 8* uint16
        }
        else {
          src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 8* uint_16
          src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 8)); // 8* uint_16
        }
        src_lo = _mm_sll_epi16(src_lo, shift);
        src_hi = _mm_sll_epi16(src_hi, shift);
        if constexpr (sizeof(pixel_t_d) == 1) {
          // upconvert always to 2 bytes
          assert(0);
        }
        else {
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), src_lo);
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x + 8), src_hi);
        }
      }
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
  }
  else
  {
    // reduce range
    const int shift_bits = source_bitdepth - target_bitdepth;
    const int round = 1 << (shift_bits - 1);
    __m128i shift = _mm_set_epi32(0, 0, 0, shift_bits);
    const auto round_simd = _mm_set1_epi16(round);

    for (int y = 0; y < src_height; y++)
    {
      for (int x = 0; x < src_width; x += 16)
      {
        if constexpr (sizeof(pixel_t_s) == 1)
          assert(0);
        // downconvert always from 2 bytes
        auto src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 8* uint_16
        auto src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 8)); // 8* uint_16
        src_lo = _mm_srl_epi16(_mm_adds_epu16(src_lo, round_simd), shift);
        src_hi = _mm_srl_epi16(_mm_adds_epu16(src_hi, round_simd), shift);
        if constexpr (sizeof(pixel_t_d) == 1) {
          // to 8 bits
          auto dst = _mm_packus_epi16(src_lo, src_hi);
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), dst);
        }
        else {
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), src_lo);
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x + 8), src_hi);
        }
      }
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
  }
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_uint_sse41(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  // limited to limited is bitshift, see in other function
  if constexpr (!fulls && !fulld) {
    convert_uint_limited_sse41< pixel_t_s, pixel_t_d>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    return;
  }

  const pixel_t_s* srcp0 = reinterpret_cast<const pixel_t_s*>(srcp);
  pixel_t_d* dstp0 = reinterpret_cast<pixel_t_d*>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  const int src_width = src_rowsize / sizeof(pixel_t_s);

  if constexpr (sizeof(pixel_t_s) == 1 && sizeof(pixel_t_d) == 2) {
    if (fulls && fulld && !chroma && source_bitdepth == 8 && target_bitdepth == 16) {
      // special case 8->16 bit full scale: * 65535 / 255 = *257
      __m128i zero = _mm_setzero_si128();
      __m128i multiplier = _mm_set1_epi16(257);

      for (int y = 0; y < src_height; y++)
      {
        for (int x = 0; x < src_width; x += 16)
        {
          __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
          __m128i src_lo = _mm_unpacklo_epi8(src, zero);             // 8* uint16
          __m128i src_hi = _mm_unpackhi_epi8(src, zero);             // 8* uint16
            // *257 mullo is faster than x*257 = (x<<8 + x) add/or solution (i7)
          __m128i res_lo = _mm_mullo_epi16(src_lo, multiplier); // lower 16 bit of multiplication is enough
          __m128i res_hi = _mm_mullo_epi16(src_hi, multiplier);
          // dstp[x] = srcp0[x] * 257; // RGB: full range 0..255 <-> 0..65535 (257 = 65535 / 255)
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), res_lo);
          _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x + 8), res_hi);
        } // for x
        dstp0 += dst_pitch;
        srcp0 += src_pitch;
      } // for y
      return;
    }
  }

  const int target_max = (1 << target_bitdepth) - 1;

  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto vect_mul_factor = _mm_set1_ps(d.mul_factor);
  auto vect_src_offset = _mm_set1_epi32(d.src_offset_i);
  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  auto vect_dst_offset_plus_round = _mm_set1_ps(dst_offset_plus_round);

  auto vect_target_max = _mm_set1_epi16(target_max);

  auto zero = _mm_setzero_si128();

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x += 16)
    {
      __m128i src_lo, src_hi;
      if constexpr (sizeof(pixel_t_s) == 1) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
        src_lo = _mm_cvtepu8_epi16(src);                   // 8* uint16
        src_hi = _mm_unpackhi_epi8(src, zero);             // 8* uint16
      }
      else {
        src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 8* uint_16
        src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 8)); // 8* uint_16
      }

      __m128i result1, result2;
      {
        // src: 8*uint16
        // convert to int32, bias, to float
        auto res_lo_i = _mm_cvtepu16_epi32(src_lo);
        auto res_hi_i = _mm_unpackhi_epi16(src_lo, zero);
        if constexpr (chroma || !fulls) {
          // src_offset is not zero
          res_lo_i = _mm_sub_epi32(res_lo_i, vect_src_offset);
          res_hi_i = _mm_sub_epi32(res_hi_i, vect_src_offset);
        }
        auto res_lo = _mm_cvtepi32_ps(res_lo_i);
        auto res_hi = _mm_cvtepi32_ps(res_hi_i);
        // multiply, bias back+round
        res_lo = _mm_add_ps(_mm_mul_ps(res_lo, vect_mul_factor), vect_dst_offset_plus_round);
        res_hi = _mm_add_ps(_mm_mul_ps(res_hi, vect_mul_factor), vect_dst_offset_plus_round);
        // convert back w/ truncate
        auto result_l = _mm_cvttps_epi32(res_lo); // no banker's rounding
        auto result_h = _mm_cvttps_epi32(res_hi);
        // 0 min is ensured by packus
        // back to 16 bit
        result1 = _mm_packus_epi32(result_l, result_h); // 8 * 16 bit pixels
        result1 = _mm_min_epu16(result1, vect_target_max);
      }

      // byte target: not yet
      if constexpr (sizeof(pixel_t_d) == 2)
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), result1);

      {
        // src: 8*uint16
        // convert to int32, bias, to float
        auto res_lo_i = _mm_cvtepu16_epi32(src_hi);
        auto res_hi_i = _mm_unpackhi_epi16(src_hi, zero);
        if constexpr (chroma || !fulls) {
          // src_offset is not zero
          res_lo_i = _mm_sub_epi32(res_lo_i, vect_src_offset);
          res_hi_i = _mm_sub_epi32(res_hi_i, vect_src_offset);
        }
        auto res_lo = _mm_cvtepi32_ps(res_lo_i);
        auto res_hi = _mm_cvtepi32_ps(res_hi_i);
        // multiply, bias back+round
        res_lo = _mm_add_ps(_mm_mul_ps(res_lo, vect_mul_factor), vect_dst_offset_plus_round);
        res_hi = _mm_add_ps(_mm_mul_ps(res_hi, vect_mul_factor), vect_dst_offset_plus_round);
        // convert back w/ truncate
        auto result_l = _mm_cvttps_epi32(res_lo);
        auto result_h = _mm_cvttps_epi32(res_hi);
        // 0 min is ensured by packus
        // back to 16 bit
        result2 = _mm_packus_epi32(result_l, result_h); // 8 * 16 bit pixels
        result2 = _mm_min_epu16(result2, vect_target_max);
      }

      if constexpr (sizeof(pixel_t_d) == 2)
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x + 8), result2);
      else {
        // byte target: store both
        auto result12 = _mm_packus_epi16(result1, result2);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), result12);
      }
    } // for x

    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

// instantiate them all
// template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
#define convert_uint_sse4_functions(uint_X_t, uint_X_dest_t) \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, false, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, false, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, false, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, false, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, true, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, true, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, true, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_sse41<uint_X_t, uint_X_dest_t, true, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

convert_uint_sse4_functions(uint8_t, uint8_t)
convert_uint_sse4_functions(uint8_t, uint16_t)
convert_uint_sse4_functions(uint16_t, uint8_t)
convert_uint_sse4_functions(uint16_t, uint16_t)

#undef convert_uint_sse4_functions

// for optimization this one has extra template parameters
// TEMPLATE_NEED_BACKSCALE to support dither_target_bits being lower than target bit depth
template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld, bool TEMPLATE_NEED_BACKSCALE, bool TEMPLATE_LOW_DITHER_BITDEPTH>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void do_convert_ordered_dither_uint_sse41(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const pixel_t_s* srcp = reinterpret_cast<const pixel_t_s*>(srcp8);
  pixel_t_d* dstp = reinterpret_cast<pixel_t_d*>(dstp8);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  // we can safely process whole 16 (8 bit src) / 32 byte (16 bit source) chunks, Avisynth ensures at least 32 byte alignment
  const int src_width = (src_rowsize / sizeof(pixel_t_s) + 15) & ~15;

  // helps compiler optimization
  if constexpr (sizeof(pixel_t_s) == 1)
    source_bitdepth = 8;
  if constexpr (sizeof(pixel_t_d) == 1) {
    target_bitdepth = 8;
    if (!TEMPLATE_NEED_BACKSCALE) {
      dither_target_bitdepth = 8;
    }
  }

  const int max_pixel_value_target = (1 << target_bitdepth) - 1;
  const int max_pixel_value_dithered = (1 << dither_target_bitdepth) - 1;
  // precheck ensures:
  // target_bitdepth >= dither_target_bitdepth
  // source_bitdepth - dither_target_bitdepth <= 8 (max precalculated table is 16x16)
  const bool odd_diff = (source_bitdepth - dither_target_bitdepth) & 1;
  const int dither_bit_diff = (source_bitdepth - dither_target_bitdepth);
  const int dither_order = (dither_bit_diff + 1) / 2;
  const int dither_mask = (1 << dither_order) - 1; // 9,10=2  11,12=4  13,14=8  15,16=16
  // 10->8: 0x01 (2x2)
  // 11->8: 0x03 (4x4)
  // 12->8: 0x03 (4x4)
  // 14->8: 0x07 (8x8)
  // 16->8: 0x0F (16x16)
  const BYTE* matrix;
  switch (dither_order) {
  case 1: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither2x2a.data_sse2 : dither2x2.data_sse2); break;
  case 2: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither4x4a.data_sse2 : dither4x4.data_sse2); break;
  case 3: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither8x8a.data_sse2 : dither8x8.data_sse2); break;
  case 4: matrix = reinterpret_cast<const BYTE*>(odd_diff ? dither16x16a.data : dither16x16.data); break; // no spec sse, already 16 bytes long
  default: return; // n/a
  }

  const int bitdiff_between_dither_and_target = target_bitdepth - dither_target_bitdepth;
  assert(TEMPLATE_NEED_BACKSCALE == (target_bitdepth != dither_target_bitdepth));  // dither to x, target to y

  assert(TEMPLATE_LOW_DITHER_BITDEPTH == (dither_target_bitdepth < 8));
  // e.g. instead of 0,1 => -0.5,+0.5;  0,1,2,3 => -1.5,-0.5,0.5,1.5
  const float half_maxcorr_value = ((1 << dither_bit_diff) - 1) / 2.0f;

  const int source_max = (1 << source_bitdepth) - 1;
  //-----------------------
  // When calculating src_pixel, src and dst are of the same bit depth
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, source_bitdepth); // both is src_bitdepth

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto src_pixel_min = 0;
  const auto src_pixel_max = source_max;
  const float mul_factor_backfromlowdither = (float)max_pixel_value_target / max_pixel_value_dithered;
  //-----------------------

  const auto mul_factor_simd = _mm_set1_ps(d.mul_factor);
  const auto mul_factor_backfromlowdither_simd = _mm_set1_ps(mul_factor_backfromlowdither);
  const auto half_maxcorr_value_simd = _mm_set1_ps(half_maxcorr_value);
  const auto zero = _mm_setzero_si128();
  const BYTE* matrix_simd16 = matrix;
  const BYTE* current_matrix_line;

  for (int y = 0; y < src_height; y++)
  {
   // int _y = (y & dither_mask) << dither_order; // for C version
    current_matrix_line = matrix_simd16 + ((y & dither_mask) << 4); // always 16 byte boundary instead of dither order
    __m128i corr = _mm_load_si128(reinterpret_cast<const __m128i*>(current_matrix_line)); // int corr = matrix[_y | (x & dither_mask)];
    // convert corr to 16 bit
    __m128i corr_lo = _mm_unpacklo_epi8(corr, zero);
    __m128i corr_hi = _mm_unpackhi_epi8(corr, zero);

    for (int x = 0; x < src_width; x += 16)
    {
      //const int corr = matrix[_y | (x & dither_mask)];

      //int src_pixel = srcp[x];
      __m128i src_lo, src_hi;
      if constexpr (sizeof(pixel_t_s) == 1) {
        auto src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 16* uint8
        src_lo = _mm_unpacklo_epi8(src, zero); // 8* uint16
        src_hi = _mm_unpackhi_epi8(src, zero); // 8* uint16
      }
      else {
        src_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); // 8* uint16
        src_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x + 8));
      }

      __m128i src_lo_lo, src_lo_hi, src_hi_lo, src_hi_hi;
      if constexpr (fulls != fulld) {
        // goint to 32 float
        // const float val = (srcp[x] - src_offset) * mul_factor + dst_offset_plus_round;
        const auto src_offset_simd = _mm_set1_epi32(d.src_offset_i);
        const auto dst_offset_plus_round_simd = _mm_set1_ps(dst_offset_plus_round);

        auto src_lo_lo_ps = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpacklo_epi16(src_lo, zero), src_offset_simd));
        auto src_lo_hi_ps = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpackhi_epi16(src_lo, zero), src_offset_simd));
        src_lo_lo_ps = _mm_add_ps(_mm_mul_ps(src_lo_lo_ps, mul_factor_simd), dst_offset_plus_round_simd);
        src_lo_hi_ps = _mm_add_ps(_mm_mul_ps(src_lo_hi_ps, mul_factor_simd), dst_offset_plus_round_simd);

        auto src_hi_lo_ps = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpacklo_epi16(src_hi, zero), src_offset_simd));
        auto src_hi_hi_ps = _mm_cvtepi32_ps(_mm_sub_epi32(_mm_unpackhi_epi16(src_hi, zero), src_offset_simd));
        src_hi_lo_ps = _mm_add_ps(_mm_mul_ps(src_hi_lo_ps, mul_factor_simd), dst_offset_plus_round_simd);
        src_hi_hi_ps = _mm_add_ps(_mm_mul_ps(src_hi_hi_ps, mul_factor_simd), dst_offset_plus_round_simd);

        // src_pixel = clamp((int)val, src_pixel_min, src_pixel_max);
        if constexpr (sizeof(pixel_t_s) == 1 && !TEMPLATE_LOW_DITHER_BITDEPTH) {
          // Source is 8 bits. This branch is ineffective if we define TEMPLATE_LOW_DITHER_BITDEPTH for all <8 bits, as we do now
          // We would reach this code for 5-7 bits when lo limit would be e.g. 4 bits
          // 8 bit source: we can go back from 32 to 16 bit int
          src_lo = _mm_packus_epi32(_mm_cvttps_epi32(src_lo_lo_ps), _mm_cvttps_epi32(src_lo_hi_ps));
          src_hi = _mm_packus_epi32(_mm_cvttps_epi32(src_hi_lo_ps), _mm_cvttps_epi32(src_hi_hi_ps));
          // 0 min is ensured by packus
          const auto src_pixel_max_simd = _mm_set1_epi16(src_pixel_max);
          src_lo = _mm_min_epu16(src_lo, src_pixel_max_simd);
          src_hi = _mm_min_epu16(src_hi, src_pixel_max_simd);
          // tmp result 16 bits in src_lo, src_hi
        }
        else {
          // go back to int32 only
          src_lo_lo = _mm_cvttps_epi32(src_lo_lo_ps);
          src_lo_hi = _mm_cvttps_epi32(src_lo_hi_ps);
          src_hi_lo = _mm_cvttps_epi32(src_hi_lo_ps);
          src_hi_hi = _mm_cvttps_epi32(src_hi_hi_ps);

          // no packus to ensure min 0
          const auto src_pixel_min_simd = _mm_set1_epi32(src_pixel_min);
          src_lo_lo = _mm_max_epi32(src_lo_lo, src_pixel_min_simd);
          src_lo_hi = _mm_max_epi32(src_lo_hi, src_pixel_min_simd);
          src_hi_lo = _mm_max_epi32(src_hi_lo, src_pixel_min_simd);
          src_hi_hi = _mm_max_epi32(src_hi_hi, src_pixel_min_simd);

          const auto src_pixel_max_simd = _mm_set1_epi32(src_pixel_max);
          src_lo_lo = _mm_min_epi32(src_lo_lo, src_pixel_max_simd);
          src_lo_hi = _mm_min_epi32(src_lo_hi, src_pixel_max_simd);
          src_hi_lo = _mm_min_epi32(src_hi_lo, src_pixel_max_simd);
          src_hi_hi = _mm_min_epi32(src_hi_hi, src_pixel_max_simd);
          // tmp result 32 bits in src_lo_lo, src_lo_hi, src_hi_lo, src_hi_hi
        }
      }
      else {
        // 10-16 bits: go into 32 bit int
        if constexpr (sizeof(pixel_t_s) == 2 || TEMPLATE_LOW_DITHER_BITDEPTH) {
          src_lo_lo = _mm_unpacklo_epi16(src_lo, zero);
          src_lo_hi = _mm_unpackhi_epi16(src_lo, zero);
          src_hi_lo = _mm_unpacklo_epi16(src_hi, zero);
          src_hi_hi = _mm_unpackhi_epi16(src_hi, zero);
          // tmp result 32 bits in src_lo_lo, src_lo_hi, src_hi_lo, src_hi_hi
        }
      }

      __m128i new_pixel_lo, new_pixel_hi;

      // scale down after adding dithering noise
      // int new_pixel = ((src_pixel + corr) >> dither_bit_diff);
      if constexpr (sizeof(pixel_t_s) == 1 && !TEMPLATE_LOW_DITHER_BITDEPTH) {
        new_pixel_lo = _mm_srai_epi16(_mm_add_epi16(src_lo, corr_lo), dither_bit_diff);
        new_pixel_hi = _mm_srai_epi16(_mm_add_epi16(src_hi, corr_hi), dither_bit_diff);
        // new_pixel_lo/hi is ready with 16 bit data
      }
      else { 
        // source bits: >8 or 8 bit but PRESHIFT
        // At 16 bit overflow can happen when 0xFFFF it dithered up. This is why we are at 32 bits int
        // Theoretically fulls == fulld && <16 bit would allow 16 bit int workflow but we do not specialize for that
        // low dither bitdepth is special as well

        /*
        int new_pixel;
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          // accurate dither: +/-
          const float corr_f = corr - half_maxcorr_value;
          new_pixel = (int)(src_pixel + corr_f) >> dither_bit_diff;
        }
        else
          new_pixel = ((src_pixel + corr) >> dither_bit_diff);
        */
        __m128i new_pixel_lo_lo, new_pixel_lo_hi, new_pixel_hi_lo, new_pixel_hi_hi;
        auto corr_lo_lo = _mm_unpacklo_epi16(corr_lo, zero);
        auto corr_lo_hi = _mm_unpackhi_epi16(corr_lo, zero);
        auto corr_hi_lo = _mm_unpacklo_epi16(corr_hi, zero);
        auto corr_hi_hi = _mm_unpackhi_epi16(corr_hi, zero);
        if constexpr(TEMPLATE_LOW_DITHER_BITDEPTH) {
          // accurately positioned to the center
          auto corr_lo_lo_ps = _mm_sub_ps(_mm_cvtepi32_ps(corr_lo_lo), half_maxcorr_value_simd);
          auto corr_lo_hi_ps = _mm_sub_ps(_mm_cvtepi32_ps(corr_lo_hi), half_maxcorr_value_simd);
          auto corr_hi_lo_ps = _mm_sub_ps(_mm_cvtepi32_ps(corr_hi_lo), half_maxcorr_value_simd);
          auto corr_hi_hi_ps = _mm_sub_ps(_mm_cvtepi32_ps(corr_hi_hi), half_maxcorr_value_simd);
          // accurate dither: +/-
          new_pixel_lo_lo = _mm_srai_epi32(_mm_cvttps_epi32(_mm_add_ps(_mm_cvtepi32_ps(src_lo_lo), corr_lo_lo_ps)), dither_bit_diff);
          new_pixel_lo_hi = _mm_srai_epi32(_mm_cvttps_epi32(_mm_add_ps(_mm_cvtepi32_ps(src_lo_hi), corr_lo_hi_ps)), dither_bit_diff);
          new_pixel_hi_lo = _mm_srai_epi32(_mm_cvttps_epi32(_mm_add_ps(_mm_cvtepi32_ps(src_hi_lo), corr_hi_lo_ps)), dither_bit_diff);
          new_pixel_hi_hi = _mm_srai_epi32(_mm_cvttps_epi32(_mm_add_ps(_mm_cvtepi32_ps(src_hi_hi), corr_hi_hi_ps)), dither_bit_diff);
        }
        else {
          // Arrgh. If dither_bit_diff would be an immediate constant (e.g. template parameter) then we get 2390 fps vs 1378!
          // for a simple Vide16bit.ConvertBits(8, dither=0, dither_bits=8)
          // const auto dither_bit_diff = 8;
          new_pixel_lo_lo = _mm_srai_epi32(_mm_add_epi32(src_lo_lo, corr_lo_lo), dither_bit_diff);
          new_pixel_lo_hi = _mm_srai_epi32(_mm_add_epi32(src_lo_hi, corr_lo_hi), dither_bit_diff);
          new_pixel_hi_lo = _mm_srai_epi32(_mm_add_epi32(src_hi_lo, corr_hi_lo), dither_bit_diff);
          new_pixel_hi_hi = _mm_srai_epi32(_mm_add_epi32(src_hi_hi, corr_hi_hi), dither_bit_diff);
        }
        new_pixel_lo = _mm_packus_epi32(new_pixel_lo_lo, new_pixel_lo_hi);
        new_pixel_hi = _mm_packus_epi32(new_pixel_hi_lo, new_pixel_hi_hi);
        // new_pixel_lo/hi is ready with 16 bit data
      }

      // scale back to the required bit depth
       // dither to x, target to y
      if constexpr (TEMPLATE_NEED_BACKSCALE) {
        /*
        new_pixel = min(new_pixel, max_pixel_value_dithered);
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          new_pixel = (int)(new_pixel * mul_factor_backfromlowdither + 0.5f);
        }
        else {
          new_pixel = new_pixel << bitdiff_between_dither_and_target;
        }
        */
        const auto max_pixel_value_dithered_simd = _mm_set1_epi16(max_pixel_value_dithered);
        new_pixel_lo = _mm_min_epu16(new_pixel_lo, max_pixel_value_dithered_simd);
        new_pixel_hi = _mm_min_epu16(new_pixel_hi, max_pixel_value_dithered_simd);
        // Interesting problem of dither_bits==1 (or in general at small dither_bits)
        // After simple slli 0,1 becomes 0,128, we'd expect 0,255 instead. So we make cosmetics.
        // dither_bits
        // 1            0,1     => 0,128        => 0,255
        // 2            0,1,2,3 => 0,64,128,192 => 0,?,?,255
        // 3            0,...,7 => 0,32,...,224 => 0,?,?,255
        // 4            0,..,15 => 0,16,...,240 => 0,?,?,255
        // 5            0,..,31 => 0,8,....,248 => 0,?,?,255
        // 6            0,..,63 => 0,4,....,252 => 0,?,?,255
        // 7            0,.,127 => 0,2.  ..,254 => 0,?,?,255
        if constexpr(TEMPLATE_LOW_DITHER_BITDEPTH) {
          // new_pixel = (int)(new_pixel * mul_factor_backfromlowdither + 0.5f);
          auto rounder_half_simd = _mm_set1_ps(0.5f);

          auto new_pixel_lo_lo_ps = _mm_cvtepi32_ps(_mm_unpacklo_epi16(new_pixel_lo, zero));
          auto new_pixel_lo_hi_ps = _mm_cvtepi32_ps(_mm_unpackhi_epi16(new_pixel_lo, zero));
          new_pixel_lo = _mm_packus_epi32(
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(new_pixel_lo_lo_ps, mul_factor_backfromlowdither_simd), rounder_half_simd)),
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(new_pixel_lo_hi_ps, mul_factor_backfromlowdither_simd), rounder_half_simd))
          );
          auto new_pixel_hi_lo_ps = _mm_cvtepi32_ps(_mm_unpacklo_epi16(new_pixel_hi, zero));
          auto new_pixel_hi_hi_ps = _mm_cvtepi32_ps(_mm_unpackhi_epi16(new_pixel_hi, zero));
          new_pixel_hi = _mm_packus_epi32(
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(new_pixel_hi_lo_ps, mul_factor_backfromlowdither_simd), rounder_half_simd)),
            _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(new_pixel_hi_hi_ps, mul_factor_backfromlowdither_simd), rounder_half_simd))
          );
        }
        else {
          new_pixel_lo = _mm_slli_epi16(new_pixel_lo, bitdiff_between_dither_and_target);
          new_pixel_hi = _mm_slli_epi16(new_pixel_hi, bitdiff_between_dither_and_target);
        }
      }

      // dstp[x] = (pixel_t_d)(min((int)new_pixel, max_pixel_value_target));
      if constexpr(sizeof(pixel_t_d) == 1)
      {
        auto result = _mm_packus_epi16(new_pixel_lo, new_pixel_hi);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
      }
      else {
        const auto max_pixel_value_target_simd = _mm_set1_epi16(max_pixel_value_target);
        // max is unnecessary for exact 16 bit pixel type but we are not specialized for that case
        new_pixel_lo = _mm_min_epu16(new_pixel_lo, max_pixel_value_target_simd);
        new_pixel_hi = _mm_min_epu16(new_pixel_hi, max_pixel_value_target_simd);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), new_pixel_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x + 8), new_pixel_hi);
      }
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_ordered_dither_uint_sse41(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const bool need_backscale = target_bitdepth != dither_target_bitdepth; // dither to x, target to y
  const bool low_dither_bitdepth = dither_target_bitdepth < 8; // 1-7 bits dither targets, need_backscale is always true, since 8 bit format is the minimum
  if (need_backscale) {
    if (low_dither_bitdepth)
      do_convert_ordered_dither_uint_sse41<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, true>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    else
    do_convert_ordered_dither_uint_sse41<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
  else {
    do_convert_ordered_dither_uint_sse41<pixel_t_s, pixel_t_d, chroma, fulls, fulld, false, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
}

// instantiate them all
// template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
// spec: fulld=false, fulls=false
#define convert_ordered_dither_uint_sse4_functions(uint_X_t, uint_X_dest_t) \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, false, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_sse41<uint_X_t, uint_X_dest_t, true, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

// dither only same of less bit depth
convert_ordered_dither_uint_sse4_functions(uint8_t, uint8_t)
convert_ordered_dither_uint_sse4_functions(uint16_t, uint8_t)
convert_ordered_dither_uint_sse4_functions(uint16_t, uint16_t)

#undef convert_ordered_dither_uint_sse4_functions


