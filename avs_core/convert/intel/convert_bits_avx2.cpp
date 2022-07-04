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


#include <avs/alignment.h>
#include <avs/minmax.h>

#ifdef AVS_WINDOWS
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

#include "convert_bits_avx2.h"
#include "../convert_bits.h"
#include "../convert_helper.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4305 4309)
#endif

template<typename pixel_t, bool chroma, bool fulls, bool fulld>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_32_to_uintN_avx2(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const float* srcp0 = reinterpret_cast<const float*>(srcp);
  pixel_t* dstp0 = reinterpret_cast<pixel_t*>(dstp);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  const int src_width = src_rowsize / sizeof(float);

  //-----------------------
  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  constexpr auto dst_pixel_min = 0;
  const auto dst_pixel_max = (1 << target_bitdepth) - 1;

  auto src_offset_ps = _mm256_set1_ps(d.src_offset);
  auto factor_ps = _mm256_set1_ps(d.mul_factor);
  auto dst_offset_plus_round_ps = _mm256_set1_ps(dst_offset_plus_round);
  auto dst_pixel_min_ps = _mm256_set1_ps((float)dst_pixel_min);
  auto dst_pixel_max_ps = _mm256_set1_ps((float)dst_pixel_max);

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x += 16) // 16 pixels at a time (64 byte - alignment is OK)
    {
      __m256i result;
      __m256i result_0, result_1;
      __m256 src_0 = _mm256_load_ps(reinterpret_cast<const float*>(srcp0 + x));
      __m256 src_1 = _mm256_load_ps(reinterpret_cast<const float*>(srcp0 + x + 8));
      if constexpr (!chroma && !fulls) {
        // when offset is different from 0
        src_0 = _mm256_sub_ps(src_0, src_offset_ps);
        src_1 = _mm256_sub_ps(src_1, src_offset_ps);
      }

      src_0 = _mm256_fmadd_ps(src_0, factor_ps, dst_offset_plus_round_ps);
      src_1 = _mm256_fmadd_ps(src_1, factor_ps, dst_offset_plus_round_ps);

      src_0 = _mm256_max_ps(_mm256_min_ps(src_0, dst_pixel_max_ps), dst_pixel_min_ps);
      src_1 = _mm256_max_ps(_mm256_min_ps(src_1, dst_pixel_max_ps), dst_pixel_min_ps);
      result_0 = _mm256_cvttps_epi32(src_0); // truncate
      result_1 = _mm256_cvttps_epi32(src_1);
      if constexpr (sizeof(pixel_t) == 2) {
        result = _mm256_packus_epi32(result_0, result_1);
        result = _mm256_permute4x64_epi64(result, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), result);
      }
      else {
        result = _mm256_packs_epi32(result_0, result_1);
        result = _mm256_permute4x64_epi64(result, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        __m128i result128_lo = _mm256_castsi256_si128(result);
        __m128i result128_hi = _mm256_extractf128_si256(result, 1);
        __m128i result128 = _mm_packus_epi16(result128_lo, result128_hi);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp0 + x), result128);
      }

      // c: 
      //const int pixel = (int)((srcp0[x] - src_offset) * mul_factor + dst_offset_plus_round);
      //dstp0[x] = pixel_t(clamp(pixel, dst_pixel_min, dst_pixel_max));
    }
    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#define convert_32_to_uintN_avx2_functions(type) \
template void convert_32_to_uintN_avx2<type, false, true, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, true, true, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, false, true, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, true, true, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, false, false, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, true, false, true>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, false, false, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_32_to_uintN_avx2<type, true, false, false>(const BYTE *srcp, BYTE *dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

convert_32_to_uintN_avx2_functions(uint8_t)
convert_32_to_uintN_avx2_functions(uint16_t)

#undef convert_32_to_uintN_avx2_functions

/*
3.7.1 test26: stop using manual _mm256_zeroupper()
This warning is left in Avisynth at one single place.
1.) compilers do it automatically
2.) clang is getting mad and produces suboptimal prologue and epilogue, saving and loading all xmm registers!
** Good (no manual zeroupper) **
  push	rsi
  .seh_pushreg rsi
  push	rdi
  .seh_pushreg rdi
  .seh_endprologue 
[...]
  pop	rdi
  pop	rsi
  vzeroupper
  ret

** Bad, manual _mm256_zeroupper() **

  vmovaps	xmmword ptr [rsp + 144], xmm15  # 16-byte Spill
  .seh_savexmm xmm15, 144
  vmovaps	xmmword ptr [rsp + 128], xmm14  # 16-byte Spill
  .seh_savexmm xmm14, 128
  vmovaps	xmmword ptr [rsp + 112], xmm13  # 16-byte Spill
  .seh_savexmm xmm13, 112
  vmovaps	xmmword ptr [rsp + 96], xmm12   # 16-byte Spill
  .seh_savexmm xmm12, 96
  vmovaps	xmmword ptr [rsp + 80], xmm11   # 16-byte Spill
  .seh_savexmm xmm11, 80
  vmovaps	xmmword ptr [rsp + 64], xmm10   # 16-byte Spill
  .seh_savexmm xmm10, 64
  vmovaps	xmmword ptr [rsp + 48], xmm9    # 16-byte Spill
  .seh_savexmm xmm9, 48
  vmovaps	xmmword ptr [rsp + 32], xmm8    # 16-byte Spill
  .seh_savexmm xmm8, 32
  vmovaps	xmmword ptr [rsp + 16], xmm7    # 16-byte Spill
  .seh_savexmm xmm7, 16
  vmovaps	xmmword ptr [rsp], xmm6         # 16-byte Spill
  .seh_savexmm xmm6, 0
  .seh_endprologue 
[...]
  pop	rdi
  pop	rsi
  vzeroupper
  vmovaps	xmm6, xmmword ptr [rsp]         # 16-byte Reload
  vmovaps	xmm7, xmmword ptr [rsp + 16]    # 16-byte Reload
  vmovaps	xmm8, xmmword ptr [rsp + 32]    # 16-byte Reload
  vmovaps	xmm9, xmmword ptr [rsp + 48]    # 16-byte Reload
  vmovaps	xmm10, xmmword ptr [rsp + 64]   # 16-byte Reload
  vmovaps	xmm11, xmmword ptr [rsp + 80]   # 16-byte Reload
  vmovaps	xmm12, xmmword ptr [rsp + 96]   # 16-byte Reload
  vmovaps	xmm13, xmmword ptr [rsp + 112]  # 16-byte Reload
  vmovaps	xmm14, xmmword ptr [rsp + 128]  # 16-byte Reload
  vmovaps	xmm15, xmmword ptr [rsp + 144]  # 16-byte Reload
  ret
*/

// YUV: bit shift 8-16 <=> 8-16 bits
// shift right or left, depending on expandrange
template<typename pixel_t_s, typename pixel_t_d>
static void convert_uint_limited_avx2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
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
      for (int x = 0; x < src_width; x += 32)
      {
        __m256i src_lo, src_hi;
        if constexpr (sizeof(pixel_t_s) == 1) {
          auto src_lo_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
          auto src_hi_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 16)); // 16* uint8
          src_lo = _mm256_cvtepu8_epi16(src_lo_128); // 16* uint16
          src_hi = _mm256_cvtepu8_epi16(src_hi_128); // 16* uint16
        }
        else {
          // 64 bytes per cycle. Ok in avs+
          src_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x)); // 16* uint_16
          src_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x + 16)); // 16* uint_16
        }
        src_lo = _mm256_sll_epi16(src_lo, shift);
        src_hi = _mm256_sll_epi16(src_hi, shift);
        if constexpr (sizeof(pixel_t_d) == 1) {
          // upconvert always to 2 bytes
          assert(0);
        }
        else {
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), src_lo);
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x + 16), src_hi);
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
    const auto round_simd = _mm256_set1_epi16(round);

    for (int y = 0; y < src_height; y++)
    {
      for (int x = 0; x < src_width; x += 32)
      {
        if constexpr (sizeof(pixel_t_s) == 1)
          assert(0);
        // downconvert always from 2 bytes
        // 64 bytes per cycle. Ok in avs+
        auto src_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x)); // 16* uint_16
        auto src_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x + 16)); // 16* uint_16
        src_lo = _mm256_srl_epi16(_mm256_adds_epu16(src_lo, round_simd), shift);
        src_hi = _mm256_srl_epi16(_mm256_adds_epu16(src_hi, round_simd), shift);
        if constexpr (sizeof(pixel_t_d) == 1) {
          // to 8 bits
          auto dst = _mm256_packus_epi16(src_lo, src_hi);
          dst = _mm256_permute4x64_epi64(dst, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), dst);
        }
        else {
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), src_lo);
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x + 16), src_hi);
        }
      }
      dstp0 += dst_pitch;
      srcp0 += src_pitch;
    }
  }
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
void convert_uint_avx2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  // limited to limited is bitshift, see in other function
  if constexpr (!fulls && !fulld) {
    convert_uint_limited_avx2<pixel_t_s, pixel_t_d>(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    return;
  }

  const pixel_t_s* srcp0 = reinterpret_cast<const pixel_t_s*>(srcp);
  pixel_t_d* dstp0 = reinterpret_cast<pixel_t_d*>(dstp);

  src_pitch = src_pitch / sizeof(pixel_t_s);
  dst_pitch = dst_pitch / sizeof(pixel_t_d);

  const int src_width = src_rowsize / sizeof(pixel_t_s);
  const int wmod32 = (src_width / 32) * 32;

  if constexpr (sizeof(pixel_t_s) == 1 && sizeof(pixel_t_d) == 2) {
    if (fulls && fulld && !chroma && source_bitdepth == 8 && target_bitdepth == 16) {
      // special case 8->16 bit full scale: * 65535 / 255 = *257
      __m256i multiplier = _mm256_set1_epi16(257);

      for (int y = 0; y < src_height; y++)
      {
        for (int x = 0; x < src_width; x += 32)
        {
          auto src_lo_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
          auto src_hi_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 16)); // 16* uint8
          auto src_lo = _mm256_cvtepu8_epi16(src_lo_128); // 16* uint16
          auto src_hi = _mm256_cvtepu8_epi16(src_hi_128); // 16* uint16

            // *257 mullo is faster than x*257 = (x<<8 + x) add/or solution (i7)
          auto res_lo = _mm256_mullo_epi16(src_lo, multiplier); // lower 16 bit of multiplication is enough
          auto res_hi = _mm256_mullo_epi16(src_hi, multiplier);
          // dstp[x] = srcp0[x] * 257; // RGB: full range 0..255 <-> 0..65535 (257 = 65535 / 255)
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), res_lo);
          _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x + 16), res_hi);
        } // for x
        // rest
        for (int x = wmod32; x < src_width; x++)
        {
          dstp0[x] = (pixel_t_d)(srcp0[x] * 257);
        }
        dstp0 += dst_pitch;
        srcp0 += src_pitch;
      } // for y
      return;
    }
  }

  const int target_max = (1 << target_bitdepth) - 1;

  bits_conv_constants d;
  get_bits_conv_constants(d, chroma, fulls, fulld, source_bitdepth, target_bitdepth);

  auto vect_mul_factor = _mm256_set1_ps(d.mul_factor);
  auto vect_src_offset = _mm256_set1_epi32(d.src_offset_i);
  auto dst_offset_plus_round = d.dst_offset + 0.5f;
  auto vect_dst_offset_plus_round = _mm256_set1_ps(dst_offset_plus_round);

  auto vect_target_max = _mm256_set1_epi16(target_max);

  auto zero = _mm256_setzero_si256();

  for (int y = 0; y < src_height; y++)
  {
    for (int x = 0; x < src_width; x += 32)
    {
      __m256i src_lo, src_hi;
      if constexpr (sizeof(pixel_t_s) == 1) {
        auto src_lo_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x)); // 16* uint8
        auto src_hi_128 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp0 + x + 16)); // 16* uint8
        src_lo = _mm256_cvtepu8_epi16(src_lo_128); // 16* uint16
        src_hi = _mm256_cvtepu8_epi16(src_hi_128); // 16* uint16
      }
      else {
        // 64 bytes per cycle. Ok in avs+
        src_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x)); // 8* uint_16
        src_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp0 + x + 16)); // 8* uint_16
      }

      __m256i result1, result2;
      {
        // src: 8*uint16
        // convert to int32, bias, to float
        auto res_lo_i = _mm256_unpacklo_epi16(src_lo, zero);
        auto res_hi_i = _mm256_unpackhi_epi16(src_lo, zero);
        if constexpr (chroma || !fulls) {
          // src_offset is not zero
          res_lo_i = _mm256_sub_epi32(res_lo_i, vect_src_offset);
          res_hi_i = _mm256_sub_epi32(res_hi_i, vect_src_offset);
        }
        auto res_lo = _mm256_cvtepi32_ps(res_lo_i);
        auto res_hi = _mm256_cvtepi32_ps(res_hi_i);
        // multiply, bias back+round
        // avx2 mode: smart fma instruction inserted even from msvc!
        res_lo = _mm256_add_ps(_mm256_mul_ps(res_lo, vect_mul_factor), vect_dst_offset_plus_round);
        res_hi = _mm256_add_ps(_mm256_mul_ps(res_hi, vect_mul_factor), vect_dst_offset_plus_round);
        // convert back w/ truncate
        auto result_l = _mm256_cvttps_epi32(res_lo); // no banker's rounding
        auto result_h = _mm256_cvttps_epi32(res_hi);
        // 0 min is ensured by packus
        // back to 16 bit
        result1 = _mm256_packus_epi32(result_l, result_h); // 8 * 16 bit pixels
        result1 = _mm256_min_epu16(result1, vect_target_max);
      }

      // byte target: not yet
      if constexpr (sizeof(pixel_t_d) == 2)
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), result1);

      {
        // src: 8*uint16
        // convert to int32, bias, to float
        auto res_lo_i = _mm256_unpacklo_epi16(src_hi, zero);
        auto res_hi_i = _mm256_unpackhi_epi16(src_hi, zero);
        if constexpr (chroma || !fulls) {
          // src_offset is not zero
          res_lo_i = _mm256_sub_epi32(res_lo_i, vect_src_offset);
          res_hi_i = _mm256_sub_epi32(res_hi_i, vect_src_offset);
        }
        auto res_lo = _mm256_cvtepi32_ps(res_lo_i);
        auto res_hi = _mm256_cvtepi32_ps(res_hi_i);
        // multiply, bias back+round
        res_lo = _mm256_add_ps(_mm256_mul_ps(res_lo, vect_mul_factor), vect_dst_offset_plus_round);
        res_hi = _mm256_add_ps(_mm256_mul_ps(res_hi, vect_mul_factor), vect_dst_offset_plus_round);
        // convert back w/ truncate
        auto result_l = _mm256_cvttps_epi32(res_lo);
        auto result_h = _mm256_cvttps_epi32(res_hi);
        // 0 min is ensured by packus
        // back to 16 bit
        result2 = _mm256_packus_epi32(result_l, result_h); // 8 * 16 bit pixels
        result2 = _mm256_min_epu16(result2, vect_target_max);
      }

      if constexpr (sizeof(pixel_t_d) == 2)
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x + 16), result2);
      else {
        // byte target: store both
        auto result12 = _mm256_packus_epi16(result1, result2);
        result12 = _mm256_permute4x64_epi64(result12, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp0 + x), result12);
      }
    } // for x

    dstp0 += dst_pitch;
    srcp0 += src_pitch;
  }
}

// instantiate them all
// template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
// spec: fulld=false, fulls=false
#define convert_uint_avx2_functions(uint_X_t, uint_X_dest_t) \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, false, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, false, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, false, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, false, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, true, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, true, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, true, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_uint_avx2<uint_X_t, uint_X_dest_t, true, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

convert_uint_avx2_functions(uint8_t, uint8_t)
convert_uint_avx2_functions(uint8_t, uint16_t)
convert_uint_avx2_functions(uint16_t, uint8_t)
convert_uint_avx2_functions(uint16_t, uint16_t)

#undef convert_uint_avx2_functions

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld, bool TEMPLATE_NEED_BACKSCALE, bool TEMPLATE_LOW_DITHER_BITDEPTH>
static void do_convert_ordered_dither_uint_avx2(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
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
  //constexpr auto src_pixel_min = 0;
  const auto src_pixel_max = source_max;
  const float mul_factor_backfromlowdither = (float)max_pixel_value_target / max_pixel_value_dithered;
  //-----------------------

  const auto mul_factor_simd = _mm256_set1_ps(d.mul_factor);
  const auto mul_factor_backfromlowdither_simd = _mm256_set1_ps(mul_factor_backfromlowdither);
  const auto half_maxcorr_value_simd = _mm256_set1_ps(half_maxcorr_value);
  const auto zero = _mm256_setzero_si256();
  const BYTE* matrix_simd16 = matrix;
  const BYTE* current_matrix_line;

  for (int y = 0; y < src_height; y++)
  {
    // int _y = (y & dither_mask) << dither_order; // for C version
    current_matrix_line = matrix_simd16 + ((y & dither_mask) << 4); // always 16 byte boundary instead of dither order
    // load and convert corr to 16 bit
    __m256i corr = _mm256_cvtepu8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(current_matrix_line))); // int corr = matrix[_y | (x & dither_mask)];

    for (int x = 0; x < src_width; x += 16)
    {
      //const int corr = matrix[_y | (x & dither_mask)];

      //int src_pixel = srcp[x];
      //__m128i src_lo, src_hi;
      __m256i src;
      if constexpr (sizeof(pixel_t_s) == 1) {
        src = _mm256_cvtepu8_epi16(_mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x))); // 16* uint8->uint16
      }
      else {
        src= _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp + x)); // 16* uint16
      }

      __m256i src_lo, src_hi;
      if constexpr (fulls != fulld) {
        // goint to 32 float
        // const float val = (srcp[x] - src_offset) * mul_factor + dst_offset_plus_round;
        const auto src_offset_simd = _mm256_set1_epi32(d.src_offset_i);
        const auto dst_offset_plus_round_simd = _mm256_set1_ps(dst_offset_plus_round);

        auto src_lo_ps = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpacklo_epi16(src, zero), src_offset_simd));
        auto src_hi_ps = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_unpackhi_epi16(src, zero), src_offset_simd));
        src_lo_ps = _mm256_add_ps(_mm256_mul_ps(src_lo_ps, mul_factor_simd), dst_offset_plus_round_simd);
        src_hi_ps = _mm256_add_ps(_mm256_mul_ps(src_hi_ps, mul_factor_simd), dst_offset_plus_round_simd);

        // src_pixel = clamp((int)val, src_pixel_min, src_pixel_max);
        if constexpr (sizeof(pixel_t_s) == 1 && !TEMPLATE_LOW_DITHER_BITDEPTH) {
          // Source is 8 bits. This branch is ineffective if we define TEMPLATE_LOW_DITHER_BITDEPTH for all <8 bits, as we do now
          // We would reach this code for 5-7 bits when lo limit would be e.g. 4 bits
          // 8 bit source: we can go back from 32 to 16 bit int
          src = _mm256_packus_epi32(_mm256_cvttps_epi32(src_lo_ps), _mm256_cvttps_epi32(src_hi_ps));
          // min is 0, which was ensured by packus
          const auto src_pixel_max_simd = _mm256_set1_epi16(src_pixel_max);
          src = _mm256_min_epu16(src, src_pixel_max_simd);
          // tmp result 16 bits in src
        }
        else {
          // go back to int32 only
          src_lo = _mm256_cvttps_epi32(src_lo_ps);
          src_hi = _mm256_cvttps_epi32(src_hi_ps);
          // min is 0, which was ensured by packus
          const auto src_pixel_max_simd = _mm256_set1_epi32(src_pixel_max);
          src_lo = _mm256_min_epi32(src_lo, src_pixel_max_simd);
          src_hi = _mm256_min_epi32(src_hi, src_pixel_max_simd);
          // tmp result 32 bits in src_lo, src_hi
        }
      }
      else {
        // 10-16 bits or preshift: go into 32 bit int
        if constexpr (sizeof(pixel_t_s) == 2 || TEMPLATE_LOW_DITHER_BITDEPTH) {
          src_lo = _mm256_unpacklo_epi16(src, zero);
          src_hi = _mm256_unpackhi_epi16(src, zero);
          // tmp result 32 bits in src_lo, src_hi
        }
      }

      __m256i new_pixel;

      // scale down after adding dithering noise
      // int new_pixel = ((src_pixel + corr) >> dither_bit_diff);
      if constexpr (sizeof(pixel_t_s) == 1 && !TEMPLATE_LOW_DITHER_BITDEPTH) {
        // when dither_target_bits is 1, then preshift==1 and dither_bit_diff==(7+1) would not fit in 16 bits
        new_pixel = _mm256_srai_epi16(_mm256_add_epi16(src, corr), dither_bit_diff);
        // new_pixel is ready with 16 bit data
      }
      else {
        // source bits: >8.
        // At 16 bit overflow can happen when 0xFFFF it dithered up. This is why we are at 32 bits int
        // Theoretically fulls == fulld && <16 bit would allow 16 bit int workflow but we do not specialize for that
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
        __m256i new_pixel_lo, new_pixel_hi;
        auto corr_lo = _mm256_unpacklo_epi16(corr, zero);
        auto corr_hi = _mm256_unpackhi_epi16(corr, zero);
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          // accurately positioned to the center
          auto corr_lo_ps = _mm256_sub_ps(_mm256_cvtepi32_ps(corr_lo), half_maxcorr_value_simd);
          auto corr_hi_ps = _mm256_sub_ps(_mm256_cvtepi32_ps(corr_hi), half_maxcorr_value_simd);
          // accurate dither: +/-
          new_pixel_lo = _mm256_srai_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_cvtepi32_ps(src_lo), corr_lo_ps)), dither_bit_diff);
          new_pixel_hi = _mm256_srai_epi32(_mm256_cvttps_epi32(_mm256_add_ps(_mm256_cvtepi32_ps(src_hi), corr_hi_ps)), dither_bit_diff);
        }
        else {
          // Arrgh. If dither_bit_diff would be an immediate constant (e.g. template parameter) then we get 2390 fps vs 1378 even for sse4.1!
          // for a simple Vide16bit.ConvertBits(8, dither=0, dither_bits=8)
          // const auto dither_bit_diff = 8;
          new_pixel_lo = _mm256_srai_epi32(_mm256_add_epi32(src_lo, corr_lo), dither_bit_diff);
          new_pixel_hi = _mm256_srai_epi32(_mm256_add_epi32(src_hi, corr_hi), dither_bit_diff);
        }
        new_pixel = _mm256_packus_epi32(new_pixel_lo, new_pixel_hi);
        // new_pixel is ready with 16 bit data
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
        const auto max_pixel_value_dithered_simd = _mm256_set1_epi16(max_pixel_value_dithered);
        new_pixel = _mm256_min_epu16(new_pixel, max_pixel_value_dithered_simd);
        // Interesting problem of dither_bits==1 (or in general at small dither_bits)
        // After simple slli 0,1 becomes 0,128, we'd expect 0,255 instead. So we make cosmetics.
        // dither_bits
        // 1            0,1     => 0,128        => 0,255
        // 2            0,1,2,3 => 0,64,128,192 => 0,85,170,255
        // 3            0,...,7 => 0,32,...,224 => 0,....255
        // 4            0,..,15 => 0,16,...,240 => 0,....255
        // 5            0,..,31 => 0,8,....,248 => 0,....255
        // 6            0,..,63 => 0,4,....,252 => 0,....255
        // 7            0,.,127 => 0,2.  ..,254 => 0,?,?,255
        if (TEMPLATE_LOW_DITHER_BITDEPTH) {
          // new_pixel = (int)(new_pixel * mul_factor_backfromlowdither + 0.5f);
          auto rounder_half_simd = _mm256_set1_ps(0.5f);

          auto new_pixel_lo_ps = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(new_pixel, zero));
          auto new_pixel_hi_ps = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(new_pixel, zero));
          new_pixel = _mm256_packus_epi32(
            _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(new_pixel_lo_ps, mul_factor_backfromlowdither_simd), rounder_half_simd)),
            _mm256_cvttps_epi32(_mm256_add_ps(_mm256_mul_ps(new_pixel_hi_ps, mul_factor_backfromlowdither_simd), rounder_half_simd))
          );
        }
        else {
          new_pixel = _mm256_slli_epi16(new_pixel, bitdiff_between_dither_and_target);
        }
      }

      // dstp[x] = (pixel_t_d)(min((int)new_pixel, max_pixel_value_target));
      if constexpr (sizeof(pixel_t_d) == 1)
      {
        // byte target: 16x 2bytes -> 16x 1 bytes like cvtepu16->epu8
        auto result12 = _mm256_packus_epi16(new_pixel, new_pixel);
        result12 = _mm256_permute4x64_epi64(result12, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm256_castsi256_si128(result12));
      }
      else {
        const auto max_pixel_value_target_simd = _mm256_set1_epi16(max_pixel_value_target);
        // max is unnecessary for exact 16 bit pixel type but we are not specialized for that case
        new_pixel = _mm256_min_epu16(new_pixel, max_pixel_value_target_simd);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x), new_pixel);
      }
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
void convert_ordered_dither_uint_avx2(const BYTE* srcp8, BYTE* dstp8, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth)
{
  const bool need_backscale = target_bitdepth != dither_target_bitdepth; // dither to x, target to y
  const bool low_dither_bitdepth = dither_target_bitdepth < 8; // 1-7 bits dither targets, need_backscale is always true, since 8 bit format is the minimum
  if (need_backscale) {
    if (low_dither_bitdepth)
      do_convert_ordered_dither_uint_avx2<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, true>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
    else
      do_convert_ordered_dither_uint_avx2<pixel_t_s, pixel_t_d, chroma, fulls, fulld, true, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
  else {
    do_convert_ordered_dither_uint_avx2<pixel_t_s, pixel_t_d, chroma, fulls, fulld, false, false>(srcp8, dstp8, src_rowsize, src_height, src_pitch, dst_pitch, source_bitdepth, target_bitdepth, dither_target_bitdepth);
  }
}

// instantiate them all
// template<typename pixel_t_s, typename pixel_t_d, bool chroma, bool fulls, bool fulld>
// spec: fulld=false, fulls=false
#define convert_ordered_dither_uint_sse4_functions(uint_X_t, uint_X_dest_t) \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, false, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, false, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, false, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, true, false>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth); \
template void convert_ordered_dither_uint_avx2<uint_X_t, uint_X_dest_t, true, true, true>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch, int source_bitdepth, int target_bitdepth, int dither_target_bitdepth);

// dither only same of less bit depth
convert_ordered_dither_uint_sse4_functions(uint8_t, uint8_t)
convert_ordered_dither_uint_sse4_functions(uint16_t, uint8_t)
convert_ordered_dither_uint_sse4_functions(uint16_t, uint16_t)

#undef convert_ordered_dither_uint_sse4_functions

