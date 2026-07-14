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


#include <avs/alignment.h>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include <immintrin.h>

#include "convert_planar_avx2.h"
#include "../convert_planar.h"
#include "../convert_helper.h"

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_LOCAL_VARIABLE

// no srai64 in AVX2, only in AVX512 + VL
// AVX2 logical shift (_mm256_srl_epi64) plus sign fill
static AVS_FORCEINLINE __m256i mm256_srai_epi64_simul(__m256i x, int count) {
  // logical right shift on 64bit lanes
  const __m128i cnt = _mm_cvtsi32_si128(count);
  const __m256i logical = _mm256_srl_epi64(x, cnt);

  // have the sign bit present in both 32 bit halves of each 64bit lane before we do a 32bit arithmetic shift.
  const __m256i hi_dup = _mm256_shuffle_epi32(x, _MM_SHUFFLE(3, 3, 1, 1));
  const __m256i sign_shifted = _mm256_sra_epi32(hi_dup, cnt);

  // mask back sign extension
  const __m256i upper_mask = _mm256_set1_epi64x(0xFFFFFFFF00000000ULL);
  const __m256i sign_ext = _mm256_and_si256(sign_shifted, upper_mask);

  return _mm256_or_si256(logical, sign_ext);
}

// packed rgb helper
static AVS_FORCEINLINE __m256i convert_yuv_to_rgb_avx2_core(const __m256i& px0189, const __m256i& px23AB, const __m256i& px45CD, const __m256i& px67EF, const __m256i& zero, const __m256i& matrix, const __m256i& round_mask_plus_rgb_offset) {
  //int b = (((int)m[0] * Y + (int)m[1] * U + (int)m[ 2] * V + 4096 + rgb_offset)>>13);

  //px01 - xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0

  auto low_lo = _mm256_madd_epi16(px0189, matrix); //xx*0 + v1*m2 | u1*m1 + y1*m0 | xx*0 + v0*m2 | u0*m1 + y0*m0
  auto low_hi = _mm256_madd_epi16(px23AB, matrix); //xx*0 + v3*m2 | u3*m1 + y3*m0 | xx*0 + v2*m2 | u2*m1 + y2*m0
  auto high_lo = _mm256_madd_epi16(px45CD, matrix);
  auto high_hi = _mm256_madd_epi16(px67EF, matrix);

  auto low_v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_lo), _mm256_castsi256_ps(low_hi), _MM_SHUFFLE(3, 1, 3, 1))); // v3*m2 | v2*m2 | v1*m2 | v0*m2
  auto high_v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(high_lo), _mm256_castsi256_ps(high_hi), _MM_SHUFFLE(3, 1, 3, 1)));

  auto low_yu = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_lo), _mm256_castsi256_ps(low_hi), _MM_SHUFFLE(2, 0, 2, 0))); // u3*m1 + y3*m0 | u2*m1 + y2*m0 | u1*m1 + y1*m0 | u0*m1 + y0*m0
  auto high_yu = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(high_lo), _mm256_castsi256_ps(high_hi), _MM_SHUFFLE(2, 0, 2, 0)));

  auto t_lo = _mm256_add_epi32(low_v, low_yu); // v3*m2 + u3*m1 + y3*m0...
  auto t_hi = _mm256_add_epi32(high_v, high_yu);

  // v3*m2 + u3*m1 + y3*m0 + 4096 + rgb_offset
  t_lo = _mm256_add_epi32(t_lo, round_mask_plus_rgb_offset);
  t_hi = _mm256_add_epi32(t_hi, round_mask_plus_rgb_offset);

  t_lo = _mm256_srai_epi32(t_lo, 13); // (v3*m2 + u3*m1 + y3*m0 + 4096) >> 13...
  t_hi = _mm256_srai_epi32(t_hi, 13);

  auto result = _mm256_packs_epi32(t_lo, t_hi);
  result = _mm256_packus_epi16(result, zero); //00 00 00 00 00 00 00 00 b15 b14 b14 b12 b11 b10 b9 b8 00 00 00 00 00 00 00 00 b7 b6 b5 b4 b3 b2 b1 b0
  return result;
}

template<int rgb_pixel_step, bool hasAlpha>
void convert_yv24_to_rgb_avx2(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix)
{
  dstp += dst_pitch * (height - 1);  // We start at last line

  size_t mod16_width = rgb_pixel_step == 3 ? width / 16 * 16 : width;
  // for rgb32 target we can process pixels beyond width, but we have 64bit alignment at target 16 pixels to 16*4=64 rgb pixels
  // if alignment would only be 32 bytes, we'd do the last cycle to process only 8 source pixels

  auto matrix_b = _mm256_set_epi16(0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b);
  auto matrix_g = _mm256_set_epi16(0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g);
  auto matrix_r = _mm256_set_epi16(0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r);

  auto zero128 = _mm_setzero_si128();
  auto zero = _mm256_setzero_si256();

  // .13 bit frac integer arithmetic
  int round_mask_plus_rgb_offset_i = (1 << 12) + (matrix.offset_rgb << 13);
  auto round_mask_plus_rgb_offset = _mm256_set1_epi32(round_mask_plus_rgb_offset_i);
  auto offset = _mm256_set_epi16(0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y);

  // 16 YUV(A) pixels --> 4x16 RGB quads = 64 bytes. Avisynth's alignment is 64 fortunately.

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x += 16) {
      __m128i src_y = _mm_load_si128(reinterpret_cast<const __m128i*>(srcY + x)); //Y15 .. Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
      __m128i src_u = _mm_load_si128(reinterpret_cast<const __m128i*>(srcU + x)); //U15 .. U7 U6 U5 U4 U3 U2 U1 U0
      __m128i src_v = _mm_load_si128(reinterpret_cast<const __m128i*>(srcV + x)); //V15 .. V7 V6 V5 V4 V3 V2 V1 V0
      [[maybe_unused]] __m128i src_a;
      if constexpr(hasAlpha)
        src_a = _mm_load_si128(reinterpret_cast<const __m128i*>(srcA + x)); //A15 .. A7 A6 A5 A4 A3 A2 A1 A0

      __m128i t1_lo = _mm_unpacklo_epi8(src_y, src_u); //U7 Y7 U6 Y6 U5 Y5 U4 Y4 U3 Y3 U2 Y2 U1 Y1 U0 Y0
      __m128i t1_hi = _mm_unpackhi_epi8(src_y, src_u); //U15 Y15 U14 Y14 U13 Y13 U12 Y12 U11 Y11 U10 Y10 U9 Y9 U8 Y8
      __m128i t2_lo = _mm_unpacklo_epi8(src_v, zero128);  //00 V7 00 V6 00 V5 00 V4 00 V3 00 V2 00 V1 00 V0
      __m128i t2_hi = _mm_unpackhi_epi8(src_v, zero128);  //00 V15 00 V14 00 V13 00 V12 00 V11 00 V10 00 V9 00 V8

      __m256i t1 = _mm256_set_m128i(t1_hi, t1_lo);
      __m256i t2 = _mm256_set_m128i(t2_hi, t2_lo);
      t1 = _mm256_permute4x64_epi64(t1, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
      t2 = _mm256_permute4x64_epi64(t2, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));

      __m256i low = _mm256_unpacklo_epi16(t1, t2); //xx V11 U11 Y11 xx V10 U10 Y10 xx V9 U9 Y9 xx V8 U8 Y8    xx V3 U3 Y3 xx V2 U2 Y2 xx V1 U1 Y1 xx V0 U0 Y0
      __m256i high = _mm256_unpackhi_epi16(t1, t2); //xx V15 U15 Y15 xx V14 U14 Y14 xx V13 U13 Y13 xx V12 U12 Y12    xx V7 U7 Y7 xx V6 U6 Y6 xx V5 U5 Y5 xx V4 U4 Y4

      __m256i px0189 = _mm256_unpacklo_epi8(low, zero);  //xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0
      __m256i px23AB = _mm256_unpackhi_epi8(low, zero);  //xx xx 00 V3 00 U3 00 Y3 xx xx 00 V2 00 U2 00 Y2
      __m256i px45CD = _mm256_unpacklo_epi8(high, zero); //xx xx 00 V5 00 U5 00 Y5 xx xx 00 V4 00 U4 00 Y4
      __m256i px67EF = _mm256_unpackhi_epi8(high, zero); //xx xx 00 V7 00 U7 00 Y7 xx xx 00 V6 00 U6 00 Y6

      px0189 = _mm256_add_epi16(px0189, offset);
      px23AB = _mm256_add_epi16(px23AB, offset);
      px45CD = _mm256_add_epi16(px45CD, offset);
      px67EF = _mm256_add_epi16(px67EF, offset);

      __m256i result_b = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_b, round_mask_plus_rgb_offset); // b15..0
      __m256i result_g = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_g, round_mask_plus_rgb_offset); // g15..0
      __m256i result_r = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_r, round_mask_plus_rgb_offset); // r15..0

      __m256i result_bg = _mm256_unpacklo_epi8(result_b, result_g); //g15 b15 g14 b14 g13 b13 g12 b12 g11 b11 g10 b10 g9 b9 g8 b8 | g7 b7 g6 b6 g5 b5 g4 b4 g3 b3 g2 b2 g1 b1 g0 b0
      __m256i alpha;
      if constexpr(hasAlpha) {
        __m128i a_lo = _mm_unpacklo_epi8(src_a, zero128);  //00 A7 00 A6 00 A5 00 A4 00 A3 00 A2 00 A1 00 A0
        __m128i a_hi = _mm_unpackhi_epi8(src_a, zero128);  //00 A15 00 A14 00 A13 00 A12 00 A11 00 A10 00 A9 00 A8
        alpha = _mm256_set_m128i(a_hi, a_lo); // a15 .. a0  at low of each m128i part
        alpha = _mm256_permute4x64_epi64(alpha, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
      }
      else
        alpha = _mm256_cmpeq_epi32(result_r, result_r); // FF FF FF FF ... default alpha transparent

      __m256i result_ra = _mm256_unpacklo_epi8(result_r, alpha);       //a7 r7 a6 r6 a5 r5 a4 r4 a3 r3 a2 r2 a1 r1 a0 r0

      __m256i result_lo = _mm256_unpacklo_epi16(result_bg, result_ra); // a11 r11 g11 b11 | a10 r10 g10 b10 | a9 r9 g9 b9 | a8 r8 g8 b8  |   a3 r3 g3 b3 | a2 r2 g2 b2 | a1 r1 g1 b1 | a0 r0 g0 b0
      __m256i result_hi = _mm256_unpackhi_epi16(result_bg, result_ra);
      // note: actual pixel indexes are different from the numbers in comments, they are kept to be follow more easily
      // Initial _mm256_permute4x64_epi64 ensures that the order here is properly argb7..argb0 and argb15..argb8

      if constexpr (rgb_pixel_step == 4) {
        //rgb32
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x * 4), result_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x * 4 + 32), result_hi);
      }
      else {
        // rgb24
        // 16*4 bytes to 16*3 bytes
        __m256i perm10 = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        __m256i shuffle_lo = _mm256_set_epi8(
            0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
            0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0
        );
        __m128i shuffle_hi_lo = _mm_set_epi8(9, 8, 6, 5,  4,  2,  1,  0, 0, 0, 0, 0, 0, 0, 0, 0);

        __m128i result_hi_lo = _mm256_extracti128_si256(result_hi, 0); // aBbBgBrB aAbAgArA a9b9g9r9 a8b8g8r8
        __m128i result_hi_hi = _mm256_extracti128_si256(result_hi, 1); // aFbFgFrF aEbEgErE aDbDgDrD aCbCgCrC
        
        __m256i result_lo_reorg = _mm256_shuffle_epi8(result_lo, shuffle_lo);
        // x  x  x  x  b7g7r7 b6g6r6 b5g5r5 b4g4r4 x  x  x  x  b3g3r3 b2g2r2 b1g1r1 b0g0r0
        result_lo_reorg = _mm256_permutevar8x32_epi32(result_lo_reorg, perm10);
        // x  x  x  x  x x x x b7g7r7 b6g6r6 b5g5r5 b4g4r4 b3g3r3 b2g2r2 b1g1r1 b0g0r0

        __m128i result_hi_lo_reorg = _mm_shuffle_epi8(result_hi_lo, shuffle_hi_lo);
        // gA rA b9 g9 r9 b8 g8 r8 x x x x x x x x

        __m256i dummy_y0 = _mm256_undefined_si256();
        auto result_hi_lo_reorg_2 = _mm256_inserti128_si256(dummy_y0, result_hi_lo_reorg, 1);
        //                                                                      
        // gA rA b9 g9|r9 b8 g8 r8|x x x  x|x x  x x|x  x x x |x x x  x|x x  x x|x  x x x   // result_hi_lo_reorg_2
        // x  x  x  x  x  x  x  x  b7g7r7 b6g6r6 b5g5r5 b4g4r4 b3g3r3 b2g2r2 b1g1r1 b0g0r0  // result_lo_reorg
        auto result_0_15 = _mm256_blend_epi32(result_lo_reorg, result_hi_lo_reorg_2, 0xC0); // 11000000
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp + x * 3), result_0_15); // not necessarily 32 bytes aligned

        // => x  x  x  x  x  x  x  x  x  x  x  x  bB gB rB bA
        __m128i shuffle_hi_lo_2 = _mm_set_epi8((char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, 14, 13, 12, 10);
        __m128i xmm4 = _mm_shuffle_epi8(result_hi_lo, shuffle_hi_lo_2);

        // => bF gF rF bE gE rE bD gD rD bC gC rC x  x  x  x
        __m128i shuffle_hi_hi = _mm_set_epi8(14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, (char)0x80, (char)0x80, (char)0x80, (char)0x80);
        __m128i xmm5 = _mm_shuffle_epi8(result_hi_hi, shuffle_hi_hi);

        // => bF gF rF bE gE rE bD gD rD bC gC rC bB gB rB bA
        __m128i result_16_23 = _mm_or_si128(xmm4, xmm5);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x * 3 + 32), result_16_23);
#if 0
        // Intel compiler can cope with it 100% optimized. No store, just shuffles and blends as above.
        alignas(32) BYTE temp[64];
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp), result_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp + 32), result_hi);
        for (int i = 0; i < 16; ++i) {
          dstp[(x + i) * 3 + 0] = temp[i * 4 + 0];
          dstp[(x + i) * 3 + 1] = temp[i * 4 + 1];
          dstp[(x + i) * 3 + 2] = temp[i * 4 + 2];
        }
#endif
      }
    }

    if constexpr (rgb_pixel_step == 3) {
      // for rgb32 (pixel_step == 4) we processed full width and more, including padded bytes
      for (size_t x = mod16_width; x < width; ++x) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + round_mask_plus_rgb_offset_i) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + round_mask_plus_rgb_offset_i) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + round_mask_plus_rgb_offset_i) >> 13);
        dstp[x * rgb_pixel_step + 0] = PixelClip(b);
        dstp[x * rgb_pixel_step + 1] = PixelClip(g);
        dstp[x * rgb_pixel_step + 2] = PixelClip(r);
        if constexpr (rgb_pixel_step == 4) { // n/a
          dstp[x * 4 + 3] = 255;
        }
      }
    }
    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
    if constexpr(hasAlpha)
      srcA += src_pitch_a;
  }
}

//instantiate
//template<int rgb_pixel_step, bool targetHasAlpha>
template void convert_yv24_to_rgb_avx2<3, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<4, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<3, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<4, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);

#define XP_LAMBDA_CAPTURE_FIX(x) (void)(x)
/*
================================================================================
YUV ↔ RGB Color Space Conversion - Unified Implementation
================================================================================

This module provides a unified, highly optimized implementation for color space
conversions between YUV and RGB formats, supporting all bit depths (8-16 bit
integer and 32-bit float) with optional bit-depth conversion.

CONVERSION DIRECTIONS:
  - YUV_TO_RGB: YUV → RGB (e.g., video decode path)
  - RGB_TO_YUV: RGB → YUV (e.g., video encode path)
  - YUV_TO_YUV: YUV → YUV (e.g., BT.601 → BT.709 matrix conversion)
  - RGB_TO_RGB: RGB → RGB (e.g., gamut/color correction)

ARITHMETIC PRECISION:
  - YUV→RGB / RGB→RGB: 13-bit fixed-point (coefficients scaled by 2^13 = 8192)
  - RGB→YUV: 15-bit fixed-point (coefficients scaled by 2^15 = 32768)
  - YUV→YUV: 14-bit fixed-point (fused matrix precision)

PLANE MEMORY LAYOUT:
  - YUV: Plane[0]=Y, Plane[1]=U, Plane[2]=V
  - RGB: Plane[0]=G, Plane[1]=B, Plane[2]=R (AviSynth convention)

================================================================================
*/

/**
 * @enum YuvRgbConversionType
 * @brief Defines the conversion workflow type for color space transformations
 *
 * Each conversion type determines:
 * - Whether to use integer or float arithmetic for the matrix multiply
 * - How input/output offsets are applied
 * - Whether bit-depth conversion is integrated into the workflow
 * - Whether 16-bit pivot trick is needed for exact 16-bit integer paths
 *
 * CONVERSION TYPE COMPARISON TABLE:
 * ┌─────────────────────┬────────────┬─────────────┬──────────────┬─────────────┬──────────────┐
 * │ Conversion Type     │ 16-bit     │ Input       │ Matrix       │ Post-Matrix │ Bit-Depth    │
 * │                     │ Pivot?     │ Offset      │ Coefficients │ Rounding    │ Scaling      │
 * ├─────────────────────┼────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
 * │ NATIVE_INT          │ Yes (16)   │ After load  │ Integer      │ Yes (shift) │ None         │
 * │                     │ No (<16)   │ (16-bit add)│ (13/15-bit)  │ Integrated  │              │
 * ├─────────────────────┼────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
 * │ FORCE_FLOAT         │ N/A        │ Pre-matrix  │ Float        │ Float+0.5   │ In scale_f   │
 * │                     │            │ (float add) │ (scaled)     │ (if int out)│              │
 * ├─────────────────────┼────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
 * │ FLOAT_OUTPUT        │ Yes (16)   │ After load  │ Integer      │ No shift    │ Post-matrix  │
 * │                     │ No (<16)   │ (16-bit add)│ (13/15-bit)  │ (int→float) │ (scale_f)    │
 * ├─────────────────────┼────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
 * │ BITCONV_INT_LIMITED │ Yes (16)   │ After load  │ Integer      │ Yes (adj.   │ Integrated   │
 * │                     │ No (<16)   │ (16-bit add)│ (13/15-bit)  │ shift)      │ into shift   │
 * ├─────────────────────┼────────────┼─────────────┼──────────────┼─────────────┼──────────────┤
 * │ BITCONV_INT_FULL    │ N/A        │ Pre-matrix  │ Float        │ Float+0.5   │ In scale_f   │
 * │                     │            │ (float add) │ (scaled)     │ (if int out)│              │
 * └─────────────────────┴────────────┴─────────────┴──────────────┴─────────────┴──────────────┘
 *
 * OFFSET APPLICATION DETAILS:
 * ┌─────────────────────┬─────────────────────────────────────────────────────────────────────┐
 * │ Conversion Type     │ Offset Application Strategy                                         │
 * ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
 * │ NATIVE_INT          │ • <16-bit: offset_in added as int16 after load                      │
 * │                     │ • 16-bit: offset_in in 32-bit patch after MADD (with pivot)         │
 * │                     │ • Output: offset_out in MADD rounding constant (scaled by 2^13/15)  │
 * ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
 * │ FORCE_FLOAT         │ • Input: offset_in added as float before matrix multiply            │
 * │                     │ • Output: offset_out added as float in matrix result                │
 * │                     │ • Coefficients pre-scaled by scale_f for bit-depth conversion       │
 * ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
 * │ FLOAT_OUTPUT        │ • Input: Same as NATIVE_INT (int16 or 32-bit patch)                 │
 * │                     │ • Output: No offset in MADD; added post-conversion in float domain  │
 * │                     │ • Result: int32 → float with scale_f, then + offset_out_f           │
 * ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
 * │ BITCONV_INT_LIMITED │ • Same as NATIVE_INT but with adjusted shift (13 - bit_diff)        │
 * │                     │ • Chroma offset uses SOURCE bit depth (half_pixel_offset)           │
 * │                     │ • Bit-depth scaling integrated into the right-shift operation       │
 * ├─────────────────────┼─────────────────────────────────────────────────────────────────────┤
 * │ BITCONV_INT_FULL    │ • Same as FORCE_FLOAT (uses float workflow internally)              │
 * │                     │ • Required for full-range bit-depth conversions (e.g., 10→8 full)   │
 * └─────────────────────┴─────────────────────────────────────────────────────────────────────┘
 *
 * 16-BIT PIVOT TECHNIQUE (for integer paths only):
 *   Problem: uint16 range [0, 65535] doesn't fit in signed int16 [-32768, 32767]
 *   Solution: Flip MSB to pivot unsigned to signed: Y_signed = Y XOR 0x8000
 *   - 0 → -32768, 65535 → 32767 (all values fit in int16)
 *   - Correction applied in 32-bit patch: add (32768 + offset_in) × coefficients
 *   - See detailed explanation at top of function
 *
 * CHROMA CENTERING:
 *   YUV Input:  U/V centered around half (e.g., 128 for 8-bit) → subtract half
 *   YUV Output: U/V centered around half → add half after matrix
 *   RGB: No centering (all values have same zero point)
 *
 * PERFORMANCE CHARACTERISTICS:
 *   - AVX2: 32 pixels per iteration (primary target, 2013+ CPUs)
 *   - SSE2: 8 pixels per iteration (legacy compatibility)
 *   - Integer MADD path: ~4000-5400 fps (1920×1080, 8-bit, Ryzen)
 *   - Float workflow: ~3200-3900 fps (10-26% slower, but necessary for some cases)
 *
 * EXAMPLE USE CASES:
 *   YUV→RGB, 10-bit→10-bit, limited range:
 *     → NATIVE_INT (integer MADD, no bit conversion)
 *
 *   YUV→RGB, 10-bit→8-bit, limited→limited:
 *     → BITCONV_INT_LIMITED (integrated bit scaling in shift)
 *
 *   YUV→RGB, 10-bit→8-bit, full→full:
 *     → BITCONV_INT_FULL (uses float workflow internally)
 *
 *   YUV→RGB, 16-bit→32-bit float:
 *     → FORCE_FLOAT (required for float input)
 *
 *   YUV→RGB, 8-bit→32-bit float, limited→full:
 *     → FLOAT_OUTPUT (integer matrix, float output with range expansion)
 *
 *   YUV(BT.601)→YUV(BT.709), 10-bit→10-bit:
 *     → NATIVE_INT with fused conversion matrix (avoids RGB intermediate)
 */

/**
 * @brief Universal color space conversion function (AVX2 optimized)
 *
 * Converts between YUV and RGB color spaces with optional bit-depth conversion.
 * Supports all combinations of 8/10/12/14/16-bit integer and 32-bit float formats.
 *
 * @tparam direction        Conversion direction (YUV_TO_RGB, RGB_TO_YUV, YUV_TO_YUV, RGB_TO_RGB)
 * @tparam pixel_t          Input pixel type (uint8_t, uint16_t, or float)
 * @tparam lessthan16bit    True if input is 8/10/12/14-bit (enables optimizations)
 * @tparam lessthan16bit_target True if output is 8/10/12/14-bit (enables clamping)
 * @tparam pixel_t_dst      Output pixel type (uint8_t, uint16_t, or float)
 * @tparam conv_type        Conversion workflow type (see YuvRgbConversionType)
 *
 * @param dstp              Output plane pointers [3] (G/Y, B/U, R/V)
 * @param dstPitch          Output plane pitches in bytes [3]
 * @param srcp              Input plane pointers [3] (G/Y, B/U, R/V)
 * @param srcPitch          Input plane pitches in bytes [3]
 * @param width             Frame width in pixels
 * @param height            Frame height in pixels
 * @param m                 Conversion matrix with coefficients and offsets
 * @param bits_per_pixel    Input bit depth (8-16)
 * @param bits_per_pixel_target Output bit depth (8-16)
 *
 * @note Matrix coefficients in ConversionMatrix must be pre-scaled:
 *       - Integer coefficients: scaled by 2^13 (YUV↔RGB) or 2^15 (RGB→YUV)
 *       - Float coefficients: unscaled, will be multiplied by scale_f internally
 *
 * @note For 16-bit integer paths, the function uses a pivot trick to work around
 *       signed int16 range limitations. See detailed comment at function start.
 *
 * @note Processes 32 pixels per iteration on AVX2. For float input/output, includes
 *       safety checks to respect 64-byte scanline alignment guarantees.
 */
template<ConversionDirection direction, typename pixel_t, bool lessthan16bit, bool lessthan16bit_target, typename pixel_t_dst, YuvRgbConversionType conv_type>
static void convert_yuv_to_planarrgb_avx2_internal(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m,
  int bits_per_pixel, int bits_per_pixel_target)
{

  constexpr int INT_ARITH_SHIFT =
    (direction == ConversionDirection::YUV_TO_RGB) ? 13 :
    (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_Y) ? 15 :
    (direction == ConversionDirection::YUV_TO_YUV) ? 14 : 13;

  // 8 bit        uint8_t
  // 10,12,14 bit uint16_t (signed range)
  // 16 bit logic:
  /*
    1. pivot the pixel:
      Convert uint16 pixel Y to a signed int16 by flipping the MSB (this is mathematically identical to Y−32768).
        0 becomes −32768.
        65535 becomes 32767.
        All values now fit in int16 without saturation.
    2. reorganize the formula (offset_in is negative when exists, e.g. -4096):
       Original: (Y + offset_y) * Cy
       Substitute
       Y = (Ysigned​ + 32768)
       => (Ysigned​ + 32768 + offset_y) * Cy​
       => (Ysigned​ * Cy​) + (32768 + offset_y) * Cy​)
    3. correction after the madd section:
       Add ((32768 + offset_y) * Cy​) to the existing 32-bit rounding/offset_in constant.
    4. Output rgb offset_in is also added to the precalculated patch.
  */

  // When input is float (pixel_t), float workflow (FORCE_FLOAT) is compulsory.
  // But not the opposite: FORCE_FLOAT and not float input is fine
  // make a static assert
  static_assert(!(std::is_floating_point<pixel_t>::value && conv_type != YuvRgbConversionType::FORCE_FLOAT), "FORCE_FLOAT conversion type is required for float input pixel type");

  constexpr bool final_is_float = std::is_floating_point<pixel_t_dst>::value;
  constexpr bool would_need_64bit_v_patch = !lessthan16bit && !final_is_float && (INT_ARITH_SHIFT == 15);
  // RGB_TO_YUV, RGB_TO_Y 16 bit origin case needs 64 bit extension at one point to avoid overflow
  // but it makes it much slower than the full-float path, so we force float path for that case as well

  // effectively full-float-inside for int full range conversion, but with the same output rules as other float conversions
  constexpr bool force_float = (conv_type == YuvRgbConversionType::FORCE_FLOAT) || would_need_64bit_v_patch;

  constexpr bool need_int_conversion_narrow_range = conv_type == YuvRgbConversionType::BITCONV_INT_LIMITED;       // full_d is false
  constexpr bool need_int_conversion_full_range = conv_type == YuvRgbConversionType::BITCONV_INT_FULL; // full_d is true
  constexpr bool need_int_conversion = conv_type == YuvRgbConversionType::BITCONV_INT_FULL || conv_type == YuvRgbConversionType::BITCONV_INT_LIMITED ||
    (conv_type == YuvRgbConversionType::FORCE_FLOAT && !final_is_float);

  constexpr bool float_matrix_workflow = force_float || need_int_conversion_full_range; // effectively full-float-inside for int full range conversion 

  // quasi-constexpr, may help optimizer
  if constexpr (std::is_same<pixel_t, uint8_t>::value) bits_per_pixel = 8;
  if constexpr (std::is_same<pixel_t_dst, uint8_t>::value) bits_per_pixel_target = 8;
  if constexpr (std::is_same<pixel_t, uint16_t>::value && !lessthan16bit) bits_per_pixel = 16;
  if constexpr (std::is_same<pixel_t_dst, uint16_t>::value && !lessthan16bit_target) bits_per_pixel_target = 16;
  if constexpr (conv_type == YuvRgbConversionType::NATIVE_INT) bits_per_pixel_target = bits_per_pixel;

  const int bit_diff = need_int_conversion ? bits_per_pixel_target - bits_per_pixel : 0;
  const int target_shift = need_int_conversion_narrow_range ? INT_ARITH_SHIFT - bit_diff : INT_ARITH_SHIFT; // int->int narrow range: integrate the bit depth conversion into the scaling back

  const int ROUNDER = (final_is_float || float_matrix_workflow) ? 0 : (1 << (target_shift - 1)); // 0 when float internal calculation is involved

  // For integer workflow + final is float:
  const float out_offset_f = m.offset_out_f_32; // for post-32-bit float conversion

  const int half_pixel_offset = 1 << (bits_per_pixel - 1); // YUV -->  // at input is float: n/a
  const int half_pixel_offset_target = 1 << (bits_per_pixel_target - 1); // For float_matrix_workflow or output is float
  const int max_pixel_value_target = (1 << bits_per_pixel_target) - 1;

  bits_conv_constants conversion_ranges;
  const bool full_scale_d = m.offset_out == 0;
  // matrix is handling only the same-bit-depth factor. Output result must be scaled as if
  // we further correct it with a post-matrix scaling for bit depth conversion which depends only
  // on the destination limite/full
  get_bits_conv_constants(conversion_ranges, false, full_scale_d, full_scale_d, bits_per_pixel, bits_per_pixel_target); // our final scale would be as if work to float

  constexpr int int_arithmetic_shift = 1 << INT_ARITH_SHIFT;
  float scale_f = conversion_ranges.mul_factor;
  // Integer matrix workflow + 32 bit final float output extra rules
  // - no rounding and scaling back INT_ARITH_SHIFT e.g. 13 for YUV-RGB, 15 for RGB-YUV bits
  // - no clamping to bit depth limits
  // - the INT_ARITH_SHIFT-bit scaling factor is integrated into the bits_per_pixel shift
  if (final_is_float && !float_matrix_workflow)
    scale_f = scale_f / int_arithmetic_shift; // int workflow, float at the end

  __m256i half = _mm256_set1_epi16((short)half_pixel_offset);  // 128
  __m256i limit = _mm256_set1_epi16((short)max_pixel_value_target); // 255

  // to be able to use it as signed 16 bit in madd; 4096+(16<<13) would not fit into i16
  // multiplier is 4096 instead of 1:
  // original   : 1      * 4096
  // needed     : 1      * (4096 + offset_rgb<<13)  (4096 + 131072 overflows i16)
  // changed to : 4096   * (1 + offset_rgb>>(13-1)
  // Except for exact 16 bit, where we do the output offset_in adjustment along with the 16 bit signed-unsigned pivot fix
  constexpr int ROUND_SCALE = 1 << (INT_ARITH_SHIFT - 1); // 1 << 12, for 13 bit integer arithmetic: "0.5"
  const __m256i m256i_round_scale = _mm256_set1_epi16(ROUND_SCALE);

  int round_mask_plus_offset_out_scaled_i;
  int round_mask_plus_offset_out_chroma_scaled_i;

  // integer workflow + exact 16 bit path: needs a special handling for the pivot and input offset_in and output rgb offset_out
  __m256i v_patch_G, v_patch_B, v_patch_R;
  __m256i sign_flip_mask = _mm256_set1_epi16((short)0x8000); // for 16 bit pivot

  // Full-range is float-workflow inside, so no need to do any of the above adjustments in
  // integer arithmetic, just do the full float conversion and clamp at the end if needed

  // Input offset_in handling
  const int offset_in_scalar = m.offset_in;
  const int offset_out_scalar = m.offset_out;

  __m256i offset_in;
  __m256 offset_in_f;
  if constexpr (float_matrix_workflow) {
    offset_in = _mm256_set1_epi32(offset_in_scalar); // float workflow, integer inputs
    offset_in_f = _mm256_set1_ps(m.offset_in_f); // float workflow, float inputs
  }
  else if constexpr (lessthan16bit) // integer workflow, lessthan16 bit path
    offset_in = _mm256_set1_epi16((short)offset_in_scalar); // input offset_in correction can happen in 16 bit arithmetic
  else // n/a, for exact 16 bit integer workflow, the offset_in is handled in the 32-bit patch together with the pivot adjustment
    offset_in = _mm256_setzero_si256();

  if constexpr (!float_matrix_workflow) {
    // integer preparations for the madd-based main loop
    if constexpr (lessthan16bit) {
      // 8-14 bit
      // offset of same magnitude as coeffs, also in simd madd
      round_mask_plus_offset_out_scaled_i = final_is_float ? 0 : (ROUNDER + (offset_out_scalar << INT_ARITH_SHIFT)) / ROUND_SCALE;
      round_mask_plus_offset_out_chroma_scaled_i = final_is_float ? 0 : (ROUNDER + (half_pixel_offset << INT_ARITH_SHIFT)) / ROUND_SCALE;
      v_patch_G = v_patch_B = v_patch_R = _mm256_setzero_si256(); // No patch needed, since the signed 16-bit workaround is not needed.
    }
    else {
      // exact 16 bit
      // keep madd simple: only handle the rounding (ROUND_SCALE * 1)
      // rgb offset is handled later in the patch, after the madd, added to the same place as the pivot adjustment
      round_mask_plus_offset_out_scaled_i = ROUNDER / ROUND_SCALE; // effectively 1 or 0 (final_is_float)
      round_mask_plus_offset_out_chroma_scaled_i = ROUNDER / ROUND_SCALE; // effectively 1 or 0 (final_is_float)

      // move BOTH the pivot and the output offset to the 32-bit patch
      // Since we have to do the patching anyway, we can combine both adjustments here
      const int luma_or_rgbin_pivot = 32768 + offset_in_scalar;
      const int chroma_pivot = 32768;
      const int offset_out_for_patch = final_is_float ? 0 : (offset_out_scalar << INT_ARITH_SHIFT); // 32-bit post-conversion adds offset in float domain.
      const int chroma_offset_out_for_patch = final_is_float ? 0 : (half_pixel_offset << INT_ARITH_SHIFT); // 32-bit post-conversion adds offset in float domain.

      if constexpr (direction == ConversionDirection::YUV_TO_RGB) {
        // for YUV->RGB, the pivot adjustment is needed for all three channels since the luma coeffs are not zero, but pivot only the Y
        v_patch_G = _mm256_set1_epi32(luma_or_rgbin_pivot * m.y_g + offset_out_for_patch);
        v_patch_B = _mm256_set1_epi32(luma_or_rgbin_pivot * m.y_b + offset_out_for_patch);
        v_patch_R = _mm256_set1_epi32(luma_or_rgbin_pivot * m.y_r + offset_out_for_patch);
      }
      else if constexpr (direction == ConversionDirection::RGB_TO_RGB) {
        // for RGB->RGB, the pivot adjustment is needed for all three channels in and aout
        v_patch_G = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.y_g + m.u_g + m.v_g) + offset_out_for_patch);
        v_patch_B = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.y_b + m.u_b + m.v_b) + offset_out_for_patch);
        v_patch_R = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.y_r + m.u_r + m.v_r) + offset_out_for_patch);
      }
      else if constexpr (direction == ConversionDirection::RGB_TO_YUV) {
        // RGB→YUV: All RGB inputs are pivoted, need to account for all three
        // Out0=Y (luma offset), Out1=U and Out2=V (chroma offset)
        // Y output = y_r*R + y_g*G + y_b*B
        if constexpr (!would_need_64bit_v_patch) {
          v_patch_G = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.y_r + m.y_g + m.y_b) + offset_out_for_patch);
          // U output = u_r*R + u_g*G + u_b*B
          v_patch_B = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.u_r + m.u_g + m.u_b) + chroma_offset_out_for_patch);
          // V output = v_r*R + v_g*G + v_b*B
          v_patch_R = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.v_r + m.v_g + m.v_b) + chroma_offset_out_for_patch);
        }
        else {
          // this path must error out, never can be used, redirected to full-float workflow for this case due to speed reasons.
          static_assert(!would_need_64bit_v_patch, "64-bit patch needed for RGB->YUV conversion, but not supported in this path. Redirected to float workflow instead.");
          v_patch_G = _mm256_set1_epi64x(luma_or_rgbin_pivot * (m.y_r + m.y_g + m.y_b) + offset_out_for_patch);
          v_patch_B = _mm256_set1_epi64x(luma_or_rgbin_pivot * (m.u_r + m.u_g + m.u_b) + chroma_offset_out_for_patch);
          v_patch_R = _mm256_set1_epi64x(luma_or_rgbin_pivot * (m.v_r + m.v_g + m.v_b) + chroma_offset_out_for_patch);
        }
      }
      else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
        // RGB→YUV: All RGB inputs are pivoted, need to account for all three
        // Out0=Y (luma offset), No chroma
        // Y output = y_r*R + y_g*G + y_b*B
        if constexpr (!would_need_64bit_v_patch) {
          v_patch_G = _mm256_set1_epi32(luma_or_rgbin_pivot * (m.y_r + m.y_g + m.y_b) + offset_out_for_patch);
          //v_patch_B = _mm256_set1_epi32(0); // no chroma
          //v_patch_R = _mm256_set1_epi32(0); // no chroma
        }
        else {
          // this path must error out, never can be used, redirected to full-float workflow for this case due to speed reasons.
          static_assert(!would_need_64bit_v_patch, "64-bit patch needed for RGB->Y conversion, but not supported in this path. Redirected to float workflow instead.");
          v_patch_G = _mm256_set1_epi64x(luma_or_rgbin_pivot * (m.y_r + m.y_g + m.y_b) + offset_out_for_patch);
          //v_patch_B = _mm256_set1_epi64x(0); // no chroma
          //v_patch_R = _mm256_set1_epi64x(0); // no chroma
        }
      }
      else if constexpr (direction == ConversionDirection::YUV_TO_YUV) {
        // for YUV->YUV
        // FIXME: untested, no AviSynth caller yet. Suspect m.y_b/m.y_r should be
        // m.u_b/m.v_r (chroma diagonal), and offset should be chroma_offset_out_for_patch.
        v_patch_G = _mm256_set1_epi32(luma_or_rgbin_pivot * m.y_g + offset_out_for_patch);
        v_patch_B = _mm256_set1_epi32(chroma_pivot * m.y_b + offset_out_for_patch);
        v_patch_R = _mm256_set1_epi32(chroma_pivot * m.y_r + offset_out_for_patch);
      }
    }
  }

  // For integer workflow + final is float
  const __m256 out_offset_f_avx2 = _mm256_set1_ps(out_offset_f);

  __m256i zero = _mm256_setzero_si256();
  const __m256 scale_f_avx2 = _mm256_set1_ps(scale_f);

  // For integer workflow
  __m256i m_uy_G, m_vr_G, m_uy_B, m_vr_B, m_uy_R, m_vr_R; // integer arithmetic, including 32-bit float target when non-float_matrix_workflow
  // For float_matrix_workflow: define coefficient accessors based on direction
  __m256 coeff_out0_in0, coeff_out0_in1, coeff_out0_in2;  // First output row
  __m256 coeff_out1_in0, coeff_out1_in1, coeff_out1_in2;  // Second output row
  __m256 coeff_out2_in0, coeff_out2_in1, coeff_out2_in2;  // Third output row

  __m256 m_offset_out_y_or_g_f, m_offset_out_u_or_b_f, m_offset_out_v_or_r_f; // three separate offsets, in YUV-RGB they are the same, in RGB-YUV they are different for luma and chroma

  if constexpr (float_matrix_workflow) {
    __m256 m_y_g_f, m_y_b_f, m_y_r_f, m_u_g_f, m_u_b_f, m_u_r_f, m_v_g_f, m_v_b_f, m_v_r_f;
    m_y_g_f = _mm256_mul_ps(_mm256_set1_ps(m.y_g_f), scale_f_avx2);
    m_y_b_f = _mm256_mul_ps(_mm256_set1_ps(m.y_b_f), scale_f_avx2);
    m_y_r_f = _mm256_mul_ps(_mm256_set1_ps(m.y_r_f), scale_f_avx2);
    if constexpr (direction != ConversionDirection::RGB_TO_Y) {
      m_u_g_f = _mm256_mul_ps(_mm256_set1_ps(m.u_g_f), scale_f_avx2);
      m_u_b_f = _mm256_mul_ps(_mm256_set1_ps(m.u_b_f), scale_f_avx2);
      m_u_r_f = _mm256_mul_ps(_mm256_set1_ps(m.u_r_f), scale_f_avx2);
      m_v_g_f = _mm256_mul_ps(_mm256_set1_ps(m.v_g_f), scale_f_avx2);
      m_v_b_f = _mm256_mul_ps(_mm256_set1_ps(m.v_b_f), scale_f_avx2);
      m_v_r_f = _mm256_mul_ps(_mm256_set1_ps(m.v_r_f), scale_f_avx2);
    }

    if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
      // RGB output: same offset for all channels
      float rgb_out_offset_scaled = m.offset_out_f * scale_f;
      m_offset_out_y_or_g_f = _mm256_set1_ps(rgb_out_offset_scaled);
      m_offset_out_u_or_b_f = _mm256_set1_ps(rgb_out_offset_scaled);
      m_offset_out_v_or_r_f = _mm256_set1_ps(rgb_out_offset_scaled);
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {

      // YUV output
      float y_out_offset = m.offset_out_f;
      float y_out_offset_scaled = y_out_offset * scale_f;

      // U/V center offset - should NOT be scaled by scale_f!
      // The matrix output is already in the target range, just add the center
      float uv_center_offset = final_is_float ? 0.0f : (float)half_pixel_offset_target;

      m_offset_out_y_or_g_f = _mm256_set1_ps(y_out_offset_scaled);
      m_offset_out_u_or_b_f = _mm256_set1_ps(uv_center_offset);
      m_offset_out_v_or_r_f = _mm256_set1_ps(uv_center_offset);
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
      // no chroma output
      float y_out_offset_scaled = m.offset_out_f * scale_f;
      m_offset_out_y_or_g_f = _mm256_set1_ps(y_out_offset_scaled);
    }

    if constexpr (direction == ConversionDirection::YUV_TO_RGB) {
      // YUV→RGB: outputs are RGB (rows G,B,R), inputs are YUV (columns Y,U,V)
      // Out0=G, Out1=B, Out2=R
      coeff_out0_in0 = m_y_g_f; coeff_out0_in1 = m_u_g_f; coeff_out0_in2 = m_v_g_f;
      coeff_out1_in0 = m_y_b_f; coeff_out1_in1 = m_u_b_f; coeff_out1_in2 = m_v_b_f;
      coeff_out2_in0 = m_y_r_f; coeff_out2_in1 = m_u_r_f; coeff_out2_in2 = m_v_r_f;
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_YUV) {
      // RGB→YUV: outputs are YUV (rows Y,U,V), inputs are RGB in memory order (G, B, R)
      // Out0=Y, Out1=U, Out2=V
      // Inputs: in0=G, in1=B, in2=R
      // Y = y_r*R + y_g*G + y_b*B
      //   = y_g*in0 + y_b*in1 + y_r*in2
      coeff_out0_in0 = m_y_g_f; coeff_out0_in1 = m_y_b_f; coeff_out0_in2 = m_y_r_f;

      // U = u_r*R + u_g*G + u_b*B
      //   = u_g*in0 + u_b*in1 + u_r*in2
      coeff_out1_in0 = m_u_g_f; coeff_out1_in1 = m_u_b_f; coeff_out1_in2 = m_u_r_f;

      // V = v_r*R + v_g*G + v_b*B
      //   = v_g*in0 + v_b*in1 + v_r*in2
      coeff_out2_in0 = m_v_g_f; coeff_out2_in1 = m_v_b_f; coeff_out2_in2 = m_v_r_f;
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
      // RGB→YUV: outputs are YUV (rows Y,U,V), inputs are RGB in memory order (G, B, R)
      // Out0=Y, No chroma output
      // Inputs: in0=G, in1=B, in2=R
      // Y = y_r*R + y_g*G + y_b*B
      //   = y_g*in0 + y_b*in1 + y_r*in2
      coeff_out0_in0 = m_y_g_f; coeff_out0_in1 = m_y_b_f; coeff_out0_in2 = m_y_r_f;
    }
    else if constexpr (direction == ConversionDirection::YUV_TO_YUV) {
      // YUV→YUV: outputs are YUV (rows Y,U,V), inputs are YUV (columns Y,U,V)
      // For BT.601→BT.709 conversion, this is a fused matrix
      // Out0=Y_out, Out1=U_out, Out2=V_out
      coeff_out0_in0 = m_y_g_f; coeff_out0_in1 = m_u_g_f; coeff_out0_in2 = m_v_g_f;
      coeff_out1_in0 = m_y_b_f; coeff_out1_in1 = m_u_b_f; coeff_out1_in2 = m_v_b_f;
      coeff_out2_in0 = m_y_r_f; coeff_out2_in1 = m_u_r_f; coeff_out2_in2 = m_v_r_f;
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_RGB) { // RGB_TO_RGB
      // RGB→RGB: outputs are RGB (rows R,G,B), inputs are RGB (columns R,G,B)
      // For gamut conversion or color correction
      // Out0=G_out, Out1=B_out, Out2=R_out
      coeff_out0_in0 = m_y_g_f; coeff_out0_in1 = m_u_g_f; coeff_out0_in2 = m_v_g_f;
      coeff_out1_in0 = m_y_b_f; coeff_out1_in1 = m_u_b_f; coeff_out1_in2 = m_v_b_f;
      coeff_out2_in0 = m_y_r_f; coeff_out2_in1 = m_u_r_f; coeff_out2_in2 = m_v_r_f;
    }
  }
  else {
    // Integer workflow coefficient setup
    int round_and_out_offset_y_or_g;
    int round_and_out_offset_u_or_b;
    int round_and_out_offset_v_or_r;

    if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
      // output RGB: same offset for all channels
      round_and_out_offset_y_or_g = round_mask_plus_offset_out_scaled_i;
      round_and_out_offset_u_or_b = round_mask_plus_offset_out_scaled_i;
      round_and_out_offset_v_or_r = round_mask_plus_offset_out_scaled_i;
    }
    else  if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {
      // output YUV: different offsets for Y vs U/V
      round_and_out_offset_y_or_g = round_mask_plus_offset_out_scaled_i;
      round_and_out_offset_u_or_b = round_mask_plus_offset_out_chroma_scaled_i;
      round_and_out_offset_v_or_r = round_mask_plus_offset_out_chroma_scaled_i;
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
      // no chroma output
      round_and_out_offset_y_or_g = round_mask_plus_offset_out_scaled_i;
    }

    // Pack coefficients based on direction
    // For MADD packing: [coeff_in1, coeff_in0] and [coeff_in2, round]

    if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
      // YUV→RGB: Out0=G, Out1=B, Out2=R
      // YUV→YUV: Out0=Y, Out1=U, Out2=V
      // G = y_g*Y + u_g*U + v_g*V
      // Inputs are: in0=Y, in1=U, in2=V
      m_uy_G = _mm256_set1_epi32((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.u_g));
      m_vr_G = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_y_or_g) << 16) | static_cast<uint16_t>(m.v_g));

      // B = y_b*Y + u_b*U + v_b*V
      m_uy_B = _mm256_set1_epi32((static_cast<uint16_t>(m.y_b) << 16) | static_cast<uint16_t>(m.u_b));
      m_vr_B = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_u_or_b) << 16) | static_cast<uint16_t>(m.v_b));

      // R = y_r*Y + u_r*U + v_r*V
      m_uy_R = _mm256_set1_epi32((static_cast<uint16_t>(m.y_r) << 16) | static_cast<uint16_t>(m.u_r));
      m_vr_R = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_v_or_r) << 16) | static_cast<uint16_t>(m.v_r));
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB) {
      // RGB→YUV: Out0=Y, Out1=U, Out2=V
      // RGB→RGB: Out0=G, Out1=B, Out2=R
      // Inputs are: in0=G, in1=B, in2=R

      // Y = y_r*R + y_g*G + y_b*B
      //   = y_g*in0 + y_b*in1 + y_r*in2
      m_uy_G = _mm256_set1_epi32((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.y_b));
      m_vr_G = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_y_or_g) << 16) | static_cast<uint16_t>(m.y_r));

      // U = u_r*R + u_g*G + u_b*B
      //   = u_g*in0 + u_b*in1 + u_r*in2
      m_uy_B = _mm256_set1_epi32((static_cast<uint16_t>(m.u_g) << 16) | static_cast<uint16_t>(m.u_b));
      m_vr_B = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_u_or_b) << 16) | static_cast<uint16_t>(m.u_r));

      // V = v_r*R + v_g*G + v_b*B
      //   = v_g*in0 + v_b*in1 + v_r*in2
      m_uy_R = _mm256_set1_epi32((static_cast<uint16_t>(m.v_g) << 16) | static_cast<uint16_t>(m.v_b));
      m_vr_R = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_v_or_r) << 16) | static_cast<uint16_t>(m.v_r));
    }
    else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
      // RGB→Y: Out0=Y, No chroma output
      // Inputs are: in0=G, in1=B, in2=R
      // Y = y_r*R + y_g*G + y_b*B
      //   = y_g*in0 + y_b*in1 + y_r*in2
      m_uy_G = _mm256_set1_epi32((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.y_b));
      m_vr_G = _mm256_set1_epi32((static_cast<uint16_t>(round_and_out_offset_y_or_g) << 16) | static_cast<uint16_t>(m.y_r));
    }
  }

  const int rowsize = width * sizeof(pixel_t);
  const int rowsize_aligned = AlignNumber(rowsize, FRAME_ALIGN); // 64
  for (int yy = 0; yy < height; yy++) {
    // 32 pixels per loop: 32 * 8 bit = 32 bytes; 32 * 16 bit = 64 bytes
    // Note: extra care needed for 32-bit float input or output, since 32xfloat is over
    // Avisynth+ guaranteed scanline alignment (64 bytes)
    // Ideally, we have to separate the processing into 32 pixel then 16 pixel chunks.
    // See ConvertBits to float AVX2 version for reference.
    // Actually, we only check a pre-calculated limit inside the loop, before the storage. (FIXME)
    // But I guess, when processing so many pixels at once, it's not that big penalty.
    // However duplicating the loop body seen below for the two float-problematic cases, that _is_ penalty :)

    for (int x = 0; x < rowsize; x += 32 * sizeof(pixel_t)) {

      // ConversionDirection::YUV_TO_RGB: y, u, v
      // ConversionDirection::RGB_TO_YUV: g, b, r

      // Regardless of the YUV/RGB source, we ise in0_1, in1_1, in2_1 for the first 16 pixels, and in0_2, in1_2, in2_2 for the second 16 pixels
      // Layout for 0-1-2: Y-U-V or G-B-R

      // integer input pixels 
      __m256i in0_1, in1_1, in2_1;
      __m256i in0_2, in1_2, in2_2;
      // float workflow
      __m256 in0_1_f_lo, in0_1_f_hi, in0_2_f_lo, in0_2_f_hi;
      __m256 in1_1_f_lo, in1_1_f_hi, in1_2_f_lo, in1_2_f_hi;
      __m256 in2_1_f_lo, in2_1_f_hi, in2_2_f_lo, in2_2_f_hi;

      // Avisynth FRAME_ALIGN == 64, so when final_is_float, we cannot process 32 pixels at once at the end of the line
      // Can I write all 32 pixels (blockA+B+C+D)?
      // yes: write all 4 blocks (blockA, B, C, D)
      // no : Write only first 2 blocks (blockA, B), 16 floats are guaranteed safe due to 64-byte alignment
      bool safe_last_16float_64bytes;
      if constexpr (final_is_float || sizeof(pixel_t) == 4)
        safe_last_16float_64bytes = (x + 32 * (int)sizeof(pixel_t)) <= rowsize_aligned;
      else
        safe_last_16float_64bytes = false;

      // Note on widening strategy (Y, U, V) for the float internal path, where series of widenings are required:
      //
      // We use unpacklo/hi_epi16 for widening Y, U and V from (uint8 to) (u)int16 to int32,
      // rather than _mm256_extracti128_si256 + (cvtepu8_epi16) + cvtepi16_epi32/cvtepu16_epi32.
      // For U and V, an additional srai_epi16 generates the sign fill word needed
      // for correct sign extension of the centered (negative-capable) chroma values.
      //
      // Although the extract+cvt approach would produce a sequential lane layout
      // that avoids the permute2f128 shuffles before the final float store, it is
      // measurably slower (Possibly port contention).

      // Load pixels, for the integer path, we pivot for exact 16 bit Y later
      if constexpr (sizeof(pixel_t) == 1) {
        // Load 32 bytes (32 pixels), unpack to two 16-bit registers
        __m256i in0_raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[0] + x)); // 0..7 8..15 16..23 24..31 Y or G
        __m256i in1_raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[1] + x)); // U or B
        __m256i in2_raw = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[2] + x)); // V or R

        if constexpr (sizeof(pixel_t_dst) == 2) {
          // 8->16 bit: pre-shuffle to avoid permutes at the end
          in0_raw = _mm256_permute4x64_epi64(in0_raw, 0xD8); // (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6)
          // 0..7, 16..23, 8..15, 24..31
          in1_raw = _mm256_permute4x64_epi64(in1_raw, 0xD8);
          in2_raw = _mm256_permute4x64_epi64(in2_raw, 0xD8);
        }
        in0_1 = _mm256_unpacklo_epi8(in0_raw, zero); in0_2 = _mm256_unpackhi_epi8(in0_raw, zero);
        // in0_1: 0..7, 8..15, in0_2: 16..23, 24..31
        in1_1 = _mm256_unpacklo_epi8(in1_raw, zero); in1_2 = _mm256_unpackhi_epi8(in1_raw, zero);
        in2_1 = _mm256_unpacklo_epi8(in2_raw, zero); in2_2 = _mm256_unpackhi_epi8(in2_raw, zero);
      }
      else if constexpr (sizeof(pixel_t) == 2) {
        // Load 64 bytes (32 pixels)
        in0_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[0] + x));
        in1_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[1] + x));
        in2_1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[2] + x));
        in0_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[0] + x + 32));
        in1_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[1] + x + 32));
        in2_2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(srcp[2] + x + 32));
      }
      else { // if constexpr (sizeof(pixel_t) == 4) {
        // this also implies full float workflow
        // Load 128 bytes (32 pixels) // safety! we are well beyond Avisynth's 64 byte alignment
        in0_1_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[0] + x));
        in0_1_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[0] + x + 32));
        in1_1_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[1] + x));
        in1_1_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[1] + x + 32));
        in2_1_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[2] + x));
        in2_1_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[2] + x + 32));
        if (safe_last_16float_64bytes) {
          in0_2_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[0] + x + 64));
          in0_2_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[0] + x + 96));
          in1_2_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[1] + x + 64));
          in1_2_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[1] + x + 96));
          in2_2_f_lo = _mm256_load_ps(reinterpret_cast<const float*>(srcp[2] + x + 64));
          in2_2_f_hi = _mm256_load_ps(reinterpret_cast<const float*>(srcp[2] + x + 96));
        }
        else {
          // fill them with something to silence the warning, they won't be stored, either.
          in0_2_f_lo = in0_2_f_hi = in1_2_f_lo = in1_2_f_hi = in2_2_f_lo = in2_2_f_hi = _mm256_setzero_ps();
        }
      }

      if constexpr (float_matrix_workflow) {
        // complete float workflow, even for integer targets, do matrixes in float math.
        // Also for maximum accuracy.

        // convert integer pixels to float, only for non-float input
        if constexpr (sizeof(pixel_t) == 4) {
          // even float input can be "limited" in Avisynth, correct with offset_in as well
          // YUV chroma is zero centered, no offset correction needed
          if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
            // YUV Source
            // Y
            in0_1_f_lo = _mm256_add_ps(in0_1_f_lo, offset_in_f);
            in0_1_f_hi = _mm256_add_ps(in0_1_f_hi, offset_in_f);
            in0_2_f_lo = _mm256_add_ps(in0_2_f_lo, offset_in_f);
            in0_2_f_hi = _mm256_add_ps(in0_2_f_hi, offset_in_f);
          }
          else {
            // RGB source: all channels need offset_in adjustments if studio RGB RGB_TO_RGB, RGB_TO_YUV, RGB_TO_Y
            in0_1_f_lo = _mm256_add_ps(in0_1_f_lo, offset_in_f);
            in0_1_f_hi = _mm256_add_ps(in0_1_f_hi, offset_in_f);
            in0_2_f_lo = _mm256_add_ps(in0_2_f_lo, offset_in_f);
            in0_2_f_hi = _mm256_add_ps(in0_2_f_hi, offset_in_f);
            in1_1_f_lo = _mm256_add_ps(in1_1_f_lo, offset_in_f);
            in1_1_f_hi = _mm256_add_ps(in1_1_f_hi, offset_in_f);
            in1_2_f_lo = _mm256_add_ps(in1_2_f_lo, offset_in_f);
            in1_2_f_hi = _mm256_add_ps(in1_2_f_hi, offset_in_f);
            in2_1_f_lo = _mm256_add_ps(in2_1_f_lo, offset_in_f);
            in2_1_f_hi = _mm256_add_ps(in2_1_f_hi, offset_in_f);
            in2_2_f_lo = _mm256_add_ps(in2_2_f_lo, offset_in_f);
            in2_2_f_hi = _mm256_add_ps(in2_2_f_hi, offset_in_f);
          }

        }
        else {
          // integer input

          // Order: move to int32, and only then adjust offset_in.
          // Superblacks would yield negative values that can only be sign-extend to int32
          // with excessive operations, see the U and V workaround.

          // We've already put the input offset_in into each int32. We use add, offset_in is stored as negative.

          // int32->float

          if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
            // YUV Source
            // Y
            in0_1_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in0_1, zero), offset_in));
            in0_1_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in0_1, zero), offset_in));
            in0_2_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in0_2, zero), offset_in));
            in0_2_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in0_2, zero), offset_in));

            // UV to sign extend
            // Though from SSE4.1 we could use _mm_cvtepi16_epi32 directly but it makes packus lane
            // crossing correction difficult (and slow). Mayne from avx512 _mm512_cvtepi16_ps (only signed 16 exists)?

            // No lane keeping cvtepi16_epi32 here either.
            // U,V, make signed, then int16->int32

            // sign-extend int16 chroma to int32, then convert to float
            // using unpacklo/hi to preserve lane layout compatible with downstream packus
            auto chroma_to_float = [&](__m256i c, __m256& f_lo, __m256& f_hi) {
              c = _mm256_sub_epi16(c, half);
              __m256i sign = _mm256_srai_epi16(c, 15); // 0xFFFF if negative, 0x0000 if positive
              f_lo = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(c, sign));
              f_hi = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(c, sign));
              };

            chroma_to_float(in1_1, in1_1_f_lo, in1_1_f_hi); // U1
            chroma_to_float(in2_1, in2_1_f_lo, in2_1_f_hi); // V1
            chroma_to_float(in1_2, in1_2_f_lo, in1_2_f_hi); // U2
            chroma_to_float(in2_2, in2_2_f_lo, in2_2_f_hi); // V2
          }
          else {
            // RGB source: only offset_in adjustment, no centering needed
            // Green
            in0_1_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in0_1, zero), offset_in));
            in0_1_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in0_1, zero), offset_in));
            in0_2_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in0_2, zero), offset_in));
            in0_2_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in0_2, zero), offset_in));
            // Blue
            in1_1_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in1_1, zero), offset_in));
            in1_1_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in1_1, zero), offset_in));
            in1_2_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in1_2, zero), offset_in));
            in1_2_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in1_2, zero), offset_in));
            // Red
            in2_1_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in2_1, zero), offset_in));
            in2_1_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in2_1, zero), offset_in));
            in2_2_f_lo = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpacklo_epi16(in2_2, zero), offset_in));
            in2_2_f_hi = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_unpackhi_epi16(in2_2, zero), offset_in));
          }
        } // end of integer to float conversion and offset_in adjustment

        // COMMON matrix multiply code - works for all 4 directions!

        auto matrix_multiply = [&](
          __m256 in0_lo, __m256 in0_hi,
          __m256 in1_lo, __m256 in1_hi,
          __m256 in2_lo, __m256 in2_hi,
          __m256& out0_lo, __m256& out0_hi,
          __m256& out1_lo, __m256& out1_hi,
          __m256& out2_lo, __m256& out2_hi)
          {
            XP_LAMBDA_CAPTURE_FIX(coeff_out0_in0);
            XP_LAMBDA_CAPTURE_FIX(coeff_out0_in1);
            XP_LAMBDA_CAPTURE_FIX(coeff_out0_in2);
            XP_LAMBDA_CAPTURE_FIX(coeff_out1_in0);
            XP_LAMBDA_CAPTURE_FIX(coeff_out1_in1);
            XP_LAMBDA_CAPTURE_FIX(coeff_out1_in2);
            XP_LAMBDA_CAPTURE_FIX(coeff_out2_in0);
            XP_LAMBDA_CAPTURE_FIX(coeff_out2_in1);
            XP_LAMBDA_CAPTURE_FIX(coeff_out2_in2);
            XP_LAMBDA_CAPTURE_FIX(m_offset_out_y_or_g_f);
            XP_LAMBDA_CAPTURE_FIX(m_offset_out_u_or_b_f);
            XP_LAMBDA_CAPTURE_FIX(m_offset_out_v_or_r_f);

            out0_lo = _mm256_fmadd_ps(coeff_out0_in2, in2_lo, _mm256_fmadd_ps(coeff_out0_in1, in1_lo, _mm256_fmadd_ps(coeff_out0_in0, in0_lo, m_offset_out_y_or_g_f)));
            out0_hi = _mm256_fmadd_ps(coeff_out0_in2, in2_hi, _mm256_fmadd_ps(coeff_out0_in1, in1_hi, _mm256_fmadd_ps(coeff_out0_in0, in0_hi, m_offset_out_y_or_g_f)));

            if constexpr (direction != ConversionDirection::RGB_TO_Y) {

              out1_lo = _mm256_fmadd_ps(coeff_out1_in2, in2_lo, _mm256_fmadd_ps(coeff_out1_in1, in1_lo, _mm256_fmadd_ps(coeff_out1_in0, in0_lo, m_offset_out_u_or_b_f)));
              out1_hi = _mm256_fmadd_ps(coeff_out1_in2, in2_hi, _mm256_fmadd_ps(coeff_out1_in1, in1_hi, _mm256_fmadd_ps(coeff_out1_in0, in0_hi, m_offset_out_u_or_b_f)));

              out2_lo = _mm256_fmadd_ps(coeff_out2_in2, in2_lo, _mm256_fmadd_ps(coeff_out2_in1, in1_lo, _mm256_fmadd_ps(coeff_out2_in0, in0_lo, m_offset_out_v_or_r_f)));
              out2_hi = _mm256_fmadd_ps(coeff_out2_in2, in2_hi, _mm256_fmadd_ps(coeff_out2_in1, in1_hi, _mm256_fmadd_ps(coeff_out2_in0, in0_hi, m_offset_out_v_or_r_f)));
            }
          };

        // output variables
        __m256 out0_1_f_lo, out0_1_f_hi, out1_1_f_lo, out1_1_f_hi, out2_1_f_lo, out2_1_f_hi;
        __m256 out0_2_f_lo, out0_2_f_hi, out1_2_f_lo, out1_2_f_hi, out2_2_f_lo, out2_2_f_hi;

        // SET#1: First 16 pixels
        matrix_multiply(
          in0_1_f_lo, in0_1_f_hi, in1_1_f_lo, in1_1_f_hi, in2_1_f_lo, in2_1_f_hi, // in0 in1 in2
          out0_1_f_lo, out0_1_f_hi, out1_1_f_lo, out1_1_f_hi, out2_1_f_lo, out2_1_f_hi // out0 out1 out2: Pixels 0..7 in lo, 8..15 in hi
        );

        // SET#2: Next 16 pixels
        matrix_multiply(
          in0_2_f_lo, in0_2_f_hi, in1_2_f_lo, in1_2_f_hi, in2_2_f_lo, in2_2_f_hi, // in0 in1 in2
          out0_2_f_lo, out0_2_f_hi, out1_2_f_lo, out1_2_f_hi, out2_2_f_lo, out2_2_f_hi // out0 out1 out2: Pixels 16..23 in lo, 24..31 in hi
        );

        auto process_from_float_plane_avx2 = [&](BYTE* plane_ptr, __m256 lo_1, __m256 hi_1, __m256 lo_2, __m256 hi_2) {
          XP_LAMBDA_CAPTURE_FIX(zero);
          XP_LAMBDA_CAPTURE_FIX(limit);

#ifdef XP_TLS
          if (final_is_float) {
#else
          if constexpr (final_is_float) {
#endif
            // float inside path, + 32 bit float output
            const int pix_idx = x / sizeof(pixel_t);
            float* f_dst = reinterpret_cast<float*>(plane_ptr) + pix_idx;

            // Define the blocks (8 pixels each) correctly by healing the lanes
            // Remember: this is still quicker that using lane-preserving cvt's earlier.
            __m256 blockA, blockB, blockC, blockD;

            if constexpr (sizeof(pixel_t) == 1) {
              // 8-bit: The lanes were swapped across res1 and res2
              blockA = _mm256_permute2f128_ps(lo_1, hi_1, 0x20); // Pixels 0-7
              blockB = _mm256_permute2f128_ps(lo_2, hi_2, 0x20); // Pixels 8-15
              blockC = _mm256_permute2f128_ps(lo_1, hi_1, 0x31); // Pixels 16-23
              blockD = _mm256_permute2f128_ps(lo_2, hi_2, 0x31); // Pixels 24-31
            }
            else if constexpr (sizeof(pixel_t) == 2) {
              // 16-bit: res1 is 0-15, res2 is 16-31
              blockA = _mm256_permute2f128_ps(lo_1, hi_1, 0x20); // Pixels 0-7
              blockB = _mm256_permute2f128_ps(lo_1, hi_1, 0x31); // Pixels 8-15
              blockC = _mm256_permute2f128_ps(lo_2, hi_2, 0x20); // Pixels 16-23
              blockD = _mm256_permute2f128_ps(lo_2, hi_2, 0x31); // Pixels 24-31
            }
            else {
              // 32-bit: no lane swap needed, but we still need to define the blocks for the tail processing
              blockA = lo_1; // Pixels 0-7
              blockB = hi_1; // Pixels 8-15
              blockC = lo_2; // Pixels 16-23
              blockD = hi_2; // Pixels 24-31
            }

            _mm256_store_ps(f_dst, blockA);
            _mm256_store_ps(f_dst + 8, blockB);

            // Safety check: only store the first half of the 32-pixel block if row width allows
            if (safe_last_16float_64bytes) {
              _mm256_store_ps(f_dst + 16, blockC);
              _mm256_store_ps(f_dst + 24, blockD);
            }
          }
          else {
            // Float workflow + integer output with rounding and clamping
            // r,g,b = static_cast<int>(r,g,b_f + 0.5f);
            // back to int32 domain
            __m256 float_rounder = _mm256_set1_ps(0.5f);
            __m256i res1_lo = _mm256_cvttps_epi32(_mm256_add_ps(lo_1, float_rounder));
            __m256i res1_hi = _mm256_cvttps_epi32(_mm256_add_ps(hi_1, float_rounder));
            __m256i res2_lo = _mm256_cvttps_epi32(_mm256_add_ps(lo_2, float_rounder));
            __m256i res2_hi = _mm256_cvttps_epi32(_mm256_add_ps(hi_2, float_rounder));

            const int pix_idx = x * sizeof(pixel_t_dst) / sizeof(pixel_t);

            __m256i p1 = _mm256_packus_epi32(res1_lo, res1_hi);
            __m256i p2 = _mm256_packus_epi32(res2_lo, res2_hi);

            if constexpr (sizeof(pixel_t_dst) == 1) {
              // X->8 bits
              __m256i final8 = _mm256_packus_epi16(p1, p2);
              if constexpr (sizeof(pixel_t) == 4) {
                // 32->8 bits
                /*
                  res1_lo: int32 [ 0  1  2  3 |  4  5  6  7]
                  res1_hi: int32 [ 8  9 10 11 | 12 13 14 15]
                  res2_lo: int32 [16 17 18 19 | 20 21 22 23]
                  res2_hi: int32 [24 25 26 27 | 28 29 30 31]
                  - after packus_epi32:
                  p1 = [0..3, 8..11 | 4..7, 12..15]
                  p2 = [16..19, 24..27 | 20..23, 28..31]
                  - after packus_epi16 p1, p2: (more lane crossing)
                  p8 = [0..3, 8..11, 16..19, 24..27 | 4..7, 12..15, 20..23, 28..31]

                  I need final8: 32 bytes [0,1,2..31]
                */
                __m256i p1 = _mm256_packus_epi32(res1_lo, res1_hi);
                __m256i p2 = _mm256_packus_epi32(res2_lo, res2_hi);
                __m256i p8 = _mm256_packus_epi16(p1, p2);
                // p8 = [0..3, 8..11, 16..19, 24..27 | 4..7, 12..15, 20..23, 28..31]
                // as 8x int32: [A, B, C, D, E, F, G, H] where we need [A,E,B,F,C,G,D,H]
                const __m256i idx = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
                final8 = _mm256_permutevar8x32_epi32(p8, idx);
              }
              else if constexpr (sizeof(pixel_t) == 2) {
                // 16->8 bits
                final8 = _mm256_permute4x64_epi64(final8, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
              }
              else {
                // 8->8 bits, no change needed
              }
              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx), final8);
            }
            else if constexpr (sizeof(pixel_t_dst) == 2) {
              // x->16 bits
              if constexpr (lessthan16bit_target) {
                p1 = _mm256_min_epi16(p1, limit);
                p2 = _mm256_min_epi16(p2, limit);
              }
              if constexpr (sizeof(pixel_t) == 1) {
                // 8->16 bits:
                /* If we'd not pre-shuffled the input for 8->16 bit, we would need to do this shuffle at the end to heal the lanes:
                // p1: 0-7  16-23; p2: 8-15 24-31
                // two shuffles per R G B plane would be needed, so 6 total, Pre-shuffling Y U V is only 3
                __m256i temp_p1 = _mm256_permute2x128_si256(p1, p2, 0x20); // 0-7, 8-15
                __m256i temp_p2 = _mm256_permute2x128_si256(p1, p2, 0x31); // 16-23, 24-31
                p1 = temp_p1;
                p2 = temp_p2;
                */
              }
              else if constexpr (sizeof(pixel_t) == 4) {
                // 32->16 bits
                p1 = _mm256_permute4x64_epi64(p1, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6)); // 0xD8
                p2 = _mm256_permute4x64_epi64(p2, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6)); // 0xD8
              }

              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx), p1);
              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx + 32), p2);
            }
          }
          };

        process_from_float_plane_avx2(dstp[0], out0_1_f_lo, out0_1_f_hi, out0_2_f_lo, out0_2_f_hi);
        if constexpr (direction != ConversionDirection::RGB_TO_Y) {
          process_from_float_plane_avx2(dstp[1], out1_1_f_lo, out1_1_f_hi, out1_2_f_lo, out1_2_f_hi);
          process_from_float_plane_avx2(dstp[2], out2_1_f_lo, out2_1_f_hi, out2_2_f_lo, out2_2_f_hi);
        }
        // end of float_matrix_workflow
      }
      else {
        // start of integer matrix arithmetic path, both for integer and float target
        if constexpr (lessthan16bit) {
          // offset_in is added, stored as negative
          if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB || direction == ConversionDirection::RGB_TO_Y) {
            // RGB sources: same input offset for G, B and R
            in0_1 = _mm256_adds_epi16(in0_1, offset_in); in0_2 = _mm256_adds_epi16(in0_2, offset_in); // G1 G2
            in1_1 = _mm256_adds_epi16(in1_1, offset_in); in1_2 = _mm256_adds_epi16(in1_2, offset_in); // B1 B2
            in2_1 = _mm256_adds_epi16(in2_1, offset_in); in2_2 = _mm256_adds_epi16(in2_2, offset_in); // R1 R2
          }
          else {
            // YUV sources: only luma. add, because offset_in is negative
            in0_1 = _mm256_adds_epi16(in0_1, offset_in); in0_2 = _mm256_adds_epi16(in0_2, offset_in); // Y1 Y2
          }
        }
        else {
          // make unsigned to signed by flipping MSB
          if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB || direction == ConversionDirection::RGB_TO_Y) {
            // RGB sources: same pivoting offset for G, B and R
            // pivot the pixel, adjust the offset_in separately, if any, later
            in0_1 = _mm256_xor_si256(in0_1, sign_flip_mask); in0_2 = _mm256_xor_si256(in0_2, sign_flip_mask); // G1 G2
            in1_1 = _mm256_xor_si256(in1_1, sign_flip_mask); in1_2 = _mm256_xor_si256(in1_2, sign_flip_mask); // B1 B2
            in2_1 = _mm256_xor_si256(in2_1, sign_flip_mask); in2_2 = _mm256_xor_si256(in2_2, sign_flip_mask); // R1 R2
          }
          else {
            // YUV sources: only pivot luma, chroma will be pivoted and centered together in the next step
            in0_1 = _mm256_xor_si256(in0_1, sign_flip_mask); in0_2 = _mm256_xor_si256(in0_2, sign_flip_mask); // Y1 Y2
          }
        }

        // YUV source: move to signed UV
        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
          in1_1 = _mm256_sub_epi16(in1_1, half); in1_2 = _mm256_sub_epi16(in1_2, half); // U1 U2
          in2_1 = _mm256_sub_epi16(in2_1, half); in2_2 = _mm256_sub_epi16(in2_2, half); // V1 V2
        }

        // pre-unpack MADD pairs
        // These are common for all R, G, B planes
        // Pair 1: [U0 Y0 U1 Y1 ... | U8 Y8 U9 Y9 ...]
        __m256i uy1_lo = _mm256_unpacklo_epi16(in1_1, in0_1);
        __m256i uy1_hi = _mm256_unpackhi_epi16(in1_1, in0_1);
        __m256i uy2_lo = _mm256_unpacklo_epi16(in1_2, in0_2);
        __m256i uy2_hi = _mm256_unpackhi_epi16(in1_2, in0_2);

        // Pair 2: [V0 Rnd V1 Rnd ... | V8 Rnd V9 Rnd ...]
        __m256i vr1_lo = _mm256_unpacklo_epi16(in2_1, m256i_round_scale);
        __m256i vr1_hi = _mm256_unpackhi_epi16(in2_1, m256i_round_scale);
        __m256i vr2_lo = _mm256_unpacklo_epi16(in2_2, m256i_round_scale);
        __m256i vr2_hi = _mm256_unpackhi_epi16(in2_2, m256i_round_scale);

        // for 16 bit, the rgb_offset is merged into the post patch adjustment
        // 13 bit fixed point arithmetic rounder 0.5 is 4096. In general: INT_ARITH_SHIFT
        // Need1:  (m.y_b   m.u_b )     (m.y_b   m.u_b)     (m.y_b   m.u_b)     (m.y_b   m.u_b)   8x16 bit
        //         (  y3      u3  )     (  y2      u2 )     (  y1      u1 )     (   y0     u0 )   8x16 bit
        // res1=  (y_b*y3 + u_b*u3)   ...                                                         4x32 bit
        // Need2:  (m.v_b   round')     (m.y_b   round')     (m.y_b   round')     (m.y_b   round')
        //         (  v3     4096 )     (  v2     4096 )     (  v1     4096 )     (  v0     4096 )
        // res2=  (yv_b*v3 + round' )  ...  round' = round + rgb_offset

        // Processing lambda - checked and benchmarked to be inlined nicely -avoids code bloat

        // For v141_xp compatibility: forces the compiler to capture a const variable
        // that would otherwise be optimized out of nested lambda scopes.

        // integer matrix arithmetic path, followed by integer-integer shift-scaling or 32-bit float conversion.
        auto process_plane = [&](BYTE* plane_ptr, __m256i m_uy, __m256i m_vr, __m256i v_patch, auto apply_float_offset_out) {
          XP_LAMBDA_CAPTURE_FIX(limit);
          XP_LAMBDA_CAPTURE_FIX(safe_last_16float_64bytes);
          // Note: v_patch is already set to the right 32 or 64 bits width based on final_is_float and lessthan16bit
          auto madd_scale = [&](__m256i uy, __m256i vr) {
            XP_LAMBDA_CAPTURE_FIX(v_patch);
            XP_LAMBDA_CAPTURE_FIX(target_shift);
            __m256i sum = _mm256_add_epi32(_mm256_madd_epi16(m_uy, uy), _mm256_madd_epi16(m_vr, vr));
            // 16-bit adjustment (signed patch, offset_in, output rgb offset_in)

#ifdef XP_TLS
            if (!final_is_float)
#else
            if constexpr (!final_is_float)
#endif
            {
              __m256i sum_lo, sum_hi;
#ifdef XP_TLS
              if (!would_need_64bit_v_patch)
#else
              if constexpr (!would_need_64bit_v_patch)
#endif
              {
                sum = _mm256_add_epi32(sum, v_patch);
              }
              else {
                // use two 64 bit sums
                // avx2 extract low/high 128 bits, add, then insert back
                sum_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sum));
                sum_lo = _mm256_add_epi64(sum_lo, v_patch);
                sum_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sum, 1));
                sum_hi = _mm256_add_epi64(sum_hi, v_patch);
              }
              // target_shift is INT_ARITH_SHIFT - modified with bit-conversion diff, bit fixed point shift
#ifdef XP_TLS
              if (!would_need_64bit_v_patch) {
#else
              if constexpr (!would_need_64bit_v_patch) {
#endif
                sum = _mm256_srai_epi32(sum, target_shift);
              }
              else {
                // _mm256_srai_epi64 is AVX512+VL only
                sum_lo = mm256_srai_epi64_simul(sum_lo, target_shift);
                sum_hi = mm256_srai_epi64_simul(sum_hi, target_shift);
              }
#ifdef XP_TLS
              if (would_need_64bit_v_patch) {
#else
              if constexpr (would_need_64bit_v_patch) {
#endif
                // pack back to 32 bit: no _mm256_cvtepi64_epi32 in AVX2
                __m256i pack_lo = _mm256_shuffle_epi32(sum_lo, _MM_SHUFFLE(3, 2, 2, 0));
                __m256i pack_hi = _mm256_shuffle_epi32(sum_hi, _MM_SHUFFLE(3, 2, 2, 0));
                __m256i mixed = _mm256_unpacklo_epi64(pack_lo, pack_hi);
                // [0, 1, 4, 5 | 2, 3, 6, 7] => [0, 1, 2, 3 | 4, 5, 6, 7]
                sum = _mm256_permute4x64_epi64(mixed, _MM_SHUFFLE(3, 1, 2, 0));
              }
            }
            else {
              // no 64-bit needed, no overflow danger in pivot, since no U and V integer offset which is the cause of overflow.
              sum = _mm256_add_epi32(sum, v_patch);
            }
            return sum;
            };

          __m256i res1_lo = madd_scale(uy1_lo, vr1_lo); // Pixels 0-3, 8-11
          __m256i res1_hi = madd_scale(uy1_hi, vr1_hi); // Pixels 4-7, 12-15
          __m256i res2_lo = madd_scale(uy2_lo, vr2_lo); // Pixels 16-19, 24-27
          __m256i res2_hi = madd_scale(uy2_hi, vr2_hi); // Pixels 20-23, 28-31

#ifdef XP_TLS
          if (final_is_float) {
#else
          if constexpr (final_is_float) {
#endif
            // when float output is needed, convert after scaling, mimic a post-ConvertBits
            const int pix_idx = x / sizeof(pixel_t);
            float* f_dst = reinterpret_cast<float*>(plane_ptr) + pix_idx;

            // Define the blocks (8 pixels each) correctly by healing the lanes
            __m256i blockA = _mm256_permute2x128_si256(res1_lo, res1_hi, 0x20); // Pixels 0-7
            __m256i blockB, blockC, blockD;

            if constexpr (sizeof(pixel_t) == 1) {
              // 8-bit: The lanes were swapped across res1 and res2
              blockB = _mm256_permute2x128_si256(res2_lo, res2_hi, 0x20); // Pixels 8-15
              blockC = _mm256_permute2x128_si256(res1_lo, res1_hi, 0x31); // Pixels 16-23
              blockD = _mm256_permute2x128_si256(res2_lo, res2_hi, 0x31); // Pixels 24-31
            }
            else {
              // 16-bit: res1 is 0-15, res2 is 16-31
              blockB = _mm256_permute2x128_si256(res1_lo, res1_hi, 0x31); // Pixels 8-15
              blockC = _mm256_permute2x128_si256(res2_lo, res2_hi, 0x20); // Pixels 16-23
              blockD = _mm256_permute2x128_si256(res2_lo, res2_hi, 0x31); // Pixels 24-31
            }

            // Convert and Store
            auto store_block = [&](float* ptr, __m256i b, auto apply_float_offset_out) {
              if (apply_float_offset_out)
                _mm256_store_ps(ptr, _mm256_fmadd_ps(_mm256_cvtepi32_ps(b), scale_f_avx2, out_offset_f_avx2));
              else // only mul, no add, when offset_out is 0 for non Y,R,G,B channels
                _mm256_store_ps(ptr, _mm256_mul_ps(_mm256_cvtepi32_ps(b), scale_f_avx2));
              };

            store_block(f_dst, blockA, apply_float_offset_out);
            store_block(f_dst + 8, blockB, apply_float_offset_out);

            // Safety check: only store the first half of the 32-pixel block if row width allows
            if (safe_last_16float_64bytes) {
              store_block(f_dst + 16, blockC, apply_float_offset_out);
              store_block(f_dst + 24, blockD, apply_float_offset_out);
            }
          }
          else {
            const int pix_idx = x * sizeof(pixel_t_dst) / sizeof(pixel_t);
            // unlike SSE2, AVX2 has packus_epi32
            __m256i p1 = _mm256_packus_epi32(res1_lo, res1_hi); // Auto-heals lanes, no permute needed
            __m256i p2 = _mm256_packus_epi32(res2_lo, res2_hi);

            if constexpr (sizeof(pixel_t_dst) == 1) {
              // X->8 bits
              __m256i final8 = _mm256_packus_epi16(p1, p2);
              if constexpr (sizeof(pixel_t) == 2) {
                // 16->8 extra shuffle
                final8 = _mm256_permute4x64_epi64(final8, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
              }
              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx), final8);
            }
            else {
              // X->16 bits
              if constexpr (lessthan16bit_target) {
                p1 = _mm256_min_epi16(p1, limit);
                p2 = _mm256_min_epi16(p2, limit);
              }
              if constexpr (sizeof(pixel_t) == 1) {
                // 8->16 bits:
                /* If we'd not pre-shuffled the input for 8->16 bit, we would need to do this shuffle at the end to heal the lanes:
                // p1: 0-7  16-23; p2: 8-15 24-31
                // two shuffles per R G B plane would be needed, so 6 total, Pre-shuffling Y U V is only 3
                __m256i temp_p1 = _mm256_permute2x128_si256(p1, p2, 0x20); // 0-7, 8-15
                __m256i temp_p2 = _mm256_permute2x128_si256(p1, p2, 0x31); // 16-23, 24-31
                p1 = temp_p1;
                p2 = temp_p2;
                */
              }
              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx), p1);
              _mm256_store_si256(reinterpret_cast<__m256i*>(plane_ptr + pix_idx + 32), p2);
            }
          }
          };

        // Process planes, using pre-packed coefficient, and the 16 bit patch if needed
        process_plane(dstp[0], m_uy_G, m_vr_G, v_patch_G, true /* apply_float_offset_out */);
        // only Y,R,G,B needs it uniformly, so for YUV, we call it with false
        // last param: apply_float_offset_out
        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
          process_plane(dstp[1], m_uy_B, m_vr_B, v_patch_B, true);
          process_plane(dstp[2], m_uy_R, m_vr_R, v_patch_R, true);
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {
          process_plane(dstp[1], m_uy_B, m_vr_B, v_patch_B, false);
          process_plane(dstp[2], m_uy_R, m_vr_R, v_patch_R, false);
        }
        else if constexpr (direction != ConversionDirection::RGB_TO_Y) {
          // no other planes
        }
      } // if float_matrix_workflow else integer_matrix_arithmetic
    } // x
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    if constexpr (direction != ConversionDirection::RGB_TO_Y) {
      dstp[1] += dstPitch[1];
      dstp[2] += dstPitch[2];
    }
  } // y
}

#undef XP_LAMBDA_CAPTURE_FIX

// Further separating cases inside, dispatcher remains relatively simple
template<ConversionDirection direction, typename pixel_t_src, bool lessthan16bit>
void convert_yuv_to_planarrgb_avx2(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m,
  int bits_per_pixel, int bits_per_pixel_target, bool force_float)
{
  // Accuracy forever
  if (force_float || std::is_floating_point<pixel_t_src>::value) {
    // int->float conversion
    // limited/full is handled automatically through scaling and offsets
    // lessthan16bit_target is still important due to clamping
    if (bits_per_pixel_target == 8) {
      convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else if (bits_per_pixel_target < 16) {
      // lessthan16bit_target is false. Need signed pack and clamping
      convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else if (bits_per_pixel_target == 16) { // == 16
      convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else { // 32 bit float target
      // limited/full is handled automatically through scaling and offsets
      // lessthan16bit_target doesn't matter since float output has no clamping
      convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, false, float, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    return;
  }

  // this is needed to avoid instantiating the full integer conversion logic when not needed,
  // it can result static_asssert failure.
  if constexpr (!std::is_floating_point<pixel_t_src>::value) {
    const bool need_conversion = bits_per_pixel_target != bits_per_pixel;
    if (!need_conversion) {
      // no conversion, just YUV to RGB
      convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, lessthan16bit, pixel_t_src, YuvRgbConversionType::NATIVE_INT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      return;
    }

    const bool full_d = m.offset_rgb == 0;

    if (bits_per_pixel_target >= 8 && bits_per_pixel <= 16) {
      // int->int conversion with range conversion (limited<->full), or upscale with no range conversion (full->full or limited->limited)
      if (bits_per_pixel_target == 8) {
        if (full_d)
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else if (bits_per_pixel_target < 16) {
        // lessthan16bit_target is false. Need signed pack and clamping
        if (full_d)
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else if (bits_per_pixel_target == 16) { // == 16
        if (full_d)
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else { // 32 bit float target
        // int->float conversion
        // limited/full is handled automatically through scaling and offsets
        // lessthan16bit_target doesn't matter since float output has no clamping
        convert_yuv_to_planarrgb_avx2_internal<direction, pixel_t_src, lessthan16bit, false, float, YuvRgbConversionType::FLOAT_OUTPUT /*n/a*/>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
    }
  }
}

//instantiate
// YUV_TO_RGB
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint8_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint16_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint16_t, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, float, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// RGB_TO_YUV
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint8_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint16_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint16_t, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, float, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// RGB_TO_Y
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint8_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint16_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint16_t, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, float, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// YUV_TO_YUV (for future use - e.g., BT.601 → BT.709)
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_YUV, uint8_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_YUV, uint16_t, true>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_YUV, uint16_t, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_YUV, float, false>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

DISABLE_WARNING_POP
