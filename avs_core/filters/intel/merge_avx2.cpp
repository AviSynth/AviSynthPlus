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


// Avisynth filter: YUV merge / Swap planes
// by Klaus Post (kp@interact.dk)
// adapted by Richard Berg (avisynth-dev@richardberg.net)
// iSSE code by Ian Brabham


// experimental simd includes for avx2 compiled files
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
#include <x86intrin.h>
// x86intrin.h includes header files for whatever instruction
// sets are specified on the compiler command line, such as: xopintrin.h, fma4intrin.h
#else
#include <immintrin.h> // MS version of immintrin.h covers AVX, AVX2 and FMA3
#endif // __GNUC__

#if !defined(__FMA__)
// Assume that all processors that have AVX2 also have FMA3
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
// Prevent error message in g++ when using FMA intrinsics with avx2:
#pragma message "It is recommended to specify also option -mfma when using -mavx2 or higher"
#else
#define __FMA__  1
#endif
#endif
// FMA3 instruction set
#if defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))  && !defined(__INTEL_COMPILER)
#include <fmaintrin.h>
#endif // __FMA__

#include "merge_avx2.h"
#include "avs/alignment.h"
#include <stdint.h>

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

/* -----------------------------------
 *            average_plane
 * -----------------------------------
 */
#define AVX2BUG_WORKAROUND
// VS2017 15.5.1..2 optimizer generates illegal instructions for vmovntdqa
// just a note, to be removed
template<typename pixel_t>
void average_plane_avx2(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height) {
  // width is RowSize here
  int mod32_width = rowsize / 32 * 32;
  int mod16_width = rowsize / 16 * 16;

  for(int y = 0; y < height; y++) {
    for(int x = 0; x < mod32_width; x+=32) {
#ifdef AVX2BUG_WORKAROUND
      __m256i src1 = _mm256_load_si256(reinterpret_cast<__m256i*>(p1 + x));
#else
      __m256i src1  = _mm256_stream_load_si256(reinterpret_cast<__m256i*>(p1+x));
#endif
      /*
      00070	8d 40 20	 lea	 eax, DWORD PTR [eax+32]

      ; 70   :       __m256i src1  = _mm256_stream_load_si256(reinterpret_cast<__m256i*>(p1+x));

      00073	c5 fe 6f 40 e0	 vmovdqu ymm0, YMMWORD PTR [eax-32]
      00078	c4 e2 7d 2a c8	 vmovntdqa ymm1, ymm0            ****CRASH HERE! ILLEGAL INSTRUCTION VS15.5.1!!!****

      ; 71   :       __m256i src2  = _mm256_stream_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2+x)));

      0007d	c5 fe 6f 44 02 e0		 vmovdqu ymm0, YMMWORD PTR [edx+eax-32]
      00083	c4 e2 7d 2a c0	 vmovntdqa ymm0, ymm0

      */
#ifdef AVX2BUG_WORKAROUND
      __m256i src2  = _mm256_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2+x)));
      #else
      __m256i src2 = _mm256_stream_load_si256(const_cast<__m256i*>(reinterpret_cast<const __m256i*>(p2 + x)));
      #endif
      __m256i dst;
      if constexpr(sizeof(pixel_t) == 1)
        dst  = _mm256_avg_epu8(src1, src2); // 16 pixels
      else // pixel_size == 2
        dst = _mm256_avg_epu16(src1, src2); // 8 pixels

      _mm256_store_si256(reinterpret_cast<__m256i*>(p1+x), dst);
    }

    for(int x = mod32_width; x < mod16_width; x+=16) {
#ifdef AVX2BUG_WORKAROUND
      __m128i src1 = _mm_load_si128(reinterpret_cast<__m128i*>(p1 + x));
      __m128i src2 = _mm_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p2 + x)));
#else
      __m128i src1  = _mm_stream_load_si128(reinterpret_cast<__m128i*>(p1+x));
      __m128i src2  = _mm_stream_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p2+x)));
#endif
      __m128i dst;
      if constexpr(sizeof(pixel_t) == 1)
        dst  = _mm_avg_epu8(src1, src2); // 8 pixels
      else // pixel_size == 2
        dst = _mm_avg_epu16(src1, src2); // 4 pixels

      _mm_store_si128(reinterpret_cast<__m128i*>(p1+x), dst);
    }

    if (mod16_width != rowsize) {
      for (size_t x = mod16_width / sizeof(pixel_t); x < rowsize/sizeof(pixel_t); ++x) {
        reinterpret_cast<pixel_t *>(p1)[x] = (int(reinterpret_cast<pixel_t *>(p1)[x]) + reinterpret_cast<const pixel_t *>(p2)[x] + 1) >> 1;
      }
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// instantiate to let them access from other modules
template void average_plane_avx2<uint8_t>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height);
template void average_plane_avx2<uint16_t>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height);


/* -----------------------------------
*       weighted_merge_planar
* -----------------------------------
*/
template<bool lessthan16bit>
void weighted_merge_planar_uint16_avx2(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i) {
  AVS_UNUSED(weight_f);
  __m128i mask_128 = _mm_set1_epi32( (weight_i << 16) + invweight_i);
  __m256i mask = _mm256_set1_epi32((weight_i << 16) + invweight_i);

  const __m128i signed_shifter_128 = _mm_set1_epi16(-32768);
  const __m256i signed_shifter = _mm256_set1_epi16(-32768);

  auto round_mask = _mm256_set1_epi32(0x4000);

  auto round_mask_128 = _mm_set1_epi32(0x4000);

  int wMod32 = (rowsize / 32) * 32;
  int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod32; x += 32) {
      __m256i px1, px2;
      px1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p1 + x)); // y7 y6 y5 y4 y3 y2 y1 y0
      px2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p2 + x)); // Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0

      if (!lessthan16bit) {
        px1 = _mm256_add_epi16(px1, signed_shifter);
        px2 = _mm256_add_epi16(px2, signed_shifter);
      }

      auto p03 = _mm256_unpacklo_epi16(px1, px2); // Y11y11 Y10y10 Y9y9 Y8y8 Y3y3 Y2y2 Y1y1 Y0y0
      auto p47 = _mm256_unpackhi_epi16(px1, px2); // Yy15-12 Yy7-4

      p03 = _mm256_madd_epi16(p03, mask); // px1 * invweight + px2 * weight
      p47 = _mm256_madd_epi16(p47, mask);

      p03 = _mm256_add_epi32(p03, round_mask);
      p47 = _mm256_add_epi32(p47, round_mask);

      p03 = _mm256_srai_epi32(p03, 15);
      p47 = _mm256_srai_epi32(p47, 15);

      auto p07 = _mm256_packs_epi32(p03, p47);
      if (!lessthan16bit) {
        p07 = _mm256_add_epi16(p07, signed_shifter);
      }

      auto result = p07;
      _mm256_store_si256(reinterpret_cast<__m256i*>(p1 + x), result);
    }

    for (int x = wMod32; x < wMod16; x += 16) {
      __m128i px1, px2;
      px1 = _mm_load_si128(reinterpret_cast<__m128i*>(p1 + x)); // y7 y6 y5 y4 y3 y2 y1 y0
      px2 = _mm_load_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(p2 + x))); // Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
      if (!lessthan16bit) {
        px1 = _mm_add_epi16(px1, signed_shifter_128);
        px2 = _mm_add_epi16(px2, signed_shifter_128);
      }

      auto p03 = _mm_unpacklo_epi16(px1, px2); // Y11y11 Y10y10 Y9y9 Y8y8 Y3y3 Y2y2 Y1y1 Y0y0
      auto p47 = _mm_unpackhi_epi16(px1, px2); // Yy15-12 Yy7-4

      p03 = _mm_madd_epi16(p03, mask_128); // px1 * invweight + px2 * weight
      p47 = _mm_madd_epi16(p47, mask_128);

      p03 = _mm_add_epi32(p03, round_mask_128);
      p47 = _mm_add_epi32(p47, round_mask_128);

      p03 = _mm_srai_epi32(p03, 15);
      p47 = _mm_srai_epi32(p47, 15);

      auto p07 = _mm_packs_epi32(p03, p47);
      if (!lessthan16bit) {
        p07 = _mm_add_epi16(p07, signed_shifter_128);
      }

      auto result = p07;
      _mm_stream_si128(reinterpret_cast<__m128i*>(p1 + x), result);
    }

    for (size_t x = wMod16 / sizeof(uint16_t); x < rowsize / sizeof(uint16_t); x++) {
      reinterpret_cast<uint16_t *>(p1)[x] = (reinterpret_cast<uint16_t *>(p1)[x] * invweight_i + reinterpret_cast<const uint16_t *>(p2)[x] * weight_i + 16384) >> 15;
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// instantiate to let them access from other modules
template void weighted_merge_planar_uint16_avx2<false>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);
template void weighted_merge_planar_uint16_avx2<true>(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);


void weighted_merge_planar_avx2(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i) {
  AVS_UNUSED(weight_f);
  auto round_mask = _mm256_set1_epi32(0x4000);
  auto zero = _mm256_setzero_si256();
  auto mask = _mm256_set_epi16(weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i);

  auto round_mask_128 = _mm_set1_epi32(0x4000);
  auto zero_128 = _mm_setzero_si128();
  auto mask_128 = _mm_set_epi16(weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i, weight_i, invweight_i);

  int wMod32 = (rowsize / 32) * 32;
  int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod32; x += 32) {
      auto px1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p1 + x)); // y15y14 ... y7y6 y5y4 y3y2 y1y0
      auto px2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(p2 + x)); // Y15Y14 ... Y7Y6 Y5Y4 Y3Y2 Y1Y0

      auto p07 = _mm256_unpacklo_epi8(px1, px2); // Y7y7 ..  Y3y3   Y2y2  Y1y1 Y0y0
      auto p815 = _mm256_unpackhi_epi8(px1, px2); //Y15y15 ..Y11y11 Y10y10 Y9y9 Y8y8

      auto p03 = _mm256_unpacklo_epi8(p07, zero);  //00Y3 00y3 00Y2 00y2 00Y1 00y1 00Y0 00y0 8*short
      auto p47 = _mm256_unpackhi_epi8(p07, zero);
      auto p811 = _mm256_unpacklo_epi8(p815, zero);
      auto p1215 = _mm256_unpackhi_epi8(p815, zero);

      p03 = _mm256_madd_epi16(p03, mask);
      p47 = _mm256_madd_epi16(p47, mask);
      p811 = _mm256_madd_epi16(p811, mask);
      p1215 = _mm256_madd_epi16(p1215, mask);

      p03 = _mm256_add_epi32(p03, round_mask);
      p47 = _mm256_add_epi32(p47, round_mask);
      p811 = _mm256_add_epi32(p811, round_mask);
      p1215 = _mm256_add_epi32(p1215, round_mask);

      p03 = _mm256_srli_epi32(p03, 15);
      p47 = _mm256_srli_epi32(p47, 15);
      p811 = _mm256_srli_epi32(p811, 15);
      p1215 = _mm256_srli_epi32(p1215, 15);

      p07 = _mm256_packs_epi32(p03, p47);
      p815 = _mm256_packs_epi32(p811, p1215);

      __m256i result = _mm256_packus_epi16(p07, p815);

      _mm256_store_si256(reinterpret_cast<__m256i*>(p1 + x), result);
    }

    for (int x = wMod32; x < wMod16; x += 16) {
      __m128i px1 = _mm_load_si128(reinterpret_cast<const __m128i*>(p1 + x)); // y15y14 ... y7y6 y5y4 y3y2 y1y0
      __m128i px2 = _mm_load_si128(reinterpret_cast<const __m128i*>(p2 + x)); // Y15Y14 ... Y7Y6 Y5Y4 Y3Y2 Y1Y0

      __m128i p07 = _mm_unpacklo_epi8(px1, px2); // Y7y7 ..  Y3y3   Y2y2  Y1y1 Y0y0
      __m128i p815 = _mm_unpackhi_epi8(px1, px2); //Y15y15 ..Y11y11 Y10y10 Y9y9 Y8y8

      __m128i p03 = _mm_unpacklo_epi8(p07, zero_128);  //00Y3 00y3 00Y2 00y2 00Y1 00y1 00Y0 00y0 8*short
      __m128i p47 = _mm_unpackhi_epi8(p07, zero_128);
      __m128i p811 = _mm_unpacklo_epi8(p815, zero_128);
      __m128i p1215 = _mm_unpackhi_epi8(p815, zero_128);

      p03 = _mm_madd_epi16(p03, mask_128);
      p47 = _mm_madd_epi16(p47, mask_128);
      p811 = _mm_madd_epi16(p811, mask_128);
      p1215 = _mm_madd_epi16(p1215, mask_128);

      p03 = _mm_add_epi32(p03, round_mask_128);
      p47 = _mm_add_epi32(p47, round_mask_128);
      p811 = _mm_add_epi32(p811, round_mask_128);
      p1215 = _mm_add_epi32(p1215, round_mask_128);

      p03 = _mm_srli_epi32(p03, 15);
      p47 = _mm_srli_epi32(p47, 15);
      p811 = _mm_srli_epi32(p811, 15);
      p1215 = _mm_srli_epi32(p1215, 15);

      p07 = _mm_packs_epi32(p03, p47);
      p815 = _mm_packs_epi32(p811, p1215);

      __m128i result = _mm_packus_epi16(p07, p815);

      _mm_store_si128(reinterpret_cast<__m128i*>(p1 + x), result);
    }

    for (size_t x = wMod16 / sizeof(uint8_t); x < rowsize / sizeof(uint8_t); x++) {
      reinterpret_cast<uint8_t *>(p1)[x] = (reinterpret_cast<uint8_t *>(p1)[x] * invweight_i + reinterpret_cast<const uint8_t *>(p2)[x] * weight_i + 16384) >> 15;
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}
