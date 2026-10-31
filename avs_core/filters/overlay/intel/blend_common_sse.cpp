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

// Overlay (c) 2003, 2004 by Klaus Post

#include <avisynth.h>

#include "blend_common_sse.h"
#include "../blend_common.h"
#include "../../../core/internal.h"

// Intrinsics base header + really required extension headers
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else 
#include <x86intrin.h> // GCC/MinGW/Clang/LLVM
#endif
#include <smmintrin.h> // SSE4.1

#include <stdint.h>
#include <type_traits>
#include <vector>

/********************************
 ********* Blend Opaque *********
 ** Use for Lighten and Darken **
 ********************************/

#ifdef X86_32
AVS_FORCEINLINE __m64 overlay_blend_opaque_mmx_core(const __m64& p1, const __m64& p2, const __m64& mask) {
  // return (mask) ? p2 : p1;
  __m64 r1 = _mm_andnot_si64(mask, p1);
  __m64 r2 = _mm_and_si64   (mask, p2);
  return _mm_or_si64(r1, r2);
}
#endif

AVS_FORCEINLINE __m128i overlay_blend_opaque_sse2_core(const __m128i& p1, const __m128i& p2, const __m128i& mask) {
  // return (mask) ? p2 : p1;
  __m128i r1 = _mm_andnot_si128(mask, p1);
  __m128i r2 = _mm_and_si128   (mask, p2);
  return _mm_or_si128(r1, r2);
}

// simd_magic_div_32, blend8_masked_sse41_row, blend16_masked_sse41_row, masked_merge_sse41_impl
// (implementation-include, no guards — each TU gets its own compiled copy)
#include "masked_merge_sse41_impl.hpp"

void masked_merge_float_sse2(BYTE* p1, const BYTE* p2, const BYTE* mask,
  int p1_pitch, int p2_pitch, int mask_pitch,
  int width, int height, float opacity_f)
{
  const int realwidth = width * sizeof(float);
  const int wMod16 = (realwidth / 16) * 16;
  const __m128 opacity_v = _mm_set1_ps(opacity_f);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      const __m128 p1_f   = _mm_loadu_ps(reinterpret_cast<const float*>(p1 + x));
      const __m128 p2_f   = _mm_loadu_ps(reinterpret_cast<const float*>(p2 + x));
      const __m128 mask_f = _mm_mul_ps(_mm_loadu_ps(reinterpret_cast<const float*>(mask + x)), opacity_v);
      // p1*(1-m) + p2*m  =  p1 + (p2-p1)*m
      _mm_storeu_ps(reinterpret_cast<float*>(p1 + x),
        _mm_add_ps(p1_f, _mm_mul_ps(_mm_sub_ps(p2_f, p1_f), mask_f)));
    }
    for (int x = wMod16 / (int)sizeof(float); x < width; x++) {
      const float m  = reinterpret_cast<const float*>(mask)[x] * opacity_f;
      const float a  = reinterpret_cast<float*>(p1)[x];
      const float b  = reinterpret_cast<const float*>(p2)[x];
      reinterpret_cast<float*>(p1)[x] = a + (b - a) * m;
    }
    p1   += p1_pitch;
    p2   += p2_pitch;
    mask += mask_pitch;
  }
}


/***************************************
 ********* Mode: Lighten/Darken ********
 ***************************************/

typedef __m128i (OverlaySseBlendOpaque)(const __m128i&, const __m128i&, const __m128i&);
typedef __m128i (OverlaySseCompare)(const __m128i&, const __m128i&, const __m128i&);
#ifdef X86_32
typedef   __m64 (OverlayMmxCompare)(const __m64&, const __m64&, const __m64&);
#endif

typedef int (OverlayCCompare)(BYTE, BYTE);


#ifdef X86_32
template<OverlayMmxCompare compare, OverlayCCompare compare_c>
AVS_FORCEINLINE void overlay_darklighten_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  __m64 zero = _mm_setzero_si64();

  int wMod8 = (width/8) * 8;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod8; x+=8) {
      // Load Y Plane
      __m64 p1_y = *(reinterpret_cast<const __m64*>(p1Y+x));
      __m64 p2_y = *(reinterpret_cast<const __m64*>(p2Y+x));

      // Compare
      __m64 cmp_result = compare(p1_y, p2_y, zero);

      // Process U Plane
      __m64 result_y = overlay_blend_opaque_mmx_core(p1_y, p2_y, cmp_result);
      *reinterpret_cast<__m64*>(p1Y+x) = result_y;

      // Process U plane
      __m64 p1_u = *(reinterpret_cast<const __m64*>(p1U+x));
      __m64 p2_u = *(reinterpret_cast<const __m64*>(p2U+x));

      __m64 result_u = overlay_blend_opaque_mmx_core(p1_u, p2_u, cmp_result);
      *reinterpret_cast<__m64*>(p1U+x) = result_u;

      // Process V plane
      __m64 p1_v = *(reinterpret_cast<const __m64*>(p1V+x));
      __m64 p2_v = *(reinterpret_cast<const __m64*>(p2V+x));

      __m64 result_v = overlay_blend_opaque_mmx_core(p1_v, p2_v, cmp_result);
      *reinterpret_cast<__m64*>(p1V+x) = result_v;
    }

    // Leftover value
    for (int x = wMod8; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core<uint8_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<uint8_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<uint8_t>(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }

  _mm_empty();
}
#endif

template <OverlaySseCompare compare, OverlayCCompare compare_c>
void overlay_darklighten_sse2(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  __m128i zero = _mm_setzero_si128();

  int wMod16 = (width/16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x+=16) {
      // Load Y Plane
      __m128i p1_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1Y+x));
      __m128i p2_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2Y+x));

      // Compare
      __m128i cmp_result = compare(p1_y, p2_y, zero);

      // Process U Plane
      __m128i result_y = overlay_blend_opaque_sse2_core(p1_y, p2_y, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1Y+x), result_y);

      // Process U plane
      __m128i p1_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1U+x));
      __m128i p2_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2U+x));

      __m128i result_u = overlay_blend_opaque_sse2_core(p1_u, p2_u, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1U+x), result_u);

      // Process V plane
      __m128i p1_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1V+x));
      __m128i p2_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2V+x));

      __m128i result_v = overlay_blend_opaque_sse2_core(p1_v, p2_v, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1V+x), result_v);
    }

    // Leftover value
    for (int x = wMod16; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core<uint8_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<uint8_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<uint8_t>(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}

template <OverlaySseCompare compare, OverlayCCompare compare_c>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void overlay_darklighten_sse41(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height)
{
  __m128i zero = _mm_setzero_si128();

  int wMod16 = (width / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      // Load Y Plane
      __m128i p1_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1Y + x));
      __m128i p2_y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2Y + x));

      // Compare
      __m128i cmp_result = compare(p1_y, p2_y, zero);

      // Process Y Plane
      __m128i result_y = _mm_blendv_epi8(p1_y, p2_y, cmp_result); // SSE4.1
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1Y + x), result_y);

      // Process U plane
      __m128i p1_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1U + x));
      __m128i p2_u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2U + x));

      __m128i result_u = _mm_blendv_epi8(p1_u, p2_u, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1U + x), result_u);

      // Process V plane
      __m128i p1_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1V + x));
      __m128i p2_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2V + x));

      __m128i result_v = _mm_blendv_epi8(p1_v, p2_v, cmp_result);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1V + x), result_v);
    }

    // Leftover value
    for (int x = wMod16; x < width; x++) {
      int mask = compare_c(p1Y[x], p2Y[x]);
      p1Y[x] = overlay_blend_opaque_c_core<uint8_t>(p1Y[x], p2Y[x], mask);
      p1U[x] = overlay_blend_opaque_c_core<uint8_t>(p1U[x], p2U[x], mask);
      p1V[x] = overlay_blend_opaque_c_core<uint8_t>(p1V[x], p2V[x], mask);
    }

    p1Y += p1_pitch;
    p1U += p1_pitch;
    p1V += p1_pitch;

    p2Y += p2_pitch;
    p2U += p2_pitch;
    p2V += p2_pitch;
  }
}

// Compare functions for lighten and darken mode
AVS_FORCEINLINE static int overlay_darken_c_cmp(BYTE p1, BYTE p2) {
  return p2 <= p1;
}

#ifdef X86_32
AVS_FORCEINLINE __m64 overlay_darken_mmx_cmp(const __m64& p1, const __m64& p2, const __m64& zero) {
  __m64 diff = _mm_subs_pu8(p2, p1);
  return _mm_cmpeq_pi8(diff, zero);
}
#endif

AVS_FORCEINLINE __m128i overlay_darken_sse_cmp(const __m128i& p1, const __m128i& p2, const __m128i& zero) {
  __m128i diff = _mm_subs_epu8(p2, p1);
  return _mm_cmpeq_epi8(diff, zero);
}

template<typename pixel_t>
AVS_FORCEINLINE int overlay_lighten_c_cmp(pixel_t p1, pixel_t p2) {
  return p2 >= p1;
}

#ifdef X86_32
AVS_FORCEINLINE __m64 overlay_lighten_mmx_cmp(const __m64& p1, const __m64& p2, const __m64& zero) {
  __m64 diff = _mm_subs_pu8(p1, p2);
  return _mm_cmpeq_pi8(diff, zero);
}
#endif

AVS_FORCEINLINE __m128i overlay_lighten_sse_cmp(const __m128i& p1, const __m128i& p2, const __m128i& zero) {
  __m128i diff = _mm_subs_epu8(p1, p2);
  return _mm_cmpeq_epi8(diff, zero);
}

#ifdef X86_32
void overlay_darken_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_mmx<overlay_darken_mmx_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_mmx(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_mmx<overlay_lighten_mmx_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
#endif

void overlay_darken_sse2(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse2<overlay_darken_sse_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_sse2(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse2<overlay_lighten_sse_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}

void overlay_darken_sse41(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse41<overlay_darken_sse_cmp, overlay_darken_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}
void overlay_lighten_sse41(BYTE *p1Y, BYTE *p1U, BYTE *p1V, const BYTE *p2Y, const BYTE *p2U, const BYTE *p2V, int p1_pitch, int p2_pitch, int width, int height) {
  overlay_darklighten_sse41<overlay_lighten_sse_cmp, overlay_lighten_c_cmp>(p1Y, p1U, p1V, p2Y, p2U, p2V, p1_pitch, p2_pitch, width, height);
}


// ---------------------------------------------------------------------------
// Family 1: weighted_merge — SSE2 (no mask, flat weight, >> 15 shift)
// weight + invweight == 32768; boundary values (0, 32768) are caller early-outs.
// All intrinsics here are SSE2-level; no GCC target attribute needed.
// ---------------------------------------------------------------------------

static void weighted_merge_uint8_sse2_impl(
  BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int rowsize, int height, int weight_i, int invweight_i)
{
  const __m128i mask       = _mm_set1_epi32(invweight_i | (weight_i << 16));
  const __m128i round_mask = _mm_set1_epi32(0x4000);
  const __m128i zero       = _mm_setzero_si128();

  const int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      __m128i px1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + x));
      __m128i px2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + x));

      __m128i p07   = _mm_unpacklo_epi8(px1, px2);
      __m128i p815  = _mm_unpackhi_epi8(px1, px2);

      __m128i p03   = _mm_unpacklo_epi8(p07, zero);
      __m128i p47   = _mm_unpackhi_epi8(p07, zero);
      __m128i p811  = _mm_unpacklo_epi8(p815, zero);
      __m128i p1215 = _mm_unpackhi_epi8(p815, zero);

      p03   = _mm_add_epi32(_mm_madd_epi16(p03,   mask), round_mask);
      p47   = _mm_add_epi32(_mm_madd_epi16(p47,   mask), round_mask);
      p811  = _mm_add_epi32(_mm_madd_epi16(p811,  mask), round_mask);
      p1215 = _mm_add_epi32(_mm_madd_epi16(p1215, mask), round_mask);

      p03   = _mm_srli_epi32(p03,   15);
      p47   = _mm_srli_epi32(p47,   15);
      p811  = _mm_srli_epi32(p811,  15);
      p1215 = _mm_srli_epi32(p1215, 15);

      p07  = _mm_packs_epi32(p03,  p47);
      p815 = _mm_packs_epi32(p811, p1215);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + x),
        _mm_packus_epi16(p07, p815));
    }

    // Scalar tail
    for (int x = wMod16; x < rowsize; ++x)
      p1[x] = (uint8_t)((p1[x] * invweight_i + p2[x] * weight_i + 16384) >> 15);

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// lessthan16bit=true:  10/12/14-bit — values fit positive int16, no pivot needed
// lessthan16bit=false: full 16-bit  — signed pivot for unsigned-to-signed madd safety
template<bool lessthan16bit>
static void weighted_merge_uint16_sse2_impl(
  BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int rowsize, int height, int weight_i, int invweight_i)
{
  const __m128i mask         = _mm_set1_epi32((weight_i << 16) + invweight_i);
  const __m128i round_mask   = _mm_set1_epi32(0x4000);
  const __m128i signed_shift = _mm_set1_epi16(-32768);

  const int wMod16 = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      __m128i px1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + x));
      __m128i px2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p2 + x));

      if constexpr (!lessthan16bit) {
        px1 = _mm_add_epi16(px1, signed_shift);
        px2 = _mm_add_epi16(px2, signed_shift);
      }

      __m128i p03 = _mm_unpacklo_epi16(px1, px2);
      __m128i p47 = _mm_unpackhi_epi16(px1, px2);

      p03 = _mm_add_epi32(_mm_madd_epi16(p03, mask), round_mask);
      p47 = _mm_add_epi32(_mm_madd_epi16(p47, mask), round_mask);

      p03 = _mm_srai_epi32(p03, 15);
      p47 = _mm_srai_epi32(p47, 15);

      __m128i p07 = _mm_packs_epi32(p03, p47);
      if constexpr (!lessthan16bit) {
        p07 = _mm_add_epi16(p07, signed_shift);
      }

      _mm_storeu_si128(reinterpret_cast<__m128i*>(p1 + x), p07);
    }

    // Scalar tail
    for (int x = wMod16 / 2; x < rowsize / 2; ++x) {
      reinterpret_cast<uint16_t*>(p1)[x] = (uint16_t)(
        (reinterpret_cast<uint16_t*>(p1)[x] * invweight_i +
         reinterpret_cast<const uint16_t*>(p2)[x] * weight_i + 16384) >> 15);
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

void weighted_merge_sse2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, int weight, int invweight, int bits_per_pixel)
{
  const int pixelsize = bits_per_pixel <= 8 ? 1 : 2;
  const int rowsize   = width * pixelsize;

  if (bits_per_pixel == 8)
    weighted_merge_uint8_sse2_impl(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
  else if (bits_per_pixel == 16)
    weighted_merge_uint16_sse2_impl<false>(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
  else  // 10, 12, 14-bit
    weighted_merge_uint16_sse2_impl<true>(p1, p2, p1_pitch, p2_pitch, rowsize, height, weight, invweight);
}

void weighted_merge_float_sse2(BYTE* p1, const BYTE* p2, int p1_pitch, int p2_pitch,
  int width, int height, float weight_f)
{
  const float invweight_f = 1.0f - weight_f;
  const auto  v_weight    = _mm_set1_ps(weight_f);
  const auto  v_invweight = _mm_set1_ps(invweight_f);
  const int   rowsize     = width * 4;
  const int   wMod16      = (rowsize / 16) * 16;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += 16) {
      auto px1 = _mm_loadu_ps(reinterpret_cast<const float*>(p1 + x));
      auto px2 = _mm_loadu_ps(reinterpret_cast<const float*>(p2 + x));
      // p1 + (p2 - p1) * w  ==  p1*(1-w) + p2*w
      // choose the latter to match C ref
      auto res = _mm_add_ps(_mm_mul_ps(px1, v_invweight), _mm_mul_ps(px2, v_weight));
      // old linear interpolation: _mm_add_ps(px1, _mm_mul_ps(_mm_sub_ps(px2, px1), v_weight))
      _mm_storeu_ps(reinterpret_cast<float*>(p1 + x), res);
    }
    // Scalar tail
    for (int x = wMod16 / 4; x < width; ++x) {
      reinterpret_cast<float*>(p1)[x] =
        reinterpret_cast<float*>(p1)[x] * invweight_f +
        reinterpret_cast<const float*>(p2)[x] * weight_f;
    }
    p1 += p1_pitch;
    p2 += p2_pitch;
  }
}

// ---------------------------------------------------------------------------
// Overlay blend masked getter — returns masked_merge_sse41_impl instantiation.
// is_chroma=false → always MASK444 (luma).
// is_chroma=true  → placement-aware maskMode (chroma).
// No target("sse4.1") needed: getter only returns a function pointer, no intrinsics.
// ---------------------------------------------------------------------------
masked_merge_fn_t* get_overlay_blend_masked_fn_sse41(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_SSE41(MaskType) \
  return is_chroma ? masked_merge_sse41_impl<MaskType> \
                   : masked_merge_sse41_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_SSE41(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_SSE41(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_SSE41(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_SSE41(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_SSE41(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_SSE41(MASK422_MPEG2)
  case MASK422_TOPLEFT:  DISPATCH_OVERLAY_BLEND_SSE41(MASK422_TOPLEFT)
  case MASK411:          DISPATCH_OVERLAY_BLEND_SSE41(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_SSE41(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_SSE41(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_SSE41(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_SSE41(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_SSE41(MASK410_TOPLEFT)
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_SSE41
  return masked_merge_sse41_impl<MASK444>; // unreachable
}

// Layer SSE4.1 dispatcher has moved to filters/intel/layer_sse41.cpp

// ---------------------------------------------------------------------------
// Per-row chroma mask preparation for the scratch path in OF_blend.cpp.
// Dispatches on MaskMode at runtime; full_opacity known at compile time.
// Defined here so masked_rowprep_sse41.hpp is only included in this TU.
// ---------------------------------------------------------------------------
template<typename pixel_t, bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void do_fill_chroma_row_sse41(
  std::vector<pixel_t>& buf, const pixel_t* luma_row,
  int luma_pitch_pixels, int chroma_w, MaskMode mode,
  int opacity_i, int half, MagicDiv magic)
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row_sse41<MASK411, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420:
    prepare_effective_mask_for_row_sse41<MASK420, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row_sse41<MASK420_MPEG2, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row_sse41<MASK420_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422:
    prepare_effective_mask_for_row_sse41<MASK422, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row_sse41<MASK422_MPEG2, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK422_TOPLEFT:
    prepare_effective_mask_for_row_sse41<MASK422_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK440:
    prepare_effective_mask_for_row_sse41<MASK440, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK410:
    prepare_effective_mask_for_row_sse41<MASK410, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row_sse41<MASK411_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row_sse41<MASK440_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row_sse41<MASK410_TOPLEFT, pixel_t, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity_i, half, magic); break;
  default: break;
  }
}

// Explicit instantiation definitions for the four combinations used by OF_blend.cpp.
template void do_fill_chroma_row_sse41<uint8_t,  true> (std::vector<uint8_t>&,  const uint8_t*,  int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_sse41<uint8_t,  false>(std::vector<uint8_t>&,  const uint8_t*,  int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_sse41<uint16_t, true> (std::vector<uint16_t>&, const uint16_t*, int, int, MaskMode, int, int, MagicDiv);
template void do_fill_chroma_row_sse41<uint16_t, false>(std::vector<uint16_t>&, const uint16_t*, int, int, MaskMode, int, int, MagicDiv);

template<bool full_opacity>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void do_fill_chroma_row_float_sse41(
  std::vector<float>& buf, const float* luma_row,
  int luma_pitch_pixels, int chroma_w, MaskMode mode,
  float opacity)
{
  switch (mode) {
  case MASK411:
    prepare_effective_mask_for_row_float_sse41<MASK411, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK420:
    prepare_effective_mask_for_row_float_sse41<MASK420, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK420_MPEG2:
    prepare_effective_mask_for_row_float_sse41<MASK420_MPEG2, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK420_TOPLEFT:
    prepare_effective_mask_for_row_float_sse41<MASK420_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK422:
    prepare_effective_mask_for_row_float_sse41<MASK422, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK422_MPEG2:
    prepare_effective_mask_for_row_float_sse41<MASK422_MPEG2, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK422_TOPLEFT:
    prepare_effective_mask_for_row_float_sse41<MASK422_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK440:
    prepare_effective_mask_for_row_float_sse41<MASK440, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK410:
    prepare_effective_mask_for_row_float_sse41<MASK410, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK411_TOPLEFT:
    prepare_effective_mask_for_row_float_sse41<MASK411_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK440_TOPLEFT:
    prepare_effective_mask_for_row_float_sse41<MASK440_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  case MASK410_TOPLEFT:
    prepare_effective_mask_for_row_float_sse41<MASK410_TOPLEFT, full_opacity>(luma_row, luma_pitch_pixels, chroma_w, buf, opacity); break;
  default: break;
  }
}


template void do_fill_chroma_row_float_sse41<true>(std::vector<float>&, const float*, int, int, MaskMode, float);
template void do_fill_chroma_row_float_sse41<false>(std::vector<float>&, const float*, int, int, MaskMode, float);

// and for float:
masked_merge_float_fn_t* get_overlay_blend_masked_float_fn_sse41(bool is_chroma, MaskMode maskMode)
{
#define DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MaskType) \
  return is_chroma ? masked_merge_float_sse41_impl<MaskType> \
                   : masked_merge_float_sse41_impl<MASK444>;

  switch (maskMode) {
  case MASK444:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK444)
  case MASK420:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK420)
  case MASK420_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK420_MPEG2)
  case MASK420_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK420_TOPLEFT)
  case MASK422:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK422)
  case MASK422_MPEG2:    DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK422_MPEG2)
  case MASK422_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK422_TOPLEFT)
  case MASK411:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK411)
  case MASK411_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK411_TOPLEFT)
  case MASK440:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK440)
  case MASK440_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK440_TOPLEFT)
  case MASK410:          DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK410)
  case MASK410_TOPLEFT:  DISPATCH_OVERLAY_BLEND_FLOAT_SSE41(MASK410_TOPLEFT)
  case MASK_MODE_COUNT: break;
  }
#undef DISPATCH_OVERLAY_BLEND_FLOAT_SSE41
  return masked_merge_float_sse41_impl<MASK444>; // unreachable
}


