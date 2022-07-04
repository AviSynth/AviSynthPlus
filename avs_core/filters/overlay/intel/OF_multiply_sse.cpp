// Avisynth+
// https://avs-plus.net
//
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

#include "avisynth.h"
#include "OF_multiply_sse.h"

#include <stdint.h>
#include <type_traits>

#include "emmintrin.h"
#include "smmintrin.h"

template<typename pixel_t>
static AVS_FORCEINLINE __m128 Fourpixels_to_floats(const pixel_t* src) {
  auto zero = _mm_setzero_si128();
  __m128i srci;
  if constexpr (sizeof(pixel_t) == 1) {
    srci = _mm_cvtsi32_si128(*(uint32_t*)src);
    srci = _mm_unpacklo_epi8(srci, zero);
    srci = _mm_unpacklo_epi16(srci, zero);
  }
  else {
    srci = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
    srci = _mm_unpacklo_epi16(srci, zero);
  }
  return _mm_cvtepi32_ps(srci);
}

template<typename pixel_t>
static AVS_FORCEINLINE __m128 FourChromapixels_to_floats(const pixel_t* src, const __m128i half) {
  auto zero = _mm_setzero_si128();
  __m128i srci;
  if constexpr (sizeof(pixel_t) == 1) {
    srci = _mm_cvtsi32_si128(*(uint32_t*)src);
    srci = _mm_unpacklo_epi8(srci, zero);
    srci = _mm_unpacklo_epi16(srci, zero);
  }
  else {
    srci = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
    srci = _mm_unpacklo_epi16(srci, zero);
  }
  srci = _mm_sub_epi32(srci, half);
  return _mm_cvtepi32_ps(srci);
}

#if 0
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
// fake half _mm_packus_epi32 (orig is SSE4.1 only)
// only four 32 bit numbers to to lower 4 16bits
static AVS_FORCEINLINE __m128i _MM_PACKUS_4_EPI32(__m128i a)
{
  const __m128i val_32 = _mm_set1_epi32(0x8000); // make signed
  const __m128i val_16 = _mm_set1_epi16(0x8000);

  a = _mm_sub_epi32(a, val_32);
  a = _mm_packs_epi32(a, a);
  a = _mm_add_epi16(a, val_16); // back to unsigned
  return a;
}
#endif

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void Store_Fourpixels(pixel_t* dst, __m128 what, const __m128 rounder) {
  what = _mm_add_ps(what, rounder); // round
  __m128i si32 = _mm_cvttps_epi32(what); // truncate
  //__m128i result_16 = _MM_PACKUS_4_EPI32(si32); // only 4 pixels
  __m128i result_16 = _mm_packus_epi32(si32, si32); // sse4.1
  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result_8 = _mm_packus_epi16(result_16, result_16);
    *(uint32_t*)(dst) = _mm_cvtsi128_si32(result_8);
  }
  else {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), result_16);
  }
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE void Store_FourChromapixels(pixel_t* dst, __m128 what, const __m128 half_plus_rounder) {
  what = _mm_add_ps(what, half_plus_rounder); // chroma offset back with rounder
  __m128i si32 = _mm_cvttps_epi32(what); // truncate
  //__m128i result_16 = _MM_PACKUS_4_EPI32(si32); // only 4 pixels, SSE2
  __m128i result_16 = _mm_packus_epi32(si32, si32); // sse4.1
  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result_8 = _mm_packus_epi16(result_16, result_16);
    *(uint32_t*)(dst) = _mm_cvtsi128_si32(result_8);
  }
  else {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), result_16);
  }
}


template<typename pixel_t, bool opacity_is_full, bool has_mask>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41(
  int bits_per_pixel,
  const float opacity_f,
  const int opacity,
  int width, int height,
  const pixel_t* ovY,
  int overlaypitch,
  pixel_t* baseY, pixel_t* baseU, pixel_t* baseV,
  int basepitch,
  const pixel_t* maskY, const pixel_t* maskU, const pixel_t* maskV,
  int maskpitch
)
{
  const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : (1 << bits_per_pixel) - 1;
  const float factor = 1.0f / max_pixel_value;
  const int half_i = sizeof(pixel_t) == 1 ? 128 : 1 << (bits_per_pixel - 1);
  const float half_f = (float)half_i;

  float factor_mul_opacity;
  if constexpr (opacity_is_full)
    factor_mul_opacity = factor * 1.0f;
  else
    factor_mul_opacity = factor * opacity_f;

  auto opacity_simd = _mm_set1_ps(opacity_f);
  auto factor_simd = _mm_set1_ps(factor);
  auto factor_mul_opacity_simd = _mm_set1_ps(factor_mul_opacity);
  auto one_ps_simd = _mm_set1_ps(1.0f);
  const __m128i half_i_simd = _mm_set1_epi32(half_i);
  const __m128 half_and_rounder_simd = _mm_set1_ps(half_f + 0.5f);
  const __m128 rounder_simd = _mm_set1_ps(0.5f);

  const int wMod4 = width / 4 * 4;

  // start processing
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod4; x += 4) {
      __m128 Y, U, V;
      // 4 pixels at a time. 8 or 16 bits to floats
      __m128 ovY_ps = Fourpixels_to_floats<pixel_t>(ovY + x);

      // generic, 8-16 bits
      if constexpr (has_mask) {
        __m128 final_opacity;
        auto overlay_opacity_minus1 = _mm_sub_ps(_mm_mul_ps(ovY_ps, factor_simd), one_ps_simd); // ovY[x] * factor - 1.0f;

        __m128 maskY_ps = Fourpixels_to_floats<pixel_t>(maskY + x);
        final_opacity = _mm_mul_ps(maskY_ps, factor_mul_opacity_simd); // maskY[x] * factor_mul_opacity;
        auto Yfactor = _mm_add_ps(one_ps_simd, _mm_mul_ps(overlay_opacity_minus1, final_opacity)); // 1.0f + overlay_opacity_minus1 * final_opacity;
        __m128 baseY_ps = Fourpixels_to_floats<pixel_t>(baseY + x);
        Y = _mm_mul_ps(baseY_ps, Yfactor); // Y = (int)(baseY[x] * Yfactor);

        __m128 maskU_ps = Fourpixels_to_floats<pixel_t>(maskU + x);
        final_opacity = _mm_mul_ps(maskU_ps, factor_mul_opacity_simd); // maskY[x] * factor_mul_opacity;
        auto Ufactor = _mm_add_ps(one_ps_simd, _mm_mul_ps(overlay_opacity_minus1, final_opacity)); // 1.0f + overlay_opacity_minus1 * final_opacity;
        __m128 baseU_ps = FourChromapixels_to_floats<pixel_t>(baseU + x, half_i_simd);
        U = _mm_mul_ps(baseU_ps, Ufactor); // U = (int)((baseU[x] - half_i) * Ufactor)         not yet: + half_i;

        __m128 maskV_ps = Fourpixels_to_floats<pixel_t>(maskV + x);
        final_opacity = _mm_mul_ps(maskV_ps, factor_mul_opacity_simd); // maskY[x] * factor_mul_opacity;
        auto Vfactor = _mm_add_ps(one_ps_simd, _mm_mul_ps(overlay_opacity_minus1, final_opacity)); // 1.0f + overlay_opacity_minus1 * final_opacity;
        __m128 baseV_ps = FourChromapixels_to_floats<pixel_t>(baseV + x, half_i_simd);
        V = _mm_mul_ps(baseV_ps, Vfactor); // V = (int)((baseV[x] - half_i) * Vfactor)         not yet: + half_i;

      }
      else {
        auto overlay_opacity_minus1 = _mm_sub_ps(_mm_mul_ps(ovY_ps, factor_simd), one_ps_simd); // ovY[x] * factor - 1.0f;
        auto common_factor = _mm_add_ps(one_ps_simd, _mm_mul_ps(overlay_opacity_minus1, opacity_simd)); // 1.0f + overlay_opacity_minus1 * opacity_f;

        auto Yfactor = common_factor;
        __m128 baseY_ps = Fourpixels_to_floats<pixel_t>(baseY + x);
        Y = _mm_mul_ps(baseY_ps, Yfactor); // Y = (int)(baseY[x] * Yfactor); 

        auto Ufactor = common_factor;
        __m128 baseU_ps = FourChromapixels_to_floats<pixel_t>(baseU + x, half_i_simd);
        U = _mm_mul_ps(baseU_ps, Ufactor); // U = (int)((baseU[x] - half_i) * Ufactor)         not yet: + half_i;

        auto Vfactor = common_factor;
        __m128 baseV_ps = FourChromapixels_to_floats<pixel_t>(baseV + x, half_i_simd);
        V = _mm_mul_ps(baseV_ps, Vfactor); // V = (int)((baseV[x] - half_i) * Vfactor)         not yet: + half_i;

      }

      Store_Fourpixels<pixel_t>(baseY + x, Y, rounder_simd);
      Store_FourChromapixels<pixel_t>(baseU + x, U, half_and_rounder_simd);
      Store_FourChromapixels<pixel_t>(baseV + x, V, half_and_rounder_simd);

    }

    for (int x = wMod4; x < width; x++) {
      // from of_mul_c
      int Y, U, V;

      // generic, 8-16 bits
      // This part re-appears in SSE2 code (non mod4 end-of-line fragment)
      // Unlike the old integer version here is proper rounding
      const float overlay_opacity_minus1 = ovY[x] * factor - 1.0f;
      if constexpr (has_mask) {
        float final_opacity;

        final_opacity = maskY[x] * factor_mul_opacity;
        auto Yfactor = 1.0f + overlay_opacity_minus1 * final_opacity;
        Y = (int)(baseY[x] * Yfactor + 0.5f);

        final_opacity = maskU[x] * factor_mul_opacity;
        auto Ufactor = 1.0f + overlay_opacity_minus1 * final_opacity;
        U = (int)(((float)baseU[x] - half_f) * Ufactor + half_f + 0.5f);

        final_opacity = maskV[x] * factor_mul_opacity;
        auto Vfactor = 1.0f + overlay_opacity_minus1 * final_opacity;
        V = (int)(((float)baseV[x] - half_f) * Vfactor + half_f + 0.5f);
      }
      else {
        const float common_factor = 1.0f + overlay_opacity_minus1 * opacity_f;

        auto Yfactor = common_factor;
        Y = (int)((float)baseY[x] * Yfactor + 0.5f);

        auto Ufactor = common_factor;
        U = (int)(((float)baseU[x] - half_f) * Ufactor + half_f + 0.5f);

        auto Vfactor = common_factor;
        V = (int)(((float)baseV[x] - half_f) * Vfactor + half_f + 0.5f);

      }

      baseU[x] = (pixel_t)U;
      baseV[x] = (pixel_t)V;
      baseY[x] = (pixel_t)Y;
    }

    if constexpr (has_mask) {
      maskY += maskpitch;
      maskU += maskpitch;
      maskV += maskpitch;
    }

    baseY += basepitch;
    baseU += basepitch;
    baseV += basepitch;

    ovY += overlaypitch;

  }
}


// instantiate
template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint8_t, false, false>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint8_t* ovY, int overlaypitch, uint8_t* baseY, uint8_t* baseU, uint8_t* baseV, int basepitch, const uint8_t* maskY, const uint8_t* maskU, const uint8_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint8_t, false, true>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint8_t* ovY, int overlaypitch, uint8_t* baseY, uint8_t* baseU, uint8_t* baseV, int basepitch, const uint8_t* maskY, const uint8_t* maskU, const uint8_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint8_t, true, false>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint8_t* ovY, int overlaypitch, uint8_t* baseY, uint8_t* baseU, uint8_t* baseV, int basepitch, const uint8_t* maskY, const uint8_t* maskU, const uint8_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint8_t, true, true>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint8_t* ovY, int overlaypitch, uint8_t* baseY, uint8_t* baseU, uint8_t* baseV, int basepitch, const uint8_t* maskY, const uint8_t* maskU, const uint8_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint16_t, false, false>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint16_t* ovY, int overlaypitch, uint16_t* baseY, uint16_t* baseU, uint16_t* baseV, int basepitch, const uint16_t* maskY, const uint16_t* maskU, const uint16_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint16_t, false, true>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint16_t* ovY, int overlaypitch, uint16_t* baseY, uint16_t* baseU, uint16_t* baseV, int basepitch, const uint16_t* maskY, const uint16_t* maskU, const uint16_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint16_t, true, false>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint16_t* ovY, int overlaypitch, uint16_t* baseY, uint16_t* baseU, uint16_t* baseV, int basepitch, const uint16_t* maskY, const uint16_t* maskU, const uint16_t* maskV, int maskpitch);

template
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void of_multiply_sse41<uint16_t, true, true>(int bits_per_pixel, const float opacity_f, const int opacity, int width, int height,
  const uint16_t* ovY, int overlaypitch, uint16_t* baseY, uint16_t* baseU, uint16_t* baseV, int basepitch, const uint16_t* maskY, const uint16_t* maskU, const uint16_t* maskV, int maskpitch);
