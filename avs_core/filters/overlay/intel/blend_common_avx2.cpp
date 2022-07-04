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
#include "blend_common_avx2.h"
#include "../blend_common.h"

#include <stdint.h>

#ifdef AVS_WINDOWS
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

template<typename pixel_t>
static AVS_FORCEINLINE __m256 Eightpixels_to_floats(const pixel_t* src) {
  __m256i srci;
  if constexpr (sizeof(pixel_t) == 1) {
    srci = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src)));
  }
  else {
    srci = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
  }
  return _mm256_cvtepi32_ps(srci);
}

template<typename pixel_t, int bits_per_pixel>
static AVS_FORCEINLINE void Store_Eightpixels(pixel_t* dst, __m256 what, const __m256 rounder) {
  what = _mm256_add_ps(what, rounder); // round
  __m256i si32 = _mm256_cvttps_epi32(what); // truncate
  __m256i result = _mm256_packus_epi32(si32, si32); // only low 8 words needed
  result = _mm256_permute4x64_epi64(result, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
  __m128i result128 = _mm256_castsi256_si128(result);
  if constexpr (sizeof(pixel_t) == 1) {
    __m128i result64 = _mm_packus_epi16(result128, result128);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), result64);
  } else {
    /* when mask is 0..1 checked then this is not possible
    if constexpr (bits_per_pixel < 16) { // otherwise no clamp needed
      constexpr int max_pixel_value = (1 << bits_per_pixel) - 1;
      auto max_pixel_value_v = _mm_set1_epi16(static_cast<uint16_t>(max_pixel_value));
      result128 = _mm_min_epu16(result128, max_pixel_value_v);
    }
    */
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), result128);
  }
}

AVS_FORCEINLINE static __m256 overlay_blend_avx2_core_new(const __m256& p1_f, const __m256& p2_f, const __m256& factor) {
  /*
  //  p1*(1-mask_f) + p2*mask_f -> p1 + (p2-p1)*mask_f
  constexpr int max_pixel_value = (1 << bits_per_pixel) - 1;
  constexpr float factor = 1.0f / max_pixel_value;
  constexpr float half_rounder = 0.5f;
  const float mask_f = mask * factor;
  const float res = p1 + (p2 - p1) * mask_f;
  int result = (int)(res + 0.5f);
  */
  // rounding not here, but before storage
  auto res = _mm256_add_ps(p1_f, _mm256_mul_ps(_mm256_sub_ps(p2_f, p1_f), factor));
  return res;
} 

template<bool has_mask, typename pixel_t, int bits_per_pixel>
void overlay_blend_avx2_uint(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f)
{

  auto rounder = _mm256_set1_ps(0.5f);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  auto factor = has_mask ? opacity_f / max_pixel_value : opacity_f;
  auto factor_v = _mm256_set1_ps(factor);

  const int realwidth = width * sizeof(pixel_t);

  // 2x8 pixels at a time
  constexpr int bytes_per_cycle = 16 * sizeof(pixel_t);
  int wMod16 = (realwidth / bytes_per_cycle) * bytes_per_cycle;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod16; x += bytes_per_cycle) {
      auto unpacked_p1 = Eightpixels_to_floats<pixel_t>((const pixel_t*)(p1 + x)); // 8x32
      auto unpacked_p2 = Eightpixels_to_floats<pixel_t>((const pixel_t*)(p2 + x)); // 8x32

      auto unpacked_p1_2 = Eightpixels_to_floats<pixel_t>((const pixel_t*)(p1 + x + bytes_per_cycle / 2)); // 8x32
      auto unpacked_p2_2 = Eightpixels_to_floats<pixel_t>((const pixel_t*)(p2 + x + bytes_per_cycle / 2)); // 8x32

      __m256 result, result_2;
      if constexpr (has_mask) {
        auto unpacked_mask = Eightpixels_to_floats<pixel_t>((const pixel_t*)(mask + x)); // 8x32
        unpacked_mask = _mm256_mul_ps(unpacked_mask, factor_v);
        result = overlay_blend_avx2_core_new(unpacked_p1, unpacked_p2, unpacked_mask);
        
        auto unpacked_mask_2 = Eightpixels_to_floats<pixel_t>((const pixel_t*)(mask + x + bytes_per_cycle / 2)); // 8x32
        unpacked_mask_2 = _mm256_mul_ps(unpacked_mask_2, factor_v);
        result_2 = overlay_blend_avx2_core_new(unpacked_p1_2, unpacked_p2_2, unpacked_mask_2);
      }
      else {
        result = overlay_blend_avx2_core_new(unpacked_p1, unpacked_p2, factor_v);
        result_2 = overlay_blend_avx2_core_new(unpacked_p1_2, unpacked_p2_2, factor_v);
      }

      Store_Eightpixels<pixel_t, bits_per_pixel>((pixel_t*)(p1 + x), result, rounder);
      Store_Eightpixels<pixel_t, bits_per_pixel>((pixel_t*)(p1 + x + bytes_per_cycle / 2), result_2, rounder);
    }

    // Leftover value

    for (int x = wMod16 / sizeof(pixel_t); x < width; x++) {
      const float new_factor = has_mask ? static_cast<float>(reinterpret_cast<const pixel_t*>(mask)[x]) * factor : factor;
      auto result = overlay_blend_c_core_simple(reinterpret_cast<pixel_t*>(p1)[x], reinterpret_cast<const pixel_t*>(p2)[x], new_factor);
      reinterpret_cast<pixel_t*>(p1)[x] = (pixel_t)(result + 0.5f);
    }

    p1 += p1_pitch;
    p2 += p2_pitch;
    if (has_mask)
      mask += mask_pitch;
  }
}

// instantiate
// mask yes/no
template void overlay_blend_avx2_uint<true, uint8_t, 8>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<true, uint16_t, 10>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<true, uint16_t, 12>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<true, uint16_t, 14>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<true, uint16_t, 16>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
//--
template void overlay_blend_avx2_uint<false, uint8_t, 8>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<false, uint16_t, 10>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<false, uint16_t, 12>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<false, uint16_t, 14>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_uint<false, uint16_t, 16>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);

template<bool has_mask>
void overlay_blend_avx2_float(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch,
  const int width, const int height, const int opacity, const float opacity_f)
{

  const int realwidth = width * sizeof(float);

  int wMod32 = (realwidth / 32) * 32;
  auto opacity_v = _mm256_set1_ps(opacity_f);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod32; x += 32) {
      auto p1_f = _mm256_loadu_ps(reinterpret_cast<const float*>(p1 + x));
      auto p2_f = _mm256_loadu_ps(reinterpret_cast<const float*>(p2 + x));
      __m256 new_mask;
      if constexpr (has_mask) {
        new_mask = _mm256_loadu_ps(reinterpret_cast<const float*>(mask + x));
        new_mask = _mm256_mul_ps(new_mask, opacity_v);
      }
      else {
        new_mask = opacity_v;
      }
      auto result = _mm256_add_ps(p1_f, _mm256_mul_ps(_mm256_sub_ps(p2_f, p1_f), new_mask)); // p1*(1-mask) + p2*mask = p1+(p2-p1)*mask

      _mm256_storeu_ps(reinterpret_cast<float*>(p1 + x), result);
    }

    // Leftover value

    for (int x = wMod32 / sizeof(float); x < width; x++) {
      auto new_mask = has_mask ? reinterpret_cast<const float*>(mask)[x] * opacity_f : opacity_f;
      auto p1x = reinterpret_cast<float*>(p1)[x];
      auto p2x = reinterpret_cast<const float*>(p2)[x];
      auto result = p1x + (p2x - p1x) * new_mask; // p1x*(1-new_mask) + p2x*mask
      reinterpret_cast<float*>(p1)[x] = result;
    }


    p1 += p1_pitch;
    p2 += p2_pitch;
    if constexpr (has_mask)
      mask += mask_pitch;
  }
}

template void overlay_blend_avx2_float<false>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);
template void overlay_blend_avx2_float<true>(BYTE* p1, const BYTE* p2, const BYTE* mask,
  const int p1_pitch, const int p2_pitch, const int mask_pitch, const int width, const int height, const int opacity, const float opacity_f);

