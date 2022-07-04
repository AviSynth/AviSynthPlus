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



// Avisynth filter: Layer
// by "poptones" (poptones@myrealbox.com)

#include "layer_sse.h"
#include "../layer.h"

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <avs/minmax.h>
#include <avs/alignment.h>
#include "../core/internal.h"
#include <emmintrin.h>
#include "../convert/convert_planar.h"
#include <algorithm>


static AVS_FORCEINLINE __m128i mask_core_sse2(__m128i& src, __m128i& alpha, __m128i& not_alpha_mask, __m128i& zero, __m128i& matrix, __m128i& round_mask) {
  __m128i not_alpha = _mm_and_si128(src, not_alpha_mask);

  __m128i pixel0 = _mm_unpacklo_epi8(alpha, zero);
  __m128i pixel1 = _mm_unpackhi_epi8(alpha, zero);

  pixel0 = _mm_madd_epi16(pixel0, matrix);
  pixel1 = _mm_madd_epi16(pixel1, matrix);

  __m128i tmp = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel0), _mm_castsi128_ps(pixel1), _MM_SHUFFLE(3, 1, 3, 1)));
  __m128i tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel0), _mm_castsi128_ps(pixel1), _MM_SHUFFLE(2, 0, 2, 0)));

  tmp = _mm_add_epi32(tmp, tmp2);
  tmp = _mm_add_epi32(tmp, round_mask);
  tmp = _mm_srli_epi32(tmp, 15);
  __m128i result_alpha = _mm_slli_epi32(tmp, 24);

  return _mm_or_si128(result_alpha, not_alpha);
}

void mask_sse2(BYTE* srcp, const BYTE* alphap, int src_pitch, int alpha_pitch, size_t width, size_t height) {
  __m128i matrix = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);
  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(16384);
  __m128i not_alpha_mask = _mm_set1_epi32(0x00FFFFFF);

  size_t width_bytes = width * 4;
  size_t width_mod16 = width_bytes / 16 * 16;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width_mod16; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i alpha = _mm_load_si128(reinterpret_cast<const __m128i*>(alphap + x));
      __m128i result = mask_core_sse2(src, alpha, not_alpha_mask, zero, matrix, round_mask);

      _mm_store_si128(reinterpret_cast<__m128i*>(srcp + x), result);
    }

    if (width_mod16 < width_bytes) {
      __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + width_bytes - 16));
      __m128i alpha = _mm_loadu_si128(reinterpret_cast<const __m128i*>(alphap + width_bytes - 16));
      __m128i result = mask_core_sse2(src, alpha, not_alpha_mask, zero, matrix, round_mask);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(srcp + width_bytes - 16), result);
    }

    srcp += src_pitch;
    alphap += alpha_pitch;
  }
}

#ifdef X86_32

static AVS_FORCEINLINE __m64 mask_core_mmx(__m64& src, __m64& alpha, __m64& not_alpha_mask, __m64& zero, __m64& matrix, __m64& round_mask) {
  __m64 not_alpha = _mm_and_si64(src, not_alpha_mask);

  __m64 pixel0 = _mm_unpacklo_pi8(alpha, zero);
  __m64 pixel1 = _mm_unpackhi_pi8(alpha, zero);

  pixel0 = _mm_madd_pi16(pixel0, matrix); //a0*0 + r0*cyr | g0*cyg + b0*cyb
  pixel1 = _mm_madd_pi16(pixel1, matrix); //a1*0 + r1*cyr | g1*cyg + b1*cyb

  __m64 tmp = _mm_unpackhi_pi32(pixel0, pixel1); // r1*cyr | r0*cyr
  __m64 tmp2 = _mm_unpacklo_pi32(pixel0, pixel1); // g1*cyg + b1*cyb | g0*cyg + b0*cyb

  tmp = _mm_add_pi32(tmp, tmp2); // r1*cyr + g1*cyg + b1*cyb | r0*cyr + g0*cyg + b0*cyb
  tmp = _mm_add_pi32(tmp, round_mask); // r1*cyr + g1*cyg + b1*cyb + 16384 | r0*cyr + g0*cyg + b0*cyb + 16384
  tmp = _mm_srli_pi32(tmp, 15); // 0 0 0 p2 | 0 0 0 p1
  __m64 result_alpha = _mm_slli_pi32(tmp, 24);

  return _mm_or_si64(result_alpha, not_alpha);
}

void mask_mmx(BYTE* srcp, const BYTE* alphap, int src_pitch, int alpha_pitch, size_t width, size_t height) {
  __m64 matrix = _mm_set_pi16(0, cyr, cyg, cyb);
  __m64 zero = _mm_setzero_si64();
  __m64 round_mask = _mm_set1_pi32(16384);
  __m64 not_alpha_mask = _mm_set1_pi32(0x00FFFFFF);

  size_t width_bytes = width * 4;
  size_t width_mod8 = width_bytes / 8 * 8;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width_mod8; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x); //pixels 0 and 1
      __m64 alpha = *reinterpret_cast<const __m64*>(alphap + x);
      __m64 result = mask_core_mmx(src, alpha, not_alpha_mask, zero, matrix, round_mask);

      *reinterpret_cast<__m64*>(srcp + x) = result;
    }

    if (width_mod8 < width_bytes) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp + width_bytes - 4));
      __m64 alpha = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(alphap + width_bytes - 4));

      __m64 result = mask_core_mmx(src, alpha, not_alpha_mask, zero, matrix, round_mask);

      *reinterpret_cast<int*>(srcp + width_bytes - 4) = _mm_cvtsi64_si32(result);
    }

    srcp += src_pitch;
    alphap += alpha_pitch;
  }
  _mm_empty();
}

#endif



void colorkeymask_sse2(BYTE* pf, int pitch, int color, int height, int width, int tolB, int tolG, int tolR) {
  unsigned int t = 0xFF000000 | (tolR << 16) | (tolG << 8) | tolB;
  __m128i tolerance = _mm_set1_epi32(t);
  __m128i colorv = _mm_set1_epi32(color);
  __m128i zero = _mm_setzero_si128();

  BYTE* endp = pf + pitch * height;

  while (pf < endp)
  {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(pf));
    __m128i gt = _mm_subs_epu8(colorv, src);
    __m128i lt = _mm_subs_epu8(src, colorv);
    __m128i absdiff = _mm_or_si128(gt, lt); //abs(color - src)

    __m128i not_passed = _mm_subs_epu8(absdiff, tolerance);
    __m128i passed = _mm_cmpeq_epi32(not_passed, zero);
    passed = _mm_slli_epi32(passed, 24);
    __m128i result = _mm_andnot_si128(passed, src);

    _mm_store_si128(reinterpret_cast<__m128i*>(pf), result);

    pf += 16;
  }
}

#ifdef X86_32

static AVS_FORCEINLINE __m64 colorkeymask_core_mmx(const __m64& src, const __m64& colorv, const __m64& tolerance, const __m64& zero) {
  __m64 gt = _mm_subs_pu8(colorv, src);
  __m64 lt = _mm_subs_pu8(src, colorv);
  __m64 absdiff = _mm_or_si64(gt, lt); //abs(color - src)

  __m64 not_passed = _mm_subs_pu8(absdiff, tolerance);
  __m64 passed = _mm_cmpeq_pi32(not_passed, zero);
  passed = _mm_slli_pi32(passed, 24);
  return _mm_andnot_si64(passed, src);
}

void colorkeymask_mmx(BYTE* srcp, int pitch, int color, int height, int width, int tolB, int tolG, int tolR) {
#pragma warning(push)
#pragma warning(disable: 4309)
  __m64 tolerance = _mm_set_pi8(0xFF, tolR, tolG, tolB, 0xFF, tolR, tolG, tolB);
#pragma warning(pop)
  __m64 colorv = _mm_set1_pi32(color);
  __m64 zero = _mm_setzero_si64();

  int mod8_width = width / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 result = colorkeymask_core_mmx(src, colorv, tolerance, zero);
      *reinterpret_cast<__m64*>(srcp + x) = result;
    }

    if (mod8_width != width) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp + width - 4));
      __m64 result = colorkeymask_core_mmx(src, colorv, tolerance, zero);
      *reinterpret_cast<int*>(srcp + width - 4) = _mm_cvtsi64_si32(result);
    }

    srcp += pitch;
  }

  _mm_empty();
}

#endif


void invert_frame_sse2(BYTE* frame, int pitch, int width, int height, int mask) {
  __m128i maskv = _mm_set1_epi32(mask);

  BYTE* endp = frame + pitch * height;

  while (frame < endp) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(frame));
    __m128i inv = _mm_xor_si128(src, maskv);
    _mm_store_si128(reinterpret_cast<__m128i*>(frame), inv);
    frame += 16;
  }
}

void invert_frame_uint16_sse2(BYTE* frame, int pitch, int width, int height, uint64_t mask64) {
  __m128i maskv = _mm_set_epi32((uint32_t)(mask64 >> 32), (uint32_t)mask64, (uint32_t)(mask64 >> 32), (uint32_t)mask64);

  BYTE* endp = frame + pitch * height;

  while (frame < endp) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(frame));
    __m128i inv = _mm_xor_si128(src, maskv);
    _mm_store_si128(reinterpret_cast<__m128i*>(frame), inv);
    frame += 16;
  }
}

#ifdef X86_32

//mod4 width (in bytes) is required
void invert_frame_mmx(BYTE* frame, int pitch, int width, int height, int mask)
{
  __m64 maskv = _mm_set1_pi32(mask);
  int mod8_width = width / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(frame + x);
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<__m64*>(frame + x) = inv;
    }

    if (mod8_width != width) {
      //last four pixels
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(frame + width - 4));
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<int*>(frame + width - 4) = _mm_cvtsi64_si32(inv);
    }
    frame += pitch;
  }
  _mm_empty();
}

void invert_plane_mmx(BYTE* frame, int pitch, int width, int height)
{
#pragma warning(push)
#pragma warning(disable: 4309)
  __m64 maskv = _mm_set1_pi8(0xFF);
#pragma warning(pop)
  int mod8_width = width / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(frame + x);
      __m64 inv = _mm_xor_si64(src, maskv);
      *reinterpret_cast<__m64*>(frame + x) = inv;
    }

    for (int x = mod8_width; x < width; ++x) {
      frame[x] = frame[x] ^ 255;
    }
    frame += pitch;
  }
  _mm_empty();
}

#endif


/*******************************
 *******   Layer Filter   ******
 *******************************/


 /* YUY2 */

template<bool use_chroma>
void layer_yuy2_mul_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int width_mod8 = width / 8 * 8;

  __m128i alpha = _mm_set1_epi16(level);
  __m128i half_alpha = _mm_srli_epi16(alpha, 1);
  __m128i luma_mask = _mm_set1_epi16(0x00FF);
  __m128i v128 = _mm_set1_epi16(128);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width_mod8; x += 8) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x * 2));
      __m128i ovr = _mm_load_si128(reinterpret_cast<const __m128i*>(ovrp + x * 2));

      __m128i src_luma = _mm_and_si128(src, luma_mask);
      __m128i ovr_luma = _mm_and_si128(ovr, luma_mask);

      __m128i src_chroma = _mm_srli_epi16(src, 8);
      __m128i ovr_chroma = _mm_srli_epi16(ovr, 8);

      __m128i luma = _mm_mullo_epi16(src_luma, ovr_luma);
      luma = _mm_srli_epi16(luma, 8);
      luma = _mm_subs_epi16(luma, src_luma);
      luma = _mm_mullo_epi16(luma, alpha);
      luma = _mm_srli_epi16(luma, 8);

      __m128i chroma;
      if (use_chroma) {
        chroma = _mm_subs_epi16(ovr_chroma, src_chroma);
        chroma = _mm_mullo_epi16(chroma, alpha);
      }
      else {
        chroma = _mm_subs_epi16(v128, src_chroma);
        chroma = _mm_mullo_epi16(chroma, half_alpha);
      }

      //it's fine, don't optimize
      chroma = _mm_srli_epi16(chroma, 8);
      chroma = _mm_slli_epi16(chroma, 8);

      __m128i dst = _mm_or_si128(luma, chroma);
      dst = _mm_add_epi8(src, dst);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x * 2), dst);
    }

    for (int x = width_mod8; x < width; ++x) {
      dstp[x * 2] = dstp[x * 2] + (((((ovrp[x * 2] * dstp[x * 2]) >> 8) - dstp[x * 2]) * level) >> 8);
      if (use_chroma) {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((ovrp[x * 2 + 1] - dstp[x * 2 + 1]) * level) >> 8);
      }
      else {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((128 - dstp[x * 2 + 1]) * (level / 2)) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_yuy2_mul_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_mul_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);


#ifdef X86_32
template<bool use_chroma>
static AVS_FORCEINLINE __m64 layer_yuy2_mul_mmx_core(const __m64& src, const __m64& ovr, const __m64& luma_mask,
  const __m64& alpha, const __m64& half_alpha, const __m64& v128) {
  __m64 src_luma = _mm_and_si64(src, luma_mask);
  __m64 ovr_luma = _mm_and_si64(ovr, luma_mask);

  __m64 src_chroma = _mm_srli_pi16(src, 8);
  __m64 ovr_chroma = _mm_srli_pi16(ovr, 8);

  __m64 luma = _mm_mullo_pi16(src_luma, ovr_luma);
  luma = _mm_srli_pi16(luma, 8);
  luma = _mm_subs_pi16(luma, src_luma);
  luma = _mm_mullo_pi16(luma, alpha);
  luma = _mm_srli_pi16(luma, 8);

  __m64 chroma;
  if (use_chroma) {
    chroma = _mm_subs_pi16(ovr_chroma, src_chroma);
    chroma = _mm_mullo_pi16(chroma, alpha);
  }
  else {
    chroma = _mm_subs_pi16(v128, src_chroma);
    chroma = _mm_mullo_pi16(chroma, half_alpha);
  }

  //it's fine, don't optimize
  chroma = _mm_srli_pi16(chroma, 8);
  chroma = _mm_slli_pi16(chroma, 8);

  __m64 dst = _mm_or_si64(luma, chroma);
  return _mm_add_pi8(src, dst);
}

template<bool use_chroma>
void layer_yuy2_mul_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int width_mod4 = width / 4 * 4;

  __m64 alpha = _mm_set1_pi16(level);
  __m64 half_alpha = _mm_srli_pi16(alpha, 1);
  __m64 luma_mask = _mm_set1_pi16(0x00FF);
  __m64 v128 = _mm_set1_pi16(128);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width_mod4; x += 4) {
      __m64 src = *reinterpret_cast<const __m64*>(dstp + x * 2);
      __m64 ovr = *reinterpret_cast<const __m64*>(ovrp + x * 2);

      *reinterpret_cast<__m64*>(dstp + x * 2) = layer_yuy2_mul_mmx_core<use_chroma>(src, ovr, luma_mask, alpha, half_alpha, v128);
    }

    if (width_mod4 != width) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + width_mod4 * 2));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + width_mod4 * 2));

      *reinterpret_cast<int*>(dstp + width_mod4 * 2) = _mm_cvtsi64_si32(layer_yuy2_mul_mmx_core<use_chroma>(src, ovr, luma_mask, alpha, half_alpha, v128));
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_yuy2_mul_mmx<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_mul_mmx<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#endif




template<bool use_chroma>
void layer_yuy2_add_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m128i alpha = _mm_set1_epi16(level);
  __m128i zero = _mm_setzero_si128();
  __m128i v128 = _mm_set1_epi32(0x00800000);
  __m128i luma_mask = _mm_set1_epi32(0x000000FF);
  int mod4_width = width / 4 * 4;

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod4_width; x += 4) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 2));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 2));

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      if (!use_chroma) {
        ovr = _mm_and_si128(ovr, luma_mask);
        ovr = _mm_or_si128(ovr, v128);
      }

      __m128i diff = _mm_subs_epi16(ovr, src);
      diff = _mm_mullo_epi16(diff, alpha);
      diff = _mm_add_epi16(diff, rounder_simd);
      diff = _mm_srli_epi16(diff, 8);

      __m128i dst = _mm_add_epi8(diff, src);
      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 2), dst);
    }

    for (int x = mod4_width; x < width; ++x) {
      dstp[x * 2] = dstp[x * 2] + (((ovrp[x * 2] - dstp[x * 2]) * level + rounder) >> 8);
      if (use_chroma) {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((ovrp[x * 2 + 1] - dstp[x * 2 + 1]) * level + rounder) >> 8);
      }
      else {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((128 - dstp[x * 2 + 1]) * level + rounder) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_yuy2_add_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_add_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#ifdef X86_32
template<bool use_chroma>
void layer_yuy2_add_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m64 alpha = _mm_set1_pi16(level);
  __m64 zero = _mm_setzero_si64();
  __m64 v128 = _mm_set1_pi32(0x00800000);
  __m64 luma_mask = _mm_set1_pi32(0x000000FF);
  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 2) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 2));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 2));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      if (!use_chroma) {
        ovr = _mm_and_si64(ovr, luma_mask); //00 00 00 YY 00 00 00 YY
        ovr = _mm_or_si64(ovr, v128); //00 128 00 YY 00 128 00 YY
      }

      __m64 diff = _mm_subs_pi16(ovr, src);
      diff = _mm_mullo_pi16(diff, alpha);
      diff = _mm_add_pi16(diff, rounder_simd);
      diff = _mm_srli_pi16(diff, 8);

      __m64 dst = _mm_add_pi8(diff, src);
      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 2) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}
// instantiate
template void layer_yuy2_add_mmx<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_add_mmx<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#endif




void layer_yuy2_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  AVS_UNUSED(level);
  int width_bytes = width * 2;
  int width_mod16 = width_bytes / 16 * 16;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width_mod16; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      __m128i ovr = _mm_load_si128(reinterpret_cast<const __m128i*>(ovrp + x));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_avg_epu8(src, ovr));
    }

    for (int x = width_mod16; x < width_bytes; ++x) {
      dstp[x] = (dstp[x] + ovrp[x] + 1) / 2;
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

#ifdef X86_32
void layer_yuy2_fast_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int width_bytes = width * 2;
  int width_mod8 = width_bytes / 8 * 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width_mod8; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(dstp + x);
      __m64 ovr = *reinterpret_cast<const __m64*>(ovrp + x);

      *reinterpret_cast<__m64*>(dstp + x) = _mm_avg_pu8(src, ovr);
    }

    if (width_mod8 != width_bytes) {
      //two last pixels
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + width_mod8 - 4));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + width_mod8 - 4));

      *reinterpret_cast<int*>(dstp + width_bytes - 4) = _mm_cvtsi64_si32(_mm_avg_pu8(src, ovr));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}
#endif


template<typename pixel_t>
void layer_genericplane_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  AVS_UNUSED(level);
  int width_bytes = width * sizeof(pixel_t);
  int width_mod16 = width_bytes / 16 * 16;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width_mod16; x += 16) {
      __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x));
      __m128i ovr = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ovrp + x));
      if constexpr (sizeof(pixel_t) == 1)
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_avg_epu8(src, ovr));
      else
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + x), _mm_avg_epu16(src, ovr));
    }

    for (int x = width_mod16 / sizeof(pixel_t); x < width; ++x) {
      reinterpret_cast<pixel_t*>(dstp)[x] = (reinterpret_cast<pixel_t*>(dstp)[x] + reinterpret_cast<const pixel_t*>(ovrp)[x] + 1) / 2;
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_genericplane_fast_sse2<uint8_t>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_genericplane_fast_sse2<uint16_t>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

template<bool use_chroma>
void layer_yuy2_subtract_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m128i alpha = _mm_set1_epi16(level);
  __m128i zero = _mm_setzero_si128();
  __m128i ff = _mm_set1_epi16(0x00FF);
  __m128i v127 = _mm_set1_epi32(0x007F0000);
  __m128i luma_mask = _mm_set1_epi32(0x000000FF);
  int mod4_width = width / 4 * 4;

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod4_width; x += 4) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 2));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 2));

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      if (!use_chroma) {
        ovr = _mm_and_si128(ovr, luma_mask);
        ovr = _mm_or_si128(ovr, v127); //255-127 on the next step will be 128
      }

      __m128i diff = _mm_subs_epi16(ff, ovr);
      diff = _mm_subs_epi16(diff, src);
      diff = _mm_mullo_epi16(diff, alpha);
      diff = _mm_add_epi16(diff, rounder_simd);
      diff = _mm_srli_epi16(diff, 8);

      __m128i dst = _mm_add_epi8(diff, src);
      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 2), dst);
    }

    for (int x = mod4_width; x < width; ++x) {
      dstp[x * 2] = dstp[x * 2] + (((255 - ovrp[x * 2] - dstp[x * 2]) * level + rounder) >> 8);
      if (use_chroma) {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((255 - ovrp[x * 2 + 1] - dstp[x * 2 + 1]) * level + rounder) >> 8);
      }
      else {
        dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((128 - dstp[x * 2 + 1]) * level + rounder) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_yuy2_subtract_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_subtract_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#ifdef X86_32
template<bool use_chroma>
void layer_yuy2_subtract_mmx(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m64 alpha = _mm_set1_pi16(level);
  __m64 zero = _mm_setzero_si64();
  __m64 ff = _mm_set1_pi16(0x00FF);
  __m64 v127 = _mm_set1_pi32(0x007F0000);
  __m64 luma_mask = _mm_set1_pi32(0x000000FF);

  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 2) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 2));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 2));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      if (!use_chroma) {
        ovr = _mm_and_si64(ovr, luma_mask);
        ovr = _mm_or_si64(ovr, v127); //255-127 on the next step will be 128
      }

      __m64 diff = _mm_subs_pi16(ff, ovr);
      diff = _mm_subs_pi16(diff, src);
      diff = _mm_mullo_pi16(diff, alpha);
      diff = _mm_add_pi16(diff, rounder_simd);
      diff = _mm_srli_pi16(diff, 8);

      __m64 dst = _mm_add_pi8(diff, src);
      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 2) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}
// instantiate
template void layer_yuy2_subtract_mmx<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_yuy2_subtract_mmx<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#endif


template<int mode>
void layer_yuy2_lighten_darken_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh) {
  int mod4_width = width / 4 * 4;

  __m128i alpha = _mm_set1_epi16(level);
  __m128i zero = _mm_setzero_si128();
  __m128i threshold = _mm_set1_epi16(thresh);

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod4_width; x += 4) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 2));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 2));

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      __m128i mask;
      if constexpr (mode == LIGHTEN) {
        __m128i tmp = _mm_add_epi16(src, threshold);
        mask = _mm_cmpgt_epi16(ovr, tmp);
      }
      else {
        __m128i tmp = _mm_sub_epi16(src, threshold);
        mask = _mm_cmpgt_epi16(tmp, ovr);
      }

      mask = _mm_shufflelo_epi16(mask, _MM_SHUFFLE(2, 2, 0, 0));
      mask = _mm_shufflehi_epi16(mask, _MM_SHUFFLE(2, 2, 0, 0));

      __m128i alpha_mask = _mm_and_si128(mask, alpha);

      __m128i diff = _mm_subs_epi16(ovr, src);
      diff = _mm_mullo_epi16(diff, alpha_mask);
      diff = _mm_add_epi16(diff, rounder_simd);
      diff = _mm_srli_epi16(diff, 8);

      __m128i dst = _mm_add_epi8(diff, src);
      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 2), dst);
    }

    for (int x = mod4_width; x < width; ++x) {
      int alpha_mask;
      if constexpr (mode == LIGHTEN)
        alpha_mask = ovrp[x * 2] > (dstp[x * 2] + thresh) ? level : 0;
      else // DARKEN
        alpha_mask = ovrp[x * 2] < (dstp[x * 2] - thresh) ? level : 0;

      dstp[x * 2] = dstp[x * 2] + (((ovrp[x * 2] - dstp[x * 2]) * alpha_mask + rounder) >> 8);
      dstp[x * 2 + 1] = dstp[x * 2 + 1] + (((ovrp[x * 2 + 1] - dstp[x * 2 + 1]) * alpha_mask + rounder) >> 8);
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_yuy2_lighten_darken_sse2<LIGHTEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template void layer_yuy2_lighten_darken_sse2<DARKEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);


#ifdef X86_32
template<int mode>
void layer_yuy2_lighten_darken_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh) {
  __m64 alpha = _mm_set1_pi16(level);
  __m64 zero = _mm_setzero_si64();
  __m64 threshold = _mm_set1_pi16(thresh);

  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 2) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 2));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 2));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      __m64 mask;
      if (mode == LIGHTEN) {
        __m64 tmp = _mm_add_pi16(src, threshold);
        mask = _mm_cmpgt_pi16(ovr, tmp);
      }
      else {
        __m64 tmp = _mm_sub_pi16(src, threshold);
        mask = _mm_cmpgt_pi16(tmp, ovr);
      }

      mask = _mm_shuffle_pi16(mask, _MM_SHUFFLE(2, 2, 0, 0));
      __m64 alpha_mask = _mm_and_si64(mask, alpha);

      __m64 diff = _mm_subs_pi16(ovr, src);
      diff = _mm_mullo_pi16(diff, alpha_mask);
      diff = _mm_add_pi16(diff, rounder_simd);
      diff = _mm_srli_pi16(diff, 8);

      __m64 dst = _mm_add_pi8(diff, src);
      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 2) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_yuy2_lighten_darken_isse<LIGHTEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template void layer_yuy2_lighten_darken_isse<DARKEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);

#endif


/* RGB32 */

//src format: xx xx xx xx | xx xx xx xx | a1 xx xx xx | a0 xx xx xx
//level_vector and one should be vectors of 32bit packed integers
static AVS_FORCEINLINE __m128i calculate_monochrome_alpha_sse2(const __m128i& src, const __m128i& level_vector, const __m128i& one) {
  __m128i alpha = _mm_srli_epi32(src, 24);
  alpha = _mm_mullo_epi16(alpha, level_vector);
  alpha = _mm_add_epi32(alpha, one);
  alpha = _mm_srli_epi32(alpha, 8);
  alpha = _mm_shufflelo_epi16(alpha, _MM_SHUFFLE(2, 2, 0, 0));
  return _mm_shuffle_epi32(alpha, _MM_SHUFFLE(1, 1, 0, 0));
}

static AVS_FORCEINLINE __m128i calculate_luma_sse2(const __m128i& src, const __m128i& rgb_coeffs, const __m128i& zero) {
  AVS_UNUSED(zero);
  __m128i temp = _mm_madd_epi16(src, rgb_coeffs);
  __m128i low = _mm_shuffle_epi32(temp, _MM_SHUFFLE(3, 3, 1, 1));
  temp = _mm_add_epi32(low, temp);
  temp = _mm_srli_epi32(temp, 15);
  __m128i result = _mm_shufflelo_epi16(temp, _MM_SHUFFLE(0, 0, 0, 0));
  return _mm_shufflehi_epi16(result, _MM_SHUFFLE(0, 0, 0, 0));
}

#ifdef X86_32
static AVS_FORCEINLINE __m64 calculate_luma_isse(const __m64& src, const __m64& rgb_coeffs, const __m64& zero) {
  __m64 temp = _mm_madd_pi16(src, rgb_coeffs);
  __m64 low = _mm_unpackhi_pi32(temp, zero);
  temp = _mm_add_pi32(low, temp);
  temp = _mm_srli_pi32(temp, 15);
  return _mm_shuffle_pi16(temp, _MM_SHUFFLE(0, 0, 0, 0));
}
#endif

template<bool use_chroma>
void layer_rgb32_mul_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int mod2_width = width / 2 * 2;

  __m128i zero = _mm_setzero_si128();
  __m128i level_vector = _mm_set1_epi32(level);
  __m128i one = _mm_set1_epi32(1);
  __m128i rgb_coeffs = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod2_width; x += 2) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 4));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 4));

      __m128i alpha = calculate_monochrome_alpha_sse2(ovr, level_vector, one);

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      __m128i luma;
      if (use_chroma) {
        luma = ovr;
      }
      else {
        luma = calculate_luma_sse2(ovr, rgb_coeffs, zero);
      }

      __m128i dst = _mm_mullo_epi16(luma, src);
      dst = _mm_srli_epi16(dst, 8);
      dst = _mm_subs_epi16(dst, src);
      dst = _mm_mullo_epi16(dst, alpha);
      dst = _mm_srli_epi16(dst, 8);
      dst = _mm_add_epi8(src, dst);

      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 4), dst);
    }

    if (width != mod2_width) {
      int x = mod2_width;
      int alpha = (ovrp[x * 4 + 3] * level + 1) >> 8;

      if (use_chroma) {
        dstp[x * 4] = dstp[x * 4] + (((((ovrp[x * 4] * dstp[x * 4]) >> 8) - dstp[x * 4]) * alpha) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((((ovrp[x * 4 + 1] * dstp[x * 4 + 1]) >> 8) - dstp[x * 4 + 1]) * alpha) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((((ovrp[x * 4 + 2] * dstp[x * 4 + 2]) >> 8) - dstp[x * 4 + 2]) * alpha) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((((ovrp[x * 4 + 3] * dstp[x * 4 + 3]) >> 8) - dstp[x * 4 + 3]) * alpha) >> 8);
      }
      else {
        int luma = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;

        dstp[x * 4] = dstp[x * 4] + (((((luma * dstp[x * 4]) >> 8) - dstp[x * 4]) * alpha) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((((luma * dstp[x * 4 + 1]) >> 8) - dstp[x * 4 + 1]) * alpha) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((((luma * dstp[x * 4 + 2]) >> 8) - dstp[x * 4 + 2]) * alpha) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((((luma * dstp[x * 4 + 3]) >> 8) - dstp[x * 4 + 3]) * alpha) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_rgb32_mul_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_mul_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);


#ifdef X86_32
template<bool use_chroma>
void layer_rgb32_mul_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m64 zero = _mm_setzero_si64();
  __m64 rgb_coeffs = _mm_set_pi16(0, cyr, cyg, cyb);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 4));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 4));

      __m64 alpha = _mm_cvtsi32_si64((ovrp[x * 4 + 3] * level + 1) >> 8);
      alpha = _mm_shuffle_pi16(alpha, _MM_SHUFFLE(0, 0, 0, 0));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      __m64 luma;
      if (use_chroma) {
        luma = ovr;
      }
      else {
        luma = calculate_luma_isse(ovr, rgb_coeffs, zero);
      }

      __m64 dst = _mm_mullo_pi16(luma, src);
      dst = _mm_srli_pi16(dst, 8);
      dst = _mm_subs_pi16(dst, src);
      dst = _mm_mullo_pi16(dst, alpha);
      dst = _mm_srli_pi16(dst, 8);
      dst = _mm_add_pi8(src, dst);

      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 4) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_rgb32_mul_isse<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_mul_isse<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#endif


template<bool use_chroma>
void layer_rgb32_add_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int mod2_width = width / 2 * 2;

  __m128i zero = _mm_setzero_si128();
  __m128i level_vector = _mm_set1_epi32(level);
  __m128i one = _mm_set1_epi32(1);
  __m128i rgb_coeffs = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod2_width; x += 2) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 4));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 4));

      __m128i alpha = calculate_monochrome_alpha_sse2(ovr, level_vector, one);

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      __m128i luma;
      if (use_chroma) {
        luma = ovr;
      }
      else {
        luma = calculate_luma_sse2(ovr, rgb_coeffs, zero);
      }

      __m128i dst = _mm_subs_epi16(luma, src);
      dst = _mm_mullo_epi16(dst, alpha);
      dst = _mm_add_epi16(dst, rounder_simd);
      dst = _mm_srli_epi16(dst, 8);
      dst = _mm_add_epi8(src, dst);

      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 4), dst);
    }

    if (width != mod2_width) {
      int x = mod2_width;
      int alpha = (ovrp[x * 4 + 3] * level + 1) >> 8;

      if (use_chroma) {
        dstp[x * 4] = dstp[x * 4] + (((ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> 8);
      }
      else {
        int luma = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;

        dstp[x * 4] = dstp[x * 4] + (((luma - dstp[x * 4]) * alpha + rounder) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((luma - dstp[x * 4 + 1]) * alpha + rounder) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((luma - dstp[x * 4 + 2]) * alpha + rounder) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((luma - dstp[x * 4 + 3]) * alpha + rounder) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_rgb32_add_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_add_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#ifdef X86_32
template<bool use_chroma>
void layer_rgb32_add_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m64 zero = _mm_setzero_si64();
  __m64 rgb_coeffs = _mm_set_pi16(0, cyr, cyg, cyb);

  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 4));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 4));

      __m64 alpha = _mm_cvtsi32_si64((ovrp[x * 4 + 3] * level + 1) >> 8);
      alpha = _mm_shuffle_pi16(alpha, _MM_SHUFFLE(0, 0, 0, 0));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      __m64 luma;
      if (use_chroma) {
        luma = ovr;
      }
      else {
        luma = calculate_luma_isse(ovr, rgb_coeffs, zero);
      }

      __m64 dst = _mm_subs_pi16(luma, src);
      dst = _mm_mullo_pi16(dst, alpha);
      dst = _mm_add_pi16(dst, rounder_simd);
      dst = _mm_srli_pi16(dst, 8);
      dst = _mm_add_pi8(src, dst);

      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 4) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_rgb32_add_isse<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_add_isse<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
#endif



void layer_rgb32_fast_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  layer_yuy2_fast_sse2(dstp, ovrp, dst_pitch, overlay_pitch, width * 2, height, level);
}

#ifdef X86_32
void layer_rgb32_fast_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  layer_yuy2_fast_isse(dstp, ovrp, dst_pitch, overlay_pitch, width * 2, height, level);
}
#endif



template<bool use_chroma>
void layer_rgb32_subtract_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  int mod2_width = width / 2 * 2;

  __m128i zero = _mm_setzero_si128();
  __m128i level_vector = _mm_set1_epi32(level);
  __m128i one = _mm_set1_epi32(1);
  __m128i rgb_coeffs = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);
  __m128i ff = _mm_set1_epi16(0x00FF);

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod2_width; x += 2) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 4));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 4));

      __m128i alpha = calculate_monochrome_alpha_sse2(ovr, level_vector, one);

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      __m128i luma;
      if (use_chroma) {
        luma = _mm_subs_epi16(ff, ovr);
      }
      else {
        luma = calculate_luma_sse2(_mm_andnot_si128(ovr, ff), rgb_coeffs, zero);
      }

      __m128i dst = _mm_subs_epi16(luma, src);
      dst = _mm_mullo_epi16(dst, alpha);
      dst = _mm_add_epi16(dst, rounder_simd);
      dst = _mm_srli_epi16(dst, 8);
      dst = _mm_add_epi8(src, dst);

      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 4), dst);
    }

    if (width != mod2_width) {
      int x = mod2_width;
      int alpha = (ovrp[x * 4 + 3] * level + 1) >> 8;

      if (use_chroma) {
        dstp[x * 4] = dstp[x * 4] + (((255 - ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((255 - ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((255 - ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((255 - ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> 8);
      }
      else {
        int luma = (cyb * (255 - ovrp[x * 4]) + cyg * (255 - ovrp[x * 4 + 1]) + cyr * (255 - ovrp[x * 4 + 2])) >> 15;

        dstp[x * 4] = dstp[x * 4] + (((luma - dstp[x * 4]) * alpha + rounder) >> 8);
        dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((luma - dstp[x * 4 + 1]) * alpha + rounder) >> 8);
        dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((luma - dstp[x * 4 + 2]) * alpha + rounder) >> 8);
        dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((luma - dstp[x * 4 + 3]) * alpha + rounder) >> 8);
      }
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_rgb32_subtract_sse2<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_subtract_sse2<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#ifdef X86_32
template<bool use_chroma>
void layer_rgb32_subtract_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  __m64 zero = _mm_setzero_si64();
  __m64 rgb_coeffs = _mm_set_pi16(0, cyr, cyg, cyb);
  __m64 ff = _mm_set1_pi16(0x00FF);

  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 4));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 4));

      __m64 alpha = _mm_cvtsi32_si64((ovrp[x * 4 + 3] * level + 1) >> 8);
      alpha = _mm_shuffle_pi16(alpha, _MM_SHUFFLE(0, 0, 0, 0));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      __m64 luma;
      if (use_chroma) {
        luma = _mm_subs_pi16(ff, ovr);
      }
      else {
        luma = calculate_luma_isse(_mm_andnot_si64(ovr, ff), rgb_coeffs, zero);
      }

      __m64 dst = _mm_subs_pi16(luma, src);
      dst = _mm_mullo_pi16(dst, alpha);
      dst = _mm_add_pi16(dst, rounder_simd);
      dst = _mm_srli_pi16(dst, 8);
      dst = _mm_add_pi8(src, dst);

      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 4) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_rgb32_subtract_isse<false>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);
template void layer_rgb32_subtract_isse<true>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level);

#endif



template<int mode>
void layer_rgb32_lighten_darken_sse2(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh) {
  int mod2_width = width / 2 * 2;

  __m128i zero = _mm_setzero_si128();
  __m128i level_vector = _mm_set1_epi32(level);
  __m128i one = _mm_set1_epi32(1);
  __m128i rgb_coeffs = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);
  __m128i threshold = _mm_set1_epi16(thresh);

  constexpr int rounder = 128;
  const __m128i rounder_simd = _mm_set1_epi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod2_width; x += 2) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(dstp + x * 4));
      __m128i ovr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ovrp + x * 4));

      __m128i alpha = calculate_monochrome_alpha_sse2(ovr, level_vector, one);

      src = _mm_unpacklo_epi8(src, zero);
      ovr = _mm_unpacklo_epi8(ovr, zero);

      __m128i luma_ovr = calculate_luma_sse2(ovr, rgb_coeffs, zero);
      __m128i luma_src = calculate_luma_sse2(src, rgb_coeffs, zero);

      __m128i mask;
      if constexpr (mode == LIGHTEN) {
        __m128i tmp = _mm_add_epi16(luma_src, threshold);
        mask = _mm_cmpgt_epi16(luma_ovr, tmp);
      }
      else {
        __m128i tmp = _mm_sub_epi16(luma_src, threshold);
        mask = _mm_cmpgt_epi16(tmp, luma_ovr);
      }

      alpha = _mm_and_si128(alpha, mask);

      __m128i dst = _mm_subs_epi16(ovr, src);
      dst = _mm_mullo_epi16(dst, alpha);
      dst = _mm_add_epi16(dst, rounder_simd);
      dst = _mm_srli_epi16(dst, 8);
      dst = _mm_add_epi8(src, dst);

      dst = _mm_packus_epi16(dst, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 4), dst);
    }

    if (width != mod2_width) {
      int x = mod2_width;
      int alpha = (ovrp[x * 4 + 3] * level + 1) >> 8;
      int luma_ovr = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;
      int luma_src = (cyb * dstp[x * 4] + cyg * dstp[x * 4 + 1] + cyr * dstp[x * 4 + 2]) >> 15;

      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;

      dstp[x * 4] = dstp[x * 4] + (((ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> 8);
      dstp[x * 4 + 1] = dstp[x * 4 + 1] + (((ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> 8);
      dstp[x * 4 + 2] = dstp[x * 4 + 2] + (((ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> 8);
      dstp[x * 4 + 3] = dstp[x * 4 + 3] + (((ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> 8);
    }

    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

// instantiate
template void layer_rgb32_lighten_darken_sse2<LIGHTEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template void layer_rgb32_lighten_darken_sse2<DARKEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);

#ifdef X86_32
template<int mode>
void layer_rgb32_lighten_darken_isse(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh) {
  __m64 zero = _mm_setzero_si64();
  __m64 rgb_coeffs = _mm_set_pi16(0, cyr, cyg, cyb);
  __m64 threshold = _mm_set1_pi16(thresh);

  constexpr int rounder = 128;
  const __m64 rounder_simd = _mm_set1_pi16(rounder);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(dstp + x * 4));
      __m64 ovr = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(ovrp + x * 4));

      __m64 alpha = _mm_cvtsi32_si64((ovrp[x * 4 + 3] * level + 1) >> 8);
      alpha = _mm_shuffle_pi16(alpha, _MM_SHUFFLE(0, 0, 0, 0));

      src = _mm_unpacklo_pi8(src, zero);
      ovr = _mm_unpacklo_pi8(ovr, zero);

      __m64 luma_ovr = calculate_luma_isse(ovr, rgb_coeffs, zero);
      __m64 luma_src = calculate_luma_isse(src, rgb_coeffs, zero);

      /*
      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;
      */

      __m64 mask;
      if (mode == LIGHTEN) {
        __m64 tmp = _mm_add_pi16(luma_src, threshold);
        mask = _mm_cmpgt_pi16(luma_ovr, tmp);
      }
      else {
        __m64 tmp = _mm_sub_pi16(luma_src, threshold);
        mask = _mm_cmpgt_pi16(tmp, luma_ovr);
      }

      alpha = _mm_and_si64(alpha, mask);

      __m64 dst = _mm_subs_pi16(ovr, src);
      dst = _mm_mullo_pi16(dst, alpha);
      dst = _mm_add_pi16(dst, rounder_simd);
      dst = _mm_srli_pi16(dst, 8);
      dst = _mm_add_pi8(src, dst);

      dst = _mm_packs_pu16(dst, zero);

      *reinterpret_cast<int*>(dstp + x * 4) = _mm_cvtsi64_si32(dst);
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
  _mm_empty();
}

// instantiate
template void layer_rgb32_lighten_darken_isse<LIGHTEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);
template void layer_rgb32_lighten_darken_isse<DARKEN>(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh);

#endif

