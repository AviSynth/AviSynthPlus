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



#include "greyscale_sse.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include <avs/config.h>
#include <avs/types.h>
#include "avs/minmax.h"

void greyscale_yuy2_sse2(BYTE *srcp, size_t /*width*/, size_t height, size_t pitch) {
  __m128i luma_mask = _mm_set1_epi16(0x00FF);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i chroma_value = _mm_set1_epi16(0x8000);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  BYTE* end_point = srcp + pitch * height;

  while(srcp < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    src = _mm_and_si128(src, luma_mask);
    src = _mm_or_si128(src, chroma_value);
    _mm_store_si128(reinterpret_cast<__m128i*>(srcp), src);

    srcp += 16;
  }
}

void greyscale_rgb32_sse2(BYTE *srcp, size_t /*width*/, size_t height, size_t pitch, int cyb, int cyg, int cyr) {
  __m128i matrix = _mm_set_epi16(0, cyr, cyg, cyb, 0, cyr, cyg, cyb);
  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(16384);
  __m128i alpha_mask = _mm_set1_epi32(0xFF000000);

  BYTE* end_point = srcp + pitch * height;

  while(srcp < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    __m128i alpha = _mm_and_si128(src, alpha_mask);
    __m128i pixel01 = _mm_unpacklo_epi8(src, zero);
    __m128i pixel23 = _mm_unpackhi_epi8(src, zero);

    pixel01 = _mm_madd_epi16(pixel01, matrix);
    pixel23 = _mm_madd_epi16(pixel23, matrix);

    __m128i tmp = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01), _mm_castsi128_ps(pixel23), _MM_SHUFFLE(3, 1, 3, 1))); // r3*cyr | r2*cyr | r1*cyr | r0*cyr
    __m128i tmp2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01), _mm_castsi128_ps(pixel23), _MM_SHUFFLE(2, 0, 2, 0)));

    tmp = _mm_add_epi32(tmp, tmp2);
    tmp = _mm_add_epi32(tmp, round_mask);
    tmp = _mm_srli_epi32(tmp, 15); // 0 0 0 p3 | 0 0 0 p2 | 0 0 0 p1 | 0 0 0 p0

    //todo: pshufb?
    __m128i result = _mm_or_si128(tmp, _mm_slli_si128(tmp, 1));
    result = _mm_or_si128(result, _mm_slli_si128(tmp, 2));
    result = _mm_or_si128(alpha, result);

    _mm_store_si128(reinterpret_cast<__m128i*>(srcp), result);

    srcp += 16;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void greyscale_rgb64_sse41(BYTE *srcp, size_t /*width*/, size_t height, size_t pitch, int cyb, int cyg, int cyr)
{
  __m128i matrix = _mm_set_epi32(0, cyr, cyg, cyb);
  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(16384);
  uint64_t mask64 = 0xFFFF000000000000ull;
  __m128i alpha_mask  = _mm_set_epi32((uint32_t)(mask64 >> 32),(uint32_t)mask64,(uint32_t)(mask64 >> 32),(uint32_t)mask64);

  BYTE* end_point = srcp + pitch * height;

  while(srcp < end_point) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp)); // 2x64bit pixels

    __m128i srclo = _mm_unpacklo_epi16(src, zero); // pixel1
    __m128i mullo = _mm_mullo_epi32(srclo, matrix); // 0, mul_r1, mul_g1, mul_b1 // sse41

    __m128i srchi = _mm_unpackhi_epi16(src, zero); // pixel2
    __m128i mulhi = _mm_mullo_epi32(srchi, matrix); // 0, mul_r2, mul_g2, mul_b2 // sse41

    __m128i alpha = _mm_and_si128(src, alpha_mask); // put back later

    // ssse3
    __m128i result = _mm_hadd_epi32(mullo, mulhi);  // 0+mul_r1 | mul_g1+mul_b1 | 0+mul_r2 | mul_g2+mul_b2
    result = _mm_hadd_epi32(result, zero);  // 0+mul_r1+mul_g1+mul_b1 | 0+mul_r2+mul_g2+mul_b2 | 0 | 0

    result = _mm_add_epi32(result, round_mask);
    result = _mm_srli_epi32(result, 15);
    // we have the greyscale value of two pixels as int32  0 0 | 0 0 | 0 p1 | 0 p0
    // we need 0 p1 p1 p1 0 p0 p0 p0

    __m128i result1 = _mm_or_si128(_mm_slli_si128(result, 2), result);
    // 0 0  | 0 0   | p1 p1 | p0 p0
    result = _mm_unpacklo_epi32(result1, result);
    // 0 p1 | p1 p1 | 0  p0 | p0 p0

    result = _mm_or_si128(alpha, result); // put back initial alpha

    _mm_store_si128(reinterpret_cast<__m128i*>(srcp), result);

    srcp += 16;
  }
}

#ifdef X86_32
void greyscale_yuy2_mmx(BYTE *srcp, size_t width, size_t height, size_t pitch) {
  bool not_mod8 = false;
  size_t loop_limit = min((pitch / 8) * 8, ((width*4 + 7) / 8) * 8);

  __m64 luma_mask = _mm_set1_pi16(0x00FF);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m64 chroma_value = _mm_set1_pi16(0x8000);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < loop_limit; x+=8) {
     __m64 src = *reinterpret_cast<const __m64*>(srcp+x);
     src = _mm_and_si64(src, luma_mask);
     src = _mm_or_si64(src, chroma_value);
     *reinterpret_cast<__m64*>(srcp+x) = src;
    }

    if (loop_limit < width) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp+width-8);
      src = _mm_and_si64(src, luma_mask);
      src = _mm_or_si64(src, chroma_value);
      *reinterpret_cast<__m64*>(srcp+width-8) = src;
    }

    srcp += pitch;
  }
 _mm_empty();
}

static AVS_FORCEINLINE __m64 greyscale_rgb32_core_mmx(__m64 &src, __m64 &alpha_mask, __m64 &zero, __m64 &matrix, __m64 &round_mask) {
  __m64 alpha = _mm_and_si64(src, alpha_mask);
  __m64 pixel0 = _mm_unpacklo_pi8(src, zero);
  __m64 pixel1 = _mm_unpackhi_pi8(src, zero);

  pixel0 = _mm_madd_pi16(pixel0, matrix); //a0*0 + r0*cyr | g0*cyg + b0*cyb
  pixel1 = _mm_madd_pi16(pixel1, matrix); //a1*0 + r1*cyr | g1*cyg + b1*cyb

  __m64 tmp = _mm_unpackhi_pi32(pixel0, pixel1); // r1*cyr | r0*cyr
  __m64 tmp2 = _mm_unpacklo_pi32(pixel0, pixel1); // g1*cyg + b1*cyb | g0*cyg + b0*cyb

  tmp = _mm_add_pi32(tmp, tmp2); // r1*cyr + g1*cyg + b1*cyb | r0*cyr + g0*cyg + b0*cyb
  tmp = _mm_add_pi32(tmp, round_mask); // r1*cyr + g1*cyg + b1*cyb + 32768 | r0*cyr + g0*cyg + b0*cyb + 32768
  tmp = _mm_srli_pi32(tmp, 15); // 0 0 0 p2 | 0 0 0 p1

  __m64 shifted = _mm_slli_si64(tmp, 8);
  tmp = _mm_or_si64(tmp, shifted); // 0 0 p2 p2 | 0 0 p1 p1
  tmp = _mm_or_si64(tmp, _mm_slli_si64(shifted, 8)); // 0 p2 p2 p2 | 0 p1 p1 p1
  return _mm_or_si64(tmp, alpha);
}

void greyscale_rgb32_mmx(BYTE *srcp, size_t width, size_t height, size_t pitch, int cyb, int cyg, int cyr) {
  __m64 matrix = _mm_set_pi16(0, cyr, cyg, cyb);
  __m64 zero = _mm_setzero_si64();
  __m64 round_mask = _mm_set1_pi32(16384);
  __m64 alpha_mask = _mm_set1_pi32(0xFF000000);

  size_t loop_limit = min((pitch / 8) * 8, ((width*4 + 7) / 8) * 8);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < loop_limit; x+=8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp+x); //pixels 0 and 1
      __m64 result = greyscale_rgb32_core_mmx(src, alpha_mask, zero, matrix, round_mask);

      *reinterpret_cast<__m64*>(srcp+x) = result;
    }

    if (loop_limit < width) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp+width-8); //pixels 0 and 1
      __m64 result = greyscale_rgb32_core_mmx(src, alpha_mask, zero, matrix, round_mask);

      *reinterpret_cast<__m64*>(srcp+width-8) = result;
    }

    srcp += pitch;
  }
  _mm_empty();
}
#endif


