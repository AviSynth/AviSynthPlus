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

// ConvertPlanar (c) 2005 by Klaus Post


#include <avs/alignment.h>
#ifdef AVS_WINDOWS
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#include "convert_planar_avx2.h"

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_planarrgb_to_yuv_uint16_avx2(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m)
{
  // generic for 10-16 bit uint16 but only used for 16 bits where unsigned 16 arithmetic makes things difficult
  // 16 bit uint16_t (unsigned range)
  __m256  half_f = _mm256_set1_ps((float)(1u << (bits_per_pixel - 1)));
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m256 offset_f = _mm256_set1_ps(m.offset_y_f);

  //__m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 8 * sizeof(uint16_t)) {
      __m256 g, b, r;
      // uint16_t: load 16 bytes: 8 pixels

      __m128i gi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[0] + x));
      __m128i bi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[1] + x));
      __m128i ri = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[2] + x));
      g = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(gi));
      b = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bi));
      r = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(ri));
      /*
      int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384)>>15);
      int U = half + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
      int V = half + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      */
      // *Y*
      {
        auto mat_r = _mm256_set1_ps(m.y_r_f);
        auto mat_g = _mm256_set1_ps(m.y_g_f);
        auto mat_b = _mm256_set1_ps(m.y_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, offset_f)));
        __m256i yi = _mm256_cvtps_epi32(y);
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[0] + x), res);
      }
      // *U*
      {
        auto mat_r = _mm256_set1_ps(m.u_r_f);
        auto mat_g = _mm256_set1_ps(m.u_g_f);
        auto mat_b = _mm256_set1_ps(m.u_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, half_f)));
        __m256i yi = _mm256_cvtps_epi32(y);
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[1] + x), res);
      }
      // *V*
      {
        auto mat_r = _mm256_set1_ps(m.v_r_f);
        auto mat_g = _mm256_set1_ps(m.v_g_f);
        auto mat_b = _mm256_set1_ps(m.v_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, half_f)));
        __m256i yi = _mm256_cvtps_epi32(y);
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[2] + x), res);
      }
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

// Instantiate them
template void convert_planarrgb_to_yuv_uint16_avx2<10>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<12>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<14>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
