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


#include "../convert.h"
#include "../convert_matrix.h"
#include "../convert_planar.h"
#include "../convert_bits.h"
#include "convert_planar_avx2.h"
#include "../filters/intel/resample_sse.h"
#include "../filters/intel/planeswap_sse.h"
#include "../filters/field.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/alignment.h>
#include <tmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <algorithm>
#include <string>



void convert_yuy2_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  __m128i luma_mask = _mm_set1_epi16(0xFF);

  for(size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 16) {
      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2));
      __m128i src2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2+16));
      src1 = _mm_and_si128(src1, luma_mask);
      src2 = _mm_and_si128(src2, luma_mask);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), _mm_packus_epi16(src1, src2));
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#ifdef X86_32
void convert_yuy2_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  __m64 luma_mask = _mm_set1_pi16(0xFF);

  for(size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 8) {
      __m64 src1 = *reinterpret_cast<const __m64*>(srcp+x*2);
      __m64 src2 = *reinterpret_cast<const __m64*>(srcp+x*2+8);
      src1 = _mm_and_si64(src1, luma_mask);
      src2 = _mm_and_si64(src2, luma_mask);
      *reinterpret_cast<__m64*>(dstp+x) = _mm_packs_pu16(src1, src2);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}
#endif


static AVS_FORCEINLINE __m128i convert_rgb_to_y8_sse2_core(const __m128i &pixel01, const __m128i &pixel23, const __m128i &pixel45, const __m128i &pixel67, __m128i& zero, __m128i &matrix, __m128i &round_mask, __m128i &offset) {
  //int Y = offset_y + ((m0 * srcp[0] + m1 * srcp[1] + m2 * srcp[2] + 16384) >> 15);
  // in general the algorithm is identical to MMX version, the only different part is getting r and g+b in appropriate registers. We use shuffling instead of unpacking here.
  __m128i pixel01m = _mm_madd_epi16(pixel01, matrix); //a1*0 + r1*cyr | g1*cyg + b1*cyb | a0*0 + r0*cyr | g0*cyg + b0*cyb
  __m128i pixel23m = _mm_madd_epi16(pixel23, matrix); //a3*0 + r3*cyr | g3*cyg + b3*cyb | a2*0 + r2*cyr | g2*cyg + b2*cyb
  __m128i pixel45m = _mm_madd_epi16(pixel45, matrix); //a5*0 + r5*cyr | g5*cyg + b5*cyb | a4*0 + r4*cyr | g4*cyg + b4*cyb
  __m128i pixel67m = _mm_madd_epi16(pixel67, matrix); //a7*0 + r7*cyr | g7*cyg + b7*cyb | a6*0 + r6*cyr | g6*cyg + b6*cyb

  __m128i pixel_0123_r = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01m), _mm_castsi128_ps(pixel23m), _MM_SHUFFLE(3, 1, 3, 1))); // r3*cyr | r2*cyr | r1*cyr | r0*cyr
  __m128i pixel_4567_r = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel45m), _mm_castsi128_ps(pixel67m), _MM_SHUFFLE(3, 1, 3, 1))); // r7*cyr | r6*cyr | r5*cyr | r4*cyr

  __m128i pixel_0123 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel01m), _mm_castsi128_ps(pixel23m), _MM_SHUFFLE(2, 0, 2, 0)));
  __m128i pixel_4567 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(pixel45m), _mm_castsi128_ps(pixel67m), _MM_SHUFFLE(2, 0, 2, 0)));

  pixel_0123 = _mm_add_epi32(pixel_0123, pixel_0123_r);
  pixel_4567 = _mm_add_epi32(pixel_4567, pixel_4567_r);

  pixel_0123 = _mm_add_epi32(pixel_0123, round_mask);
  pixel_4567 = _mm_add_epi32(pixel_4567, round_mask);

  pixel_0123 = _mm_srai_epi32(pixel_0123, 15);
  pixel_4567 = _mm_srai_epi32(pixel_4567, 15);

  __m128i result = _mm_packs_epi32(pixel_0123, pixel_4567);

  result = _mm_adds_epi16(result, offset);
  result = _mm_packus_epi16(result, zero);

  return result;
}

void convert_rgb32_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  __m128i matrix_v = _mm_set_epi16(0, matrix.y_r, matrix.y_g, matrix.y_b, 0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m128i zero = _mm_setzero_si128();
  __m128i offset = _mm_set1_epi16(matrix.offset_y);
  __m128i round_mask = _mm_set1_epi32(16384);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x+=8) {
      __m128i src0123 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4)); //pixels 0, 1, 2 and 3
      __m128i src4567 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+16));//pixels 4, 5, 6 and 7

      __m128i pixel01 = _mm_unpacklo_epi8(src0123, zero);
      __m128i pixel23 = _mm_unpackhi_epi8(src0123, zero);
      __m128i pixel45 = _mm_unpacklo_epi8(src4567, zero);
      __m128i pixel67 = _mm_unpackhi_epi8(src4567, zero);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_v, round_mask, offset));
    }

    srcp -= src_pitch;
    dstp += dst_pitch;
  }
}


void convert_rgb24_to_y8_sse2(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  __m128i matrix_v = _mm_set_epi16(0, matrix.y_r, matrix.y_g, matrix.y_b, 0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m128i zero = _mm_setzero_si128();
  __m128i offset = _mm_set1_epi16(matrix.offset_y);
  __m128i round_mask = _mm_set1_epi32(16384);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x+=8) {
      __m128i pixel01 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3)); //pixels 0 and 1
      __m128i pixel23 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+6)); //pixels 2 and 3
      __m128i pixel45 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+12)); //pixels 4 and 5
      __m128i pixel67 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+18)); //pixels 6 and 7


      //0 0 0 0 0 0 0 0 | x x r1 g1 b1 r0 g0 b0  -> 0 x 0 x 0 r1 0 g1 | 0 b1 0 r0 0 g0 0 b0 -> 0 r1 0 g1 0 b1 0 r0 | 0 b1 0 r0 0 g0 0 b0 -> 0 r1 0 r1 0 g1 0 b1 | 0 b1 0 r0 0 g0 0 b0
      pixel01 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel01, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel23 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel23, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel45 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel45, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel67 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel67, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_v, round_mask, offset));
    }

    srcp -= src_pitch;
    dstp += dst_pitch;
  }
}


#ifdef X86_32

#pragma warning(push)
#pragma warning(disable: 4799)
static AVS_FORCEINLINE int convert_rgb_to_y8_mmx_core(const __m64 &pixel0, const __m64 &pixel1, const __m64 &pixel2, const __m64 &pixel3, __m64& zero, __m64 &matrix, __m64 &round_mask, __m64 &offset) {
  //int Y = offset_y + ((m0 * srcp[0] + m1 * srcp[1] + m2 * srcp[2] + 16384) >> 15);

  __m64 pixel0m = _mm_madd_pi16(pixel0, matrix); //a0*0 + r0*cyr | g0*cyg + b0*cyb
  __m64 pixel1m = _mm_madd_pi16(pixel1, matrix); //a1*0 + r1*cyr | g1*cyg + b1*cyb
  __m64 pixel2m = _mm_madd_pi16(pixel2, matrix); //a2*0 + r2*cyr | g2*cyg + b2*cyb
  __m64 pixel3m = _mm_madd_pi16(pixel3, matrix); //a3*0 + r3*cyr | g3*cyg + b3*cyb

  __m64 pixel_01_r = _mm_unpackhi_pi32(pixel0m, pixel1m); // r1*cyr | r0*cyr
  __m64 pixel_23_r = _mm_unpackhi_pi32(pixel2m, pixel3m); // r3*cyr | r2*cyr

  __m64 pixel_01 = _mm_unpacklo_pi32(pixel0m, pixel1m); //g1*cyg + b1*cyb | g0*cyg + b0*cyb
  __m64 pixel_23 = _mm_unpacklo_pi32(pixel2m, pixel3m); //g3*cyg + b3*cyb | g2*cyg + b2*cyb

  pixel_01 = _mm_add_pi32(pixel_01, pixel_01_r); // r1*cyr + g1*cyg + b1*cyb | r0*cyr + g0*cyg + b0*cyb
  pixel_23 = _mm_add_pi32(pixel_23, pixel_23_r); // r3*cyr + g3*cyg + b3*cyb | r2*cyr + g2*cyg + b2*cyb

  pixel_01 = _mm_add_pi32(pixel_01, round_mask); //r1*cyr + g1*cyg + b1*cyb + 16384 | r0*cyr + g0*cyg + b0*cyb + 16384
  pixel_23 = _mm_add_pi32(pixel_23, round_mask); //r3*cyr + g3*cyg + b3*cyb + 16384 | r2*cyr + g2*cyg + b2*cyb + 16384

  pixel_01 = _mm_srai_pi32(pixel_01, 15); //0 | p1 | 0 | p0
  pixel_23 = _mm_srai_pi32(pixel_23, 15); //0 | p3 | 0 | p2

  __m64 result = _mm_packs_pi32(pixel_01, pixel_23); //p3 | p2 | p1 | p0

  result = _mm_adds_pi16(result, offset);
  result = _mm_packs_pu16(result, zero); //0 0 0 0 p3 p2 p1 p0

  return _mm_cvtsi64_si32(result);
}
#pragma warning(pop)

void convert_rgb32_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  __m64 matrix_v = _mm_set_pi16(0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m64 zero = _mm_setzero_si64();
  __m64 offset = _mm_set1_pi16(matrix.offset_y);
  __m64 round_mask = _mm_set1_pi32(16384);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x+=4) {
      __m64 src01 = *reinterpret_cast<const __m64*>(srcp+x*4); //pixels 0 and 1
      __m64 src23 = *reinterpret_cast<const __m64*>(srcp+x*4+8);//pixels 2 and 3

      __m64 pixel0 = _mm_unpacklo_pi8(src01, zero); //a0 r0 g0 b0
      __m64 pixel1 = _mm_unpackhi_pi8(src01, zero); //a1 r1 g1 b1
      __m64 pixel2 = _mm_unpacklo_pi8(src23, zero); //a2 r2 g2 b2
      __m64 pixel3 = _mm_unpackhi_pi8(src23, zero); //a3 r3 g3 b3

      *reinterpret_cast<int*>(dstp+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_v, round_mask, offset);
    }

    srcp -= src_pitch;
    dstp += dst_pitch;
  }
  _mm_empty();
}


void convert_rgb24_to_y8_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  __m64 matrix_v = _mm_set_pi16(0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m64 zero = _mm_setzero_si64();
  __m64 offset = _mm_set1_pi16(matrix.offset_y);
  __m64 round_mask = _mm_set1_pi32(16384);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 4) {
      __m64 pixel0 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3)); //pixel 0
      __m64 pixel1 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+3)); //pixel 1
      __m64 pixel2 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+6)); //pixel 2
      __m64 pixel3 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+9)); //pixel 3

      pixel0 = _mm_unpacklo_pi8(pixel0, zero);
      pixel1 = _mm_unpacklo_pi8(pixel1, zero);
      pixel2 = _mm_unpacklo_pi8(pixel2, zero);
      pixel3 = _mm_unpacklo_pi8(pixel3, zero);

      *reinterpret_cast<int*>(dstp+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_v, round_mask, offset);
    }

    srcp -= src_pitch;
    dstp += dst_pitch;
  }
  _mm_empty();
}

#endif // X86_32




void convert_rgb32_to_yv24_sse2(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  srcp += src_pitch * (height-1);

  __m128i matrix_y = _mm_set_epi16(0, matrix.y_r, matrix.y_g, matrix.y_b, 0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m128i matrix_u = _mm_set_epi16(0, matrix.u_r, matrix.u_g, matrix.u_b, 0, matrix.u_r, matrix.u_g, matrix.u_b);
  __m128i matrix_v = _mm_set_epi16(0, matrix.v_r, matrix.v_g, matrix.v_b, 0, matrix.v_r, matrix.v_g, matrix.v_b);

  __m128i zero = _mm_setzero_si128();
  __m128i offset = _mm_set1_epi16(matrix.offset_y);
  __m128i round_mask = _mm_set1_epi32(16384);
  __m128i v128 = _mm_set1_epi16(128);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 8) {
      __m128i src0123 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4)); //pixels 0, 1, 2 and 3
      __m128i src4567 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+16));//pixels 4, 5, 6 and 7

      __m128i pixel01 = _mm_unpacklo_epi8(src0123, zero);
      __m128i pixel23 = _mm_unpackhi_epi8(src0123, zero);
      __m128i pixel45 = _mm_unpacklo_epi8(src4567, zero);
      __m128i pixel67 = _mm_unpackhi_epi8(src4567, zero);

      __m128i result_y = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_y, round_mask, offset);
      __m128i result_u = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_u, round_mask, v128);
      __m128i result_v = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_v, round_mask, v128);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstY+x), result_y);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstU+x), result_u);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstV+x), result_v);
    }

    srcp -= src_pitch;
    dstY += dst_pitch_y;
    dstU += UVpitch;
    dstV += UVpitch;
  }
}

void convert_rgb24_to_yv24_sse2(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  srcp += src_pitch * (height-1);

  size_t mod8_width = width / 8 * 8;

  __m128i matrix_y = _mm_set_epi16(0, matrix.y_r, matrix.y_g, matrix.y_b, 0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m128i matrix_u = _mm_set_epi16(0, matrix.u_r, matrix.u_g, matrix.u_b, 0, matrix.u_r, matrix.u_g, matrix.u_b);
  __m128i matrix_v = _mm_set_epi16(0, matrix.v_r, matrix.v_g, matrix.v_b, 0, matrix.v_r, matrix.v_g, matrix.v_b);

  __m128i zero = _mm_setzero_si128();
  __m128i offset = _mm_set1_epi16(matrix.offset_y);
  __m128i round_mask = _mm_set1_epi32(16384);
  __m128i v128 = _mm_set1_epi16(128);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m128i pixel01 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3)); //pixels 0 and 1
      __m128i pixel23 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+6)); //pixels 2 and 3
      __m128i pixel45 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+12)); //pixels 4 and 5
      __m128i pixel67 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x*3+18)); //pixels 6 and 7

      //0 0 0 0 0 0 0 0 | x x r1 g1 b1 r0 g0 b0  -> 0 x 0 x 0 r1 0 g1 | 0 b1 0 r0 0 g0 0 b0 -> 0 r1 0 g1 0 b1 0 r0 | 0 b1 0 r0 0 g0 0 b0 -> 0 r1 0 r1 0 g1 0 b1 | 0 b1 0 r0 0 g0 0 b0
      pixel01 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel01, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel23 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel23, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel45 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel45, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel67 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel67, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));

      __m128i result_y = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_y, round_mask, offset);
      __m128i result_u = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_u, round_mask, v128);
      __m128i result_v = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_v, round_mask, v128);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstY+x), result_y);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstU+x), result_u);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstV+x), result_v);
    }

    if (mod8_width != width) {
      __m128i pixel01 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+width*3-24)); //pixels 0 and 1
      __m128i pixel23 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+width*3-18)); //pixels 2 and 3
      __m128i pixel45 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+width*3-12)); //pixels 4 and 5
      __m128i pixel67 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+width*3-6)); //pixels 6 and 7

      pixel01 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel01, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel23 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel23, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel45 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel45, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));
      pixel67 = _mm_shufflehi_epi16(_mm_shuffle_epi32(_mm_unpacklo_epi8(pixel67, zero), _MM_SHUFFLE(2, 1, 1, 0)), _MM_SHUFFLE(0, 3, 2, 1));

      __m128i result_y = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_y, round_mask, offset);
      __m128i result_u = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_u, round_mask, v128);
      __m128i result_v = convert_rgb_to_y8_sse2_core(pixel01, pixel23, pixel45, pixel67, zero, matrix_v, round_mask, v128);

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstY+width-8), result_y);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstU+width-8), result_u);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstV+width-8), result_v);
    }

    srcp -= src_pitch;
    dstY += dst_pitch_y;
    dstU += UVpitch;
    dstV += UVpitch;
  }
}

#ifdef X86_32

void convert_rgb32_to_yv24_mmx(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix& matrix) {
  srcp += src_pitch * (height-1);

  __m64 matrix_y = _mm_set_pi16(0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m64 matrix_u = _mm_set_pi16(0, matrix.u_r, matrix.u_g, matrix.u_b);
  __m64 matrix_v = _mm_set_pi16(0, matrix.v_r, matrix.v_g, matrix.v_b);

  __m64 zero = _mm_setzero_si64();
  __m64 offset = _mm_set1_pi16(matrix.offset_y);
  __m64 round_mask = _mm_set1_pi32(16384);
  __m64 v128 = _mm_set1_pi16(128);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 4) {
      __m64 src01 = *reinterpret_cast<const __m64*>(srcp+x*4); //pixels 0 and 1
      __m64 src23 = *reinterpret_cast<const __m64*>(srcp+x*4+8);//pixels 2 and 3

      __m64 pixel0 = _mm_unpacklo_pi8(src01, zero); //a0 r0 g0 b0
      __m64 pixel1 = _mm_unpackhi_pi8(src01, zero); //a1 r1 g1 b1
      __m64 pixel2 = _mm_unpacklo_pi8(src23, zero); //a2 r2 g2 b2
      __m64 pixel3 = _mm_unpackhi_pi8(src23, zero); //a3 r3 g3 b3

      *reinterpret_cast<int*>(dstY+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_y, round_mask, offset);
      *reinterpret_cast<int*>(dstU+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_u, round_mask, v128);
      *reinterpret_cast<int*>(dstV+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_v, round_mask, v128);
    }

    srcp -= src_pitch;
    dstY += dst_pitch_y;
    dstU += UVpitch;
    dstV += UVpitch;
  }
  _mm_empty();
}

void convert_rgb24_to_yv24_mmx(BYTE* dstY, BYTE* dstU, BYTE* dstV, const BYTE*srcp, size_t dst_pitch_y, size_t UVpitch, size_t src_pitch, size_t width, size_t height, const ConversionMatrix &matrix) {
  srcp += src_pitch * (height-1);

  size_t mod4_width = width / 4 * 4;

  __m64 matrix_y = _mm_set_pi16(0, matrix.y_r, matrix.y_g, matrix.y_b);
  __m64 matrix_u = _mm_set_pi16(0, matrix.u_r, matrix.u_g, matrix.u_b);
  __m64 matrix_v = _mm_set_pi16(0, matrix.v_r, matrix.v_g, matrix.v_b);

  __m64 zero = _mm_setzero_si64();
  __m64 offset = _mm_set1_pi16(matrix.offset_y);
  __m64 round_mask = _mm_set1_pi32(16384);
  __m64 v128 = _mm_set1_pi16(128);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod4_width; x+=4) {
      __m64 pixel0 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3)); //pixel 0
      __m64 pixel1 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+3)); //pixel 1
      __m64 pixel2 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+6)); //pixel 2
      __m64 pixel3 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+9)); //pixel 3

      pixel0 = _mm_unpacklo_pi8(pixel0, zero);
      pixel1 = _mm_unpacklo_pi8(pixel1, zero);
      pixel2 = _mm_unpacklo_pi8(pixel2, zero);
      pixel3 = _mm_unpacklo_pi8(pixel3, zero);

      *reinterpret_cast<int*>(dstY+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_y, round_mask, offset);
      *reinterpret_cast<int*>(dstU+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_u, round_mask, v128);
      *reinterpret_cast<int*>(dstV+x) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_v, round_mask, v128);
    }

    if (mod4_width != width) {
      __m64 pixel0 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+width*3-12)); //pixel 0
      __m64 pixel1 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+width*3-9)); //pixel 1
      __m64 pixel2 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+width*3-6)); //pixel 2
      __m64 pixel3 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+width*3-3)); //pixel 3

      pixel0 = _mm_unpacklo_pi8(pixel0, zero);
      pixel1 = _mm_unpacklo_pi8(pixel1, zero);
      pixel2 = _mm_unpacklo_pi8(pixel2, zero);
      pixel3 = _mm_unpacklo_pi8(pixel3, zero);

      *reinterpret_cast<int*>(dstY+width-4) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_y, round_mask, offset);
      *reinterpret_cast<int*>(dstU+width-4) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_u, round_mask, v128);
      *reinterpret_cast<int*>(dstV+width-4) = convert_rgb_to_y8_mmx_core(pixel0, pixel1, pixel2, pixel3, zero, matrix_v, round_mask, v128);
    }

    srcp -= src_pitch;
    dstY += dst_pitch_y;
    dstU += UVpitch;
    dstV += UVpitch;
  }
  _mm_empty();
}

#endif

template<typename pixel_t, int bits_per_pixel>
void convert_planarrgb_to_yuv_uint8_14_sse2(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 8 bit        uint8_t
  // 10,12,14 bit uint16_t (signed range)
  __m128i half = _mm_set1_epi16((short)(1 << (bits_per_pixel - 1)));  // 128
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128i offset = _mm_set1_epi16((short)m.offset_y);

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(pixel_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 8 * sizeof(pixel_t)) {
      __m128i res1, res2;
      __m128i m_bg, m_rR;
      __m128i bg0123, bg4567;
      __m128i ar0123, ar4567;
      __m128i g, b, r;
      // cant handle 16 at a time, only 2x4 8bits pixels (4x32_mul_result=128 bit)
      if constexpr(sizeof(pixel_t) == 1) {
        g = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x)), zero);
        b = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x)), zero);
        r = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x)), zero);
      }
      else { // uint16_t pixels, 14 bits OK, but 16 bit pixels are unsigned, cannot madd
        g = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[0] + x));
        b = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[1] + x));
        r = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[2] + x));
      }
      // Need1:  (m.y_b   m.y_g)     (m.y_b   m.y_g)     (m.y_b   m.y_g)     (m.y_b   m.y_g)   8x16 bit
      //         (  b3     g3  )     (  b2      g2 )     (  b1      g1 )     (   b0     g0 )   8x16 bit
      // res1=  (y_b*b3 + y_g*g3)   (y_b*b2 + y_g*g2)   (y_b*b1 + y_g*g1)   (y_b*b0 + y_g*g0)  4x32 bit
      // Need2:  (m.y_r   round)     (m.y_r   round)     (m.y_r   round)     (m.y_r   round)
      //         (  r3      1  )     (  r2      1  )     (  r1      1  )     (  r0      1  )
      // res2=  (y_r*r3 + round )   (y_r*r2 + round )   (y_r*r1 + round )   (y_r*r0 + round )
      // Y result 4x32 bit = offset + ((res1 + res2) >> 15)
      // UV result 4x32 bit = half + ((res1_u_or_v + res2_u_or_v) >> 15)
      // *Y* ----------------
      m_bg = _mm_set1_epi32(int(((uint16_t)(m.y_g) << 16) | (uint16_t)m.y_b)); // green and blue
      m_rR = _mm_set1_epi32(int(((uint16_t)(16384) << 16) | (uint16_t)m.y_r)); // rounding 15 bit >> 1   and red

      bg0123 = _mm_unpacklo_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg0123);
      ar0123 = _mm_unpacklo_epi16(r, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_rR, ar0123);
      __m128i y_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2), 15);

      bg4567 = _mm_unpackhi_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg4567);
      ar4567 = _mm_unpackhi_epi16(r, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_rR, ar4567);
      __m128i y_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2), 15);

      __m128i y = _mm_add_epi16(_mm_packs_epi32(y_lo, y_hi), offset); // 2x4x32 -> 2x4xuint16_t
      if constexpr(sizeof(pixel_t) == 1) {
        y = _mm_packus_epi16(y, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0]+x), y);
      }
      else {
        y = _mm_min_epi16(y, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[0]+x), y);
      }

      // *U* ----------------
      m_bg = _mm_set1_epi32(int(((uint16_t)(m.u_g) << 16) | (uint16_t)m.u_b)); // green and blue
      m_rR = _mm_set1_epi32(int(((uint16_t)(16384) << 16) | (uint16_t)m.u_r)); // rounding 15 bit >> 1   and red

      bg0123 = _mm_unpacklo_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg0123);
      ar0123 = _mm_unpacklo_epi16(r, _mm_set1_epi16(1));
      res2   = _mm_madd_epi16(m_rR, ar0123);
      __m128i u_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2),15);

      bg4567 = _mm_unpackhi_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg4567);
      ar4567 = _mm_unpackhi_epi16(r, _mm_set1_epi16(1));
      res2   = _mm_madd_epi16(m_rR, ar4567);
      __m128i u_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2),15);

      __m128i u = _mm_add_epi16(_mm_packs_epi32(u_lo, u_hi), half); // 2x4x32 -> 2x4xuint16_t

      if constexpr(sizeof(pixel_t) == 1) {
        u = _mm_packus_epi16(u, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1]+x), u);
      }
      else {
        u = _mm_min_epi16(u, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[1]+x), u);
      }
      // *V* ----------------
      m_bg = _mm_set1_epi32(int(((uint16_t)(m.v_g) << 16) | (uint16_t)m.v_b)); // green and blue
      m_rR = _mm_set1_epi32(int(((uint16_t)(16384) << 16) | (uint16_t)m.v_r)); // rounding 15 bit >> 1   and red

      bg0123 = _mm_unpacklo_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg0123);
      ar0123 = _mm_unpacklo_epi16(r, _mm_set1_epi16(1));
      res2   = _mm_madd_epi16(m_rR, ar0123);
      __m128i v_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2),15);

      bg4567 = _mm_unpackhi_epi16(b, g);
      res1 = _mm_madd_epi16(m_bg, bg4567);
      ar4567 = _mm_unpackhi_epi16(r, _mm_set1_epi16(1));
      res2   = _mm_madd_epi16(m_rR, ar4567);
      __m128i v_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2),15);

      __m128i v = _mm_add_epi16(_mm_packs_epi32(v_lo, v_hi), half); // 2x4x32 -> 2x4xuint16_t

      if constexpr(sizeof(pixel_t) == 1) {
        v = _mm_packus_epi16(v, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2]+x), v);
      }
      else {
        v = _mm_min_epi16(v, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[2]+x), v);
      }
      /*
        int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384) >> 15);
        int U = half_i + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
        int V = half_i + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      }
      */
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<typename pixel_t, int bits_per_pixel>
template void convert_planarrgb_to_yuv_uint8_14_sse2<uint8_t, 8>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_planarrgb_to_yuv_uint8_14_sse2<uint16_t, 10>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_planarrgb_to_yuv_uint8_14_sse2<uint16_t, 12>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_planarrgb_to_yuv_uint8_14_sse2<uint16_t, 14>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);


void convert_planarrgb_to_yuv_float_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 32 bit float
  __m128 offset_f = _mm_set1_ps(m.offset_y_f);

  const int rowsize = width * sizeof(float);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(float)) {
      __m128 sum1, sum2;
      __m128 mul_r, mul_g, mul_b;
      __m128 mat_r, mat_g, mat_b;
      __m128 g, b, r;
      // float: load 16 bytes: 4 pixels
      g = _mm_load_ps(reinterpret_cast<const float *>(srcp[0] + x));
      b = _mm_load_ps(reinterpret_cast<const float *>(srcp[1] + x));
      r = _mm_load_ps(reinterpret_cast<const float *>(srcp[2] + x));
      /*
      int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384)>>15);
      int U = half + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
      int V = half + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      */
      // *Y*
      mat_r = _mm_set1_ps(m.y_r_f);
      mat_g = _mm_set1_ps(m.y_g_f);
      mat_b = _mm_set1_ps(m.y_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, offset_f);
      __m128 y = _mm_add_ps(sum1, sum2);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[0] + x), y);
      // *U*
      mat_r = _mm_set1_ps(m.u_r_f);
      mat_g = _mm_set1_ps(m.u_g_f);
      mat_b = _mm_set1_ps(m.u_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      //sum2 = _mm_add_ps(mul_b, half_f); no chroma 0.5 shift, center is 0
      __m128 u = _mm_add_ps(sum1, mul_b);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[1] + x), u);
      // *V*
      mat_r = _mm_set1_ps(m.v_r_f);
      mat_g = _mm_set1_ps(m.v_g_f);
      mat_b = _mm_set1_ps(m.v_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      //sum2 = _mm_add_ps(mul_b, half_f); no chroma 0.5 shift, center is 0
      __m128 v = _mm_add_ps(sum1, mul_b);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[2] + x), v);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_planarrgb_to_yuv_uint16_sse41(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // generic for 10-16 bit uint16 but only used for 16 bits where unsigned 16 arithmetic makes things difficult
  // 16 bit uint16_t (unsigned range)

  __m128  half_f = _mm_set1_ps((float)(1u << (bits_per_pixel - 1)));
  __m128i limit  = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128 offset_f = _mm_set1_ps(m.offset_y_f);

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(uint16_t)) {
      __m128 sum1, sum2;
      __m128 mul_r, mul_g, mul_b;
      __m128 mat_r, mat_g, mat_b;
      __m128 g, b, r;
      // uint16_t: load 8 bytes: 4 pixels
      __m128i gi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x));
      __m128i bi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x));
      __m128i ri = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x));
      g = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gi,zero));
      b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(bi,zero));
      r = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ri,zero));
      /*
      int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384)>>15);
      int U = half + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
      int V = half + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      */
      // *Y*
      mat_r = _mm_set1_ps(m.y_r_f);
      mat_g = _mm_set1_ps(m.y_g_f);
      mat_b = _mm_set1_ps(m.y_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, offset_f);
      __m128 y = _mm_add_ps(sum1, sum2);
      __m128i yi = _mm_cvtps_epi32(y); // no extra rounding, cvtps rounds to nearest
      yi = _mm_packus_epi32(yi, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        yi = _mm_min_epi16(yi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0] + x), yi);
      // *U*
      mat_r = _mm_set1_ps(m.u_r_f);
      mat_g = _mm_set1_ps(m.u_g_f);
      mat_b = _mm_set1_ps(m.u_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, half_f); // 8-16 bit U has chroma offset
      __m128 u = _mm_add_ps(sum1, sum2);
      __m128i ui = _mm_cvtps_epi32(u); // no extra rounding, cvtps rounds to nearest
      ui = _mm_packus_epi32(ui, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        ui = _mm_min_epi16(ui, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + x), ui);
      // *V*
      mat_r = _mm_set1_ps(m.v_r_f);
      mat_g = _mm_set1_ps(m.v_g_f);
      mat_b = _mm_set1_ps(m.v_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, half_f); // 8-16 bit V has chroma offset
      __m128 v = _mm_add_ps(sum1, sum2);
      __m128i vi = _mm_cvtps_epi32(v); // no extra rounding, cvtps rounds to nearest
      vi = _mm_packus_epi32(vi, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        vi = _mm_min_epi16(vi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + x), vi);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<int bits_per_pixel>
template void convert_planarrgb_to_yuv_uint16_sse41<10>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse41<12>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse41<14>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse41<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);


template<int bits_per_pixel>
void convert_planarrgb_to_yuv_uint16_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // generic for 10-16 bit uint16 but only used for 16 bits where unsigned 16 arithmetic makes things difficult
  // 16 bit uint16_t (unsigned range)

  __m128  half_f = _mm_set1_ps((float)(1u << (bits_per_pixel - 1)));
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128 offset_f = _mm_set1_ps(m.offset_y_f);

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(uint16_t)) {
      __m128 sum1, sum2;
      __m128 mul_r, mul_g, mul_b;
      __m128 mat_r, mat_g, mat_b;
      __m128 g, b, r;
      // uint16_t: load 8 bytes: 4 pixels
      __m128i gi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x));
      __m128i bi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x));
      __m128i ri = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x));
      g = _mm_cvtepi32_ps(_mm_unpacklo_epi16(gi, zero));
      b = _mm_cvtepi32_ps(_mm_unpacklo_epi16(bi, zero));
      r = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ri, zero));
      /*
      int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384)>>15);
      int U = half + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
      int V = half + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      */
      // *Y*
      mat_r = _mm_set1_ps(m.y_r_f);
      mat_g = _mm_set1_ps(m.y_g_f);
      mat_b = _mm_set1_ps(m.y_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, offset_f);
      __m128 y = _mm_add_ps(sum1, sum2);
      __m128i yi = _mm_cvtps_epi32(y); // no extra rounding, cvtps rounds to nearest
      yi = _MM_PACKUS_EPI32(yi, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        yi = _mm_min_epi16(yi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0] + x), yi);
      // *U*
      mat_r = _mm_set1_ps(m.u_r_f);
      mat_g = _mm_set1_ps(m.u_g_f);
      mat_b = _mm_set1_ps(m.u_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, half_f);
      __m128 u = _mm_add_ps(sum1, sum2);
      __m128i ui = _mm_cvtps_epi32(u); // no extra rounding, cvtps rounds to nearest
      ui = _MM_PACKUS_EPI32(ui, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        ui = _mm_min_epi16(ui, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + x), ui);
      // *V*
      mat_r = _mm_set1_ps(m.v_r_f);
      mat_g = _mm_set1_ps(m.v_g_f);
      mat_b = _mm_set1_ps(m.v_b_f);
      mul_r = _mm_mul_ps(r, mat_r);
      mul_g = _mm_mul_ps(g, mat_g);
      mul_b = _mm_mul_ps(b, mat_b);
      sum1 = _mm_add_ps(mul_r, mul_g);
      sum2 = _mm_add_ps(mul_b, half_f);
      __m128 v = _mm_add_ps(sum1, sum2);
      __m128i vi = _mm_cvtps_epi32(v); // no extra rounding, cvtps rounds to nearest
      vi = _MM_PACKUS_EPI32(vi, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        vi = _mm_min_epi16(vi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + x), vi);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<int bits_per_pixel>
template void convert_planarrgb_to_yuv_uint16_sse2<10>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse2<12>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse2<14>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_sse2<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);


template<typename pixel_t, int bits_per_pixel>
void convert_yuv_to_planarrgb_uint8_14_sse2(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 8 bit        uint8_t
  // 10,12,14 bit uint16_t (signed range)
  __m128i half = _mm_set1_epi16((short)(1 << (bits_per_pixel - 1)));  // 128
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128i offset = _mm_set1_epi16((short)(m.offset_y));

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(pixel_t);
  for (int yy = 0; yy < height; yy++) {
    // if not mod16 then still no trouble, process non-visible pixels, we have 32 byte aligned in avs+
    for (int x = 0; x < rowsize; x += 8 * sizeof(pixel_t)) {
      __m128i res1, res2;
      __m128i m_uy, m_vR; // bg=uy rR=vR, g=y b=u r=v R=rounding
      __m128i uy0123, uy4567;
      __m128i xv0123, xv4567;
      __m128i y, u, v;
      // cant handle 16 at a time, only 2x4 8bits pixels (4x32_mul_result=128 bit)
      if constexpr(sizeof(pixel_t) == 1) {
        y = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x)), zero);
        u = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x)), zero);
        v = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x)), zero);
      }
      else { // uint16_t pixels, 14 bits OK, but 16 bit pixels are unsigned, cannot madd
        y = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[0] + x));
        u = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[1] + x));
        v = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp[2] + x));
      }
      y = _mm_adds_epi16(y, offset); // offset is negative
      u = _mm_subs_epi16(u, half);
      v = _mm_subs_epi16(v, half);
      /*
      // int b = (((int64_t)matrix.y_b * Y + (int64_t)matrix.u_b * U + (int64_t)matrix.v_b * V + 4096)>>13);
      // int g = (((int64_t)matrix.y_g * Y + (int64_t)matrix.u_g * U + (int64_t)matrix.v_g * V + 4096)>>13);
      // int r = (((int64_t)matrix.y_r * Y + (int64_t)matrix.u_r * U + (int64_t)matrix.v_r * V + 4096)>>13);
      */
      // Need1:  (m.y_b   m.u_b )     (m.y_b   m.u_b)     (m.y_b   m.u_b)     (m.y_b   m.u_b)   8x16 bit
      //         (  y3      u3  )     (  y2      u2 )     (  y1      u1 )     (   y0     u0 )   8x16 bit
      // res1=  (y_b*y3 + u_b*u3)   ...                                                         4x32 bit
      // Need2:  (m.v_b   round )     (m.y_b   round)     (m.y_b   round)     (m.y_b   round)
      //         (  v3      1   )     (  v2      1  )     (  v1      1  )     (  v0      1  )
      // res2=  (yv_b*v3 + round )  ...
      // *G* ----------------
      m_uy = _mm_set1_epi32(int((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.u_g))); // y and u
      m_vR = _mm_set1_epi32(int((static_cast<uint16_t>(4096) << 16) | static_cast<uint16_t>(m.v_g))); // rounding 13 bit >> 1 and v

      uy0123 = _mm_unpacklo_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy0123);
      xv0123 = _mm_unpacklo_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv0123);
      __m128i g_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      uy4567 = _mm_unpackhi_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy4567);
      xv4567 = _mm_unpackhi_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv4567);
      __m128i g_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      __m128i g = _mm_packs_epi32(g_lo, g_hi); // 2x4x32 -> 2x4xuint16_t
      if constexpr(sizeof(pixel_t) == 1) {
        g = _mm_packus_epi16(g, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0]+x), g);
      }
      else {
        g = _mm_max_epi16(_mm_min_epi16(g, limit), zero); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[0]+x), g);
      }
      // *B* ----------------
      m_uy = _mm_set1_epi32(int((static_cast<uint16_t>(m.y_b) << 16) | static_cast<uint16_t>(m.u_b))); // y and u
      m_vR = _mm_set1_epi32(int((static_cast<uint16_t>(4096) << 16) | static_cast<uint16_t>(m.v_b))); // rounding 13 bit >> 1 and v

      uy0123 = _mm_unpacklo_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy0123);
      xv0123 = _mm_unpacklo_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv0123);
      __m128i b_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      uy4567 = _mm_unpackhi_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy4567);
      xv4567 = _mm_unpackhi_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv4567);
      __m128i b_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      __m128i b = _mm_packs_epi32(b_lo, b_hi); // 2x4x32 -> 2x4xuint16_t
      if constexpr(sizeof(pixel_t) == 1) {
        b = _mm_packus_epi16(b, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1]+x), b);
      }
      else {
        b = _mm_max_epi16(_mm_min_epi16(b, limit), zero); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[1]+x), b);
      }
      // *R* ----------------
      m_uy = _mm_set1_epi32(int((static_cast<uint16_t>(m.y_r) << 16) | static_cast<uint16_t>(m.u_r))); // y and u
      m_vR = _mm_set1_epi32(int((static_cast<uint16_t>(4096) << 16) | static_cast<uint16_t>(m.v_r))); // rounding 13 bit >> 1 and v

      uy0123 = _mm_unpacklo_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy0123);
      xv0123 = _mm_unpacklo_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv0123);
      __m128i r_lo = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      uy4567 = _mm_unpackhi_epi16(u, y);
      res1 = _mm_madd_epi16(m_uy, uy4567);
      xv4567 = _mm_unpackhi_epi16(v, _mm_set1_epi16(1));
      res2 = _mm_madd_epi16(m_vR, xv4567);
      __m128i r_hi = _mm_srai_epi32(_mm_add_epi32(res1, res2), 13);

      __m128i r = _mm_packs_epi32(r_lo, r_hi); // 2x4x32 -> 2x4xuint16_t
      if constexpr(sizeof(pixel_t) == 1) {
        r = _mm_packus_epi16(r, zero);   // 8x uint16_t -> 8x uint_8
        _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2]+x), r);
      }
      else {
        r = _mm_max_epi16(_mm_min_epi16(r, limit), zero); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[2]+x), r);
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

//instantiate
//template<typename pixel_t, int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint8_14_sse2<uint8_t, 8>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_sse2<uint16_t, 10>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_sse2<uint16_t, 12>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_sse2<uint16_t, 14>(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m);

void convert_yuv_to_planarrgb_float_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 32 bit float
  __m128  offset_f = _mm_set1_ps(m.offset_y_f);

  const int rowsize = width * sizeof(float);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(float)) {
      __m128 sum1, res;
      __m128 mul_y, mul_u, mul_v;
      __m128 mat_y, mat_u, mat_v;
      __m128 y, u, v;
      // float: load 16 bytes: 4 pixels
      y = _mm_load_ps(reinterpret_cast<const float *>(srcp[0] + x));
      u = _mm_load_ps(reinterpret_cast<const float *>(srcp[1] + x));
      v = _mm_load_ps(reinterpret_cast<const float *>(srcp[2] + x));
      y = _mm_add_ps(y, offset_f); // offset is negative
      //u = _mm_sub_ps(u, half_f); // chroma is already 0 centered
      //v = _mm_sub_ps(v, half_f); // chroma is already 0 centered
      // *G*
      mat_y = _mm_set1_ps(m.y_g_f);  mat_u = _mm_set1_ps(m.u_g_f); mat_v = _mm_set1_ps(m.v_g_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[0] + x), res);
      // *B*
      mat_y = _mm_set1_ps(m.y_b_f);  mat_u = _mm_set1_ps(m.u_b_f); mat_v = _mm_set1_ps(m.v_b_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[1] + x), res);
      // *R*
      mat_y = _mm_set1_ps(m.y_r_f);  mat_u = _mm_set1_ps(m.u_r_f); mat_v = _mm_set1_ps(m.v_r_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      // no clamp
      _mm_store_ps(reinterpret_cast<float *>(dstp[2] + x), res);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void convert_yuv_to_planarrgb_uint16_sse41(BYTE *(&dstp)[3], int (&dstPitch)[3], const BYTE *(&srcp)[3], const int (&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 16 bit uint16_t (unsigned range)
  __m128  half_f = _mm_set1_ps((float)(1u << (bits_per_pixel - 1))); // 8-16 bit chroma has shift
  __m128i limit  = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128  offset_f = _mm_set1_ps(m.offset_y_f);

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(uint16_t)) {
      __m128 sum1, res;
      __m128 mul_y, mul_u, mul_v;
      __m128 mat_y, mat_u, mat_v;
      __m128 y, u, v;
      // uint16_t: load 8 bytes: 4 pixels
      __m128i yi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x));
      __m128i ui = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x));
      __m128i vi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x));
      y = _mm_cvtepi32_ps(_mm_unpacklo_epi16(yi,zero));
      u = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ui,zero));
      v = _mm_cvtepi32_ps(_mm_unpacklo_epi16(vi,zero));

      y = _mm_add_ps(y, offset_f); // offset is negative
      u = _mm_sub_ps(u, half_f);
      v = _mm_sub_ps(v, half_f);
      // *G*
      mat_y = _mm_set1_ps(m.y_g_f);  mat_u = _mm_set1_ps(m.u_g_f); mat_v = _mm_set1_ps(m.v_g_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      __m128i resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _mm_packus_epi32(resi, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0] + x), resi);
      // *B*
      mat_y = _mm_set1_ps(m.y_b_f);  mat_u = _mm_set1_ps(m.u_b_f); mat_v = _mm_set1_ps(m.v_b_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _mm_packus_epi32(resi, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + x), resi);
      // *R*
      mat_y = _mm_set1_ps(m.y_r_f);  mat_u = _mm_set1_ps(m.u_r_f); mat_v = _mm_set1_ps(m.v_r_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _mm_packus_epi32(resi, zero);
      if constexpr(bits_per_pixel<16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + x), resi);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint16_sse41<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);


template<int bits_per_pixel>
void convert_yuv_to_planarrgb_uint16_sse2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 16 bit uint16_t (unsigned range)
  __m128  half_f = _mm_set1_ps((float)(1u << (bits_per_pixel - 1)));
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m128  offset_f = _mm_set1_ps(m.offset_y_f);

  __m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 4 * sizeof(uint16_t)) {
      __m128 sum1, res;
      __m128 mul_y, mul_u, mul_v;
      __m128 mat_y, mat_u, mat_v;
      __m128 y, u, v;
      // uint16_t: load 8 bytes: 4 pixels
      __m128i yi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[0] + x));
      __m128i ui = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[1] + x));
      __m128i vi = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(srcp[2] + x));
      y = _mm_cvtepi32_ps(_mm_unpacklo_epi16(yi, zero));
      u = _mm_cvtepi32_ps(_mm_unpacklo_epi16(ui, zero));
      v = _mm_cvtepi32_ps(_mm_unpacklo_epi16(vi, zero));

      y = _mm_add_ps(y, offset_f); // offset is negative
      u = _mm_sub_ps(u, half_f);
      v = _mm_sub_ps(v, half_f);
      // *G*
      mat_y = _mm_set1_ps(m.y_g_f);  mat_u = _mm_set1_ps(m.u_g_f); mat_v = _mm_set1_ps(m.v_g_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      __m128i resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _MM_PACKUS_EPI32(resi, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[0] + x), resi);
      // *B*
      mat_y = _mm_set1_ps(m.y_b_f);  mat_u = _mm_set1_ps(m.u_b_f); mat_v = _mm_set1_ps(m.v_b_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _MM_PACKUS_EPI32(resi, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + x), resi);
      // *R*
      mat_y = _mm_set1_ps(m.y_r_f);  mat_u = _mm_set1_ps(m.u_r_f); mat_v = _mm_set1_ps(m.v_r_f);
      mul_y = _mm_mul_ps(y, mat_y);
      mul_u = _mm_mul_ps(u, mat_u);
      mul_v = _mm_mul_ps(v, mat_v);
      sum1 = _mm_add_ps(mul_y, mul_u);
      res = _mm_add_ps(sum1, mul_v);
      resi = _mm_cvtps_epi32(res); // no extra rounding, cvtps rounds to nearest
      resi = _MM_PACKUS_EPI32(resi, zero); // simulation
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resi = _mm_min_epi16(resi, limit); // clamp 10,12,14 bit
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + x), resi);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint16_sse2<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);

// packed rgb helper
static AVS_FORCEINLINE __m128i convert_yuv_to_rgb_sse2_core(const __m128i &px01, const __m128i &px23, const __m128i &px45, const __m128i &px67, const __m128i& zero, const __m128i &matrix, const __m128i &round_mask) {
  //int b = (((int)m[0] * Y + (int)m[1] * U + (int)m[ 2] * V + 4096)>>13);

  //px01 - xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0

  __m128i low_lo  = _mm_madd_epi16(px01, matrix); //xx*0 + v1*m2 | u1*m1 + y1*m0 | xx*0 + v0*m2 | u0*m1 + y0*m0
  __m128i low_hi  = _mm_madd_epi16(px23, matrix); //xx*0 + v3*m2 | u3*m1 + y3*m0 | xx*0 + v2*m2 | u2*m1 + y2*m0
  __m128i high_lo = _mm_madd_epi16(px45, matrix);
  __m128i high_hi = _mm_madd_epi16(px67, matrix);

  __m128i low_v  = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(low_lo), _mm_castsi128_ps(low_hi), _MM_SHUFFLE(3, 1, 3, 1))); // v3*m2 | v2*m2 | v1*m2 | v0*m2
  __m128i high_v = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(high_lo), _mm_castsi128_ps(high_hi), _MM_SHUFFLE(3, 1, 3, 1)));

  __m128i low_yu  = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(low_lo), _mm_castsi128_ps(low_hi), _MM_SHUFFLE(2, 0, 2, 0))); // u3*m1 + y3*m0 | u2*m1 + y2*m0 | u1*m1 + y1*m0 | u0*m1 + y0*m0
  __m128i high_yu = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(high_lo), _mm_castsi128_ps(high_hi), _MM_SHUFFLE(2, 0, 2, 0)));

  __m128i t_lo = _mm_add_epi32(low_v, low_yu); // v3*m2 + u3*m1 + y3*m0...
  __m128i t_hi = _mm_add_epi32(high_v, high_yu);

  t_lo = _mm_add_epi32(t_lo, round_mask); // v3*m2 + u3*m1 + y3*m0 + 4096...
  t_hi = _mm_add_epi32(t_hi, round_mask);

  t_lo = _mm_srai_epi32(t_lo, 13); // (v3*m2 + u3*m1 + y3*m0 + 4096) >> 13...
  t_hi = _mm_srai_epi32(t_hi, 13);

  __m128i result = _mm_packs_epi32(t_lo, t_hi);
  result = _mm_packus_epi16(result, zero); //00 00 00 00 00 00 00 00 b7 b6 b5 b4 b3 b2 b1 b0
  return result;
}

template<int rgb_pixel_step, bool hasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_yv24_to_rgb_ssse3(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix)
{
  dstp += dst_pitch * (height-1);  // We start at last line

  size_t mod8_width = rgb_pixel_step == 3 ? width / 8 * 8 : width; // for rgb32 target we may process pixels beyond width, but we have 32bit alignment at target

  __m128i matrix_b = _mm_set_epi16(0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b);
  __m128i matrix_g = _mm_set_epi16(0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g);
  __m128i matrix_r = _mm_set_epi16(0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r);

  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(4096);
  __m128i offset = _mm_set_epi16(0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y);
  __m128i pixels0123_mask = _mm_set_epi8(0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);
  __m128i pixels4567_mask = _mm_set_epi8(4, 2, 1, 0, 0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5);
  __m128i ssse3_merge_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m128i src_y = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcY+x)); //0 0 0 0 0 0 0 0 Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
      __m128i src_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcU+x)); //0 0 0 0 0 0 0 0 U7 U6 U5 U4 U3 U2 U1 U0
      __m128i src_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcV+x)); //0 0 0 0 0 0 0 0 V7 V6 V5 V4 V3 V2 V1 V0
      __m128i src_a;
      if(hasAlpha)
        src_a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcA+x)); //0 0 0 0 0 0 0 0 A7 A6 A5 A4 A3 A2 A1 A0

      __m128i t1 = _mm_unpacklo_epi8(src_y, src_u); //U7 Y7 U6 Y6 U5 Y5 U4 Y4 U3 Y3 U2 Y2 U1 Y1 U0 Y0
      __m128i t2 = _mm_unpacklo_epi8(src_v, zero);  //00 V7 00 V6 00 V5 00 V4 00 V3 00 V2 00 V1 00 V0

      __m128i low  = _mm_unpacklo_epi16(t1, t2); //xx V3 U3 Y3 xx V2 U2 Y2 xx V1 U1 Y1 xx V0 U0 Y0
      __m128i high = _mm_unpackhi_epi16(t1, t2); //xx V7 U7 Y7 xx V6 U6 Y6 xx V5 U5 Y5 xx V4 U4 Y4

      __m128i px01 = _mm_unpacklo_epi8(low, zero);  //xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0
      __m128i px23 = _mm_unpackhi_epi8(low, zero);  //xx xx 00 V3 00 U3 00 Y3 xx xx 00 V2 00 U2 00 Y2
      __m128i px45 = _mm_unpacklo_epi8(high, zero); //xx xx 00 V5 00 U5 00 Y5 xx xx 00 V4 00 U4 00 Y4
      __m128i px67 = _mm_unpackhi_epi8(high, zero); //xx xx 00 V7 00 U7 00 Y7 xx xx 00 V6 00 U6 00 Y6

      px01 = _mm_add_epi16(px01, offset);
      px23 = _mm_add_epi16(px23, offset);
      px45 = _mm_add_epi16(px45, offset);
      px67 = _mm_add_epi16(px67, offset);

      __m128i result_b = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_b, round_mask); //00 00 00 00 00 00 00 00 b7 b6 b5 b4 b3 b2 b1 b0
      __m128i result_g = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_g, round_mask); //00 00 00 00 00 00 00 00 g7 g6 g5 g4 g3 g2 g1 g0
      __m128i result_r = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_r, round_mask); //00 00 00 00 00 00 00 00 r7 r6 r5 r4 r3 r2 r1 r0

      __m128i result_bg = _mm_unpacklo_epi8(result_b, result_g); //g7 b7 g6 b6 g5 b5 g4 b4 g3 b3 g2 b2 g1 b1 g0 b0
      __m128i alpha;
      if(hasAlpha)
        alpha = src_a; // a7 .. a0
      else
        alpha = _mm_cmpeq_epi32(result_r, result_r); // FF FF FF FF ... default alpha transparent

      __m128i result_ra = _mm_unpacklo_epi8(result_r, alpha);       //a7 r7 a6 r6 a5 r5 a4 r4 a3 r3 a2 r2 a1 r1 a0 r0

      __m128i result_lo = _mm_unpacklo_epi16(result_bg, result_ra);
      __m128i result_hi = _mm_unpackhi_epi16(result_bg, result_ra);

      if constexpr(rgb_pixel_step == 4) {
        //rgb32
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x*4),    result_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x*4+16), result_hi);
      }
      else {
        //rgb24
        //"fast" SSSE3 version
        __m128i px0123 = _mm_shuffle_epi8(result_lo, pixels0123_mask); //xxxx xxxx b3g3 r3b2 g2r2 b1g1 r1b0 g0r0
        __m128i dst567 = _mm_shuffle_epi8(result_hi, pixels4567_mask); //r5b4 g4r4 xxxx xxxx b7g7 r7b6 g6r6 b5g5

        __m128i dst012345 = _mm_or_si128(
          _mm_andnot_si128(ssse3_merge_mask, px0123),
          _mm_and_si128(ssse3_merge_mask, dst567)
        ); //r5b4 g4r4 b3g3 r3b2 g2r2 b1g1 r1b0 g0r0

        _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + x * 3), dst012345);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x * 3 + 16), dst567);

      }
    }

    if constexpr(rgb_pixel_step == 3) {
      // for rgb32 (pixel_step == 4) we processed full width and more, including padded 8 bytes
      for (size_t x = mod8_width; x < width; ++x) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + 4096) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + 4096) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + 4096) >> 13);
        dstp[x*rgb_pixel_step + 0] = PixelClip(b);
        dstp[x*rgb_pixel_step + 1] = PixelClip(g);
        dstp[x*rgb_pixel_step + 2] = PixelClip(r);
        if constexpr(rgb_pixel_step == 4) { // n/a
          dstp[x * 4 + 3] = 255;
        }
      }
    }
    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
    if(hasAlpha)
      srcA += src_pitch_a;
  }
}

//instantiate
//template<int rgb_pixel_step, bool hasAlpha>
template void convert_yv24_to_rgb_ssse3<3, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_ssse3<4, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_ssse3<3, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_ssse3<4, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);


template<int rgb_pixel_step, bool hasAlpha>
void convert_yv24_to_rgb_sse2(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix) {
  dstp += dst_pitch * (height - 1);  // We start at last line

  size_t mod8_width = rgb_pixel_step == 3 ? width / 8 * 8 : width; // for rgb32 target we may process pixels beyond width, but we have 32bit alignment at target

  __m128i matrix_b = _mm_set_epi16(0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b);
  __m128i matrix_g = _mm_set_epi16(0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g);
  __m128i matrix_r = _mm_set_epi16(0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r);

  __m128i zero = _mm_setzero_si128();
  __m128i round_mask = _mm_set1_epi32(4096);
  __m128i offset = _mm_set_epi16(0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x += 8) {
      __m128i src_y = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcY + x)); //0 0 0 0 0 0 0 0 Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
      __m128i src_u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcU + x)); //0 0 0 0 0 0 0 0 U7 U6 U5 U4 U3 U2 U1 U0
      __m128i src_v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcV + x)); //0 0 0 0 0 0 0 0 V7 V6 V5 V4 V3 V2 V1 V0
      __m128i src_a;
      if (hasAlpha)
        src_a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcA + x)); //0 0 0 0 0 0 0 0 A7 A6 A5 A4 A3 A2 A1 A0

      __m128i t1 = _mm_unpacklo_epi8(src_y, src_u); //U7 Y7 U6 Y6 U5 Y5 U4 Y4 U3 Y3 U2 Y2 U1 Y1 U0 Y0
      __m128i t2 = _mm_unpacklo_epi8(src_v, zero);  //00 V7 00 V6 00 V5 00 V4 00 V3 00 V2 00 V1 00 V0

      __m128i low = _mm_unpacklo_epi16(t1, t2); //xx V3 U3 Y3 xx V2 U2 Y2 xx V1 U1 Y1 xx V0 U0 Y0
      __m128i high = _mm_unpackhi_epi16(t1, t2); //xx V7 U7 Y7 xx V6 U6 Y6 xx V5 U5 Y5 xx V4 U4 Y4

      __m128i px01 = _mm_unpacklo_epi8(low, zero);  //xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0
      __m128i px23 = _mm_unpackhi_epi8(low, zero);  //xx xx 00 V3 00 U3 00 Y3 xx xx 00 V2 00 U2 00 Y2
      __m128i px45 = _mm_unpacklo_epi8(high, zero); //xx xx 00 V5 00 U5 00 Y5 xx xx 00 V4 00 U4 00 Y4
      __m128i px67 = _mm_unpackhi_epi8(high, zero); //xx xx 00 V7 00 U7 00 Y7 xx xx 00 V6 00 U6 00 Y6

      px01 = _mm_add_epi16(px01, offset);
      px23 = _mm_add_epi16(px23, offset);
      px45 = _mm_add_epi16(px45, offset);
      px67 = _mm_add_epi16(px67, offset);

      __m128i result_b = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_b, round_mask); //00 00 00 00 00 00 00 00 b7 b6 b5 b4 b3 b2 b1 b0
      __m128i result_g = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_g, round_mask); //00 00 00 00 00 00 00 00 g7 g6 g5 g4 g3 g2 g1 g0
      __m128i result_r = convert_yuv_to_rgb_sse2_core(px01, px23, px45, px67, zero, matrix_r, round_mask); //00 00 00 00 00 00 00 00 r7 r6 r5 r4 r3 r2 r1 r0

      __m128i result_bg = _mm_unpacklo_epi8(result_b, result_g); //g7 b7 g6 b6 g5 b5 g4 b4 g3 b3 g2 b2 g1 b1 g0 b0
      __m128i alpha;
      if (hasAlpha)
        alpha = src_a; // a7 .. a0
      else
        alpha = _mm_cmpeq_epi32(result_r, result_r); // FF FF FF FF ... default alpha transparent

      __m128i result_ra = _mm_unpacklo_epi8(result_r, alpha);       //a7 r7 a6 r6 a5 r5 a4 r4 a3 r3 a2 r2 a1 r1 a0 r0

      __m128i result_lo = _mm_unpacklo_epi16(result_bg, result_ra);
      __m128i result_hi = _mm_unpackhi_epi16(result_bg, result_ra);

      if constexpr (rgb_pixel_step == 4) {
        //rgb32
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x * 4), result_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x * 4 + 16), result_hi);
      }
      else {
        //rgb24
        alignas(16) BYTE temp[32];
        //slow SSE2 version
        _mm_store_si128(reinterpret_cast<__m128i*>(temp), result_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(temp + 16), result_hi);

        for (int i = 0; i < 8; ++i) {
          *reinterpret_cast<int*>(dstp + (x + i) * 3) = *reinterpret_cast<int*>(temp + i * 4);
        }
        //last pixel
        dstp[(x + 7) * 3 + 0] = temp[7 * 4 + 0];
        dstp[(x + 7) * 3 + 1] = temp[7 * 4 + 1];
        dstp[(x + 7) * 3 + 2] = temp[7 * 4 + 2];
      }
    }

    if constexpr (rgb_pixel_step == 3) {
      // for rgb32 (pixel_step == 4) we processed full width and more, including padded 8 bytes
      for (size_t x = mod8_width; x < width; ++x) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + 4096) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + 4096) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + 4096) >> 13);
        dstp[x*rgb_pixel_step + 0] = PixelClip(b);
        dstp[x*rgb_pixel_step + 1] = PixelClip(g);
        dstp[x*rgb_pixel_step + 2] = PixelClip(r);
        if constexpr (rgb_pixel_step == 4) { // n/a
          dstp[x * 4 + 3] = 255;
        }
      }
    }
    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
    if (hasAlpha)
      srcA += src_pitch_a;
  }
}

//instantiate
//template<int rgb_pixel_step, bool hasAlpha>
template void convert_yv24_to_rgb_sse2<3, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_sse2<4, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_sse2<3, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_sse2<4, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, const BYTE*srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix &matrix);


#ifdef X86_32

static AVS_FORCEINLINE __m64 convert_yuv_to_rgb_mmx_core(const __m64 &px0, const __m64 &px1, const __m64 &px2, const __m64 &px3, const __m64& zero, const __m64 &matrix, const __m64 &round_mask) {
  //int b = (((int)m[0] * Y + (int)m[1] * U + (int)m[ 2] * V + 4096)>>13);

  //px01 - xx xx 00 V0 00 U0 00 Y0

  __m64 low_lo  = _mm_madd_pi16(px0, matrix); //xx*0 + v1*m2 | u1*m1 + y1*m0 | xx*0 + v0*m2 | u0*m1 + y0*m0
  __m64 low_hi  = _mm_madd_pi16(px1, matrix); //xx*0 + v3*m2 | u3*m1 + y3*m0 | xx*0 + v2*m2 | u2*m1 + y2*m0
  __m64 high_lo = _mm_madd_pi16(px2, matrix);
  __m64 high_hi = _mm_madd_pi16(px3, matrix);

  __m64 low_v = _mm_unpackhi_pi32(low_lo, low_hi); // v1*m2 | v0*m2
  __m64 high_v = _mm_unpackhi_pi32(high_lo, high_hi);

  __m64 low_yu = _mm_unpacklo_pi32(low_lo, low_hi); // u1*m1 + y1*m0 | u0*m1 + y0*m0
  __m64 high_yu = _mm_unpacklo_pi32(high_lo, high_hi);

  __m64 t_lo = _mm_add_pi32(low_v, low_yu); // v3*m2 + u3*m1 + y3*m0...
  __m64 t_hi = _mm_add_pi32(high_v, high_yu);

  t_lo = _mm_add_pi32(t_lo, round_mask); // v3*m2 + u3*m1 + y3*m0 + 4096...
  t_hi = _mm_add_pi32(t_hi, round_mask);

  t_lo = _mm_srai_pi32(t_lo, 13); // (v3*m2 + u3*m1 + y3*m0 + 4096) >> 13...
  t_hi = _mm_srai_pi32(t_hi, 13);

  __m64 result = _mm_packs_pi32(t_lo, t_hi);
  result = _mm_packs_pu16(result, zero); //00 00 00 00 b3 b2 b1 b0
  return result;
}

template<int rgb_pixel_step>
void convert_yv24_to_rgb_mmx(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t width, size_t height, const ConversionMatrix &matrix) {
  dstp += dst_pitch * (height-1);  // We start at last line

  size_t mod4_width = rgb_pixel_step == 3 ? width / 4 * 4 : width;

  __m64 matrix_b = _mm_set_pi16(0, matrix.v_b, matrix.u_b, matrix.y_b);
  __m64 matrix_g = _mm_set_pi16(0, matrix.v_g, matrix.u_g, matrix.y_g);
  __m64 matrix_r = _mm_set_pi16(0, matrix.v_r, matrix.u_r, matrix.y_r);

  __m64 zero = _mm_setzero_si64();
  __m64 round_mask = _mm_set1_pi32(4096);
  __m64 ff = _mm_set1_pi32(0xFFFFFFFF);
  __m64 offset = _mm_set_pi16(0, -128, -128, matrix.offset_y);
  __m64 low_pixel_mask = _mm_set_pi32(0, 0x00FFFFFF);
  __m64 high_pixel_mask = _mm_set_pi32(0x00FFFFFF, 0);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod4_width; x+=4) {
      __m64 src_y = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcY+x)); //0 0 0 0 Y3 Y2 Y1 Y0
      __m64 src_u = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcU+x)); //0 0 0 0 U3 U2 U1 U0
      __m64 src_v = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcV+x)); //0 0 0 0 V3 V2 V1 V0

      __m64 t1 = _mm_unpacklo_pi8(src_y, src_u); //U3 Y3 U2 Y2 U1 Y1 U0 Y0
      __m64 t2 = _mm_unpacklo_pi8(src_v, zero);  //00 V3 00 V2 00 V1 00 V0

      __m64 low  = _mm_unpacklo_pi16(t1, t2); //xx V1 U1 Y1 xx V0 U0 Y0
      __m64 high = _mm_unpackhi_pi16(t1, t2); //xx V3 U3 Y3 xx V2 U2 Y2

      __m64 px0 = _mm_unpacklo_pi8(low, zero);  //xx xx 00 V0 00 U0 00 Y0
      __m64 px1 = _mm_unpackhi_pi8(low, zero);  //xx xx 00 V1 00 U1 00 Y1
      __m64 px2 = _mm_unpacklo_pi8(high, zero); //xx xx 00 V2 00 U2 00 Y2
      __m64 px3 = _mm_unpackhi_pi8(high, zero); //xx xx 00 V3 00 U3 00 Y3

      px0 = _mm_add_pi16(px0, offset);
      px1 = _mm_add_pi16(px1, offset);
      px2 = _mm_add_pi16(px2, offset);
      px3 = _mm_add_pi16(px3, offset);

      __m64 result_b = convert_yuv_to_rgb_mmx_core(px0, px1, px2, px3, zero, matrix_b, round_mask); //00 00 00 00 b3 b2 b1 b0
      __m64 result_g = convert_yuv_to_rgb_mmx_core(px0, px1, px2, px3, zero, matrix_g, round_mask); //00 00 00 00 g3 g2 g1 g0
      __m64 result_r = convert_yuv_to_rgb_mmx_core(px0, px1, px2, px3, zero, matrix_r, round_mask); //00 00 00 00 r3 r2 r1 r0

      __m64 result_bg = _mm_unpacklo_pi8(result_b, result_g); //g3 b3 g2 b2 g1 b1 g0 b0
      __m64 result_ra = _mm_unpacklo_pi8(result_r, ff);       //a3 r3 a2 r2 a1 r1 a0 r0

      __m64 result_lo = _mm_unpacklo_pi16(result_bg, result_ra);
      __m64 result_hi = _mm_unpackhi_pi16(result_bg, result_ra);

      if (rgb_pixel_step == 4) {
        //rgb32
        *reinterpret_cast<__m64*>(dstp+x*4) = result_lo;
        *reinterpret_cast<__m64*>(dstp+x*4+8) = result_hi;
      } else {
        __m64 p0 = _mm_and_si64(result_lo, low_pixel_mask); //0000 0000 00r0 g0b0
        __m64 p1 = _mm_and_si64(result_lo, high_pixel_mask); //00r1 g1b1 0000 0000
        __m64 p2 = _mm_and_si64(result_hi, low_pixel_mask); //0000 0000 00r2 g2b2
        __m64 p3 = _mm_and_si64(result_hi, high_pixel_mask); //00r3 g3b3 0000 0000

        __m64 dst01 = _mm_or_si64(p0, _mm_srli_si64(p1, 8)); //0000 r1g1 b1r0 g0b0
        p3 = _mm_srli_si64(p3, 24); //0000 0000 r3g3 b300

        __m64 dst012 = _mm_or_si64(dst01, _mm_slli_si64(p2, 48));  //g2b2 r1g1 b1r0 g0b0
        __m64 dst23 = _mm_or_si64(p3, _mm_srli_si64(p2, 16)); //0000 0000 r3g3 b3r2

        *reinterpret_cast<__m64*>(dstp+x*3) = dst012;
        *reinterpret_cast<int*>(dstp+x*3+8) = _mm_cvtsi64_si32(dst23);
      }
    }

    if (rgb_pixel_step == 3) {
      for (size_t x = mod4_width; x < width; ++x) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + 4096) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + 4096) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + 4096) >> 13);
        dstp[x*rgb_pixel_step + 0] = PixelClip(b);
        dstp[x*rgb_pixel_step + 1] = PixelClip(g);
        dstp[x*rgb_pixel_step + 2] = PixelClip(r);
        if (rgb_pixel_step == 4) {
          dstp[x * 4 + 3] = 255;
        }
      }
    }

    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
  }
  _mm_empty();
}

//instantiate
//template<int rgb_pixel_step>
template void convert_yv24_to_rgb_mmx<3>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t width, size_t height, const ConversionMatrix &matrix);
template void convert_yv24_to_rgb_mmx<4>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE*srcV, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t width, size_t height, const ConversionMatrix &matrix);

#endif




void convert_yuy2_to_yv16_sse2(const BYTE *srcp, BYTE *dstp_y, BYTE *dstp_u, BYTE *dstp_v, size_t src_pitch, size_t dst_pitch_y, size_t dst_pitch_uv, size_t width, size_t height)
{
  width /= 2;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 8) {
      __m128i p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 4));      // V3 Y7 U3 Y6 V2 Y5 U2 Y4 V1 Y3 U1 Y2 V0 Y1 U0 Y0
      __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 4 + 16)); // V7 Yf U7 Ye V6 Yd U6 Yc V5 Yb U5 Ya V4 Y9 U4 Y8

      __m128i p2 = _mm_unpacklo_epi8(p0, p1); // V5 V1 Yb Y3 U5 U1 Ya Y2 V4 V0 Y9 Y1 U4 U0 Y8 Y0
      __m128i p3 = _mm_unpackhi_epi8(p0, p1); // V7 V3 Yf Y7 U7 U3 Ye Y6 V6 V2 Yd Y5 U6 U2 Yc Y4

      p0 = _mm_unpacklo_epi8(p2, p3); // V6 V4 V2 V0 Yd Y9 Y5 Y1 U6 U4 U2 U0 Yc Y8 Y4 Y0
      p1 = _mm_unpackhi_epi8(p2, p3); // V7 V5 V3 V1 Yf Yb Y7 Y3 U7 U5 U3 U1 Ye Ya Y6 Y2

      p2 = _mm_unpacklo_epi8(p0, p1); // U7 U6 U5 U4 U3 U2 U1 U0 Ye Yc Ya Y8 Y6 Y4 Y2 Y0
      p3 = _mm_unpackhi_epi8(p0, p1); // V7 V6 V5 V4 V3 V2 V1 V0 Yf Yd Yb Y9 Y7 Y5 Y3 Y1

      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp_u + x), _mm_srli_si128(p2, 8));
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp_v + x), _mm_srli_si128(p3, 8));
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp_y + x * 2), _mm_unpacklo_epi8(p2, p3));
    }

    srcp += src_pitch;
    dstp_y += dst_pitch_y;
    dstp_u += dst_pitch_uv;
    dstp_v += dst_pitch_uv;
  }
}


#ifdef X86_32

void convert_yuy2_to_yv16_mmx(const BYTE *srcp, BYTE *dstp_y, BYTE *dstp_u, BYTE *dstp_v, size_t src_pitch, size_t dst_pitch_y, size_t dst_pitch_uv, size_t width, size_t height)
{
  width /= 2;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; x += 4) {
      __m64 p0 = *reinterpret_cast<const __m64*>(srcp + x * 4);     // V1 Y3 U1 Y2 V0 Y1 U0 Y0
      __m64 p1 = *reinterpret_cast<const __m64*>(srcp + x * 4 + 8); // V3 Y7 U3 Y6 V2 Y5 U2 Y4

      __m64 p2 = _mm_unpacklo_pi8(p0, p1); // V2 V0 Y5 Y1 U2 U0 Y4 Y0
      __m64 p3 = _mm_unpackhi_pi8(p0, p1); // V3 V1 Y7 Y3 U3 U1 Y6 Y2

      p0 = _mm_unpacklo_pi8(p2, p3); // U3 U2 U1 U0 Y6 Y4 Y2 Y0
      p1 = _mm_unpackhi_pi8(p2, p3); // V3 V2 V1 V0 Y7 Y5 Y3 Y1

      *reinterpret_cast<int*>(dstp_u + x) = _mm_cvtsi64_si32(_mm_srli_si64(p0, 32));
      *reinterpret_cast<int*>(dstp_v + x) = _mm_cvtsi64_si32(_mm_srli_si64(p1, 32));
      *reinterpret_cast<__m64*>(dstp_y + x * 2) = _mm_unpacklo_pi8(p0, p1);
    }

    srcp += src_pitch;
    dstp_y += dst_pitch_y;
    dstp_u += dst_pitch_uv;
    dstp_v += dst_pitch_uv;
  }
  _mm_empty();
}

#endif


void convert_yv16_to_yuy2_sse2(const BYTE *srcp_y, const BYTE *srcp_u, const BYTE *srcp_v, BYTE *dstp, size_t src_pitch_y, size_t src_pitch_uv, size_t dst_pitch, size_t width, size_t height)
{
  width /= 2;

  for (size_t yy=0; yy<height; yy++) {
    for (size_t x=0; x<width; x+=8) {

      __m128i y = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp_y + x*2));
      __m128i u = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp_u + x));
      __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp_v + x));

      __m128i uv = _mm_unpacklo_epi8(u, v);
      __m128i yuv_lo = _mm_unpacklo_epi8(y, uv);
      __m128i yuv_hi = _mm_unpackhi_epi8(y, uv);

      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x*4), yuv_lo);
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x*4 + 16), yuv_hi);
    }

    srcp_y += src_pitch_y;
    srcp_u += src_pitch_uv;
    srcp_v += src_pitch_uv;
    dstp += dst_pitch;
  }
}

#ifdef X86_32
void convert_yv16_to_yuy2_mmx(const BYTE *srcp_y, const BYTE *srcp_u, const BYTE *srcp_v, BYTE *dstp, size_t src_pitch_y, size_t src_pitch_uv, size_t dst_pitch, size_t width, size_t height)
{
  width /= 2;

  for (size_t y=0; y<height; y++) {
    for (size_t x=0; x<width; x+=4) {
      __m64 y = *reinterpret_cast<const __m64*>(srcp_y + x*2);
      __m64 u = *reinterpret_cast<const __m64*>(srcp_u + x);
      __m64 v = *reinterpret_cast<const __m64*>(srcp_v + x);

      __m64 uv = _mm_unpacklo_pi8(u, v);
      __m64 yuv_lo = _mm_unpacklo_pi8(y, uv);
      __m64 yuv_hi = _mm_unpackhi_pi8(y, uv);

      *reinterpret_cast<__m64*>(dstp + x*4) = yuv_lo;
      *reinterpret_cast<__m64*>(dstp + x*4+8) = yuv_hi;
    }

    srcp_y += src_pitch_y;
    srcp_u += src_pitch_uv;
    srcp_v += src_pitch_uv;
    dstp += dst_pitch;
  }
  _mm_empty();
}
#endif




