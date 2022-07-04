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

#include <avisynth.h>
#include <avs/minmax.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <emmintrin.h>

void compare_sse2(uint32_t mask, int increment,
                         const BYTE * f1ptr, int pitch1,
                         const BYTE * f2ptr, int pitch2,
                         int rowsize, int height,
                         int &SAD_sum, int &SD_sum, int &pos_D,  int &neg_D, double &SSD_sum)
{
  // rowsize multiple of 16 for YUV Planar, RGB32 and YUY2; 12 for RGB24
  // increment must be 3 for RGB24 and 4 for others

  int64_t issd = 0;
  __m128i sad_vector = _mm_setzero_si128(); //sum of absolute differences
  __m128i sd_vector = _mm_setzero_si128(); // sum of differences
  __m128i positive_diff = _mm_setzero_si128();
  __m128i negative_diff = _mm_setzero_si128();
  __m128i zero = _mm_setzero_si128();

  __m128i mask64 = _mm_set_epi32(0, 0, 0, mask);
  if (increment == 3) {
    mask64 = _mm_or_si128(mask64, _mm_slli_si128(mask64, 3));
    mask64 = _mm_or_si128(mask64, _mm_slli_si128(mask64, 6));
  } else {
    mask64 = _mm_or_si128(mask64, _mm_slli_si128(mask64, 4));
    mask64 = _mm_or_si128(mask64, _mm_slli_si128(mask64, 8));
  }



  for (int y = 0; y < height; ++y) {
    __m128i row_ssd = _mm_setzero_si128();  // sum of squared differences (row_SSD)

    for (int x = 0; x < rowsize; x+=increment*4) {
      __m128i src1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f1ptr+x));
      __m128i src2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f2ptr+x));

      src1 = _mm_and_si128(src1, mask64);
      src2 = _mm_and_si128(src2, mask64);

      __m128i diff_1_minus_2 = _mm_subs_epu8(src1, src2);
      __m128i diff_2_minus_1 = _mm_subs_epu8(src2, src1);

      positive_diff = _mm_max_epu8(positive_diff, diff_1_minus_2);
      negative_diff = _mm_max_epu8(negative_diff, diff_2_minus_1);

      __m128i absdiff1 = _mm_sad_epu8(diff_1_minus_2, zero);
      __m128i absdiff2 = _mm_sad_epu8(diff_2_minus_1, zero);

      sad_vector = _mm_add_epi32(sad_vector, absdiff1);
      sad_vector = _mm_add_epi32(sad_vector, absdiff2);

      sd_vector = _mm_add_epi32(sd_vector, absdiff1);
      sd_vector = _mm_sub_epi32(sd_vector, absdiff2);

      __m128i ssd = _mm_or_si128(diff_1_minus_2, diff_2_minus_1);
      __m128i ssd_lo = _mm_unpacklo_epi8(ssd, zero);
      __m128i ssd_hi = _mm_unpackhi_epi8(ssd, zero);
      ssd_lo   = _mm_madd_epi16(ssd_lo, ssd_lo);
      ssd_hi   = _mm_madd_epi16(ssd_hi, ssd_hi);
      row_ssd = _mm_add_epi32(row_ssd, ssd_lo);
      row_ssd = _mm_add_epi32(row_ssd, ssd_hi);
    }

    f1ptr += pitch1;
    f2ptr += pitch2;

    __m128i tmp = _mm_srli_si128(row_ssd, 8);
    row_ssd = _mm_add_epi32(row_ssd, tmp);
    tmp = _mm_srli_si128(row_ssd, 4);
    row_ssd = _mm_add_epi32(row_ssd, tmp);

    issd += _mm_cvtsi128_si32(row_ssd);
  }

  SAD_sum += _mm_cvtsi128_si32(sad_vector);
  SAD_sum += _mm_cvtsi128_si32(_mm_srli_si128(sad_vector, 8));
  SD_sum  += _mm_cvtsi128_si32(sd_vector);
  SD_sum += _mm_cvtsi128_si32(_mm_srli_si128(sd_vector, 8));

  BYTE posdiff_tmp[16];
  BYTE negdiff_tmp[16];
  _mm_store_si128(reinterpret_cast<__m128i*>(posdiff_tmp), positive_diff);
  _mm_store_si128(reinterpret_cast<__m128i*>(negdiff_tmp), negative_diff);

  SSD_sum += (double)issd;

  neg_D = -neg_D; // 160801! false neg_D fix for isse

  for (int i = 0; i < increment*4; ++i) {
    pos_D = max(pos_D, (int)(posdiff_tmp[i]));
    neg_D = max(neg_D, (int)(negdiff_tmp[i]));
  }

  neg_D = -neg_D;
}

#ifdef X86_32

void compare_isse(uint32_t mask, int increment,
                         const BYTE * f1ptr, int pitch1,
                         const BYTE * f2ptr, int pitch2,
                         int rowsize, int height,
                         int &SAD_sum, int &SD_sum, int &pos_D,  int &neg_D, double &SSD_sum)
{
  // rowsize multiple of 8 for YUV Planar, RGB32 and YUY2; 6 for RGB24
  // increment must be 3 for RGB24 and 4 for others

  int64_t issd = 0;
  __m64 sad_vector = _mm_setzero_si64(); //sum of absolute differences
  __m64 sd_vector = _mm_setzero_si64(); // sum of differences
  __m64 positive_diff = _mm_setzero_si64();
  __m64 negative_diff = _mm_setzero_si64();
  __m64 zero = _mm_setzero_si64();

  __m64 mask64 = _mm_set_pi32(0, mask);
  mask64 = _mm_or_si64(mask64, _mm_slli_si64(mask64, increment*8));


  for (int y = 0; y < height; ++y) {
    __m64 row_ssd = _mm_setzero_si64();  // sum of squared differences (row_SSD)

    for (int x = 0; x < rowsize; x+=increment*2) {
      __m64 src1 = *reinterpret_cast<const __m64*>(f1ptr+x);
      __m64 src2 = *reinterpret_cast<const __m64*>(f2ptr+x);

      src1 = _mm_and_si64(src1, mask64);
      src2 = _mm_and_si64(src2, mask64);

      __m64 diff_1_minus_2 = _mm_subs_pu8(src1, src2);
      __m64 diff_2_minus_1 = _mm_subs_pu8(src2, src1);

      positive_diff = _mm_max_pu8(positive_diff, diff_1_minus_2);
      negative_diff = _mm_max_pu8(negative_diff, diff_2_minus_1);

      __m64 absdiff1 = _mm_sad_pu8(diff_1_minus_2, zero);
      __m64 absdiff2 = _mm_sad_pu8(diff_2_minus_1, zero);

      sad_vector = _mm_add_pi32(sad_vector, absdiff1);
      sad_vector = _mm_add_pi32(sad_vector, absdiff2);

      sd_vector = _mm_add_pi32(sd_vector, absdiff1);
      sd_vector = _mm_sub_pi32(sd_vector, absdiff2);

      __m64 ssd = _mm_or_si64(diff_1_minus_2, diff_2_minus_1);
      __m64 ssd_lo = _mm_unpacklo_pi8(ssd, zero);
      __m64 ssd_hi = _mm_unpackhi_pi8(ssd, zero);
      ssd_lo   = _mm_madd_pi16(ssd_lo, ssd_lo);
      ssd_hi   = _mm_madd_pi16(ssd_hi, ssd_hi);
      row_ssd = _mm_add_pi32(row_ssd, ssd_lo);
      row_ssd = _mm_add_pi32(row_ssd, ssd_hi);
    }

    f1ptr += pitch1;
    f2ptr += pitch2;

    __m64 tmp = _mm_unpackhi_pi32(row_ssd, zero);
    row_ssd = _mm_add_pi32(row_ssd, tmp);

    issd += _mm_cvtsi64_si32(row_ssd);
  }

  SAD_sum += _mm_cvtsi64_si32(sad_vector);
  SD_sum  += _mm_cvtsi64_si32(sd_vector);

  BYTE posdiff_tmp[8];
  BYTE negdiff_tmp[8];
  *reinterpret_cast<__m64*>(posdiff_tmp) = positive_diff;
  *reinterpret_cast<__m64*>(negdiff_tmp) = negative_diff;
  _mm_empty();

  SSD_sum += (double)issd;

  neg_D = -neg_D; // 160801! false neg_D fix for isse

  for (int i = 0; i < increment*2; ++i) {
    pos_D = max(pos_D, (int)(posdiff_tmp[i]));
    neg_D = max(neg_D, (int)(negdiff_tmp[i]));
  }

  neg_D = -neg_D;
}

#endif



