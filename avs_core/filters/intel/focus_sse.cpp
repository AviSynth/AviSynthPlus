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
#include "../focus.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include "../core/internal.h"
#include <stdint.h>

/****************************************
 ***  AdjustFocus helper classes     ***
 ***  Originally by Ben R.G.         ***
 ***  MMX code by Marc FD            ***
 ***  Adaptation and bugfixes sh0dan ***
 ***  Code actually requires ISSE!   ***
 ***  Not anymore - pure MMX    IanB ***
 ***  Implement boundary proc.  IanB ***
 ***  Impl. full 8bit MMX proc. IanB ***
 ***************************************/

 // for linker reasons these forceinlined functions appear in C, intel sse2 and avx2 source as well
template<typename pixel_t>
static AVS_FORCEINLINE void af_horizontal_planar_process_line_c(pixel_t left, BYTE* dstp8, size_t row_size, int center_weight, int outer_weight) {
  size_t x;
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width - 1; ++x) {
    pixel_t temp = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x + 1]) * (weight_t)outer_weight));
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight));
}

static AVS_FORCEINLINE void af_horizontal_planar_process_line_uint16_c(uint16_t left, BYTE* dstp8, size_t row_size, int center_weight, int outer_weight, int bits_per_pixel) {
  size_t x;
  typedef uint16_t pixel_t;
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const int max_pixel_value = (1 << bits_per_pixel) - 1; // clamping on 10-12-14-16 bitdepth
  typedef std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width - 1; ++x) {
    pixel_t temp = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x + 1]) * (weight_t)outer_weight), max_pixel_value);
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight), max_pixel_value);
}

static AVS_FORCEINLINE void af_horizontal_planar_process_line_float_c(float left, float* dstp, size_t row_size, float center_weight, float outer_weight) {
  size_t x;
  size_t width = row_size / sizeof(float);
  for (x = 0; x < width - 1; ++x) {
    float temp = dstp[x] * center_weight + (left + dstp[x + 1]) * outer_weight;
    left = dstp[x];
    dstp[x] = temp;
  }
  dstp[x] = dstp[x] * center_weight + (left + dstp[x]) * outer_weight;
}

void af_vertical_sse2_float(BYTE* line_buf, BYTE* dstp, const int height, const int pitch, const int row_size, const float amount) {

  const float center_weight = amount;
  const float outer_weight = (1.0f - amount) / 2.0f;

  __m128 center_weight_simd = _mm_set1_ps(center_weight);
  __m128 outer_weight_simd = _mm_set1_ps(outer_weight);

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 16) {
      __m128 upper = _mm_load_ps(reinterpret_cast<const float*>(line_buf + x));
      __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));
      __m128 lower = _mm_load_ps(reinterpret_cast<const float*>(dstp + pitch + x));
      _mm_store_ps(reinterpret_cast<float*>(line_buf + x), center);

      __m128 tmp1 = _mm_mul_ps(center, center_weight_simd);
      __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, lower), outer_weight_simd);
      __m128 result = _mm_add_ps(tmp1, tmp2);

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 16) {
    __m128 upper = _mm_load_ps(reinterpret_cast<const float*>(line_buf + x));
    __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));

    __m128 tmp1 = _mm_mul_ps(center, center_weight_simd);
    __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, center), outer_weight_simd); // last line: center instead of lower
    __m128 result = _mm_add_ps(tmp1, tmp2);

    _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
  }
}


static AVS_FORCEINLINE __m128i af_blend_sse2(__m128i &upper, __m128i &center, __m128i &lower, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i outer_tmp = _mm_add_epi16(upper, lower);
  __m128i center_tmp = _mm_mullo_epi16(center, center_weight);

  outer_tmp = _mm_mullo_epi16(outer_tmp, outer_weight);

  __m128i result = _mm_adds_epi16(center_tmp, outer_tmp);
  result = _mm_adds_epi16(result, center_tmp);
  result = _mm_adds_epi16(result, round_mask);
  return _mm_srai_epi16(result, 7);
}

static AVS_FORCEINLINE __m128i af_blend_uint16_t_sse2(__m128i &upper, __m128i &center, __m128i &lower, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i outer_tmp = _mm_add_epi32(upper, lower);
  __m128i center_tmp = _MM_MULLO_EPI32(center, center_weight); // sse2: mullo simulation
  outer_tmp = _MM_MULLO_EPI32(outer_tmp, outer_weight);

  __m128i result = _mm_add_epi32(center_tmp, outer_tmp);
  result = _mm_add_epi32(result, center_tmp);
  result = _mm_add_epi32(result, round_mask);
  return _mm_srai_epi32(result, 7);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE __m128i af_blend_uint16_t_sse41(__m128i &upper, __m128i &center, __m128i &lower, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask)
{
  __m128i outer_tmp = _mm_add_epi32(upper, lower);
  __m128i center_tmp = _mm_mullo_epi32(center, center_weight);
  outer_tmp = _mm_mullo_epi32(outer_tmp, outer_weight);

  __m128i result = _mm_add_epi32(center_tmp, outer_tmp);
  result = _mm_add_epi32(result, center_tmp);
  result = _mm_add_epi32(result, round_mask);
  return _mm_srai_epi32(result, 7);
}

static AVS_FORCEINLINE __m128 af_blend_float_sse2(__m128 &upper, __m128 &center, __m128 &lower, __m128 &center_weight, __m128 &outer_weight) {
  __m128 tmp1 = _mm_mul_ps(center, center_weight);
  __m128 tmp2 = _mm_mul_ps(_mm_add_ps(upper, lower), outer_weight);
  return _mm_add_ps(tmp1, tmp2);
}


static AVS_FORCEINLINE __m128i af_unpack_blend_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask, __m128i &zero) {
  __m128i left_lo = _mm_unpacklo_epi8(left, zero);
  __m128i left_hi = _mm_unpackhi_epi8(left, zero);
  __m128i center_lo = _mm_unpacklo_epi8(center, zero);
  __m128i center_hi = _mm_unpackhi_epi8(center, zero);
  __m128i right_lo = _mm_unpacklo_epi8(right, zero);
  __m128i right_hi = _mm_unpackhi_epi8(right, zero);

  __m128i result_lo = af_blend_sse2(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m128i result_hi = af_blend_sse2(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);

  return _mm_packus_epi16(result_lo, result_hi);
}

static AVS_FORCEINLINE __m128i af_unpack_blend_uint16_t_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask, __m128i &zero) {
  __m128i left_lo = _mm_unpacklo_epi16(left, zero);
  __m128i left_hi = _mm_unpackhi_epi16(left, zero);
  __m128i center_lo = _mm_unpacklo_epi16(center, zero);
  __m128i center_hi = _mm_unpackhi_epi16(center, zero);
  __m128i right_lo = _mm_unpacklo_epi16(right, zero);
  __m128i right_hi = _mm_unpackhi_epi16(right, zero);

  __m128i result_lo = af_blend_uint16_t_sse2(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m128i result_hi = af_blend_uint16_t_sse2(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);
  return _MM_PACKUS_EPI32(result_lo, result_hi); // sse4.1 simul
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE __m128i af_unpack_blend_uint16_t_sse41(__m128i &left, __m128i &center, __m128i &right, __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask, __m128i &zero)
{
  __m128i left_lo = _mm_unpacklo_epi16(left, zero);
  __m128i left_hi = _mm_unpackhi_epi16(left, zero);
  __m128i center_lo = _mm_unpacklo_epi16(center, zero);
  __m128i center_hi = _mm_unpackhi_epi16(center, zero);
  __m128i right_lo = _mm_unpacklo_epi16(right, zero);
  __m128i right_hi = _mm_unpackhi_epi16(right, zero);

  __m128i result_lo = af_blend_uint16_t_sse41(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m128i result_hi = af_blend_uint16_t_sse41(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);
  return _mm_packus_epi32(result_lo, result_hi);
}


void af_vertical_uint16_t_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount) {
  // amount was: half_amount (32768). Full: 65536 (2**16)
  // now it becomes 2**(16-9)=2**7 scale
  int t = (amount + 256) >> 9; // 16-9 = 7 -> shift in
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 16) {
      __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      __m128i lower = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + pitch + x));
      _mm_store_si128(reinterpret_cast<__m128i*>(line_buf + x), center);

      __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
      __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
      __m128i center_lo = _mm_unpacklo_epi16(center, zero);
      __m128i center_hi = _mm_unpackhi_epi16(center, zero);
      __m128i lower_lo = _mm_unpacklo_epi16(lower, zero);
      __m128i lower_hi = _mm_unpackhi_epi16(lower, zero);

      __m128i result_lo = af_blend_uint16_t_sse2(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m128i result_hi = af_blend_uint16_t_sse2(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m128i result = _MM_PACKUS_EPI32(result_lo, result_hi); // sse4.1 simul

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 16) {
    __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));

    __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
    __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
    __m128i center_lo = _mm_unpacklo_epi16(center, zero);
    __m128i center_hi = _mm_unpackhi_epi16(center, zero);

    __m128i result_lo = af_blend_uint16_t_sse2(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m128i result_hi = af_blend_uint16_t_sse2(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m128i result = _MM_PACKUS_EPI32(result_lo, result_hi); // sse4.1 simul

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_vertical_uint16_t_sse41(BYTE* line_buf, BYTE* dstp, int height, int pitch, int row_size, int amount)
{
  // amount was: half_amount (32768). Full: 65536 (2**16)
  // now it becomes 2**(16-9)=2**7 scale
  int t = (amount + 256) >> 9; // 16-9 = 7 -> shift in
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height - 1; ++y) {
    for (int x = 0; x < row_size; x += 16) {
      __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      __m128i lower = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + pitch + x));
      _mm_store_si128(reinterpret_cast<__m128i*>(line_buf + x), center);

      __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
      __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
      __m128i center_lo = _mm_unpacklo_epi16(center, zero);
      __m128i center_hi = _mm_unpackhi_epi16(center, zero);
      __m128i lower_lo = _mm_unpacklo_epi16(lower, zero);
      __m128i lower_hi = _mm_unpackhi_epi16(lower, zero);

      __m128i result_lo = af_blend_uint16_t_sse41(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m128i result_hi = af_blend_uint16_t_sse41(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m128i result = _mm_packus_epi32(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < row_size; x += 16) {
    __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf + x));
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));

    __m128i upper_lo = _mm_unpacklo_epi16(upper, zero);
    __m128i upper_hi = _mm_unpackhi_epi16(upper, zero);
    __m128i center_lo = _mm_unpacklo_epi16(center, zero);
    __m128i center_hi = _mm_unpackhi_epi16(center, zero);

    __m128i result_lo = af_blend_uint16_t_sse41(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m128i result_hi = af_blend_uint16_t_sse41(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m128i result = _mm_packus_epi32(result_lo, result_hi);

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
  }
}

void af_vertical_sse2(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount) {
  short t = (amount + 256) >> 9;
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height-1; ++y) {
    for (int x = 0; x < width; x+= 16) {
      __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf+x));
      __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));
      __m128i lower = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+pitch+x));
      _mm_store_si128(reinterpret_cast<__m128i*>(line_buf+x), center);

      __m128i upper_lo = _mm_unpacklo_epi8(upper, zero);
      __m128i upper_hi = _mm_unpackhi_epi8(upper, zero);
      __m128i center_lo = _mm_unpacklo_epi8(center, zero);
      __m128i center_hi = _mm_unpackhi_epi8(center, zero);
      __m128i lower_lo = _mm_unpacklo_epi8(lower, zero);
      __m128i lower_hi = _mm_unpackhi_epi8(lower, zero);

      __m128i result_lo = af_blend_sse2(upper_lo, center_lo, lower_lo, center_weight, outer_weight, round_mask);
      __m128i result_hi = af_blend_sse2(upper_hi, center_hi, lower_hi, center_weight, outer_weight, round_mask);

      __m128i result = _mm_packus_epi16(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < width; x+= 16) {
    __m128i upper = _mm_load_si128(reinterpret_cast<const __m128i*>(line_buf+x));
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));

    __m128i upper_lo = _mm_unpacklo_epi8(upper, zero);
    __m128i upper_hi = _mm_unpackhi_epi8(upper, zero);
    __m128i center_lo = _mm_unpacklo_epi8(center, zero);
    __m128i center_hi = _mm_unpackhi_epi8(center, zero);

    __m128i result_lo = af_blend_sse2(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m128i result_hi = af_blend_sse2(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m128i result = _mm_packus_epi16(result_lo, result_hi);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
  }
}

#ifdef X86_32

static AVS_FORCEINLINE __m64 af_blend_mmx(__m64 &upper, __m64 &center, __m64 &lower, __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask) {
  __m64 outer_tmp = _mm_add_pi16(upper, lower);
  __m64 center_tmp = _mm_mullo_pi16(center, center_weight);

  outer_tmp = _mm_mullo_pi16(outer_tmp, outer_weight);

  __m64 result = _mm_adds_pi16(center_tmp, outer_tmp);
  result = _mm_adds_pi16(result, center_tmp);
  result = _mm_adds_pi16(result, round_mask);
  return _mm_srai_pi16(result, 7);
}

static AVS_FORCEINLINE __m64 af_unpack_blend_mmx(__m64 &left, __m64 &center, __m64 &right, __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask, __m64 &zero) {
  __m64 left_lo = _mm_unpacklo_pi8(left, zero);
  __m64 left_hi = _mm_unpackhi_pi8(left, zero);
  __m64 center_lo = _mm_unpacklo_pi8(center, zero);
  __m64 center_hi = _mm_unpackhi_pi8(center, zero);
  __m64 right_lo = _mm_unpacklo_pi8(right, zero);
  __m64 right_hi = _mm_unpackhi_pi8(right, zero);

  __m64 result_lo = af_blend_mmx(left_lo, center_lo, right_lo, center_weight, outer_weight, round_mask);
  __m64 result_hi = af_blend_mmx(left_hi, center_hi, right_hi, center_weight, outer_weight, round_mask);

  return _mm_packs_pu16(result_lo, result_hi);
}

void af_vertical_mmx(BYTE* line_buf, BYTE* dstp, int height, int pitch, int width, int amount) {
  short t = (amount + 256) >> 9;
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();

  for (int y = 0; y < height-1; ++y) {
    for (int x = 0; x < width; x+= 8) {
      __m64 upper = *reinterpret_cast<const __m64*>(line_buf+x);
      __m64 center = *reinterpret_cast<const __m64*>(dstp+x);
      __m64 lower = *reinterpret_cast<const __m64*>(dstp+pitch+x);
      *reinterpret_cast<__m64*>(line_buf+x) = center;

      __m64 result = af_unpack_blend_mmx(upper, center, lower, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast<__m64*>(dstp+x) = result;
    }
    dstp += pitch;
  }

  //last line
  for (int x = 0; x < width; x+= 8) {
    __m64 upper = *reinterpret_cast<const __m64*>(line_buf+x);
    __m64 center = *reinterpret_cast<const __m64*>(dstp+x);

    __m64 upper_lo = _mm_unpacklo_pi8(upper, zero);
    __m64 upper_hi = _mm_unpackhi_pi8(upper, zero);
    __m64 center_lo = _mm_unpacklo_pi8(center, zero);
    __m64 center_hi = _mm_unpackhi_pi8(center, zero);

    __m64 result_lo = af_blend_mmx(upper_lo, center_lo, center_lo, center_weight, outer_weight, round_mask);
    __m64 result_hi = af_blend_mmx(upper_hi, center_hi, center_hi, center_weight, outer_weight, round_mask);

    __m64 result = _mm_packs_pu16(result_lo, result_hi);

    *reinterpret_cast<__m64*>(dstp+x) = result;
  }
  _mm_empty();
}

#endif


// --------------------------------------


//implementation is not in-place. Unaligned reads will be slow on older intels but who cares
void af_horizontal_rgb32_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 4;
  size_t loop_limit = width_bytes - 16;
  //int center_weight_c = int(amount*2);
  //int outer_weight_c = int(32768-amount);

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();
//#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
//#pragma warning(default: 4309)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4));
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 4));

    result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x+=16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4));
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp+x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));
    right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 4));

    result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);


    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

void af_horizontal_rgb64_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  // width is really width
  size_t width_bytes = width * 4 * sizeof(uint16_t);
  size_t loop_limit = width_bytes - 16;

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
  //#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
  //#pragma warning(default: 4309)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4*sizeof(uint16_t))); // move right by one 4*uint16_t pixelblock
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 8));

    result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x += 16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4 * sizeof(uint16_t)));
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4 * sizeof(uint16_t)));

      result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp + x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4 * sizeof(uint16_t)));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));
    right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 4 * sizeof(uint16_t)));

    result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);


    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_horizontal_rgb64_sse41(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount)
{
  // width is really width
  size_t width_bytes = width * 4 * sizeof(uint16_t);
  size_t loop_limit = width_bytes - 16;

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
  //#pragma warning(disable: 4309)
  __m128i left_mask = _mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);
  //#pragma warning(default: 4309)

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4 * sizeof(uint16_t))); // move right by one 4*uint16_t pixelblock
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 8));

    result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x += 16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4 * sizeof(uint16_t)));
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4 * sizeof(uint16_t)));

      result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4 * sizeof(uint16_t)));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));
    right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 4 * sizeof(uint16_t)));

    result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + loop_limit), result);


    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#ifdef X86_32

void af_horizontal_rgb32_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 4;
  size_t loop_limit = width_bytes - 8;
  int center_weight_c = amount*2;
  int outer_weight_c = 32768-amount;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();
  //#pragma warning(disable: 4309)
  __m64 left_mask = _mm_set_pi32(0, 0xFFFFFFFF);
  __m64 right_mask = _mm_set_pi32(0xFFFFFFFF, 0);
  //#pragma warning(default: 4309)

  __m64 center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = *reinterpret_cast<const __m64*>(srcp);
    right = *reinterpret_cast<const __m64*>(srcp + 4);
    left = _mm_or_si64(_mm_and_si64(center, left_mask), _mm_slli_si64(center, 32));

    result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

    *reinterpret_cast< __m64*>(dstp) = result;

    for (size_t x = 8; x < loop_limit; x+=8) {
      left = *reinterpret_cast<const __m64*>(srcp + x - 4);
      center = *reinterpret_cast<const __m64*>(srcp + x);
      right = *reinterpret_cast<const __m64*>(srcp + x + 4);

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast< __m64*>(dstp+x) = result;
    }

    left = *reinterpret_cast<const __m64*>(srcp + loop_limit - 4);
    center = *reinterpret_cast<const __m64*>(srcp + loop_limit);
    right = _mm_or_si64(_mm_and_si64(center, right_mask), _mm_srli_si64(center, 32));

    result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

    *reinterpret_cast< __m64*>(dstp + loop_limit) = result;

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}

#endif



static AVS_FORCEINLINE __m128i af_blend_yuy2_sse2(__m128i &left, __m128i &center, __m128i &right, __m128i &luma_mask,
                                             __m128i &center_weight, __m128i &outer_weight, __m128i &round_mask) {
  __m128i left_luma = _mm_and_si128(left, luma_mask); //0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0 0 Y-1 0 Y-2
  __m128i center_luma = _mm_and_si128(center, luma_mask); //0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0
  __m128i right_luma = _mm_and_si128(right, luma_mask); //0 Y9 0 Y8 0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2

  left_luma = _mm_or_si128(
    _mm_srli_si128(left_luma, 2), // 0  0 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1 0 Y0 0 Y-1
    _mm_slli_si128(right_luma, 6) // 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0  0 0  0 0  0
    ); // Y6..Y0 (Y-1)

  right_luma = _mm_or_si128(
    _mm_srli_si128(center_luma, 2),//0 0  0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 Y1
    _mm_slli_si128(right_luma, 2)  //0 Y8 0 Y7 0 Y6 0 Y5 0 Y4 0 Y3 0 Y2 0 0
    ); // Y8..Y1

  __m128i result_luma = af_blend_sse2(left_luma, center_luma, right_luma, center_weight, outer_weight, round_mask);

  __m128i left_chroma = _mm_srli_epi16(left, 8); //0 V 0 U 0 V 0 U
  __m128i center_chroma = _mm_srli_epi16(center, 8); //0 V 0 U 0 V 0 U
  __m128i right_chroma = _mm_srli_epi16(right, 8); //0 V 0 U 0 V 0 U

  __m128i result_chroma = af_blend_sse2(left_chroma, center_chroma, right_chroma, center_weight, outer_weight, round_mask);

  __m128i lo_lu_hi_co = _mm_packus_epi16(result_luma, result_chroma); // U3 V3 U2 V2 U1 V1 U0 V0 Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
  __m128i result = _mm_unpacklo_epi8(lo_lu_hi_co, _mm_srli_si128(lo_lu_hi_co, 8)); // U3 Y7 V3 Y6 U2 Y5 V2 Y4 U1 Y3 V1 Y2 U0 Y1 V0 Y0
  return result;
}


void af_horizontal_yuy2_sse2(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 2;
  size_t loop_limit = width_bytes - 16;

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
  __m128i left_mask_small = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0x00FF, 0);
  __m128i right_mask_small = _mm_set_epi16(0, 0x00FF, 0, 0, 0, 0, 0, 0);
  __m128i luma_mask = _mm_set1_epi16(0xFF);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m128i center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp));//V1 Y3 U1 Y2 V0 Y1 U0 Y0
    right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4));//V2 Y5 U2 Y4 V1 Y3 U1 Y2

    //todo: now this is dumb
    left = _mm_or_si128(
      _mm_and_si128(center, left_mask),
      _mm_slli_si128(center, 4)
      );//V0 Y1 U0 Y0 V0 Y1 U0 Y0
    left = _mm_or_si128(
      _mm_andnot_si128(left_mask_small, left),
      _mm_and_si128(_mm_slli_si128(center, 2), left_mask_small)
      );//V0 Y1 U0 Y0 V0 Y0 U0 Y0

    result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    _mm_store_si128(reinterpret_cast< __m128i*>(dstp), result);

    for (size_t x = 16; x < loop_limit; x+=16) {
      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x - 4));//V0 Y1 U0 Y0 V-1 Y-1 U-1 Y-2
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)); //V1 Y3 U1 Y2 V0 Y1 U0 Y0
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x + 4));//V2 Y5 U2 Y4 V1 Y3 U1 Y2

      result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

      _mm_store_si128(reinterpret_cast< __m128i*>(dstp+x), result);
    }

    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit - 4));
    center = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + loop_limit));  //V1 Y3 U1 Y2 V0 Y1 U0 Y0

    //todo: now this is dumb2
    right = _mm_or_si128(
      _mm_and_si128(center, right_mask),
      _mm_srli_si128(center, 4)
      );//V1 Y3 U1 Y2 V1 Y3 U1 Y2

    right = _mm_or_si128(
      _mm_andnot_si128(right_mask_small, right),
      _mm_and_si128(_mm_srli_si128(center, 2), right_mask_small)
      );//V1 Y3 U1 Y3 V1 Y3 U1 Y2

    result = af_blend_yuy2_sse2(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    _mm_storeu_si128(reinterpret_cast< __m128i*>(dstp + loop_limit), result);

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}



#ifdef X86_32
// -------------------------------------
// Blur/Sharpen Horizontal YUY2 MMX Code
// -------------------------------------
//
static AVS_FORCEINLINE __m64 af_blend_yuy2_mmx(__m64 &left, __m64 &center, __m64 &right, __m64 &luma_mask,
                           __m64 &center_weight, __m64 &outer_weight, __m64 &round_mask) {
  __m64 left_luma = _mm_and_si64(left, luma_mask); //0 Y1 0 Y0 0 Y-1 0 Y-2
  __m64 center_luma = _mm_and_si64(center, luma_mask); //0 Y3 0 Y2 0 Y1 0 Y0
  __m64 right_luma = _mm_and_si64(right, luma_mask); //0 Y5 0 Y4 0 Y3 0 Y2

  left_luma = _mm_or_si64(
    _mm_srli_si64(left_luma, 16), // 0 0 0 Y1 0 Y0 0 Y-1
    _mm_slli_si64(right_luma, 48) // 0 Y2 0 0 0 0 0 0
    );

  right_luma = _mm_or_si64(
    _mm_srli_si64(center_luma, 16),//0 0 0 Y3 0 Y2 0 Y1
    _mm_slli_si64(right_luma, 16)//0 Y4 0 Y3 0 Y2 0 0
    );

  __m64 result_luma = af_blend_mmx(left_luma, center_luma, right_luma, center_weight, outer_weight, round_mask);

  __m64 left_chroma = _mm_srli_pi16(left, 8); //0 V 0 U 0 V 0 U
  __m64 center_chroma = _mm_srli_pi16(center, 8); //0 V 0 U 0 V 0 U
  __m64 right_chroma = _mm_srli_pi16(right, 8); //0 V 0 U 0 V 0 U

  __m64 result_chroma = af_blend_mmx(left_chroma, center_chroma, right_chroma, center_weight, outer_weight, round_mask);

  __m64 lo_lu_hi_co = _m_packuswb(result_luma, result_chroma); // U1 V1 U0 V0 Y3 Y2 Y1 Y0
  __m64 result = _mm_unpacklo_pi8(lo_lu_hi_co, _mm_srli_si64(lo_lu_hi_co, 32)); // U1 Y3 V1 Y2 U0 Y1 V0 Y0
  return result;
}


void af_horizontal_yuy2_mmx(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, size_t amount) {
  size_t width_bytes = width * 2;
  size_t loop_limit = width_bytes - 8;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m64 left_mask = _mm_set_pi32(0, 0xFFFFFFFF);
  __m64 right_mask = _mm_set_pi32(0xFFFFFFFF, 0);
  __m64 left_mask_small = _mm_set_pi16(0, 0, 0x00FF, 0);
  __m64 right_mask_small = _mm_set_pi16(0, 0x00FF, 0, 0);
  __m64 luma_mask = _mm_set1_pi16(0xFF);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m64 center, right, left, result;

  for (size_t y = 0; y < height; ++y) {
    center = *reinterpret_cast<const __m64*>(srcp);//V1 Y3 U1 Y2 V0 Y1 U0 Y0
    right = *reinterpret_cast<const __m64*>(srcp + 4);//V2 Y5 U2 Y4 V1 Y3 U1 Y2

    //todo: now this is dumb
    left = _mm_or_si64(
      _mm_and_si64(center, left_mask),
      _mm_slli_si64(center, 32)
      );//V0 Y1 U0 Y0 V0 Y1 U0 Y0
    left = _mm_or_si64(
      _mm_andnot_si64(left_mask_small, left),
      _mm_and_si64(_mm_slli_si64(center, 16), left_mask_small)
      );//V0 Y1 U0 Y0 V0 Y0 U0 Y0

    result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    *reinterpret_cast< __m64*>(dstp) = result;

    for (size_t x = 8; x < loop_limit; x+=8) {
      left = *reinterpret_cast<const __m64*>(srcp + x - 4);//V0 Y1 U0 Y0 V-1 Y-1 U-1 Y-2
      center = *reinterpret_cast<const __m64*>(srcp + x); //V1 Y3 U1 Y2 V0 Y1 U0 Y0
      right = *reinterpret_cast<const __m64*>(srcp + x + 4);//V2 Y5 U2 Y4 V1 Y3 U1 Y2

      __m64 result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

      *reinterpret_cast< __m64*>(dstp+x) = result;
    }

    left = *reinterpret_cast<const __m64*>(srcp + loop_limit - 4);
    center = *reinterpret_cast<const __m64*>(srcp + loop_limit);  //V1 Y3 U1 Y2 V0 Y1 U0 Y0

    //todo: now this is dumb2
    right = _mm_or_si64(
      _mm_and_si64(center, right_mask),
      _mm_srli_si64(center, 32)
      );//V1 Y3 U1 Y2 V1 Y3 U1 Y2
    right = _mm_or_si64(
      _mm_andnot_si64(right_mask_small, right),
      _mm_and_si64(_mm_srli_si64(center, 16), right_mask_small)
      );//V1 Y3 U1 Y3 V1 Y3 U1 Y2

    result = af_blend_yuy2_mmx(left, center, right, luma_mask, center_weight, outer_weight, round_mask);

    *reinterpret_cast< __m64*>(dstp + loop_limit) = result;

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}


#endif


void af_horizontal_planar_sse2(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount) {
  size_t mod16_width = (width / 16) * 16;
  size_t sse_loop_limit = width == mod16_width ? mod16_width - 16 : mod16_width;
  int center_weight_c = int(amount*2);
  int outer_weight_c = int(32768-amount);

  short t = short((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi16(t);
  __m128i outer_weight = _mm_set1_epi16(64 - t);
  __m128i round_mask = _mm_set1_epi16(0x40);
  __m128i zero = _mm_setzero_si128();
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFF);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i right_mask = _mm_set_epi8(0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m128i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp));
    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+1));
    left = _mm_or_si128(_mm_and_si128(center, left_mask),  _mm_slli_si128(center, 1));

    __m128i result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+15));
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x+= 16) {
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x+1));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp+x+15));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), result);
    }

    //right border
    if(mod16_width == width) { //width is mod8, process with mmx
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp+mod16_width-16));
      right = _mm_or_si128(_mm_and_si128(center, right_mask),  _mm_srli_si128(center, 1));

      result = af_unpack_blend_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+mod16_width-16), result);
    } else { //some stuff left
      BYTE l = _mm_cvtsi128_si32(left) & 0xFF;
      af_horizontal_planar_process_line_c<uint8_t>(l, dstp+mod16_width, width-mod16_width, center_weight_c, outer_weight_c);

    }

    dstp += pitch;
  }
}

void af_horizontal_planar_uint16_t_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel) {
  size_t mod16_width = (row_size / 16) * 16;
  size_t sse_loop_limit = row_size == mod16_width ? mod16_width - 16 : mod16_width;
  int center_weight_c = int(amount * 2);
  int outer_weight_c = int(32768 - amount);

  int t = int((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i left_mask = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0xFFFF); // 0, 0, 0, 0, 0, 0, 0, FFFF
  __m128i right_mask = _mm_set_epi16(0xFFFF, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m128i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp));
    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + 2));
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 2));

    __m128i result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + (16 - 2)));
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x += 16) {
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + 2));

      result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + (16 - 2)));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }

    //right border
    if (mod16_width == row_size) { //width is mod8, process with mmx
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + mod16_width - 16));
      right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 2));

      result = af_unpack_blend_uint16_t_sse2(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + mod16_width - 16), result);
    }
    else { //some stuff left
      uint16_t l = _mm_cvtsi128_si32(left) & 0xFFFF;
      af_horizontal_planar_process_line_uint16_c(l, dstp + mod16_width, row_size - mod16_width, center_weight_c, outer_weight_c, bits_per_pixel);
    }

    dstp += pitch;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void af_horizontal_planar_uint16_t_sse41(BYTE* dstp, size_t height, size_t pitch, size_t row_size, size_t amount, int bits_per_pixel)
{
  size_t mod16_width = (row_size / 16) * 16;
  size_t sse_loop_limit = row_size == mod16_width ? mod16_width - 16 : mod16_width;
  int center_weight_c = int(amount * 2);
  int outer_weight_c = int(32768 - amount);

  int t = int((amount + 256) >> 9);
  __m128i center_weight = _mm_set1_epi32(t);
  __m128i outer_weight = _mm_set1_epi32(64 - t);
  __m128i round_mask = _mm_set1_epi32(0x40);
  __m128i zero = _mm_setzero_si128();
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i left_mask = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0xFFFF); // 0, 0, 0, 0, 0, 0, 0, FFFF
  __m128i right_mask = _mm_set_epi16(0xFFFF, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m128i left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128i center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp));
    __m128i right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + 2));
    left = _mm_or_si128(_mm_and_si128(center, left_mask), _mm_slli_si128(center, 2));

    __m128i result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + (16 - 2)));
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x += 16) {
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + x));
      right = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + 2));

      result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);

      left = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dstp + x + (16 - 2)));

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), result);
    }

    //right border
    if (mod16_width == row_size) { //width is mod8, process with mmx
      center = _mm_load_si128(reinterpret_cast<const __m128i*>(dstp + mod16_width - 16));
      right = _mm_or_si128(_mm_and_si128(center, right_mask), _mm_srli_si128(center, 2));

      result = af_unpack_blend_uint16_t_sse41(left, center, right, center_weight, outer_weight, round_mask, zero);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + mod16_width - 16), result);
    }
    else { //some stuff left
      uint16_t l = _mm_cvtsi128_si32(left) & 0xFFFF;
      af_horizontal_planar_process_line_uint16_c(l, dstp + mod16_width, row_size - mod16_width, center_weight_c, outer_weight_c, bits_per_pixel);
    }

    dstp += pitch;
  }
}

void af_horizontal_planar_float_sse2(BYTE* dstp, size_t height, size_t pitch, size_t row_size, float amount) {
  const float center_weight = amount;
  const float outer_weight = (1.0f - amount) / 2.0f;

  __m128 center_weight_simd = _mm_set1_ps(center_weight);
  __m128 outer_weight_simd = _mm_set1_ps(outer_weight);

  size_t mod16_width = (row_size / 16) * 16;
  size_t sse_loop_limit = row_size == mod16_width ? mod16_width - 16 : mod16_width;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m128i left_mask = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
  __m128i right_mask = _mm_set_epi32(0xFFFFFFFF, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m128 left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m128 center = _mm_load_ps(reinterpret_cast<const float*>(dstp));
    __m128 right = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + sizeof(float)));
    left = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(_mm_castps_si128(center), left_mask), _mm_slli_si128(_mm_castps_si128(center), sizeof(float))));

    __m128 result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);
    left = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + (16 - sizeof(float))));
    _mm_store_ps(reinterpret_cast<float*>(dstp), result);

    //main processing loop
    for (size_t x = 16; x < sse_loop_limit; x += 16) {
      center = _mm_load_ps(reinterpret_cast<const float*>(dstp + x));
      right = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + x + sizeof(float)));

      result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);

      left = _mm_loadu_ps(reinterpret_cast<const float*>(dstp + x + (16 - sizeof(float))));

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), result);
    }

    //right border
    if (mod16_width == row_size) { //width is mod8, process with mmx
      center = _mm_load_ps(reinterpret_cast<const float*>(dstp + mod16_width - 16));
      right = _mm_castsi128_ps(_mm_or_si128(_mm_and_si128(_mm_castps_si128(center), right_mask), _mm_srli_si128(_mm_castps_si128(center), sizeof(float))));

      result = af_blend_float_sse2(left, center, right, center_weight_simd, outer_weight_simd);

      _mm_store_ps(reinterpret_cast<float*>(dstp + mod16_width - 16), result);
    }
    else { //some stuff left
      float l = _mm_cvtss_f32(left);
      af_horizontal_planar_process_line_float_c(l, (float *)(dstp + mod16_width), row_size - mod16_width, center_weight, outer_weight);
    }

    dstp += pitch;
  }
}


#ifdef X86_32

void af_horizontal_planar_mmx(BYTE* dstp, size_t height, size_t pitch, size_t width, size_t amount) {
  size_t mod8_width = (width / 8) * 8;
  size_t mmx_loop_limit = width == mod8_width ? mod8_width - 8 : mod8_width;
  int center_weight_c = amount*2;
  int outer_weight_c = 32768-amount;

  short t = short((amount + 256) >> 9);
  __m64 center_weight = _mm_set1_pi16(t);
  __m64 outer_weight = _mm_set1_pi16(64 - t);
  __m64 round_mask = _mm_set1_pi16(0x40);
  __m64 zero = _mm_setzero_si64();
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
  __m64 left_mask = _mm_set_pi8(0, 0, 0, 0, 0, 0, 0, 0xFF);
  __m64 right_mask = _mm_set_pi8(0xFF, 0, 0, 0, 0, 0, 0, 0);
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  __m64 left;

  for (size_t y = 0; y < height; ++y) {
    //left border
    __m64 center = *reinterpret_cast<const __m64*>(dstp);
    __m64 right = *reinterpret_cast<const __m64*>(dstp+1);
    left = _mm_or_si64(_mm_and_si64(center, left_mask),  _mm_slli_si64(center, 8));

    __m64 result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);
    left = *reinterpret_cast<const __m64*>(dstp+7);
    *reinterpret_cast<__m64*>(dstp) = result;

    //main processing loop
    for (size_t x = 8; x < mmx_loop_limit; x+= 8) {
      center = *reinterpret_cast<const __m64*>(dstp+x);
      right = *reinterpret_cast<const __m64*>(dstp+x+1);

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);
      left = *reinterpret_cast<const __m64*>(dstp+x+7);

      *reinterpret_cast<__m64*>(dstp+x) = result;
    }

    //right border
    if(mod8_width == width) { //width is mod8, process with mmx
      center = *reinterpret_cast<const __m64*>(dstp+mod8_width-8);
      right = _mm_or_si64(_mm_and_si64(center, right_mask),  _mm_srli_si64(center, 8));

      result = af_unpack_blend_mmx(left, center, right, center_weight, outer_weight, round_mask, zero);

      *reinterpret_cast<__m64*>(dstp+mod8_width-8) = result;
    } else { //some stuff left
      BYTE l = _mm_cvtsi64_si32(left) & 0xFF;
      af_horizontal_planar_process_line_c<uint8_t>(l, dstp+mod8_width, width-mod8_width, center_weight_c, outer_weight_c);
    }

    dstp += pitch;
  }
  _mm_empty();
}


#endif









#if 0
static AVS_FORCEINLINE __m128i ts_multiply_repack_sse2(const __m128i &src, const __m128i &div, __m128i &halfdiv, __m128i &zero) {
  __m128i acc = _mm_madd_epi16(src, div);
  acc = _mm_add_epi32(acc, halfdiv);
  acc = _mm_srli_epi32(acc, 15);
  acc = _mm_packs_epi32(acc, acc);
  return _mm_packus_epi16(acc, zero);
}
#endif

static inline __m128i _mm_cmple_epu8(__m128i x, __m128i y)
{
  // Returns 0xFF where x <= y:
  return _mm_cmpeq_epi8(_mm_min_epu8(x, y), x);
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static inline __m128i _mm_cmple_epu16_sse41(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm_cmpeq_epi16(_mm_min_epu16(x, y), x);
}

static inline __m128i _mm_cmple_epu16_sse2(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

// fast: maxThreshold (255) simple accumulate for average
template<bool maxThreshold>
void accumulate_line_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div) {
  // threshold: 8 bits: 2 bytes in a word. for YUY2: luma<<8 | chroma
  // 16 bits: 16 bit value (orig threshold scaled by bits_per_pixel)
  __m128i halfdiv_vector = _mm_set1_epi16(1); // High16(0x10000)
  __m128i div_vector = _mm_set1_epi16(65536 / (planes + 1)); // mulhi
  __m128i thresh = _mm_set1_epi16(threshold);

  for (size_t x = 0; x < width; x+=16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane+x));
    __m128i zero = _mm_setzero_si128();
     __m128i low = _mm_unpacklo_epi8(current, zero);
     __m128i high = _mm_unpackhi_epi8(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi8(p, zero);
        add_high = _mm_unpackhi_epi8(p, zero);
      } else {
        auto pc = _mm_subs_epu8(p, current); // r2507-
        auto cp = _mm_subs_epu8(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu8(abs_cp, thresh);

        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c
        add_low = _mm_unpacklo_epi8(blended, zero);
        add_high = _mm_unpackhi_epi8(blended, zero);
      }

      low = _mm_adds_epu16(low, add_low);
      high = _mm_adds_epu16(high, add_high);
    }

    // non SSSE3, no _mm_mulhrs_epi16
    // (x*2 * 65536/N + 65536) / 65536 / 2
    // Hi16(x*2 * 65536/N + 1) >> 1
    low = _mm_mulhi_epu16(_mm_slli_epi16(low, 1), div_vector);
    low = _mm_adds_epu16(low, halfdiv_vector);
    low = _mm_srli_epi16(low, 1);
    high = _mm_mulhi_epu16(_mm_slli_epi16(high, 1), div_vector);
    high = _mm_adds_epu16(high, halfdiv_vector);
    high = _mm_srli_epi16(high, 1);
    __m128i acc = _mm_packus_epi16(low, high);

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane+x), acc);
  }
}

// instantiate
template void accumulate_line_sse2<false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);
template void accumulate_line_sse2<true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);


template<bool maxThreshold>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void accumulate_line_ssse3(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div)
{
  // threshold: 8 bits: 2 bytes in a word. for YUY2: luma<<8 | chroma
  // 16 bits: 16 bit value (orig threshold scaled by bits_per_pixel)
  __m128i div_vector = _mm_set1_epi16(div);

  __m128i thresh = _mm_set1_epi16(threshold);

  for (size_t x = 0; x < width; x += 16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane + x));
    __m128i zero = _mm_setzero_si128();
    __m128i low = _mm_unpacklo_epi8(current, zero);
    __m128i high = _mm_unpackhi_epi8(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi8(p, zero);
        add_high = _mm_unpackhi_epi8(p, zero);
      }
      else {
        auto pc = _mm_subs_epu8(p, current); // r2507-
        auto cp = _mm_subs_epu8(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu8(abs_cp, thresh);

        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c
        add_low = _mm_unpacklo_epi8(blended, zero);
        add_high = _mm_unpackhi_epi8(blended, zero);
      }

      low = _mm_adds_epu16(low, add_low);
      high = _mm_adds_epu16(high, add_high);
    }

      // SSSE3: _mm_mulhrs_epi16: r0 := INT16(((a0 * b0) + 0x4000) >> 15)
    low = _mm_mulhrs_epi16(low, div_vector);
    high = _mm_mulhrs_epi16(high, div_vector);
    __m128i acc = _mm_packus_epi16(low, high);

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane + x), acc);
  }
}

// instantiate
template void accumulate_line_ssse3<false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);
template void accumulate_line_ssse3<true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div);


// fast: maxThreshold (255) simple accumulate for average
template<bool maxThreshold, bool lessThan16bit>
void accumulate_line_16_sse2(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel) {
  // threshold:
  // 10-16 bits: orig threshold scaled by (bits_per_pixel-8)
  int max_pixel_value = (1 << bits_per_pixel) - 1;
  __m128i limit = _mm_set1_epi16(max_pixel_value); //used for clamping when 10-14 bits
  __m128 div_vector = _mm_set1_ps(1.0f / (planes + 1));
  __m128i thresh = _mm_set1_epi16(threshold);


  for (size_t x = 0; x < rowsize; x+=16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane+x));
    __m128i zero = _mm_setzero_si128();
    __m128i low, high;
    low = _mm_unpacklo_epi16(current, zero);
    high = _mm_unpackhi_epi16(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi16(p, zero);
        add_high = _mm_unpackhi_epi16(p, zero);
      } else {
        auto pc = _mm_subs_epu16(p, current); // r2507-
        auto cp = _mm_subs_epu16(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu16_sse2(abs_cp, thresh);
        /*
        __m128i p_greater_t = _mm_subs_epu16(p, thresh);
        __m128i c_greater_t = _mm_subs_epu16(current, thresh);
        __m128i over_thresh = _mm_or_si128(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

        __m128i leq_thresh = _mm_cmpeq_epi16(over_thresh, zero); //abs diff lower or equal to threshold
        */

        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c

        add_low = _mm_unpacklo_epi16(blended, zero);
        add_high = _mm_unpackhi_epi16(blended, zero);
      }
      low = _mm_add_epi32(low, add_low);
      high = _mm_add_epi32(high, add_high);
    }

    __m128i acc;
    //__m128 half = _mm_set1_ps(0.5f); // no need rounder, _mm_cvtps_epi32 default is round-to-nearest, unless we use _mm_cvttps_epi32 which truncates
    low = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(low), div_vector));
    high = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(high), div_vector));
    acc = _MM_PACKUS_EPI32(low, high); // sse4.1 simul
    if (lessThan16bit)
      acc = _MM_MIN_EPU16(acc, limit); // sse4.1 simul

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane+x), acc);
  }
}


// instantiate
template void accumulate_line_16_sse2<false, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse2<false, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse2<true, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse2<true, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);


// fast: maxThreshold (255) simple accumulate for average
template<bool maxThreshold, bool lessThan16bit>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
void accumulate_line_16_sse41(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel)
{
  // threshold:
  // 10-16 bits: orig threshold scaled by (bits_per_pixel-8)
  int max_pixel_value = (1 << bits_per_pixel) - 1;
  __m128i limit = _mm_set1_epi16(max_pixel_value); //used for clamping when 10-14 bits
  // halfdiv_vector = _mm_set1_epi32(1); // n/a
  __m128 div_vector = _mm_set1_ps(1.0f / (planes + 1));
  __m128i thresh = _mm_set1_epi16(threshold);


  for (size_t x = 0; x < rowsize; x += 16) {
    __m128i current = _mm_load_si128(reinterpret_cast<const __m128i*>(c_plane + x));
    __m128i zero = _mm_setzero_si128();
    __m128i low, high;
    low = _mm_unpacklo_epi16(current, zero);
    high = _mm_unpackhi_epi16(current, zero);

    for (int plane = planes - 1; plane >= 0; --plane) {
      __m128i p = _mm_load_si128(reinterpret_cast<const __m128i*>(planeP[plane] + x));

      __m128i add_low, add_high;
      if (maxThreshold) {
        // fast: simple accumulate for average
        add_low = _mm_unpacklo_epi16(p, zero);
        add_high = _mm_unpackhi_epi16(p, zero);
      }
      else {
        auto pc = _mm_subs_epu16(p, current); // r2507-
        auto cp = _mm_subs_epu16(current, p);
        auto abs_cp = _mm_or_si128(pc, cp);
        auto leq_thresh = _mm_cmple_epu16_sse41(abs_cp, thresh);
        /*
        __m128i p_greater_t = _mm_subs_epu16(p, thresh);
        __m128i c_greater_t = _mm_subs_epu16(current, thresh);
        __m128i over_thresh = _mm_or_si128(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

        __m128i leq_thresh = _mm_cmpeq_epi16(over_thresh, zero); //abs diff lower or equal to threshold
        */

        __m128i andop = _mm_and_si128(leq_thresh, p);
        __m128i andnop = _mm_andnot_si128(leq_thresh, current);
        __m128i blended = _mm_or_si128(andop, andnop); //abs(p-c) <= thresh ? p : c

        add_low = _mm_unpacklo_epi16(blended, zero);
        add_high = _mm_unpackhi_epi16(blended, zero);
      }
      low = _mm_add_epi32(low, add_low);
      high = _mm_add_epi32(high, add_high);
    }

    __m128i acc;
    //__m128 half = _mm_set1_ps(0.5f); // no need rounder, _mm_cvtps_epi32 default is round-to-nearest, unless we use _mm_cvttps_epi32 which truncates
    low = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(low), div_vector));
    high = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(high), div_vector));
    acc = _mm_packus_epi32(low, high); // sse41
    if (lessThan16bit)
      acc = _mm_min_epu16(acc, limit); // sse41

    _mm_store_si128(reinterpret_cast<__m128i*>(c_plane + x), acc);
  }
}

// instantiate
// fast: maxThreshold (255) simple accumulate for average
template void accumulate_line_16_sse41<false, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse41<false, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse41<true, false>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);
template void accumulate_line_16_sse41<true, true>(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, int threshold, int div, int bits_per_pixel);


#ifdef X86_32

static AVS_FORCEINLINE __m64 ts_multiply_repack_mmx(const __m64 &src, const __m64 &div, __m64 &halfdiv, __m64 &zero) {
  __m64 acc = _mm_madd_pi16(src, div);
  acc = _mm_add_pi32(acc, halfdiv);
  acc = _mm_srli_pi32(acc, 15);
  acc = _mm_packs_pi32(acc, acc);
  return _mm_packs_pu16(acc, zero);
}

//thresh and div must always be 16-bit integers. Thresh is 2 packed bytes and div is a single 16-bit number
void accumulate_line_mmx(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, int threshold, int div) {
  __m64 halfdiv_vector = _mm_set1_pi32(16384);
  __m64 div_vector = _mm_set1_pi16(div);

  for (size_t x = 0; x < width; x+=8) {
    __m64 current = *reinterpret_cast<const __m64*>(c_plane+x);
    __m64 zero = _mm_setzero_si64();
    __m64 low = _mm_unpacklo_pi8(current, zero);
    __m64 high = _mm_unpackhi_pi8(current, zero);
    __m64 thresh = _mm_set1_pi16(threshold);

    for(int plane = planes-1; plane >= 0; --plane) {
      __m64 p = *reinterpret_cast<const __m64*>(planeP[plane]+x);

      __m64 p_greater_t = _mm_subs_pu8(p, thresh);
      __m64 c_greater_t = _mm_subs_pu8(current, thresh);
      __m64 over_thresh = _mm_or_si64(p_greater_t, c_greater_t); //abs(p-c) - t == (satsub(p,c) | satsub(c,p)) - t =kinda= satsub(p,t) | satsub(c,t)

      __m64 leq_thresh = _mm_cmpeq_pi8(over_thresh, zero); //abs diff lower or equal to threshold

      __m64 andop = _mm_and_si64(leq_thresh, p);
      __m64 andnop = _mm_andnot_si64(leq_thresh, current);
      __m64 blended = _mm_or_si64(andop, andnop); //abs(p-c) <= thresh ? p : c

      __m64 add_low = _mm_unpacklo_pi8(blended, zero);
      __m64 add_high = _mm_unpackhi_pi8(blended, zero);

      low = _mm_adds_pu16(low, add_low);
      high = _mm_adds_pu16(high, add_high);
    }

    __m64 low_low   = ts_multiply_repack_mmx(_mm_unpacklo_pi16(low, zero), div_vector, halfdiv_vector, zero);
    __m64 low_high  = ts_multiply_repack_mmx(_mm_unpackhi_pi16(low, zero), div_vector, halfdiv_vector, zero);
    __m64 high_low  = ts_multiply_repack_mmx(_mm_unpacklo_pi16(high, zero), div_vector, halfdiv_vector, zero);
    __m64 high_high = ts_multiply_repack_mmx(_mm_unpackhi_pi16(high, zero), div_vector, halfdiv_vector, zero);

    low = _mm_unpacklo_pi16(low_low, low_high);
    high = _mm_unpacklo_pi16(high_low, high_high);

   __m64 acc = _mm_unpacklo_pi32(low, high);

    *reinterpret_cast<__m64*>(c_plane+x) = acc;
  }
  _mm_empty();
}

#endif



// may also used from conditionalfunctions
// packed rgb template masks out alpha plane for RGB32
template<bool packedRGB3264>
int calculate_sad_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod16_width = rowsize / 16 * 16;
  int result = 0;
  __m128i sum = _mm_setzero_si128();

  __m128i rgb_mask;
  if (packedRGB3264) {
    rgb_mask = _mm_set1_epi32(0x00FFFFFF);
  }

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+=16) {
      __m128i cur = _mm_load_si128(reinterpret_cast<const __m128i*>(cur_ptr + x));
      __m128i other = _mm_load_si128(reinterpret_cast<const __m128i*>(other_ptr + x));
      if (packedRGB3264) {
        cur = _mm_and_si128(cur, rgb_mask);  // mask out A channel
        other = _mm_and_si128(other, rgb_mask);
      }
      __m128i sad = _mm_sad_epu8(cur, other);
      sum = _mm_add_epi32(sum, sad);
    }
    if (mod16_width != rowsize) {
      if (packedRGB3264)
        for (size_t x = mod16_width / 4; x < rowsize / 4; x += 4) {
          result += std::abs(cur_ptr[x*4+0] - other_ptr[x*4+0]) +
            std::abs(cur_ptr[x*4+1] - other_ptr[x*4+1]) +
            std::abs(cur_ptr[x*4+2] - other_ptr[x*4+2]);
          // no alpha
        }
      else
        for (size_t x = mod16_width; x < rowsize; ++x) {
          result += std::abs(cur_ptr[x] - other_ptr[x]);
        }
    }
    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  __m128i upper = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
  sum = _mm_add_epi32(sum, upper);
  result += _mm_cvtsi128_si32(sum);
  return result;
}

// instantiate
template int calculate_sad_sse2<false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template int calculate_sad_sse2<true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);


// works for uint8_t, but there is a specific, bit faster function above
// also used from conditionalfunctions
// packed rgb template masks out alpha plane for RGB32/RGB64
template<typename pixel_t, bool packedRGB3264>
int64_t calculate_sad_8_or_16_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod16_width = rowsize / 16 * 16;

  __m128i zero = _mm_setzero_si128();
  int64_t totalsum = 0; // fullframe SAD exceeds int32 at 8+ bit

  __m128i rgb_mask;
  if (packedRGB3264) {
    if constexpr(sizeof(pixel_t) == 1)
      rgb_mask = _mm_set1_epi32(0x00FFFFFF);
    else
      rgb_mask = _mm_set_epi32(0x0000FFFF,0xFFFFFFFF,0x0000FFFF,0xFFFFFFFF);
  }

  for ( size_t y = 0; y < height; y++ )
  {
    __m128i sum = _mm_setzero_si128(); // for one row int is enough
    for ( size_t x = 0; x < mod16_width; x+=16 )
    {
      __m128i src1, src2;
      src1 = _mm_load_si128((__m128i *) (cur_ptr + x));   // 16 bytes or 8 words
      src2 = _mm_load_si128((__m128i *) (other_ptr + x));
      if (packedRGB3264) {
        src1 = _mm_and_si128(src1, rgb_mask); // mask out A channel
        src2 = _mm_and_si128(src2, rgb_mask);
      }
      if constexpr(sizeof(pixel_t) == 1) {
        // this is uint_16 specific, but leave here for sample
        sum = _mm_add_epi32(sum, _mm_sad_epu8(src1, src2)); // sum0_32, 0, sum1_32, 0
      }
      else if constexpr(sizeof(pixel_t) == 2) {
        __m128i greater_t = _mm_subs_epu16(src1, src2); // unsigned sub with saturation
        __m128i smaller_t = _mm_subs_epu16(src2, src1);
        __m128i absdiff = _mm_or_si128(greater_t, smaller_t); //abs(s1-s2)  == (satsub(s1,s2) | satsub(s2,s1))
        // 8 x uint16 absolute differences
        sum = _mm_add_epi32(sum, _mm_unpacklo_epi16(absdiff, zero));
        sum = _mm_add_epi32(sum, _mm_unpackhi_epi16(absdiff, zero));
        // sum0_32, sum1_32, sum2_32, sum3_32
      }
    }
    // summing up partial sums
    if constexpr(sizeof(pixel_t) == 2) {
      // at 16 bits: we have 4 integers for sum: a0 a1 a2 a3
      __m128i a0_a1 = _mm_unpacklo_epi32(sum, zero); // a0 0 a1 0
      __m128i a2_a3 = _mm_unpackhi_epi32(sum, zero); // a2 0 a3 0
      sum = _mm_add_epi32( a0_a1, a2_a3 ); // a0+a2, 0, a1+a3, 0
      /* SSSE3: told to be not too fast
      sum = _mm_hadd_epi32(sum, zero);  // A1+A2, B1+B2, 0+0, 0+0
      sum = _mm_hadd_epi32(sum, zero);  // A1+A2+B1+B2, 0+0+0+0, 0+0+0+0, 0+0+0+0
      */
    }

    // sum here: two 32 bit partial result: sum1 0 sum2 0
    __m128i sum_hi = _mm_unpackhi_epi64(sum, zero);
    // or: __m128i sum_hi = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
    sum = _mm_add_epi32(sum, sum_hi);
    int rowsum = _mm_cvtsi128_si32(sum);

    // rest
    if (mod16_width != rowsize) {
      if (packedRGB3264)
        for (size_t x = mod16_width / sizeof(pixel_t) / 4; x < rowsize / sizeof(pixel_t) / 4; x += 4) {
          rowsum += std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+0] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+0]) +
            std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+1] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+1]) +
            std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x*4+2] - reinterpret_cast<const pixel_t *>(other_ptr)[x*4+2]);
          // no alpha
        }
      else
        for (size_t x = mod16_width / sizeof(pixel_t); x < rowsize / sizeof(pixel_t); ++x) {
          rowsum += std::abs(reinterpret_cast<const pixel_t *>(cur_ptr)[x] - reinterpret_cast<const pixel_t *>(other_ptr)[x]);
        }
    }

    totalsum += rowsum;

    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  return totalsum;
}

// instantiate
template int64_t calculate_sad_8_or_16_sse2<uint8_t, false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template int64_t calculate_sad_8_or_16_sse2<uint8_t, true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template int64_t calculate_sad_8_or_16_sse2<uint16_t, false>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template int64_t calculate_sad_8_or_16_sse2<uint16_t, true>(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);


#ifdef X86_32
int calculate_sad_isse(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  size_t mod8_width = rowsize / 8 * 8;
  int result = 0;
  __m64 sum = _mm_setzero_si64();
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 cur = *reinterpret_cast<const __m64*>(cur_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      __m64 sad = _mm_sad_pu8(cur, other);
      sum = _mm_add_pi32(sum, sad);
    }
    if (mod8_width != rowsize) {
      for (size_t x = mod8_width; x < rowsize; ++x) {
        result += std::abs(cur_ptr[x] - other_ptr[x]);
      }
    }

    cur_ptr += cur_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}
#endif







