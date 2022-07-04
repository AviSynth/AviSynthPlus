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

#include <avs/alignment.h>
#ifdef AVS_WINDOWS
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifndef _mm256_set_m128
#define _mm256_set_m128(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

#include "convert_rgb_avx2.h"

// minimum width: 48*2 bytes
template<typename pixel_t, bool targetHasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_rgb_to_rgbp_avx2(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel)
{
  // RGB24: 2x3x16 bytes cycle, 2x16*(RGB) 8bit pixels
  // RGB48: 2x3x16 bytes cycle, 2x8*(RGB) 16bit pixels
  // 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF
  // BGRBGRBGRBGRBGRB GRBGRBGRBGRBGRBG RBGRBGRBGRBGRBGR // 8 bit
  // B G R B G R B G  R B G R B G R B  G R B G R B G R  // 16 bit
  // 1111111111112222 2222222233333333 3333444444444444

  constexpr int pixels_at_a_time = (sizeof(pixel_t) == 1) ? 32 : 16;
  const int wmod = (width / pixels_at_a_time) * pixels_at_a_time; // 8 pixels for 8 bit, 4 pixels for 16 bit
  __m256i mask;
  if constexpr(sizeof(pixel_t) == 1)
    mask = _mm256_set_epi8(15, 14, 13, 12, 11, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 0,
      15, 14, 13, 12, 11, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 0); // same for both lanes
  else
    mask = _mm256_set_epi8(15, 14, 13, 12, 11, 10, 5, 4, 9, 8, 3, 2, 7, 6, 1, 0,
      15, 14, 13, 12, 11, 10, 5, 4, 9, 8, 3, 2, 7, 6, 1, 0); // same for both lanes

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m256i max_pixel_value;
  if constexpr(sizeof(pixel_t) == 1)
    max_pixel_value = _mm256_set1_epi8(0xFF);
  else
    max_pixel_value = _mm256_set1_epi16((1 << bits_per_pixel) - 1); // bits_per_pixel is 16
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// read-optimized
#define SRC_ADDRESS_ADVANCES
#ifdef SRC_ADDRESS_ADVANCES
  srcp -= src_pitch * (height - 1); // source packed RGB is upside down
  dstp[0] += dst_pitch[0] * (height - 1);
  dstp[1] += dst_pitch[1] * (height - 1);
  dstp[2] += dst_pitch[2] * (height - 1);
  if (targetHasAlpha)
    dstp[3] += dst_pitch[3] * (height - 1);
#endif

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wmod; x += pixels_at_a_time) {
      auto BGRA_1_Lo48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time));
      auto BGRA_2_Lo48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 16));
      auto BGRA_3_Lo48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 32));

      auto BGRA_1_Hi48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 0 + 48));
      auto BGRA_2_Hi48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 16 + 48));
      auto BGRA_3_Hi48 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 32 + 48));

      auto BGRA_1 = _mm256_set_m128i(BGRA_1_Hi48, BGRA_1_Lo48);
      auto BGRA_2 = _mm256_set_m128i(BGRA_2_Hi48, BGRA_2_Lo48);
      auto BGRA_3 = _mm256_set_m128i(BGRA_3_Hi48, BGRA_3_Lo48);

      auto pack_lo = _mm256_shuffle_epi8(BGRA_1, mask); // 111111111111: BBBBGGGGRRRR and rest: BGRB | BBGGRR and rest: BBGG
      BGRA_1 = _mm256_alignr_epi8(BGRA_2, BGRA_1, 12);
      auto pack_hi = _mm256_shuffle_epi8(BGRA_1, mask); // 222222222222: BBBBGGGGRRRR | BBGGRR
      BGRA_2 = _mm256_alignr_epi8(BGRA_3, BGRA_2, 8);
      auto pack_lo2 = _mm256_shuffle_epi8(BGRA_2, mask); // 333333333333: BBBBGGGGRRRR | BBGGRR
      BGRA_3 = _mm256_srli_si256(BGRA_3, 4); // to use the same mask
      auto pack_hi2 = _mm256_shuffle_epi8(BGRA_3, mask); // 444444444444: BBBBGGGGRRRR | BBGGRR

      auto BG1 = _mm256_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      auto BG2 = _mm256_unpacklo_epi32(pack_lo2, pack_hi2);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      auto RA1 = _mm256_unpackhi_epi32(pack_lo, pack_hi);   // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      auto RA2 = _mm256_unpackhi_epi32(pack_lo2, pack_hi2);  // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      auto B = _mm256_unpacklo_epi64(BG1, BG2);
      _mm256_stream_si256(reinterpret_cast<__m256i *>(dstp[1] + x * sizeof(pixel_t)), B); // B
      auto G = _mm256_unpackhi_epi64(BG1, BG2);
      _mm256_stream_si256(reinterpret_cast<__m256i *>(dstp[0] + x * sizeof(pixel_t)), G); // G
      auto R = _mm256_unpacklo_epi64(RA1, RA2);
      _mm256_stream_si256(reinterpret_cast<__m256i *>(dstp[2] + x * sizeof(pixel_t)), R); // R
      if (targetHasAlpha)
        _mm256_stream_si256(reinterpret_cast<__m256i *>(dstp[3] + x * sizeof(pixel_t)), max_pixel_value); // A
    }
    // rest, unaligned but simd
    if (wmod != width) {
      size_t x = (width - pixels_at_a_time);
      auto BGRA_1_Lo48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time));
      auto BGRA_2_Lo48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 16));
      auto BGRA_3_Lo48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 32));

      auto BGRA_1_Hi48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 0 + 48));
      auto BGRA_2_Hi48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 16 + 48));
      auto BGRA_3_Hi48 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 * 48 / pixels_at_a_time + 32 + 48));

      auto BGRA_1 = _mm256_set_m128i(BGRA_1_Hi48, BGRA_1_Lo48);
      auto BGRA_2 = _mm256_set_m128i(BGRA_2_Hi48, BGRA_2_Lo48);
      auto BGRA_3 = _mm256_set_m128i(BGRA_3_Hi48, BGRA_3_Lo48);

      auto pack_lo = _mm256_shuffle_epi8(BGRA_1, mask); // 111111111111: BBBBGGGGRRRR and rest: BGRB | BBGGRR and rest: BBGG
      BGRA_1 = _mm256_alignr_epi8(BGRA_2, BGRA_1, 12);
      auto pack_hi = _mm256_shuffle_epi8(BGRA_1, mask); // 222222222222: BBBBGGGGRRRR | BBGGRR
      BGRA_2 = _mm256_alignr_epi8(BGRA_3, BGRA_2, 8);
      auto pack_lo2 = _mm256_shuffle_epi8(BGRA_2, mask); // 333333333333: BBBBGGGGRRRR | BBGGRR
      BGRA_3 = _mm256_srli_si256(BGRA_3, 4); // to use the same mask
      auto pack_hi2 = _mm256_shuffle_epi8(BGRA_3, mask); // 444444444444: BBBBGGGGRRRR | BBGGRR

      auto BG1 = _mm256_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      auto BG2 = _mm256_unpacklo_epi32(pack_lo2, pack_hi2);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      auto RA1 = _mm256_unpackhi_epi32(pack_lo, pack_hi);   // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      auto RA2 = _mm256_unpackhi_epi32(pack_lo2, pack_hi2);  // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      auto B = _mm256_unpacklo_epi64(BG1, BG2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp[1] + x * sizeof(pixel_t)), B); // B
      auto G = _mm256_unpackhi_epi64(BG1, BG2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp[0] + x * sizeof(pixel_t)), G); // G
      auto R = _mm256_unpacklo_epi64(RA1, RA2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp[2] + x * sizeof(pixel_t)), R); // R
      if (targetHasAlpha)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp[3] + x * sizeof(pixel_t)), max_pixel_value); // A
    }
#ifdef SRC_ADDRESS_ADVANCES
    srcp += src_pitch; // source packed RGB is upside down
    dstp[0] -= dst_pitch[0];
    dstp[1] -= dst_pitch[1];
    dstp[2] -= dst_pitch[2];
    if (targetHasAlpha)
      dstp[3] -= dst_pitch[3];
#else
    srcp -= src_pitch; // source packed RGB is upside down
    dstp[0] += dst_pitch[0];
    dstp[1] += dst_pitch[1];
    dstp[2] += dst_pitch[2];
    if (targetHasAlpha)
      dstp[3] += dst_pitch[3];
#endif
  }
#undef SRC_ADDRESS_ADVANCES
}

// Instantiate them
template void convert_rgb_to_rgbp_avx2<uint8_t, false>(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_avx2<uint8_t, true>(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_avx2<uint16_t, false>(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_avx2<uint16_t, true>(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
