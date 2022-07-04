
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
#include <cstdint>
#include <emmintrin.h>
#include <algorithm>

// sum: sad with zero
double get_sum_of_pixels_sse2(const uint8_t* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod16_width = width / 16 * 16;
  int64_t result = 0;
  __m128i sum = _mm_setzero_si128();
  __m128i zero = _mm_setzero_si128();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+=16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i sad = _mm_sad_epu8(src, zero);
      sum = _mm_add_epi32(sum, sad);
    }

    for (size_t x = mod16_width; x < width; ++x) {
      result += srcp[x];
    }

    srcp += pitch;
  }
  __m128i upper = _mm_castps_si128(_mm_movehl_ps(_mm_setzero_ps(), _mm_castsi128_ps(sum)));
  sum = _mm_add_epi32(sum, upper);
  result += _mm_cvtsi128_si32(sum);
  return (double)result;
}

#ifdef X86_32
double get_sum_of_pixels_isse(const uint8_t* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod8_width = width / 8 * 8;
  int64_t result = 0;
  __m64 sum = _mm_setzero_si64();
  __m64 zero = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 sad = _mm_sad_pu8(src, zero);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += srcp[x];
    }

    srcp += pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return (double)result;
}

size_t get_sad_isse(const uint8_t* src_ptr, const uint8_t* other_ptr, size_t height, size_t width, size_t src_pitch, size_t other_pitch) {
  size_t mod8_width = width / 8 * 8;
  size_t result = 0;
  __m64 sum = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 src = *reinterpret_cast<const __m64*>(src_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      __m64 sad = _mm_sad_pu8(src, other);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += std::abs(src_ptr[x] - other_ptr[x]);
    }

    src_ptr += src_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}

size_t get_sad_rgb_isse(const uint8_t* src_ptr, const uint8_t* other_ptr, size_t height, size_t width, size_t src_pitch, size_t other_pitch) {
  size_t mod8_width = width / 8 * 8;
  size_t result = 0;
  __m64 rgb_mask = _mm_set1_pi32(0x00FFFFFF);
  __m64 sum = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 src = *reinterpret_cast<const __m64*>(src_ptr + x);
      __m64 other = *reinterpret_cast<const __m64*>(other_ptr + x);
      src = _mm_and_si64(src, rgb_mask);
      other = _mm_and_si64(other, rgb_mask);
      __m64 sad = _mm_sad_pu8(src, other);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += std::abs(src_ptr[x] - other_ptr[x]);
    }

    src_ptr += src_pitch;
    other_ptr += other_pitch;
  }
  result += _mm_cvtsi64_si32(sum);
  _mm_empty();
  return result;
}

#endif

