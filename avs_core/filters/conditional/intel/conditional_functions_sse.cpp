
// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://avisynth.nl

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

// Intrinsics base header + really required extension headers
#if defined(_MSC_VER)
#include <intrin.h> // MSVC
#else 
#include <x86intrin.h> // GCC/MinGW/Clang/LLVM
#endif

#include <algorithm>

// sum: sad with zero
double get_sum_of_pixels_sse2(const uint8_t* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod16_width = width / 16 * 16;
  int64_t result = 0; // fullframe sum exceeds int32 for large frames
  __m128i sum = _mm_setzero_si128();
  __m128i zero = _mm_setzero_si128();

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+=16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i sad = _mm_sad_epu8(src, zero);
      sum = _mm_add_epi64(sum, sad); // sad provides 64bit lanes
    }

    for (size_t x = mod16_width; x < width; ++x) {
      result += srcp[x];
    }

    srcp += pitch;
  }
  sum = _mm_add_epi64(sum, _mm_unpackhi_epi64(sum, sum)); // lane0 + lane1
  int64_t val;
  _mm_storel_epi64((__m128i*)&val, sum);
  return (double)(result + val);
}

// 10-16 bit: sum of uint16_t pixels.
// Using pivot unsigned-signed trick to use madd, with post-correction
double get_sum_of_pixels_uint16_sse2(const uint8_t* srcp, size_t height, size_t width, size_t pitch) {
  const size_t rowsize = width * sizeof(uint16_t);
  const size_t mod16_rowsize = rowsize / 16 * 16;
  const size_t mod8_width = mod16_rowsize / sizeof(uint16_t);
  int64_t result = 0; // fullframe sum exceeds int32 for large frames
  const __m128i bias = _mm_set1_epi16((short)0x8000); // sign flip
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i zero = _mm_setzero_si128();
  __m128i total = _mm_setzero_si128(); // 2x int64

  for (size_t y = 0; y < height; ++y) {
    __m128i rowsum = _mm_setzero_si128(); // 4x int32
    // 8 pixels per iteration, 16 bytes, 8x uint16_t
    for (size_t x = 0; x < mod16_rowsize; x += 16) {
      __m128i src = _mm_xor_si128(_mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x)), bias);
      rowsum = _mm_add_epi32(rowsum, _mm_madd_epi16(src, ones));
    }
    // sign extend the 4 x int32 row sums to int64 and add to the total
    const __m128i sign = _mm_cmpgt_epi32(zero, rowsum);
    total = _mm_add_epi64(total, _mm_unpacklo_epi32(rowsum, sign)); // sign extend 2xint32.lo to 2xint64
    total = _mm_add_epi64(total, _mm_unpackhi_epi32(rowsum, sign));// sign extend 2xint32.hi to 2xint64
    result += (int64_t)32768 * (int64_t)mod8_width; // correct pivot bias back
    // scalar tail
    const uint16_t* srcp16 = reinterpret_cast<const uint16_t*>(srcp);
    for (size_t x = mod8_width; x < width; ++x) {
      result += srcp16[x];
    }

    srcp += pitch;
  }
  total = _mm_add_epi64(total, _mm_unpackhi_epi64(total, total)); // lane0 + lane1
  int64_t val;
  _mm_storel_epi64((__m128i*)&val, total);
  return (double)(result + val);
}

#ifdef X86_32
double get_sum_of_pixels_isse(const uint8_t* srcp, size_t height, size_t width, size_t pitch) {
  size_t mod8_width = width / 8 * 8;
  int64_t result = 0; // fullframe sum exceeds int32 for large frames
  __m64 zero = _mm_setzero_si64();

  for (size_t y = 0; y < height; ++y) {
    __m64 sum = _mm_setzero_si64(); // for one row int32 is enough
    for (size_t x = 0; x < mod8_width; x+=8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 sad = _mm_sad_pu8(src, zero);
      sum = _mm_add_pi32(sum, sad);
    }

    for (size_t x = mod8_width; x < width; ++x) {
      result += srcp[x];
    }
    result += _mm_cvtsi64_si32(sum);

    srcp += pitch;
  }
  _mm_empty();
  return (double)result;
}


#endif

