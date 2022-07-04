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
#include <emmintrin.h>


//todo: think of a way to do this with pavgb
static AVS_FORCEINLINE __m128i vertical_reduce_sse2_blend(__m128i& src, __m128i& src_next, __m128i& src_next2, __m128i& zero, __m128i& two) {
  __m128i src_unpck_lo = _mm_unpacklo_epi8(src, zero);
  __m128i src_unpck_hi = _mm_unpackhi_epi8(src, zero);

  __m128i src_next_unpck_lo = _mm_unpacklo_epi8(src_next, zero);
  __m128i src_next_unpck_hi = _mm_unpackhi_epi8(src_next, zero);

  __m128i src_next2_unpck_lo = _mm_unpacklo_epi8(src_next2, zero);
  __m128i src_next2_unpck_hi = _mm_unpackhi_epi8(src_next2, zero);

  __m128i acc_lo = _mm_adds_epu16(src_next_unpck_lo, src_next_unpck_lo);
  acc_lo = _mm_adds_epu16(acc_lo, src_unpck_lo);
  acc_lo = _mm_adds_epu16(acc_lo, src_next2_unpck_lo);

  __m128i acc_hi = _mm_adds_epu16(src_next_unpck_hi, src_next_unpck_hi);
  acc_hi = _mm_adds_epu16(acc_hi, src_unpck_hi);
  acc_hi = _mm_adds_epu16(acc_hi, src_next2_unpck_hi);

  acc_lo = _mm_adds_epu16(acc_lo, two);
  acc_hi = _mm_adds_epu16(acc_hi, two);

  acc_lo = _mm_srai_epi16(acc_lo, 2);
  acc_hi = _mm_srai_epi16(acc_hi, 2);

  return _mm_packus_epi16(acc_lo, acc_hi);
}

void vertical_reduce_sse2(BYTE* dstp, const BYTE* srcp, int dst_pitch, int src_pitch, size_t width, size_t height) {
  const BYTE* srcp_next = srcp + src_pitch;
  const BYTE* srcp_next2 = srcp + src_pitch * 2;
  __m128i zero = _mm_setzero_si128();
  __m128i two = _mm_set1_epi16(2);

  for (size_t y = 0; y < height - 1; ++y) {
    for (size_t x = 0; x < width; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp_next + x));
      __m128i src_next2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp_next2 + x));

      __m128i avg = vertical_reduce_sse2_blend(src, src_next, src_next2, zero, two);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch * 2;
    srcp_next += src_pitch * 2;
    srcp_next2 += src_pitch * 2;
  }
  //last line
  for (size_t x = 0; x < width; x += 16) {
    __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
    __m128i src_next = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp_next + x));

    __m128i avg = vertical_reduce_sse2_blend(src, src_next, src_next, zero, two);

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), avg);
  }
}

#ifdef X86_32

//todo: think of a way to do this with pavgb
static AVS_FORCEINLINE __m64 vertical_reduce_mmx_blend(__m64& src, __m64& src_next, __m64& src_next2, __m64& zero, __m64& two) {
  __m64 src_unpck_lo = _mm_unpacklo_pi8(src, zero);
  __m64 src_unpck_hi = _mm_unpackhi_pi8(src, zero);

  __m64 src_next_unpck_lo = _mm_unpacklo_pi8(src_next, zero);
  __m64 src_next_unpck_hi = _mm_unpackhi_pi8(src_next, zero);

  __m64 src_next2_unpck_lo = _mm_unpacklo_pi8(src_next2, zero);
  __m64 src_next2_unpck_hi = _mm_unpackhi_pi8(src_next2, zero);

  __m64 acc_lo = _mm_adds_pu16(src_next_unpck_lo, src_next_unpck_lo);
  acc_lo = _mm_adds_pu16(acc_lo, src_unpck_lo);
  acc_lo = _mm_adds_pu16(acc_lo, src_next2_unpck_lo);

  __m64 acc_hi = _mm_adds_pu16(src_next_unpck_hi, src_next_unpck_hi);
  acc_hi = _mm_adds_pu16(acc_hi, src_unpck_hi);
  acc_hi = _mm_adds_pu16(acc_hi, src_next2_unpck_hi);

  acc_lo = _mm_adds_pu16(acc_lo, two);
  acc_hi = _mm_adds_pu16(acc_hi, two);

  acc_lo = _mm_srai_pi16(acc_lo, 2);
  acc_hi = _mm_srai_pi16(acc_hi, 2);

  return _mm_packs_pu16(acc_lo, acc_hi);
}

void vertical_reduce_mmx(BYTE* dstp, const BYTE* srcp, int dst_pitch, int src_pitch, size_t width, size_t height) {
  const BYTE* srcp_next = srcp + src_pitch;
  const BYTE* srcp_next2 = srcp + src_pitch * 2;
  __m64 zero = _mm_setzero_si64();
  __m64 two = _mm_set1_pi16(2);

  size_t mod8_width = width / 8 * 8;

  for (size_t y = 0; y < height - 1; ++y) {
    for (size_t x = 0; x < mod8_width; x += 8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 src_next = *reinterpret_cast<const __m64*>(srcp_next + x);
      __m64 src_next2 = *reinterpret_cast<const __m64*>(srcp_next2 + x);

      __m64 avg = vertical_reduce_mmx_blend(src, src_next, src_next2, zero, two);

      *reinterpret_cast<__m64*>(dstp + x) = avg;
    }

    if (mod8_width != width) {
      size_t x = width - 8;
      __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
      __m64 src_next = *reinterpret_cast<const __m64*>(srcp_next + x);
      __m64 src_next2 = *reinterpret_cast<const __m64*>(srcp_next2 + x);

      __m64 avg = vertical_reduce_mmx_blend(src, src_next, src_next2, zero, two);

      *reinterpret_cast<__m64*>(dstp + x) = avg;
    }

    dstp += dst_pitch;
    srcp += src_pitch * 2;
    srcp_next += src_pitch * 2;
    srcp_next2 += src_pitch * 2;
  }
  //last line
  for (size_t x = 0; x < mod8_width; x += 8) {
    __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
    __m64 src_next = *reinterpret_cast<const __m64*>(srcp_next + x);

    __m64 avg = vertical_reduce_mmx_blend(src, src_next, src_next, zero, two);

    *reinterpret_cast<__m64*>(dstp + x) = avg;
  }

  if (mod8_width != width) {
    size_t x = width - 8;
    __m64 src = *reinterpret_cast<const __m64*>(srcp + x);
    __m64 src_next = *reinterpret_cast<const __m64*>(srcp_next + x);

    __m64 avg = vertical_reduce_mmx_blend(src, src_next, src_next, zero, two);

    *reinterpret_cast<__m64*>(dstp + x) = avg;
  }

  _mm_empty();
}
#endif



