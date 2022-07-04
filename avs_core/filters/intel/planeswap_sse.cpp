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


// Avisynth filter:  Swap planes
// by Klaus Post
// adapted by Richard Berg (avisynth-dev@richardberg.net)
// iSSE code by Ian Brabham

#include <avs/config.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif
#include "planeswap_sse.h"
#include <emmintrin.h>
#include <tmmintrin.h>
#include "stdint.h"



/**************************************
 *  Swap - swaps UV on planar maps
 **************************************/

void yuy2_swap_sse2(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch, int width, int height)
{
  const __m128i mask = _mm_set1_epi16(0x00FF);

  for (int y = 0; y < height; ++y ) {
    for (int x = 0; x < width; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i swapped = _mm_shufflelo_epi16(src, _MM_SHUFFLE(2, 3, 0, 1));
      swapped = _mm_shufflehi_epi16(swapped, _MM_SHUFFLE(2, 3, 0, 1));
      swapped = _mm_or_si128(_mm_and_si128(mask, src), _mm_andnot_si128(mask, swapped));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), swapped);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void yuy2_swap_ssse3(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch, int width, int height)
{
  const __m128i mask = _mm_set_epi8(13, 14, 15, 12, 9, 10, 11, 8, 5, 6, 7, 4, 1, 2, 3, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 16) {
      __m128i src = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x));
      __m128i dst = _mm_shuffle_epi8(src, mask);
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), dst);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

#ifdef X86_32
void yuy2_swap_isse(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch, int width, int height)
{
  __m64 mask = _mm_set1_pi16(0x00FF);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x+= 8) {
      __m64 src = *reinterpret_cast<const __m64*>(srcp+x);
      __m64 swapped = _mm_shuffle_pi16(src, _MM_SHUFFLE(2, 3, 0, 1));
      swapped = _mm_or_si64(_mm_and_si64(mask, src), _mm_andnot_si64(mask, swapped));
      *reinterpret_cast<__m64*>(dstp + x) = swapped;
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
  _mm_empty();
}
#endif

void yuy2_uvtoy_sse2(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch, int dst_width, int height, int pos)
{
  const __m128i chroma = _mm_set1_epi32(0x80008000);
  const __m128i mask = _mm_set1_epi32(0x000000FF);
  pos *= 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < dst_width; x += 16) {
      __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 2 * x));
      __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 2 * x + 16));
      s0 = _mm_and_si128(mask, _mm_srli_epi32(s0, pos));
      s1 = _mm_and_si128(mask, _mm_srli_epi32(s1, pos));
      s0 = _mm_packs_epi32(s0, s1);
      s0 = _mm_or_si128(s0, chroma);
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), s0);
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

void yuy2_uvtoy8_sse2(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch, int dst_width, int height, int pos)
{
  const __m128i mask = _mm_set1_epi32(0x000000FF);
  pos *= 8;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < dst_width; x += 8) {
      __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4 * x));
      __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + 4 * x + 16));
      s0 = _mm_and_si128(mask, _mm_srli_epi32(s0, pos));
      s1 = _mm_and_si128(mask, _mm_srli_epi32(s1, pos));
      s0 = _mm_packs_epi32(s0, s1);
      s0 = _mm_packus_epi16(s0, s0);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), s0);
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

template <bool has_clipY>
void yuy2_ytouv_sse2(const BYTE* srcp_y, const BYTE* srcp_u, const BYTE* srcp_v, BYTE* dstp, int pitch_y, int pitch_u, int pitch_v, int dst_pitch, int dst_rowsize, int height)
{
  const __m128i mask = _mm_set1_epi16(0x00FF);
  const __m128i zero = _mm_setzero_si128();
  const __m128i fill = _mm_set1_epi16(0x007e);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < dst_rowsize; x += 32) {
      __m128i u = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp_u + x / 2));
      __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp_v + x / 2));
      __m128i uv = _mm_or_si128(_mm_and_si128(u, mask), _mm_slli_epi16(v, 8));
      __m128i uv_lo = _mm_unpacklo_epi8(zero, uv);
      __m128i uv_hi = _mm_unpackhi_epi8(zero, uv);
      if (has_clipY) {
        __m128i y_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp_y + x));
        __m128i y_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp_y + x + 16));
        uv_lo = _mm_or_si128(uv_lo, _mm_and_si128(y_lo, mask));
        uv_hi = _mm_or_si128(uv_hi, _mm_and_si128(y_hi, mask));
      }
      else {
        uv_lo = _mm_or_si128(uv_lo, fill);
        uv_hi = _mm_or_si128(uv_hi, fill);
      }
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x), uv_lo);
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp + x + 16), uv_hi);
    }
    srcp_y += pitch_y;
    srcp_u += pitch_u;
    srcp_v += pitch_v;
    dstp += dst_pitch;
  }
}

template void yuy2_ytouv_sse2<false>(const BYTE* srcp_y, const BYTE* srcp_u, const BYTE* srcp_v, BYTE* dstp, int pitch_y, int pitch_u, int pitch_v, int dst_pitch, int dst_rowsize, int height);
template void yuy2_ytouv_sse2<true>(const BYTE* srcp_y, const BYTE* srcp_u, const BYTE* srcp_v, BYTE* dstp, int pitch_y, int pitch_u, int pitch_v, int dst_pitch, int dst_rowsize, int height);
