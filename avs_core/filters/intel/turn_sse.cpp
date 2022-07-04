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

/*
** Turn. version 0.1
** (c) 2003 - Ernst Peché
**
*/

#include "../turn.h"
#include "turn_sse.h"
#include <tmmintrin.h>
#include <cstdint>


static AVS_FORCEINLINE __m128i movehl(const __m128i& x)
{
    __m128 ps = _mm_castsi128_ps(x);
    return _mm_castps_si128(_mm_movehl_ps(ps, ps));
}


// This pattern seems faster than the others.
static AVS_FORCEINLINE void transpose_8x8x8_sse2(const BYTE* srcp, BYTE* dstp, int src_pitch, int dst_pitch)
{
    __m128i a07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 0)); //a0 a1 a2 a3 a4 a5 a6 a7
    __m128i b07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 1)); //b0 b1 b2 b3 b4 b5 b6 b7
    __m128i c07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 2)); //c0 c1 c2 c3 c4 c5 c6 c7
    __m128i d07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 3)); //d0 d1 d2 d3 d4 d5 d6 d7
    __m128i e07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 4)); //e0 e1 e2 e3 e4 e5 e6 e7
    __m128i f07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 5)); //f0 f1 f2 f3 f4 f5 f6 f7
    __m128i g07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 6)); //g0 g1 g2 g3 g4 g5 g6 g7
    __m128i h07 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 7)); //h0 h1 h2 h3 h4 h5 h6 h7

    __m128i ea07 = _mm_unpacklo_epi8(e07, a07); //e0 a0 e1 a1 e2 a2 e3 a3 e4 a4 e5 a5 e6 a6 e7 a7
    __m128i fb07 = _mm_unpacklo_epi8(f07, b07); //f0 b0 f1 b1 f2 b2 f3 b3 f4 b4 f5 b5 f6 b6 f7 b7
    __m128i gc07 = _mm_unpacklo_epi8(g07, c07); //g0 c0 g1 c1 g2 c2 g3 c3 g4 c4 g5 c5 g6 c6 g7 c7
    __m128i hd07 = _mm_unpacklo_epi8(h07, d07); //h0 d0 h1 d1 h2 d2 h3 d3 h4 d4 h5 d5 h6 d6 h7 d7

    __m128i geca03 = _mm_unpacklo_epi8(gc07, ea07); //g0 e0 c0 a0 g1 e1 c1 a1 g2 e2 c2 a2 g3 e3 c3 a3
    __m128i geca47 = _mm_unpackhi_epi8(gc07, ea07); //g4 e4 c4 a4 g5 e5 c5 a5 g6 e6 c6 a6 g7 e7 c7 a7
    __m128i hfdb03 = _mm_unpacklo_epi8(hd07, fb07); //h0 f0 d0 b0 h1 f1 d1 b1 h2 f2 d2 b2 h3 f3 d3 b3
    __m128i hfdb47 = _mm_unpackhi_epi8(hd07, fb07); //h4 f4 d4 b4 h5 f5 d5 b5 h6 f6 d6 b6 h7 f7 d7 b7

    __m128i hgfedcba01 = _mm_unpacklo_epi8(hfdb03, geca03); //h0 g0 f0 e0 d0 c0 b0 a0 h1 g1 f1 e1 d1 c1 b1 a1
    __m128i hgfedcba23 = _mm_unpackhi_epi8(hfdb03, geca03); //h2 g2 f2 e2 d2 c2 b2 a2 h3 g3 f3 e3 d3 c3 b3 a3
    __m128i hgfedcba45 = _mm_unpacklo_epi8(hfdb47, geca47); //h4 g4 f4 e4 d4 c4 b4 a4 h5 g5 f5 e5 d5 c5 b5 a5
    __m128i hgfedcba67 = _mm_unpackhi_epi8(hfdb47, geca47); //h6 g6 f6 e6 d6 c6 b6 a6 h7 g7 f7 e7 d7 c7 b7 a7

    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 0), hgfedcba01);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 1), movehl(hgfedcba01));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 2), hgfedcba23);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 3), movehl(hgfedcba23));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 4), hgfedcba45);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 5), movehl(hgfedcba45));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 6), hgfedcba67);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + dst_pitch * 7), movehl(hgfedcba67));
}


void turn_right_plane_8_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    const BYTE* s0 = srcp;
    int w = src_rowsize & ~7;
    int h = src_height & ~7;

    for (int y = 0; y < h; y += 8)
    {
        BYTE* d0 = dstp + src_height - 8  - y;
        for (int x = 0; x < w; x += 8)
        {
            transpose_8x8x8_sse2(s0 + x, d0, src_pitch, dst_pitch);
            d0 += dst_pitch * 8;
        }
        s0 += src_pitch * 8;
    }

    if (src_rowsize != w)
    {
        turn_right_plane_8_c(srcp + w, dstp + w * dst_pitch, src_rowsize - w, src_height, src_pitch, dst_pitch);
    }

    if (src_height != h)
    {
        turn_right_plane_8_c(srcp + h * src_pitch, dstp, src_rowsize, src_height - h, src_pitch, dst_pitch);
    }
}


void turn_left_plane_8_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_8_sse2(srcp + (src_height - 1) * src_pitch, dstp + (src_rowsize - 1) * dst_pitch, src_rowsize, src_height, -src_pitch, -dst_pitch);
}



static AVS_FORCEINLINE void transpose_16x4x8_sse2(const BYTE* srcp, BYTE* dstp, const int src_pitch, const int dst_pitch)
{
    __m128i a03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 0)); //a0 a1 a2 a3
    __m128i b03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 1)); //b0 b1 b2 b3
    __m128i c03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 2)); //c0 c1 c2 c3
    __m128i d03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 3)); //d0 d1 d2 d3
    __m128i e03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 4)); //e0 e1 e2 e3
    __m128i f03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 5)); //f0 f1 f2 f3
    __m128i g03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 6)); //g0 g1 g2 g3
    __m128i h03 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp + src_pitch * 7)); //h0 h1 h2 h3

    __m128i ae03 = _mm_unpacklo_epi16(a03, e03); //a0 e0 a1 e1 a2 e2 a3 e3
    __m128i bf03 = _mm_unpacklo_epi16(b03, f03); //b0 f0 b1 f1 b2 f2 b3 f3
    __m128i cg03 = _mm_unpacklo_epi16(c03, g03); //c0 g0 c1 g1 c2 g2 c3 g3
    __m128i dh03 = _mm_unpacklo_epi16(d03, h03); //d0 h0 d1 h1 d2 h2 d3 h3

    __m128i aceg01 = _mm_unpacklo_epi16(ae03, cg03); //a0 c0 e0 g0 a1 c1 e1 g1
    __m128i aceg23 = _mm_unpackhi_epi16(ae03, cg03); //a2 c2 e2 g2 a3 c3 e3 g3
    __m128i bdfh01 = _mm_unpacklo_epi16(bf03, dh03); //b0 d0 f0 h0 b1 d1 f1 h1
    __m128i bdfh23 = _mm_unpackhi_epi16(bf03, dh03); //b2 d2 f2 h2 b3 d3 f3 h3

    __m128i abcdefgh0 = _mm_unpacklo_epi16(aceg01, bdfh01); //a0 b0 c0 d0 e0 f0 g0 h0
    __m128i abcdefgh1 = _mm_unpackhi_epi16(aceg01, bdfh01); //a1 b1 c1 d1 e1 f1 g1 h1
    __m128i abcdefgh2 = _mm_unpacklo_epi16(aceg23, bdfh23); //a2 b2 c2 d2 e2 f2 g2 h2
    __m128i abcdefgh3 = _mm_unpackhi_epi16(aceg23, bdfh23); //a3 b3 c3 d3 e3 f3 g3 h3

    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 0), abcdefgh0);
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 1), abcdefgh1);
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 2), abcdefgh2);
    _mm_store_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 3), abcdefgh3);
}


void turn_right_plane_16_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    const BYTE* s0 = srcp + src_pitch * (src_height - 1);
    int w = src_rowsize & ~7;
    int h = src_height & ~7;

    for (int y = 0; y < h; y += 8)
    {
        BYTE* d0 = dstp + y * 2;
        for (int x = 0; x < w; x += 8)
        {
            transpose_16x4x8_sse2(s0 + x, d0, -src_pitch, dst_pitch);
            d0 += 4 * dst_pitch;
        }
        s0 -= 8 * src_pitch;
    }
    if (src_rowsize != w)
    {
        turn_right_plane_16_c(srcp + w, dstp + dst_pitch * w / 2, src_rowsize - w, src_height, src_pitch, dst_pitch);
    }

    if (src_height != h)
    {
        turn_right_plane_16_c(srcp, dstp + h * 2, src_rowsize, src_height - h, src_pitch, dst_pitch);
    }
}


void turn_left_plane_16_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_16_sse2(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 2 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}



static AVS_FORCEINLINE void transpose_32x4x4_sse2(const BYTE* srcp, BYTE* dstp, const int src_pitch, const int dst_pitch)
{
    __m128i a03 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch * 0)); //a0 a1 a2 a3
    __m128i b03 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch * 1)); //b0 b1 b2 b3
    __m128i c03 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch * 2)); //c0 c1 c2 c3
    __m128i d03 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + src_pitch * 3)); //d0 d1 d2 d3

    __m128i ac01 = _mm_unpacklo_epi32(a03, c03); //a0 c0 a1 c1
    __m128i ac23 = _mm_unpackhi_epi32(a03, c03); //a2 c2 a3 c3
    __m128i bd01 = _mm_unpacklo_epi32(b03, d03); //b0 d0 b1 d1
    __m128i bd23 = _mm_unpackhi_epi32(b03, d03); //b2 d2 b3 d3

    __m128i abcd0 = _mm_unpacklo_epi32(ac01, bd01); //a0 b0 c0 d0
    __m128i abcd1 = _mm_unpackhi_epi32(ac01, bd01); //a1 b1 c1 d1
    __m128i abcd2 = _mm_unpacklo_epi32(ac23, bd23); //a2 b2 c2 d2
    __m128i abcd3 = _mm_unpackhi_epi32(ac23, bd23); //a3 b3 c3 d3

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 0), abcd0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 1), abcd1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 2), abcd2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_pitch * 3), abcd3);
}


void turn_right_plane_32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    const BYTE* s0 = srcp + src_pitch * (src_height - 1);
    int w = src_rowsize & ~15;
    int h = src_height & ~3;

    for (int y = 0; y < h; y += 4)
    {
        BYTE* d0 = dstp + y * 4;
        for (int x = 0; x < w; x += 16)
        {
            transpose_32x4x4_sse2(s0 + x, d0, -src_pitch, dst_pitch);
            d0 += 4 * dst_pitch;
        }
        s0 -= 4 * src_pitch;
    }

    if (src_rowsize != w)
    {
        turn_right_plane_32_c(srcp + w, dstp + w / 4 * dst_pitch, src_rowsize - w, src_height, src_pitch, dst_pitch);
    }

    if (src_height != h)
    {
        turn_right_plane_32_c(srcp, dstp + h * 4, src_rowsize, src_height - h, src_pitch, dst_pitch);
    }
}

void turn_left_plane_32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_32_sse2(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 4 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}


// on RGB, TurnLeft and TurnRight are reversed.

void turn_left_rgb32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_32_sse2(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb32_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_left_plane_32_sse2(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}




static inline void turn_right_plane_64_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    const BYTE* s0 = srcp + src_pitch * (src_height - 1);
    int w = src_rowsize & ~15;
    int h = src_height & ~1;

    for (int y = 0; y < h; y += 2)
    {
        BYTE* d0 = dstp + y * 8;
        for (int x = 0; x < w; x += 16)
        {
            __m128i a01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0 + x));               // a0 a1
            __m128i b01 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0 + x - src_pitch));   // b0 b1
            __m128i ab0 = _mm_unpacklo_epi64(a01, b01); // a0 b0
            __m128i ab1 = _mm_unpackhi_epi64(a01, b01); // a1 b1
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d0), ab0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d0 + dst_pitch), ab1);
            d0 += 2 * dst_pitch;
        }
        s0 -= 2 * src_pitch;
    }

    if (src_rowsize != w)
    {
        turn_right_plane_c<uint64_t>(srcp + w, dstp + w / 8 * dst_pitch, 8, src_height, src_pitch, dst_pitch);
    }

    if (src_height != h)
    {
        turn_right_plane_c<uint64_t>(srcp, dstp + h * 8, src_rowsize, 1, src_pitch, dst_pitch);
    }
}


void turn_left_rgb64_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_64_sse2(srcp, dstp, src_rowsize, src_height, src_pitch, dst_pitch);
}


void turn_right_rgb64_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
    turn_right_plane_64_sse2(srcp + src_pitch * (src_height - 1), dstp + dst_pitch * (src_rowsize / 8 - 1), src_rowsize, src_height, -src_pitch, -dst_pitch);
}





template <typename T>
void turn_180_plane_sse2(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
  const BYTE* s0 = srcp;
  BYTE* d0 = dstp + dst_pitch * (src_height - 1) + src_rowsize - 16;
  const int w = src_rowsize & ~15;

  for (int y = 0; y < src_height; ++y)
  {
    for (int x = 0; x < w; x += 16)
    {
      __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0 + x));
      if constexpr (sizeof(T) == 8) // RGB64
        src = _mm_shuffle_epi32(src, _MM_SHUFFLE(1, 0, 3, 2));
      else if constexpr (sizeof(T) == 4) // RGB32
        src = _mm_shuffle_epi32(src, _MM_SHUFFLE(0, 1, 2, 3));
      else { // uint16_t, uint8_t
        src = _mm_shuffle_epi32(src, _MM_SHUFFLE(0, 1, 2, 3));
        src = _mm_shufflelo_epi16(src, _MM_SHUFFLE(2, 3, 0, 1));
        src = _mm_shufflehi_epi16(src, _MM_SHUFFLE(2, 3, 0, 1));

        if constexpr (sizeof(T) == 1)
          src = _mm_or_si128(_mm_srli_epi16(src, 8), _mm_slli_epi16(src, 8));
      }
      _mm_storeu_si128(reinterpret_cast<__m128i*>(d0 - x), src);
    }
    s0 += src_pitch;
    d0 -= dst_pitch;
  }

  if (src_rowsize != w)
  {
    turn_180_plane_c<T>(srcp + w, dstp, src_rowsize - w, src_height, src_pitch, dst_pitch);
  }
}

// instantiate
template void turn_180_plane_sse2<uint8_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void turn_180_plane_sse2<uint16_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void turn_180_plane_sse2<uint32_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void turn_180_plane_sse2<uint64_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);


template <typename T>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void turn_180_plane_ssse3(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch)
{
  const BYTE* s0 = srcp;
  BYTE* d0 = dstp + dst_pitch * (src_height - 1) + src_rowsize - 16;
  const int w = src_rowsize & ~15;

  for (int y = 0; y < src_height; ++y)
  {
    for (int x = 0; x < w; x += 16)
    {
      __m128i src = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0 + x));
      if constexpr (sizeof(T) == 8) // RGB64
        src = _mm_shuffle_epi32(src, _MM_SHUFFLE(1, 0, 3, 2));
      else if constexpr (sizeof(T) == 4) // RGB32
        src = _mm_shuffle_epi32(src, _MM_SHUFFLE(0, 1, 2, 3));
      else { // uint16_t, uint8_t
        // SSSE3
        __m128i pshufb_mask;
        if constexpr (sizeof(T) == 1)
          pshufb_mask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        else if constexpr (sizeof(T) == 2)
          pshufb_mask = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

        src = _mm_shuffle_epi8(src, pshufb_mask);
      }
      _mm_storeu_si128(reinterpret_cast<__m128i*>(d0 - x), src);
    }
    s0 += src_pitch;
    d0 -= dst_pitch;
  }

  if (src_rowsize != w)
  {
    turn_180_plane_c<T>(srcp + w, dstp, src_rowsize - w, src_height, src_pitch, dst_pitch);
  }
}

// instantiate
template void turn_180_plane_ssse3<uint8_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
template void turn_180_plane_ssse3<uint16_t>(const BYTE* srcp, BYTE* dstp, int src_rowsize, int src_height, int src_pitch, int dst_pitch);
