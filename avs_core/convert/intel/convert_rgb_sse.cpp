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


#include "../convert_rgb.h"

#include <tmmintrin.h>
#include <avs/alignment.h>




#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb48_to_rgb64_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  size_t mod16_width = sizeof(uint16_t)*(width & (~size_t(7)));
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m128i mask0 = _mm_set_epi8(0x80, 0x80, 11, 10, 9, 8, 7, 6, 0x80, 0x80, 5, 4, 3, 2, 1, 0);
  __m128i mask1 = _mm_set_epi8(0x80, 0x80, 15, 14, 13, 12, 11, 10, 0x80, 0x80, 9, 8, 7, 6, 5, 4);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  __m128i alpha = _mm_set_epi32(0xFFFF0000,0x00000000,0xFFFF0000,0x00000000);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+= 16) {
      __m128i src0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3));
      // 7...       ...0
      // B G|R B G R B G  #0 #1 (#2)       x*3+0
      // G R B G|R B G R  (#2) #3 #4 (#5)  x*3+16
      // R B G R B G|R B  (#5) #6 #7       x*3+32
      __m128i dst = _mm_or_si128(alpha, _mm_shuffle_epi8(src0, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+4*x), dst);

      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3+16));
      __m128i tmp = _mm_alignr_epi8(src1, src0, 12);
      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(tmp, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+16), dst);

      __m128i src2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3+32));
      tmp = _mm_alignr_epi8(src2, src1, 8);
      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(tmp, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+32), dst);

      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(src2, mask1));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+48), dst);
    }

    for (size_t x = mod16_width/sizeof(uint16_t); x < width; ++x) {
      reinterpret_cast<uint16_t *>(dstp)[x*4+0] = reinterpret_cast<const uint16_t *>(srcp)[x*3+0];
      reinterpret_cast<uint16_t *>(dstp)[x*4+1] = reinterpret_cast<const uint16_t *>(srcp)[x*3+1];
      reinterpret_cast<uint16_t *>(dstp)[x*4+2] = reinterpret_cast<const uint16_t *>(srcp)[x*3+2];
      reinterpret_cast<uint16_t *>(dstp)[x*4+3] = 65535;
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb24_to_rgb32_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  size_t mod16_width = (width + 3) & (~size_t(15)); //when the modulo is more than 13, a problem does not happen
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m128i mask0 = _mm_set_epi8(0x80, 11, 10, 9, 0x80, 8, 7, 6, 0x80, 5, 4, 3, 0x80, 2, 1, 0);
  __m128i mask1 = _mm_set_epi8(0x80, 15, 14, 13, 0x80, 12, 11, 10, 0x80, 9, 8, 7, 0x80, 6, 5, 4);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  __m128i alpha = _mm_set1_epi32(0xFF000000);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+= 16) {
      __m128i src0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3));
      __m128i dst = _mm_or_si128(alpha, _mm_shuffle_epi8(src0, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+4*x), dst);

      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3+16));
      __m128i tmp = _mm_alignr_epi8(src1, src0, 12);
      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(tmp, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+16), dst);

      __m128i src2 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*3+32));
      tmp = _mm_alignr_epi8(src2, src1, 8);
      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(tmp, mask0));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+32), dst);

      dst = _mm_or_si128(alpha, _mm_shuffle_epi8(src2, mask1));
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*4+48), dst);
    }

    for (size_t x = mod16_width; x < width; ++x) {
      dstp[x*4+0] = srcp[x*3+0];
      dstp[x*4+1] = srcp[x*3+1];
      dstp[x*4+2] = srcp[x*3+2];
      dstp[x*4+3] = 255;
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

#ifdef X86_32

void convert_rgb24_to_rgb32_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  size_t mod4_width = width & (~size_t(3));
  __m64 alpha = _mm_set1_pi32(0xFF000000);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod4_width; x+= 4) {
      __m64 src0 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+0)); //0000 0000 r1g0 b0r0
      __m64 src1 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+3)); //0000 0000 r2g1 b1r1
      __m64 src2 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+6)); //0000 0000 r3g2 b2r2
      __m64 src3 = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x*3+9)); //0000 0000 r4g3 b3r3

      __m64 dst01 = _mm_or_si64(src0, _mm_slli_si64(src1, 32)); //r2g1 b1r1 r1g0 b0r0
      __m64 dst23 = _mm_or_si64(src2, _mm_slli_si64(src3, 32)); //r4g3 b3r3 r3g2 b2r2

      dst01 = _mm_or_si64(dst01, alpha);
      dst23 = _mm_or_si64(dst23, alpha);

      *reinterpret_cast<__m64*>(dstp+x*4) = dst01;
      *reinterpret_cast<__m64*>(dstp+x*4+8) = dst23;
    }

    for (size_t x = mod4_width; x < width; ++x) {
      dstp[x*4+0] = srcp[x*3+0];
      dstp[x*4+1] = srcp[x*3+1];
      dstp[x*4+2] = srcp[x*3+2];
      dstp[x*4+3] = 255;
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }

  _mm_empty();
}

#endif // X86_32








#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb64_to_rgb48_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  size_t mod16_width = sizeof(uint16_t) * ((width) & (~size_t(7))); // perhaps width+2 is still o.k
  __m128i mask0 = _mm_set_epi8(13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0, 15, 14, 7, 6); // BBGGRRBBGGRRAAAA
  __m128i mask1 = _mm_set_epi8(15, 14, 7, 6, 13, 12, 11, 10, 9, 8, 5, 4, 3, 2, 1, 0); // AAAABBGGRRBBGGRR

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+= 16) {
      __m128i src0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4));    //a1b1 g1r1 a0b0 g0r0
      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+16)); //a3b3 g3r3 a2b2 g2r2
      src0 = _mm_shuffle_epi8(src0, mask0);         //b1g1 r1b0 g0r0 a1a0
      src1 = _mm_shuffle_epi8(src1, mask1);         //a3a2 b3g3 r3b2 g2r2
      __m128i dst = _mm_alignr_epi8(src1, src0, 4); //g2r2 b1g1 r1b0 g0r0
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3), dst);

      src0 = _mm_slli_si128(src1, 4);       // b3g3 r3b2 g2r2 XXXX
      src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 4 + 32)); // a5b5 g5r5 a4b4 g4r4
      src1 = _mm_shuffle_epi8(src1, mask1); // a5a4 b5g5 r5b4 g4r4
      dst = _mm_alignr_epi8(src1, src0, 8); // r5b4 g4r4 b3g3 r3b2
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3+16), dst);

      src0 = _mm_slli_si128(src1, 4);        // b5g5 r5b4 g4r4 XXXX
      src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+48)); // a7b7 g7r7 a6b6 g6r6
      src1 = _mm_shuffle_epi8(src1, mask1);  // a7a6 b7g7 r7b6 g6r6
      dst = _mm_alignr_epi8(src1, src0, 12); // b7g7 r7b6 g6r6 b5g5
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3+32), dst);
    }

    for (size_t x = mod16_width/sizeof(uint16_t); x < width; ++x) {
      reinterpret_cast<uint16_t *>(dstp)[x*3+0] = reinterpret_cast<const uint16_t *>(srcp)[x*4+0];
      reinterpret_cast<uint16_t *>(dstp)[x*3+1] = reinterpret_cast<const uint16_t *>(srcp)[x*4+1];
      reinterpret_cast<uint16_t *>(dstp)[x*3+2] = reinterpret_cast<const uint16_t *>(srcp)[x*4+2];
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

//todo: think how to port to sse2 without tons of shuffles or (un)packs
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb32_to_rgb24_ssse3(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  size_t mod16_width = (width + 3) & (~size_t(15)); //when the modulo is more than 13, a problem does not happen
  __m128i mask0 = _mm_set_epi8(14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, 15, 11, 7, 3);
  __m128i mask1 = _mm_set_epi8(15, 11, 7, 3, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x+= 16) {
      __m128i src0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4));    //a3b3 g3r3 a2b2 g2r2 a1b1 g1r1 a0b0 g0r0
      __m128i src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+16)); //a7b7 g7r7 a6b6 g6r6 a5b5 g5r5 a4b4 g4r4
      src0 = _mm_shuffle_epi8(src0, mask0);         //b3g3 r3b2 g2r2 b1g1 r1b0 g0r0 a3a2 a1a0
      src1 = _mm_shuffle_epi8(src1, mask1);         //a7a6 a5a4 b7g7 r7b6 g6r6 b5g5 r5b4 g4r4
      __m128i dst = _mm_alignr_epi8(src1, src0, 4); //r5b4 g4r4 b3g3 r3b2 g2r2 b1g1 r1b0 g0r0
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3), dst);

      src0 = _mm_slli_si128(src1, 4);       //b7g7 r7b6 g6r6 b5g5 r5b4 g4r4 XXXX XXXX
      src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 4 + 32)); //aBbB gBrB aAbA gArA a9b9 g9r9 a8b8 g8r8
      src1 = _mm_shuffle_epi8(src1, mask1); //aBaA a9a8 bBgB rBbA gArA b9g9 r9b8 g8r8
      dst = _mm_alignr_epi8(src1, src0, 8); //gArA b9g9 r9b8 g8r8 b7g7 r7b6 g6r6 b5g5
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3+16), dst);

      src0 = _mm_slli_si128(src1, 4);        //bBgB rBbA gArA b9g9 r9b8 g8r8 XXXX XXXX
      src1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*4+48)); //aFbF gFrF aEbE gErE aDbD gDrD aCbC gCrC
      src1 = _mm_shuffle_epi8(src1, mask1);  //aFaE aDaC bFgF rFbE gErE bDgD rDbC gCrC
      dst = _mm_alignr_epi8(src1, src0, 12); //bFgF rFbE gErE bDgD rDbC gCrC bBgB rBbA
      _mm_stream_si128(reinterpret_cast<__m128i*>(dstp+x*3+32), dst);
    }

    for (size_t x = mod16_width; x < width; ++x) {
      dstp[x*3+0] = srcp[x*4+0];
      dstp[x*3+1] = srcp[x*4+1];
      dstp[x*3+2] = srcp[x*4+2];
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

#ifdef X86_32

void convert_rgb32_to_rgb24_mmx(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  size_t mod4_width = width & (~size_t(3));
  __m64 low_pixel_mask = _mm_set_pi32(0, 0x00FFFFFF);
  __m64 high_pixel_mask = _mm_set_pi32(0x00FFFFFF, 0);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod4_width; x+= 4) {
      __m64 src01 = *reinterpret_cast<const __m64*>(srcp+x*4); //a1r1 g1b1 a0r0 g0b0
      __m64 src23 = *reinterpret_cast<const __m64*>(srcp+x*4+8); //a3r3 g3b3 a2r2 g2b2

      __m64 p0 = _mm_and_si64(src01, low_pixel_mask); //0000 0000 00r0 g0b0
      __m64 p1 = _mm_and_si64(src01, high_pixel_mask); //00r1 g1b1 0000 0000
      __m64 p2 = _mm_and_si64(src23, low_pixel_mask); //0000 0000 00r2 g2b2
      __m64 p3 = _mm_and_si64(src23, high_pixel_mask); //00r3 g3b3 0000 0000

      __m64 dst01 = _mm_or_si64(p0, _mm_srli_si64(p1, 8)); //0000 r1g1 b1r0 g0b0
      p3 = _mm_srli_si64(p3, 24); //0000 0000 r3g3 b300

      __m64 dst012 = _mm_or_si64(dst01, _mm_slli_si64(p2, 48));  //g2b2 r1g1 b1r0 g0b0
      __m64 dst23 = _mm_or_si64(p3, _mm_srli_si64(p2, 16)); //0000 0000 r3g3 b3r2

      *reinterpret_cast<__m64*>(dstp+x*3) = dst012;
      *reinterpret_cast<int*>(dstp+x*3+8) = _mm_cvtsi64_si32(dst23);
    }

    for (size_t x = mod4_width; x < width; ++x) {
      dstp[x*3+0] = srcp[x*4+0];
      dstp[x*3+1] = srcp[x*4+1];
      dstp[x*3+2] = srcp[x*4+2];
    }

    srcp += src_pitch;
    dstp += dst_pitch;
  }

  _mm_empty();
}

#endif // X86_32




// minimum width: 48 bytes
template<typename pixel_t, bool targetHasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgb_to_rgbp_ssse3(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel)
{
  // RGB24: 3x16 bytes cycle, 16*(RGB) 8bit pixels
  // RGB48: 3x16 bytes cycle, 8*(RGB) 16bit pixels
  // 0123456789ABCDEF 0123456789ABCDEF 0123456789ABCDEF
  // BGRBGRBGRBGRBGRB GRBGRBGRBGRBGRBG RBGRBGRBGRBGRBGR // 8 bit
  // B G R B G R B G  R B G R B G R B  G R B G R B G R  // 16 bit
  // 1111111111112222 2222222233333333 3333444444444444

  const int pixels_at_a_time = (sizeof(pixel_t) == 1) ? 16 : 8;
  const int wmod = (width / pixels_at_a_time) * pixels_at_a_time; // 8 pixels for 8 bit, 4 pixels for 16 bit
  __m128i mask;
  if constexpr(sizeof(pixel_t) == 1)
    mask = _mm_set_epi8(15, 14, 13, 12, 11, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 0);
  else
    mask = _mm_set_epi8(15, 14, 13, 12, 11, 10, 5, 4, 9, 8, 3, 2, 7, 6, 1, 0);

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m128i max_pixel_value;
  if constexpr(sizeof(pixel_t) == 1)
    max_pixel_value = _mm_set1_epi8(0xFF);
  else
    max_pixel_value = _mm_set1_epi16((1 << bits_per_pixel) - 1); // bits_per_pixel is 16
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  for (int y = height; y > 0; --y) {
    __m128i BGRA_1, BGRA_2, BGRA_3;
    for (int x = 0; x < wmod; x += pixels_at_a_time) {
      BGRA_1 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time));
      BGRA_2 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time + 16));
      BGRA_3 = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time + 32));

      auto pack_lo = _mm_shuffle_epi8(BGRA_1, mask); // 111111111111: BBBBGGGGRRRR and rest: BGRB | BBGGRR and rest: BBGG
      BGRA_1 = _mm_alignr_epi8(BGRA_2, BGRA_1, 12);
      auto pack_hi = _mm_shuffle_epi8(BGRA_1, mask); // 222222222222: BBBBGGGGRRRR | BBGGRR
      BGRA_2 = _mm_alignr_epi8(BGRA_3, BGRA_2, 8);
      auto pack_lo2 = _mm_shuffle_epi8(BGRA_2, mask); // 333333333333: BBBBGGGGRRRR | BBGGRR
      BGRA_3 = _mm_srli_si128(BGRA_3, 4); // to use the same mask
      auto pack_hi2 = _mm_shuffle_epi8(BGRA_3, mask); // 444444444444: BBBBGGGGRRRR | BBGGRR

      __m128i BG1 = _mm_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      __m128i BG2 = _mm_unpacklo_epi32(pack_lo2, pack_hi2);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      __m128i RA1 = _mm_unpackhi_epi32(pack_lo, pack_hi);   // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      __m128i RA2 = _mm_unpackhi_epi32(pack_lo2, pack_hi2);  // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      __m128i B = _mm_unpacklo_epi64(BG1, BG2);
      _mm_store_si128(reinterpret_cast<__m128i *>(dstp[1] + x * sizeof(pixel_t)), B); // B
      __m128i G = _mm_unpackhi_epi64(BG1, BG2);
      _mm_store_si128(reinterpret_cast<__m128i *>(dstp[0] + x * sizeof(pixel_t)), G); // G
      __m128i R = _mm_unpacklo_epi64(RA1, RA2);
      _mm_store_si128(reinterpret_cast<__m128i *>(dstp[2] + x * sizeof(pixel_t)), R); // R
      if (targetHasAlpha)
        _mm_store_si128(reinterpret_cast<__m128i *>(dstp[3] + x * sizeof(pixel_t)), max_pixel_value); // A
    }
    // rest, unaligned but simd
    if (wmod != width) {
      // width = 17: 0..7 8..15, 16
      // last_start = 1 (9..16 8 pixels)  width - pixels_at_a_time
      size_t x = (width - pixels_at_a_time);
      BGRA_1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time));
      BGRA_2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time + 16));
      BGRA_3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + x * 48 / pixels_at_a_time + 32));

      auto pack_lo = _mm_shuffle_epi8(BGRA_1, mask);  // 111111111111: BBBBGGGGRRRR and rest: BGRB | BBGGRR and rest: BBGG
      BGRA_1 = _mm_alignr_epi8(BGRA_2, BGRA_1, 12);
      auto pack_hi = _mm_shuffle_epi8(BGRA_1, mask); // 222222222222: BBBBGGGGRRRR | BBGGRR
      BGRA_2 = _mm_alignr_epi8(BGRA_3, BGRA_2, 8);
      auto pack_lo2 = _mm_shuffle_epi8(BGRA_2, mask); // 333333333333: BBBBGGGGRRRR | BBGGRR
      BGRA_3 = _mm_srli_si128(BGRA_3, 4); // to use the same mask
      auto pack_hi2 = _mm_shuffle_epi8(BGRA_3, mask); // 444444444444: BBBBGGGGRRRR | BBGGRR

      __m128i BG1 = _mm_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      __m128i BG2 = _mm_unpacklo_epi32(pack_lo2, pack_hi2);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      __m128i RA1 = _mm_unpackhi_epi32(pack_lo, pack_hi);   // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      __m128i RA2 = _mm_unpackhi_epi32(pack_lo2, pack_hi2);  // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      __m128i B = _mm_unpacklo_epi64(BG1, BG2);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dstp[1] + x * sizeof(pixel_t)), B); // B
      __m128i G = _mm_unpackhi_epi64(BG1, BG2);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dstp[0] + x * sizeof(pixel_t)), G); // G
      __m128i R = _mm_unpacklo_epi64(RA1, RA2);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(dstp[2] + x * sizeof(pixel_t)), R); // R
      if (targetHasAlpha)
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dstp[3] + x * sizeof(pixel_t)), max_pixel_value); // A
    }
    srcp -= src_pitch; // source packed RGB is upside down
    dstp[0] += dst_pitch[0];
    dstp[1] += dst_pitch[1];
    dstp[2] += dst_pitch[2];
    if (targetHasAlpha)
      dstp[3] += dst_pitch[3];
  }
}

//instantiate
//template<typename pixel_t, bool targetHasAlpha>
template void convert_rgb_to_rgbp_ssse3<uint8_t, false>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_ssse3<uint8_t, true>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_ssse3<uint16_t, false>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);
template void convert_rgb_to_rgbp_ssse3<uint16_t, true>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height, int bits_per_pixel);


// minimum width: 32 bytes (8 RGBA pixels for 8 bits, 4 RGBA pixels for 16 bits)
template<typename pixel_t, bool targetHasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("ssse3")))
#endif
void convert_rgba_to_rgbp_ssse3(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int (&dst_pitch)[4], int width, int height)
{
  const int pixels_at_a_time = (sizeof(pixel_t) == 1) ? 8 : 4;
  const int wmod = (width / pixels_at_a_time) * pixels_at_a_time; // 8 pixels for 8 bit, 4 pixels for 16 bit
  __m128i mask;
  if constexpr(sizeof(pixel_t) == 1)
    mask = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  else
    mask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);

  for (int y = height; y > 0; --y) {
    __m128i BGRA_lo, BGRA_hi;
    for (int x = 0; x < wmod; x+=pixels_at_a_time) {
      BGRA_lo = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x*32/pixels_at_a_time));    // 8bit: *4 pixels 16bit:*2 pixels
      BGRA_hi = _mm_load_si128(reinterpret_cast<const __m128i *>(srcp + x*32/pixels_at_a_time + 16));
      __m128i pack_lo, pack_hi, eightbytes_of_pixels;
      if constexpr(sizeof(pixel_t) == 1) {
        pack_lo = _mm_shuffle_epi8(BGRA_lo, mask); // BBBBGGGGRRRRAAAA
        pack_hi = _mm_shuffle_epi8(BGRA_hi, mask); // BBBBGGGGRRRRAAAA
      }
      else if constexpr(sizeof(pixel_t) == 2) {
        pack_lo = _mm_shuffle_epi8(BGRA_lo, mask); // BBGGRRAA
        pack_hi = _mm_shuffle_epi8(BGRA_hi, mask); // BBGGRRAA
      }
      eightbytes_of_pixels = _mm_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + x * sizeof(pixel_t)), eightbytes_of_pixels); // B
      _mm_storeh_pd(reinterpret_cast<double *>(dstp[0] + x * sizeof(pixel_t)), _mm_castsi128_pd(eightbytes_of_pixels)); // G
      eightbytes_of_pixels = _mm_unpackhi_epi32(pack_lo, pack_hi); // RRRR_lo RRRR_hi AAAA_lo AAAA_hi
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + x * sizeof(pixel_t)), eightbytes_of_pixels); // R
      if(targetHasAlpha)
        _mm_storeh_pd(reinterpret_cast<double *>(dstp[3] + x * sizeof(pixel_t)), _mm_castsi128_pd(eightbytes_of_pixels)); // A
    }
    // rest, unaligned but simd
    if (wmod != width) {
      // width = 17: 0..7 8..15, 16
      // last_start = 1 (9..16 8 pixels)  width - pixels_at_a_time
      size_t last_start = (width - pixels_at_a_time);
      BGRA_lo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + last_start * 32 / pixels_at_a_time));
      BGRA_hi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(srcp + last_start * 32 / pixels_at_a_time + 16));
      __m128i pack_lo, pack_hi, eightbytes_of_pixels;
      if constexpr(sizeof(pixel_t) == 1) {
        pack_lo = _mm_shuffle_epi8(BGRA_lo, mask); // BBBBGGGGRRRRAAAA
        pack_hi = _mm_shuffle_epi8(BGRA_hi, mask); // BBBBGGGGRRRRAAAA
      }
      else if constexpr(sizeof(pixel_t) == 2) {
        pack_lo = _mm_shuffle_epi8(BGRA_lo, mask); // BBGGRRAA
        pack_hi = _mm_shuffle_epi8(BGRA_hi, mask); // BBGGRRAA
      }
      eightbytes_of_pixels = _mm_unpacklo_epi32(pack_lo, pack_hi);  // BBBB_lo BBBB_hi GGGG_lo GGGG_hi
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[1] + last_start * sizeof(pixel_t)), eightbytes_of_pixels); // B
      _mm_storeh_pd(reinterpret_cast<double *>(dstp[0] + last_start * sizeof(pixel_t)), _mm_castsi128_pd(eightbytes_of_pixels)); // G
      eightbytes_of_pixels = _mm_unpackhi_epi32(pack_lo, pack_hi);
      _mm_storel_epi64(reinterpret_cast<__m128i *>(dstp[2] + last_start * sizeof(pixel_t)), eightbytes_of_pixels); // R
      if (targetHasAlpha)
        _mm_storeh_pd(reinterpret_cast<double *>(dstp[3] + last_start * sizeof(pixel_t)), _mm_castsi128_pd(eightbytes_of_pixels)); // A
    }
    srcp -= src_pitch; // source packed RGB is upside down
    dstp[0] += dst_pitch[0];
    dstp[1] += dst_pitch[1];
    dstp[2] += dst_pitch[2];
    if (targetHasAlpha)
      dstp[3] += dst_pitch[3];
  }
}

//instantiate
//template<typename pixel_t, bool targetHasAlpha>
template void convert_rgba_to_rgbp_ssse3<uint8_t, false>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height);
template void convert_rgba_to_rgbp_ssse3<uint8_t, true>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height);
template void convert_rgba_to_rgbp_ssse3<uint16_t, false>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height);
template void convert_rgba_to_rgbp_ssse3<uint16_t, true>(const BYTE* srcp, BYTE* (&dstp)[4], int src_pitch, int(&dst_pitch)[4], int width, int height);

// always to rgb32/64
template<typename pixel_t, bool hasSrcAlpha>
void convert_rgbp_to_rgba_sse2(const BYTE *(&srcp)[4], BYTE * dstp, int (&src_pitch)[4], int dst_pitch, int width, int height) {
  const int pixels_at_a_time = 8 / sizeof(pixel_t); // 8x uint8_t, 4xuint16_t 8 bytes

  const int wmod = (width / pixels_at_a_time) * pixels_at_a_time;

  const __m128i transparent = _mm_set1_epi8((char)0xFF); // 0xFFFF for uint16_t

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wmod; x+=pixels_at_a_time) {
      __m128i R, G, B, A;
      G = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[0] + x*sizeof(pixel_t))); // 8 bytes G7..G0 or G3..G0
      B = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[1] + x*sizeof(pixel_t))); // 8 bytes
      R = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[2] + x*sizeof(pixel_t))); // 8 bytes
      if (hasSrcAlpha)
        A = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[3] + x*sizeof(pixel_t))); // 8 bytes from Alpha plane
      else
        A = transparent;
      if constexpr(sizeof(pixel_t) == 1) {
        __m128i BG = _mm_unpacklo_epi8(B, G); // G7 B7 .. G0 B0
        __m128i RA = _mm_unpacklo_epi8(R, A); // A7 R7 .. A0 R0
        __m128i BGRA_lo = _mm_unpacklo_epi16(BG, RA); // 0..3
        _mm_store_si128(reinterpret_cast<__m128i * >(dstp + x * 4), BGRA_lo);
        __m128i BGRA_hi = _mm_unpackhi_epi16(BG, RA); // 4..7
        _mm_store_si128(reinterpret_cast<__m128i * >(dstp + x * 4 + 16), BGRA_hi);
        // target: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3 lsb->msb
      }
      else {
        __m128i BG = _mm_unpacklo_epi16(B, G); // G7 B7 .. G0 B0
        __m128i RA = _mm_unpacklo_epi16(R, A); // A7 R7 .. A0 R0
        __m128i BGRA_lo = _mm_unpacklo_epi32(BG, RA); // 0..3
        _mm_store_si128(reinterpret_cast<__m128i * >(dstp + x * 8), BGRA_lo);
        __m128i BGRA_hi = _mm_unpackhi_epi32(BG, RA); // 4..7
        _mm_store_si128(reinterpret_cast<__m128i * >(dstp + x * 8 + 16), BGRA_hi);
        // target: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3 lsb->msb
      }
    }
    // rest, overlapped but simd
    if (wmod != width) {
      // width = 17: 0..7 8..15, 16
      // last_start = 1 (9..16 8 pixels)  width - pixels_at_a_time
      // width = 8 -> x = 0
      //         9 -> x = 1
      size_t x = (width - pixels_at_a_time);
      __m128i R, G, B, A;
      G = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[0] + x*sizeof(pixel_t))); // 8 bytes G7..G0 or G3..G0
      B = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[1] + x*sizeof(pixel_t))); // 8 bytes
      R = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[2] + x*sizeof(pixel_t))); // 8 bytes
      if (hasSrcAlpha)
        A = _mm_loadl_epi64(reinterpret_cast<const __m128i * > (srcp[3] + x*sizeof(pixel_t))); // 8 bytes from Alpha plane
      else
        A = transparent;
      if constexpr(sizeof(pixel_t) == 1) {
        __m128i BG = _mm_unpacklo_epi8(B, G); // G7 B7 .. G0 B0
        __m128i RA = _mm_unpacklo_epi8(R, A); // A7 R7 .. A0 R0
        __m128i BGRA_lo = _mm_unpacklo_epi16(BG, RA); // 0..3
        _mm_storeu_si128(reinterpret_cast<__m128i * >(dstp + x * 4), BGRA_lo);
        __m128i BGRA_hi = _mm_unpackhi_epi16(BG, RA); // 4..7
        _mm_storeu_si128(reinterpret_cast<__m128i * >(dstp + x * 4 + 16), BGRA_hi);
        // target: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3 lsb->msb
      }
      else {
        __m128i BG = _mm_unpacklo_epi16(B, G); // G7 B7 .. G0 B0
        __m128i RA = _mm_unpacklo_epi16(R, A); // A7 R7 .. A0 R0
        __m128i BGRA_lo = _mm_unpacklo_epi32(BG, RA); // 0..3
        _mm_storeu_si128(reinterpret_cast<__m128i * >(dstp + x * 8), BGRA_lo);
        __m128i BGRA_hi = _mm_unpackhi_epi32(BG, RA); // 4..7
        _mm_storeu_si128(reinterpret_cast<__m128i * >(dstp + x * 8 + 16), BGRA_hi);
        // target: B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3 lsb->msb
      }
    }

    dstp -= dst_pitch; // source packed RGB is upside down
    srcp[0] += src_pitch[0];
    srcp[1] += src_pitch[1];
    srcp[2] += src_pitch[2];
    if (hasSrcAlpha)
      srcp[3] += src_pitch[3];
  }
}

//instantiate
//template<typename pixel_t, bool hasSrcAlpha>
template void convert_rgbp_to_rgba_sse2<uint8_t, false>(const BYTE* (&srcp)[4], BYTE* dstp, int(&src_pitch)[4], int dst_pitch, int width, int height);
template void convert_rgbp_to_rgba_sse2<uint8_t, true>(const BYTE* (&srcp)[4], BYTE* dstp, int(&src_pitch)[4], int dst_pitch, int width, int height);
template void convert_rgbp_to_rgba_sse2<uint16_t, false>(const BYTE* (&srcp)[4], BYTE* dstp, int(&src_pitch)[4], int dst_pitch, int width, int height);
template void convert_rgbp_to_rgba_sse2<uint16_t, true>(const BYTE* (&srcp)[4], BYTE* dstp, int(&src_pitch)[4], int dst_pitch, int width, int height);


