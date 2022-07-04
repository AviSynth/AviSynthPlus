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

// Overlay (c) 2003, 2004 by Klaus Post

#include "444convert_sse.h"
#include "../core/internal.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include <avs/alignment.h>

// fast in-place conversions from and to 4:4:4

/***** YV12 -> YUV 4:4:4   ******/

template<typename pixel_t>
static void convert_yv12_chroma_to_yv24_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int src_width, int src_height) {
  src_width *= sizeof(pixel_t);
  int mod8_width = src_width / 8 * 8;
  for (int y = 0; y < src_height; ++y) {
    for (int x = 0; x < mod8_width; x+=8) {
      // 0 0 0 0 0 0 0 0 U7 U6 U5 U4 U3 U2 U1 U0 for 8 bits
      // 0 0 0 0 U3 U2 U1 U0 for 16 bits
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x));
      if constexpr(sizeof(pixel_t) == 1)
        src = _mm_unpacklo_epi8(src, src); //U7 U7 U6 U6 U5 U5 U4 U4 U3 U3 U2 U2 U1 U1 U0 U0
      else
        src = _mm_unpacklo_epi16(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x*2), src);
      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x*2 + dst_pitch), src);
    }

    if (mod8_width != src_width) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+src_width - 8));
      if constexpr(sizeof(pixel_t) == 1)
        src = _mm_unpacklo_epi8(src, src); //U7 U7 U6 U6 U5 U5 U4 U4 U3 U3 U2 U2 U1 U1 U0 U0
      else
        src = _mm_unpacklo_epi16(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + (src_width * 2) - 16), src);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + (src_width * 2) - 16 + dst_pitch), src);
    }

    dstp += dst_pitch*2;
    srcp += src_pitch;
  }
}


#ifdef X86_32

static void convert_yv12_chroma_to_yv24_mmx(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int src_width, int src_height) {
  int mod4_width = src_width / 4 * 4;
  for (int y = 0; y < src_height; ++y) {
    for (int x = 0; x < mod4_width; x+=4) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp+x)); //0 0 0 0 U3 U2 U1 U0
      src = _mm_unpacklo_pi8(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      *reinterpret_cast<__m64*>(dstp+x*2) = src;
      *reinterpret_cast<__m64*>(dstp+x*2 + dst_pitch) = src;
    }

    if (mod4_width != src_width) {
      __m64 src = _mm_cvtsi32_si64(*reinterpret_cast<const int*>(srcp-4)); //0 0 0 0 U3 U2 U1 U0
      src = _mm_unpacklo_pi8(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      *reinterpret_cast<__m64*>(dstp + (src_width * 2) - 8) = src;
      *reinterpret_cast<__m64*>(dstp + (src_width * 2) - 8 + dst_pitch) = src;
    }

    dstp += dst_pitch*2;
    srcp += src_pitch;
  }
  _mm_empty();
}

#endif // X86_32


template<typename pixel_t>
static void convert_yv12_chroma_to_yv24_c(BYTE *dstp8, const BYTE *srcp8, int dst_pitch, int src_pitch, int src_width, int src_height) {
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);
  for (int y = 0; y < src_height; ++y) {
    for (int x = 0; x < src_width; ++x) {
      dstp[x*2]             = srcp[x];
      dstp[x*2+1]           = srcp[x];
      dstp[x*2+dst_pitch]   = srcp[x];
      dstp[x*2+dst_pitch+1] = srcp[x];
    }
    dstp += dst_pitch*2;
    srcp += src_pitch;
  }
}

void Convert444FromYV12(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env)
{
  AVS_UNUSED(bits_per_pixel);
  env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), src->GetReadPtr(PLANAR_Y),src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y), src->GetHeight());

  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  int srcUVpitch = src->GetPitch(PLANAR_U);

  BYTE* dstU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = dst->GetWritePtr(PLANAR_V);

  int dstUVpitch = dst->GetPitch(PLANAR_U);

  int width = src->GetRowSize(PLANAR_U) / pixelsize;
  int height = src->GetHeight(PLANAR_U);

  if ((pixelsize == 1 || pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(dstU, 16) && IsPtrAligned(dstV, 16))
  {
    if (pixelsize == 1) {
      convert_yv12_chroma_to_yv24_sse2<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
      convert_yv12_chroma_to_yv24_sse2<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
    }
    else if (pixelsize == 2) {
      convert_yv12_chroma_to_yv24_sse2<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
      convert_yv12_chroma_to_yv24_sse2<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
    }
  }
  else
#ifdef X86_32
    if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_MMX))
    {
      convert_yv12_chroma_to_yv24_mmx(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
      convert_yv12_chroma_to_yv24_mmx(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
    }
    else
#endif
    {
      if (pixelsize == 1) {
        convert_yv12_chroma_to_yv24_c<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv12_chroma_to_yv24_c<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      } else if(pixelsize == 2) {
        convert_yv12_chroma_to_yv24_c<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv12_chroma_to_yv24_c<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      }
      else {
        convert_yv12_chroma_to_yv24_c<float>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv12_chroma_to_yv24_c<float>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      }
    }

  env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A),
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), dst->GetRowSize(PLANAR_A), dst->GetHeight(PLANAR_A));


}

/***** YV16 -> YUV 4:4:4   ******/

template<typename pixel_t>
static void convert_yv16_chroma_to_yv24_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int src_width, int src_height) {
  src_width *= sizeof(pixel_t);
  int mod8_width = src_width / 8 * 8;
  for (int y = 0; y < src_height; ++y) {
    for (int x = 0; x < mod8_width; x+=8) {
      // 0 0 0 0 0 0 0 0 U7 U6 U5 U4 U3 U2 U1 U0 for 8 bits
      // 0 0 0 0 U3 U2 U1 U0 for 16 bits
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+x));
      if constexpr(sizeof(pixel_t) == 1)
        src = _mm_unpacklo_epi8(src, src); //U7 U7 U6 U6 U5 U5 U4 U4 U3 U3 U2 U2 U1 U1 U0 U0
      else
        src = _mm_unpacklo_epi16(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x*2), src);
    }

    if (mod8_width != src_width) {
      __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(srcp+src_width - 8)); //0 0 0 0 0 0 0 0 U8 U7 U6 U5 U4 U3 U2 U1 U0
      if constexpr(sizeof(pixel_t)==1)
        src = _mm_unpacklo_epi8(src, src); //U7 U7 U6 U6 U5 U5 U4 U4 U3 U3 U2 U2 U1 U1 U0 U0
      else
        src = _mm_unpacklo_epi16(src, src); //U3 U3 U2 U2 U1 U1 U0 U0

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + (src_width * 2) - 16), src);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}


template<typename pixel_t>
static void convert_yv16_chroma_to_yv24_c(BYTE *dstp8, const BYTE *srcp8, int dst_pitch, int src_pitch, int src_width, int src_height) {
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);
  for (int y = 0; y < src_height; ++y) {
    for (int x = 0; x < src_width; ++x) {
      dstp[x*2]             = srcp[x];
      dstp[x*2+1]           = srcp[x];
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

void Convert444FromYV16(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env)
{
  AVS_UNUSED(bits_per_pixel);
  env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), src->GetReadPtr(PLANAR_Y),src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y), src->GetHeight());

  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  int srcUVpitch = src->GetPitch(PLANAR_U);

  BYTE* dstU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = dst->GetWritePtr(PLANAR_V);

  int dstUVpitch = dst->GetPitch(PLANAR_U);

  int width = src->GetRowSize(PLANAR_U) / pixelsize;
  int height = src->GetHeight(PLANAR_U);

  if ((pixelsize == 1 || pixelsize==2) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(dstU, 16) && IsPtrAligned(dstV, 16))
  {
    if (pixelsize == 1) {
      convert_yv16_chroma_to_yv24_sse2<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
      convert_yv16_chroma_to_yv24_sse2<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
    }
    else if (pixelsize == 2) {
      convert_yv16_chroma_to_yv24_sse2<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
      convert_yv16_chroma_to_yv24_sse2<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
    }
  }
  else
    {
      if (pixelsize == 1) {
        convert_yv16_chroma_to_yv24_c<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv16_chroma_to_yv24_c<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      } else if(pixelsize == 2) {
        convert_yv16_chroma_to_yv24_c<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv16_chroma_to_yv24_c<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      }
      else {
        convert_yv16_chroma_to_yv24_c<float>(dstU, srcU, dstUVpitch, srcUVpitch, width, height);
        convert_yv16_chroma_to_yv24_c<float>(dstV, srcV, dstUVpitch, srcUVpitch, width, height);
      }
    }

  env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A),
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), dst->GetRowSize(PLANAR_A), dst->GetHeight(PLANAR_A));

}

/***** YUY2 -> YUV 4:4:4   ******/

void Convert444FromYUY2(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  AVS_UNUSED(pixelsize);
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(env);

  const BYTE* srcP = src->GetReadPtr();
  int srcPitch = src->GetPitch();

  BYTE* dstY = dst->GetWritePtr(PLANAR_Y);
  BYTE* dstU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = dst->GetWritePtr(PLANAR_V);

  int dstPitch = dst->GetPitch();

  int w = src->GetRowSize() / 2;
  int h = src->GetHeight();

  for (int y=0; y<h; y++) {
    for (int x=0; x<w; x+=2) {
      int x2 = x<<1;
      dstY[x]   = srcP[x2];
      dstU[x]   = dstU[x+1] = srcP[x2+1];
      dstV[x]   = dstV[x+1] = srcP[x2+3];
      dstY[x+1] = srcP[x2+2];
    }
    srcP+=srcPitch;

    dstY+=dstPitch;
    dstU+=dstPitch;
    dstV+=dstPitch;
  }
}

// YV24->YV12 float types
// Quick simple 2x2 averaging, no mpeg2 or mpeg1 placement involved
static AVS_FORCEINLINE __m128 convert_yv24_chroma_block_to_yv12_float_sse2(const __m128 &src_line0_p0, const __m128 &src_line1_p0, const __m128 &src_line0_p1, const __m128 &src_line1_p1, const __m128 &onefourth) {
  __m128 avg1f = _mm_add_ps(src_line0_p0, src_line1_p0); // vertical sum
  __m128 avg2f = _mm_add_ps(src_line0_p1, src_line1_p1);
  // ABCD -> a3, a2+a3, a1, a1+a0
  avg1f = _mm_add_ps(avg1f, _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(avg1f), 32)));
  avg2f = _mm_add_ps(avg2f, _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(avg2f), 32)));
  return _mm_mul_ps(_mm_shuffle_ps(avg1f, avg2f, _MM_SHUFFLE(2, 0, 2, 0)), onefourth);
}

static void convert_yv24_chroma_to_yv12_float_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  int mod16_width = dst_width / 16 * 16;
  const __m128 onefourth = _mm_set1_ps(0.25f);

  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x += 16) {
      __m128 src_line0_p0 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2));
      __m128 src_line0_p1 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2 + 16));
      __m128 src_line1_p0 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2 + src_pitch));
      __m128 src_line1_p1 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2 + src_pitch + 16));

      __m128 avg = convert_yv24_chroma_block_to_yv12_float_sse2(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, onefourth);

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), avg);
    }

    if (mod16_width != dst_width) {
      __m128 src_line0_p0 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 - 32));
      __m128 src_line0_p1 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 - 16));
      __m128 src_line1_p0 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 + src_pitch - 32));
      __m128 src_line1_p1 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 + src_pitch - 16));

      __m128 avg = convert_yv24_chroma_block_to_yv12_float_sse2(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, onefourth);

      _mm_storeu_ps(reinterpret_cast<float*>(dstp + dst_width - 16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch * 2;
  }
}

// YV24->YV12, uint8_t uint16-t
// Quick simple 2x2 averaging, no mpeg2 or mpeg1 placement involved
// 16bit: SSE4 option

// Reason of "XOR FFFFFFFF...":
// Note! (((a+b+1) >> 1) + ((c+d+1) >> 1) + 1) >> 1 = (a+b+c+d+3) >> 2
/*
      should be (a+b+c+d+2) >> 2
      average_round_down(average_round_up(a, b), average_round_up(c, d))
    = average_round_down(pavgb(a, b), pavgb(c, d))
    = ~pavgb(~pavgb(a, b), ~pavgb(c, d))
*/
template<typename pixel_t>
static AVS_FORCEINLINE __m128i convert_yv24_chroma_block_to_yv12_sse2(const __m128i &src_line0_p0, const __m128i &src_line1_p0, const __m128i &src_line0_p1, const __m128i &src_line1_p1, const __m128i &ffff, const __m128i &mask) {
  __m128i avg1, avg2;
  if constexpr(sizeof(pixel_t) == 1) {
    avg1 = _mm_avg_epu8(src_line0_p0, src_line1_p0);
    avg2 = _mm_avg_epu8(src_line0_p1, src_line1_p1);
  }
  else { // if constexpr(sizeof(pixel_t) == 2)
    avg1 = _mm_avg_epu16(src_line0_p0, src_line1_p0);
    avg2 = _mm_avg_epu16(src_line0_p1, src_line1_p1);
  }

  __m128i avg1x = _mm_xor_si128(avg1, ffff);
  __m128i avg2x = _mm_xor_si128(avg2, ffff);

  if constexpr(sizeof(pixel_t) == 1) {
    __m128i avg1_sh = _mm_srli_epi16(avg1x, 8);
    __m128i avg2_sh = _mm_srli_epi16(avg2x, 8);
    avg1 = _mm_avg_epu8(avg1x, avg1_sh);
    avg2 = _mm_avg_epu8(avg2x, avg2_sh);
  }
  else if constexpr(sizeof(pixel_t) == 2) {
    __m128i avg1_sh = _mm_srli_epi32(avg1x, 16);
    __m128i avg2_sh = _mm_srli_epi32(avg2x, 16);
    avg1 = _mm_avg_epu16(avg1x, avg1_sh);
    avg2 = _mm_avg_epu16(avg2x, avg2_sh);
  }

  avg1 = _mm_and_si128(avg1, mask);
  avg2 = _mm_and_si128(avg2, mask);

  __m128i packed;
  if constexpr(sizeof(pixel_t) == 1)
    packed = _mm_packus_epi16(avg1, avg2);
  else if constexpr(sizeof(pixel_t) == 2) {
    packed = _MM_PACKUS_EPI32(avg1, avg2); // SSE4.1 simul for SSE2
  }
  return _mm_xor_si128(packed, ffff);
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE __m128i convert_yv24_chroma_block_to_yv12_sse41(const __m128i &src_line0_p0, const __m128i &src_line1_p0, const __m128i &src_line0_p1, const __m128i &src_line1_p1, const __m128i &ffff, const __m128i &mask)
{
  __m128i avg1, avg2;
  if constexpr (sizeof(pixel_t) == 1) {
    avg1 = _mm_avg_epu8(src_line0_p0, src_line1_p0);
    avg2 = _mm_avg_epu8(src_line0_p1, src_line1_p1);
  }
  else { // if constexpr(sizeof(pixel_t) == 2)
    avg1 = _mm_avg_epu16(src_line0_p0, src_line1_p0);
    avg2 = _mm_avg_epu16(src_line0_p1, src_line1_p1);
  }

  __m128i avg1x = _mm_xor_si128(avg1, ffff);
  __m128i avg2x = _mm_xor_si128(avg2, ffff);

  if constexpr (sizeof(pixel_t) == 1) {
    __m128i avg1_sh = _mm_srli_epi16(avg1x, 8);
    __m128i avg2_sh = _mm_srli_epi16(avg2x, 8);
    avg1 = _mm_avg_epu8(avg1x, avg1_sh);
    avg2 = _mm_avg_epu8(avg2x, avg2_sh);
  }
  else if constexpr (sizeof(pixel_t) == 2) {
    __m128i avg1_sh = _mm_srli_epi32(avg1x, 16);
    __m128i avg2_sh = _mm_srli_epi32(avg2x, 16);
    avg1 = _mm_avg_epu16(avg1x, avg1_sh);
    avg2 = _mm_avg_epu16(avg2x, avg2_sh);
  }

  avg1 = _mm_and_si128(avg1, mask);
  avg2 = _mm_and_si128(avg2, mask);

  __m128i packed;
  if constexpr (sizeof(pixel_t) == 1)
    packed = _mm_packus_epi16(avg1, avg2);
  else if constexpr (sizeof(pixel_t) == 2) {
    packed = _mm_packus_epi32(avg1, avg2); // SSE4
  }
  return _mm_xor_si128(packed, ffff);
}

template<typename pixel_t>
static void convert_yv24_chroma_to_yv12_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  int mod16_width = dst_width / 16 * 16;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m128i ffff = _mm_set1_epi8(0xFF); // rounding
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  __m128i mask;
  if constexpr(sizeof(pixel_t) == 1)
    mask = _mm_set1_epi16(0x00FF);
  else
    mask = _mm_set1_epi32(0x0000FFFF);


  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x+=16) {
      __m128i src_line0_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2));
      __m128i src_line0_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2+16));
      __m128i src_line1_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2+src_pitch));
      __m128i src_line1_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2+src_pitch+16));

      __m128i avg = convert_yv24_chroma_block_to_yv12_sse2<pixel_t>(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), avg);
    }

    if (mod16_width != dst_width) {
      __m128i src_line0_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2-32));
      __m128i src_line0_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2-16));
      __m128i src_line1_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2+src_pitch-32));
      __m128i src_line1_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2+src_pitch-16));

      __m128i avg = convert_yv24_chroma_block_to_yv12_sse2<pixel_t>(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+dst_width-16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch*2;
  }
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void convert_yv24_chroma_to_yv12_sse41(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height)
{
  int mod16_width = dst_width / 16 * 16;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m128i ffff = _mm_set1_epi8(0xFF); // rounding
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  __m128i mask;
  if constexpr (sizeof(pixel_t) == 1)
    mask = _mm_set1_epi16(0x00FF);
  else
    mask = _mm_set1_epi32(0x0000FFFF);


  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x += 16) {
      __m128i src_line0_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2));
      __m128i src_line0_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 + 16));
      __m128i src_line1_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 + src_pitch));
      __m128i src_line1_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 + src_pitch + 16));

      __m128i avg = convert_yv24_chroma_block_to_yv12_sse41<pixel_t>(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), avg);
    }

    if (mod16_width != dst_width) {
      __m128i src_line0_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 - 32));
      __m128i src_line0_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 - 16));
      __m128i src_line1_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 + src_pitch - 32));
      __m128i src_line1_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 + src_pitch - 16));

      __m128i avg = convert_yv24_chroma_block_to_yv12_sse41<pixel_t>(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_width - 16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch * 2;
  }
}

#ifdef X86_32

static AVS_FORCEINLINE __m64 convert_yv24_chroma_block_to_yv12_isse(const __m64 &src_line0_p0, const __m64 &src_line1_p0, const __m64 &src_line0_p1, const __m64 &src_line1_p1, const __m64 &ffff, const __m64 &mask) {
  __m64 avg1 = _mm_avg_pu8(src_line0_p0, src_line1_p0);
  __m64 avg2 = _mm_avg_pu8(src_line0_p1, src_line1_p1);

  __m64 avg1x = _mm_xor_si64(avg1, ffff);
  __m64 avg2x = _mm_xor_si64(avg2, ffff);

  __m64 avg1_sh = _mm_srli_pi16(avg1x, 8);
  __m64 avg2_sh = _mm_srli_pi16(avg2x, 8);

  avg1 = _mm_avg_pu8(avg1x, avg1_sh);
  avg2 = _mm_avg_pu8(avg2x, avg2_sh);

  avg1 = _mm_and_si64(avg1, mask);
  avg2 = _mm_and_si64(avg2, mask);

  __m64 packed = _mm_packs_pu16(avg1, avg2);
  return _mm_xor_si64(packed, ffff);
}

static void convert_yv24_chroma_to_yv12_isse(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  int mod8_width = dst_width / 8 * 8;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif
  __m64 ffff = _mm_set1_pi8(0xFF);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  __m64 mask = _mm_set1_pi16(0x00FF);

  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod8_width; x+=8) {
      __m64 src_line0_p0 = *reinterpret_cast<const __m64*>(srcp+x*2);
      __m64 src_line0_p1 = *reinterpret_cast<const __m64*>(srcp+x*2+8);
      __m64 src_line1_p0 = *reinterpret_cast<const __m64*>(srcp+x*2+src_pitch);
      __m64 src_line1_p1 = *reinterpret_cast<const __m64*>(srcp+x*2+src_pitch+8);

      __m64 avg = convert_yv24_chroma_block_to_yv12_isse(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      *reinterpret_cast<__m64*>(dstp+x) = avg;
    }

    if (mod8_width != dst_width) {
      __m64 src_line0_p0 = *reinterpret_cast<const __m64*>(srcp+dst_width*2-16);
      __m64 src_line0_p1 = *reinterpret_cast<const __m64*>(srcp+dst_width*2-8);
      __m64 src_line1_p0 = *reinterpret_cast<const __m64*>(srcp+dst_width*2+src_pitch-16);
      __m64 src_line1_p1 = *reinterpret_cast<const __m64*>(srcp+dst_width*2+src_pitch-8);

      __m64 avg = convert_yv24_chroma_block_to_yv12_isse(src_line0_p0, src_line1_p0, src_line0_p1, src_line1_p1, ffff, mask);

      *reinterpret_cast<__m64*>(dstp+dst_width-8) = avg;
    }

    dstp += dst_pitch;
    srcp += src_pitch*2;
  }
  _mm_empty();
}

#endif // X86_32

template<typename pixel_t>
static void convert_yv24_chroma_to_yv12_c(BYTE *dstp8, const BYTE *srcp8, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);
  for (int y = 0; y < dst_height; y++) {
    for (int x = 0; x < dst_width; x++) {
      if constexpr (sizeof(pixel_t) == 4)
        dstp[x] = (srcp[x * 2] + srcp[x * 2 + 1] + srcp[x * 2 + src_pitch] + srcp[x * 2 + src_pitch + 1]) * 0.25f; // /4
      else
        dstp[x] = (srcp[x * 2] + srcp[x * 2 + 1] + srcp[x * 2 + src_pitch] + srcp[x * 2 + src_pitch + 1] + 2) >> 2;
    }
    srcp += src_pitch * 2;
    dstp += dst_pitch;
  }
}

// Quick YV24->YV16 chroma
static AVS_FORCEINLINE __m128 convert_yv24_chroma_block_to_yv16_float_sse2(const __m128 &src_line0_p0, const __m128 &src_line0_p1, const __m128 &half) {
  // A3 A2 A1 A0 B3 B2 B1 B0 -> A3 + A2, A1 + A0, B3 + B2, B1 + B0
  __m128 avg1 = src_line0_p0;
  __m128 avg2 = src_line0_p1;
  // ABCD -> a3, a2+a3, a1, a1+a0
  avg1 = _mm_add_ps(avg1, _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(avg1), 32)));
  // ABCD -> b3, b2+b3, b1, b1+a0
  avg2 = _mm_add_ps(avg2, _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(avg2), 32)));
  return _mm_mul_ps(_mm_shuffle_ps(avg1, avg2, _MM_SHUFFLE(2, 0, 2, 0)), half);
}

// min width: dst_width >= 16
static void convert_yv24_chroma_to_yv16_float_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  int mod16_width = dst_width / 16 * 16;
  const __m128 half = _mm_set1_ps(0.5f); // averaging

  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x += 16) {
      __m128 src_line0_p0 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2));
      __m128 src_line0_p1 = _mm_load_ps(reinterpret_cast<const float*>(srcp + x * 2 + 16));

      __m128 avg = convert_yv24_chroma_block_to_yv16_float_sse2(src_line0_p0, src_line0_p1, half);

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), avg);
    }

    if (mod16_width != dst_width) {
      __m128 src_line0_p0 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 - 32));
      __m128 src_line0_p1 = _mm_loadu_ps(reinterpret_cast<const float*>(srcp + dst_width * 2 - 16));

      __m128 avg = convert_yv24_chroma_block_to_yv16_float_sse2(src_line0_p0, src_line0_p1, half);

      _mm_storeu_ps(reinterpret_cast<float*>(dstp + dst_width - 16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

// 16 bit: SSE4 option
// uint8_t, uint16_t
template<typename pixel_t>
static AVS_FORCEINLINE __m128i convert_yv24_chroma_block_to_yv16_sse2(const __m128i &src_line0_p0, const __m128i &src_line0_p1, const __m128i &mask) {
  __m128i avg1, avg2;

  if constexpr(sizeof(pixel_t) == 1) {
    __m128i avg1_sh = _mm_srli_epi16(src_line0_p0, 8);
    __m128i avg2_sh = _mm_srli_epi16(src_line0_p1, 8);

    avg1 = _mm_avg_epu8(src_line0_p0, avg1_sh);
    avg2 = _mm_avg_epu8(src_line0_p1, avg2_sh);
  }
  else { // if constexpr(sizeof(pixel_t) == 2)
    __m128i avg1_sh = _mm_srli_epi32(src_line0_p0, 16);
    __m128i avg2_sh = _mm_srli_epi32(src_line0_p1, 16);

    avg1 = _mm_avg_epu16(src_line0_p0, avg1_sh);
    avg2 = _mm_avg_epu16(src_line0_p1, avg2_sh);
  }

  avg1 = _mm_and_si128(avg1, mask);
  avg2 = _mm_and_si128(avg2, mask);

  __m128i packed;
  if constexpr(sizeof(pixel_t) == 1)
    packed = _mm_packus_epi16(avg1, avg2);
  else { // if constexpr(sizeof(pixel_t) == 2)
    packed = _MM_PACKUS_EPI32(avg1, avg2); // SSE4.1 simul for SSE2
  }
  return packed;
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static AVS_FORCEINLINE __m128i convert_yv24_chroma_block_to_yv16_sse41(const __m128i &src_line0_p0, const __m128i &src_line0_p1, const __m128i &mask)
{
  __m128i avg1, avg2;

  if constexpr (sizeof(pixel_t) == 1) {
    __m128i avg1_sh = _mm_srli_epi16(src_line0_p0, 8);
    __m128i avg2_sh = _mm_srli_epi16(src_line0_p1, 8);

    avg1 = _mm_avg_epu8(src_line0_p0, avg1_sh);
    avg2 = _mm_avg_epu8(src_line0_p1, avg2_sh);
  }
  else { // if constexpr(sizeof(pixel_t) == 2)
    __m128i avg1_sh = _mm_srli_epi32(src_line0_p0, 16);
    __m128i avg2_sh = _mm_srli_epi32(src_line0_p1, 16);

    avg1 = _mm_avg_epu16(src_line0_p0, avg1_sh);
    avg2 = _mm_avg_epu16(src_line0_p1, avg2_sh);
  }

  avg1 = _mm_and_si128(avg1, mask);
  avg2 = _mm_and_si128(avg2, mask);

  __m128i packed;
  if constexpr (sizeof(pixel_t) == 1)
    packed = _mm_packus_epi16(avg1, avg2);
  else { // if constexpr(sizeof(pixel_t) == 2)
    packed = _mm_packus_epi32(avg1, avg2); // SSE4
  }
  return packed;
}

// uint8_t, uint16_t
template<typename pixel_t>
static void convert_yv24_chroma_to_yv16_sse2(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  int mod16_width = dst_width / 16 * 16;

  __m128i mask;
  if constexpr(sizeof(pixel_t) == 1)
    mask = _mm_set1_epi16(0x00FF);
  else
    mask = _mm_set1_epi32(0x0000FFFF);

  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x+=16) {
      __m128i src_line0_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2));
      __m128i src_line0_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp+x*2+16));

      __m128i avg = convert_yv24_chroma_block_to_yv16_sse2<pixel_t>(src_line0_p0, src_line0_p1, mask);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp+x), avg);
    }

    if (mod16_width != dst_width) {
      __m128i src_line0_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2-32));
      __m128i src_line0_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp+dst_width*2-16));

      __m128i avg = convert_yv24_chroma_block_to_yv16_sse2<pixel_t>(src_line0_p0, src_line0_p1, mask);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp+dst_width-16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<typename pixel_t>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("sse4.1")))
#endif
static void convert_yv24_chroma_to_yv16_sse41(BYTE *dstp, const BYTE *srcp, int dst_pitch, int src_pitch, int dst_width, const int dst_height)
{
  int mod16_width = dst_width / 16 * 16;

  __m128i mask;
  if constexpr (sizeof(pixel_t) == 1)
    mask = _mm_set1_epi16(0x00FF);
  else
    mask = _mm_set1_epi32(0x0000FFFF);

  for (int y = 0; y < dst_height; ++y) {
    for (int x = 0; x < mod16_width; x += 16) {
      __m128i src_line0_p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2));
      __m128i src_line0_p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp + x * 2 + 16));

      __m128i avg = convert_yv24_chroma_block_to_yv16_sse41<pixel_t>(src_line0_p0, src_line0_p1, mask);

      _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), avg);
    }

    if (mod16_width != dst_width) {
      __m128i src_line0_p0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 - 32));
      __m128i src_line0_p1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(srcp + dst_width * 2 - 16));

      __m128i avg = convert_yv24_chroma_block_to_yv16_sse41<pixel_t>(src_line0_p0, src_line0_p1, mask);

      _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + dst_width - 16), avg);
    }

    dstp += dst_pitch;
    srcp += src_pitch;
  }
}

template<typename pixel_t>
static void convert_yv24_chroma_to_yv16_c(BYTE *dstp8, const BYTE *srcp8, int dst_pitch, int src_pitch, int dst_width, const int dst_height) {
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp8);
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  dst_pitch /= sizeof(pixel_t);
  src_pitch /= sizeof(pixel_t);
  for (int y=0; y < dst_height; y++) {
    for (int x=0; x < dst_width; x++) {
      if constexpr (sizeof(pixel_t) == 4)
        dstp[x] = (srcp[x * 2] + srcp[x * 2 + 1]) * 0.5f;
      else
        dstp[x] = (srcp[x * 2] + srcp[x * 2 + 1] + 1) >> 1;
    }
    srcp+=src_pitch;
    dstp+=dst_pitch;
  }
}

void ConvertYToYV12Chroma(BYTE *dst, BYTE *src, int dstpitch, int srcpitch, int pixelsize, int w, int h, IScriptEnvironment* env)
{
  if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src, 16) && IsPtrAligned(dst, 16))
  {
    if (pixelsize == 1)
      convert_yv24_chroma_to_yv12_sse2<uint8_t>(dst, src, dstpitch, srcpitch, w, h);
    else if (pixelsize == 2) {
      if (env->GetCPUFlags() & CPUF_SSE4) { // packus_epi32
        convert_yv24_chroma_to_yv12_sse41<uint16_t>(dst, src, dstpitch, srcpitch, w*pixelsize, h);
      }
      else {
        convert_yv24_chroma_to_yv12_sse2<uint16_t>(dst, src, dstpitch, srcpitch, w*pixelsize, h);
      }
    }
    else
      convert_yv24_chroma_to_yv12_float_sse2(dst, src, dstpitch, srcpitch, w*pixelsize, h);
  }
  else {
    if(pixelsize==1)
      convert_yv24_chroma_to_yv12_c<uint8_t>(dst, src, dstpitch, srcpitch, w, h);
    else if (pixelsize == 2)
      convert_yv24_chroma_to_yv12_c<uint16_t>(dst, src, dstpitch, srcpitch, w, h);
    else // if (pixelsize == 4)
      convert_yv24_chroma_to_yv12_c<float>(dst, src, dstpitch, srcpitch, w, h);
  }
}

void ConvertYToYV16Chroma(BYTE *dst, BYTE *src, int dstpitch, int srcpitch, int pixelsize, int w, int h, IScriptEnvironment* env)
{
  if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src, 16) && IsPtrAligned(dst, 16)
    && w * pixelsize >= 16) // last chunk is also simd, but working on a right-aligned 32 bytes -> 8 bytes. Also simd but unaligned.
  {
    if (pixelsize == 1)
      convert_yv24_chroma_to_yv16_sse2<uint8_t>(dst, src, dstpitch, srcpitch, w, h);
    else if (pixelsize == 2) {
      if (env->GetCPUFlags() & CPUF_SSE4) { // packus_epi32
        convert_yv24_chroma_to_yv16_sse41<uint16_t>(dst, src, dstpitch, srcpitch, w*pixelsize, h);
      }
      else {
        convert_yv24_chroma_to_yv16_sse2<uint16_t>(dst, src, dstpitch, srcpitch, w*pixelsize, h);
      }
    }
    else {
      convert_yv24_chroma_to_yv16_float_sse2(dst, src, dstpitch, srcpitch, w*pixelsize, h);
    }
  }
  else {
    if(pixelsize==1)
      convert_yv24_chroma_to_yv16_c<uint8_t>(dst, src, dstpitch, srcpitch, w, h);
    else if (pixelsize == 2)
      convert_yv24_chroma_to_yv16_c<uint16_t>(dst, src, dstpitch, srcpitch, w, h);
    else // if (pixelsize == 4)
      convert_yv24_chroma_to_yv16_c<float>(dst, src, dstpitch, srcpitch, w, h);
  }
}

void Convert444ToYV16(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env)
{
  AVS_UNUSED(bits_per_pixel);
  env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y),
    src->GetReadPtr(PLANAR_Y), src->GetPitch(), dst->GetRowSize(PLANAR_Y), dst->GetHeight());

  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  int srcUVpitch = src->GetPitch(PLANAR_U);

  BYTE* dstU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = dst->GetWritePtr(PLANAR_V);

  int dstUVpitch = dst->GetPitch(PLANAR_U);

  int w = dst->GetRowSize(PLANAR_U);
  int h = dst->GetHeight(PLANAR_U);

  if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcU, 16) && IsPtrAligned(srcV, 16) && IsPtrAligned(dstU, 16) && IsPtrAligned(dstV, 16))
  {
    if (pixelsize == 1) {
      convert_yv24_chroma_to_yv16_sse2<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
      convert_yv24_chroma_to_yv16_sse2<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
    }
    else if (pixelsize == 2) {
      if (env->GetCPUFlags() & CPUF_SSE4) { // packus_epi32
        convert_yv24_chroma_to_yv16_sse41<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv16_sse41<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else {
        convert_yv24_chroma_to_yv16_sse2<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv16_sse2<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
    }
    else {
      convert_yv24_chroma_to_yv16_float_sse2(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
      convert_yv24_chroma_to_yv16_float_sse2(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
    }
  }
  else {
      if(pixelsize==1) {
        convert_yv24_chroma_to_yv16_c<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv16_c<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else if (pixelsize == 2) {
        convert_yv24_chroma_to_yv16_c<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv16_c<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else { // if (pixelsize == 4)
        convert_yv24_chroma_to_yv16_c<float>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv16_c<float>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
  }

  env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A),
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), dst->GetRowSize(PLANAR_A), dst->GetHeight(PLANAR_A));
}


void Convert444ToYV12(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env)
{
  AVS_UNUSED(bits_per_pixel);
  env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y),
    src->GetReadPtr(PLANAR_Y), src->GetPitch(), dst->GetRowSize(PLANAR_Y), dst->GetHeight());

  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  int srcUVpitch = src->GetPitch(PLANAR_U);

  BYTE* dstU = dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = dst->GetWritePtr(PLANAR_V);

  int dstUVpitch = dst->GetPitch(PLANAR_U);

  int w = dst->GetRowSize(PLANAR_U);
  int h = dst->GetHeight(PLANAR_U);

  if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcU, 16) && IsPtrAligned(srcV, 16) && IsPtrAligned(dstU, 16) && IsPtrAligned(dstV, 16))
  {
    if (pixelsize == 1) {
      convert_yv24_chroma_to_yv12_sse2<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
      convert_yv24_chroma_to_yv12_sse2<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
    }
    else if (pixelsize == 2) {
      if (env->GetCPUFlags() & CPUF_SSE4) {  // packus_epi32
        convert_yv24_chroma_to_yv12_sse41<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv12_sse41<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else {
        convert_yv24_chroma_to_yv12_sse2<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv12_sse2<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
    }
    else {
      convert_yv24_chroma_to_yv12_float_sse2(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
      convert_yv24_chroma_to_yv12_float_sse2(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
    }
  }
  else {
#ifdef X86_32
    if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_INTEGER_SSE))
    {
      convert_yv24_chroma_to_yv12_isse(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
      convert_yv24_chroma_to_yv12_isse(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
    }
    else
#endif
    {
      if(pixelsize==1) {
        convert_yv24_chroma_to_yv12_c<uint8_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv12_c<uint8_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else if (pixelsize == 2) {
        convert_yv24_chroma_to_yv12_c<uint16_t>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv12_c<uint16_t>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
      else { // if (pixelsize == 4)
        convert_yv24_chroma_to_yv12_c<float>(dstU, srcU, dstUVpitch, srcUVpitch, w, h);
        convert_yv24_chroma_to_yv12_c<float>(dstV, srcV, dstUVpitch, srcUVpitch, w, h);
      }
    }
  }

  env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A),
    src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), dst->GetRowSize(PLANAR_A), dst->GetHeight(PLANAR_A));

}

/*****   YUV 4:4:4 -> YUY2   *******/

void Convert444ToYUY2(PVideoFrame &src, PVideoFrame &dst, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(env);

  const BYTE* srcY = src->GetReadPtr(PLANAR_Y);
  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  int srcPitch = src->GetPitch();

  BYTE* dstP = dst->GetWritePtr();

  int dstPitch = dst->GetPitch();

  int w = src->GetRowSize() / pixelsize;
  int h = src->GetHeight();

  for (int y=0; y<h; y++) {
    for (int x=0; x<w; x+=2) {
      int x2 = x<<1;
      dstP[x2]   = srcY[x];
      dstP[x2+1] = (srcU[x] + srcU[x+1] + 1)>>1;
      dstP[x2+2] = srcY[x+1];
      dstP[x2+3] = (srcV[x] + srcV[x+1] + 1)>>1;
    }
    srcY+=srcPitch;
    srcU+=srcPitch;
    srcV+=srcPitch;
    dstP+=dstPitch;
  }
}
