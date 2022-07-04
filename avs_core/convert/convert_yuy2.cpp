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

#include "convert_yuy2.h"
#ifdef INTEL_INTRINSICS
#include "intel/convert_yuy2_sse.h"
#endif
#include "convert.h"
#include "convert_matrix.h"
#include "convert_helper.h"
#include "convert_planar.h"
#include <avs/alignment.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif


/**********************************
 *******   Convert to YUY2   ******
 *********************************/

ConvertToYUY2::ConvertToYUY2(PClip _child, bool _dupl, bool _interlaced, const char *matrix_name, IScriptEnvironment* env)
  : GenericVideoFilter(_child), interlaced(_interlaced),src_cs(vi.pixel_type)
{
  AVS_UNUSED(_dupl);
  if (vi.height&3 && vi.IsYV12() && interlaced)
    env->ThrowError("ConvertToYUY2: Cannot convert from interlaced YV12 if height is not multiple of 4. Use Crop!");

  if (vi.height&1 && vi.IsYV12() )
    env->ThrowError("ConvertToYUY2: Cannot convert from YV12 if height is not even. Use Crop!");

  if (vi.width & 1)
    env->ThrowError("ConvertToYUY2: Image width must be even. Use Crop!");

  if (matrix_name && !vi.IsRGB())
    env->ThrowError("ConvertToYUY2: invalid \"matrix\" parameter (RGB data only)");

  auto frame0 = _child->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);
  matrix_parse_merge_with_props(vi, matrix_name, props, theMatrix, theColorRange, env);

  const int shift = 15;
  const int bits_per_pixel = 8;
  if (!do_BuildMatrix_Rgb2Yuv(theMatrix, theColorRange, shift, bits_per_pixel, /*ref*/matrix))
    env->ThrowError("ConvertToYUY2: invalid \"matrix\" parameter");

  vi.pixel_type = VideoInfo::CS_YUY2;
}

// 1-2-1 kernel version: convert_rgb_to_yuy2
// 0-1-0 kernel version: convert_rgb_back_to_yuy2_c

// 1-2-1 Kernel version
template<bool TV_range>
static void convert_rgb_to_yuy2_c(
  const BYTE* rgb,
                                  BYTE* yuv, const int yuv_offset,
                                  const int rgb_offset, const int rgb_inc,
  int width, int height, const ConversionMatrix& matrix)
{
  /*
    int Y = matrix.offset_y + (((int)matrix.y_b * b + (int)matrix.y_g * g + (int)matrix.y_r * r + 16384) >> 15);
    int U = 128 + (((int)matrix.u_b * b + (int)matrix.u_g * g + (int)matrix.u_r * r + 16384) >> 15);
    int V = 128 + (((int)matrix.v_b * b + (int)matrix.v_g * g + (int)matrix.v_r * r + 16384) >> 15);

    For U and V this is optimized by using ku and kv (though not that precise)
  */
  constexpr int PRECBITS = 15;
  constexpr int PRECRANGE = 1 << PRECBITS; // 32768
  constexpr int bias = TV_range ? 0x84000 : 0x4000; //  16.5 * 32768 : 0.5 * 32768

  for (int y = height; y > 0; --y)
  {
    // Use left most pixel for edge condition
    int y0 = (matrix.y_b * rgb[0] + matrix.y_g * rgb[1] + matrix.y_r * rgb[2] + bias) >> PRECBITS;
    const BYTE* rgb_prev   = rgb;
    for (int x = 0; x < width; x += 2)
    {
      const BYTE* const rgb_next = rgb + rgb_inc;
      // y1 and y2 can't overflow
      const int y1 = (matrix.y_b * rgb[0] + matrix.y_g * rgb[1] + matrix.y_r * rgb[2] + bias) >> PRECBITS;
      yuv[0]               = y1;
      const int y2 = (matrix.y_b * rgb_next[0] + matrix.y_g * rgb_next[1] + matrix.y_r * rgb_next[2] + bias) >> PRECBITS;
      yuv[2]               = y2;
      if constexpr (!TV_range) {
        const int scaled_y = y0 + y1 * 2 + y2; // 1-2-1 kernel
        const int b_y = (rgb_prev[0] + rgb[0] * 2 + rgb_next[0]) - scaled_y;
        yuv[1] = PixelClip((b_y * matrix.ku + (128 << (PRECBITS + 2)) + (1 << (PRECBITS + 1))) >> (PRECBITS + 2));  // u
        const int r_y = (rgb_prev[2] + rgb[2] * 2 + rgb_next[2]) - scaled_y;
        yuv[3] = PixelClip((r_y * matrix.kv + (128 << (PRECBITS + 2)) + (1 << (PRECBITS + 1))) >> (PRECBITS + 2));  // v
      }
      else {
        const int scaled_y = (y0 + y1 * 2 + y2 - (16 * 4)) * int(255.0 / 219.0 * PRECRANGE + 0.5);
        const int b_y = ((rgb_prev[0] + rgb[0] * 2 + rgb_next[0]) << PRECBITS) - scaled_y;
        yuv[1] = PixelClip(((b_y >> (PRECBITS + 2 - 6)) * matrix.ku + (128 << (PRECBITS + 6)) + (1 << (PRECBITS + 5))) >> (PRECBITS + 6));  // u
        const int r_y = ((rgb_prev[2] + rgb[2] * 2 + rgb_next[2]) << PRECBITS) - scaled_y;
        yuv[3] = PixelClip(((r_y >> (PRECBITS + 2 - 6)) * matrix.kv + (128 << (PRECBITS + 6)) + (1 << (PRECBITS + 5))) >> (PRECBITS + 6));  // v
      }
      y0       = y2;

      rgb_prev = rgb_next;
      rgb      = rgb_next + rgb_inc;
      yuv     += 4;
    }
    rgb += rgb_offset;
    yuv += yuv_offset;
  }
}

// matrix multiplication for u and v like in generic planar conversions
// kept for reference
#if 0
static void convert_rgb_to_yuy2_generic_c(const bool pcrange, const BYTE* rgb,
  BYTE* yuv, const int yuv_offset,
  const int rgb_offset, const int rgb_inc,
  int width, int height, ConversionMatrix& matrix) {
/*
    int Y = matrix.offset_y + (((int)matrix.y_b * b + (int)matrix.y_g * g + (int)matrix.y_r * r + 16384) >> 15);
    int U = 128 + (((int)matrix.u_b * b + (int)matrix.u_g * g + (int)matrix.u_r * r + 16384) >> 15);
    int V = 128 + (((int)matrix.v_b * b + (int)matrix.v_g * g + (int)matrix.v_r * r + 16384) >> 15);
*/
  constexpr int PRECBITS = 15;
  constexpr int ROUNDER = 1 << (PRECBITS - 1); // for 1<<(15-1)
  constexpr int ROUNDER4X = ROUNDER << 2;

  for (int y = height; y > 0; --y)
  {
    // Use left most pixel for edge condition
    int y0 = matrix.offset_y + (((int)matrix.y_b * rgb[0] + (int)matrix.y_g * rgb[1] + (int)matrix.y_r * rgb[2] + ROUNDER) >> PRECBITS);
    const BYTE* rgb_prev = rgb;
    for (int x = 0; x < width; x += 2)
    {
      const BYTE* const rgb_next = rgb + rgb_inc;
      // y1 and y2 can't overflow
      const int y1 = matrix.offset_y + (((int)matrix.y_b * rgb[0] + (int)matrix.y_g * rgb[1] + (int)matrix.y_r * rgb[2] + ROUNDER) >> PRECBITS);
      yuv[0] = y1;
      const int y2 = matrix.offset_y + (((int)matrix.y_b * rgb_next[0] + (int)matrix.y_g * rgb_next[1] + (int)matrix.y_r * rgb_next[2] + ROUNDER) >> PRECBITS);
      yuv[2] = y2;

      int b = (rgb_prev[0] + rgb[0] * 2 + rgb_next[0]);
      int g = (rgb_prev[1] + rgb[1] * 2 + rgb_next[1]);
      int r = (rgb_prev[2] + rgb[2] * 2 + rgb_next[2]);

      int u = 128 + (((int)matrix.u_b * b + (int)matrix.u_g * g + (int)matrix.u_r * r + ROUNDER4X) >> (PRECBITS + 2));
      int v = 128 + (((int)matrix.v_b * b + (int)matrix.v_g * g + (int)matrix.v_r * r + ROUNDER4X) >> (PRECBITS + 2));

      yuv[1] = PixelClip(u);
      yuv[3] = PixelClip(v);

      y0 = y2;

      rgb_prev = rgb_next;
      rgb = rgb_next + rgb_inc;
      yuv += 4;
    }
    rgb += rgb_offset;
    yuv += yuv_offset;
  }
}
#endif

/* YV12 -> YUY2 conversion */


static inline void copy_yv12_line_to_yuy2_c(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, BYTE* dstp, int width) {
  for (int x = 0; x < width / 2; ++x) {
    dstp[x*4] = srcY[x*2];
    dstp[x*4+2] = srcY[x*2+1];
    dstp[x*4+1] = srcU[x];
    dstp[x*4+3] = srcV[x];
  }
}

void convert_yv12_to_yuy2_progressive_c(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height) {
  //first two lines
  copy_yv12_line_to_yuy2_c(srcY, srcU, srcV, dstp, src_width);
  copy_yv12_line_to_yuy2_c(srcY+src_pitch_y, srcU, srcV, dstp+dst_pitch, src_width);

  //last two lines. Easier to do them here
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-2),
    srcU + src_pitch_uv * ((height/2)-1),
    srcV + src_pitch_uv * ((height/2)-1),
    dstp + dst_pitch * (height-2),
    src_width
    );
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-1),
    srcU + src_pitch_uv * ((height/2)-1),
    srcV + src_pitch_uv * ((height/2)-1),
    dstp + dst_pitch * (height-1),
    src_width
    );

  srcY += src_pitch_y*2;
  srcU += src_pitch_uv;
  srcV += src_pitch_uv;
  dstp += dst_pitch*2;

  for (int y = 2; y < height-2; y+=2) {
    for (int x = 0; x < src_width / 2; ++x) {
      dstp[x*4] = srcY[x*2];
      dstp[x*4+2] = srcY[x*2+1];

      //avg(avg(a, b)-1, b)
      dstp[x*4+1] = ((((srcU[x-src_pitch_uv] + srcU[x] + 1) / 2) + srcU[x]) / 2);
      dstp[x*4+3] = ((((srcV[x-src_pitch_uv] + srcV[x] + 1) / 2) + srcV[x]) / 2);

      dstp[x*4 + dst_pitch] = srcY[x*2 + src_pitch_y];
      dstp[x*4+2 + dst_pitch] = srcY[x*2+1 + src_pitch_y];

      dstp[x*4+1 + dst_pitch] = ((((srcU[x] + srcU[x+src_pitch_uv] + 1) / 2) + srcU[x]) / 2);
      dstp[x*4+3 + dst_pitch] = ((((srcV[x] + srcV[x+src_pitch_uv] + 1) / 2) + srcV[x]) / 2);
    }
    srcY += src_pitch_y*2;
    dstp += dst_pitch*2;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
  }
}

void convert_yv12_to_yuy2_interlaced_c(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, int src_width, int src_pitch_y, int src_pitch_uv, BYTE *dstp, int dst_pitch, int height) {
  //first four lines
  copy_yv12_line_to_yuy2_c(srcY, srcU, srcV, dstp, src_width);
  copy_yv12_line_to_yuy2_c(srcY + src_pitch_y*2, srcU, srcV, dstp + dst_pitch*2, src_width);
  copy_yv12_line_to_yuy2_c(srcY + src_pitch_y, srcU + src_pitch_uv, srcV + src_pitch_uv, dstp + dst_pitch, src_width);
  copy_yv12_line_to_yuy2_c(srcY + src_pitch_y*3, srcU + src_pitch_uv, srcV + src_pitch_uv, dstp + dst_pitch*3, src_width);

  //last four lines. Easier to do them here
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-4),
    srcU + src_pitch_uv * ((height/2)-2),
    srcV + src_pitch_uv * ((height/2)-2),
    dstp + dst_pitch * (height-4),
    src_width
    );
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-2),
    srcU + src_pitch_uv * ((height/2)-2),
    srcV + src_pitch_uv * ((height/2)-2),
    dstp + dst_pitch * (height-2),
    src_width
    );
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-3),
    srcU + src_pitch_uv * ((height/2)-1),
    srcV + src_pitch_uv * ((height/2)-1),
    dstp + dst_pitch * (height-3),
    src_width
    );
  copy_yv12_line_to_yuy2_c(
    srcY + src_pitch_y * (height-1),
    srcU + src_pitch_uv * ((height/2)-1),
    srcV + src_pitch_uv * ((height/2)-1),
    dstp + dst_pitch * (height-1),
    src_width
    );

  srcY += src_pitch_y * 4;
  srcU += src_pitch_uv * 2;
  srcV += src_pitch_uv * 2;
  dstp += dst_pitch * 4;

  for (int y = 4; y < height-4; y+= 2) {
    for (int x = 0; x < src_width / 2; ++x) {
      dstp[x*4] = srcY[x*2];
      dstp[x*4+2] = srcY[x*2+1];

      dstp[x*4+1] = ((((srcU[x-src_pitch_uv*2] + srcU[x] + 1) / 2) + srcU[x]) / 2);
      dstp[x*4+3] = ((((srcV[x-src_pitch_uv*2] + srcV[x] + 1) / 2) + srcV[x]) / 2);

      dstp[x*4 + dst_pitch*2] = srcY[x*2 + src_pitch_y*2];
      dstp[x*4+2 + dst_pitch*2] = srcY[x*2+1 + src_pitch_y*2];

      dstp[x*4+1 + dst_pitch*2] = ((((srcU[x] + srcU[x+src_pitch_uv*2] + 1) / 2) + srcU[x]) / 2);
      dstp[x*4+3 + dst_pitch*2] = ((((srcV[x] + srcV[x+src_pitch_uv*2] + 1) / 2) + srcV[x]) / 2);
    }

    if (y % 4 == 0) {
      //top field processed, jumb to the bottom
      srcY += src_pitch_y;
      dstp += dst_pitch;
      srcU += src_pitch_uv;
      srcV += src_pitch_uv;
    } else {
      //bottom field processed, jump to the next top
      srcY += src_pitch_y*3;
      srcU += src_pitch_uv;
      srcV += src_pitch_uv;
      dstp += dst_pitch*3;
    }
  }
}


/* YUY2 -> YV12 conversion */


void convert_yuy2_to_yv12_progressive_c(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height) {
  //src_width is twice the luma width of yv12 frame
  const BYTE* srcp = src;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < src_width / 2 ; ++x) {
      dstY[x] = srcp[x*2];
    }
    dstY += dst_pitchY;
    srcp += src_pitch;
  }


  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < src_width / 4; ++x) {
      dstU[x] = (src[x*4+1] + src[x*4+1+src_pitch] + 1) / 2;
      dstV[x] = (src[x*4+3] + src[x*4+3+src_pitch] + 1) / 2;
    }
    dstU += dst_pitchUV;
    dstV += dst_pitchUV;
    src += src_pitch * 2;
  }
}

void convert_yuy2_to_yv12_interlaced_c(const BYTE* src, int src_width, int src_pitch, BYTE* dstY, BYTE* dstU, BYTE* dstV, int dst_pitchY, int dst_pitchUV, int height) {
  const BYTE* srcp = src;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < src_width / 2 ; ++x) {
      dstY[x] = srcp[x*2];
    }
    dstY += dst_pitchY;
    srcp += src_pitch;
  }

  for (int y = 0; y < height / 2; y+=2) {
    for (int x = 0; x < src_width / 4; ++x) {
      dstU[x] = ((src[x*4+1] + src[x*4+1+src_pitch*2] + 1) / 2 + src[x*4+1]) / 2;
      dstV[x] = ((src[x*4+3] + src[x*4+3+src_pitch*2] + 1) / 2 + src[x*4+3]) / 2;
    }
    dstU += dst_pitchUV;
    dstV += dst_pitchUV;
    src += src_pitch;

    for (int x = 0; x < src_width / 4; ++x) {
      dstU[x] = ((src[x*4+1] + src[x*4+1+src_pitch*2] + 1) / 2 + src[x*4+1+src_pitch*2]) / 2;
      dstV[x] = ((src[x*4+3] + src[x*4+3+src_pitch*2] + 1) / 2 + src[x*4+3+src_pitch*2]) / 2;
    }
    dstU += dst_pitchUV;
    dstV += dst_pitchUV;
    src += src_pitch*3;
  }
} 

PVideoFrame __stdcall ConvertToYUY2::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  if (((src_cs&VideoInfo::CS_YV12)==VideoInfo::CS_YV12)||((src_cs&VideoInfo::CS_I420)==VideoInfo::CS_I420)) {
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);
    BYTE* dstp = dst->GetWritePtr();
    const BYTE* srcp_y = src->GetReadPtr(PLANAR_Y);
    const BYTE* srcp_u = src->GetReadPtr(PLANAR_U);
    const BYTE* srcp_v = src->GetReadPtr(PLANAR_V);
    int src_pitch_y = src->GetPitch(PLANAR_Y);
    int src_pitch_uv = src->GetPitch(PLANAR_U);
    int dst_pitch = dst->GetPitch();
    int src_heigh = dst->GetHeight();

    if (interlaced) {
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        convert_yv12_to_yuy2_interlaced_sse2(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
      }
      else
#ifdef X86_32
      if (env->GetCPUFlags() & CPUF_INTEGER_SSE)
      {
        convert_yv12_to_yuy2_interlaced_isse(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
      }
      else
#endif
#endif
      {
        convert_yv12_to_yuy2_interlaced_c(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
      }
    } else {
#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_SSE2)
      {
        convert_yv12_to_yuy2_progressive_sse2(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
      }
      else
#ifdef X86_32
        if (env->GetCPUFlags() & CPUF_INTEGER_SSE)
        {
          convert_yv12_to_yuy2_progressive_isse(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
        }
        else
#endif
#endif
        {
          convert_yv12_to_yuy2_progressive_c(srcp_y, srcp_u, srcp_v, src->GetRowSize(PLANAR_Y), src_pitch_y, src_pitch_uv, dstp, dst_pitch ,src_heigh);
        }
    }
    return dst;
  }

  // convert from rgb

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  BYTE* yuv = dst->GetWritePtr();

  auto props = env->getFramePropsRW(dst);
  update_Matrix_and_ColorRange(props, theMatrix, theColorRange, env);

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSE2)
  {
    if ((src_cs & VideoInfo::CS_BGR32) == VideoInfo::CS_BGR32) {
      convert_rgb_to_yuy2_sse2<4>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    } else {
      convert_rgb_to_yuy2_sse2<3>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    }
    return dst;
  }

#ifdef X86_32
  if (env->GetCPUFlags() & CPUF_MMX)
  {
    if ((src_cs & VideoInfo::CS_BGR32) == VideoInfo::CS_BGR32) {
      convert_rgb_to_yuy2_mmx<4>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    } else {
      convert_rgb_to_yuy2_mmx<3>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    }
    return dst;
  }
#endif
#endif
  const BYTE* rgb = src->GetReadPtr() + (vi.height-1) * src->GetPitch();

  const int yuv_offset = dst->GetPitch() - dst->GetRowSize();
  const int rgb_offset = -src->GetPitch() - src->GetRowSize();
  const int rgb_inc = ((src_cs&VideoInfo::CS_BGR32)==VideoInfo::CS_BGR32) ? 4 : 3;

  // this reference C matches best to the actual sse2 implementations
  if (0 != matrix.offset_y)
    convert_rgb_to_yuy2_c<true>(rgb, yuv, yuv_offset, rgb_offset, rgb_inc, vi.width, vi.height, matrix); // rec
  else
    convert_rgb_to_yuy2_c<false>(rgb, yuv, yuv_offset, rgb_offset, rgb_inc, vi.width, vi.height, matrix); // PC

  // or using similar matrix multiplication inside like in other RGB to planar YUV, but it differs from sse implementation
  // convert_rgb_to_yuy2_new_c(false, rgb, yuv, yuv_offset, rgb_offset, rgb_inc, vi.width, vi.height, matrix);

  return dst;
}


AVSValue __cdecl ConvertToYUY2::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsYUY2())
    return clip;

  const bool haveOpts = args[3].Defined() || args[4].Defined();

  if (clip->GetVideoInfo().BitsPerComponent() != 8) {
    env->ThrowError("ConvertToYUY2: only 8 bit sources are supported");
  }

  if (clip->GetVideoInfo().IsPlanar()) {
    if (haveOpts || !clip->GetVideoInfo().IsYV12()) {
      // We have no direct conversions. Go to YV16.
      AVSValue new_args[5] = { clip, args[1], args[2], args[3], args[4] };
      clip = ConvertToPlanarGeneric::CreateYUV422(AVSValue(new_args, 5), (void *)0,  env).AsClip(); // (void *)0: restricted to 8 bits
    }
  }

  if (clip->GetVideoInfo().IsYV16())
    return new ConvertYV16ToYUY2(clip,  env);

  if (haveOpts)
    env->ThrowError("ConvertToYUY2: ChromaPlacement and ChromaResample options are not supported.");

  const bool i=args[1].AsBool(false);
  return new ConvertToYUY2(clip, false, i, args[2].AsString(0), env);
}



/****************************************************
 ******* Convert back to YUY2                  ******
 ******* this only uses Chroma from left pixel ******
 ******* to be used, when signal already has   ******
 ******* been YUY2 to avoid deterioration      ******
 ****************************************************/

ConvertBackToYUY2::ConvertBackToYUY2(PClip _child, const char *matrix, IScriptEnvironment* env)
  : ConvertToYUY2(_child, true, false, matrix, env)
{
  if (!_child->GetVideoInfo().IsRGB() && !_child->GetVideoInfo().IsYV24())
    env->ThrowError("ConvertBackToYUY2: Use ConvertToYUY2 to convert non-RGB material to YUY2.");
}

static void convert_yv24_back_to_yuy2_c(const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, BYTE* dstp, int pitchY, int pitchUV, int dpitch, int height, int width) {
  for (int y=0; y < height; ++y) {
    for (int x=0; x < width; x+=2) {
      dstp[x*2+0] = srcY[x];
      dstp[x*2+1] = srcU[x];
      dstp[x*2+2] = srcY[x+1];
      dstp[x*2+3] = srcV[x];
    }
    srcY += pitchY;
    srcU += pitchUV;
    srcV += pitchUV;
    dstp += dpitch;
  }
}


template<bool TV_range>
static void convert_rgb_back_to_yuy2_c(BYTE* yuv, const BYTE* rgb, int rgb_offset, int yuv_offset, int height, int width, int rgb_inc, const ConversionMatrix& matrix) {
  /* Existing 0-1-0 Kernel version */
  /* As noted u and v is calculated only from left pixel of adjacent rgb pairs */

  constexpr int PRECBITS = 15;
  constexpr int PRECRANGE = 1 << PRECBITS; // 32768
  constexpr int bias = TV_range ? 0x84000 : 0x4000; //  16.5 * 32768 : 0.5 * 32768

  for (int y = height; y > 0; --y)
  {
    for (int x = 0; x < width; x += 2)
    {
      const BYTE* const rgb_next = rgb + rgb_inc;
      // y1 and y2 can't overflow
      yuv[0] = (matrix.y_b * rgb[0] + matrix.y_g * rgb[1] + matrix.y_r * rgb[2] + bias) >> PRECBITS;
      yuv[2] = (matrix.y_b * rgb_next[0] + matrix.y_g * rgb_next[1] + matrix.y_r * rgb_next[2] + bias) >> PRECBITS;
      if constexpr (!TV_range) {
        int scaled_y = yuv[0];
        int b_y = rgb[0] - scaled_y;
        yuv[1] = Scaled15bitPixelClip(b_y * matrix.ku + (128 << PRECBITS));  // u
        int r_y = rgb[2] - scaled_y;
        yuv[3] = Scaled15bitPixelClip(r_y * matrix.kv + (128 << PRECBITS));  // v
      }
      else {
        int scaled_y = (yuv[0] - 16) * int(255.0 / 219.0 * PRECRANGE + 0.5);
        int b_y = ((rgb[0]) << PRECBITS) - scaled_y;
        yuv[1] = PixelClip(((b_y >> (PRECBITS - 6)) * matrix.ku + (128 << (PRECBITS + 6)) + (1 << (PRECBITS + 5))) >> (PRECBITS + 6));  // u
        int r_y = ((rgb[2]) << PRECBITS) - scaled_y;
        yuv[3] = PixelClip(((r_y >> (PRECBITS - 6)) * matrix.kv + (128 << (PRECBITS + 6)) + (1 << (PRECBITS + 5))) >> (PRECBITS + 6));  // v
      }
      rgb = rgb_next + rgb_inc;
      yuv += 4;
    }
    rgb += rgb_offset;
    yuv += yuv_offset;
  }
}


PVideoFrame __stdcall ConvertBackToYUY2::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  if ((src_cs&VideoInfo::CS_YV24)==VideoInfo::CS_YV24)
  {
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);
    BYTE* dstp = dst->GetWritePtr();
    const int dpitch  = dst->GetPitch();

    const BYTE* srcY = src->GetReadPtr(PLANAR_Y);
    const BYTE* srcU = src->GetReadPtr(PLANAR_U);
    const BYTE* srcV = src->GetReadPtr(PLANAR_V);

    const int pitchY  = src->GetPitch(PLANAR_Y);
    const int pitchUV = src->GetPitch(PLANAR_U);

#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2)
    { 
      convert_yv24_back_to_yuy2_sse2(srcY, srcU, srcV, dstp, pitchY, pitchUV, dpitch, vi.height, vi.width);
    }
    else
#ifdef X86_32
    if (env->GetCPUFlags() & CPUF_MMX)
    {  // Use MMX
      convert_yv24_back_to_yuy2_mmx(srcY, srcU, srcV, dstp, pitchY, pitchUV, dpitch, vi.height, vi.width);
    }
    else
#endif
#endif
    {
      convert_yv24_back_to_yuy2_c(srcY, srcU, srcV, dstp, pitchY, pitchUV, dpitch, vi.height, vi.width);
    }
    return dst;
  }

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  BYTE* yuv = dst->GetWritePtr();

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSE2)
  {
    if ((src_cs & VideoInfo::CS_BGR32) == VideoInfo::CS_BGR32) {
      convert_rgb_back_to_yuy2_sse2<4>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    } else {
      convert_rgb_back_to_yuy2_sse2<3>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    }
    return dst;
  }

#ifdef X86_32
  if (env->GetCPUFlags() & CPUF_MMX)
  {
    if ((src_cs & VideoInfo::CS_BGR32) == VideoInfo::CS_BGR32) {
      convert_rgb_back_to_yuy2_mmx<4>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    } else {
      convert_rgb_back_to_yuy2_mmx<3>(src->GetReadPtr(), dst->GetWritePtr(), src->GetPitch(), dst->GetPitch(), vi.width, vi.height, matrix);
    }
    return dst;
  }
#endif
#endif

  const BYTE* rgb = src->GetReadPtr() + (vi.height-1) * src->GetPitch(); // Last line

  const int yuv_offset = dst->GetPitch() - dst->GetRowSize();
  const int rgb_offset = -src->GetPitch() - src->GetRowSize(); // moving upwards
  const int rgb_inc = (src_cs&VideoInfo::CS_BGR32)==VideoInfo::CS_BGR32 ? 4 : 3;

  if(0 != matrix.offset_y)
    convert_rgb_back_to_yuy2_c<true>(yuv, rgb, rgb_offset, yuv_offset, vi.height, vi.width, rgb_inc, matrix);
  else
    convert_rgb_back_to_yuy2_c<false>(yuv, rgb, rgb_offset, yuv_offset, vi.height, vi.width, rgb_inc, matrix);

  return dst;
}

AVSValue __cdecl ConvertBackToYUY2::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  if (!clip->GetVideoInfo().IsYUY2())
    return new ConvertBackToYUY2(clip, args[1].AsString(0), env);

  return clip;
}
