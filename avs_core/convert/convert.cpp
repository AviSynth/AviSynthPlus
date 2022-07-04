// Avisynth v2.5.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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


#include "convert.h"
#include "convert_matrix.h"
#include "convert_helper.h"
#include "convert_bits.h"
#include "convert_planar.h"
#include "convert_rgb.h"
#include "convert_yuy2.h"

#ifdef INTEL_INTRINSICS
#include "intel/convert_sse.h"
#include "intel/convert_yuy2_sse.h"
#endif

#include <avs/alignment.h>
#include <avs/minmax.h>
#include <avs/config.h>
#include <tuple>
#include <map>
#include <algorithm>

#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Convert_filters[] = {       // matrix can be "rec601", "rec709", "PC.601" or "PC.709" or "rec2020" or "PC.2020"
  { "ConvertToRGB",   BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)0 },
  { "ConvertToRGB24", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)24 },
  { "ConvertToRGB32", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)32 },
  { "ConvertToRGB48", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)48 },
  { "ConvertToRGB64", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)64 },
  { "ConvertToPlanarRGB",  BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)-1 },
  { "ConvertToPlanarRGBA", BUILTIN_FUNC_PREFIX, "c[matrix]s[interlaced]b[ChromaInPlacement]s[chromaresample]s", ConvertToRGB::Create, (void *)-2 },
  { "ConvertToY8",    BUILTIN_FUNC_PREFIX, "c[matrix]s", ConvertToY::Create, (void*)0 }, // user_data == 0 -> only 8 bit sources
  { "ConvertToYV12",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s", ConvertToYV12::Create, (void*)0 },
  { "ConvertToYV24",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s", ConvertToPlanarGeneric::CreateYUV444, (void*)0},
  { "ConvertToYV16",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s", ConvertToPlanarGeneric::CreateYUV422, (void*)0},
  { "ConvertToYV411", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s", ConvertToPlanarGeneric::CreateYV411, (void*)0},
  { "ConvertToYUY2",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s", ConvertToYUY2::Create },
  { "ConvertBackToYUY2", BUILTIN_FUNC_PREFIX, "c[matrix]s", ConvertBackToYUY2::Create },
  { "ConvertToY",       BUILTIN_FUNC_PREFIX, "c[matrix]s", ConvertToY::Create, (void*)1 }, // user_data == 1 -> any bit depth sources
  { "ConvertToYUV411", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s", ConvertToPlanarGeneric::CreateYV411, (void*)1}, // alias for ConvertToYV411, 8 bit check later
  { "ConvertToYUV420",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s", ConvertToPlanarGeneric::CreateYUV420, (void*)1},
  { "ConvertToYUV422",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s", ConvertToPlanarGeneric::CreateYUV422, (void*)1},
  { "ConvertToYUV444",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s", ConvertToPlanarGeneric::CreateYUV444, (void*)1},
  { "ConvertTo8bit",  BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)8 },
  { "ConvertTo16bit", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)16 },
  { "ConvertToFloat", BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)32 },
  { "ConvertBits",    BUILTIN_FUNC_PREFIX, "c[bits]i[truerange]b[dither]i[dither_bits]i[fulls]b[fulld]b", ConvertBits::Create, (void *)0 },
  { "AddAlphaPlane",  BUILTIN_FUNC_PREFIX, "c[mask].", AddAlphaPlane::Create},
  { "RemoveAlphaPlane",  BUILTIN_FUNC_PREFIX, "c", RemoveAlphaPlane::Create},
  { 0 }
};


/****************************************
*******   Convert to RGB / RGBA   ******
***************************************/

// YUY2 only
ConvertToRGB::ConvertToRGB( PClip _child, bool rgb24, const char* matrix_name,
                           IScriptEnvironment* env )
                           : GenericVideoFilter(_child)
{
  auto frame0 = _child->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);
  matrix_parse_merge_with_props(vi, matrix_name, props, theMatrix, theColorRange, env);

  const int shift = 16; // for integer arithmetic; YUY2 is using 16 bits, later is divided back by 4 or 8
  const int bits_per_pixel = 8; // YUY2
  if (!do_BuildMatrix_Yuv2Rgb(theMatrix, theColorRange, shift, bits_per_pixel, /*ref*/matrix))
    env->ThrowError("ConvertToRGB: invalid \"matrix\" parameter");

  theOutMatrix = Matrix_e::AVS_MATRIX_RGB;
  theOutColorRange = ColorRange_e::AVS_RANGE_FULL;

  // these constants are used with intentional minus operator in core calculations
  matrix.v_g = -matrix.v_g;
  matrix.u_g = -matrix.u_g;
  matrix.offset_y = -matrix.offset_y;

  vi.pixel_type = rgb24 ? VideoInfo::CS_BGR24 : VideoInfo::CS_BGR32;
}

template<int rgb_size>
static void convert_yuy2_to_rgb_c(const BYTE *srcp, BYTE* dstp, int src_pitch, int dst_pitch, int height, int width, int crv, int cgv, int cgu, int cbu, int cy, int tv_scale) {
  srcp += height * src_pitch;
  for (int y = height; y > 0; --y) {
    srcp -= src_pitch;
    int x;
    for (x = 0; x < width-2; x+=2) {
      int scaled_y0 = (srcp[x*2+0] - tv_scale) * cy;
      int u0 = srcp[x*2+1]-128;
      int v0 = srcp[x*2+3]-128;
      int scaled_y1 = (srcp[x*2+2] - tv_scale) * cy;
      int u1 = srcp[x*2+5]-128;
      int v1 = srcp[x*2+7]-128;

      dstp[x*rgb_size + 0] = ScaledPixelClip(scaled_y0 + u0 * cbu);                 // blue
      dstp[x*rgb_size + 1] = ScaledPixelClip(scaled_y0 - u0 * cgu - v0 * cgv); // green
      dstp[x*rgb_size + 2] = ScaledPixelClip(scaled_y0            + v0 * crv); // red

      dstp[(x+1)*rgb_size + 0] = ScaledPixelClip(scaled_y1 + (u0+u1) * (cbu / 2));                     // blue
      dstp[(x+1)*rgb_size + 1] = ScaledPixelClip(scaled_y1 - (u0+u1) * (cgu / 2) - (v0+v1) * (cgv/2)); // green
      dstp[(x+1)*rgb_size + 2] = ScaledPixelClip(scaled_y1                       + (v0+v1) * (crv/2)); // red

      if constexpr(rgb_size == 4) {
        dstp[x*4+3] = 255;
        dstp[x*4+7] = 255;
      }
    }

    int scaled_y0 = (srcp[x*2+0] - tv_scale) * cy;
    int scaled_y1 = (srcp[x*2+2] - tv_scale) * cy;
    int u = srcp[x*2+1]-128;
    int v = srcp[x*2+3]-128;

    dstp[x*rgb_size + 0]     = ScaledPixelClip(scaled_y0 + u * cbu);                 // blue
    dstp[x*rgb_size + 1]     = ScaledPixelClip(scaled_y0 - u * cgu - v * cgv); // green
    dstp[x*rgb_size + 2]     = ScaledPixelClip(scaled_y0           + v * crv); // red

    dstp[(x+1)*rgb_size + 0] = ScaledPixelClip(scaled_y1 + u * cbu);                 // blue
    dstp[(x+1)*rgb_size + 1] = ScaledPixelClip(scaled_y1 - u * cgu - v * cgv); // green
    dstp[(x+1)*rgb_size + 2] = ScaledPixelClip(scaled_y1           + v * crv); // red

    if constexpr(rgb_size == 4) {
      dstp[x*4+3] = 255;
      dstp[x*4+7] = 255;
    }
    dstp += dst_pitch;
  }
}

// YUY2 only
PVideoFrame __stdcall ConvertToRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  const int src_pitch = src->GetPitch();
  const BYTE* srcp = src->GetReadPtr();

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  auto props = env->getFramePropsRW(dst);
  update_Matrix_and_ColorRange(props, theOutMatrix, theOutColorRange, env);
  update_ChromaLocation(props, -1, env); // RGB target: delete _ChromaLocation

  const int dst_pitch = dst->GetPitch();
  BYTE* dstp = dst->GetWritePtr();
  int tv_scale = matrix.offset_y;


#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSE2) {
    if (vi.IsRGB32()) {
      convert_yuy2_to_rgb_sse2<4>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
      matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    } else {
      convert_yuy2_to_rgb_sse2<3>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
        matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    }
  }
  else
#ifdef X86_32
  if (env->GetCPUFlags() & CPUF_INTEGER_SSE) {
    if (vi.IsRGB32()) {
      convert_yuy2_to_rgb_isse<4>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
        matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    } else {
      convert_yuy2_to_rgb_isse<3>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
        matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    }
  }
  else
#endif
#endif
  {
    if (vi.IsRGB32()) {
      convert_yuy2_to_rgb_c<4>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
        matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    } else {
      convert_yuy2_to_rgb_c<3>(srcp, dstp, src_pitch, dst_pitch, vi.height, vi.width,
        matrix.v_r, matrix.v_g, matrix.u_g, matrix.u_b, matrix.y_r, tv_scale);
    }
  }
  return dst;
}

// general for all colorspaces
// however class is constructed only for YUY2 input
AVSValue __cdecl ConvertToRGB::Create(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  const bool haveOpts = args[3].Defined() || args[4].Defined();
  PClip clip = args[0].AsClip();
  const char* const matrix_name = args[1].AsString(0);
  VideoInfo vi = clip->GetVideoInfo();

  // common Create for all CreateRGB24/32/48/64/Planar(RGBP:-1, RGPAP:-2) using user_data
  int target_rgbtype = (int)reinterpret_cast<intptr_t>(user_data);
  // -1,-2: Planar RGB(A)
  //  0: not specified (leave if input is packed RGB, convert to rgb32/64 input colorspace dependent)
  // 24,32,48,64: RGB24/32/48/64

  if (target_rgbtype == 0 && vi.BitsPerComponent() != 8 && vi.BitsPerComponent() != 16)
    env->ThrowError("ConvertToRGB: conversion is allowed only from 8 or 16 bit colorspaces");

  // planar YUV-like
  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    bool needConvertFinalBitdepth = false;
    int finalBitdepth = -1;

    AVSValue new_args[5] = { clip, args[2], args[1], args[3], args[4] };
    // conversion to planar or packed RGB is always from 444
    clip = ConvertToPlanarGeneric::CreateYUV444(AVSValue(new_args, 5), (void *)1, env).AsClip(); // (void *)1: not restricted to 8 bits
    if ((target_rgbtype == 24 || target_rgbtype == 32)) {
      if (vi.BitsPerComponent() != 8) {
        needConvertFinalBitdepth = true;
        finalBitdepth = 8;
        target_rgbtype = (target_rgbtype == 24) ? -1 : -2; // planar rgb intermediate
      }
    }
    else if ((target_rgbtype == 48 || target_rgbtype == 64)) {
      if (vi.BitsPerComponent() != 16) {
        needConvertFinalBitdepth = true;
        finalBitdepth = 16;
        target_rgbtype = (target_rgbtype == 48) ? -1 : -2; // planar rgb intermediate
      }
    }
    int rgbtype_param = 0;
    bool reallyConvert = true;
    switch (target_rgbtype)
    {
    case -1: case -2:
        rgbtype_param = target_rgbtype; break; // planar RGB(A)
    case 0:
        rgbtype_param = vi.ComponentSize() == 1 ? 4 : 8; break; // input bitdepth adaptive
    case 24:
        rgbtype_param = 3; break; // RGB24
    case 32:
        rgbtype_param = 4; break; // RGB32
    case 48: {
            // instead of C code of YUV444P16->RGB48
            // we convert to PlanarRGB then to RGB48 (both is fast)
          AVSValue new_args2[5] = { clip, args[1], args[2], args[3], args[4] };
          clip = ConvertToRGB::Create(AVSValue(new_args2, 5), (void *)-1, env).AsClip();
          vi = clip->GetVideoInfo();
          reallyConvert = false;
          rgbtype_param = 6; // old option RGB48 target, slow C
        }
        break; // RGB48
    case 64: {
        // instead of C code of YUV(A)444P16->RGB64
        // we convert to PlanarRGB(A) then to RGB64 (both is fast)
        AVSValue new_args2[5] = { clip, args[1], args[2], args[3], args[4] };
        clip = ConvertToRGB::Create(AVSValue(new_args2, 5), vi.IsYUVA() ? (void *)-2 : (void *)-1, env).AsClip();
        vi = clip->GetVideoInfo();
        reallyConvert = false;
        rgbtype_param = 8; // old option RGB64 target, slow C
      }
      break; // RGB64
    }
    if (reallyConvert) {
      clip = new ConvertYUV444ToRGB(clip, matrix_name, rgbtype_param, env);

      if (needConvertFinalBitdepth) {
        // from any planar rgb(a) -> rgb24/32/48/64
        clip = new ConvertBits(clip, -1 /*dither_type*/, finalBitdepth /*target_bitdepth*/, true /*assume_truerange*/, 
          ColorRange_e::AVS_RANGE_FULL /*fulls*/, ColorRange_e::AVS_RANGE_FULL /*fulld*/, 
          8 /*n/a dither_bitdepth*/, env);
        vi = clip->GetVideoInfo();

        // source here is always a 8/16bit planar RGB(A), finally it has to be converted to RGB24/32/48/64
        const bool isRGBA = target_rgbtype == -2;
        clip = new PlanarRGBtoPackedRGB(clip, isRGBA);
        vi = clip->GetVideoInfo();
      }
      return clip;
    }
  }

  if (haveOpts)
    env->ThrowError("ConvertToRGB: ChromaPlacement and ChromaResample options are not supported.");

  // planar RGB-like source
  if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    if (target_rgbtype < 0) // planar to planar
    {
      if (vi.IsPlanarRGB()) {
        if (target_rgbtype == -1)
          return clip;
        // prgb->prgba create with default alpha
        return new AddAlphaPlane(clip, nullptr, 0.0f, false, env);
      }
      // planar rgba source
      if (target_rgbtype == -2)
        return clip;
      return new RemoveAlphaPlane(clip, env);
    }

    // planar to packed 24/32/48/64
    bool needConvertFinalBitdepth = false;
    int finalBitdepth = -1;

    if (target_rgbtype == 24 || target_rgbtype == 32) {
      if (vi.BitsPerComponent() != 8) {
        needConvertFinalBitdepth = true;
        finalBitdepth = 8;
      }
    }
    else if (target_rgbtype == 48 || target_rgbtype == 64) {
      if (vi.BitsPerComponent() != 16) {
        needConvertFinalBitdepth = true;
        finalBitdepth = 16;
      }
    }

    if (needConvertFinalBitdepth) {
      // from any bitdepth planar rgb(a) -> 8/16 bits
      clip = new ConvertBits(clip, -1 /*dither_type*/, finalBitdepth /*target_bitdepth*/, true /*assume_truerange*/, 
        ColorRange_e::AVS_RANGE_FULL /*fulls*/, ColorRange_e::AVS_RANGE_FULL /*fulld*/, 
        8 /*n/a dither_bitdepth*/, env);
      vi = clip->GetVideoInfo();
    }

    bool hasAlpha = target_rgbtype == 32 || target_rgbtype == 64 ||
      (target_rgbtype == 0 && vi.IsPlanarRGBA());

    return new PlanarRGBtoPackedRGB(clip, hasAlpha);
  }

  // YUY2
  if (vi.IsYUV()) // at this point IsYUV means YUY2 (non-planar)
  {
    if (target_rgbtype == 48 || target_rgbtype == 64)
      env->ThrowError("ConvertToRGB: conversion from YUY2 is allowed only to 8 bits");
    if (target_rgbtype < 0) {
      // rgb32 intermediate is faster
      clip = new ConvertToRGB(clip, false, matrix_name, env); // YUY2->RGB32
      return new PackedRGBtoPlanarRGB(clip, true, target_rgbtype == -2);
    }
    else
      return new ConvertToRGB(clip, target_rgbtype == 24, matrix_name, env);
  }

  // conversions from packed RGB

  if (target_rgbtype == 24 || target_rgbtype == 32) {
    if (vi.ComponentSize() != 1) {
      // 64->32, 48->24
      clip = new ConvertBits(clip, -1 /*dither_type*/, 8 /*target_bitdepth*/, true /*assume_truerange*/, 
        ColorRange_e::AVS_RANGE_FULL /*fulls*/, ColorRange_e::AVS_RANGE_FULL /*fulld*/, 
        8 /*n/a dither_bitdepth*/, env);
      vi = clip->GetVideoInfo(); // new format
    }
  }
  else if (target_rgbtype == 48 || target_rgbtype == 64) {
    if (vi.ComponentSize() != 2) {
      // 32->64, 24->48
      clip = new ConvertBits(clip, -1 /*dither_type*/, 16 /*target_bitdepth*/, true /*assume_truerange*/, 
        ColorRange_e::AVS_RANGE_FULL /*fulls*/, ColorRange_e::AVS_RANGE_FULL /*fulld*/, 
        8 /*n/a dither_bitdepth*/, env);
      vi = clip->GetVideoInfo(); // new format
    }
  }

  if(target_rgbtype==32 || target_rgbtype==64)
      if (vi.IsRGB24() || vi.IsRGB48())
          return new RGBtoRGBA(clip); // 24->32 or 48->64

  if(target_rgbtype==24 || target_rgbtype==48)
      if (vi.IsRGB32() || vi.IsRGB64())
          return new RGBAtoRGB(clip); // 32->24 or 64->48

  // <0: target is planar RGB(A)
  if (target_rgbtype < 0) {
    // RGB24/32/48/64 ->
    const bool isSrcRGBA = vi.IsRGB32() || vi.IsRGB64();
    const bool isTargetRGBA = target_rgbtype == -2;
    return new PackedRGBtoPlanarRGB(clip, isSrcRGBA, isTargetRGBA);
  }

  return clip;
}


/**********************************
*******   Convert to YV12   ******
*********************************/

// for YUY2->YV12 only
// all other sources use ConvertToPlanarGeneric
ConvertToYV12::ConvertToYV12(PClip _child, bool _interlaced, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  interlaced(_interlaced)
{
  if (vi.width & 1)
    env->ThrowError("ConvertToYV12: Image width must be multiple of 2");

  if (interlaced && (vi.height & 3))
    env->ThrowError("ConvertToYV12: Interlaced image height must be multiple of 4");

  if ((!interlaced) && (vi.height & 1))
    env->ThrowError("ConvertToYV12: Image height must be multiple of 2");

  if (!vi.IsYUY2())
    env->ThrowError("ConvertToYV12: Source must be YUY2.");

  vi.pixel_type = VideoInfo::CS_YV12;
}

PVideoFrame __stdcall ConvertToYV12::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  if (interlaced) {
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2)
    {
      convert_yuy2_to_yv12_interlaced_sse2(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
        dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
        dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
    }
    else
#ifdef X86_32
      if ((env->GetCPUFlags() & CPUF_INTEGER_SSE))
      {
        convert_yuy2_to_yv12_interlaced_isse(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
          dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
          dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
      }
      else
#endif
#endif
      {
        convert_yuy2_to_yv12_interlaced_c(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
          dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
          dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
      }
  }
  else
  {
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2)
    {
      convert_yuy2_to_yv12_progressive_sse2(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
        dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
        dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
    }
    else
#ifdef X86_32
      if ((env->GetCPUFlags() & CPUF_INTEGER_SSE))
      {
        convert_yuy2_to_yv12_progressive_isse(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
          dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
          dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
      }
      else
#endif
#endif
      {
        convert_yuy2_to_yv12_progressive_c(src->GetReadPtr(), src->GetRowSize(), src->GetPitch(),
          dst->GetWritePtr(PLANAR_Y), dst->GetWritePtr(PLANAR_U), dst->GetWritePtr(PLANAR_V),
          dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), src->GetHeight());
      }
  }

  return dst;

}


/**********************************
*******   Convert to YV12   ******
*********************************/


AVSValue __cdecl ConvertToYV12::Create(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  if (only_8bit && vi.BitsPerComponent() != 8)
    env->ThrowError("ConvertToYV12: only 8 bit sources allowed");

  if (vi.IsYUY2() && !args[3].Defined() && !args[4].Defined() && !args[5].Defined())  // User has not requested options, do it fast!
    return new ConvertToYV12(clip,args[1].AsBool(false),env);

  return ConvertToPlanarGeneric::CreateYUV420(args, NULL,env);
}

AVSValue AddAlphaPlane::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  bool isMaskDefined = args[1].Defined();
  bool maskIsClip = false;
  // if mask is not defined and videoformat has Alpha then we return
  if(isMaskDefined && !args[1].IsClip() && !args[1].IsFloat())
    env->ThrowError("AddAlphaPlane: mask parameter should be clip or number");
  const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
  if (!isMaskDefined && (vi.IsPlanarRGBA() || vi.IsYUVA() || vi.IsRGB32() || vi.IsRGB64()))
    return args[0].AsClip();
  PClip alphaClip = nullptr;
  if (isMaskDefined && args[1].IsClip()) {
    const VideoInfo& viAlphaClip = args[1].AsClip()->GetVideoInfo();
    maskIsClip = true;
    if(viAlphaClip.BitsPerComponent() != vi.BitsPerComponent())
      env->ThrowError("AddAlphaPlane: alpha clip is of different bit depth");
    if(viAlphaClip.width != vi.width || viAlphaClip.height != vi.height )
      env->ThrowError("AddAlphaPlane: alpha clip is of different size");
    if (viAlphaClip.IsY())
      alphaClip = args[1].AsClip();
    else if (viAlphaClip.NumComponents() == 4) {
      AVSValue new_args[1] = { args[1].AsClip() };
      alphaClip = env->Invoke("ExtractA", AVSValue(new_args, 1)).AsClip();
    } else {
      env->ThrowError("AddAlphaPlane: alpha clip should be greyscale or should have alpha plane");
    }
    // alphaClip is always greyscale here
  }
  float maskAsFloat = -1.0f;
  if (!maskIsClip)
    maskAsFloat = (float)args[1].AsFloat(-1.0f);
  if (vi.IsRGB24()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    PClip child = env->Invoke("ConvertToRGB32", AVSValue(new_args, 1)).AsClip();
    return new AddAlphaPlane(child, alphaClip, maskAsFloat, isMaskDefined, env);
  } else if(vi.IsRGB48()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    PClip child = env->Invoke("ConvertToRGB64", AVSValue(new_args, 1)).AsClip();
    return new AddAlphaPlane(child, alphaClip, maskAsFloat, isMaskDefined, env);
  }
  return new AddAlphaPlane(args[0].AsClip(), alphaClip, maskAsFloat, isMaskDefined, env);
}

AddAlphaPlane::AddAlphaPlane(PClip _child, PClip _alphaClip, float _mask_f, bool isMaskDefined, IScriptEnvironment* env)
  : GenericVideoFilter(_child), alphaClip(_alphaClip)
{
  if(vi.IsYUY2())
    env->ThrowError("AddAlphaPlane: YUY2 is not allowed");
  if(vi.IsY())
    env->ThrowError("AddAlphaPlane: greyscale source is not allowed");
  if(vi.IsYUV() && !vi.Is420() && !vi.Is422() && !vi.Is444()) // e.g. 410
    env->ThrowError("AddAlphaPlane: YUV format not supported, must be 420, 422 or 444");
  if(!vi.IsYUV() && !vi.IsYUVA() && !vi.IsRGB())
    env->ThrowError("AddAlphaPlane: format not supported");

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (vi.IsYUV()) {
    int pixel_type = vi.pixel_type;
    if (vi.IsYV12())
      pixel_type = VideoInfo::CS_YV12;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_YUV) | VideoInfo::CS_YUVA;
    vi.pixel_type = new_pixel_type;
  } else if(vi.IsPlanarRGB()) {
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_RGB_TYPE) | VideoInfo::CS_RGBA_TYPE;
    vi.pixel_type = new_pixel_type;
  }
  // RGB24 and RGB48 already converted to 32/64
  // RGB32, RGB64, YUVA and RGBA: no change

  // mask parameter. If none->max transparency

  if (!alphaClip) {
    int max_pixel_value = (1 << bits_per_pixel) - 1; // n/a for float
    if (!isMaskDefined) {
      mask_f = 1.0f;
      mask = max_pixel_value;
    }
    else {
      mask_f = _mask_f;
      mask = (mask_f < 0) ? 0 : (mask_f > max_pixel_value) ? max_pixel_value : (int)mask_f;
      mask = clamp(mask, 0, max_pixel_value);
      // no clamp for float
    }
  }
}

PVideoFrame AddAlphaPlane::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  if(vi.IsPlanar())
  {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    // copy existing 3 planes
    for (int p = 0; p < 3; ++p) {
      const int plane = planes[p];
      env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
           src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
    }
  } else {
    // Packed RGB, already converted to RGB32 or RGB64
    env->BitBlt(dst->GetWritePtr(), dst->GetPitch(), src->GetReadPtr(),
      src->GetPitch(), src->GetRowSize(), src->GetHeight());
  }

  if (vi.IsPlanarRGBA() || vi.IsYUVA()) {
    if (alphaClip) {
      PVideoFrame srcAlpha = alphaClip->GetFrame(n, env);
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), srcAlpha->GetReadPtr(PLANAR_Y),
        srcAlpha->GetPitch(PLANAR_Y), srcAlpha->GetRowSize(PLANAR_Y), srcAlpha->GetHeight(PLANAR_Y));
    }
    else {
      // default constant
      const int dst_pitchA = dst->GetPitch(PLANAR_A);
      BYTE* dstp_a = dst->GetWritePtr(PLANAR_A);
      const int heightA = dst->GetHeight(PLANAR_A);

      switch (vi.ComponentSize())
      {
      case 1:
        fill_plane<BYTE>(dstp_a, heightA, dst_pitchA, mask);
        break;
      case 2:
        fill_plane<uint16_t>(dstp_a, heightA, dst_pitchA, mask);
        break;
      case 4:
        fill_plane<float>(dstp_a, heightA, dst_pitchA, mask_f);
        break;
      }
    }
    return dst;
  }
  // RGB32 and RGB64

  BYTE* pf = dst->GetWritePtr();
  int pitch = dst->GetPitch();
  int rowsize = dst->GetRowSize();
  int height = dst->GetHeight();
  int width = vi.width;

  if (alphaClip) {
    // fill by alpha clip already converted to grey-only
    PVideoFrame srcAlpha = alphaClip->GetFrame(n, env);
    const BYTE* srcp_a = srcAlpha->GetReadPtr(PLANAR_Y);
    size_t pitch_a = srcAlpha->GetPitch(PLANAR_Y);

    pf += pitch * (vi.height - 1); // start from bottom: packed RGB is upside down

    if (vi.IsRGB32()) {
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
          pf[x*4+3] = srcp_a[x];
        }
        pf -= pitch; // packed RGB is upside down
        srcp_a += pitch_a;
      }
    }
    else if (vi.IsRGB64()) {
      rowsize /= sizeof(uint16_t);
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x ++) {
          reinterpret_cast<uint16_t *>(pf)[x*4+3] = reinterpret_cast<const uint16_t *>(srcp_a)[x];
        }
        pf -= pitch; // packed RGB is upside down
        srcp_a += pitch_a;
      }
    }
  }
  else {
    // fill with constant
    if (vi.IsRGB32()) {
      for (int y = 0; y < height; y++) {
        for (int x = 3; x < rowsize; x += 4) {
          pf[x] = mask;
        }
        pf += pitch;
      }
    }
    else if (vi.IsRGB64()) {
      rowsize /= sizeof(uint16_t);
      for (int y = 0; y < height; y++) {
        for (int x = 3; x < rowsize; x += 4) {
          reinterpret_cast<uint16_t *>(pf)[x] = mask;
        }
        pf += pitch;
      }
    }
  }

  return dst;
}

AVSValue RemoveAlphaPlane::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  // if videoformat has no Alpha then we return
  const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();
  if(vi.IsPlanar() && (vi.IsYUV() || vi.IsPlanarRGB())) // planar and no alpha
    return args[0].AsClip();
  if (vi.IsYUY2()) // YUY2: no alpha
    return args[0].AsClip();
  if(vi.IsRGB24() || vi.IsRGB48()) // packed RGB and no alpha
    return args[0].AsClip();
  if (vi.IsRGB32()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    return env->Invoke("ConvertToRGB24", AVSValue(new_args, 1)).AsClip();
  }
  if (vi.IsRGB64()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    return env->Invoke("ConvertToRGB48", AVSValue(new_args, 1)).AsClip();
  }
  return new RemoveAlphaPlane(args[0].AsClip(), env);
}

RemoveAlphaPlane::RemoveAlphaPlane(PClip _child, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{
  if(vi.IsYUY2())
    env->ThrowError("RemoveAlphaPlane: YUY2 is not allowed");
  if(vi.IsY())
    env->ThrowError("RemoveAlphaPlane: greyscale source is not allowed");

  if (vi.IsYUVA()) {
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_YUVA) | VideoInfo::CS_YUV;
    vi.pixel_type = new_pixel_type;
  } else if(vi.IsPlanarRGBA()) {
    int pixel_type = vi.pixel_type;
    int new_pixel_type = (pixel_type & ~VideoInfo::CS_RGBA_TYPE) | VideoInfo::CS_RGB_TYPE;
    vi.pixel_type = new_pixel_type;
  }
}

PVideoFrame RemoveAlphaPlane::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  // Packed RGB: already handled in ::Create through Invoke 32->24 or 64->48 conversion
  // only planar here
  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  // Abuse Subframe to snatch the YUV/GBR planes
  return env->SubframePlanar(src, 0, src->GetPitch(planes[0]), src->GetRowSize(planes[0]), src->GetHeight(planes[0]),0,0,src->GetPitch(planes[1]));

#if 0
  // BitBlt version. Kept for reference
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  // copy 3 planes w/o alpha
  for (int p = 0; p < 3; ++p) {
    const int plane = planes[p];
    env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
      src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
  }
return dst;
#endif
}

