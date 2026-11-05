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

#include "transform.h"
#include "../convert/convert_matrix.h"
#include "../convert/convert_helper.h"
#include <avs/minmax.h>
#include "../core/bitblt.h"
#include <stdint.h>
#include "resample_functions.h"
#include "../convert/convert_planar.h"
#include "combine.h"
#include <vector>
#include <cmath>



/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Transform_filters[] = {
  { "FlipVertical",   BUILTIN_FUNC_PREFIX, "c", FlipVertical::Create },
  { "FlipHorizontal", BUILTIN_FUNC_PREFIX, "c", FlipHorizontal::Create },
  { "Crop",           BUILTIN_FUNC_PREFIX, "ciiii[align]b", Crop::Create },              // left, top, width, height *OR*
                                                  //  left, top, -right, -bottom (VDub style)
  { "CropBottom", BUILTIN_FUNC_PREFIX, "ci", Create_CropBottom },      // bottom amount
  { "AddBorders", BUILTIN_FUNC_PREFIX, "ciiii[color]i[color_yuv]i[resample]s[param1]f[param2]f[param3]f[r]i", AddBorders::Create },  // left, top, right, bottom [,color] [,color_yuv]
  { "Letterbox",  BUILTIN_FUNC_PREFIX, "cii[x1]i[x2]i[color]i[color_yuv]i[resample]s[param1]f[param2]f[param3]f[r]i", Create_Letterbox },       // top, bottom, [left], [right] [,color] [,color_yuv]
  { 0 }
};





/********************************
 *******   Flip Vertical   ******
 ********************************/

PVideoFrame FlipVertical::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();
  int row_size = src->GetRowSize();
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  env->BitBlt(dstp, dst_pitch, srcp + (vi.height-1) * src_pitch, -src_pitch, row_size, vi.height);

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  int planeUB = isRGBPfamily ? PLANAR_B : PLANAR_U;
  int planeVR = isRGBPfamily ? PLANAR_R : PLANAR_V;

  if (src->GetPitch(planeUB)) {
    srcp = src->GetReadPtr(planeUB);
    dstp = dst->GetWritePtr(planeUB);
    row_size = src->GetRowSize(planeUB);
    src_pitch = src->GetPitch(planeUB);
    dst_pitch = dst->GetPitch(planeUB);
    env->BitBlt(dstp, dst_pitch, srcp + (src->GetHeight(planeUB)-1) * src_pitch, -src_pitch, row_size, src->GetHeight(planeUB));

    srcp = src->GetReadPtr(planeVR);
    dstp = dst->GetWritePtr(planeVR);
    env->BitBlt(dstp, dst_pitch, srcp + (src->GetHeight(planeVR)-1) * src_pitch, -src_pitch, row_size, src->GetHeight(planeVR));

    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    {
      srcp = src->GetReadPtr(PLANAR_A);
      dstp = dst->GetWritePtr(PLANAR_A);
      row_size = src->GetRowSize(PLANAR_A);
      src_pitch = src->GetPitch(PLANAR_A);
      dst_pitch = dst->GetPitch(PLANAR_A);
      env->BitBlt(dstp, dst_pitch, srcp + (src->GetHeight(PLANAR_A)-1) * src_pitch, -src_pitch, row_size, src->GetHeight(PLANAR_A));
    }
  }
  return dst;
}

AVSValue __cdecl FlipVertical::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  AVS_UNUSED(env);
  return new FlipVertical(args[0].AsClip());
}



/********************************
 *******   Flip Horizontal   ******
 ********************************/

template<typename pixel_t>
static void flip_horizontal_plane_c(BYTE* dstp, const BYTE* srcp, int dst_pitch, int src_pitch, int width, int height) {
  width = width / sizeof(pixel_t); // width is called with GetRowSize value
  dstp += (width - 1) * sizeof(pixel_t);
  for (int y = 0; y < height; y++) { // Loop planar luma.
    for (int x = 0; x < width; x++) {
      (reinterpret_cast<pixel_t *>(dstp))[-x] = (reinterpret_cast<const pixel_t *>(srcp))[x];
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

PVideoFrame FlipHorizontal::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();
  int width = src->GetRowSize();
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  int height = src->GetHeight();
  if (vi.IsYUY2()) { // Avoid flipping UV in YUY2 mode.
    srcp += width;
    srcp -= 4;
    for (int y = 0; y<height; y++) {
      for (int x = 0; x<width; x += 4) {
        dstp[x] = srcp[-x+2];
        dstp[x+1] = srcp[-x+1];
        dstp[x+2] = srcp[-x];
        dstp[x+3] = srcp[-x+3];
      }
      srcp += src_pitch;
      dstp += dst_pitch;
    }
    return dst;
  }

  typedef void(*FlipFuncPtr) (BYTE * dstp, const BYTE * srcp, int dst_pitch, int src_pitch, int width, int height);
  FlipFuncPtr flip_h_func;

  if (vi.IsPlanar()) {
    switch (vi.ComponentSize()) // AVS16
    {
    case 1: flip_h_func = flip_horizontal_plane_c<uint8_t>; break;
    case 2: flip_h_func = flip_horizontal_plane_c<uint16_t>; break;
    default: // 4 float
       flip_h_func = flip_horizontal_plane_c<float>; break;
    }
    flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);

    bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
    int planeUB = isRGBPfamily ? PLANAR_B : PLANAR_U;
    int planeVR = isRGBPfamily ? PLANAR_R : PLANAR_V;

    if (src->GetPitch(planeUB)) {
      srcp = src->GetReadPtr(planeUB);
      dstp = dst->GetWritePtr(planeUB);
      width = src->GetRowSize(planeUB);
      src_pitch = src->GetPitch(planeUB);
      dst_pitch = dst->GetPitch(planeUB);
      height = src->GetHeight(planeUB);
      flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);

      srcp = src->GetReadPtr(planeVR);
      dstp = dst->GetWritePtr(planeVR);

      flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);

      if (vi.IsYUVA() || vi.IsPlanarRGBA())
      {
        srcp = src->GetReadPtr(PLANAR_A);
        dstp = dst->GetWritePtr(PLANAR_A);
        width = src->GetRowSize(PLANAR_A);
        src_pitch = src->GetPitch(PLANAR_A);
        dst_pitch = dst->GetPitch(PLANAR_A);
        height = src->GetHeight(PLANAR_A);
        flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);
      }
    }
    return dst;
  }

  // width is GetRowSize
  if (vi.IsRGB32()) { // fast method
    flip_h_func = flip_horizontal_plane_c<uint32_t>;
    flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);
  }
  else if (vi.IsRGB64()) {
    flip_h_func = flip_horizontal_plane_c<uint64_t>;
    flip_h_func(dstp, srcp, dst_pitch, src_pitch, width, height);
  }
  else if (vi.IsRGB24()) {
    dstp += width - 3;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x += 3) {
          dstp[-x + 0] = srcp[x + 0];
          dstp[-x + 1] = srcp[x + 1];
          dstp[-x + 2] = srcp[x + 2];
      }
      srcp += src_pitch;
      dstp += dst_pitch;
    }
  }
  else if (vi.IsRGB48()) {
    dstp += width - 3 * sizeof(uint16_t);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width / 2 /*sizeof(uint16_t)*/; x += 3) {
        reinterpret_cast<uint16_t*>(dstp)[-x + 0] = reinterpret_cast<const uint16_t*>(srcp)[x + 0];
        reinterpret_cast<uint16_t*>(dstp)[-x + 1] = reinterpret_cast<const uint16_t*>(srcp)[x + 1];
        reinterpret_cast<uint16_t*>(dstp)[-x + 2] = reinterpret_cast<const uint16_t*>(srcp)[x + 2];
      }
      srcp += src_pitch;
      dstp += dst_pitch;
    }
  }
  return dst;
}


AVSValue __cdecl FlipHorizontal::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  AVS_UNUSED(env);
  return new FlipHorizontal(args[0].AsClip());
}





/******************************
 *******   Crop Filter   ******
 *****************************/

Crop::Crop(int _left, int _top, int _width, int _height, bool _align, PClip _child, IScriptEnvironment* env)
 : GenericVideoFilter(_child), align(FRAME_ALIGN - 1), xsub(0), ysub(0)
{
  AVS_UNUSED(_align);
  // _align parameter exists only for the backward compatibility.

  /* Negative values -> VDub-style syntax
     Namely, Crop(a, b, -c, -d) will crop c pixels from the right and d pixels from the bottom.
     Flags on 0 values too since AFAICT it's much more useful to this syntax than the standard one. */
  if ( (_left<0) || (_top<0) )
    env->ThrowError("Crop: Top and Left must be more than 0");

  if (_width <= 0)
      _width = vi.width - _left + _width;
  if (_height <= 0)
      _height = vi.height - _top + _height;

  if (_width <=0)
    env->ThrowError("Crop: Destination width is 0 or less.");

  if (_height<=0)
    env->ThrowError("Crop: Destination height is 0 or less.");

  if (_left + _width > vi.width || _top + _height > vi.height)
    env->ThrowError("Crop: you cannot use crop to enlarge or 'shift' a clip");

  isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  hasAlpha = vi.IsPlanarRGBA() || vi.IsYUVA();

  if (vi.IsYUV() || vi.IsYUVA()) {
    if (vi.NumComponents() > 1 && !vi.IsYA()) { // Y+A has no chroma
      xsub=vi.GetPlaneWidthSubsampling(PLANAR_U);
      ysub=vi.GetPlaneHeightSubsampling(PLANAR_U);
    }
    const int xmask = (1 << xsub) - 1;
    const int ymask = (1 << ysub) - 1;

    // YUY2, etc, ... can only crop to even pixel boundaries horizontally
    if (_left   & xmask)
      env->ThrowError("Crop: YUV image can only be cropped by Mod %d (left side).", xmask+1);
    if (_width  & xmask)
      env->ThrowError("Crop: YUV image can only be cropped by Mod %d (right side).", xmask+1);
    if (_top    & ymask)
      env->ThrowError("Crop: YUV image can only be cropped by Mod %d (top).", ymask+1);
    if (_height & ymask)
      env->ThrowError("Crop: YUV image can only be cropped by Mod %d (bottom).", ymask+1);
  } else if (!isRGBPfamily) {
    // RGB is upside-down
    _top = vi.height - _height - _top;
  }

  left_bytes = vi.BytesFromPixels(_left);
  top = _top;
  vi.width = _width;
  vi.height = _height;

}


PVideoFrame Crop::GetFrame(int n, IScriptEnvironment* env_)
{
  InternalEnvironment* IEnv = GetAndRevealCamouflagedEnv(env_); // though here a static cast would do
  IScriptEnvironment* env = static_cast<IScriptEnvironment*>(IEnv);

  PVideoFrame frame = child->GetFrame(n, env);

  int plane0 = isRGBPfamily ? PLANAR_G : PLANAR_Y;
  int plane1 = isRGBPfamily ? PLANAR_B : PLANAR_U;
  int plane2 = isRGBPfamily ? PLANAR_R : PLANAR_V;

  const BYTE* srcp0 = frame->GetReadPtr(plane0) + top *  frame->GetPitch(plane0) + left_bytes;
  const BYTE* srcp1 = frame->GetReadPtr(plane1) + (top>>ysub) *  frame->GetPitch(plane1) + (left_bytes>>xsub);
  const BYTE* srcp2 = frame->GetReadPtr(plane2) + (top>>ysub) *  frame->GetPitch(plane2) + (left_bytes>>xsub);

  size_t _align;

  if (frame->GetPitch(plane1) && (!vi.IsYV12() || env->PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentTest)))
    _align = this->align & ((size_t)srcp0|(size_t)srcp1|(size_t)srcp2);
  else
    _align = this->align & (size_t)srcp0;

  // Ignore alignment for CUDA. Clip should be explicitly aligned by Align()
  if (0 != _align && (IEnv->GetDeviceType() == DEV_TYPE_CPU)) {
    PVideoFrame dst = env->NewVideoFrameP(vi, &frame, (int)align+1);

    env->BitBlt(dst->GetWritePtr(plane0), dst->GetPitch(plane0), srcp0,
      frame->GetPitch(plane0), dst->GetRowSize(plane0), dst->GetHeight(plane0));

    env->BitBlt(dst->GetWritePtr(plane1), dst->GetPitch(plane1), srcp1,
      frame->GetPitch(plane1), dst->GetRowSize(plane1), dst->GetHeight(plane1));

    env->BitBlt(dst->GetWritePtr(plane2), dst->GetPitch(plane2), srcp2,
      frame->GetPitch(plane2), dst->GetRowSize(plane2), dst->GetHeight(plane2));

    if(hasAlpha)
      env->BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), frame->GetReadPtr(PLANAR_A) + top *  frame->GetPitch(PLANAR_A) + left_bytes,
        frame->GetPitch(PLANAR_A), dst->GetRowSize(PLANAR_A), dst->GetHeight(PLANAR_A));

    return dst;
  }

  // subframe is preserving frame properties
  if (!frame->GetPitch(plane1)) {
    // No U/B (and thus V/R) plane (zero pitch)
    if (hasAlpha) {
      // Only Y+A can reach here.
      // dummy 0 offset/pitch for the unused U/V args
      // Y and A are both full sized.
      return env->SubframePlanarA(frame, top * frame->GetPitch() + left_bytes, frame->GetPitch(), vi.RowSize(), vi.height,
        0, 0, 0, top * frame->GetPitch(PLANAR_A) + left_bytes);
    }
    return env->Subframe(frame, top * frame->GetPitch() + left_bytes, frame->GetPitch(), vi.RowSize(), vi.height);
  }
  else {
    if (hasAlpha) {

      return env->SubframePlanarA(frame, top * frame->GetPitch() + left_bytes, frame->GetPitch(), vi.RowSize(), vi.height,
        (top >> ysub) * frame->GetPitch(plane1) + (left_bytes >> xsub),
        (top >> ysub) * frame->GetPitch(plane2) + (left_bytes >> xsub),
        frame->GetPitch(plane1), top * frame->GetPitch(PLANAR_A) + left_bytes);
    }
    else {
      return env->SubframePlanar(frame, top * frame->GetPitch() + left_bytes, frame->GetPitch(), vi.RowSize(), vi.height,
        (top >> ysub) * frame->GetPitch(plane1) + (left_bytes >> xsub),
        (top >> ysub) * frame->GetPitch(plane2) + (left_bytes >> xsub),
        frame->GetPitch(plane1));
    }
  }
}

int __stdcall Crop::SetCacheHints(int cachehints, int frame_range) {
  AVS_UNUSED(frame_range);
  switch (cachehints) {
  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;
  case CACHE_GET_DEV_TYPE:
    return GetDeviceTypes(child) & (DEV_TYPE_CPU | DEV_TYPE_CUDA);
  }
  return 0;
}


AVSValue __cdecl Crop::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Crop( args[1].AsInt(), args[2].AsInt(), args[3].AsInt(), args[4].AsInt(), args[5].AsBool(true),
                   args[0].AsClip(), env );
}





/******************************
 *******   Add Borders   ******
 *****************************/

AddBorders::AddBorders(int _left, int _top, int _right, int _bot, int _clr, bool _force_color_as_yuv, PClip _child, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  left(max(0, _left)), top(max(0, _top)), right(max(0, _right)), bot(max(0, _bot)),
  clr(_clr),
  xsub(0), ysub(0),
  force_color_as_yuv(_force_color_as_yuv)

{
  if (vi.IsYUV() || vi.IsYUVA()) {
    if (vi.NumComponents() > 1 && !vi.IsYA()) { // Y+A has no chroma
      xsub=vi.GetPlaneWidthSubsampling(PLANAR_U);
      ysub=vi.GetPlaneHeightSubsampling(PLANAR_U);
    }

    const int xmask = (1 << xsub) - 1;
    const int ymask = (1 << ysub) - 1;

    // YUY2, etc, ... can only add even amounts
    if (_left  & xmask)
      env->ThrowError("AddBorders: YUV image can only add by Mod %d (left side).", xmask+1);
    if (_right & xmask)
      env->ThrowError("AddBorders: YUV image can only add by Mod %d (right side).", xmask+1);

    if (_top   & ymask)
      env->ThrowError("AddBorders: YUV image can only add by Mod %d (top).", ymask+1);
    if (_bot   & ymask)
      env->ThrowError("AddBorders: YUV image can only add by Mod %d (bottom).", ymask+1);
  } else if (!vi.IsPlanarRGB() && !vi.IsPlanarRGBA()){
    // RGB is upside-down
    int t = top; top = bot; bot = t;
  }
  vi.width += left+right;
  vi.height += top+bot;
}

template<typename pixel_t>
static inline pixel_t GetHbdColorFromByte(uint8_t color, bool fullscale, int bits_per_pixel, bool chroma)
{
  if constexpr(sizeof(pixel_t) == 1) return color;
  else if constexpr(sizeof(pixel_t) == 2) return (pixel_t)(fullscale ? (color * ((1 << bits_per_pixel)-1)) / 255 : (int)color << (bits_per_pixel - 8));
  else {
    if (chroma)
      return (pixel_t)uv8tof(color);  // float, scale, 128=0.0f
    else
      return (pixel_t)(fullscale ? c8tof(color) : y8tof(color)); // float, scale to [0..1]
  }
}

template<typename pixel_t>
static void addborders_planar(PVideoFrame &dst, PVideoFrame &src, VideoInfo &vi, int top, int bot, int left, int right, int color, bool isYUV, bool force_color_as_yuv, int bits_per_pixel)
{
  const unsigned int colr = isYUV && !force_color_as_yuv ? RGB2YUV_Rec601(color) : color;
  const unsigned char YBlack=(unsigned char)((colr >> 16) & 0xff);
  const unsigned char UBlack=(unsigned char)((colr >>  8) & 0xff);
  const unsigned char VBlack=(unsigned char)((colr      ) & 0xff);
  const unsigned char ABlack=(unsigned char)((colr >> 24) & 0xff);

  int planes_yuva[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_ya[2]   = { PLANAR_Y, PLANAR_A };
  int planes_rgb[4]  = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = vi.IsYA() ? planes_ya : isYUV ? planes_yuva : planes_rgb;
  uint8_t colors_yuva[4] = { YBlack, UBlack, VBlack, ABlack };
  uint8_t colors_ya[2]   = { YBlack, ABlack };
  uint8_t colors_rgb[4]  = { UBlack, VBlack, YBlack, ABlack }; // mapping for planar RGB
  uint8_t *colors = vi.IsYA() ? colors_ya : isYUV ? colors_yuva : colors_rgb;
  for (int p = 0; p < vi.NumComponents(); p++)
  {
    int plane = planes[p];
    int src_pitch = src->GetPitch(plane);
    int src_rowsize = src->GetRowSize(plane);
    int src_height = src->GetHeight(plane);

    int dst_pitch = dst->GetPitch(plane);

    int xsub=vi.GetPlaneWidthSubsampling(plane);
    int ysub=vi.GetPlaneHeightSubsampling(plane);

    const int initial_black = (top >> ysub) * dst_pitch + vi.BytesFromPixels(left >> xsub);
    const int middle_black = dst_pitch - src_rowsize;
    const int final_black = (bot >> ysub) * dst_pitch + vi.BytesFromPixels(right >> xsub) +
                             (dst_pitch - dst->GetRowSize(plane));

    const bool chroma = plane == PLANAR_U || plane == PLANAR_V;

    // float alpha is full range. For integers alpha is still a shift here.
    const bool fullscale = !isYUV || (sizeof(pixel_t) == 4 && plane == PLANAR_A);
    pixel_t current_color = GetHbdColorFromByte<pixel_t>(colors[p], fullscale, bits_per_pixel, chroma);

    BYTE *dstp = dst->GetWritePtr(plane);
    // copy original
    BitBlt(dstp+initial_black, dst_pitch, src->GetReadPtr(plane), src_pitch, src_rowsize, src_height);
    // add top
    for (size_t a = 0; a<initial_black / sizeof(pixel_t); a++) {
      reinterpret_cast<pixel_t *>(dstp)[a] = current_color;
    }
    // middle right + left (fill overflows from right to left)
    dstp += initial_black + src_rowsize;
    for (int y = src_height-1; y>0; --y) {
      for (size_t b = 0; b<middle_black / sizeof(pixel_t); b++) {
        reinterpret_cast<pixel_t *>(dstp)[b] = current_color;
      }
      dstp += dst_pitch;
    }
    // bottom
    for (size_t c = 0; c<final_black / sizeof(pixel_t); c++)
      reinterpret_cast<pixel_t *>(dstp)[c] = current_color;
  }
}

PVideoFrame AddBorders::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  if (vi.IsPlanar()) {
    int bits_per_pixel = vi.BitsPerComponent();
    bool isYUV = vi.IsYUV() || vi.IsYUVA();
    switch(vi.ComponentSize()) {
    case 1: addborders_planar<uint8_t>(dst, src, vi, top, bot, left, right, clr, isYUV, force_color_as_yuv /*like MODE_COLOR_YUV in BlankClip */, bits_per_pixel); break;
    case 2: addborders_planar<uint16_t>(dst, src, vi, top, bot, left, right,  clr, isYUV, force_color_as_yuv, bits_per_pixel); break;
    default: //case 4:
      addborders_planar<float>(dst, src, vi, top, bot, left, right, clr, isYUV, force_color_as_yuv, bits_per_pixel); break;
    }
    return dst;
  }

  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();
  const int src_pitch = src->GetPitch();
  const int dst_pitch = dst->GetPitch();
  const int src_row_size = src->GetRowSize();
  const int dst_row_size = dst->GetRowSize();
  const int src_height = src->GetHeight();

  const int initial_black = top * dst_pitch + vi.BytesFromPixels(left);
  const int middle_black = dst_pitch - src_row_size;
  const int final_black = bot * dst_pitch + vi.BytesFromPixels(right)
    + (dst_pitch - dst_row_size);

  if (vi.IsYUY2()) {
    const unsigned int colr = force_color_as_yuv ? clr : RGB2YUV_Rec601(clr);
    const uint32_t black = (colr>>16) * 0x010001 + ((colr>>8)&255) * 0x0100 + (colr&255) * 0x01000000;

    BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);
    for (int a = 0; a<initial_black; a += 4) {
      *(uint32_t*)(dstp+a) = black;
    }
    dstp += initial_black + src_row_size;
    for (int y = src_height-1; y>0; --y) {
      for (int b = 0; b<middle_black; b += 4) {
        *(uint32_t*)(dstp+b) = black;
      }
      dstp += dst_pitch;
    }
    for (int c = 0; c<final_black; c += 4) {
      *(uint32_t*)(dstp+c) = black;
    }
  } else if (vi.IsRGB24()) {
    const unsigned char  clr0 = (unsigned char)(clr & 0xFF);
    const unsigned short clr1 = (unsigned short)(clr >> 8);
    const int leftbytes = vi.BytesFromPixels(left);
    const int leftrow = src_row_size + leftbytes;
    const int rightbytes = vi.BytesFromPixels(right);
    const int rightrow = dst_pitch - dst_row_size + rightbytes;

    BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);
    /* Cannot use *_black optimisation as pitch may not be mod 3 */
    for (int y = top; y>0; --y) {
      for (int i = 0; i<dst_row_size; i += 3) {
        dstp[i] = clr0;
        *(uint16_t*)(dstp+i+1) = clr1;
      }
      dstp += dst_pitch;
    }
    for (int y = src_height; y>0; --y) {
      for (int i = 0; i<leftbytes; i += 3) {
        dstp[i] = clr0;
        *(uint16_t*)(dstp+i+1) = clr1;
      }
      dstp += leftrow;
      for (int i = 0; i<rightbytes; i += 3) {
        dstp[i] = clr0;
        *(uint16_t*)(dstp+i+1) = clr1;
      }
      dstp += rightrow;
    }
    for (int y = bot; y>0; --y) {
      for (int i = 0; i<dst_row_size; i += 3) {
        dstp[i] = clr0;
        *(uint16_t*)(dstp+i+1) = clr1;
      }
      dstp += dst_pitch;
    }
  }
  else if (vi.IsRGB32()) {
    BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);
    for (int i = 0; i<initial_black; i += 4) {
      *(uint32_t*)(dstp+i) = clr;
    }
    dstp += initial_black + src_row_size;
    for (int y = src_height-1; y>0; --y) {
      for (int i = 0; i<middle_black; i += 4) {
        *(uint32_t*)(dstp+i) = clr;
      }
      dstp += dst_pitch;
    } // for y
    for (int i = 0; i<final_black; i += 4) {
      *(uint32_t*)(dstp+i) = clr;
    }
  } else if (vi.IsRGB48()) {
    const uint16_t  clr0 = GetHbdColorFromByte<uint16_t>(clr & 0xFF, true, 16, false);
    uint32_t clr1 =
      ((uint32_t)GetHbdColorFromByte<uint16_t>((clr >> 16) & 0xFF, true, 16, false) << (8 * 2)) +
      ((uint32_t)GetHbdColorFromByte<uint16_t>((clr >> 8) & 0xFF, true, 16, false));
    const int leftbytes = vi.BytesFromPixels(left);
    const int leftrow = src_row_size + leftbytes;
    const int rightbytes = vi.BytesFromPixels(right);
    const int rightrow = dst_pitch - dst_row_size + rightbytes;

    BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);
    /* Cannot use *_black optimisation as pitch may not be mod 3 */
    for (int y = top; y>0; --y) {
      for (int i = 0; i<dst_row_size; i += 6) {
        *(uint16_t*)(dstp+i) = clr0;
        *(uint32_t*)(dstp+i+2) = clr1;
      }
      dstp += dst_pitch;
    }
    for (int y = src_height; y>0; --y) {
      for (int i = 0; i<leftbytes; i += 6) {
        *(uint16_t*)(dstp+i) = clr0;
        *(uint32_t*)(dstp+i+2) = clr1;
      }
      dstp += leftrow;
      for (int i = 0; i<rightbytes; i += 6) {
        *(uint16_t*)(dstp+i) = clr0;
        *(uint32_t*)(dstp+i+2) = clr1;
      }
      dstp += rightrow;
    }
    for (int y = bot; y>0; --y) {
      for (int i = 0; i<dst_row_size; i += 6) {
        *(uint16_t*)(dstp+i) = clr0;
        *(uint32_t*)(dstp+i+2) = clr1;
      }
      dstp += dst_pitch;
    }
  } else if (vi.IsRGB64()) {
    BitBlt(dstp+initial_black, dst_pitch, srcp, src_pitch, src_row_size, src_height);

    uint64_t clr64 =
      ((uint64_t)GetHbdColorFromByte<uint16_t>((clr >> 24) & 0xFF, true, 16, false) << (24 * 2)) +
      ((uint64_t)GetHbdColorFromByte<uint16_t>((clr >> 16) & 0xFF, true, 16, false) << (16 * 2)) +
      ((uint64_t)GetHbdColorFromByte<uint16_t>((clr >> 8) & 0xFF, true, 16, false) << (8 * 2)) +
      ((uint64_t)GetHbdColorFromByte<uint16_t>((clr) & 0xFF, true, 16, false));

    for (int i = 0; i<initial_black; i += 8) {
      *(uint64_t*)(dstp+i) = clr64;
    }
    dstp += initial_black + src_row_size;
    for (int y = src_height-1; y>0; --y) {
      for (int i = 0; i<middle_black; i += 8) {
        *(uint64_t*)(dstp+i) = clr64;
      }
      dstp += dst_pitch;
    } // for y
    for (int i = 0; i<final_black; i += 8) {
      *(uint64_t*)(dstp+i) = clr64;
    }
  }

  return dst;
}

// Optional transient area filtering when radius > 0
// The primary aim to blur the border neighborhood to prevent excessive ringing on
// a following upscaleing process
static PClip AddBorderPostProcess(PClip child,
  int left, int top, int right, int bottom,
  const AVSValue& _resample, const AVSValue& _param1, const AVSValue& _param2, const AVSValue& _param3, const AVSValue& _flt_rad,
  int forced_chroma_placement,
  IScriptEnvironment* env) {

  // filtering radius, default 0: no transient area filtering
  int r = _flt_rad.AsInt(0);

  if (r == 0)
    return child;

  // Default: only the border side gets the blur treatment
  // Minus "r": only the outer - added border - part is blurred
  // though this is usually not enough to prevent ringing when upscaling.
  const bool both_side = (r > 0);
  if (r < 0)
    r = -r;

  const VideoInfo& vi = child->GetVideoInfo();

  /*if (vi.IsYUY2())
    env->ThrowError("AddBorders: YUY2 transient filtering not supported, use YV16.");
    */

  AVSValue param1 = _param1;
  AVSValue param2 = _param2;
  AVSValue param3 = _param3;

  // evaluate resample parameters only radius > 0
  const char* resampler_name = _resample.AsString("gauss");
  if (_stricmp(resampler_name, "gauss") == 0) {
    // for gauss, defaults are different from what we use in chroma resamplers.
    // here we use this filter for blurring (convolution use)
    // p = 10, b = 2.71828, s = 0
    param1 = _param1.AsDblDef(10); // p
    param2 = _param2.AsDblDef(2.718281828); // b
    param3 = _param3.AsDblDef(0); // 0
  }
  ResamplingFunction* filter = getResampler(resampler_name, param1, param2, param3, false, env);
  if (filter == nullptr)
    env->ThrowError("AddBorders: unknown resampler name: %s", resampler_name);

  const bool grey = vi.IsY();
  const bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  const int shift_w = (vi.IsPlanar() && !grey && !isRGBPfamily) ? vi.GetPlaneWidthSubsampling(PLANAR_U) : 0;
  const int shift_h = (vi.IsPlanar() && !grey && !isRGBPfamily) ? vi.GetPlaneHeightSubsampling(PLANAR_U) : 0;
  const int shift = max(shift_w, shift_h); // worst case

  // adjust radius for worst case chroma
  r = (r + (1 << shift) - 1) & ~((1 << shift) - 1); // chroma friendly

  const int MIN_FILTERING_EXTENT = 10; // +/-
  // but if one of the margins are smaller, e.g. 5, the the 2*10 is achieved as 5+15 pixel wide (high) parts
  // We always filter a larger area, but only copy the 2*radius (r) pixels from it.
  // a radius plus the filter support size overrides
  int filtering_width = max(MIN_FILTERING_EXTENT, r + (int)std::ceil(filter->support()));

  // adjust the filtering width for the worst case, considering the chroma subsampling
  filtering_width = (filtering_width + (1 << shift) - 1) & ~((1 << shift) - 1);
  // active area before adding the new borders
  const int orig_width = vi.width - left - right;
  const int orig_height = vi.height - top - bottom;

  // End of safety adjustments
  // Radius "r" and "filtering_width" are both +/- values and are chroma and resizer friendly.
  // left, right, top, bottom are already chroma subsample friendly values

  /*
  Create up to 8 sections.
  
  left, top, right, bottom and the four corner bars mark the convolution (blur) area.
  - left and right needs only force horizontal resizer
  - top and bottom bars force vertical resizer
  - corners force both H and V resizer
  +---+-------------------+
  | oo|ooooooooooooooo|oo |
  +-o-+-----------------o-+
  | o |               | o |
  | o |               | o |
  | o |               | o |
  | o |               | o |
  | o |               | o |
  | o |               | o |
  +-o-+---------------+-o-+
  | oo|ooooooooooooooo|oo |
  +---+---------------+---+
  */

  typedef struct {
    PClip clip; // a smart pointer
    int extent_outer_horiz, extent_inner_horiz; // may not be symmetric
    int extent_outer_vert, extent_inner_vert;
    int crop_x, crop_y;
    int target_x, target_y;
    int target_width, target_height;
    int src_x, src_y;
    int src_width, src_height;
    int force;
    //  0 - return unchanged if no resize needed
    //  1 - force H - Horizontal resizing phase, For vertical bars
    //  2 - force V - Vertical resizing phase, For horizontal bars
    //  3 - force H and V for corner rectangles
  } BarSection;

  std::vector<BarSection> bars;

  /* Example on LEFT and RIGHT vertical bar
  
    extent_outer extent_inner                extent_inner extent_outer
        <------>|<--------->                  <------>|<--------->

              r*2                                   r*2 
             |<>|<>|           orig_width          |<>|<>|
                 <------------------------------------>
        +----+--|--+-------+  ^               +----+--|--+-------+
        |    |  |  |       |  |               |    |  |  |       |
        |    |  |  |       |  |orig_height    |    |  |  |       |
        |    |  |  |       |  |               |    |  |  |       |
        |    |  |  |       |  |               |    |  |  |       |
        |    |  |  |       |  |               |    |  |  |       |
        +----+--|--+-------+  V               +----+--|--+-------+

  The extent_inner+extent_outer area is cropped and blurred (filtered) with the given algoritm, e.g "gauss".
  From this blurred Clip only the (2*r) * orig_height rectangle is copied back onto the transient area.
  */

  // Define bar sections data for iteration

  struct BarSectionTemplate {
    int side;           // 0=left, 1=right, 2=top, 3=bottom 4=top-left 5=top-right 6=bottom-left 7=bottom-right
    int direction;      // 1=horizontally, 2=vertically filtere, 3=both (corners)
  };

  const BarSectionTemplate templates[] = {
    {0, 1}, // left bar (horizontally filtered)
    {1, 1}, // right bar (horizontally filtered)
    {2, 2}, // top bar (vertically filtered)
    {3, 2}, // bottom bar (vertically filtered)
    {4, 3}, // top left rectangle (H-V filtered)
    {5, 3}, // top right rectangle (H-V filtered)
    {6, 3}, // bottom left rectangle (H-V filtered)
    {7, 3}  // bottom right rectangle (H-V filtered)
  };

  // First cycle: initialize common parameters and calculate dimensions
  for (int i = 0; i < 8; i++) {
    BarSection bar = {};

    const bool isHorizontallyFiltered = (templates[i].direction == 1) || (templates[i].direction == 3);
    const bool isVerticallyFiltered = (templates[i].direction == 2) || (templates[i].direction == 3);
    const int side = templates[i].side;

    // Calculate border to use based on side
    int borderSize, borderSize2;
    switch (side) {
    case 0: borderSize = left; borderSize2 = 0;  break;
    case 1: borderSize = right; borderSize2 = 0; break;
    case 2: borderSize = 0; borderSize2 = top; break;
    case 3: borderSize = 0; borderSize2 = bottom; break;
    case 4: borderSize = left; borderSize2 = top; break;
    case 5: borderSize = right; borderSize2 = top; break;
    case 6: borderSize = left;  borderSize2 = bottom; break;
    case 7: borderSize = right; borderSize2 = bottom; break;
    }

    int safe_flt_rad = r;
    if (borderSize > 0) { // same as isHorizontallyFiltered
      bar.extent_outer_horiz = std::min(filtering_width, borderSize);
      bar.extent_inner_horiz = 2 * filtering_width - bar.extent_outer_horiz;
    }
    if (borderSize2 > 0) { // same as(isVerticallyFiltered
      bar.extent_outer_vert = std::min(filtering_width, borderSize2);
      bar.extent_inner_vert = 2 * filtering_width - bar.extent_outer_vert;
    }

    if (isHorizontallyFiltered) {
      safe_flt_rad = std::min(safe_flt_rad, borderSize);
      bar.extent_inner_horiz = std::min(bar.extent_inner_horiz, orig_width);
    }
    if (isVerticallyFiltered) {
      safe_flt_rad = std::min(safe_flt_rad, borderSize2);
      bar.extent_inner_vert = std::min(bar.extent_inner_vert, orig_height);
    }

    int safe_flt_rad_inner = both_side ? safe_flt_rad : 0;

    int safe_flt_rad_inner_horiz = std::min(bar.extent_inner_horiz, safe_flt_rad_inner);
    int safe_flt_rad_inner_vert = std::min(bar.extent_inner_vert, safe_flt_rad_inner);

    // Set force direction
    bar.force = templates[i].direction;

    // Set dimensions based on direction and side
    if (isHorizontallyFiltered && isVerticallyFiltered) { // corners
      bar.target_width = bar.extent_outer_horiz + bar.extent_inner_horiz;
      bar.target_height = bar.extent_outer_vert + bar.extent_inner_vert;
      bar.src_width = safe_flt_rad + safe_flt_rad_inner_horiz;
      bar.src_height = safe_flt_rad + safe_flt_rad_inner_vert;

      if (side == 4 || side == 6) {  // Top Left Bottom Left
        bar.crop_x = left - bar.extent_outer_horiz;
        bar.target_x = left - safe_flt_rad;
        bar.src_x = bar.extent_outer_horiz - safe_flt_rad;
      }
      else {  // Top Right Bottom Right
        bar.crop_x = left + orig_width - bar.extent_inner_horiz;
        bar.target_x = left + orig_width - safe_flt_rad_inner_horiz;
        bar.src_x = bar.extent_inner_horiz - safe_flt_rad_inner_horiz;
      }

      if (side == 4 || side == 5) {  // Top Left Top Right
        bar.crop_y = top - bar.extent_outer_vert;
        bar.target_y = top - safe_flt_rad;
        bar.src_y = bar.extent_outer_vert - safe_flt_rad;
      }
      else {  // Bottom Left Bottom Right
        bar.crop_y = top + orig_height - bar.extent_inner_vert;
        bar.target_y = top + orig_height - safe_flt_rad_inner_vert;
        bar.src_y = bar.extent_inner_vert - safe_flt_rad_inner_vert;
      }
    }
    else if (isHorizontallyFiltered) {
      // Vertical but horizontally filtered bars on the left/right side
      bar.target_width = bar.extent_outer_horiz + bar.extent_inner_horiz;
      bar.target_height = orig_height;
      bar.src_width = safe_flt_rad + safe_flt_rad_inner_horiz;
      bar.src_height = orig_height;

      if (side == 0) {  // Left
        bar.crop_x = left - bar.extent_outer_horiz;;
        bar.crop_y = top;
        bar.target_x = left - safe_flt_rad;
        bar.target_y = top;
        bar.src_x = bar.extent_outer_horiz - safe_flt_rad;

      }
      else {  // Right
        bar.crop_x = left + orig_width - bar.extent_inner_horiz;
        bar.crop_y = top;
        bar.target_x = left + orig_width - safe_flt_rad_inner_horiz;
        bar.target_y = top;
        bar.src_x = bar.extent_inner_horiz - safe_flt_rad_inner_horiz;
      }
     
      bar.src_y = 0;
    }
    else {
      // Horizontal, but vertically filtered bars (top/bottom)
      bar.target_width = orig_width;
      bar.target_height = bar.extent_outer_vert + bar.extent_inner_vert;
      bar.src_width = orig_width;
      bar.src_height = safe_flt_rad + safe_flt_rad_inner_vert;

      if (side == 2) {  // Top
        bar.crop_x = left;
        bar.crop_y = top - bar.extent_outer_vert;
        bar.target_x = left;
        bar.target_y = top - safe_flt_rad;
        bar.src_y = bar.extent_outer_vert - safe_flt_rad;
      }
      else {  // Bottom
        bar.crop_x = left;
        bar.crop_y = top + orig_height - bar.extent_inner_vert;
        bar.target_x = left;
        bar.target_y = top + orig_height - safe_flt_rad_inner_vert;
        bar.src_y = bar.extent_inner_vert - safe_flt_rad_inner_vert;
      }

      bar.src_x = 0;
    }

    // avoid "Resize: Source image too small for this resize method. Width=%d, Support=%d", source_size, int(ceil(filter_support))"
    // int source_size, double crop_size, int target_size
    // all are the same here
    // whether H or V resizing is forced, only check for that dimension
    bool ok = true;
    if (isHorizontallyFiltered && bar.target_width <= 0) ok = false;
    if (isVerticallyFiltered && bar.target_height <= 0) ok = false;
    // size is indifferent since avs 3.7.4, resizers are robust, accept anything, 
    // no other size constraint check is needed.

    if(ok)
      bars.push_back(bar);
  }

  std::vector<PClip> child_array = { child };
  std::vector<int> position_array = { };

  const bool preserve_center = true;
  const char* placement_name = nullptr; 
  // along with forced_chroma_placement, does not read frame props again in eight child resizers
  // _ChromaLocation - if any - was read in AddBorders - once - and passed here

  for (auto& bar : bars) {
    // or use [src_left]f[src_top]f[src_width]f[src_height]f
    // resizer parameters set for convolution (unchanged dimensions) filter
    /* When we directly pass source position, it's not possible to govern the directional "force", anyway, it would just call crop as well.
    AVSValue args_left_top_w_h[4] = {bar.crop_x, bar.crop_y, bar.target_width, bar.target_height}; // left, top, width (auto), height (auto)
    bar.clip = FilteredResize::CreateResize(child, bar.target_width, bar.target_height, args_left_top_w_h, bar.force, filter, env);
    */
    AVSValue args_left_top_w_h[4] = { 0, 0, AVSValue(), AVSValue() }; // left, top, width (auto), height (auto)

    bar.clip = FilteredResize::CreateResize(
      new Crop(bar.crop_x, bar.crop_y, bar.target_width, bar.target_height, 0, child, env),
      bar.target_width, bar.target_height, args_left_top_w_h, bar.force, filter, 
      preserve_center, placement_name, forced_chroma_placement,
      env);


    // Add to vector
    child_array.push_back(bar.clip);
    position_array.push_back(bar.target_x);
    position_array.push_back(bar.target_y);
    // we can copy only a smaller part from the filtered area
    // After blurring e.g. a 10x10 rectangle, we'd copy only from a radius of +/-1 or +/-2
    position_array.push_back(bar.src_x);
    position_array.push_back(bar.src_y);
    position_array.push_back(bar.src_width);
    position_array.push_back(bar.src_height);
  }

  PClip clip = new MultiOverlay(child_array, position_array, env);

  delete filter;

  return clip;
}


AVSValue __cdecl AddBorders::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  // [0][1][2][3][4]  [5]       [6]        [7]       [8]       [9]      [10]  [11]
  //  c  i  i  i  i [color]i[color_yuv]i[resample]s[param1]f[param2]f[param3]f[r]i
  // int left, int top, int right, int bottom

  // similar to BlankClip
  int color = args[5].AsInt(0);
  bool color_as_yuv = false;

  const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();

  if (args[6].Defined()) {
    if (color != 0) // Not quite 100% test
      env->ThrowError("AddBorders: color and color_yuv are mutually exclusive");

    if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("AddBorders: color_yuv only valid for YUV color spaces");
    color = args[6].AsInt(); // override
    color_as_yuv = true;
  }

  const int left = max(0, args[1].AsInt());
  const int top = max(0, args[2].AsInt());
  const int right = max(0, args[3].AsInt());
  const int bottom = max(0, args[4].AsInt());

  int forced_chroma_placement = ChromaLocation_e::AVS_CHROMA_UNUSED;
  if (vi.Is411() || vi.Is420() || vi.Is422() || vi.Is440() || vi.Is410()) {
     auto frame0 = args[0].AsClip()->GetFrame(0, env);
    const AVSMap* props = env->getFramePropsRO(frame0);
    if (props) {
      if (env->propNumElements(props, "_ChromaLocation") > 0) {
        forced_chroma_placement = (int)env->propGetIntSaturated(props, "_ChromaLocation", 0, nullptr);
      }
    }
  }

  // here we have read _ChromaLocation, in order to pass it directly to the many blurring resizers to prevent them
  // to read up to 8x GetFrame(0) in their init for getting the same frame properties like we had.

  PClip clip = new AddBorders( left, top, right, bottom, color, color_as_yuv, args[0].AsClip(), env);

  // args[7], args[8], args[9], args[10], args[11], // [resample]s[param1]f[param2]f[param3]f[r]i

  clip = AddBorderPostProcess(clip, left, top, right, bottom, args[7], args[8], args[9], args[10], args[11], forced_chroma_placement, env);

  return clip;
}


/**********************************
 *******   Factory Methods   ******
 *********************************/


AVSValue __cdecl Create_Letterbox(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  int top = args[1].AsInt();
  int bottom = args[2].AsInt();
  int left = args[3].AsInt(0);
  int right = args[4].AsInt(0);
  int color = args[5].AsInt(0);
  const VideoInfo& vi = clip->GetVideoInfo();

  // [0][1][2][3]   [4]   [5]        [6]          [7]       [8]       [9]      [10]  [11]
  //  c  i  i [x1]i [x2]i [color]i [color_yuv]i[resample]s[param1]f[param2]f[param3]f[r]i
  //   top, bottom, [left], [right] [,color] [,color_yuv]

  // similar to BlankClip/AddBorders
  bool color_as_yuv = false;
  if (args[6].Defined()) {
    if (color != 0) // Not quite 100% test
      env->ThrowError("LetterBox: color and color_yuv are mutually exclusive");

    if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("LetterBox: color_yuv only valid for YUV color spaces");
    color = args[6].AsInt(); // override
    color_as_yuv = true;
  }

  if ( (top<0) || (bottom<0) || (left<0) || (right<0) )
    env->ThrowError("LetterBox: You cannot specify letterboxing less than 0.");
  if (top+bottom>=vi.height) // Must be >= otherwise it is interpreted wrong by crop()
    env->ThrowError("LetterBox: You cannot specify letterboxing that is bigger than the picture (height).");
  if (right+left>=vi.width) // Must be >= otherwise it is interpreted wrong by crop()
    env->ThrowError("LetterBox: You cannot specify letterboxing that is bigger than the picture (width).");

  if (vi.IsYUV() || vi.IsYUVA()) {
    int xsub = 0;
    int ysub = 0;

    if (vi.NumComponents() > 1 && !vi.IsYA()) { // Y+A has no chroma
      xsub=vi.GetPlaneWidthSubsampling(PLANAR_U);
      ysub=vi.GetPlaneHeightSubsampling(PLANAR_U);
    }
    const int xmask = (1 << xsub) - 1;
    const int ymask = (1 << ysub) - 1;

    // YUY2, etc, ... can only operate to even pixel boundaries
    if (left  & xmask)
      env->ThrowError("LetterBox: YUV images width must be divideable by %d (left side).", xmask+1);
    if (right & xmask)
      env->ThrowError("LetterBox: YUV images width must be divideable by %d (right side).", xmask+1);

    if (top & ymask)
      env->ThrowError("LetterBox: YUV images height must be divideable by %d (top).", ymask+1);
    if (bottom & ymask)
      env->ThrowError("LetterBox: YUV images height must be divideable by %d (bottom).", ymask+1);
  }

  left = max(0, left);
  top = max(0, top);
  right = max(0, right);
  bottom = max(0, bottom);

  int forced_chroma_placement = ChromaLocation_e::AVS_CHROMA_UNUSED;
  if (vi.Is411() || vi.Is420() || vi.Is422() || vi.Is440() || vi.Is410()) {
    auto frame0 = clip->GetFrame(0, env);
    const AVSMap* props = env->getFramePropsRO(frame0);
    if (props) {
      if (env->propNumElements(props, "_ChromaLocation") > 0) {
        forced_chroma_placement = (int)env->propGetIntSaturated(props, "_ChromaLocation", 0, nullptr);
      }
    }
  }

  clip = new AddBorders(left, top, right, bottom, color, color_as_yuv, new Crop(left, top, vi.width-left-right, vi.height-top-bottom, 0, clip, env), env);

  // args[7], args[8], args[9], args[10], args[11], // [resample]s[param1]f[param2]f[param3]f[r]i

  clip = AddBorderPostProcess(clip, left, top, right, bottom, args[7], args[8], args[9], args[10], args[11], forced_chroma_placement, env);

  return clip;

}


AVSValue __cdecl Create_CropBottom(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();
  return new Crop(0, 0, vi.width, vi.height - args[1].AsInt(), 0, clip, env);
}
