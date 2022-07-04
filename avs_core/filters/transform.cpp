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

#include "transform.h"
#include "../convert/convert_matrix.h"
#include <avs/minmax.h>
#include "../core/bitblt.h"
#include <stdint.h>



/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Transform_filters[] = {
  { "FlipVertical",   BUILTIN_FUNC_PREFIX, "c", FlipVertical::Create },
  { "FlipHorizontal", BUILTIN_FUNC_PREFIX, "c", FlipHorizontal::Create },
  { "Crop",           BUILTIN_FUNC_PREFIX, "ciiii[align]b", Crop::Create },              // left, top, width, height *OR*
                                                  //  left, top, -right, -bottom (VDub style)
  { "CropBottom", BUILTIN_FUNC_PREFIX, "ci", Create_CropBottom },      // bottom amount
  { "AddBorders", BUILTIN_FUNC_PREFIX, "ciiii[color]i[color_yuv]i", AddBorders::Create },  // left, top, right, bottom [,color] [,color_yuv]
  { "Letterbox",  BUILTIN_FUNC_PREFIX, "cii[x1]i[x2]i[color]i[color_yuv]i", Create_Letterbox },       // top, bottom, [left], [right] [,color] [,color_yuv]
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
    if (vi.NumComponents() > 1) {
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
  InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);
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
  if (0 != _align && (env->GetDeviceType() == DEV_TYPE_CPU)) {
    PVideoFrame dst = static_cast<IScriptEnvironment*>(env)->NewVideoFrameP(vi, &frame, (int)align+1);

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
  if (!frame->GetPitch(plane1))
    return env->Subframe(frame, top * frame->GetPitch() + left_bytes, frame->GetPitch(), vi.RowSize(), vi.height);
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
 : GenericVideoFilter(_child), left(max(0,_left)), top(max(0,_top)), right(max(0,_right)), bot(max(0,_bot)), clr(_clr), xsub(0), ysub(0), force_color_as_yuv(_force_color_as_yuv)
{
  if (vi.IsYUV() || vi.IsYUVA()) {
    if (vi.NumComponents() > 1) {
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
      return (pixel_t)c8tof(color); // float, scale to [0..1]
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

  int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *planes = isYUV ? planesYUV : planesRGB;
  uint8_t colorsYUV[4] = { YBlack, UBlack, VBlack, ABlack };
  uint8_t colorsRGB[4] = { UBlack, VBlack, YBlack, ABlack }; // mapping for planar RGB
  uint8_t *colors = isYUV ? colorsYUV : colorsRGB;
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

    pixel_t current_color = GetHbdColorFromByte<pixel_t>(colors[p], !isYUV, bits_per_pixel, chroma);

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



AVSValue __cdecl AddBorders::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  // [0][1][2][3][4]  [5]       [6]
  //  c  i  i  i  i [color]i[color_yuv]i

  // similar to BlankClip
  int color = args[5].AsInt(0);
  bool color_as_yuv = false;
  if (args[6].Defined()) {
    if (color != 0) // Not quite 100% test
      env->ThrowError("AddBorders: color and color_yuv are mutually exclusive");

    const VideoInfo &vi = args[0].AsClip()->GetVideoInfo();

    if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("AddBorders: color_yuv only valid for YUV color spaces");
    color = args[6].AsInt(); // override
    color_as_yuv = true;
  }

  return new AddBorders( args[1].AsInt(), args[2].AsInt(), args[3].AsInt(),
                         args[4].AsInt(), color, color_as_yuv, args[0].AsClip(), env);
}





/******************************
 *******   Fill Border   ******
 *****************************/


 /*  This function fills up the right side of the picture on planar images with duplicates of the rightmost pixel
  *   TODO: Implement fast ISSE routines
  */

// PF dead function?
FillBorder::FillBorder(PClip _clip) : GenericVideoFilter(_clip) {
}

PVideoFrame __stdcall FillBorder::GetFrame(int n, IScriptEnvironment* env) {

  PVideoFrame src = child->GetFrame(n, env);
  if (src->GetRowSize(PLANAR_Y)==src->GetRowSize(PLANAR_Y_ALIGNED)) return src;  // No need to fill extra pixels

  // !! when GetWritePtr would be size_t, revise this. Be cautious when you subtract two unsigned size_t variables
  unsigned char* Ydata = src->GetWritePtr(PLANAR_U) - (src->GetOffset(PLANAR_U) - src->GetOffset(PLANAR_Y)); // Nasty hack, to avoid "MakeWritable" - never, EVER do this at home!
  unsigned char* Udata = src->GetWritePtr(PLANAR_U);
  unsigned char* Vdata = src->GetWritePtr(PLANAR_V);

  int fillp=src->GetRowSize(PLANAR_Y_ALIGNED) - src->GetRowSize(PLANAR_Y);
  int h=src->GetHeight(PLANAR_Y);

  Ydata = &Ydata[src->GetRowSize(PLANAR_Y)-1];
  for (int y=0; y<h; y++) {
    for (int x=1; x<=fillp; x++) {
      Ydata[x]=Ydata[0];
    }
    Ydata+=src->GetPitch(PLANAR_Y);
  }

  fillp=src->GetRowSize(PLANAR_U_ALIGNED) - src->GetRowSize(PLANAR_U);
  Udata = &Udata[src->GetRowSize(PLANAR_U)-1];
  Vdata = &Vdata[src->GetRowSize(PLANAR_V)-1];
  h=src->GetHeight(PLANAR_U);

  for (int y=0; y<h; y++) {
    for (int x=1; x<=fillp; x++) {
      Udata[x]=Udata[0];
      Vdata[x]=Vdata[0];
    }
    Udata+=src->GetPitch(PLANAR_U);
    Vdata+=src->GetPitch(PLANAR_V);
  }
  return src;
}


PClip FillBorder::Create(PClip clip)
{
  if (!clip->GetVideoInfo().IsPlanar()) {  // If not planar, already ok.
    return clip;
  }
  else
    return new FillBorder(clip);
}





/**********************************
 *******   Factory Methods   ******
 *********************************/


AVSValue __cdecl Create_Letterbox(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  int top = args[1].AsInt();
  int bot = args[2].AsInt();
  int left = args[3].AsInt(0);
  int right = args[4].AsInt(0);
  int color = args[5].AsInt(0);
  const VideoInfo& vi = clip->GetVideoInfo();

  // [0][1][2][3]   [4]   [5]        [6]
  //  c  i  i [x1]i [x2]i [color]i [color_yuv]i"  // top, bottom, [left], [right] [,color] [,color_yuv]

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

  if ( (top<0) || (bot<0) || (left<0) || (right<0) )
    env->ThrowError("LetterBox: You cannot specify letterboxing less than 0.");
  if (top+bot>=vi.height) // Must be >= otherwise it is interpreted wrong by crop()
    env->ThrowError("LetterBox: You cannot specify letterboxing that is bigger than the picture (height).");
  if (right+left>=vi.width) // Must be >= otherwise it is interpreted wrong by crop()
    env->ThrowError("LetterBox: You cannot specify letterboxing that is bigger than the picture (width).");

  if (vi.IsYUV() || vi.IsYUVA()) {
    int xsub = 0;
    int ysub = 0;

    if (vi.NumComponents() > 1) {
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

    if (top   & ymask)
      env->ThrowError("LetterBox: YUV images height must be divideable by %d (top).", ymask+1);
    if (bot   & ymask)
      env->ThrowError("LetterBox: YUV images height must be divideable by %d (bottom).", ymask+1);
  }
  return new AddBorders(left, top, right, bot, color, color_as_yuv, new Crop(left, top, vi.width-left-right, vi.height-top-bot, 0, clip, env), env);
}


AVSValue __cdecl Create_CropBottom(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();
  return new Crop(0, 0, vi.width, vi.height - args[1].AsInt(), 0, clip, env);
}
