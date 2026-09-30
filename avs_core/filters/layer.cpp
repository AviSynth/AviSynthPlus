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



// Avisynth filter: Layer
// by "poptones" (poptones@myrealbox.com)

#include <avisynth.h>
#include "layer.h"
#include "merge.h"
#ifdef INTEL_INTRINSICS
#include "intel/layer_sse.h"
#include "intel/layer_avx2.h"
#include "intel/layer_sse41.h"
#endif
#ifdef NEON_INTRINSICS
#include "overlay/aarch64/layer_neon.h"
#endif
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif

#include <avs/alignment.h>
#include "../core/internal.h"
#include "../convert/convert_planar.h"
#include <algorithm>
#include <vector>

// `placement`: chroma location of the (base) clip, same syntax, precedence and default as
// ConvertToYUV4xx's ChromaInPlacement (explicit -> _ChromaLocation -> format default).
// Not that unlike Overlay, Layer does not convert: both clips have the same format and are
// assumed to share the siting.
// Only the mask downsampling kernel depends on it, mapped to the nearest implemented
// variant (see chromaLocationToMaskPlacement in blend_common.h).
static int getPlacement(const AVSValue& _placement, PClip clip, IScriptEnvironment* env) {
  const int chromaloc = ResolveChromaLocation(clip, _placement.AsString(nullptr), env);
  return chromaLocationToMaskPlacement(chromaloc, clip->GetVideoInfo());
}

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Layer_filters[] = {
  { "Mask",         BUILTIN_FUNC_PREFIX, "cc", Mask::Create },     // clip, mask
  { "ColorKeyMask", BUILTIN_FUNC_PREFIX, "ci[]i[]i[]i", ColorKeyMask::Create },    // clip, color, tolerance[B, toleranceG, toleranceR]
  { "ResetMask",    BUILTIN_FUNC_PREFIX, "c[mask]f[opacity]f", ResetMask::Create },
  { "Invert",       BUILTIN_FUNC_PREFIX, "c[channels]s", Invert::Create },
  { "ShowAlpha",    BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)3 }, // AVS+ also for YUVA, PRGBA
  { "ShowRed",      BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)2 },
  { "ShowGreen",    BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)1 },
  { "ShowBlue",     BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)0 },
  { "ShowY",        BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)4 }, // AVS+
  { "ShowU",        BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)5 }, // AVS+
  { "ShowV",        BUILTIN_FUNC_PREFIX, "c[pixel_type]s", ShowChannel::Create, (void*)6 }, // AVS+
  { "MergeRGB",     BUILTIN_FUNC_PREFIX, "ccc[pixel_type]s", MergeRGB::Create, (void*)0 },
  { "MergeARGB",    BUILTIN_FUNC_PREFIX, "cccc[pixel_type]s", MergeRGB::Create, (void*)1 },
  { "Layer",        BUILTIN_FUNC_PREFIX, "cc[op]s[level]i[x]i[y]i[threshold]i[use_chroma]b[opacity]f[placement]s", Layer::Create },
  /**
    * Layer(clip, overlayclip, operation, amount, xpos, ypos, [threshold=0], [use_chroma=true])
   **/
  { "Subtract", BUILTIN_FUNC_PREFIX, "cc", Subtract::Create },
  { NULL }
};


/******************************
 *******   Mask Filter   ******
 ******************************/

Mask::Mask(PClip _child1, PClip _child2, IScriptEnvironment* env)
  : child1(_child1), child2(_child2)
{
  const VideoInfo& vi1 = child1->GetVideoInfo();
  const VideoInfo& vi2 = child2->GetVideoInfo();
  if (vi1.width != vi2.width || vi1.height != vi2.height)
    env->ThrowError("Mask error: image dimensions don't match");
  if (!((vi1.IsRGB32() && vi2.IsRGB32()) ||
    (vi1.IsRGB64() && vi2.IsRGB64()) ||
    (vi1.IsPlanarRGBA() && vi2.IsPlanarRGBA()))
    )
    env->ThrowError("Mask error: sources must be RGB32, RGB64 or Planar RGBA");

  if (vi1.BitsPerComponent() != vi2.BitsPerComponent())
    env->ThrowError("Mask error: Components are not of the same bit depths");

  vi = vi1;

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  mask_frames = vi2.num_frames;
}


template<typename pixel_t>
static void mask_c(BYTE* srcp8, const BYTE* alphap8, int src_pitch, int alpha_pitch, size_t width, size_t height) {
  pixel_t* srcp = reinterpret_cast<pixel_t*>(srcp8);
  const pixel_t* alphap = reinterpret_cast<const pixel_t*>(alphap8);

  src_pitch /= sizeof(pixel_t);
  alpha_pitch /= sizeof(pixel_t);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      srcp[x * 4 + 3] = (cyb * alphap[x * 4 + 0] + cyg * alphap[x * 4 + 1] + cyr * alphap[x * 4 + 2] + 16384) >> 15;
    }
    srcp += src_pitch;
    alphap += alpha_pitch;
  }
}

template<typename pixel_t>
static void mask_planar_rgb_c(BYTE* dstp8, const BYTE* srcp_r8, const BYTE* srcp_g8, const BYTE* srcp_b8, int dst_pitch, int src_pitch, size_t width, size_t height, int bits_per_pixel) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* srcp_r = reinterpret_cast<const pixel_t*>(srcp_r8);
  const pixel_t* srcp_g = reinterpret_cast<const pixel_t*>(srcp_g8);
  const pixel_t* srcp_b = reinterpret_cast<const pixel_t*>(srcp_b8);
  src_pitch /= sizeof(pixel_t);
  dst_pitch /= sizeof(pixel_t);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      dstp[x] = ((cyb * srcp_b[x] + cyg * srcp_g[x] + cyr * srcp_r[x] + 16384) >> 15);
    }
    dstp += dst_pitch;
    srcp_r += src_pitch;
    srcp_g += src_pitch;
    srcp_b += src_pitch;
  }
}

static void mask_planar_rgb_float_c(BYTE* dstp8, const BYTE* srcp_r8, const BYTE* srcp_g8, const BYTE* srcp_b8, int dst_pitch, int src_pitch, size_t width, size_t height) {

  float* dstp = reinterpret_cast<float*>(dstp8);
  const float* srcp_r = reinterpret_cast<const float*>(srcp_r8);
  const float* srcp_g = reinterpret_cast<const float*>(srcp_g8);
  const float* srcp_b = reinterpret_cast<const float*>(srcp_b8);
  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      dstp[x] = cyb_f * srcp_b[x] + cyg_f * srcp_g[x] + cyr_f * srcp_r[x];
    }
    dstp += dst_pitch;
    srcp_r += src_pitch;
    srcp_g += src_pitch;
    srcp_b += src_pitch;
  }
}

PVideoFrame __stdcall Mask::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src1 = child1->GetFrame(n, env);
  PVideoFrame src2 = child2->GetFrame(std::min(n, mask_frames - 1), env);

  env->MakeWritable(&src1);

  if (vi.IsPlanar()) {
    // planar RGB
    BYTE* dstp = src1->GetWritePtr(PLANAR_A); // destination Alpha plane

    const BYTE* srcp_g = src2->GetReadPtr(PLANAR_G);
    const BYTE* srcp_b = src2->GetReadPtr(PLANAR_B);
    const BYTE* srcp_r = src2->GetReadPtr(PLANAR_R);

    const int dst_pitch = src1->GetPitch();
    const int src_pitch = src2->GetPitch();

    // clip1_alpha = greyscale(clip2)
    if (pixelsize == 1)
      mask_planar_rgb_c<uint8_t>(dstp, srcp_r, srcp_g, srcp_b, dst_pitch, src_pitch, vi.width, vi.height, bits_per_pixel);
    else if (pixelsize == 2)
      mask_planar_rgb_c<uint16_t>(dstp, srcp_r, srcp_g, srcp_b, dst_pitch, src_pitch, vi.width, vi.height, bits_per_pixel);
    else
      mask_planar_rgb_float_c(dstp, srcp_r, srcp_g, srcp_b, dst_pitch, src_pitch, vi.width, vi.height);
  }
  else {
    // Packed RGB32/64
    BYTE* src1p = src1->GetWritePtr();
    const BYTE* src2p = src2->GetReadPtr();

    const int src1_pitch = src1->GetPitch();
    const int src2_pitch = src2->GetPitch();

    // clip1_alpha = greyscale(clip2)
#ifdef INTEL_INTRINSICS
    // SIMD versions need at least 16 bytes (4 RGB32 pixels) per row.
    // Reason: unaligned row-tail step processes the last 16 bytes,
    if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2) && vi.width >= 4)
    {
      mask_avx2(src1p, src2p, src1_pitch, src2_pitch, vi.width, vi.height);
    }
    else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && vi.width >= 4)
    {
      mask_sse2(src1p, src2p, src1_pitch, src2_pitch, vi.width, vi.height);
    }
    else
#endif
      {
        if (pixelsize == 1) { // RGB32
          mask_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, vi.width, vi.height);
        }
        else { // if (pixelsize == 2) RGB64
          mask_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, vi.width, vi.height);
        }
      }
  }

  return src1;
}

AVSValue __cdecl Mask::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Mask(args[0].AsClip(), args[1].AsClip(), env);
}


/**************************************
 *******   ColorKeyMask Filter   ******
 **************************************/


ColorKeyMask::ColorKeyMask(PClip _child, int _color, int _tolB, int _tolG, int _tolR, IScriptEnvironment* env)
  : GenericVideoFilter(_child), color(_color & 0xffffff), tolB(_tolB & 0xff), tolG(_tolG & 0xff), tolR(_tolR & 0xff)
{
  if (!vi.IsRGB32() && !vi.IsRGB64() && !vi.IsPlanarRGBA())
    env->ThrowError("ColorKeyMask: requires RGB32, RGB64 or Planar RGBA input");
  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  max_pixel_value = (1 << bits_per_pixel) - 1;

  auto rgbcolor8to16 = [](uint8_t color8, int max_pixel_value) { return (uint16_t)(color8 * max_pixel_value / 255); };

  uint64_t r = rgbcolor8to16((color >> 16) & 0xFF, max_pixel_value);
  uint64_t g = rgbcolor8to16((color >> 8) & 0xFF, max_pixel_value);
  uint64_t b = rgbcolor8to16((color) & 0xFF, max_pixel_value);
  uint64_t a = rgbcolor8to16((color >> 24) & 0xFF, max_pixel_value);
  color64 = (a << 48) + (r << 32) + (g << 16) + (b);
  tolR16 = rgbcolor8to16(tolR & 0xFF, max_pixel_value); // scale tolerance
  tolG16 = rgbcolor8to16(tolG & 0xFF, max_pixel_value);
  tolB16 = rgbcolor8to16(tolB & 0xFF, max_pixel_value);
}


template<typename pixel_t>
static void colorkeymask_c(BYTE* pf8, int pitch, int R, int G, int B, int height, int rowsize, int tolB, int tolG, int tolR) {
  pixel_t* pf = reinterpret_cast<pixel_t*>(pf8);
  rowsize /= sizeof(pixel_t);
  pitch /= sizeof(pixel_t);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < rowsize; x += 4) {
      if (IsClose(pf[x], B, tolB) && IsClose(pf[x + 1], G, tolG) && IsClose(pf[x + 2], R, tolR))
        pf[x + 3] = 0;
    }
    pf += pitch;
  }
}

template<typename pixel_t>
static void colorkeymask_planar_c(const BYTE* pfR8, const BYTE* pfG8, const BYTE* pfB8, BYTE* pfA8, int pitch, int R, int G, int B, int height, int width, int tolB, int tolG, int tolR) {
  const pixel_t* pfR = reinterpret_cast<const pixel_t*>(pfR8);
  const pixel_t* pfG = reinterpret_cast<const pixel_t*>(pfG8);
  const pixel_t* pfB = reinterpret_cast<const pixel_t*>(pfB8);
  pixel_t* pfA = reinterpret_cast<pixel_t*>(pfA8);
  pitch /= sizeof(pixel_t);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (IsClose(pfB[x], B, tolB) && IsClose(pfG[x], G, tolG) && IsClose(pfR[x], R, tolR))
        pfA[x] = 0;
    }
    pfR += pitch;
    pfG += pitch;
    pfB += pitch;
    pfA += pitch;
  }
}

static void colorkeymask_planar_float_c(const BYTE* pfR8, const BYTE* pfG8, const BYTE* pfB8, BYTE* pfA8, int pitch, float R, float G, float B, int height, int width, float tolB, float tolG, float tolR) {
  typedef float pixel_t;
  const pixel_t* pfR = reinterpret_cast<const pixel_t*>(pfR8);
  const pixel_t* pfG = reinterpret_cast<const pixel_t*>(pfG8);
  const pixel_t* pfB = reinterpret_cast<const pixel_t*>(pfB8);
  pixel_t* pfA = reinterpret_cast<pixel_t*>(pfA8);
  pitch /= sizeof(pixel_t);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (IsCloseFloat(pfB[x], B, tolB) && IsCloseFloat(pfG[x], G, tolG) && IsCloseFloat(pfR[x], R, tolR))
        pfA[x] = 0;
    }
    pfR += pitch;
    pfG += pitch;
    pfB += pitch;
    pfA += pitch;
  }
}


PVideoFrame __stdcall ColorKeyMask::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  env->MakeWritable(&frame);

  BYTE* pf = frame->GetWritePtr();
  const int pitch = frame->GetPitch();
  const int rowsize = frame->GetRowSize();

  if (vi.IsPlanarRGBA()) {
    const BYTE* pf_g = frame->GetReadPtr(PLANAR_G);
    const BYTE* pf_b = frame->GetReadPtr(PLANAR_B);
    const BYTE* pf_r = frame->GetReadPtr(PLANAR_R);
    BYTE* pf_a = frame->GetWritePtr(PLANAR_A);

    const int pitch = frame->GetPitch();
    const int width = vi.width;

    if (pixelsize == 1) {
      const int R = (color >> 16) & 0xff;
      const int G = (color >> 8) & 0xff;
      const int B = color & 0xff;
      colorkeymask_planar_c<uint8_t>(pf_r, pf_g, pf_b, pf_a, pitch, R, G, B, vi.height, width, tolB, tolG, tolR);
    }
    else if (pixelsize == 2) {
      const int R = (color64 >> 32) & 0xffff;
      const int G = (color64 >> 16) & 0xffff;
      const int B = color64 & 0xffff;
      colorkeymask_planar_c<uint16_t>(pf_r, pf_g, pf_b, pf_a, pitch, R, G, B, vi.height, width, tolB16, tolG16, tolR16);
    }
    else { // float
      const float R = ((color >> 16) & 0xff) / 255.0f;
      const float G = ((color >> 8) & 0xff) / 255.0f;
      const float B = (color & 0xff) / 255.0f;
      colorkeymask_planar_float_c(pf_r, pf_g, pf_b, pf_a, pitch, R, G, B, vi.height, width, tolB / 255.0f, tolG / 255.0f, tolR / 255.0f);
    }
  }
  else {
    // RGB32, RGB64
#ifdef INTEL_INTRINSICS
    if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
    {
      colorkeymask_avx2(pf, pitch, color, vi.height, rowsize, tolB, tolG, tolR);
    }
    else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
    {
      colorkeymask_sse2(pf, pitch, color, vi.height, rowsize, tolB, tolG, tolR);
    }
    else
#endif
      {
        if (pixelsize == 1) {
          const int R = (color >> 16) & 0xff;
          const int G = (color >> 8) & 0xff;
          const int B = color & 0xff;
          colorkeymask_c<uint8_t>(pf, pitch, R, G, B, vi.height, rowsize, tolB, tolG, tolR);
        }
        else {
          const int R = (color64 >> 32) & 0xffff;
          const int G = (color64 >> 16) & 0xffff;
          const int B = color64 & 0xffff;
          colorkeymask_c<uint16_t>(pf, pitch, R, G, B, vi.height, rowsize, tolB16, tolG16, tolR16);
        }
      }
  }

  return frame;
}

AVSValue __cdecl ColorKeyMask::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  enum { CHILD, COLOR, TOLERANCE_B, TOLERANCE_G, TOLERANCE_R };
  return new ColorKeyMask(args[CHILD].AsClip(),
    args[COLOR].AsInt(0),
    args[TOLERANCE_B].AsInt(10),
    args[TOLERANCE_G].AsInt(args[TOLERANCE_B].AsInt(10)),
    args[TOLERANCE_R].AsInt(args[TOLERANCE_B].AsInt(10)), env);
}


/********************************
 ******  ResetMask filter  ******
 ********************************/


ResetMask::ResetMask(PClip _child, AVSValue _mask_f, AVSValue _opacity_f, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{
  if (!(vi.IsRGB32() || vi.IsRGB64() || vi.IsPlanarRGBA() || vi.IsYUVA()))
    env->ThrowError("ResetMask: format has no alpha channel");

  mask_f = _mask_f.AsFloatf(-1.0f);
  const bool mask_defined = _mask_f.Defined();
  const bool opacity_defined = _opacity_f.Defined();
  if (mask_defined && mask_f < 0.0f) {
    env->ThrowError("ResetMask: mask value must be non-negative");
  }
  // only for integers
  int max_pixel_value = vi.BitsPerComponent() <= 16 ? (1 << vi.BitsPerComponent()) - 1 : /*n/a*/ 255;
  // mask and mask_f are the exact unscaled values to put into A channel (integer/float)
  if (!mask_defined) {
    mask_f = 1.0f;
    mask = max_pixel_value;
  }
  else {
    // mask is defined - convert to integer for non-float formats
    if (vi.ComponentSize() < 4) {
      // integer format - use mask value directly
      mask = (int)(mask_f + 0.5f);
    }
    // for float formats, mask_f is already set correctly
  }
  // no check, allow rounding errors
  /*
  if (opacity_defined && _opacity_f.AsFloat(1.0f) < 0.0f) {
    env->ThrowError("ResetMask: opacity value must be non-negative");
  }
  if (opacity_defined && _opacity_f.AsFloat(1.0f) > 1.0f) {
    env->ThrowError("ResetMask: opacity value must be between 0.0 and 1.0");
  }
  */
  if (opacity_defined) {
    // if opacity is defined, calculate mask_f from opacity
    mask_f = _opacity_f.AsFloatf(1.0f);
    if (mask_f < 0.0f) mask_f = 0.0f;
    if (mask_f > 1.0f) mask_f = 1.0f;
    mask = (int)(mask_f * max_pixel_value + 0.5f);
  }

  // mask (bit depth dependent): unscaled value. If none -> max OPACITY (fully opaque)
  //
  // opacity parameter (0.0 to 1.0):
  // opacity = 1.0 means OPAQUE (fully visible, mask value 255/1023/etc. : white)
  // opacity = 0.0 means TRANSPARENT (invisible, mask value 0 : black)
  // when opacity is set, it overrides mask

  mask = std::min(std::max(mask, 0), max_pixel_value);
  mask_f = std::min(std::max(mask_f, 0.0f), 1.0f);
}


PVideoFrame ResetMask::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame f = child->GetFrame(n, env);
  env->MakeWritable(&f);

  if (vi.IsPlanarRGBA() || vi.IsYUVA()) {
    const int dst_rowsizeA = f->GetRowSize(PLANAR_A);
    const int dst_pitchA = f->GetPitch(PLANAR_A);
    BYTE* dstp_a = f->GetWritePtr(PLANAR_A);
    const int heightA = f->GetHeight(PLANAR_A);

    switch (vi.ComponentSize())
    {
    case 1:
      fill_plane<BYTE>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, mask);
      break;
    case 2:
      fill_plane<uint16_t>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, mask);
      break;
    case 4:
      fill_plane<float>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, mask_f);
      break;
    }
    return f;
  }
  // RGB32 and RGB64

  BYTE* pf = f->GetWritePtr();
  int pitch = f->GetPitch();
  int rowsize = f->GetRowSize();
  int height = f->GetHeight();

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
        reinterpret_cast<uint16_t*>(pf)[x] = mask;
      }
      pf += pitch;
    }
  }

  return f;
}


AVSValue ResetMask::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new ResetMask(args[0].AsClip(), args[1], args[2], env);
}


/********************************
 ******  Invert filter  ******
 ********************************/


Invert::Invert(PClip _child, const char* _channels, IScriptEnvironment* env)
  : GenericVideoFilter(_child)
{
  doB = doG = doR = doA = doY = doU = doV = false;

  for (int k = 0; _channels[k] != '\0'; ++k) {
    switch (_channels[k]) {
    case 'B':
    case 'b':
      doB = true;
      break;
    case 'G':
    case 'g':
      doG = true;
      break;
    case 'R':
    case 'r':
      doR = true;
      break;
    case 'A':
    case 'a':
      doA = vi.IsYUVA() || vi.IsPlanarRGBA(); // includes Y+Alpha (2 components)
      break;
    case 'Y':
    case 'y':
      doY = true;
      break;
    case 'U':
    case 'u':
      doU = (vi.NumComponents() > 1) && !vi.IsYA(); // Y+Alpha has no chroma
      break;
    case 'V':
    case 'v':
      doV = (vi.NumComponents() > 1) && !vi.IsYA(); // Y+Alpha has no chroma
      break;
    default:
      break;
    }
  }
  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();
  if (vi.IsYUY2()) {
    mask = doY ? 0x00ff00ff : 0;
    mask |= doU ? 0x0000ff00 : 0;
    mask |= doV ? 0xff000000 : 0;
  }
  else if (vi.IsRGB32()) {
    mask = doB ? 0x000000ff : 0;
    mask |= doG ? 0x0000ff00 : 0;
    mask |= doR ? 0x00ff0000 : 0;
    mask |= doA ? 0xff000000 : 0;
  }
  else if (vi.IsRGB64()) {
    mask64 = doB ? 0x000000000000ffffull : 0;
    mask64 |= (doG ? 0x00000000ffff0000ull : 0);
    mask64 |= (doR ? 0x0000ffff00000000ull : 0);
    mask64 |= (doA ? 0xffff000000000000ull : 0);
  }
  else {
    mask = 0xffffffff;
    mask64 = (1 << bits_per_pixel) - 1;
    mask64 |= (mask64 << 48) | (mask64 << 32) | (mask64 << 16); // works for 10 bit, too
    // RGB24/48 is special case no use of this mask
  }
}


//mod4 width is required
static void invert_frame_inplace_c(BYTE* frame, int pitch, int width, int height, int mask) {
  for (int y = 0; y < height; ++y) {
    int* intptr = reinterpret_cast<int*>(frame);

    for (int x = 0; x < width / 4; ++x) {
      intptr[x] = intptr[x] ^ mask;
    }
    frame += pitch;
  }
}

static void invert_frame_uint16_inplace_c(BYTE* frame, int pitch, int width, int height, uint64_t mask64) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 8; ++x) {
      reinterpret_cast<uint64_t*>(frame)[x] = reinterpret_cast<uint64_t*>(frame)[x] ^ mask64;
    }
    frame += pitch;
  }
}

// called for uint8_t, uint16_t and float planar, chroma planes are inverted differently than luma plane
// R G B are treated the same way as luma.
// We assume full-range.
// 3.7.6 minor change: chroma: uint8_t, uint16_t: pivot around half and not xor FF/FFFF for chroma
// Note: this filter is so simple that it is optimized from C to same speed as SIMD in release
// Also, it is memory-bound AVX2 is not quicker than SSE2.
// lessthan16bits helps optimizing the exact 16 bit case.
// We use this very same C source for AVX2, where it is optimized even with 2x256 bit paths
template<typename pixel_t, bool lessthan16bits, bool chroma>
static void invert_plane_c(uint8_t* dstp, const uint8_t* srcp, int src_pitch, int dst_pitch, int width, int height, int bits_per_pixel) {
  if constexpr (std::is_same_v<pixel_t, float>) {
    if constexpr (chroma) {
      // For chroma planes, invert around 0.0 -> negate
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<float*>(dstp)[x] = -reinterpret_cast<const float*>(srcp)[x];
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
    else {
      // For luma plane, invert around 1.0 -> 1.0 - value
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<float*>(dstp)[x] = 1.0f - reinterpret_cast<const float*>(srcp)[x];
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
    return;
  }
  // 8 bit
  if constexpr (std::is_same_v<pixel_t, uint8_t>) {
    constexpr int max_pixel_value = 255;
    if constexpr (chroma) {
      constexpr int half = 128;
      // For chroma planes, invert around 128 -> negate
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<uint8_t*>(dstp)[x] = std::min(2 * half - reinterpret_cast<const uint8_t*>(srcp)[x], max_pixel_value);
          // chroma invert: -(srcp[x] - half) + half
          // = 2*half - srcp[x] = (1 << bits_per_pixel) - srcp[x]
          // Watch for src==0, must top at max_pixel_value
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
    else {
      // For luma plane, 255-x which is xor 0xff
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<uint8_t*>(dstp)[x] = max_pixel_value - reinterpret_cast<const uint8_t*>(srcp)[x];
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
    return;
  }
  // 10-16 bit uint16_t, luma: max_pixel_value - x which is xor with max_pixel_value, chroma: half - x
  if constexpr (std::is_same_v<pixel_t, uint16_t>) {
    if constexpr (!lessthan16bits)
      bits_per_pixel = 16; // quasi constexpr for optimization
    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    if constexpr (chroma) {
      const int half = 1 << (bits_per_pixel - 1);
      // For chroma planes, invert around mid-point (2^(bits-1))
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<uint16_t*>(dstp)[x] = std::min(2 * half - reinterpret_cast<const uint16_t*>(srcp)[x], max_pixel_value);
          // chroma invert: -(srcp[x] - half) + half
          // = 2*half - srcp[x] = (1 << bits_per_pixel) - srcp[x]
          // Watch for src==0, must top at max_pixel_value
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
    else {
      // For luma plane, max_pixel_value - x
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          reinterpret_cast<uint16_t*>(dstp)[x] = max_pixel_value - reinterpret_cast<const uint16_t*>(srcp)[x];
        }
        srcp += src_pitch;
        dstp += dst_pitch;
      }
    }
  }
}

// YUY2 and packed RGB formats
static void invert_frame_inplace(BYTE* frame, int pitch, int rowsize, int height, int mask, uint64_t mask64, int pixelsize, IScriptEnvironment* env) {
#ifdef INTEL_INTRINSICS
  if ((pixelsize == 1 || pixelsize == 2) && (env->GetCPUFlags() & CPUF_AVX2))
  {
    if (pixelsize == 1)
      invert_frame_inplace_avx2(frame, pitch, rowsize, height, mask);
    else
      invert_frame_uint16_inplace_avx2(frame, pitch, rowsize, height, mask64);
  }
  else if ((pixelsize == 1 || pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2))
  {
    if (pixelsize == 1)
      invert_frame_inplace_sse2(frame, pitch, rowsize, height, mask);
    else
      invert_frame_uint16_inplace_sse2(frame, pitch, rowsize, height, mask64);
  }
  else
#endif
  {
    if (pixelsize == 1)
      invert_frame_inplace_c(frame, pitch, rowsize, height, mask);
    else
      invert_frame_uint16_inplace_c(frame, pitch, rowsize, height, mask64);
  }
}

// Function pointer type definition
using invert_plane_fn_t = void(*)(uint8_t*, const uint8_t*, int, int, int, int, int);

// width is in pixels, not bytes.
// AVX2: explicit SIMD kernels (invert_plane_avx2_u8/u16/f32).
// SSE2: explicit SIMD kernels (invert_plane_sse2_u8/u16/f32).
// Fallback: C templates (invert_plane_c).
static void invert_plane(uint8_t* dstp, const uint8_t* srcp, int src_pitch, int dst_pitch, int width, int height, int bits_per_pixel, bool chroma, IScriptEnvironment* env) {
  const int pixelsize = bits_per_pixel == 8 ? 1 : bits_per_pixel <= 16 ? 2 : 4; // 8 bit = 1 byte, 10-16 bit = 2 bytes, float = 4 bytes
#ifdef INTEL_INTRINSICS
  const bool avx2 = env->GetCPUFlags() & CPUF_AVX2;
  const bool sse2 = env->GetCPUFlags() & CPUF_SSE2;
#endif

  invert_plane_fn_t fn = nullptr;

  switch (pixelsize) {
  case 1:
#ifdef INTEL_INTRINSICS
    if (avx2)
      fn = chroma ? invert_plane_avx2_u8<true> : invert_plane_avx2_u8<false>;
    else if (sse2)
      fn = chroma ? invert_plane_sse2_u8<true> : invert_plane_sse2_u8<false>;
    else
#endif
      fn = chroma ? invert_plane_c<uint8_t, true /*lt16b* n/a */, true> : invert_plane_c<uint8_t, true /*lt16b* n/a */, false>;
    break;
  case 2:
#ifdef INTEL_INTRINSICS
    if (avx2)
      fn = chroma ? invert_plane_avx2_u16<true> : invert_plane_avx2_u16<false>;
    else if (sse2)
      fn = chroma ? invert_plane_sse2_u16<true> : invert_plane_sse2_u16<false>;
    else
#endif
      if (bits_per_pixel < 16)
        fn = chroma ? invert_plane_c<uint16_t, true /*lt16b**/, true> : invert_plane_c<uint16_t, true /*lt16b**/, false>;
      else
        fn = chroma ? invert_plane_c<uint16_t, false /*lt16b**/, true> : invert_plane_c<uint16_t, false /*lt16b**/, false>;
    break;
  case 4:
#ifdef INTEL_INTRINSICS
    if (avx2)
      fn = chroma ? invert_plane_avx2_f32<true> : invert_plane_avx2_f32<false>;
    else if (sse2)
      fn = chroma ? invert_plane_sse2_f32<true> : invert_plane_sse2_f32<false>;
    else
#endif
      fn = chroma ? invert_plane_c<float, false /*lt16b* n/a */, true> : invert_plane_c<float, false /*lt16b* n/a */, false>;
    break;
  }

  fn(dstp, srcp, src_pitch, dst_pitch, width, height, bits_per_pixel);
}

PVideoFrame Invert::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  if (vi.IsPlanar()) {

    // We do not use worst case MakeWritable, do it per plane.
    // Read and Invert only those planes which are needed, BitBlt the rest.

    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    // Helper function to process or copy a plane
    auto process_plane = [&](int plane, bool do_process, bool is_chroma) {
      if (do_process) {
        invert_plane(
          dst->GetWritePtr(plane),
          src->GetReadPtr(plane),
          src->GetPitch(plane),
          dst->GetPitch(plane),
          dst->GetRowSize(plane) / pixelsize,
          dst->GetHeight(plane),
          bits_per_pixel,
          is_chroma,
          env
        );
      }
      else {
        env->BitBlt(
          dst->GetWritePtr(plane),
          dst->GetPitch(plane),
          src->GetReadPtr(plane),
          src->GetPitch(plane),
          src->GetRowSize(plane | PLANAR_ALIGNED),  // Use aligned row size for BitBlt optimization
          src->GetHeight(plane)
        );
      }
      };

    // YUV/YUVA
    if (vi.IsYUV() || vi.IsYUVA()) {
      process_plane(PLANAR_Y, doY, false);
      process_plane(PLANAR_U, doU, true);
      process_plane(PLANAR_V, doV, true);
    }

    // Planar RGB/RGBA
    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      process_plane(PLANAR_G, doG, false);
      process_plane(PLANAR_B, doB, false);
      process_plane(PLANAR_R, doR, false);
    }

    // Alpha channel
    if (vi.IsPlanarRGBA() || vi.IsYUVA()) {
      process_plane(PLANAR_A, doA, false);
    }

    return dst;
  }

  // packed formats, we do a full copy of the frame before inverting
  env->MakeWritable(&src);

  BYTE* pf = src->GetWritePtr();
  int pitch = src->GetPitch();
  int rowsize = src->GetRowSize();
  int height = src->GetHeight();

  if (vi.IsYUY2() || vi.IsRGB32() || vi.IsRGB64()) {
    // packed pixels, 4x1 or 4x2 bytes, all can be treated the same way with a mask
    // YUY2 is simply xored even in its chroma component. Not really correct, but YUY2 is a compability format.
    // we won't convert it to and from YV16, keep its pre-3.7.6 behavior.
    invert_frame_inplace(pf, pitch, rowsize, height, mask, mask64, pixelsize, env);
  }
  else if (vi.IsRGB24()) {
    int rMask = doR ? 0xff : 0;
    int gMask = doG ? 0xff : 0;
    int bMask = doB ? 0xff : 0;
    for (int i = 0; i < height; i++) {

      for (int j = 0; j < rowsize; j += 3) {
        pf[j + 0] = pf[j + 0] ^ bMask;
        pf[j + 1] = pf[j + 1] ^ gMask;
        pf[j + 2] = pf[j + 2] ^ rMask;
      }
      pf += pitch;
    }
  }
  else if (vi.IsRGB48()) {
    int rMask = doR ? 0xffff : 0;
    int gMask = doG ? 0xffff : 0;
    int bMask = doB ? 0xffff : 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < rowsize / pixelsize; j += 3) {
        reinterpret_cast<uint16_t*>(pf)[j + 0] ^= bMask;
        reinterpret_cast<uint16_t*>(pf)[j + 1] ^= gMask;
        reinterpret_cast<uint16_t*>(pf)[j + 2] ^= rMask;
      }
      pf += pitch;
    }
  }

  return src;
}


AVSValue Invert::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Invert(args[0].AsClip(), args[0].AsClip()->GetVideoInfo().IsRGB() ? args[1].AsString("RGBA") : args[1].AsString("YUVA"), env);
}


/**********************************
 ******  ShowChannel filter  ******
 **********************************/


ShowChannel::ShowChannel(PClip _child, const char* pixel_type, int _channel, IScriptEnvironment* env)
  : GenericVideoFilter(_child), channel(_channel), input_type(_child->GetVideoInfo().pixel_type),
  pixelsize(_child->GetVideoInfo().ComponentSize()), bits_per_pixel(_child->GetVideoInfo().BitsPerComponent())
{
  static const char* const ShowText[7] = { "Blue", "Green", "Red", "Alpha", "Y", "U", "V" };

  input_type_is_packed_rgb = vi.IsRGB() && !vi.IsPlanar();
  input_type_is_planar_rgb = vi.IsPlanarRGB();
  input_type_is_planar_rgba = vi.IsPlanarRGBA();
  input_type_is_yuva = vi.IsYUVA();
  input_type_is_yuv = vi.IsYUV() && vi.IsPlanar();
  input_type_is_planar = vi.IsPlanar();

  int orig_channel = channel;

  // A channel
  // IsYUVA() is also true for YA()
  if ((channel == 3) && !vi.IsRGB32() && !vi.IsRGB64() && !vi.IsPlanarRGBA() && !vi.IsYUVA())
    env->ThrowError("ShowAlpha: RGB32, RGB64, Planar RGBA, YUVA or YA data only");

  // R, G, B channel
  if ((channel >= 0) && (channel <= 2) && !vi.IsRGB())
    env->ThrowError("Show%s: plane is valid only with RGB or planar RGB(A) source", ShowText[channel]);

  // Y, U, V channel (4,5,6)
  if ((channel >= 4) && (channel <= 6)) {
    if (!vi.IsYUV() && !vi.IsYUVA())
      env->ThrowError("Show%s: plane is valid only with YUV(A) source", ShowText[channel]);
    if (channel != 4 && (vi.IsY() || vi.IsYA()))
      env->ThrowError("Show%s: invalid plane for greyscale or Y+Alpha source", ShowText[channel]);
    channel -= 4; // map to 0,1,2
  }

  int target_bits_per_pixel;

  const int orig_width = vi.width;
  const int orig_height = vi.height;

  if (input_type_is_yuv || input_type_is_yuva)
  {
    if (channel == 1 || channel == 2) // U or V: target can be smaller than Y
    {
      vi.width >>= vi.GetPlaneWidthSubsampling(PLANAR_U);
      vi.height >>= vi.GetPlaneHeightSubsampling(PLANAR_U);
    }
  }

  const bool empty_pixel_type = pixel_type == nullptr || *pixel_type == 0;

  if (!lstrcmpi(pixel_type, "rgb") || (vi.IsRGB() && empty_pixel_type)) {
    // target is RGB, rgb (packed) is adaptively 32 or 64 bits
    //                rgb (planar) is of any bit depths
    if (vi.IsPlanar()) {
      // YUV, planar RGB or Y
      // always alphaless planar RGB output
      switch (bits_per_pixel) {
      case 8: vi.pixel_type = VideoInfo::CS_RGBP8; break;
      case 10: vi.pixel_type = VideoInfo::CS_RGBP10; break;
      case 12: vi.pixel_type = VideoInfo::CS_RGBP12; break;
      case 14: vi.pixel_type = VideoInfo::CS_RGBP14; break;
      case 16: vi.pixel_type = VideoInfo::CS_RGBP16; break;
      case 32: vi.pixel_type = VideoInfo::CS_RGBPS; break;
      }
    }
    else if (vi.IsRGB()) {
      // packed RGB source
      switch (bits_per_pixel) {
      case 8: vi.pixel_type = VideoInfo::CS_BGR32; break; // bit-depth adaptive
      case 16: vi.pixel_type = VideoInfo::CS_BGR64; break;
      default: env->ThrowError("Show%s: source must be 8 or 16 bits", ShowText[orig_channel]);
      }
    }
    else {
      env->ThrowError("Show%s: unsupported source format", ShowText[orig_channel]);
    }
    target_bits_per_pixel = bits_per_pixel;
  }
  else if (!lstrcmpi(pixel_type, "yuv") || ((vi.IsYUV() || vi.IsYUVA()) && empty_pixel_type)) {
    // target is YUV, rgb (packed) is adaptively 32 or 64 bits
    //                rgb (planar) is of any bit depths
    //                YUV,Y: 420
      // RGB source, when only 'yuv' is given, convert to 444
    switch (bits_per_pixel) {
    case 8: vi.pixel_type = VideoInfo::CS_YV24; break;
    case 10: vi.pixel_type = VideoInfo::CS_YUV444P10; break;
    case 12: vi.pixel_type = VideoInfo::CS_YUV444P12; break;
    case 14: vi.pixel_type = VideoInfo::CS_YUV444P14; break;
    case 16: vi.pixel_type = VideoInfo::CS_YUV444P16; break;
    case 32: vi.pixel_type = VideoInfo::CS_YUV444PS; break;
    }
    target_bits_per_pixel = bits_per_pixel;
  }
  else {
    // explicitely given output pixel type
    
    // first try
    // Append bit depth and check
    std::string format = pixel_type;
    // RGBP --> RGBPS, Y -> Y16, YUV420 -> YUV420P10
    if (!lstrcmpi(pixel_type, "y")) {
      format = format + std::to_string(bits_per_pixel); // Y8..Y16, also Y32
    }
    else if (!lstrcmpi(pixel_type, "rgbp") || !lstrcmpi(pixel_type, "rgbap")) {
      if (bits_per_pixel == 32)
        format = format + "S"; // RGBAPS
      else
        format = format + std::to_string(bits_per_pixel); // RGBP16
    }
    else {
      // hopefully like "yuv420" or "yuva444"
      if (bits_per_pixel == 32)
        format = format + "PS"; // YUV420PS
      else
        format = format + "P" + std::to_string(bits_per_pixel); // YUV420P16
    }

    int new_pixel_type = GetPixelTypeFromName(format.c_str());
    if (new_pixel_type == VideoInfo::CS_UNKNOWN) {
      new_pixel_type = GetPixelTypeFromName(pixel_type);
      if (new_pixel_type == VideoInfo::CS_UNKNOWN)
        env->ThrowError("Show%s: invalid pixel_type!", ShowText[orig_channel]);
    }
    // new output format
    vi.pixel_type = new_pixel_type;

    if (vi.IsYUY2()) {
      if (vi.width & 1) {
        env->ThrowError("Show%s: width must be mod 2 for yuy2", ShowText[orig_channel]);
      }
    }
    if (vi.Is420()) {
      if (vi.width & 1) {
        env->ThrowError("Show%s: width must be mod 2 for 4:2:0 target", ShowText[orig_channel]);
      }
      if (vi.height & 1) {
        env->ThrowError("Show%s: height must be mod 2 for 4:2:0 target", ShowText[orig_channel]);
      }
    }
    if (vi.Is422()) {
      if (vi.width & 1) {
        env->ThrowError("Show%s: width must be mod 2 for 4:2:2 target", ShowText[orig_channel]);
      }
    }
    if (vi.Is411()) {
      if (vi.width & 3) {
        env->ThrowError("Show%s: width must be mod 4 for 4:1:1 target", ShowText[orig_channel]);
      }
    }
    if (vi.Is440()) {
      // 4:4:0 has no horizontal chroma subsampling, so no width constraint here.
      if (vi.height & 1) {
        env->ThrowError("Show%s: height must be mod 2 for 4:4:0 target", ShowText[orig_channel]);
      }
    }
    if (vi.Is410()) {
      if (vi.width & 3) {
        env->ThrowError("Show%s: width must be mod 4 for 4:1:0 target", ShowText[orig_channel]);
      }
      if (vi.height & 3) {
        env->ThrowError("Show%s: height must be mod 4 for 4:1:0 target", ShowText[orig_channel]);
      }
    }

    target_bits_per_pixel = vi.BitsPerComponent();
  }

  if (target_bits_per_pixel != bits_per_pixel)
    env->ThrowError("Show%s: source bit depth must be %d for %s", ShowText[orig_channel], target_bits_per_pixel, pixel_type);

  target_hasalpha = vi.IsRGB32() || vi.IsRGB64() || vi.IsPlanarRGBA() || vi.IsYUVA();
  source_hasalpha = input_type == VideoInfo::CS_BGR32 || input_type == VideoInfo::CS_BGR64 || input_type_is_planar_rgba || input_type_is_yuva;

  if (target_hasalpha && source_hasalpha && (vi.width != orig_width || vi.height != orig_height)) {
    env->ThrowError("Show%s: subsampled source plane and alpha-aware source and destination format: alpha dimensions must be the same", ShowText[orig_channel]);
  }

}


template<typename pixel_t, bool source_hasalpha, bool target_hasalpha>
static void planar_to_packedrgb(uint8_t* dstp, int dst_pitch, const uint8_t* srcp, const uint8_t* srcp_a, int src_pitch, int width, int height)
{
  // packed RGB is upside-down
  dstp += (height - 1) * dst_pitch;
  constexpr int target_rgb_step = target_hasalpha ? 4 : 3;
  constexpr int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : 65535;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; j++) {
      pixel_t* curr_dstp = reinterpret_cast<pixel_t*>(dstp);
      curr_dstp[j * target_rgb_step + 0] = curr_dstp[j * target_rgb_step + 1] = curr_dstp[j * target_rgb_step + 2] = reinterpret_cast<const pixel_t*>(srcp)[j];
      if constexpr(target_hasalpha) {
        const int alpha = source_hasalpha ? reinterpret_cast<const pixel_t*>(srcp_a)[j] : max_pixel_value;
        curr_dstp[j * target_rgb_step + 3] = alpha;
      }
    }
    srcp += src_pitch;
    if constexpr(source_hasalpha)
      srcp_a += src_pitch; // alpha has the same pitch
    dstp -= dst_pitch;
  }
}

template<typename pixel_t, bool source_hasalpha, bool target_hasalpha>
static void packed_to_packedrgb(uint8_t* dstp, int dst_pitch, const uint8_t* srcp, int pitch, int width, int height, int channel)
{
  constexpr int target_rgb_step = target_hasalpha ? 4 : 3;
  constexpr int source_rgb_step = source_hasalpha ? 4 : 3;
  constexpr int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : 65535;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; j++) {
      pixel_t* curr_dstp = reinterpret_cast<pixel_t*>(dstp);
      const int pixel = reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + channel];
      curr_dstp[j * target_rgb_step + 0] = curr_dstp[j * target_rgb_step + 1] = curr_dstp[j * target_rgb_step + 2] = pixel;
      if constexpr(target_hasalpha) {
        const int alpha = source_hasalpha ? reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + 3] : max_pixel_value;
        curr_dstp[j * target_rgb_step + 3] = alpha; // alpha
      }
    }
    srcp += pitch;
    dstp += dst_pitch;
  }
}

template<typename pixel_t, bool source_hasalpha, bool target_hasalpha>
static void packed_to_planarrgb(uint8_t* dstp_r, uint8_t* dstp_g, uint8_t* dstp_b, uint8_t* dstp_a, int dst_pitch, const uint8_t* srcp, int src_pitch, int width, int height, int channel)
{
  constexpr int source_rgb_step = source_hasalpha ? 4 : 3;
  constexpr int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : 65535;

  // packed RGB is upside-down
  srcp += (height - 1) * src_pitch;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int pixel = reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + channel];
      reinterpret_cast<pixel_t*>(dstp_g)[j] =
        reinterpret_cast<pixel_t*>(dstp_b)[j] =
        reinterpret_cast<pixel_t*>(dstp_r)[j] = pixel;
      if constexpr (target_hasalpha) {
        const int alpha = source_hasalpha ? reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + 3] : max_pixel_value;
        reinterpret_cast<pixel_t*>(dstp_a)[j] = alpha;
      }
    }
    srcp -= src_pitch;
    dstp_g += dst_pitch;
    dstp_b += dst_pitch;
    dstp_r += dst_pitch;
    if constexpr (target_hasalpha)
      dstp_a += dst_pitch;
  }
}

template<typename pixel_t, bool source_hasalpha, bool target_hasalpha>
static void packed_to_luma_alpha(uint8_t* dstp_y, uint8_t* dstp_a, int dst_pitch, const uint8_t* srcp, int src_pitch, int width, int height, int channel)
{
  constexpr int source_rgb_step = source_hasalpha ? 4 : 3;
  constexpr int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : 65535;

  // packed RGB is upside-down
  srcp += (height - 1) * src_pitch;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int pixel = reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + channel];
      reinterpret_cast<pixel_t*>(dstp_y)[j] = pixel;
      if constexpr (target_hasalpha) {
        const int alpha = source_hasalpha ? reinterpret_cast<const pixel_t*>(srcp)[j * source_rgb_step + 3] : max_pixel_value;
        reinterpret_cast<pixel_t*>(dstp_a)[j] = alpha;
      }
    }
    srcp -= src_pitch;
    dstp_y += dst_pitch;
    if constexpr (target_hasalpha)
      dstp_a += dst_pitch;
  }
}


PVideoFrame ShowChannel::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);

  // for planar these will be reread for proper plane
  const BYTE* srcp = src->GetReadPtr();
  const int height = src->GetHeight();
  const int pitch = src->GetPitch();
  const int rowsize = src->GetRowSize();

  const float chroma_center_f = 0.0f;

  if (input_type_is_packed_rgb) {
    PVideoFrame dst = env->NewVideoFrameP(vi, &src);

    if (!vi.IsRGB()) {
      // delete _Matrix when target is not an RGB
      auto props = env->getFramePropsRW(dst);
      env->propDeleteKey(props, "_Matrix");
    }

    const int source_rgb_step = source_hasalpha ? 4 : 3;
    const int w = rowsize / pixelsize / source_rgb_step;

    if (vi.IsRGB() && !vi.IsPlanar())
    {
      // packed RGB to packed RGB
      BYTE* dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      if (pixelsize == 1) {
        if (!source_hasalpha && !target_hasalpha)
          packed_to_packedrgb<uint8_t, false, false>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_packedrgb<uint8_t, false, true>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_packedrgb<uint8_t, true, false>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_packedrgb<uint8_t, true, true>(dstp, dstpitch, srcp, pitch, w, height, channel);
      }
      else {
        if (!source_hasalpha && !target_hasalpha)
          packed_to_packedrgb<uint16_t, false, false>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_packedrgb<uint16_t, false, true>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_packedrgb<uint16_t, true, false>(dstp, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_packedrgb<uint16_t, true, true>(dstp, dstpitch, srcp, pitch, w, height, channel);
      }
    }
    else if (vi.pixel_type == VideoInfo::CS_YUY2)
    {
      // packed RGB to YUY2
      BYTE* dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      // RGB is upside-down
      srcp += (height - 1) * pitch;

      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < w; j++) {
          dstp[j * 2 + 0] = srcp[j * source_rgb_step + channel];
          dstp[j * 2 + 1] = 128;
        }
        srcp -= pitch;
        dstp += dstpitch;
      }
    }
    else if (vi.IsYUV() || vi.IsYUVA() || vi.IsY())
    {
      // packed RGB -> Y, YUV(A)
      BYTE* dstp = dst->GetWritePtr();
      int dstpitch = dst->GetPitch();

      BYTE* dstp_a = target_hasalpha ? dst->GetWritePtr(PLANAR_A) : nullptr;

      if (pixelsize == 1) {
        if (!source_hasalpha && !target_hasalpha)
          packed_to_luma_alpha<uint8_t, false, false>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_luma_alpha<uint8_t, false, true>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_luma_alpha<uint8_t, true, false>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_luma_alpha<uint8_t, true, true>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
      }
      else {
        // 16 bit
        if (!source_hasalpha && !target_hasalpha)
          packed_to_luma_alpha<uint16_t, false, false>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_luma_alpha<uint16_t, false, true>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_luma_alpha<uint16_t, true, false>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_luma_alpha<uint16_t, true, true>(dstp, dstp_a, dstpitch, srcp, pitch, w, height, channel);
      }

      // fill chroma neutral
      if (!vi.IsY())
      {
        int uvrowsize = dst->GetRowSize(PLANAR_U);
        int uvpitch = dst->GetPitch(PLANAR_U);
        int dstheight = dst->GetHeight(PLANAR_U);
        BYTE* dstp_u = dst->GetWritePtr(PLANAR_U);
        BYTE* dstp_v = dst->GetWritePtr(PLANAR_V);
        switch (pixelsize) {
        case 1: fill_chroma<BYTE>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, (BYTE)0x80); break;
        case 2: fill_chroma<uint16_t>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, 1 << (vi.BitsPerComponent() - 1)); break;
        case 4:
          fill_chroma<float>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, chroma_center_f);
          break;
        }
      }
    }
    else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
    {  // packed RGB -> Planar RGB 8/16 bit
      BYTE* dstp_g = dst->GetWritePtr(PLANAR_G);
      BYTE* dstp_b = dst->GetWritePtr(PLANAR_B);
      BYTE* dstp_r = dst->GetWritePtr(PLANAR_R);
      int dstpitch = dst->GetPitch();

      BYTE* dstp_a = target_hasalpha ? dst->GetWritePtr(PLANAR_A) : nullptr;

      if (pixelsize == 1) {
        if (!source_hasalpha && !target_hasalpha)
          packed_to_planarrgb<uint8_t, false, false>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_planarrgb<uint8_t, false, true>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_planarrgb<uint8_t, true, false>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_planarrgb<uint8_t, true, true>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
      }
      else {
        // 16 bit
        if (!source_hasalpha && !target_hasalpha)
          packed_to_planarrgb<uint16_t, false, false>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (!source_hasalpha && target_hasalpha)
          packed_to_planarrgb<uint16_t, false, true>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else if (source_hasalpha && !target_hasalpha)
          packed_to_planarrgb<uint16_t, true, false>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
        else // if (source_hasalpha && target_hasalpha)
          packed_to_planarrgb<uint16_t, true, true>(dstp_r, dstp_g, dstp_b, dstp_a, dstpitch, srcp, pitch, w, height, channel);
      }
    }
    return dst;
  } // end of packed rgb source
  
  if (input_type_is_planar_rgb || input_type_is_planar_rgba || input_type_is_yuv || input_type_is_yuva) {
    // planar source
    const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* planes = (input_type_is_planar_rgb || input_type_is_planar_rgba) ? planesRGB : planesYUV;
    int final_channel = channel;
    // RGB channels: B=0 G=1 R=2 (like packed)
    // Planar order: G=0 B=1 R=2
    if (input_type_is_planar_rgb || input_type_is_planar_rgba) {
      // exchange B and G
      if (channel == 0) final_channel = 1;
      else if (channel == 1) final_channel = 0;
    }
    const int plane = planes[final_channel];

    const BYTE* srcp = src->GetReadPtr(plane); // source plane
    const BYTE* srcp_a = source_hasalpha ? src->GetReadPtr(PLANAR_A) : nullptr;

    const int width = src->GetRowSize(plane) / pixelsize;
    const int height = src->GetHeight(plane);
    const int pitch = src->GetPitch(plane);

    if (vi.IsRGB() && !vi.IsPlanar())
    {
      // planar RGBP/YUVA -> packed RGB
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      BYTE* dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      if (!input_type_is_planar_rgb && !input_type_is_planar_rgba) {
        auto props = env->getFramePropsRW(dst);
        // delete _Matrix and ChromaLocation when source is not RGB
        env->propDeleteKey(props, "_Matrix");
        env->propDeleteKey(props, "_ChromaLocation");
      }

      if (bits_per_pixel == 8) {
        if (target_hasalpha) {
          if (source_hasalpha)
            planar_to_packedrgb<uint8_t, true, true>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
          else
            planar_to_packedrgb<uint8_t, false, true>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
        }
        else {
          planar_to_packedrgb<uint8_t, false, false>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
        }
      }
      else {
        // 16 bits
        if (target_hasalpha) {
          if (source_hasalpha)
            planar_to_packedrgb<uint16_t, true, true>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
          else
            planar_to_packedrgb<uint16_t, false, true>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
        }
        else {
          planar_to_packedrgb<uint16_t, false, false>(dstp, dstpitch, srcp, srcp_a, pitch, width, height);
        }
      }

      return dst;
    }
    else if (vi.pixel_type == VideoInfo::CS_YUY2) // RGB(A)P/YUVA->YUY2
    {
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);
      BYTE* dstp = dst->GetWritePtr();
      const int dstpitch = dst->GetPitch();

      auto props = env->getFramePropsRW(dst);
      env->propDeleteKey(props, "_Matrix");
      env->propDeleteKey(props, "_ChromaLocation");

      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; j++) {
          dstp[j * 2 + 0] = srcp[j];
          dstp[j * 2 + 1] = 128;
        }
        srcp += pitch;
        dstp += dstpitch;
      }
      return dst;
    }
    else
    { // planar to planar
      // RGB(A)P/YUVA -> YV12/16/24/Y8 + 16bit
      PVideoFrame dst = env->NewVideoFrameP(vi, &src);

      // remove frame props if either src or target is not RGB
      if (!(input_type_is_planar_rgb || input_type_is_planar_rgba || vi.IsRGB())) {
        // RGB origin to YUV
        auto props = env->getFramePropsRW(dst);
        env->propDeleteKey(props, "_Matrix");
        if(input_type_is_yuv || input_type_is_yuva)
          env->propDeleteKey(props, "_ChromaLocation");
      }

      if (vi.IsYUV() || vi.IsYUVA() || vi.IsY()) // Y8, YV12, Y16, YUV420P16, etc.
      {
        BYTE* dstp = dst->GetWritePtr();
        int dstpitch = dst->GetPitch();

        // copy source plane to luma
        env->BitBlt(dstp, dstpitch, srcp, pitch, width * pixelsize, height);
        // fill UV with neutral
        if (!vi.IsY())
        {
          int uvrowsize = dst->GetRowSize(PLANAR_U);
          int uvpitch = dst->GetPitch(PLANAR_U);
          int dstheight = dst->GetHeight(PLANAR_U);
          BYTE* dstp_u = dst->GetWritePtr(PLANAR_U);
          BYTE* dstp_v = dst->GetWritePtr(PLANAR_V);
          switch (pixelsize) {
          case 1: fill_chroma<uint8_t>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, (uint8_t)0x80); break;
          case 2: fill_chroma<uint16_t>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, 1 << (vi.BitsPerComponent() - 1)); break;
          case 4: fill_chroma<float>(dstp_u, dstp_v, dstheight, uvrowsize, uvpitch, chroma_center_f); break;
          }
        }
      }
      else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
      {  // RGBP(A)/YUVA -> Planar RGB
        BYTE* dstp_g = dst->GetWritePtr(PLANAR_G);
        BYTE* dstp_b = dst->GetWritePtr(PLANAR_B);
        BYTE* dstp_r = dst->GetWritePtr(PLANAR_R);

        int dstpitch = dst->GetPitch();
        int dstwidth = dst->GetRowSize() / pixelsize;

        // copy to all channels
        if (pixelsize == 1) {
          for (int i = 0; i < height; ++i) {
            for (int j = 0; j < dstwidth; ++j) {
              dstp_g[j] = dstp_b[j] = dstp_r[j] = srcp[j];
            }
            srcp += pitch;
            dstp_g += dstpitch;
            dstp_b += dstpitch;
            dstp_r += dstpitch;
          }
        }
        else if (pixelsize == 2) {
          for (int i = 0; i < height; ++i) {
            for (int j = 0; j < dstwidth; ++j) {
              reinterpret_cast<uint16_t*>(dstp_g)[j] =
                reinterpret_cast<uint16_t*>(dstp_b)[j] =
                reinterpret_cast<uint16_t*>(dstp_r)[j] = reinterpret_cast<const uint16_t*>(srcp)[j];
            }
            srcp += pitch;
            dstp_g += dstpitch;
            dstp_b += dstpitch;
            dstp_r += dstpitch;
          }
        }
        else { // pixelsize==4
          for (int i = 0; i < height; ++i) {
            for (int j = 0; j < dstwidth; ++j) {
              reinterpret_cast<float*>(dstp_g)[j] =
                reinterpret_cast<float*>(dstp_b)[j] =
                reinterpret_cast<float*>(dstp_r)[j] = reinterpret_cast<const float*>(srcp)[j];
            }
            srcp += pitch;
            dstp_g += dstpitch;
            dstp_b += dstpitch;
            dstp_r += dstpitch;
          }
        }
      }
      if (target_hasalpha) {
        // fill alpha with transparent
        const int dst_rowsizeA = dst->GetRowSize(PLANAR_A);
        const int dst_pitchA = dst->GetPitch(PLANAR_A);
        BYTE* dstp_a = dst->GetWritePtr(PLANAR_A);
        const int heightA = dst->GetHeight(PLANAR_A);

        if (source_hasalpha) {
          // copy source alpha plane to target alpha plane
          env->BitBlt(dstp_a, dst_pitchA, srcp_a, pitch, width* pixelsize, height);
        }
        else {
          switch (vi.ComponentSize())
          {
          case 1:
            fill_plane<uint8_t>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, 0xFF);
            break;
          case 2:
            fill_plane<uint16_t>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, (1 << vi.BitsPerComponent()) - 1);
            break;
          case 4:
            fill_plane<float>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, 1.0f);
            break;
          }
        }
      }
      return dst;
    }
  } // planar RGB(A) or YUVA source

  env->ThrowError("ShowChannel: unexpected end of function");
  return src;
}


AVSValue ShowChannel::Create(AVSValue args, void* channel, IScriptEnvironment* env)
{
  // yuy2 is autoconverted to YV16
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();

  if (vi.IsYUY2()) {
    AVSValue new_args[1] = { clip };
    clip = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }
  return new ShowChannel(clip, args[1].AsString(""), (int)(size_t)channel, env);
}


/**********************************
 ******  MergeRGB filter  ******
 **********************************/


MergeRGB::MergeRGB(PClip _child, PClip _blue, PClip _green, PClip _red, PClip _alpha,
  const char* pixel_type, IScriptEnvironment* env)
  : GenericVideoFilter(_child), blue(_blue), green(_green), red(_red), alpha(_alpha),
  viB(blue->GetVideoInfo()), viG(green->GetVideoInfo()), viR(red->GetVideoInfo()),
  viA(((alpha) ? alpha : child)->GetVideoInfo()), myname((alpha) ? "MergeARGB" : "MergeRGB")
{
  vi = viR; // comparison base

  if ((vi.BitsPerComponent() != viB.BitsPerComponent()) || (vi.BitsPerComponent() != viG.BitsPerComponent()) ||
    (vi.BitsPerComponent() != viR.BitsPerComponent()) || (vi.BitsPerComponent() != viA.BitsPerComponent()))
    env->ThrowError("%s: All clips must have the same bit depth.", myname);

  if ((vi.width != viB.width) || (vi.width != viG.width) || (vi.width != viR.width) || (vi.width != viA.width))
    env->ThrowError("%s: All clips must have the same width.", myname);

  if ((vi.height != viB.height) || (vi.height != viG.height) || (vi.height != viR.height) || (vi.height != viA.height))
    env->ThrowError("%s: All clips must have the same height.", myname);

  const int is_any_planar_rgb =
    viR.IsPlanarRGB() || viR.IsPlanarRGBA() ||
    viG.IsPlanarRGB() || viG.IsPlanarRGBA() ||
    viB.IsPlanarRGB() || viB.IsPlanarRGBA() ||
    viA.IsPlanarRGB() || viA.IsPlanarRGBA();

  const int bits_per_pixel = viR.BitsPerComponent();
  const bool empty_pixel_type = pixel_type == nullptr || *pixel_type == 0;

  // planar rgb target if
  // - pixel_type is "rgb" or
  // - pixel_type not specified and
  //   - bit depth is not 8 or 16 (cannot have packed rgb representation) or
  //   - any of the input clips is planar rgb

  if (!lstrcmpi(pixel_type, "rgb") ||
    (empty_pixel_type && (is_any_planar_rgb || (bits_per_pixel != 8 && bits_per_pixel != 16))))
  {
    switch (bits_per_pixel) {
    case 8: vi.pixel_type = alpha ? VideoInfo::CS_RGBAP8 : VideoInfo::CS_RGBP8; break;
    case 10: vi.pixel_type = alpha ? VideoInfo::CS_RGBAP10 : VideoInfo::CS_RGBP10; break;
    case 12: vi.pixel_type = alpha ? VideoInfo::CS_RGBAP12 : VideoInfo::CS_RGBP12; break;
    case 14: vi.pixel_type = alpha ? VideoInfo::CS_RGBAP14 : VideoInfo::CS_RGBP14; break;
    case 16: vi.pixel_type = alpha ? VideoInfo::CS_RGBAP16 : VideoInfo::CS_RGBP16; break;
    case 32: vi.pixel_type = alpha ? VideoInfo::CS_RGBAPS : VideoInfo::CS_RGBPS; break;
    }
  }
  else if (empty_pixel_type && vi.BitsPerComponent() == 8) {
    // default for 8 bit
    vi.pixel_type = VideoInfo::CS_BGR32;
  }
  else if (empty_pixel_type && vi.BitsPerComponent() == 16) {
    // default for 16 bit
    vi.pixel_type = VideoInfo::CS_BGR64;
  }
  else {
    // explicitely given output pixel type

    // first try
    // Append bit depth and check
    std::string format = pixel_type;
    // RGBP --> RGBPS, Y -> Y16, YUV420 -> YUV420P10
    if (!lstrcmpi(pixel_type, "y")) {
      format = format + std::to_string(bits_per_pixel); // Y8..Y16, also Y32
    }
    else if (!lstrcmpi(pixel_type, "rgbp") || !lstrcmpi(pixel_type, "rgbap")) {
      if (bits_per_pixel == 32)
        format = format + "S"; // RGBAPS
      else
        format = format + std::to_string(bits_per_pixel); // RGBP16
    }
    else {
      // hopefully like "yuv420" or "yuva444"
      if (bits_per_pixel == 32)
        format = format + "PS"; // YUV420PS
      else
        format = format + "P" + std::to_string(bits_per_pixel); // YUV420P16
    }

    int new_pixel_type = GetPixelTypeFromName(format.c_str());
    if (new_pixel_type == VideoInfo::CS_UNKNOWN) {
      new_pixel_type = GetPixelTypeFromName(pixel_type);
      if (new_pixel_type == VideoInfo::CS_UNKNOWN)
        env->ThrowError("%s: invalid pixel_type!", myname);
    }
    // new output format
    vi.pixel_type = new_pixel_type;
  }

  if (vi.BitsPerComponent() != bits_per_pixel)
    env->ThrowError("%s: target bit depth (%d) must match with sources (%d)", myname, vi.BitsPerComponent(), bits_per_pixel);

  if (!vi.IsRGB())
    env->ThrowError("%s: target format must be an RGB format", myname);

  if(alpha && vi.NumComponents() != 4)
    env->ThrowError("MergeARGB: target format must have an alpha channel");

  // When not in ARGB mode, target is still allowed to have an alpha channel.
  // If no alpha source is given, target alpha will be filled by default 0 value.

  if (alpha && (viA.IsRGB24() || viA.IsRGB48() || viA.IsPlanarRGB()))
    env->ThrowError("MergeARGB: Alpha source channel cannot be obtained from RGB24, RGB48 or alphaless planar RGB");
}


PVideoFrame MergeRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame B = blue->GetFrame(n, env);
  PVideoFrame G = green->GetFrame(n, env);
  PVideoFrame R = red->GetFrame(n, env);
  PVideoFrame A = (alpha) ? alpha->GetFrame(n, env) : 0;

  // choose one for frame property source: R
  PVideoFrame dst = env->NewVideoFrameP(vi, &R);

  auto props = env->getFramePropsRW(dst);
  if (!viR.IsRGB()) {
    // delete _Matrix and ChromaLocation when source is not RGB
    env->propDeleteKey(props, "_Matrix");
    env->propDeleteKey(props, "_ChromaLocation");
  }

  const int pixelsize = viR.ComponentSize();

  if (!vi.IsPlanar()) {
    // target is packed RGB
    const int height = dst->GetHeight();
    const int pitch = dst->GetPitch();
    const int rowsize = dst->GetRowSize();
    const int modulo = pitch - rowsize;

    BYTE* dstp = dst->GetWritePtr();


    int planeB = viB.IsPlanar() && viB.IsRGB() ? PLANAR_B : vi.IsRGB() ? 0 : PLANAR_Y;
    int planeG = viG.IsPlanar() && viG.IsRGB() ? PLANAR_G : vi.IsRGB() ? 0 : PLANAR_Y;
    int planeR = viR.IsPlanar() && viR.IsRGB() ? PLANAR_R : vi.IsRGB() ? 0 : PLANAR_Y;
    int planeA = viA.IsPlanar() && viA.IsRGB() ? PLANAR_A : vi.IsRGB() ? 0 : PLANAR_Y;

    // RGB is upside-down, backscan any Planar to match
    const int Bpitch = (viB.IsPlanar()) ? -(B->GetPitch(planeB)) : B->GetPitch();
    const int Gpitch = (viG.IsPlanar()) ? -(G->GetPitch(planeG)) : G->GetPitch();
    const int Rpitch = (viR.IsPlanar()) ? -(R->GetPitch(planeR)) : R->GetPitch();

    // Bump any RGB channels, move any YUV channels to last line
    const BYTE* Bp = B->GetReadPtr(planeB) + (viB.IsPlanar() ? Bpitch * (1 - height) : 0);
    const BYTE* Gp = G->GetReadPtr(planeG) + (viG.IsPlanar() ? Gpitch * (1 - height) : (1 * pixelsize));
    const BYTE* Rp = R->GetReadPtr(planeR) + (viR.IsPlanar() ? Rpitch * (1 - height) : (2 * pixelsize));

    // Adjustment from the end of 1 line to the start of the next
    const int Bmodulo = Bpitch - B->GetRowSize(planeB);
    const int Gmodulo = Gpitch - G->GetRowSize(planeG);
    const int Rmodulo = Rpitch - R->GetRowSize(planeR);

    // Number of bytes per pixel (1, 2, 3 or 4 .. 8)
    const int Bstride = viB.IsPlanar() ? pixelsize : (viB.BitsPerPixel() >> 3);
    const int Gstride = viG.IsPlanar() ? pixelsize : (viG.BitsPerPixel() >> 3);
    const int Rstride = viR.IsPlanar() ? pixelsize : (viR.BitsPerPixel() >> 3);

    // End of VFB
    BYTE const* yend = dstp + pitch * height;

    if (alpha) { // ARGB mode
      const int Apitch = (viA.IsPlanar()) ? -(A->GetPitch(planeA)) : A->GetPitch();
      const BYTE* Ap = A->GetReadPtr(planeA) + (viA.IsPlanar() ? Apitch * (1 - height) : (3 * pixelsize));
      const int Amodulo = Apitch - A->GetRowSize(planeA);
      const int Astride = viA.IsPlanar() ? pixelsize : (viA.BitsPerPixel() >> 3);

      switch (pixelsize) {
      case 1:
        while (dstp < yend) {
          BYTE const* xend = dstp + rowsize;
          while (dstp < xend) {
            *dstp++ = *Bp; Bp += Bstride;
            *dstp++ = *Gp; Gp += Gstride;
            *dstp++ = *Rp; Rp += Rstride;
            *dstp++ = *Ap; Ap += Astride;
          }
          dstp += modulo;
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
          Ap += Amodulo;
        }
        break;
      case 2:
      {
        uint16_t* dstp16 = reinterpret_cast<uint16_t*>(dstp);
        uint16_t const* yend16 = dstp16 + pitch * height / sizeof(uint16_t);
        while (dstp16 < yend16) {
          uint16_t const* xend16 = dstp16 + rowsize / sizeof(uint16_t);
          while (dstp16 < xend16) {
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Bp); Bp += Bstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Gp); Gp += Gstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Rp); Rp += Rstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Ap); Ap += Astride;
          }
          dstp16 += modulo / sizeof(uint16_t);
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
          Ap += Amodulo;
        }
      }
      break;
      default:
        env->ThrowError("%s: float pixel type not supported", myname);
        break;
      }
    }
    else if (vi.pixel_type == VideoInfo::CS_BGR32 || vi.pixel_type == VideoInfo::CS_BGR64) { // RGB32 mode
      switch (pixelsize) {
      case 1:
        while (dstp < yend) {
          BYTE const* xend = dstp + rowsize;
          while (dstp < xend) {
            *dstp++ = *Bp; Bp += Bstride;
            *dstp++ = *Gp; Gp += Gstride;
            *dstp++ = *Rp; Rp += Rstride;
            *dstp++ = 0; // default Alpha is 0!
          }
          dstp += modulo;
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
        }
        break;
      case 2:
      {
        uint16_t* dstp16 = reinterpret_cast<uint16_t*>(dstp);
        uint16_t const* yend16 = dstp16 + pitch * height / sizeof(uint16_t);
        while (dstp16 < yend16) {
          uint16_t const* xend16 = dstp16 + rowsize / sizeof(uint16_t);
          while (dstp16 < xend16) {
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Bp); Bp += Bstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Gp); Gp += Gstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Rp); Rp += Rstride;
            *dstp16++ = 0; // default Alpha is 0!
          }
          dstp16 += modulo / sizeof(uint16_t);
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
        }
      }
      break;
      default:
        env->ThrowError("%s: float pixel type not supported", myname);
        break;
      }
    }
    else if (vi.pixel_type == VideoInfo::CS_BGR24 || vi.pixel_type == VideoInfo::CS_BGR48) { // RGB24 mode
      switch (pixelsize) {
      case 1:
        while (dstp < yend) {
          BYTE const* xend = dstp + rowsize;
          while (dstp < xend) {
            *dstp++ = *Bp; Bp += Bstride;
            *dstp++ = *Gp; Gp += Gstride;
            *dstp++ = *Rp; Rp += Rstride;
          }
          dstp += modulo;
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
        }
        break;
      case 2:
      {
        uint16_t* dstp16 = reinterpret_cast<uint16_t*>(dstp);
        uint16_t const* yend16 = dstp16 + pitch * height / sizeof(uint16_t);
        while (dstp16 < yend16) {
          uint16_t const* xend16 = dstp16 + rowsize / sizeof(uint16_t);
          while (dstp16 < xend16) {
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Bp); Bp += Bstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Gp); Gp += Gstride;
            *dstp16++ = *reinterpret_cast<const uint16_t*>(Rp); Rp += Rstride;
          }
          dstp16 += modulo / sizeof(uint16_t);
          Bp += Bmodulo;
          Gp += Gmodulo;
          Rp += Rmodulo;
        }
      }
      break;
      default:
        env->ThrowError("%s: float pixel type not supported", myname);
        break;
      }
    }
    else
      env->ThrowError("%s: unexpected end of function", myname);

    return dst;
  }

  // target is planar RGB
  for (int p = 0; p < vi.NumComponents(); p++) {
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int plane = planes_r[p];

    const VideoInfo& vi_src =
      plane == PLANAR_G ? viG :
      plane == PLANAR_B ? viB :
      plane == PLANAR_R ? viR : viA;

    PVideoFrame& src =
      plane == PLANAR_G ? G :
      plane == PLANAR_B ? B :
      plane == PLANAR_R ? R : A;

    // actual plane source is not a packed RGB type

    if (p == 3 && !alpha) {
      // fill alpha, but not ARGB mode

      const int dst_rowsizeA = dst->GetRowSize(PLANAR_A);
      const int dst_pitchA = dst->GetPitch(PLANAR_A);
      BYTE* dstp_a = dst->GetWritePtr(PLANAR_A);
      const int heightA = dst->GetHeight(PLANAR_A);
      // unlike in other filters, default alpha value here is 0 instead of max transparent 255
      switch (vi.ComponentSize())
      {
      case 1:
        fill_plane<uint8_t>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, 0);
        break;
      case 2:
        fill_plane<uint16_t>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, 0);
        break;
      case 4:
        fill_plane<float>(dstp_a, heightA, dst_rowsizeA, dst_pitchA, 0.0f);
        break;
      }
    }
    else {
      if (vi_src.IsPlanar())
      {
        // plane copy
        const int plane_src = vi_src.IsRGB() ? plane : PLANAR_Y;
        env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane_src),
          src->GetPitch(plane_src), src->GetRowSize(plane_src), src->GetHeight(plane_src));
      }
      else {
        // fill a plane from packed RGB
        // packed RGB -> Y, YUV(A)
        BYTE* dstp = dst->GetWritePtr(plane);
        int dstpitch = dst->GetPitch(plane);


        const BYTE* srcp = src->GetReadPtr();
        const int pitch = src->GetPitch();
        const int packed_rgb_channel =
          plane == PLANAR_G ? 1 :
          plane == PLANAR_B ? 0 :
          plane == PLANAR_R ? 2 : 3;

        if (pixelsize == 1) {
          if (vi_src.NumComponents() == 3)
            packed_to_luma_alpha<uint8_t, false, false>(dstp, nullptr, dstpitch, srcp, pitch, vi.width, vi.height, packed_rgb_channel);
          else
            packed_to_luma_alpha<uint8_t, true, false>(dstp, nullptr, dstpitch, srcp, pitch, vi.width, vi.height, packed_rgb_channel);
        }
        else {
          if (vi_src.NumComponents() == 3)
            packed_to_luma_alpha<uint16_t, false, false>(dstp, nullptr, dstpitch, srcp, pitch, vi.width, vi.height, packed_rgb_channel);
          else
            packed_to_luma_alpha<uint16_t, true, false>(dstp, nullptr, dstpitch, srcp, pitch, vi.width, vi.height, packed_rgb_channel);
        }
      }
    }
  }
  return dst;
}


AVSValue MergeRGB::Create(AVSValue args, void* mode, IScriptEnvironment* env)
{
  if (mode) // ARGB
    return new MergeRGB(args[0].AsClip(), args[3].AsClip(), args[2].AsClip(), args[1].AsClip(), args[0].AsClip(), args[4].AsString(""), env);
  else      // RGB[type]
    return new MergeRGB(args[0].AsClip(), args[2].AsClip(), args[1].AsClip(), args[0].AsClip(), 0, args[3].AsString(""), env);
}


/*******************************
 *******   Layer Filter   ******
 *******************************/

// Packed RGB (RGB24/32/48/64) and YUY2 are pre-converted in Create; this constructor
// receives only planar formats.  Post-conversion back to the original format is done in Create.
Layer::Layer(PClip _child1, PClip _child2, PClip _mask_child, const char _op[], int _lev, int _x, int _y,
  int _t, bool _chroma, float _opacity, int _placement, IScriptEnvironment* env)
  : child1(_child1), child2(_child2), mask_child(_mask_child), Op(_op), levelB(_lev), ofsX(_x), ofsY(_y),
  chroma(_chroma), opacity(_opacity), placement(_placement)
{
  const VideoInfo& vi1 = child1->GetVideoInfo();
  const VideoInfo& vi2 = child2->GetVideoInfo();

  // PlanarRGB base + PlanarRGBA overlay is valid: overlay A is the blend weight, base has no alpha.
  // All other format mismatches (including bit-depth) remain errors.
  const bool rgb_alpha_mismatch =
    (vi1.IsPlanarRGB() && vi2.IsPlanarRGBA()) && vi1.BitsPerComponent() == vi2.BitsPerComponent();
  if (vi1.pixel_type != vi2.pixel_type && !vi1.IsSameColorspace(vi2) && !rgb_alpha_mismatch) // i420 and YV12 are matched OK
    env->ThrowError("Layer: image formats don't match");

  vi = vi1;

  // hasAlpha: overlay supplies a per-pixel blend weight via its alpha plane.
  // Derived from overlay (vi2): it is the overlay's alpha that acts as the mask.
  // After pre-conversion in Create (when mode is non-"Add"/"Subtract", packed RGB32/64
  // normally appear here as PlanarRGBA (non-native path).
  // For the native packed path (Add/Subtract with use_chroma=true),
  // vi2 stays as RGB32, we still fill hasAlpha for the level->opacity calc;
  // however the packed RGB32 (kept for speed reasons) blend kernel reads alpha directly.
  // RGB32 'add' + chroma=true uses magicdiv and opacity in their calc
  // Or from a separate mask_child clip.
  hasAlpha = vi2.IsYUVA() || vi2.IsPlanarRGBA() || vi2.IsRGB32() || vi2.IsRGB64();

  // process_alpha_channel: both clips have an alpha plane → blend A exactly like the colour
  // channels (same op formula).  This reproduces the historical RGB32 behaviour where the
  // MMX code applied the blend uniformly across all four packed bytes including alpha.
  // Used for planar rgb part, legacy packed rgb32 is working like this by default.
  process_alpha_channel = hasAlpha && (vi1.IsYUVA() || vi1.IsPlanarRGBA());
  bits_per_pixel = vi.BitsPerComponent();

  const bool levelSpecified = levelB >= 0;
  const bool strengthSpecified = opacity >= 0.0f;

  if (levelSpecified && strengthSpecified)
    env->ThrowError("Layer: cannot specify both level and opacity");
  if (levelSpecified && bits_per_pixel == 32)
    env->ThrowError("Layer: cannot specify level for 32 bit float format");

  if (levelSpecified)
  {
    if (hasAlpha)
      opacity = (float)levelB / ((1 << bits_per_pixel) + 1); // gives 1.0f for 257 (@8bit) and 65537 (@16 bits)
      // originally levelB was used in formula: (alpha*level + 1) / range_size,
      // now level is calculated from opacity as: level = opacity * ((1 << bits_per_pixel) + 1)
    else
      opacity = (float)levelB / ((1 << bits_per_pixel)); // YUY2 or other non-Alpha, gives 1.0f for 256 (@8bit)
    // we'll calculate back the level as: level = opacity * ((1 << bits_per_pixel))
  }
  else if (!strengthSpecified)
    opacity = 1.0f;

  if (opacity < 0.0f) opacity = 0.0f;
  if (opacity > 1.0f) opacity = 1.0f;

  constexpr bool snap_offsets_to_chroma = false;
  // Can be turned on for legacy subsampling-friendly behaviour (pre-3.7.6 Layer snapped the offsets)
  // Chroma width calculation is already made ready to accept odd positions.
  // "Overlay" does not snap.
  if (snap_offsets_to_chroma) {
    if ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) {
      // make offsets subsampling friendly
      ofsX = ofsX & ~((1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1);
      ofsY = ofsY & ~((1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1);
    }
  }
  // For the sake of completeness, packed rgb formats still exist, but for most modes they are converted to planar RGB.
  if (vi.IsRGB32() || vi.IsRGB64() || vi.IsRGB24() || vi.IsRGB48())
    ofsY = vi.height - vi2.height - ofsY; // packed RGB is upside down

  xdest = (ofsX < 0) ? 0 : ofsX;
  ydest = (ofsY < 0) ? 0 : ofsY;

  xsrc = (ofsX < 0) ? (0 - ofsX) : 0;
  ysrc = (ofsY < 0) ? (0 - ofsY) : 0;

  xcount = (vi.width < (ofsX + vi2.width)) ? (vi.width - xdest) : (vi2.width - xsrc);
  ycount = (vi.height < (ofsY + vi2.height)) ? (vi.height - ydest) : (vi2.height - ysrc);

  if (!(!lstrcmpi(Op, "Mul") || !lstrcmpi(Op, "Add") || !lstrcmpi(Op, "Fast") ||
    !lstrcmpi(Op, "Subtract") || !lstrcmpi(Op, "Lighten") || !lstrcmpi(Op, "Darken") ||
    !lstrcmpi(Op, "mulovr")))
    env->ThrowError("Layer supports the following ops: Fast, Lighten, Darken, Add, Subtract, Mul, mulovr");

  if (!lstrcmpi(Op, "mulovr") && !vi.IsYUV() && !vi.IsYUVA())
    env->ThrowError("Layer mulovr: YUV(A) formats only");

  if (!chroma)
  {
    if (!lstrcmpi(Op, "Darken")) env->ThrowError("Layer: monochrome darken illegal op");
    if (!lstrcmpi(Op, "Lighten")) env->ThrowError("Layer: monochrome lighten illegal op");
    if (!lstrcmpi(Op, "Fast")) env->ThrowError("Layer: this mode not allowed in FAST; use ADD instead");
  }

  // autoscale ThresholdParam from 8 bit base
  // todo check validity
  if (bits_per_pixel == 32)
    ThresholdParam = _t; // n/a
  else
    ThresholdParam = _t << (bits_per_pixel - 8);
  ThresholdParam_f = _t / 255.0f;

  overlay_frames = vi2.num_frames;
}


// simple averaging
template<typename pixel_t>
static void layer_genericplane_fast_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  AVS_UNUSED(level);
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dstp[x] = (dstp[x] + ovrp[x] + 1) / 2;
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

static void layer_genericplane_fast_f_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, float level) {
  AVS_UNUSED(level);
  float* dstp = reinterpret_cast<float*>(dstp8);
  const float* ovrp = reinterpret_cast<const float*>(ovrp8);
  dst_pitch /= sizeof(float);
  overlay_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      dstp[x] = (dstp[x] + ovrp[x]) * 0.5f;
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

/* RGB32 */

// For Full Strength: 8 bit Level must be 257, 16 bit must be 65537!
// in 8 bit:   (255*257+1)/256 = (65535+1)/256 = 256 -> alpha_max = 256
// in 16 bit:  (65535*65537+1)/65536 = 65536, x=? 7FFFFFFF, x=65537 -> alpha_max = 65536

template<typename pixel_t>
static void layer_rgb32_mul_chroma_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;

      dstp[x * 4 + 0] = (pixel_t)(dstp[x * 4 + 0] + ((((((calc_t)ovrp[x * 4 + 0] * dstp[x * 4 + 0]) >> SHIFT) - dstp[x * 4 + 0]) * alpha) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + ((((((calc_t)ovrp[x * 4 + 1] * dstp[x * 4 + 1]) >> SHIFT) - dstp[x * 4 + 1]) * alpha) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + ((((((calc_t)ovrp[x * 4 + 2] * dstp[x * 4 + 2]) >> SHIFT) - dstp[x * 4 + 2]) * alpha) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + ((((((calc_t)ovrp[x * 4 + 3] * dstp[x * 4 + 3]) >> SHIFT) - dstp[x * 4 + 3]) * alpha) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

template<typename pixel_t>
static void layer_rgb32_mul_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;
      calc_t luma = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;

      dstp[x * 4 + 0] = (pixel_t)(dstp[x * 4 + 0] + (((((luma * dstp[x * 4 + 0]) >> SHIFT) - dstp[x * 4 + 0]) * alpha) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + (((((luma * dstp[x * 4 + 1]) >> SHIFT) - dstp[x * 4 + 1]) * alpha) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + (((((luma * dstp[x * 4 + 2]) >> SHIFT) - dstp[x * 4 + 2]) * alpha) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + (((((luma * dstp[x * 4 + 3]) >> SHIFT) - dstp[x * 4 + 3]) * alpha) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}


template<typename pixel_t>
static void layer_rgb32_add_chroma_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  constexpr int rounder = sizeof(pixel_t) == 1 ? 128 : 32768;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;

      dstp[x * 4] = (pixel_t)(dstp[x * 4] + ((((calc_t)ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + ((((calc_t)ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + ((((calc_t)ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + ((((calc_t)ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

template<typename pixel_t>
static void layer_rgb32_add_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  constexpr int rounder = sizeof(pixel_t) == 1 ? 128 : 32768;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;
      calc_t luma = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;

      dstp[x * 4] = (pixel_t)(dstp[x * 4] + (((luma - dstp[x * 4]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + (((luma - dstp[x * 4 + 1]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + (((luma - dstp[x * 4 + 2]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + (((luma - dstp[x * 4 + 3]) * alpha + rounder) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}



template<typename pixel_t>
static void layer_rgb32_fast_c(BYTE* dstp, const BYTE* ovrp, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  layer_genericplane_fast_c<pixel_t>(dstp, ovrp, dst_pitch, overlay_pitch, width * 4, height, level);
}


template<typename pixel_t>
static void layer_rgb32_subtract_chroma_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  const calc_t MAX_PIXEL_VALUE = sizeof(pixel_t) == 1 ? 255 : 65535;
  constexpr int rounder = sizeof(pixel_t) == 1 ? 128 : 32768;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;

      dstp[x * 4] = (pixel_t)(dstp[x * 4] + (((MAX_PIXEL_VALUE - ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + (((MAX_PIXEL_VALUE - ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + (((MAX_PIXEL_VALUE - ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + (((MAX_PIXEL_VALUE - ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}

template<typename pixel_t>
static void layer_rgb32_subtract_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  const calc_t MAX_PIXEL_VALUE = sizeof(pixel_t) == 1 ? 255 : 65535;
  constexpr int rounder = sizeof(pixel_t) == 1 ? 128 : 32768;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;
      calc_t luma = (cyb * (MAX_PIXEL_VALUE - ovrp[x * 4]) + cyg * (MAX_PIXEL_VALUE - ovrp[x * 4 + 1]) + cyr * (MAX_PIXEL_VALUE - ovrp[x * 4 + 2])) >> 15;

      dstp[x * 4] = (pixel_t)(dstp[x * 4] + (((luma - dstp[x * 4]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + (((luma - dstp[x * 4 + 1]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + (((luma - dstp[x * 4 + 2]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + (((luma - dstp[x * 4 + 3]) * alpha + rounder) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}


template<int mode, typename pixel_t>
static void layer_rgb32_lighten_darken_c(BYTE* dstp8, const BYTE* ovrp8, int dst_pitch, int overlay_pitch, int width, int height, int level, int thresh) {
  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* ovrp = reinterpret_cast<const pixel_t*>(ovrp8);
  dst_pitch /= sizeof(pixel_t);
  overlay_pitch /= sizeof(pixel_t);
  const int SHIFT = sizeof(pixel_t) == 1 ? 8 : 16;

  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type calc_t;

  constexpr int rounder = sizeof(pixel_t) == 1 ? 128 : 32768;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      calc_t alpha = ((calc_t)ovrp[x * 4 + 3] * level + 1) >> SHIFT;
      int luma_ovr = (cyb * ovrp[x * 4] + cyg * ovrp[x * 4 + 1] + cyr * ovrp[x * 4 + 2]) >> 15;
      int luma_src = (cyb * dstp[x * 4] + cyg * dstp[x * 4 + 1] + cyr * dstp[x * 4 + 2]) >> 15;

      if constexpr (mode == LIGHTEN)
        alpha = luma_ovr > luma_src + thresh ? alpha : 0;
      else // DARKEN
        alpha = luma_ovr < luma_src - thresh ? alpha : 0;

      dstp[x * 4] = (pixel_t)(dstp[x * 4] + ((((calc_t)ovrp[x * 4] - dstp[x * 4]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 1] = (pixel_t)(dstp[x * 4 + 1] + ((((calc_t)ovrp[x * 4 + 1] - dstp[x * 4 + 1]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 2] = (pixel_t)(dstp[x * 4 + 2] + ((((calc_t)ovrp[x * 4 + 2] - dstp[x * 4 + 2]) * alpha + rounder) >> SHIFT));
      dstp[x * 4 + 3] = (pixel_t)(dstp[x * 4 + 3] + ((((calc_t)ovrp[x * 4 + 3] - dstp[x * 4 + 3]) * alpha + rounder) >> SHIFT));
    }
    dstp += dst_pitch;
    ovrp += overlay_pitch;
  }
}


// chroma placement to mask helpers
// yuv add, subtract, mul, lighten, darken
// included in base and the avx2 source module, to get different optimizations
#include "layer.hpp"



PVideoFrame __stdcall Layer::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src1 = child1->GetFrame(n, env);

  if (xcount <= 0 || ycount <= 0) return src1;

  PVideoFrame src2 = child2->GetFrame(std::min(n, overlay_frames - 1), env);
  
  env->MakeWritable(&src1);

  const int mylevel = hasAlpha ? (int)(opacity * ((1 << bits_per_pixel) + 1) + 0.5f) : (int)(opacity * (1 << bits_per_pixel) + 0.5f);

  // For functions using magic-div (/max_val) arithmetic,
  // opacity_i in [0..max_pixel_value], same convention as masked_merge.
  const int opacity_i = (int)(opacity * ((1 << bits_per_pixel) - 1) + 0.5f);

  // The inner-loop functions that consume mylevel use a power-of-two shift (>> bpp) for the
  // blend, not division by max_val.  With a ÷(max_val+1) = ÷256 (@8bit) divisor, a weight
  // of max_val (255) maps to 255/256 — never to 1.0 — so a mask/alpha pixel of 255 cannot
  // reach the maximum output value (gives 254 instead of 255).
  // The inflated level compensates: the two-step product alpha*level can reach the next
  // power-of-two (256 @8bit, 65536 @16bit), restoring exact values at both extremes.
  // This fixes both endpoints but leaves intermediate weights slightly asymmetric because
  // 255 equal mask steps are mapped through a 256-wide divisor.
  // The masked_merge family avoids this by dividing by max_val directly (see blend_common.h).
  //   Alpha mode:     opacity 1.0 → 257 (@8bit), 65537 (@16bit): alpha*257>>8 reaches 256 when alpha=255
  //   Non-alpha mode: opacity 1.0 → 256 (@8bit): direct flat weight, >> 8 at full weight gives src

  const int pixelsize = vi.ComponentSize();

  const int height = ycount; // these may be divided by subsampling factor
  const int width = xcount;

  if (vi.IsRGB32() || vi.IsRGB64())
  {
    const int src1_pitch = src1->GetPitch();
    const int src2_pitch = src2->GetPitch();
    BYTE* src1p = src1->GetWritePtr();
    const BYTE* src2p = src2->GetReadPtr();

    int rgb_step = vi.BytesFromPixels(1); // 4 or 8 bytes/pixelgroup

    src1p += (src1_pitch * ydest) + (xdest * rgb_step);
    src2p += (src2_pitch * ysrc) + (xsrc * rgb_step);

    // note: ThresholdParam is not scaled automatically
    int thresh = std::max(0, std::min(ThresholdParam, (1 << bits_per_pixel) - 1)); // limit threshold, old method was: & 0xFF

    // only "Add" and "Subtract" is live code for RGB32/64, others are preconverted to planar RGBA in Create.
    // Kept for reference.

    // packed rgb "Mul": dead code, kept for reference
    if (!lstrcmpi(Op, "Mul"))
    {
      if (chroma)
      {
#ifdef INTEL_INTRINSICS
        if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
        {
          layer_rgb32_mul_avx2<true>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
        {
          layer_rgb32_mul_sse2<true>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else
#endif
        {
          if (pixelsize == 1)
            layer_rgb32_mul_chroma_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
          else
            layer_rgb32_mul_chroma_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
      }
      else // Mul, chroma==false
      {
#ifdef INTEL_INTRINSICS
        if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
        {
          layer_rgb32_mul_avx2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
        {
          layer_rgb32_mul_sse2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else
#endif
        {
          if (pixelsize == 1)
            layer_rgb32_mul_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
          else
            layer_rgb32_mul_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
      }
    }
    if (!lstrcmpi(Op, "Add"))
    {
      if (chroma)
      {
        // rgb32 packed rgb "Add" + chroma=true: NOT dead code, kept for speed, too many scripts are using

        // Packed RGBA Add/chroma (and pre-inverted Subtract/chroma):
        // overlay alpha acts as per-pixel blend weight, or a separate mask
        // clip supplies the original alpha when the overlay was pre-inverted.
        // Uses magic-div (÷max_val) arithmetic

        // mask_child is set for pre-inverted Subtract; nullptr for plain Add.
        // The mask clip (ExtractA of original overlay) has 1 channel with 1 byte/pixel
        // for RGB32 and 2 bytes/pixel for RGB64; offset by (ysrc, xsrc) as the overlay.
        const BYTE* maskp8   = nullptr;
        int         maskpitch = 0;
        PVideoFrame mask_frame;
        if (mask_child) {
          mask_frame = mask_child->GetFrame(std::min(n, overlay_frames - 1), env);
          maskpitch  = mask_frame->GetPitch();
          maskp8     = mask_frame->GetReadPtr() + maskpitch * ysrc + xsrc * pixelsize;
        }
        const bool has_separate_mask = (mask_child != nullptr);

        layer_packedrgb_blend_c_t* blend_fn;
#ifdef INTEL_INTRINSICS
        // for 8+ bits these may return the C reference, but compiled with avx2/sse4.1 arch.
        if ((env->GetCPUFlags() & CPUF_AVX2))
          get_layer_packedrgb_blend_functions_avx2(has_separate_mask, bits_per_pixel, &blend_fn);
        else if ((env->GetCPUFlags() & CPUF_SSE4_1))
          get_layer_packedrgb_blend_functions_sse41(has_separate_mask, bits_per_pixel, &blend_fn);
        else
#endif
          get_layer_packedrgb_blend_functions(has_separate_mask, bits_per_pixel, &blend_fn);
        blend_fn(src1p, src2p, maskp8, src1_pitch, src2_pitch, maskpitch, width, height, opacity_i);
      }
      else // Add, chroma == false, Special "Layer" case, not a simple MaskedMerge.
      {
#ifdef INTEL_INTRINSICS
        if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
        {
          layer_rgb32_add_avx2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
        {
          layer_rgb32_add_sse2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
        else
#endif
        {
          if (pixelsize == 1)
            layer_rgb32_add_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
          else
            layer_rgb32_add_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        }
      }
    }
    // packed rgb "Mul": dead code, kept for reference
    if (!lstrcmpi(Op, "Lighten"))
    {
      // Copy overlay_clip over base_clip in areas where overlay_clip is lighter by threshold.
      // only chroma == true
#ifdef INTEL_INTRINSICS
      if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
      {
        layer_rgb32_lighten_darken_avx2<LIGHTEN>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
      else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
      {
        layer_rgb32_lighten_darken_sse2<LIGHTEN>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
      else
#endif 
      {
        if (pixelsize == 1)
          layer_rgb32_lighten_darken_c<LIGHTEN, uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
        else
          layer_rgb32_lighten_darken_c<LIGHTEN, uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
    }
    // packed rgb "Mul": dead code, kept for reference
    if (!lstrcmpi(Op, "Darken"))
    {
      // Copy overlay_clip over base_clip in areas where overlay_clip is darker by threshold.
      // only chroma == true
#ifdef INTEL_INTRINSICS
      if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
      {
        layer_rgb32_lighten_darken_avx2<DARKEN>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
      else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
      {
        layer_rgb32_lighten_darken_sse2<DARKEN>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
      else
#endif
      {
        if (pixelsize == 1)
          layer_rgb32_lighten_darken_c<DARKEN, uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
        else
          layer_rgb32_lighten_darken_c<DARKEN, uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel, thresh);
      }
    }
    if (!lstrcmpi(Op, "Fast"))
    {
      // Like add, but without masking.
      // use_chroma must be true; level and threshold are not used.
      // The result is simply the average of base_clip and overlay_clip.
      // only chroma == true
#ifdef INTEL_INTRINSICS
      // yes, alignment check, we can have x offsets
      // But this avx2 does not require it
      if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
      {
        layer_rgb32_fast_avx2(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
      else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(src1p, 16) && IsPtrAligned(src2p, 16))
      {
        layer_rgb32_fast_sse2(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
      else
#endif
      {
        if (pixelsize == 1)
          layer_rgb32_fast_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        else // rgb64
          layer_rgb32_fast_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
    }
    if (!lstrcmpi(Op, "Subtract"))
    {
      // use_chroma=true Subtract was redirected to Add in Create() with pre-inverted
      // overlay and a separate mask clip, so only use_chroma=false reaches here.
      // use_chroma=false: flat-weight subtract without alpha mask.
#ifdef INTEL_INTRINSICS
      if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_AVX2))
      {
        layer_rgb32_subtract_avx2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
      else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2))
      {
        layer_rgb32_subtract_sse2<false>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
      else
#endif
      {
        if (pixelsize == 1)
          layer_rgb32_subtract_c<uint8_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
        else
          layer_rgb32_subtract_c<uint16_t>(src1p, src2p, src1_pitch, src2_pitch, width, height, mylevel);
      }
    }
  }
  else if (vi.IsYUV() || vi.IsYUVA())
  {
    // planar YUV(A) source

    if (!lstrcmpi(Op, "Lighten") || !lstrcmpi(Op, "Darken"))
    {
      const bool isLighten = !lstrcmpi(Op, "Lighten");
      // only chroma == true

      const int src1_pitch = src1->GetPitch(PLANAR_Y);
      const int src2_pitch = src2->GetPitch(PLANAR_Y);
      const int src1_pitchUV = src1->GetPitch(PLANAR_U);
      const int src2_pitchUV = src2->GetPitch(PLANAR_U);

      const bool grey = vi.IsY();

      const int ws = grey ? 1 : vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int hs = grey ? 1 : vi.GetPlaneHeightSubsampling(PLANAR_U);

      BYTE* src1p = src1->GetWritePtr(PLANAR_Y) + src1_pitch * (ydest)+(xdest)*pixelsize; // in-place source and destination
      const BYTE* src2p = src2->GetReadPtr(PLANAR_Y) + src2_pitch * (ysrc)+(xsrc)*pixelsize; // overlay

      BYTE* src1p_u = grey ? nullptr : src1->GetWritePtr(PLANAR_U) + src1_pitchUV * (ydest >> hs) + (xdest >> ws) * pixelsize;
      const BYTE* src2p_u = grey ? nullptr : src2->GetReadPtr(PLANAR_U) + src2_pitchUV * (ysrc >> hs) + (xsrc >> ws) * pixelsize;

      BYTE* src1p_v = grey ? nullptr : src1->GetWritePtr(PLANAR_V) + src1_pitchUV * (ydest >> hs) + (xdest >> ws) * pixelsize;
      const BYTE* src2p_v = grey ? nullptr : src2->GetReadPtr(PLANAR_V) + src2_pitchUV * (ysrc >> hs) + (xsrc >> ws) * pixelsize;

      // target alpha channel is unaffected
      // Until we support providing a separate mask clip, we pass A plane pointer for overlay alpha
      const BYTE* maskp = hasAlpha ? src2->GetReadPtr(PLANAR_A) + src2_pitch * ysrc + xsrc * pixelsize : nullptr; // overlay alpha
      const int mask_pitch = src2->GetPitch(PLANAR_A);

      // called only once, for all planes

      layer_yuv_lighten_darken_c_t* layer_fn = nullptr;
      layer_yuv_lighten_darken_f_c_t* layer_f_fn = nullptr;

      // fills layer_yuv_lighten_darken_c_t or layer_yuv_lighten_darken_f_c_t depending on video format, and bits_per_pixel.
      get_layer_yuv_lighten_darken_functions(isLighten, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);

      // ThresholdParam was scaled from the 8 bit world
      if (layer_fn && bits_per_pixel != 32) {
        layer_fn(
          src1p, src1p_u, src1p_v,
          src2p, src2p_u, src2p_v, maskp,
          src1_pitch, src1_pitchUV,
          src2_pitch, src2_pitchUV,
          mask_pitch,
          width, height, opacity_i, ThresholdParam, bits_per_pixel);
      }
      else if (layer_f_fn && bits_per_pixel == 32) {
        layer_f_fn(
          src1p, src1p_u, src1p_v,
          src2p, src2p_u, src2p_v, maskp,
          src1_pitch, src1_pitchUV,
          src2_pitch, src2_pitchUV,
          mask_pitch,
          width, height, opacity, ThresholdParam_f);
      }
      // lighten/darken end
    }
    else if (!lstrcmpi(Op, "mulovr"))
    {
      const int src1_pitch   = src1->GetPitch(PLANAR_Y);
      const int src2_pitch   = src2->GetPitch(PLANAR_Y);
      const int src1_pitchUV = src1->GetPitch(PLANAR_U);

      const bool grey = vi.IsY();
      const int ws = grey ? 1 : vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int hs = grey ? 1 : vi.GetPlaneHeightSubsampling(PLANAR_U);

      BYTE*       src1p   = src1->GetWritePtr(PLANAR_Y) + src1_pitch   * ydest + xdest * pixelsize;
      const BYTE* src2p   = src2->GetReadPtr (PLANAR_Y) + src2_pitch   * ysrc  + xsrc  * pixelsize;
      BYTE*       src1p_u = grey ? nullptr : src1->GetWritePtr(PLANAR_U) + src1_pitchUV * (ydest >> hs) + (xdest >> ws) * pixelsize;
      BYTE*       src1p_v = grey ? nullptr : src1->GetWritePtr(PLANAR_V) + src1_pitchUV * (ydest >> hs) + (xdest >> ws) * pixelsize;

      // mulovr uses only overlay Y; overlay UV is ignored.
      // mask: overlay alpha plane (if hasAlpha), luma-resolution, same stride as overlay Y.
      const BYTE* maskp      = hasAlpha ? src2->GetReadPtr(PLANAR_A) + src2_pitch * ysrc + xsrc * pixelsize : nullptr;
      const int   mask_pitch = hasAlpha ? src2->GetPitch(PLANAR_A) : 0;

      layer_yuv_mulovr_c_t*   layer_fn   = nullptr;
      layer_yuv_mulovr_f_c_t* layer_f_fn = nullptr;
      get_layer_yuv_mulovr_functions(hasAlpha, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);

      if (layer_fn)
        layer_fn(src1p, src1p_u, src1p_v, src2p, maskp,
          src1_pitch, src1_pitchUV, src2_pitch, mask_pitch,
          width, height, opacity_i, bits_per_pixel);
      else if (layer_f_fn)
        layer_f_fn(src1p, src1p_u, src1p_v, src2p, maskp,
          src1_pitch, src1_pitchUV, src2_pitch, mask_pitch,
          width, height, opacity);
    }
    else {
      // not Lighten, Darken, or mulovr — process planes individually.
      // Add (Subtract is pre-inverted Add), Fast, Mul will follow
      // intentionally do not process alpha plane here (handled elsewhere)
      // YA is using only PLANAR_Y (index 0)
      const int maxPlanes = vi.IsYA() ? 1 : std::min(vi.NumComponents(), 3);
      const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
      for (int channel = 0; channel < maxPlanes; channel++)
      {
        const int plane = planesYUV[channel];

        const int src1_pitch = src1->GetPitch(plane);
        int src2_pitch = src2->GetPitch(plane); // not const, maybe changed later

        const int ws = vi.GetPlaneWidthSubsampling(plane);
        const int hs = vi.GetPlaneHeightSubsampling(plane);

        // For chroma planes, the processing width/height is reduced by the subsampling factor.
        // old formula: assuming that xdest and ydest was already snapped to the chroma grid
        //const int currentwidth  = width  >> ws;
        //const int currentheight = height >> hs;

        // Ceiling formula: account for odd xdest/ydest so the last partial chroma
        // column/row is included when xdest (or ydest) is misaligned to the chroma grid.
        const int ws_mask = (1 << ws) - 1;
        const int hs_mask = (1 << hs) - 1;
        const int currentwidth = (ws > 0) ? ((xdest & ws_mask) + width + ws_mask) >> ws : width;
        const int currentheight = (hs > 0) ? ((ydest & hs_mask) + height + hs_mask) >> hs : height;

        BYTE* src1p = src1->GetWritePtr(plane) + src1_pitch * (ydest >> hs) + (xdest >> ws) * pixelsize; // destination
        const BYTE* src2p = src2->GetReadPtr(plane) + src2_pitch * (ysrc >> hs) + (xsrc >> ws) * pixelsize; // source plane

        const BYTE* maskp = hasAlpha ? src2->GetReadPtr(PLANAR_A) + src2->GetPitch(PLANAR_A) * (ysrc) + (xsrc) * pixelsize : nullptr; // alpha plane from Overlay
        const int mask_pitch = hasAlpha ? src2->GetPitch(PLANAR_A) : 0;

        const bool is_chroma = plane != PLANAR_Y;

        // Special "Add" and "Mul" use_chroma == false case for chroma planes
        // Overlay chroma data is ignored, blending occurs toward grey (e.g. 128) instead of the chroma source pixel.
        // We can "Fake" a chroma scanline buffer of chroma center ("half") by passing a pointer to a local variable, and ignoring the pitch.
        // This is a special "Layer" mode, if we pass a prepared overlay plane, MaskedMerge or Weighted Merge is transparently using this
        // buffer.
        std::vector <uint8_t> chroma_scanline_buffer; // only used if is_chroma && !chroma
        if (!lstrcmpi(Op, "Add") || !lstrcmpi(Op, "Mul")) {
          if (is_chroma && !chroma) {
            chroma_scanline_buffer.resize(pixelsize * currentwidth);
            if (bits_per_pixel == 8)
              std::fill((uint8_t *)chroma_scanline_buffer.data(), (uint8_t *)chroma_scanline_buffer.data() + chroma_scanline_buffer.size(), 128);
            else if (bits_per_pixel <= 16)
              std::fill((uint16_t *)chroma_scanline_buffer.data(), (uint16_t *)chroma_scanline_buffer.data() + chroma_scanline_buffer.size() / sizeof(uint16_t), 1 << (bits_per_pixel - 1));
            else // float
              std::fill((float*)chroma_scanline_buffer.data(), (float*)chroma_scanline_buffer.data() + chroma_scanline_buffer.size() / sizeof(float), 0.0f); // chroma center: 0.0f
            src2p = chroma_scanline_buffer.data();
            src2_pitch = 0; // pitch is 0, so this prepared scanline is used for every line of the chroma plane.
          }
        }
        // preparation end

        if (!lstrcmpi(Op, "Mul"))
        {
          layer_yuv_mul_c_t* layer_fn = nullptr;
          layer_yuv_mul_f_c_t* layer_f_fn = nullptr;

#ifdef INTEL_INTRINSICS
          if (env->GetCPUFlags() & CPUF_AVX2) {
            // no SIMD yet, wrapper for the avx2 module, which calls its local scoped get_layer_yuv_mul_functions the included layer.hpp
            get_layer_yuv_mul_functions_avx2(is_chroma, hasAlpha, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
          }
          else
#endif
            get_layer_yuv_mul_functions(is_chroma, hasAlpha, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);

          if (layer_fn && bits_per_pixel != 32) {
            layer_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
              currentwidth, currentheight, opacity_i, bits_per_pixel);
          }
          else if (layer_f_fn && bits_per_pixel == 32) {
            layer_f_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
              currentwidth, currentheight, opacity);
          }
        }
        else if (!lstrcmpi(Op, "Add") || !lstrcmpi(Op, "Fast"))
        {
          // Note: "Subtract" is handled by pre-inverting the overlay in Layer::Create
          // for planar YUV(A) formats.  GetFrame never sees Op="Subtract" for these.
          const bool is_fast = !lstrcmpi(Op, "Fast");
          if (is_fast) {
            // "Fast" is weighted merge with exact 0.5 weight, and it is optimized inside the called merge function
            const bool use_padded_width = false; // exact width
            // in merge.cpp/h
            merge_plane(src1p, src2p, src1_pitch, src2_pitch, currentwidth * pixelsize /*src_rowsize*/, currentheight, 0.5f, bits_per_pixel, use_padded_width, env);
          }
          else if (!hasAlpha) {
            // check for weighted merge: this call is automatically dispatches to the proper bit_depth, CPU opt
            const bool use_padded_width = false; // exact width
            // in merge.cpp/h
            merge_plane(src1p, src2p, src1_pitch, src2_pitch, currentwidth * pixelsize /*src_rowsize*/, currentheight, opacity, bits_per_pixel, use_padded_width, env);
          }
          else {
            // masked merge, use the unified masked blend functions
            layer_yuv_add_c_t* layer_fn = nullptr;
            layer_yuv_add_f_c_t* layer_f_fn = nullptr;

#ifdef INTEL_INTRINSICS
            if (env->GetCPUFlags() & CPUF_AVX2) {
              // AVX2 masked merge with chroma placement support
              get_layer_yuv_masked_add_functions_avx2(is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
            }
            else if (env->GetCPUFlags() & CPUF_SSE4_1) {
              // SSE4.1 masked merge with chroma placement support
              get_layer_yuv_masked_add_functions_sse41(is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
            }
            else
#elif defined(NEON_INTRINSICS)
            if (env->GetCPUFlags() & CPUF_ARM_NEON)
              get_layer_yuv_masked_add_functions_neon(is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
            else
#endif
            {
              get_layer_yuv_add_masked_functions(is_chroma, hasAlpha, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
            }

            if (layer_fn && bits_per_pixel != 32) {
              layer_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
                currentwidth, currentheight, opacity_i, bits_per_pixel);
            }
            else if (layer_f_fn && bits_per_pixel == 32) {
              layer_f_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
                currentwidth, currentheight, opacity);
            }
          }
        }
      } // for one channel
    } // if lighten/darken else

  }
  else if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
  {
    BYTE* dstp[4];
    const BYTE* ovrp[4];
    const int dstp_pitch = src1->GetPitch(PLANAR_G); // same for all
    const int ovrp_pitch = src2->GetPitch(PLANAR_G); // same for all

    const int maxPlanes = vi.NumComponents();
    const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    for (int channel = 0; channel < maxPlanes; channel++)
    {
      const int plane = planesRGB[channel];
      dstp[channel] = src1->GetWritePtr(plane) + dstp_pitch * ydest + xdest * pixelsize; // destination
      ovrp[channel] = src2->GetReadPtr(plane) + ovrp_pitch * ysrc + xsrc * pixelsize; // overlay
    }

    // Per-pixel blend weight (maskp): normally the overlay's A plane.
    // For Subtract, mask_child holds ExtractA of the pre-Invert overlay (original A = weight);
    // ovrp[3] then points to the inverted A, used only as the alpha blend target when
    // process_alpha_channel=true.  For all other ops mask_child is nullptr.
    PVideoFrame mask_frame;
    const BYTE* maskp = nullptr;
    int mask_pitch = 0;
    if (hasAlpha) {
      if (mask_child) {
        mask_frame = mask_child->GetFrame(std::min(n, overlay_frames - 1), env);
        maskp = mask_frame->GetReadPtr(PLANAR_Y) + mask_frame->GetPitch(PLANAR_Y) * ysrc + xsrc * pixelsize;
        mask_pitch = mask_frame->GetPitch(PLANAR_Y);
      }
      else {
        // Direct add/mul/lighten/darken: overlay A plane is the blend weight.
        maskp = src2->GetReadPtr(PLANAR_A) + ovrp_pitch * ysrc + xsrc * pixelsize;
        mask_pitch = ovrp_pitch;
      }
    }

    if (!lstrcmpi(Op, "Mul"))
    {
      // called only once, for all planes
      layer_planarrgb_mul_c_t* layer_fn = nullptr;
      layer_planarrgb_mul_f_c_t* layer_f_fn = nullptr;

#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_AVX2) {
        get_layer_planarrgb_mul_functions_avx2(chroma, hasAlpha, process_alpha_channel, bits_per_pixel, &layer_fn, &layer_f_fn);
      }
      else
#endif
      {
        get_layer_planarrgb_mul_functions(chroma, hasAlpha, process_alpha_channel, bits_per_pixel, &layer_fn, &layer_f_fn);
      }

      if (layer_fn && bits_per_pixel != 32)
        layer_fn(dstp, ovrp, maskp, dstp_pitch, ovrp_pitch, mask_pitch, width, height, opacity_i, bits_per_pixel);
      else if (layer_f_fn && bits_per_pixel == 32)
        layer_f_fn(dstp, ovrp, maskp, dstp_pitch, ovrp_pitch, mask_pitch, width, height, opacity);
      // planar_rgba mul end
    }
    else if (!lstrcmpi(Op, "Add") || !lstrcmpi(Op, "Fast"))
    {
      // no subsampling
      // use the unified weigthed blend or masked blend function depending on the presence of alpha,
      // for both Add and Fast (Fast is just a special case of weighted blend with fixed 0.5 weight, and may be optimized internally in the called function)
      const int currentwidth = width;
      const int currentheight = height;
      // Note: "Subtract" is handled by pre-inverting the overlay in Layer::Create
      // for planar YUV(A) formats.  GetFrame never sees Op="Subtract" for these.
      const bool is_fast = !lstrcmpi(Op, "Fast");
      if (is_fast || !hasAlpha) {
        const float actual_opacity = is_fast ? 0.5f : opacity; // "Add" with opacity=0.5 is the same as "Fast", but "Fast" is optimized for this case, and may be faster than a general weighted merge.
        // "Fast" is weighted merge with exact 0.5 weight, and it is optimized inside the called merge function
        const bool use_padded_width = false; // exact width

        for (int channel = 0; channel < maxPlanes; channel++)
        {
          uint8_t *src1p = dstp[channel];
          const uint8_t *src2p = ovrp[channel];
          int src1_pitch = dstp_pitch;
          int src2_pitch = ovrp_pitch;
          // in merge.cpp/h
          merge_plane(src1p, src2p, src1_pitch, src2_pitch, currentwidth * pixelsize /*src_rowsize*/, currentheight, actual_opacity, bits_per_pixel, use_padded_width, env);
        }
      }
      else {
        // masked merge, use the unified masked blend functions
        layer_yuv_add_c_t* layer_fn = nullptr;
        layer_yuv_add_f_c_t* layer_f_fn = nullptr;

        const bool effective_is_chroma = false; // rgb: all planes are treated as luma for blending purposes
        // placement is n/a

#ifdef INTEL_INTRINSICS
        if (env->GetCPUFlags() & CPUF_AVX2) {
          // AVX2 masked merge with chroma placement support
          get_layer_yuv_masked_add_functions_avx2(effective_is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
        }
        else if (env->GetCPUFlags() & CPUF_SSE4_1) {
          // SSE4.1 masked merge with chroma placement support
          get_layer_yuv_masked_add_functions_sse41(effective_is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
        }
        else
#elif defined(NEON_INTRINSICS)
        if (env->GetCPUFlags() & CPUF_ARM_NEON)
          get_layer_yuv_masked_add_functions_neon(effective_is_chroma, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
        else
#endif
        {
          get_layer_yuv_add_masked_functions(effective_is_chroma, hasAlpha, placement, vi, bits_per_pixel, &layer_fn, &layer_f_fn);
        }

        for (int channel = 0; channel < maxPlanes; channel++)
        {
          uint8_t* src1p = dstp[channel];
          const uint8_t* src2p = ovrp[channel];
          int src1_pitch = dstp_pitch;
          int src2_pitch = ovrp_pitch;

          if (layer_fn && bits_per_pixel != 32) {
            layer_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
              currentwidth, currentheight, opacity_i, bits_per_pixel);
          }
          else if (layer_f_fn && bits_per_pixel == 32) {
            layer_f_fn(src1p, src2p, maskp, src1_pitch, src2_pitch, mask_pitch,
              currentwidth, currentheight, opacity);
          }
        }
      }
    } // Add/Fast
    else if (!lstrcmpi(Op, "Lighten") || !lstrcmpi(Op, "Darken"))
    {
      const bool isLighten = !lstrcmpi(Op, "Lighten");
      // only chroma == true
      // Copy overlay_clip over base_clip in areas where overlay_clip is lighter by threshold.
      // called only once, for all planes
      // integer 8-16 bits version
      layer_planarrgb_lighten_darken_c_t* layer_fn = nullptr;
      // 32 bit float version
      layer_planarrgb_lighten_darken_f_c_t* layer_f_fn = nullptr;

#ifdef INTEL_INTRINSICS
      if (env->GetCPUFlags() & CPUF_AVX2) {
        // wrapper for the avx2 module, which calls its local scoped get_layer_planarrgb_lighten_darken_functions the included layer.hpp
        get_layer_planarrgb_lighten_darken_functions_avx2(isLighten, hasAlpha, process_alpha_channel, bits_per_pixel, &layer_fn, &layer_f_fn);
      }
      else
#endif
        get_layer_planarrgb_lighten_darken_functions(isLighten, hasAlpha, process_alpha_channel, bits_per_pixel, &layer_fn, &layer_f_fn);

      if (layer_fn && bits_per_pixel != 32)
        layer_fn(dstp, ovrp, maskp, dstp_pitch, ovrp_pitch, mask_pitch, width, height, opacity_i, ThresholdParam, bits_per_pixel);
      else if (layer_f_fn && bits_per_pixel == 32)
        layer_f_fn(dstp, ovrp, maskp, dstp_pitch, ovrp_pitch, mask_pitch, width, height, opacity, ThresholdParam_f);
      // planar rgba lighten/darken end
    }
    // Layer planar RGB(A) end
  }

  return src1;
}


AVSValue __cdecl Layer::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  const VideoInfo& vi1 = args[0].AsClip()->GetVideoInfo();
  const VideoInfo& vi2 = args[1].AsClip()->GetVideoInfo();

  // "Add" and "Subtract" (use_chroma=true) have native interleaved paths for RGB32
  // (masked_blend_packedrgba_avx2/sse41/c — magic-div, correct at both endpoints).
  // For those ops the packed clips are passed straight through; no round-trip conversion
  // to PlanarRGBA is needed.  Subtract pre-inverts the overlay and extracts the original
  // alpha as a separate clip (mask_child) below — see the comment on the Subtract redirect.
  // All other ops (Mul, Lighten, Darken, Fast, ...) still require the planar path, and
  // RGB24/RGB48/RGB64/YUY2 always convert regardless of op.
  const char* opStr = args[2].AsString("Add");
  const bool packedRGBNativePath =
    vi1.IsRGB32() &&
    (!lstrcmpi(opStr, "Add") || !lstrcmpi(opStr, "Subtract"));

  // Convert base clip to planar when needed.
  // RGB24/RGB48 → PlanarRGB; RGB32/RGB64 → PlanarRGBA (unless native path).
  // ConvertToPlanarRGB(A) handles the packed-RGB vertical flip.
  PClip clip1;
  if (!packedRGBNativePath && (vi1.IsRGB32() || vi1.IsRGB64())) {
    AVSValue new_args[1] = { args[0].AsClip() };
    clip1 = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi1.IsRGB24() || vi1.IsRGB48()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    clip1 = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi1.IsYUY2()) {
    AVSValue new_args[1] = { args[0].AsClip() };
    clip1 = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }
  else {
    clip1 = args[0].AsClip();
  }

  // Convert overlay clip to planar when needed.
  // For the native packed path the overlay stays as RGB32/RGB64 so GetFrame can read
  // the interleaved alpha directly (Add) or from a separate extracted clip (Subtract).
  // For the planar path, RGB32/RGB64 → PlanarRGBA so that the A plane is available
  // as the per-pixel blend weight.
  PClip clip2;
  if (!packedRGBNativePath && (vi2.IsRGB32() || vi2.IsRGB64())) {
    AVSValue new_args[1] = { args[1].AsClip() };
    clip2 = env->Invoke("ConvertToPlanarRGBA", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi2.IsRGB24() || vi2.IsRGB48()) {
    AVSValue new_args[1] = { args[1].AsClip() };
    clip2 = env->Invoke("ConvertToPlanarRGB", AVSValue(new_args, 1)).AsClip();
  }
  else if (vi2.IsYUY2()) {
    AVSValue new_args[1] = { args[1].AsClip() };
    clip2 = env->Invoke("ConvertToYV16", AVSValue(new_args, 1)).AsClip();
  }
  else {
    clip2 = args[1].AsClip();
  }

  // When op="Subtract", pre-invert the overlay clip and redirect to "Add".
  // This eliminates the subtract template dimension in layer.hpp and the SIMD dispatchers.
  // Equivalence is exact for Y/U/V/R/G/B planes.
  //
  // Alpha treatment: alpha is just another plane — Invert() inverts ALL channels including A.
  // For overlay clips that have an alpha plane (YUVA / PlanarRGBA / RGB32 / RGB64), we must
  // save the original (pre-invert) A before calling Invert, because:
  //   - The original A is the per-pixel blend *weight* for all colour channels.
  //   - The inverted A is the blend *target* for the alpha channel itself when both clips
  //     have alpha (process_alpha_channel=true, or packed-RGB native path).
  // ExtractA is a free SubplanarFrame alias (no copy) for planar formats; for packed
  // RGB32/RGB64 it produces a separate 8/16-bit Y grayscale clip.
  // The result is stored in mask_child and wired into GetFrame as the per-pixel mask.
  // For overlay clips without alpha, and for all non-Subtract operations, mask_child stays
  // nullptr and GetFrame reads the mask directly from the overlay's alpha channel.
  //
  // Native packed RGB32 Subtract (use_chroma=true): treated uniformly here —
  // ExtractA + Invert + redirect to Add — so no subtract logic remains in any kernel.
  // use_chroma=false Subtract keeps the old formula-based kernels (no alpha mask).
  const bool use_chroma_arg = args[7].AsBool(true);
  PClip mask_clip = nullptr;
  if (!lstrcmpi(opStr, "Subtract")) {
    const VideoInfo& vi_c2 = clip2->GetVideoInfo();
    if (packedRGBNativePath && use_chroma_arg) {
      // Native packed RGB32 Subtract (use_chroma=true):
      // extract original alpha as separate mask, pre-invert the overlay, redirect.
      AVSValue ea_args[1] = { AVSValue(clip2) };
      mask_clip = env->Invoke("ExtractA", AVSValue(ea_args, 1)).AsClip();
      AVSValue inv_args[1] = { AVSValue(clip2) };
      clip2 = env->Invoke("Invert", AVSValue(inv_args, 1)).AsClip();
      opStr = "Add";
    }
    else if (vi_c2.IsYUV() || vi_c2.IsYUVA() || vi_c2.IsPlanarRGB() || vi_c2.IsPlanarRGBA()) {
      // Save pre-invert alpha as the per-pixel blend weight.
      if (vi_c2.IsYUVA() || vi_c2.IsPlanarRGBA()) {
        AVSValue ea_args[1] = { AVSValue(clip2) };
        mask_clip = env->Invoke("ExtractA", AVSValue(ea_args, 1)).AsClip();
      }
      // Invert ALL channels — alpha is treated uniformly like any other plane.
      AVSValue inv_args[1] = { AVSValue(clip2) };
      clip2 = env->Invoke("Invert", AVSValue(inv_args, 1)).AsClip();
      opStr = "Add";
    }
  }

  Layer* Result = new Layer(clip1, clip2, mask_clip, opStr, args[3].AsInt(-1),
    args[4].AsInt(0), args[5].AsInt(0), args[6].AsInt(0),
    args[7].AsBool(true), // chroma
    args[8].AsFloatf(-1.0f), // opacity
    getPlacement(args[9], clip1, env), // chroma placement -> mask kernel variant
    env);

  if (vi1.IsRGB24()) {
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToRGB24", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi1.IsRGB48()) {
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToRGB48", AVSValue(new_args2, 1)).AsClip();
  }
  else if (!packedRGBNativePath && vi1.IsRGB32()) {
    // Non-native op: result is PlanarRGBA, convert back to packed.
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToRGB32", AVSValue(new_args2, 1)).AsClip();
  }
  else if (!packedRGBNativePath && vi1.IsRGB64()) {
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToRGB64", AVSValue(new_args2, 1)).AsClip();
  }
  else if (vi1.IsYUY2()) {
    AVSValue new_args2[1] = { Result };
    return env->Invoke("ConvertToYUY2", AVSValue(new_args2, 1)).AsClip();
  }

  return Result;

}



/**********************************
 *******   Subtract Filter   ******
 *********************************/
bool Subtract::DiffFlag = false;
BYTE Subtract::LUT_Diff8[513];

Subtract::Subtract(PClip _child1, PClip _child2, IScriptEnvironment* env)
  : child1(_child1), child2(_child2)
{
  VideoInfo vi1 = child1->GetVideoInfo();
  VideoInfo vi2 = child2->GetVideoInfo();

  if (vi1.width != vi2.width || vi1.height != vi2.height)
    env->ThrowError("Subtract: image dimensions don't match");

  if (!(vi1.IsSameColorspace(vi2)))
    env->ThrowError("Subtract: image formats don't match");

  vi = vi1;
  vi.num_frames = std::max(vi1.num_frames, vi2.num_frames);
  vi.num_audio_samples = std::max(vi1.num_audio_samples, vi2.num_audio_samples);

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (!DiffFlag) { // Init the global Diff table
    DiffFlag = true;
    for (int i = 0; i <= 512; i++) LUT_Diff8[i] = std::max(0, std::min(255, i - 129));
    // 0 ..  129  130 131   ... 255 256 257 258     384 ... 512
    // 0 ..   0    1   2  3 ... 126 127 128 129 ... 255 ... 255
  }
}

template<typename pixel_t, int midpixel, bool chroma>
static void subtract_plane(BYTE* src1p, const BYTE* src2p, int src1_pitch, int src2_pitch, int width, int height, int bits_per_pixel)
{
  typedef typename std::conditional < sizeof(pixel_t) == 4, float, int>::type limits_t;

  const limits_t limit_lo = sizeof(pixel_t) <= 2 ? 0 : (limits_t)(chroma ? uv8tof(0) : c8tof(0));
  const limits_t limit_hi = sizeof(pixel_t) == 1 ? 255 : sizeof(pixel_t) == 2 ? ((1 << bits_per_pixel) - 1) : (limits_t)(chroma ? uv8tof(255) : c8tof(255));
  const limits_t equal_luma = sizeof(pixel_t) == 1 ? midpixel : sizeof(pixel_t) == 2 ? (midpixel << (bits_per_pixel - 8)) : (limits_t)(chroma ? uv8tof(midpixel) : c8tof(midpixel));
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      limits_t val = reinterpret_cast<const pixel_t*>(src2p)[x] - reinterpret_cast<const pixel_t*>(src1p)[x] + equal_luma;
      // 126: luma of equality
      reinterpret_cast<pixel_t*>(src1p)[x] =
        (pixel_t)std::min(std::max(val, limit_lo), limit_hi);
    }
    src1p += src1_pitch;
    src2p += src2_pitch;
  }
}

PVideoFrame __stdcall Subtract::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src1 = child1->GetFrame(n, env);
  PVideoFrame src2 = child2->GetFrame(n, env);

  env->MakeWritable(&src1);

  BYTE* src1p = src1->GetWritePtr();
  const BYTE* src2p = src2->GetReadPtr();
  int row_size = src1->GetRowSize();
  int src1_pitch = src1->GetPitch();
  int src2_pitch = src2->GetPitch();

  int width = row_size / pixelsize;
  int height = vi.height;

  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    // alpha
    if (pixelsize == 1) {
      // LUT is a bit faster than clamp version
      for (int y = 0; y < vi.height; y++) {
        for (int x = 0; x < row_size; x++) {
          src1p[x] = LUT_Diff8[src1p[x] - src2p[x] + 126 + 129];
        }
        src1p += src1->GetPitch();
        src2p += src2->GetPitch();
      }
    }
    else if (pixelsize == 2)
      subtract_plane<uint16_t, 126, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
    else //if (pixelsize==4)
      subtract_plane<float, 126, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);

    // chroma
    row_size = src1->GetRowSize(PLANAR_U);
    if (row_size) {
      width = row_size / pixelsize;
      height = src1->GetHeight(PLANAR_U);
      src1_pitch = src1->GetPitch(PLANAR_U);
      src2_pitch = src2->GetPitch(PLANAR_U);
      // U_plane exists
      BYTE* src1p = src1->GetWritePtr(PLANAR_U);
      const BYTE* src2p = src2->GetReadPtr(PLANAR_U);
      BYTE* src1pV = src1->GetWritePtr(PLANAR_V);
      const BYTE* src2pV = src2->GetReadPtr(PLANAR_V);

      if (pixelsize == 1) {
        // LUT is a bit faster than clamp version
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            src1p[x] = LUT_Diff8[src1p[x] - src2p[x] + 128 + 129];
            src1pV[x] = LUT_Diff8[src1pV[x] - src2pV[x] + 128 + 129];
          }
          src1p += src1_pitch;
          src2p += src2_pitch;
          src1pV += src1_pitch;
          src2pV += src2_pitch;
        }
      }
      else if (pixelsize == 2) {
        subtract_plane<uint16_t, 128, true>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
        subtract_plane<uint16_t, 128, true>(src1pV, src2pV, src1_pitch, src2_pitch, width, height, bits_per_pixel);
      }
      else { //if (pixelsize==4)
        subtract_plane<float, 128, true>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
        subtract_plane<float, 128, true>(src1pV, src2pV, src1_pitch, src2_pitch, width, height, bits_per_pixel);
      }
    }
    return src1;
  } // End planar YUV

  // For YUY2, 50% gray is about (126,128,128) instead of (128,128,128).  Grr...
  if (vi.IsYUY2()) {
    for (int y = 0; y < vi.height; ++y) {
      for (int x = 0; x < row_size; x += 2) {
        src1p[x] = LUT_Diff8[src1p[x] - src2p[x] + 126 + 129];
        src1p[x + 1] = LUT_Diff8[src1p[x + 1] - src2p[x + 1] + 128 + 129];
      }
      src1p += src1->GetPitch();
      src2p += src2->GetPitch();
    }
  }
  else { // RGB
    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) {
      const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };

      // do not diff Alpha
      for (int p = 0; p < 3; p++) {
        const int plane = planesRGB[p];
        src1p = src1->GetWritePtr(plane);
        src2p = src2->GetReadPtr(plane);
        src1_pitch = src1->GetPitch(plane);
        src2_pitch = src2->GetPitch(plane);
        if (pixelsize == 1)
          subtract_plane<uint8_t, 128, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
        else if (pixelsize == 2)
          subtract_plane<uint16_t, 128, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
        else
          subtract_plane<float, 128, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
      }
    }
    else { // packed RGB
      if (pixelsize == 1) {
        for (int y = 0; y < vi.height; ++y) {
          for (int x = 0; x < row_size; ++x)
            src1p[x] = LUT_Diff8[src1p[x] - src2p[x] + 128 + 129];

          src1p += src1->GetPitch();
          src2p += src2->GetPitch();
        }
      }
      else { // pixelsize == 2: RGB48, RGB64
        // width is getrowsize based here: ok.
        subtract_plane<uint16_t, 128, false>(src1p, src2p, src1_pitch, src2_pitch, width, height, bits_per_pixel);
      }
    }
  }
  return src1;
}



AVSValue __cdecl Subtract::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new Subtract(args[0].AsClip(), args[1].AsClip(), env);
}
