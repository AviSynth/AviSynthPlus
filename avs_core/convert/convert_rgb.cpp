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


#include "convert_rgb.h"
#ifdef INTEL_INTRINSICS
#include "intel/convert_rgb_sse.h"
#include "intel/convert_rgb_avx2.h"
#endif
#include <avs/alignment.h>


/*************************************
 *******   RGB Helper Classes   ******
 *************************************/

RGBtoRGBA::RGBtoRGBA(PClip src)
  : GenericVideoFilter(src)
{
  vi.pixel_type = src->GetVideoInfo().ComponentSize() == 1 ? VideoInfo::CS_BGR32 : VideoInfo::CS_BGR64;
}

static void convert_rgb24_to_rgb32_c(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  for (size_t y = height; y > 0; --y) {
    for (size_t x = 0; x < width; ++x) {
      *reinterpret_cast<int*>(dstp + x*4) = *reinterpret_cast<const int*>(srcp+x*3) | 0xFF000000;
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

static void convert_rgb48_to_rgb64_c(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  for (size_t y = height; y > 0; --y) {
    for (size_t x = 0; x < width; ++x) {
      *reinterpret_cast<uint64_t*>(dstp + x*8) = *reinterpret_cast<const uint64_t*>(srcp+x*6) | 0xFFFF000000000000ULL;
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

PVideoFrame __stdcall RGBtoRGBA::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  const BYTE *srcp = src->GetReadPtr();
  BYTE *dstp = dst->GetWritePtr();
  const int src_pitch = src->GetPitch();
  const int dst_pitch = dst->GetPitch();

  int pixelsize = vi.ComponentSize();

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSSE3) {
    if(pixelsize==1)
      convert_rgb24_to_rgb32_ssse3(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    else
      convert_rgb48_to_rgb64_ssse3(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
  }
  else
#ifdef X86_32
    if (pixelsize==1 && (env->GetCPUFlags() & CPUF_MMX))
    {
      convert_rgb24_to_rgb32_mmx(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    }
    else
#endif
#endif
    {
      if (pixelsize == 1)
        convert_rgb24_to_rgb32_c(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
      else
        convert_rgb48_to_rgb64_c(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    }
  return dst;
}




RGBAtoRGB::RGBAtoRGB(PClip src)
: GenericVideoFilter(src)
{
  vi.pixel_type = src->GetVideoInfo().ComponentSize() == 1 ? VideoInfo::CS_BGR24 : VideoInfo::CS_BGR48;
}

static void convert_rgb32_to_rgb24_c(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  for (size_t y = height; y > 0; --y) {
    size_t x;
    for (x = 0; x < width-1; ++x) {
      *reinterpret_cast<int*>(dstp+x*3) = *reinterpret_cast<const int*>(srcp+x*4);
    }
    //last pixel
    dstp[x*3+0] = srcp[x*4+0];
    dstp[x*3+1] = srcp[x*4+1];
    dstp[x*3+2] = srcp[x*4+2];

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

static void convert_rgb64_to_rgb48_c(const BYTE *srcp, BYTE *dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height) {
  for (size_t y = height; y > 0; --y) {
    size_t x;
    for (x = 0; x < width-1; ++x) { // width-1 really!
      *reinterpret_cast<uint64_t*>(dstp+x*6) = *reinterpret_cast<const uint64_t*>(srcp+x*8);
    }
    //last pixel
    reinterpret_cast<uint16_t*>(dstp)[x*3+0] = reinterpret_cast<const uint16_t*>(srcp)[x*4+0];
    reinterpret_cast<uint16_t*>(dstp)[x*3+1] = reinterpret_cast<const uint16_t*>(srcp)[x*4+1];
    reinterpret_cast<uint16_t*>(dstp)[x*3+2] = reinterpret_cast<const uint16_t*>(srcp)[x*4+2];

    srcp += src_pitch;
    dstp += dst_pitch;
  }
}

PVideoFrame __stdcall RGBAtoRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  const BYTE *srcp = src->GetReadPtr();
  BYTE *dstp = dst->GetWritePtr();
  size_t src_pitch = src->GetPitch();
  size_t dst_pitch = dst->GetPitch();

  int pixelsize = vi.ComponentSize();

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSSE3) {
    if(pixelsize==1)
      convert_rgb32_to_rgb24_ssse3(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    else
      convert_rgb64_to_rgb48_ssse3(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
  }
  else
#ifdef X86_32
  if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_MMX))
  {
    convert_rgb32_to_rgb24_mmx(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
  }
  else
#endif
#endif
  {
    if(pixelsize==1)
      convert_rgb32_to_rgb24_c(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    else
      convert_rgb64_to_rgb48_c(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
  }
  return dst;
}

PackedRGBtoPlanarRGB::PackedRGBtoPlanarRGB(PClip src, bool _sourceHasAlpha, bool _targetHasAlpha)
  : GenericVideoFilter(src), sourceHasAlpha(_sourceHasAlpha), targetHasAlpha(_targetHasAlpha)
{
  vi.pixel_type = src->GetVideoInfo().ComponentSize() == 1 ?
    (targetHasAlpha ? VideoInfo::CS_RGBAP : VideoInfo::CS_RGBP) :
    (targetHasAlpha ? VideoInfo::CS_RGBAP16 : VideoInfo::CS_RGBP16);
}

template<typename pixel_t, int src_numcomponents>
static void convert_rgb_to_rgbp_c(const BYTE *srcp, BYTE * (&dstp)[4], int src_pitch, int (&dst_pitch)[4], size_t width, size_t height, int bits_per_pixel) {
  bool targetHasAlpha = (dst_pitch[3] != 0);
  const int max_pixel_value = (1 << bits_per_pixel) - 1;
  for (size_t y = height; y > 0; --y) {
    size_t x;
    // not proud of it but it works
    for (x = 0; x < width; ++x) {
      pixel_t B = reinterpret_cast<const pixel_t *>(srcp)[x*src_numcomponents + 0];
      pixel_t G = reinterpret_cast<const pixel_t *>(srcp)[x*src_numcomponents + 1];
      pixel_t R = reinterpret_cast<const pixel_t *>(srcp)[x*src_numcomponents + 2];
      pixel_t A = 0;
      if(targetHasAlpha)
        A = (src_numcomponents==4) ? reinterpret_cast<const pixel_t *>(srcp)[x*src_numcomponents + 3] : max_pixel_value;
      reinterpret_cast<pixel_t *>(dstp[0])[x] = G;
      reinterpret_cast<pixel_t *>(dstp[1])[x] = B;
      reinterpret_cast<pixel_t *>(dstp[2])[x] = R;
      if(targetHasAlpha)
        reinterpret_cast<pixel_t *>(dstp[3])[x] = A;
    }

    srcp -= src_pitch; // source packed RGB is upside down
    dstp[0] += dst_pitch[0];
    dstp[1] += dst_pitch[1];
    dstp[2] += dst_pitch[2];
    if (targetHasAlpha)
      dstp[3] += dst_pitch[3];
  }
}

PVideoFrame __stdcall PackedRGBtoPlanarRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  int src_pitch = src->GetPitch();
  const BYTE *srcp = src->GetReadPtr();
  BYTE *dstp[4] = {dst->GetWritePtr(PLANAR_G),dst->GetWritePtr(PLANAR_B),dst->GetWritePtr(PLANAR_R),dst->GetWritePtr(PLANAR_A)};
  int dst_pitch[4] = {dst->GetPitch(PLANAR_G),dst->GetPitch(PLANAR_B),dst->GetPitch(PLANAR_R),dst->GetPitch(PLANAR_A)};

  int pixelsize = vi.ComponentSize();

  srcp += src_pitch * (vi.height - 1); // start from bottom: packed RGB is upside down

#ifdef INTEL_INTRINSICS
  const bool targetHasAlpha = vi.IsPlanarRGBA();
#endif

  if(pixelsize==1)
  {
    // targetHasAlpha decision in convert function
    if (sourceHasAlpha) {
      // RGB32->RGBP8
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSSE3) && vi.width >= 8) {
        if (targetHasAlpha)
          convert_rgba_to_rgbp_ssse3<uint8_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
        else
          convert_rgba_to_rgbp_ssse3<uint8_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
      }
      else
#endif
        convert_rgb_to_rgbp_c<uint8_t, 4>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
    }
    else {
#ifdef INTEL_INTRINSICS
      // RGB24->RGB(A)P8, works with 48byte blocks (16xRGB), min width is 16 (SSSE3, 32 (AVX2)
      if ((env->GetCPUFlags() & CPUF_AVX2) && vi.width >= 32) {
        if (targetHasAlpha)
          convert_rgb_to_rgbp_avx2<uint8_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
        else
          convert_rgb_to_rgbp_avx2<uint8_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
      }
      else if ((env->GetCPUFlags() & CPUF_SSSE3) && vi.width >= 16) {
        if (targetHasAlpha)
          convert_rgb_to_rgbp_ssse3<uint8_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
        else
          convert_rgb_to_rgbp_ssse3<uint8_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
      }
      else
#endif
        convert_rgb_to_rgbp_c<uint8_t, 3>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 8);
    }
  }
  else {
    if (sourceHasAlpha) {
      // RGB32->RGBP16, RGBAP16
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSSE3) && vi.width >= 4) {
        if (targetHasAlpha)
          convert_rgba_to_rgbp_ssse3<uint16_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
        else
          convert_rgba_to_rgbp_ssse3<uint16_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
      }
      else
#endif
        convert_rgb_to_rgbp_c<uint16_t, 4>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
    }
    else {
      // RGB48->RGB(A)P16, works with 48byte blocks (8xRGB), min width is 8 (SSSE3), 16 (AVX2)
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_AVX2) && vi.width >= 16) {
        if (targetHasAlpha)
          convert_rgb_to_rgbp_avx2<uint16_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
        else
          convert_rgb_to_rgbp_avx2<uint16_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
      }
      else if ((env->GetCPUFlags() & CPUF_SSSE3) && vi.width >= 8) {
        if (targetHasAlpha)
          convert_rgb_to_rgbp_ssse3<uint16_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
        else
          convert_rgb_to_rgbp_ssse3<uint16_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
      }
      else
#endif
        convert_rgb_to_rgbp_c<uint16_t, 3>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height, 16);
    }
  }
  return dst;
}

PlanarRGBtoPackedRGB::PlanarRGBtoPackedRGB(PClip src, bool _targetHasAlpha)
  : GenericVideoFilter(src), targetHasAlpha(_targetHasAlpha)
{
  vi.pixel_type = src->GetVideoInfo().ComponentSize() == 1 ?
    (targetHasAlpha ? VideoInfo::CS_BGR32 : VideoInfo::CS_BGR24) : // PlanarRGB(A)->RGB24/32
    (targetHasAlpha ? VideoInfo::CS_BGR64 : VideoInfo::CS_BGR48);  // PlanarRGB(A)->RGB48/64
}

template<typename pixel_t, int target_numcomponents>
static void convert_rgbp_to_rgb_c(const BYTE *(&srcp)[4], BYTE * dstp, int (&src_pitch)[4], int dst_pitch, size_t width, size_t height) {
  bool hasSrcAlpha = (src_pitch[3] != 0);
  for (size_t y = 0; y < height; y++) {
    size_t x;
    // not proud of it but it works
    for (x = 0; x < width; ++x) {
      const pixel_t G = reinterpret_cast<const pixel_t *>(srcp[0])[x];
      const pixel_t B = reinterpret_cast<const pixel_t *>(srcp[1])[x];
      const pixel_t R = reinterpret_cast<const pixel_t *>(srcp[2])[x];
      reinterpret_cast<pixel_t *>(dstp)[x*target_numcomponents+0] = B;
      reinterpret_cast<pixel_t *>(dstp)[x*target_numcomponents+1] = G;
      reinterpret_cast<pixel_t *>(dstp)[x*target_numcomponents+2] = R;
      if constexpr (target_numcomponents == 4) { // either from A channel or default transparent constant
        const pixel_t A = hasSrcAlpha ? reinterpret_cast<const pixel_t *>(srcp[3])[x] : (1 << (8 * sizeof(pixel_t))) - 1; // 255/65535
        reinterpret_cast<pixel_t *>(dstp)[x*target_numcomponents + 3] = A;
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

PVideoFrame __stdcall PlanarRGBtoPackedRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  int dst_pitch = dst->GetPitch();
  BYTE *dstp = dst->GetWritePtr();
  const BYTE *srcp[4] = {src->GetReadPtr(PLANAR_G),src->GetReadPtr(PLANAR_B),src->GetReadPtr(PLANAR_R),src->GetReadPtr(PLANAR_A)};
  int src_pitch[4] = {src->GetPitch(PLANAR_G),src->GetPitch(PLANAR_B),src->GetPitch(PLANAR_R),src->GetPitch(PLANAR_A)};

  int pixelsize = vi.ComponentSize();

  dstp += dst_pitch * (vi.height - 1); // start from bottom: packed RGB is upside down

#ifdef INTEL_INTRINSICS
  bool hasSrcAlpha = (src_pitch[3] != 0); // Planar RGBA
#endif
  bool hasTargetAlpha = (vi.NumComponents() == 4);

  if(pixelsize==1)
  {
    if(!hasTargetAlpha) // RGB24
      convert_rgbp_to_rgb_c<uint8_t, 3>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    else {// RGBA32
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSE2) && vi.width >= 4) {
        if(hasSrcAlpha)
          convert_rgbp_to_rgba_sse2<uint8_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
        else
          convert_rgbp_to_rgba_sse2<uint8_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
      }
      else
#endif
        convert_rgbp_to_rgb_c<uint8_t, 4>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    }
  } else {
    if(!hasTargetAlpha)
      convert_rgbp_to_rgb_c<uint16_t, 3>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    else { // RGBA64
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSE2) && vi.width >= 4) {
        if(hasSrcAlpha)
          convert_rgbp_to_rgba_sse2<uint16_t, true>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
        else
          convert_rgbp_to_rgba_sse2<uint16_t, false>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
      }
      else 
#endif
        convert_rgbp_to_rgb_c<uint16_t, 4>(srcp, dstp, src_pitch, dst_pitch, vi.width, vi.height);
    }
  }
  return dst;
}
