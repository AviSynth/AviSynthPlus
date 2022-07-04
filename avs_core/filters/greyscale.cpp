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


#include "greyscale.h"
#ifdef INTEL_INTRINSICS
#include "intel/greyscale_sse.h"
#endif
#include "../core/internal.h"
#include <avs/alignment.h>
#include <avs/minmax.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <stdint.h>
#include "../convert/convert_planar.h"
#include "../convert/convert.h"
#include "../convert/convert_helper.h"


/*************************************
 *******   Convert to Greyscale ******
 ************************************/

extern const AVSFunction Greyscale_filters[] = {
  { "Greyscale", BUILTIN_FUNC_PREFIX, "c[matrix]s", Greyscale::Create },       // matrix can be "rec601", "rec709" or "Average" or "rec2020"
  { "Grayscale", BUILTIN_FUNC_PREFIX, "c[matrix]s", Greyscale::Create },
  { 0 }
};

Greyscale::Greyscale(PClip _child, const char* matrix_name, IScriptEnvironment* env)
 : GenericVideoFilter(_child)
{
  if (matrix_name && !vi.IsRGB())
    env->ThrowError("GreyScale: invalid \"matrix\" parameter (RGB data only)");

  // originally there was no PC range here
  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  if (vi.IsRGB()) {
    matrix_parse_merge_with_props(vi, matrix_name, nullptr, theMatrix, theColorRange, env);
    if(theColorRange == ColorRange_e::AVS_RANGE_FULL && theMatrix != Matrix_e::AVS_MATRIX_AVERAGE)
      env->ThrowError("GreyScale: only limited range matrix definition or \"Average\" is allowed.");
    // and then we make it full range because the result will get back into RGB planes, so no range conversion occurs.
    theColorRange = ColorRange_e::AVS_RANGE_FULL;
    const int shift = 15; // internally 15 bits precision, still no overflow in calculations

    if (!do_BuildMatrix_Rgb2Yuv(theMatrix, theColorRange, shift, bits_per_pixel, /*ref*/greyMatrix))
      env->ThrowError("GreyScale: Unknown matrix.");
  }
  // greyscale does not change color space, rgb remains rgb
  // Leave matrix and range frame properties as is.
}

template<typename pixel_t, int pixel_step>
static void greyscale_packed_rgb_c(BYTE *srcp8, int src_pitch, int width, int height, int cyb, int cyg, int cyr) {
  pixel_t *srcp = reinterpret_cast<pixel_t *>(srcp8);
  src_pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      srcp[x*pixel_step+0] = srcp[x*pixel_step+1] = srcp[x*pixel_step+2] =
        (cyb*srcp[x*pixel_step+0] + cyg*srcp[x*pixel_step+1] + cyr*srcp[x*pixel_step+2] + 16384) >> 15;
    }
    srcp += src_pitch;
  }
}

template<typename pixel_t>
static void greyscale_planar_rgb_c(BYTE *srcp_r8, BYTE *srcp_g8, BYTE *srcp_b8, int src_pitch, int width, int height, int cyb, int cyg, int cyr) {
  pixel_t *srcp_r = reinterpret_cast<pixel_t *>(srcp_r8);
  pixel_t *srcp_g = reinterpret_cast<pixel_t *>(srcp_g8);
  pixel_t *srcp_b = reinterpret_cast<pixel_t *>(srcp_b8);
  src_pitch /= sizeof(pixel_t);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      srcp_b[x] = srcp_g[x] = srcp_r[x] =
        ((cyb*srcp_b[x] + cyg*srcp_g[x] + cyr*srcp_r[x] + 16384) >> 15);
    }
    srcp_r += src_pitch;
    srcp_g += src_pitch;
    srcp_b += src_pitch;
  }
}

static void greyscale_planar_rgb_float_c(BYTE *srcp_r8, BYTE *srcp_g8, BYTE *srcp_b8, int src_pitch, int width, int height, float cyb, float cyg, float cyr) {
  float *srcp_r = reinterpret_cast<float *>(srcp_r8);
  float *srcp_g = reinterpret_cast<float *>(srcp_g8);
  float *srcp_b = reinterpret_cast<float *>(srcp_b8);
  src_pitch /= sizeof(float);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      srcp_b[x] = srcp_g[x] = srcp_r[x] = cyb*srcp_b[x] + cyg*srcp_g[x] + cyr*srcp_r[x];
    }
    srcp_r += src_pitch;
    srcp_g += src_pitch;
    srcp_b += src_pitch;
  }
}


PVideoFrame Greyscale::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame frame = child->GetFrame(n, env);
  if (vi.NumComponents() == 1)
    return frame;

  env->MakeWritable(&frame);
  BYTE* srcp = frame->GetWritePtr();
  int pitch = frame->GetPitch();
  int height = vi.height;
  int width = vi.width;

  // greyscale does not change color space, rgb remains rgb
  // Leave matrix and range frame properties as is.

  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    // planar YUV, set UV plane to neutral
    BYTE* dstp_u = frame->GetWritePtr(PLANAR_U);
    BYTE* dstp_v = frame->GetWritePtr(PLANAR_V);
    const int height = frame->GetHeight(PLANAR_U);
    const int dst_pitch = frame->GetPitch(PLANAR_U);
    switch (vi.ComponentSize())
    {
    case 1:
      fill_chroma<BYTE>(dstp_u, dstp_v, height, dst_pitch, 0x80); // in convert_planar
      break;
    case 2:
      fill_chroma<uint16_t>(dstp_u, dstp_v, height, dst_pitch, 1 << (vi.BitsPerComponent() - 1));
      break;
    case 4:
#ifdef FLOAT_CHROMA_IS_HALF_CENTERED
      const float shift = 0.5f;
#else
      const float shift = 0.0f;
#endif
      fill_chroma<float>(dstp_u, dstp_v, height, dst_pitch, shift);
      break;
    }
    return frame;
  }

  if (vi.IsYUY2()) {
#ifdef INTEL_INTRINSICS
    if ((env->GetCPUFlags() & CPUF_SSE2) && width > 4 && IsPtrAligned(srcp, 16)) {
      greyscale_yuy2_sse2(srcp, width, height, pitch);
    } else
#ifdef X86_32
      if ((env->GetCPUFlags() & CPUF_MMX) && width > 2) {
        greyscale_yuy2_mmx(srcp, width, height, pitch);
      } else
#endif
#endif
      {
        for (int y = 0; y<height; ++y)
        {
          for (int x = 0; x<width; x++)
            srcp[x*2+1] = 128;
          srcp += pitch;
        }
      }

      return frame;
  }
#ifdef INTEL_INTRINSICS
  if(vi.IsRGB64()) {
    if ((env->GetCPUFlags() & CPUF_SSE4_1) && IsPtrAligned(srcp, 16)) {
      greyscale_rgb64_sse41(srcp, width, height, pitch, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      return frame;
    }
  }

  if (vi.IsRGB32()) {
    if ((env->GetCPUFlags() & CPUF_SSE2) && IsPtrAligned(srcp, 16)) {
      greyscale_rgb32_sse2(srcp, width, height, pitch, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      return frame;
    }
#ifdef X86_32
    else if (env->GetCPUFlags() & CPUF_MMX) {
      greyscale_rgb32_mmx(srcp, width, height, pitch, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      return frame;
    }
#endif
  }
#endif

  if (vi.IsRGB())
  {  // RGB C.
    if (vi.IsPlanarRGB() || vi.IsPlanarRGBA())
    {
      BYTE* srcp_g = frame->GetWritePtr(PLANAR_G);
      BYTE* srcp_b = frame->GetWritePtr(PLANAR_B);
      BYTE* srcp_r = frame->GetWritePtr(PLANAR_R);

      const int src_pitch = frame->GetPitch(); // same for all planes

      if (pixelsize == 1)
        greyscale_planar_rgb_c<uint8_t>(srcp_r, srcp_g, srcp_b, src_pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      else if (pixelsize == 2)
        greyscale_planar_rgb_c<uint16_t>(srcp_r, srcp_g, srcp_b, src_pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      else
        greyscale_planar_rgb_float_c(srcp_r, srcp_g, srcp_b, src_pitch, vi.width, vi.height, greyMatrix.y_b_f, greyMatrix.y_g_f, greyMatrix.y_r_f);

      return frame;
    }
    // packed RGB

    const int rgb_inc = vi.IsRGB32() || vi.IsRGB64() ? 4 : 3;

    if (pixelsize == 1) { // rgb24/32
      if (rgb_inc == 3)
        greyscale_packed_rgb_c<uint8_t, 3>(srcp, pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      else
        greyscale_packed_rgb_c<uint8_t, 4>(srcp, pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
    }
    else { // rgb48/64
      if (rgb_inc == 3)
        greyscale_packed_rgb_c<uint16_t, 3>(srcp, pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
      else
        greyscale_packed_rgb_c<uint16_t, 4>(srcp, pitch, vi.width, vi.height, greyMatrix.y_b, greyMatrix.y_g, greyMatrix.y_r);
    }

  }
  return frame;
}


AVSValue __cdecl Greyscale::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  const VideoInfo& vi = clip->GetVideoInfo();

  if (vi.NumComponents() == 1)
    return clip;

  return new Greyscale(clip, args[1].AsString(0), env);
}
