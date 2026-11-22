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

#include "focus.h"
#ifdef INTEL_INTRINSICS
#include "intel/focus_sse.h"
#include "intel/focus_avx2.h"
#elif defined(NEON_INTRINSICS)
#include "aarch64/focus_neon.h"
#include "aarch64/focus_dp.h"
#endif
#include <cmath>
#include <vector>
#include <avs/alignment.h>
#include <avs/minmax.h>
#include "../core/internal.h"
#include <stdint.h>


/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Focus_filters[] = {
  { "Blur",           BUILTIN_FUNC_PREFIX, "cf[]f[mmx]b", Create_Blur },                     // amount [-1.0 - 1.5849625] -- log2(3)
  { "Sharpen",        BUILTIN_FUNC_PREFIX, "cf[]f[mmx]b", Create_Sharpen },               // amount [-1.5849625 - 1.0]
  { "TemporalSoften", BUILTIN_FUNC_PREFIX, "ciii[scenechange]i[mode]i", TemporalSoften::Create }, // radius, luma_threshold, chroma_threshold
  { "SpatialSoften",  BUILTIN_FUNC_PREFIX, "ciii", SpatialSoften::Create },   // radius, luma_threshold, chroma_threshold
  { 0 }
};





/****************************************
 ***  AdjustFocus helper classes     ***
 ***  Originally by Ben R.G.         ***
 ***  MMX code by Marc FD            ***
 ***  Adaptation and bugfixes sh0dan ***
 ***  Code actually requires ISSE!   ***
 ***  Not anymore - pure MMX    IanB ***
 ***  Implement boundary proc.  IanB ***
 ***  Impl. full 8bit MMX proc. IanB ***
 ***************************************/

AdjustFocusV::AdjustFocusV(double _amount, PClip _child)
: GenericVideoFilter(_child), amountd(pow(2.0, _amount)) {
    half_amount = int(32768 * amountd + 0.5);
}

template<typename pixel_t>
static void af_vertical_c(BYTE* line_buf8, BYTE* dstp8, const int height, const int pitch8, const int width, const int half_amount, int bits_per_pixel) {
  const int outer_weight_15 = af_outer_weight_q15(half_amount); // same arithmetic as the SIMD paths
  const int max_pixel_value = (1 << bits_per_pixel) - 1;

  pixel_t * dstp = reinterpret_cast<pixel_t *>(dstp8);
  pixel_t * line_buf = reinterpret_cast<pixel_t *>(line_buf8);
  int pitch = pitch8 / sizeof(pixel_t);

  for (int y = height-1; y>0; --y) {
    for (int x = 0; x < width; ++x) {
      const pixel_t a = (pixel_t)af_kernel_int(line_buf[x], dstp[x], dstp[x+pitch], outer_weight_15, max_pixel_value);
      line_buf[x] = dstp[x];
      dstp[x] = a;
    }
    dstp += pitch;
  }
  for (int x = 0; x < width; ++x) // Last row - map centre as lower
    dstp[x] = (pixel_t)af_kernel_int(line_buf[x], dstp[x], dstp[x], outer_weight_15, max_pixel_value);
}

static void af_vertical_c_float(BYTE* line_buf8, BYTE* dstp8, const int height, const int pitch8, const int width, const float amount) {
    float *dstp = reinterpret_cast<float *>(dstp8);
    float *line_buf = reinterpret_cast<float *>(line_buf8);
    int pitch = pitch8 / sizeof(float);

    const float center_weight = amount;
    const float outer_weight = (1.0f - amount) / 2.0f;

    for (int y = height-1; y>0; --y) {
        for (int x = 0; x < width; ++x) {
            float a = dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight;
            line_buf[x] = dstp[x];
            dstp[x] = a;
        }
        dstp += pitch;
    }
    for (int x = 0; x < width; ++x) { // Last row - map centre as lower
        dstp[x] = dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight;
    }
}

template<typename pixel_t>
static void af_vertical_process(BYTE* line_buf, BYTE* dstp, size_t height, size_t pitch, size_t row_size, int half_amount, int bits_per_pixel, IScriptEnvironment* env) {
  size_t width = row_size / sizeof(pixel_t);
  // only for 8/16 bit, float separated
#ifdef INTEL_INTRINSICS
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_AVX2) && width >= 32) {
    //pitch of aligned frames is always >= 32 so we'll just process some garbage if width is not mod32
    af_vertical_avx2(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_SSSE3) && width >= 16) {
    //pitch of aligned frames is always >= 16 so we'll just process some garbage if width is not mod16
    af_vertical_ssse3(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_SSE2) && width >= 16) {
    af_vertical_sse2(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_AVX2) && row_size >= 32) {
    af_vertical_uint16_t_avx2(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount, bits_per_pixel);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE4_1) && row_size >= 16) {
    af_vertical_uint16_t_sse41(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount, bits_per_pixel);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE2) && row_size >= 16) {
    af_vertical_uint16_t_sse2(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount, bits_per_pixel);
  }
  else
#endif
  {
    af_vertical_c<pixel_t>(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount, bits_per_pixel);
  }
}

static void af_vertical_process_float(BYTE* line_buf, BYTE* dstp, size_t height, size_t pitch, size_t row_size, double amountd, IScriptEnvironment* env) {
    size_t width = row_size / sizeof(float);
#ifdef INTEL_INTRINSICS
    if ((env->GetCPUFlags() & CPUF_SSE2) && width >= 16) {
        //pitch of aligned frames is always >= 16 so we'll just process some garbage if width is not mod16
        af_vertical_sse2_float(line_buf, dstp, (int)height, (int)pitch, (int)row_size, (float)amountd);
    } 
    else
#endif
    {
      af_vertical_c_float(line_buf, dstp, (int)height, (int)pitch, (int)width, (float)amountd);
    }
}

// --------------------------------
// Vertical Blur/Sharpen
// --------------------------------

PVideoFrame __stdcall AdjustFocusV::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);

    env->MakeWritable(&src);

    BYTE* line_buf = reinterpret_cast<BYTE*>(env->Allocate(AlignNumber(src->GetRowSize(), FRAME_ALIGN), FRAME_ALIGN, AVS_POOLED_ALLOC));
    if (!line_buf) {
        env->ThrowError("AdjustFocusV: Could not reserve memory.");
    }

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    if (vi.IsPlanar()) {
      const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
      const int planesYA[2]  = { PLANAR_Y, PLANAR_A};
      const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
      // IsYA() must be checked before IsYUVA(): IsYUVA() is also true for YA
      const int *planes = vi.IsYA() ? planesYA : vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;
      const int cplanes_to_filter = (vi.IsY() || vi.IsYA()) ? 1 : 3; // greyscale: Y only

      for (int cplane = 0; cplane < cplanes_to_filter; cplane++) {
            int plane = planes[cplane];
            BYTE* dstp = src->GetWritePtr(plane);
            int pitch = src->GetPitch(plane);
            int row_size = src->GetRowSize(plane);
            int height = src->GetHeight(plane);
            memcpy(line_buf, dstp, row_size); // First row - map centre as upper

            switch (pixelsize) {
            case 1: af_vertical_process<uint8_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env); break;
            case 2: af_vertical_process<uint16_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env); break;
            default: // 4: float
                af_vertical_process_float(line_buf, dstp, height, pitch, row_size, amountd, env); break;
            }
        }
    }
    else {
        BYTE* dstp = src->GetWritePtr();
        int pitch = src->GetPitch();
        int row_size = vi.RowSize();
        int height = vi.height;
        memcpy(line_buf, dstp, row_size); // First row - map centre as upper
        if (pixelsize == 1)
          af_vertical_process<uint8_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env);
        else
          af_vertical_process<uint16_t>(line_buf, dstp, height, pitch, row_size, half_amount, bits_per_pixel, env);
    }

    env->Free(line_buf);
    return src;
}


AdjustFocusH::AdjustFocusH(double _amount, PClip _child)
: GenericVideoFilter(_child), amountd(pow(2.0, _amount)) {
    half_amount = int(32768 * amountd + 0.5);
}


// ---------------------------------------------------
// Blur/Sharpen Horizontal RGB24/32/48/64 C++ Code
// ---------------------------------------------------

// Not in-place (like the SSE2 versions): src -> dst, each output only from source pixels, so the row is one
// flat loop over the components, vectorized by GCC and Clang. The in-place version carried the original
// left neighbour from pixel to pixel and stayed scalar. num_comp: components per pixel (3 or 4).
// First and last pixel: the missing neighbour is the pixel itself.
// Same pivot (Q15) arithmetic as the SIMD paths (see af_outer_weight_q15): fits int32 for 16 bit too.
template<typename pixel_t, int num_comp>
static void af_horizontal_packed_rgb_c(BYTE* dstp8, const BYTE* srcp8, size_t dst_pitch8, size_t src_pitch8, size_t height, size_t width, int half_amount) {
  const int outer_weight_15 = af_outer_weight_q15(half_amount);
  constexpr int max_pixel_value = sizeof(pixel_t) == 1 ? 255 : 65535;
  auto kernel = [=](int left, int center, int right) {
    return (pixel_t)af_kernel_int(left, center, right, outer_weight_15, max_pixel_value);
  };

  pixel_t* dstp = reinterpret_cast<pixel_t*>(dstp8);
  const pixel_t* srcp = reinterpret_cast<const pixel_t*>(srcp8);
  const size_t dst_pitch = dst_pitch8 / sizeof(pixel_t);
  const size_t src_pitch = src_pitch8 / sizeof(pixel_t);
  const size_t count = width * num_comp;

  for (size_t y = height; y > 0; --y)
  {
    if (width == 1) {
      for (int c = 0; c < num_comp; c++)
        dstp[c] = kernel(srcp[c], srcp[c], srcp[c]);
    }
    else {
      for (int c = 0; c < num_comp; c++) // first pixel: left neighbour = itself
        dstp[c] = kernel(srcp[c], srcp[c], srcp[c + num_comp]);
      for (size_t i = num_comp; i < count - num_comp; i++)
        dstp[i] = kernel(srcp[i - num_comp], srcp[i], srcp[i + num_comp]);
      for (size_t i = count - num_comp; i < count; i++) // last pixel: right neighbour = itself
        dstp[i] = kernel(srcp[i - num_comp], srcp[i], srcp[i]);
    }
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}


// -------------------------------------
// Blur/Sharpen Horizontal YUY2 C++ Code
// -------------------------------------

// Not in-place (like the SSE2 version), same pivot arithmetic as the SIMD paths. Luma: neighbours at +-2 bytes,
// chroma (U with U, V with V) at +-4 bytes; at the row ends the missing neighbour is the sample itself.
// The inner macropixels (Y U Y V) have all their neighbours inside the row: one flat loop, vectorized by GCC and Clang.
static void af_horizontal_yuy2_c(BYTE* dstp, const BYTE* srcp, size_t dst_pitch, size_t src_pitch, size_t height, size_t width, int half_amount) {
  const int outer_weight_15 = af_outer_weight_q15(half_amount);
  auto kernel = [=](int left, int center, int right) {
    return (BYTE)af_kernel_int(left, center, right, outer_weight_15, 255);
  };
  const size_t count = width * 2; // width is even: whole macropixels
  // pointers by value: a by-reference capture keeps Clang from vectorizing the inner loop
  auto edge_macropixel = [=](BYTE* d, const BYTE* s, size_t i0) {
    for (size_t i = i0; i < i0 + 4; i++) {
      const size_t step = (i & 1) ? 4 : 2;
      d[i] = kernel(i >= step ? s[i - step] : s[i], s[i], i + step < count ? s[i + step] : s[i]);
    }
  };

  for (size_t y = height; y > 0; --y)
  {
    edge_macropixel(dstp, srcp, 0);
    for (size_t i = 4; i < count - 4; i += 4) {
      dstp[i + 0] = kernel(srcp[i - 2], srcp[i + 0], srcp[i + 2]);
      dstp[i + 1] = kernel(srcp[i - 3], srcp[i + 1], srcp[i + 5]);
      dstp[i + 2] = kernel(srcp[i + 0], srcp[i + 2], srcp[i + 4]);
      dstp[i + 3] = kernel(srcp[i - 1], srcp[i + 3], srcp[i + 7]);
    }
    edge_macropixel(dstp, srcp, count - 4); // width 2: the same macropixel again
    dstp += dst_pitch;
    srcp += src_pitch;
  }
}


// -------------------------------------
// Blur/Sharpen Horizontal YV12 C++ Code
// -------------------------------------

// for linker reasons this forceinlined function appears in C and intel sse2 source as well (8-16 bit: focus.h)
static AVS_FORCEINLINE void af_horizontal_planar_process_line_float_c(float left, float* dstp, size_t row_size, float center_weight, float outer_weight) {
  size_t x;
  size_t width = row_size / sizeof(float);
  for (x = 0; x < width - 1; ++x) {
    float temp = dstp[x] * center_weight + (left + dstp[x + 1]) * outer_weight;
    left = dstp[x];
    dstp[x] = temp;
  }
  dstp[x] = dstp[x] * center_weight + (left + dstp[x]) * outer_weight;
}

template<typename pixel_t>
static void af_horizontal_planar_c(BYTE* dstp8, size_t height, size_t pitch8, size_t row_size, size_t half_amount, int bits_per_pixel)
{
    pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
    size_t pitch = pitch8 / sizeof(pixel_t);
    const int outer_weight_15 = af_outer_weight_q15(half_amount); // same arithmetic as the SIMD paths
    const int max_pixel_value = (1 << bits_per_pixel) - 1;
    for (size_t y = height; y>0; --y) {
        af_horizontal_planar_process_line_c<pixel_t>(dstp[0], (BYTE *)dstp, row_size, outer_weight_15, max_pixel_value);
        dstp += pitch;
    }
}

static void af_horizontal_planar_float_c(BYTE* dstp8, size_t height, size_t pitch8, size_t row_size, float amount)
{
    float* dstp = reinterpret_cast<float *>(dstp8);
    size_t pitch = pitch8 / sizeof(float);
    float center_weight = amount;
    float outer_weight = (1.0f - amount) / 2.0f;
    float left;
    for (size_t y = height; y>0; --y) {
        left = dstp[0];
        af_horizontal_planar_process_line_float_c(left, dstp, row_size, center_weight, outer_weight);
        dstp += pitch;
    }
}

static void copy_frame(const PVideoFrame &src, PVideoFrame &dst, IScriptEnvironment *env, const int *planes, int plane_count) {
  for (int p = 0; p < plane_count; p++) {
    int plane = planes[p];
    env->BitBlt(dst->GetWritePtr(plane), dst->GetPitch(plane), src->GetReadPtr(plane),
      src->GetPitch(plane), src->GetRowSize(plane), src->GetHeight(plane));
  }
}

// ----------------------------------
// Blur/Sharpen Horizontal GetFrame()
// ----------------------------------

PVideoFrame __stdcall AdjustFocusH::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const int planesYUVA[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  const int planesYA[2] = { PLANAR_Y, PLANAR_A };
  const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  const int* planes = vi.IsYA() ? planesYA : vi.IsYUV() || vi.IsYUVA() ? planesYUVA : planesRGB;

  int pixelsize = vi.ComponentSize();

  if (vi.IsPlanar()) {
    copy_frame(src, dst, env, planes, vi.NumComponents()); //planar processing is always in-place
    int bits_per_pixel = vi.BitsPerComponent();
    // Only the non-alpha planes get the blur/sharpen
    // alpha was already copied by copy_frame above
    const int cplanes_to_filter = (vi.IsY() || vi.IsYA()) ? 1 : 3; // greyscale: Y only
    for (int cplane = 0; cplane < cplanes_to_filter; cplane++) {
      int plane = planes[cplane];
      int row_size = dst->GetRowSize(plane);
      BYTE* q = dst->GetWritePtr(plane);
      int pitch = dst->GetPitch(plane);
      int height = dst->GetHeight(plane);
#ifdef INTEL_INTRINSICS
      if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_AVX2) && row_size > 32) {
        af_horizontal_planar_avx2(q, height, pitch, row_size, half_amount);
      }
      else if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_SSSE3) && row_size > 16) {
        af_horizontal_planar_ssse3(q, height, pitch, row_size, half_amount);
      }
      else if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_SSE2) && row_size > 16) {
        af_horizontal_planar_sse2(q, height, pitch, row_size, half_amount);
      }
      else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_AVX2) && row_size > 32) {
        af_horizontal_planar_uint16_t_avx2(q, height, pitch, row_size, half_amount, bits_per_pixel);
      }
      else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_SSE4_1) && row_size > 16) {
        af_horizontal_planar_uint16_t_sse41(q, height, pitch, row_size, half_amount, bits_per_pixel);
      }
      else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_SSE2) && row_size > 16) {
        af_horizontal_planar_uint16_t_sse2(q, height, pitch, row_size, half_amount, bits_per_pixel);
      }
      else if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_SSE2) && row_size > 16) {
        af_horizontal_planar_float_sse2(q, height, pitch, row_size, (float)amountd);
      }
      else
#elif defined(NEON_INTRINSICS)
      // any width: the NEON functions do the row ends per pixel
      if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_ARM_NEON)) {
        af_horizontal_planar_neon(q, height, pitch, row_size, half_amount);
      }
      else if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_ARM_NEON)) {
        af_horizontal_planar_uint16_t_neon(q, height, pitch, row_size, half_amount, bits_per_pixel);
      }
      else if (pixelsize == 4 && (env->GetCPUFlags() & CPUF_ARM_NEON)) {
        af_horizontal_planar_float_neon(q, height, pitch, row_size, (float)amountd);
      }
      else
#endif
      {
        switch (pixelsize) {
        case 1: af_horizontal_planar_c<uint8_t>(q, height, pitch, row_size, half_amount, bits_per_pixel); break;
        case 2: af_horizontal_planar_c<uint16_t>(q, height, pitch, row_size, half_amount, bits_per_pixel); break;
        default: // 4: float
          af_horizontal_planar_float_c(q, height, pitch, row_size, (float)amountd); break;
        }
      }
    }
  }
  else {
    if (vi.IsYUY2()) {
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSE2) && vi.width > 8) {
        af_horizontal_yuy2_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      }
      else
#endif
      {
        af_horizontal_yuy2_c(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      }
    }
    else if (vi.IsRGB32() || vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
      if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && vi.width > 4) {
        //this one is NOT in-place
        af_horizontal_rgb32_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      }
      else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE4_1) && vi.width > 2) {
        //this one is NOT in-place
        af_horizontal_rgb64_sse41(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount); // really width
      }
      else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && vi.width > 2) {
        //this one is NOT in-place
        af_horizontal_rgb64_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount); // really width
      }
      else
#endif
      {
        if (pixelsize == 1)
          af_horizontal_packed_rgb_c<uint8_t, 4>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
        else
          af_horizontal_packed_rgb_c<uint16_t, 4>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      }
    }
    else if (vi.IsRGB24() || vi.IsRGB48()) {
      if (pixelsize == 1)
        af_horizontal_packed_rgb_c<uint8_t, 3>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      else
        af_horizontal_packed_rgb_c<uint16_t, 3>(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
    }
  }

  return dst;
}


/************************************************
 *******   Sharpen/Blur Factory Methods   *******
 ***********************************************/

AVSValue __cdecl Create_Sharpen(AVSValue args, void*, IScriptEnvironment* env)
{
  const double amountH = args[1].AsFloat(), amountV = args[2].AsDblDef(amountH);

  if (std::isnan(amountH) || std::isnan(amountV) ||
      amountH < -1.5849625 || amountH > 1.0 || amountV < -1.5849625 || amountV > 1.0) // log2(3)
    env->ThrowError("Sharpen: arguments must be in the range -1.58 to 1.0");

  if (fabs(amountH) < 0.00002201361136) { // log2(1+1/65536)
    if (fabs(amountV) < 0.00002201361136) {
      return args[0].AsClip();
    }
    else {
      return new AdjustFocusV(amountV, args[0].AsClip());
    }
  }
  else {
    if (fabs(amountV) < 0.00002201361136) {
      return new AdjustFocusH(amountH, args[0].AsClip());
    }
    else {
      return new AdjustFocusH(amountH, new AdjustFocusV(amountV, args[0].AsClip()));
    }
  }
}

AVSValue __cdecl Create_Blur(AVSValue args, void*, IScriptEnvironment* env)
{
  const double amountH = args[1].AsFloat(), amountV = args[2].AsDblDef(amountH);

  if (std::isnan(amountH) || std::isnan(amountV) ||
      amountH < -1.0 || amountH > 1.5849625 || amountV < -1.0 || amountV > 1.5849625) // log2(3)
    env->ThrowError("Blur: arguments must be in the range -1.0 to 1.58");

  if (fabs(amountH) < 0.00002201361136) { // log2(1+1/65536)
    if (fabs(amountV) < 0.00002201361136) {
      return args[0].AsClip();
    }
    else {
      return new AdjustFocusV(-amountV, args[0].AsClip());
    }
  }
  else {
    if (fabs(amountV) < 0.00002201361136) {
      return new AdjustFocusH(-amountH, args[0].AsClip());
    }
    else {
      return new AdjustFocusH(-amountH, new AdjustFocusV(-amountV, args[0].AsClip()));
    }
  }
}




/***************************
 ****  TemporalSoften  *****
 **************************/

TemporalSoften::TemporalSoften( PClip _child, unsigned radius, unsigned luma_thresh,
                                unsigned chroma_thresh, int _scenechange, IScriptEnvironment* env )
  : GenericVideoFilter(_child),
  scenechange(_scenechange),
  luma_threshold(min(luma_thresh, 255u)),
  chroma_threshold(min(chroma_thresh, 255u)),
  kernel(2 * min(radius, (unsigned int)MAX_RADIUS) + 1)
{

  // declares the frame window read per output frame; the AviSynth+ cache ignores it (see AvsCache::SetCacheHints)
  // Comment added because it may serve later as test for memory-limited cashe shrink tests
  child->SetCacheHints(CACHE_WINDOW,kernel);

  if (vi.IsRGB24() || vi.IsRGB48()) {
    env->ThrowError("TemporalSoften: RGB24/48 Not supported, use ConvertToRGB32/48().");
  }

  if ((vi.IsRGB32() || vi.IsRGB64()) && (vi.width&1)) {
    env->ThrowError("TemporalSoften: RGB32/64 source must be multiple of 2 in width.");
  }

  if ((vi.IsYUY2()) && (vi.width&3)) {
    env->ThrowError("TemporalSoften: YUY2 source must be multiple of 4 in width.");
  }

  if (scenechange >= 255) {
    scenechange = 0;
  }

  if (scenechange>0 && (vi.IsRGB32() || vi.IsRGB64())) {
      env->ThrowError("TemporalSoften: Scenechange not available on RGB32/64");
  }

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  // original scenechange parameter always 0-255
  int factor;
  if (vi.IsPlanar()) // Y/YUV, no Planar RGB here
    factor = 1; // bitdepth independent. sad normalizes
  else
    factor = vi.BytesFromPixels(1) / pixelsize; // /pixelsize: correction for packed 16 bit rgb
  // 3.7.6: removed (width/32)*32. The original (2002) ISSE SAD summed only the first
  // (rowsize/32)*32 bytes of each row
  // Current SAD routines include the leftover columns since ages.
  scenechange *= (int64_t)vi.width * vi.height * factor; // 64 bit: exceeds int32 e.g. at 8K


  int c = 0;
  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    if (luma_thresh>0) {
      planes[c].planeId = PLANAR_Y;
      planes[c++].threshold = luma_thresh;
    }
    if (chroma_thresh>0 && vi.NumComponents() >= 3) { // Y/YA: no chroma planes
      planes[c].planeId = PLANAR_V;
      planes[c++].threshold =chroma_thresh;
      planes[c].planeId = PLANAR_U;
      planes[c++].threshold = chroma_thresh;
    }
  } else if (vi.IsYUY2()) {
    planes[c].planeId= DEFAULT_PLANE; // 0
    planes[c++].threshold=luma_thresh|(chroma_thresh<<8);
  } else if (vi.IsRGB()) {  // For RGB We use Luma.
    if (vi.IsPlanar()) {
      planes[c].planeId = PLANAR_G;
      planes[c++].threshold = luma_thresh;
      planes[c].planeId = PLANAR_B;
      planes[c++].threshold = luma_thresh;
      planes[c].planeId = PLANAR_R;
      planes[c++].threshold = luma_thresh;
    }
    else { // packed RGB
      planes[c].planeId = DEFAULT_PLANE; // 0
      planes[c++].threshold = luma_thresh;
    }
  }
  plane_count = c;
}

//offset is the initial value of x. Used when C routine processes only parts of frames after SSE/MMX paths do their job.
template<typename pixel_t, bool maxThreshold>
static void accumulate_line_c(BYTE* _c_plane, const BYTE** planeP, int planes, int offset, size_t rowsize, BYTE _threshold, int div, int bits_per_pixel) {
  pixel_t *c_plane = reinterpret_cast<pixel_t *>(_c_plane);

  typedef typename std::conditional < sizeof(pixel_t) <= 2, int, float>::type sum_t;
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type bigsum_t;
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, int>::type threshold_t;

  size_t width = rowsize / sizeof(pixel_t);

  threshold_t threshold = _threshold; // parameter 0..255
  if (std::is_floating_point<pixel_t>::value)
    threshold = (threshold_t)(threshold / 255.0f); // float
  else if constexpr(sizeof(pixel_t) == 2) {
    threshold = threshold * (uint16_t)(1 << (bits_per_pixel - 8)); // uint16_t, 10 bit: *4 16bit: *256
  }

  float average_multiplier = 0.0f;
  if constexpr(sizeof(pixel_t) == 2) {
    // Match the 16-bit SIMD paths: float reciprocal, round half up.
    average_multiplier = 1.0f / static_cast<float>(planes + 1);
  }

  for (size_t x = offset; x < width; ++x) {
    pixel_t current = c_plane[x];
    sum_t sum = current;

    for (int plane = planes - 1; plane >= 0; plane--) {
      pixel_t p = reinterpret_cast<const pixel_t *>(planeP[plane])[x];
      if (maxThreshold) {
        sum += p; // simple frame average mode
      }
      else {
        sum_t absdiff = std::abs(current - p);

        if (absdiff <= threshold) {
          sum += p;
        }
        else {
          sum += current;
        }
      }
    }
    if (std::is_floating_point<pixel_t>::value)
      c_plane[x] = (pixel_t)(sum / (planes + 1)); // float: simple average
    else if constexpr(sizeof(pixel_t) == 2)
      c_plane[x] = (pixel_t)(int)(static_cast<float>(sum) * average_multiplier + 0.5f); // half up, match SIMD paths
    else
      c_plane[x] = (pixel_t)(((bigsum_t)sum * div + 16384) >> 15); // div = 32768/(planes+1) for integer arithmetic
  }
}

static void accumulate_line_yuy2_c(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, BYTE threshold_luma, BYTE threshold_chroma, int div) {
  for (size_t x = 0; x < width; x+=2) {
    BYTE current_y = c_plane[x];
    BYTE current_c = c_plane[x+1];
    size_t sum_y = current_y;
    size_t sum_c = current_c;

    for (int plane = planes - 1; plane >= 0; plane--) {
      BYTE p_y = planeP[plane][x];
      BYTE p_c = planeP[plane][x+1];
      size_t absdiff_y = std::abs(current_y - p_y);
      size_t absdiff_c = std::abs(current_c - p_c);

      if (absdiff_y <= threshold_luma) {
        sum_y += p_y;
      } else {
        sum_y += current_y;
      }

      if (absdiff_c <= threshold_chroma) {
        sum_c += p_c;
      } else {
        sum_c += current_c;
      }
    }

    c_plane[x] = (BYTE)((sum_y * div + 16384) >> 15);
    c_plane[x+1] = (BYTE)((sum_c * div + 16384) >> 15);
  }
}

static void accumulate_line_yuy2(BYTE* c_plane, const BYTE** planeP, int planes, size_t width, BYTE threshold_luma, BYTE threshold_chroma, int div, IScriptEnvironment* env)
{
#ifdef INTEL_INTRINSICS
  if ((env->GetCPUFlags() & CPUF_SSE2) && width >= 16) {
    accumulate_line_sse2<false>(c_plane, planeP, planes, width, threshold_luma | (threshold_chroma << 8), div);
  } else
#ifdef X86_32
  if ((env->GetCPUFlags() & CPUF_MMX) && width >= 8) {
    accumulate_line_mmx(c_plane, planeP, planes, width, threshold_luma | (threshold_chroma << 8), div); //yuy2 is always at least mod8
  } else
#endif
#endif
    accumulate_line_yuy2_c(c_plane, planeP, planes, width, threshold_luma, threshold_chroma, div);
}

static void accumulate_line(BYTE* c_plane, const BYTE** planeP, int planes, size_t rowsize, BYTE threshold, int div, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
  // threshold == 255: simple average
  bool maxThreshold = (threshold == 255);
#ifdef INTEL_INTRINSICS
  const BYTE* planeP_rest[16]; // AVX2: offset plane pointers for the rest of the row
  if ((pixelsize <= 2) && (env->GetCPUFlags() & CPUF_AVX2) && rowsize >= 32) {
    // AVX2 for the mod32 width
    const size_t rowsize_mod32 = rowsize / 32 * 32;
    const int thresh16 = threshold << (bits_per_pixel - 8); // for 10-16 bits
    if (pixelsize == 1) {
      if (maxThreshold)
        accumulate_line_avx2<true>(c_plane, planeP, planes, rowsize_mod32, threshold | (threshold << 8), div);
      else
        accumulate_line_avx2<false>(c_plane, planeP, planes, rowsize_mod32, threshold | (threshold << 8), div);
    }
    else {
      if (maxThreshold) {
        if (bits_per_pixel < 16) accumulate_line_16_avx2<true, true>(c_plane, planeP, planes, rowsize_mod32, thresh16, bits_per_pixel);
        else accumulate_line_16_avx2<true, false>(c_plane, planeP, planes, rowsize_mod32, thresh16, bits_per_pixel);
      }
      else {
        if (bits_per_pixel < 16) accumulate_line_16_avx2<false, true>(c_plane, planeP, planes, rowsize_mod32, thresh16, bits_per_pixel);
        else accumulate_line_16_avx2<false, false>(c_plane, planeP, planes, rowsize_mod32, thresh16, bits_per_pixel);
      }
    }
    if (rowsize_mod32 == rowsize)
      return;
    // The rest <32 bytes falls through SSSE3/SSE4.1/C with modded pointers and sizes
    c_plane += rowsize_mod32;
    for (int i = 0; i < planes; i++)
      planeP_rest[i] = planeP[i] + rowsize_mod32;
    planeP = planeP_rest;
    rowsize -= rowsize_mod32;
  }
  if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE4) && rowsize >= 16) {
    // <maxThreshold, lessThan16bit>
    if(maxThreshold) {
      if(bits_per_pixel < 16)
        accumulate_line_16_sse41<true, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse41<true, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
    else {
      if (bits_per_pixel < 16)
        accumulate_line_16_sse41<false, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse41<false, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
  } else if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
    // <maxThreshold, lessThan16bit>
    if(maxThreshold) {
      if(bits_per_pixel < 16)
        accumulate_line_16_sse2<true, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<true, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
    else {
      if (bits_per_pixel < 16)
        accumulate_line_16_sse2<false, true>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
      else
        accumulate_line_16_sse2<false, false>(c_plane, planeP, planes, rowsize, threshold << (bits_per_pixel - 8), div, bits_per_pixel);
    }
  }
  else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSSE3) && rowsize >= 16) {
    if (maxThreshold) // <maxThreshold
      accumulate_line_ssse3<true>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
    else
      accumulate_line_ssse3<false>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
  } else if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
    if (maxThreshold)
      accumulate_line_sse2<true>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
    else
      accumulate_line_sse2<false>(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);
  }
  else
#ifdef X86_32
  if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_MMX) && rowsize >= 8) {
    size_t mod8_width = rowsize / 8 * 8;
    accumulate_line_mmx(c_plane, planeP, planes, rowsize, threshold | (threshold << 8), div);

    if (mod8_width != rowsize) {
      accumulate_line_c<uint8_t, false>(c_plane, planeP, planes, mod8_width, rowsize - mod8_width, threshold, div, bits_per_pixel);
    }
  } else
#endif
#elif defined(NEON_INTRINSICS)
  // any width: the NEON function does the rest of the row per pixel
  if (env->GetCPUFlags() & CPUF_ARM_NEON) {
    switch (pixelsize) {
    case 1:
      if (maxThreshold) accumulate_line_neon<uint8_t, true>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      else accumulate_line_neon<uint8_t, false>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      break;
    case 2:
      if (maxThreshold) accumulate_line_neon<uint16_t, true>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      else accumulate_line_neon<uint16_t, false>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      break;
    case 4:
      if (maxThreshold) accumulate_line_neon<float, true>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      else accumulate_line_neon<float, false>(c_plane, planeP, planes, rowsize, threshold, div, bits_per_pixel);
      break;
    }
  }
  else
#endif

    {
    switch(pixelsize) {
    case 1:
      if (maxThreshold)
        accumulate_line_c<uint8_t, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<uint8_t, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    case 2:
      if (maxThreshold)
        accumulate_line_c<uint16_t, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<uint16_t, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    case 4:
      if (maxThreshold)
        accumulate_line_c<float, true>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      else
        accumulate_line_c<float, false>(c_plane, planeP, planes, 0, rowsize, threshold, div, bits_per_pixel);
      break;
    }
   }
}

template<typename pixel_t>
static int64_t calculate_sad_c(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height)
{
  const pixel_t *ptr1 = reinterpret_cast<const pixel_t *>(cur_ptr);
  const pixel_t *ptr2 = reinterpret_cast<const pixel_t *>(other_ptr);
  size_t width = rowsize / sizeof(pixel_t);
  cur_pitch /= sizeof(pixel_t);
  other_pitch /= sizeof(pixel_t);

  // for fullframe float may loose precision
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, double, int64_t>::type sum_t;
  // for one row int is enough and faster than int64
  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, int>::type sumrow_t;
  sum_t sum = 0;

  for (size_t y = 0; y < height; ++y) {
    sumrow_t sumrow = 0;
    for (size_t x = 0; x < width; ++x) {
      sumrow += std::abs(ptr1[x] - ptr2[x]);
    }
    sum += sumrow;
    ptr1 += cur_pitch;
    ptr2 += other_pitch;
  }
  if (std::is_floating_point<pixel_t>::value)
    return (int64_t)(sum * 255); // scale 0..1 based sum to 8 bit range
  else
    return (int64_t)sum; // for int, scaling to 8 bit range is done outside
}

// sum of byte-diffs.
static int64_t calculate_sad(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height, int pixelsize, int bits_per_pixel, IScriptEnvironment* env) {
#ifdef INTEL_INTRINSICS
  // todo: sse for float
  if ((pixelsize == 1) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
    return calculate_sad_8_or_16_sse2<uint8_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#ifdef X86_32
  if ((pixelsize ==1 ) && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && rowsize >= 8) {
    return calculate_sad_isse<false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#endif
  // sse2 uint16_t
  if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
    return calculate_sad_8_or_16_sse2<uint16_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel-8);
  }
#endif
#ifdef NEON_INTRINSICS
  if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_ARM_DOTPROD))
    return calculate_sad_8_dp<false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  if (env->GetCPUFlags() & CPUF_ARM_NEON) {
    if (pixelsize == 1)
      return calculate_sad_8_or_16_neon<uint8_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
    if (pixelsize == 2)
      return calculate_sad_8_or_16_neon<uint16_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel - 8);
  }
#endif

  switch(pixelsize) {
  case 1: return calculate_sad_c<uint8_t>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  case 2: return calculate_sad_c<uint16_t>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel-8); // scale back to 8 bit range;
  default: // case 4
    return calculate_sad_c<float>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
}

PVideoFrame TemporalSoften::GetFrame(int n, IScriptEnvironment* env)
{
  int radius = (kernel-1) / 2;

  // Just skip if silly settings

  // returns the frame unchanged when nothing is left to process
  // (e.g. Y/YA + luma_threshold = 0 --> empty plane list)
  if ((!luma_threshold && !chroma_threshold) || !radius || plane_count == 0)
  {
    PVideoFrame ret = child->GetFrame(n, env); // P.F.
    return ret;
  }

  bool planeDisabled[16];

  for (int p = 0; p<16; p++) {
    planeDisabled[p] = false;
  }

  std::vector<PVideoFrame> frames;
  frames.reserve(kernel);

  for (int p = n - radius; p <= n + radius; ++p) {
    frames.emplace_back(child->GetFrame(clamp(p, 0, vi.num_frames - 1), env));
  }

  // P.F. 16.04.06 leak fix r1841 after 8 days of bug chasing:
  // Reason #1 of the random QTGMC memory leaks (stuck frame refcounts)
  // MakeWritable alters the pointer if it is not yet writeable, thus the original PVideoFrame won't be freed (refcount decremented)
  // To fix this, we leave the frame[] array in its place and copy frame[radius] to CenterFrame and make further write operations on this new frame.
  // env->MakeWritable(&frames[radius]); // old culprit line. if not yet writeable -> gives another pointer
  PVideoFrame CenterFrame = frames[radius];
  env->MakeWritable(&CenterFrame);

  for (int c = 0; c < plane_count; c++) {
    const BYTE* planeP[16];
    const BYTE* planeP2[16];
    int planePitch[16];
    int planePitch2[16];

    int current_thresh = planes[c].threshold;  // Threshold for current plane.
    int d = 0;
    for (int i = 0; i<radius; i++) { // Fetch all planes sequencially
      planePitch[d] = frames[i]->GetPitch(planes[c].planeId);
      planeP[d++] = frames[i]->GetReadPtr(planes[c].planeId);
    }

//    BYTE* c_plane = frames[radius]->GetWritePtr(planes[c]);
    BYTE* c_plane = CenterFrame->GetWritePtr(planes[c].planeId); // P.F. using CenterFrame for write access

    for (int i = 1; i<=radius; i++) { // Fetch all planes sequencially
      planePitch[d] = frames[radius+i]->GetPitch(planes[c].planeId);
      planeP[d++] = frames[radius+i]->GetReadPtr(planes[c].planeId);
    }

    int rowsize = CenterFrame->GetRowSize(planes[c].planeId|PLANAR_ALIGNED);
    int h = CenterFrame->GetHeight(planes[c].planeId);
    int pitch = CenterFrame->GetPitch(planes[c].planeId);

    if (scenechange>0) {
      int d2 = 0;
      bool skiprest = false;
      for (int i = radius-1; i>=0; i--) { // Check frames backwards
        if ((!skiprest) && (!planeDisabled[i])) {
          int64_t sad = calculate_sad(c_plane, planeP[i], pitch, planePitch[i], CenterFrame->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
          if (sad < scenechange) {
            planePitch2[d2] = planePitch[i];
            planeP2[d2++] = planeP[i];
          } else {
            skiprest = true;
          }
          planeDisabled[i] = skiprest;  // Disable this frame on next plane (so that Y can affect UV)
        } else {
          planeDisabled[i] = true;
        }
      }
      skiprest = false;
      for (int i = radius; i < 2*radius; i++) { // Check forward frames
        if ((!skiprest)  && (!planeDisabled[i])) {   // Disable this frame on next plane (so that Y can affect UV)
          int64_t sad = calculate_sad(c_plane, planeP[i], pitch, planePitch[i], CenterFrame->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
          if (sad < scenechange) {
            planePitch2[d2] = planePitch[i];
            planeP2[d2++] = planeP[i];
          } else {
            skiprest = true;
          }
          planeDisabled[i] = skiprest;
        } else {
          planeDisabled[i] = true;
        }
      }

      //Copy back
      for (int i = 0; i<d2; i++) {
        planeP[i] = planeP2[i];
        planePitch[i] = planePitch2[i];
      }
      d = d2;
    }

    if (d < 1)
    {
      // Memory leak reason #2 r1841: this wasn't here before return
      for (int i = 0; i < kernel; ++i)
        frames[i] = nullptr;
      // return frames[radius];
      return CenterFrame; // return the modified frame
    }

    int c_div = 32768/(d+1);  // We also have the tetplane included, thus d+1.
    if (current_thresh) {
      // for threshold==255 -> simple average
      for (int y = 0; y<h; y++) { // One line at the time
        if (vi.IsYUY2()) {
          accumulate_line_yuy2(c_plane, planeP, d, rowsize, luma_threshold, chroma_threshold, c_div, env);
        } else {
          accumulate_line(c_plane, planeP, d, rowsize, current_thresh, c_div, pixelsize, bits_per_pixel, env);
        }
        for (int p = 0; p<d; p++)
          planeP[p] += planePitch[p];
        c_plane += pitch;
      }
    } else { // Just maintain the plane
    }
  }

  //  PVideoFrame result = frames[radius]; // we are using CenterFrame instead
  //  return result;
  return CenterFrame;
}


AVSValue __cdecl TemporalSoften::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new TemporalSoften( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),
                             args[3].AsInt(), args[4].AsInt(0),/*args[5].AsInt(1),*/env ); //ignore mode parameter
}



/****************************
 ****  Spatial Soften   *****
 ***************************/

SpatialSoften::SpatialSoften( PClip _child, int _radius, unsigned _luma_threshold,
                              unsigned _chroma_threshold, IScriptEnvironment* env )
  : GenericVideoFilter(_child),
    luma_threshold(_luma_threshold), chroma_threshold(_chroma_threshold),
    diameter(_radius * 2 + 1)
{
  if (!vi.IsYUY2())
    env->ThrowError("SpatialSoften: requires YUY2 input");
}


PVideoFrame SpatialSoften::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  int row_size = src->GetRowSize();

  for (int y=0; y<vi.height; ++y)
  {
    const BYTE* line[65];    // better not make diameter bigger than this...
    for (int h=0; h<diameter; ++h)
      line[h] = &srcp[src_pitch * clamp(y+h-(diameter>>1), 0, vi.height-1)];
    int x;

    int edge = (diameter+1) & -4;
    for (x=0; x<edge; ++x)  // diameter-1 == (diameter>>1) * 2
      dstp[y*dst_pitch + x] = srcp[y*src_pitch + x];
    for (; x < row_size - edge; x+=2)
    {
      int cnt=0, _y=0, _u=0, _v=0;
      int xx = x | 3;
      int Y = srcp[y*src_pitch + x], U = srcp[y*src_pitch + xx - 2], V = srcp[y*src_pitch + xx];
      for (int h=0; h<diameter; ++h)
      {
        for (int w = -diameter+1; w < diameter; w += 2)
        {
          int xw = (x+w) | 3;
          if (IsClose(line[h][x+w], Y, luma_threshold) && IsClose(line[h][xw-2], U,
                      chroma_threshold) && IsClose(line[h][xw], V, chroma_threshold))
          {
            ++cnt; _y += line[h][x+w]; _u += line[h][xw-2]; _v += line[h][xw];
          }
        }
      }
      dstp[y*dst_pitch + x] = (_y + (cnt>>1)) / cnt;
      if (!(x&3)) {
        dstp[y*dst_pitch + x+1] = (_u + (cnt>>1)) / cnt;
        dstp[y*dst_pitch + x+3] = (_v + (cnt>>1)) / cnt;
      }
    }
    for (; x<row_size; ++x)
      dstp[y*dst_pitch + x] = srcp[y*src_pitch + x];
  }

  return dst;
}


AVSValue __cdecl SpatialSoften::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  return new SpatialSoften( args[0].AsClip(), args[1].AsInt(), args[2].AsInt(),
                            args[3].AsInt(), env );
}
