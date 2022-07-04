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

#include "focus.h"
#ifdef INTEL_INTRINSICS
#include "intel/focus_sse.h"
#include "intel/focus_avx2.h"
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
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5
  int max_pixel_value = (1 << bits_per_pixel) - 1;

  pixel_t * dstp = reinterpret_cast<pixel_t *>(dstp8);
  pixel_t * line_buf = reinterpret_cast<pixel_t *>(line_buf8);
  int pitch = pitch8 / sizeof(pixel_t);

  for (int y = height-1; y>0; --y) {
    for (int x = 0; x < width; ++x) {
      pixel_t a;
      // Note: ScaledPixelClip is overloaded. With int64_t parameter and uint16_t result works for 16 bit
      if constexpr(sizeof(pixel_t) == 1)
        a = ScaledPixelClip((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight));
      else
        a = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x+pitch]) * outer_weight), max_pixel_value);
      line_buf[x] = dstp[x];
      dstp[x] = a;
    }
    dstp += pitch;
  }
  for (int x = 0; x < width; ++x) { // Last row - map centre as lower
    if constexpr(sizeof(pixel_t) == 1)
      dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight));
    else
      dstp[x] = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * center_weight + (line_buf[x] + dstp[x]) * outer_weight), max_pixel_value);
  }
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
  else
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_SSE2) && width >= 16) {
    //pitch of aligned frames is always >= 16 so we'll just process some garbage if width is not mod16
    af_vertical_sse2(line_buf, dstp, (int)height, (int)pitch, (int)width, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_AVX2) && row_size >= 32) {
    af_vertical_uint16_t_avx2(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE4_1) && row_size >= 16) {
    af_vertical_uint16_t_sse41(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else if (sizeof(pixel_t) == 2 && (env->GetCPUFlags() & CPUF_SSE2) && row_size >= 16) {
    af_vertical_uint16_t_sse2(line_buf, dstp, (int)height, (int)pitch, (int)row_size, half_amount);
  }
  else
#ifdef X86_32
  if (sizeof(pixel_t) == 1 && (env->GetCPUFlags() & CPUF_MMX) && width >= 8)
  {
    size_t mod8_width = width / 8 * 8;
    af_vertical_mmx(line_buf, dstp, height, pitch, mod8_width, half_amount);
    if (mod8_width != width) {
      //yes, this is bad for caching. MMX shouldn't be used these days anyway
      af_vertical_c<uint8_t>(line_buf, dstp + mod8_width, height, pitch, width - mod8_width, half_amount, bits_per_pixel);
    }
  } else
#endif
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
      const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
      const int *planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

      for (int cplane = 0; cplane < 3; cplane++) {
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


// --------------------------------------
// Blur/Sharpen Horizontal RGB32 C++ Code
// --------------------------------------

template<typename pixel_t, typename weight_t>
static AVS_FORCEINLINE void af_horizontal_rgb32_process_line_c(pixel_t b_left, pixel_t g_left, pixel_t r_left, pixel_t a_left, pixel_t *dstp, size_t width, weight_t center_weight, weight_t outer_weight) {
  size_t x;
  for (x = 0; x < width-1; ++x)
  {
    pixel_t b = ScaledPixelClip((weight_t)(dstp[x*4+0] * center_weight + (b_left + dstp[x*4+4]) * outer_weight));
    b_left = dstp[x*4+0];
    dstp[x*4+0] = b;
    pixel_t g = ScaledPixelClip((weight_t)(dstp[x*4+1] * center_weight + (g_left + dstp[x*4+5]) * outer_weight));
    g_left = dstp[x*4+1];
    dstp[x*4+1] = g;
    pixel_t r = ScaledPixelClip((weight_t)(dstp[x*4+2] * center_weight + (r_left + dstp[x*4+6]) * outer_weight));
    r_left = dstp[x*4+2];
    dstp[x*4+2] = r;
    pixel_t a = ScaledPixelClip((weight_t)(dstp[x*4+3] * center_weight + (a_left + dstp[x*4+7]) * outer_weight));
    a_left = dstp[x*4+3];
    dstp[x*4+3] = a;
  }
  dstp[x*4+0] = ScaledPixelClip((weight_t)(dstp[x*4+0] * center_weight + (b_left + dstp[x*4+0]) * outer_weight));
  dstp[x*4+1] = ScaledPixelClip((weight_t)(dstp[x*4+1] * center_weight + (g_left + dstp[x*4+1]) * outer_weight));
  dstp[x*4+2] = ScaledPixelClip((weight_t)(dstp[x*4+2] * center_weight + (r_left + dstp[x*4+2]) * outer_weight));
  dstp[x*4+3] = ScaledPixelClip((weight_t)(dstp[x*4+3] * center_weight + (a_left + dstp[x*4+3]) * outer_weight));
}

template<typename pixel_t>
static void af_horizontal_rgb32_64_c(BYTE* dstp8, size_t height, size_t pitch8, size_t width, int half_amount) {
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5

  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  size_t pitch = pitch8 / sizeof(pixel_t);

  for (size_t y = height; y>0; --y)
  {
    pixel_t b_left = dstp[0];
    pixel_t g_left = dstp[1];
    pixel_t r_left = dstp[2];
    pixel_t a_left = dstp[3];
    af_horizontal_rgb32_process_line_c<pixel_t, weight_t>(b_left, g_left, r_left, a_left, dstp, width, center_weight, outer_weight);
    dstp += pitch;
  }

}


// -------------------------------------
// Blur/Sharpen Horizontal YUY2 C++ Code
// -------------------------------------

static void af_horizontal_yuy2_c(BYTE* p, int height, int pitch, int width, int amount) {
  const int center_weight = amount*2;
  const int outer_weight = 32768-amount;
  for (int y0 = height; y0>0; --y0)
  {
    BYTE yy = p[0];
    BYTE uv = p[1];
    BYTE vu = p[3];
    int x;
    for (x = 0; x < width-2; ++x)
    {
      BYTE y = ScaledPixelClip(p[x*2+0] * center_weight + (yy + p[x*2+2]) * outer_weight);
      yy   = p[x*2+0];
      p[x*2+0] = y;
      BYTE w = ScaledPixelClip(p[x*2+1] * center_weight + (uv + p[x*2+5]) * outer_weight);
      uv   = vu;
      vu   = p[x*2+1];
      p[x*2+1] = w;
    }
    BYTE y     = ScaledPixelClip(p[x*2+0] * center_weight + (yy + p[x*2+2]) * outer_weight);
    yy       = p[x*2+0];
    p[x*2+0] = y;
    p[x*2+1] = ScaledPixelClip(p[x*2+1] * center_weight + (uv + p[x*2+1]) * outer_weight);
    p[x*2+2] = ScaledPixelClip(p[x*2+2] * center_weight + (yy + p[x*2+2]) * outer_weight);
    p[x*2+3] = ScaledPixelClip(p[x*2+3] * center_weight + (vu + p[x*2+3]) * outer_weight);

    p += pitch;
  }
}


// --------------------------------------
// Blur/Sharpen Horizontal RGB24 C++ Code
// --------------------------------------

template<typename pixel_t>
static void af_horizontal_rgb24_48_c(BYTE* dstp8, int height, int pitch8, int width, int half_amount) {
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t;
  // kernel:[(1-1/2^_amount)/2, 1/2^_amount, (1-1/2^_amount)/2]
  weight_t center_weight = half_amount*2;    // *2: 16 bit scaled arithmetic, but the converted amount parameter scaled is only 15 bits
  weight_t outer_weight = 32768-half_amount; // (1-1/2^_amount)/2  32768 = 0.5

  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp8);
  int pitch = pitch8 / sizeof(pixel_t);
  for (int y = height; y>0; --y)
  {
    pixel_t bb = dstp[0];
    pixel_t gg = dstp[1];
    pixel_t rr = dstp[2];
    int x;
    for (x = 0; x < width-1; ++x)
    {
      // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
      pixel_t b = ScaledPixelClip((weight_t)(dstp[x*3+0] * center_weight + (bb + dstp[x*3+3]) * outer_weight));
      bb = dstp[x*3+0]; dstp[x*3+0] = b;
      pixel_t g = ScaledPixelClip((weight_t)(dstp[x*3+1] * center_weight + (gg + dstp[x*3+4]) * outer_weight));
      gg = dstp[x*3+1]; dstp[x*3+1] = g;
      pixel_t r = ScaledPixelClip((weight_t)(dstp[x*3+2] * center_weight + (rr + dstp[x*3+5]) * outer_weight));
      rr = dstp[x*3+2]; dstp[x*3+2] = r;
    }
    dstp[x*3+0] = ScaledPixelClip((weight_t)(dstp[x*3+0] * center_weight + (bb + dstp[x*3+0]) * outer_weight));
    dstp[x*3+1] = ScaledPixelClip((weight_t)(dstp[x*3+1] * center_weight + (gg + dstp[x*3+1]) * outer_weight));
    dstp[x*3+2] = ScaledPixelClip((weight_t)(dstp[x*3+2] * center_weight + (rr + dstp[x*3+2]) * outer_weight));
    dstp += pitch;
  }
}

// -------------------------------------
// Blur/Sharpen Horizontal YV12 C++ Code
// -------------------------------------

// for linker reasons these forceinlined functions appear in C, intel sse2 and avx2 source as well
template<typename pixel_t>
static AVS_FORCEINLINE void af_horizontal_planar_process_line_c(pixel_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight) {
  size_t x;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  typedef typename std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight));
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClip((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight));
}

static AVS_FORCEINLINE void af_horizontal_planar_process_line_uint16_c(uint16_t left, BYTE *dstp8, size_t row_size, int center_weight, int outer_weight, int bits_per_pixel) {
  size_t x;
  typedef uint16_t pixel_t;
  pixel_t* dstp = reinterpret_cast<pixel_t *>(dstp8);
  const int max_pixel_value = (1 << bits_per_pixel) - 1; // clamping on 10-12-14-16 bitdepth
  typedef std::conditional < sizeof(pixel_t) == 1, int, int64_t>::type weight_t; // for calling the right ScaledPixelClip()
  size_t width = row_size / sizeof(pixel_t);
  for (x = 0; x < width-1; ++x) {
    pixel_t temp = (pixel_t)ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x+1]) * (weight_t)outer_weight), max_pixel_value);
    left = dstp[x];
    dstp[x] = temp;
  }
  // ScaledPixelClip has 2 overloads: BYTE/uint16_t (int/int64 i)
  dstp[x] = ScaledPixelClipEx((weight_t)(dstp[x] * (weight_t)center_weight + (left + dstp[x]) * (weight_t)outer_weight), max_pixel_value);
}

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
    int center_weight = int(half_amount*2);
    int outer_weight = int(32768-half_amount);
    pixel_t left;
    for (size_t y = height; y>0; --y) {
        left = dstp[0];
        if constexpr(sizeof(pixel_t) == 1)
          af_horizontal_planar_process_line_c<pixel_t>(left, (BYTE *)dstp, row_size, center_weight, outer_weight);
        else
          af_horizontal_planar_process_line_uint16_c(left, (BYTE *)dstp, row_size, center_weight, outer_weight, bits_per_pixel);
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

  const int planesYUV[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A};
  const int planesRGB[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A};
  const int *planes = vi.IsYUV() || vi.IsYUVA() ? planesYUV : planesRGB;

  int pixelsize = vi.ComponentSize();

  if (vi.IsPlanar()) {
    copy_frame(src, dst, env, planes, vi.NumComponents() ); //planar processing is always in-place
    int bits_per_pixel = vi.BitsPerComponent();
    for(int cplane=0;cplane<3;cplane++) {
      int plane = planes[cplane];
      int row_size = dst->GetRowSize(plane);
      BYTE* q = dst->GetWritePtr(plane);
      int pitch = dst->GetPitch(plane);
      int height = dst->GetHeight(plane);
#ifdef INTEL_INTRINSICS
      if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_AVX2) && row_size > 32) {
        af_horizontal_planar_avx2(q, height, pitch, row_size, half_amount);
      }
      else
        if (pixelsize==1 && (env->GetCPUFlags() & CPUF_SSE2) && row_size > 16) {
        af_horizontal_planar_sse2(q, height, pitch, row_size, half_amount);
      } else
#ifdef X86_32
        if (pixelsize == 1 && (env->GetCPUFlags() & CPUF_MMX) && row_size > 8) {
          af_horizontal_planar_mmx(q,height,pitch,row_size,half_amount);
        } else
#endif
        if (pixelsize == 2 && (env->GetCPUFlags() & CPUF_AVX2) && row_size > 32) {
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
  } else {
    if (vi.IsYUY2()) {
      BYTE* q = dst->GetWritePtr();
      const int pitch = dst->GetPitch();
#ifdef INTEL_INTRINSICS
      if ((env->GetCPUFlags() & CPUF_SSE2) && vi.width>8) {
        af_horizontal_yuy2_sse2(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#ifdef X86_32
      if ((env->GetCPUFlags() & CPUF_MMX) && vi.width>8) {
        af_horizontal_yuy2_mmx(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#endif
#endif
      {
        copy_frame(src, dst, env, planesYUV, 1); //in-place
        af_horizontal_yuy2_c(q,vi.height,pitch,vi.width,half_amount);
      }
    }
    else if (vi.IsRGB32() || vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
      if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_SSE2) && vi.width>4) {
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
#ifdef X86_32
      if ((pixelsize==1) && (env->GetCPUFlags() & CPUF_MMX) && vi.width > 2)
      { //so as this one
        af_horizontal_rgb32_mmx(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), vi.height, vi.width, half_amount);
      } else
#endif
#endif
      {
        copy_frame(src, dst, env, planesYUV, 1);
        if(pixelsize==1)
          af_horizontal_rgb32_64_c<uint8_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
        else
          af_horizontal_rgb32_64_c<uint16_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
      }
    } else if (vi.IsRGB24() || vi.IsRGB48()) {
      copy_frame(src, dst, env, planesYUV, 1);
      if(pixelsize==1)
        af_horizontal_rgb24_48_c<uint8_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
      else
        af_horizontal_rgb24_48_c<uint16_t>(dst->GetWritePtr(), vi.height, dst->GetPitch(), vi.width, half_amount);
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

  if (amountH < -1.5849625 || amountH > 1.0 || amountV < -1.5849625 || amountV > 1.0) // log2(3)
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

  if (amountH < -1.0 || amountH > 1.5849625 || amountV < -1.0 || amountV > 1.5849625) // log2(3)
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
  scenechange *= ((vi.width/32)*32)*vi.height*factor; // why /*32?


  int c = 0;
  if (vi.IsPlanar() && (vi.IsYUV() || vi.IsYUVA())) {
    if (luma_thresh>0) {
      planes[c].planeId = PLANAR_Y;
      planes[c++].threshold = luma_thresh;
    }
    if (chroma_thresh>0) {
      planes[c].planeId = PLANAR_V;
      planes[c++].threshold =chroma_thresh;
      planes[c].planeId = PLANAR_U;
      planes[c++].threshold = chroma_thresh;
    }
  } else if (vi.IsYUY2()) {
    planes[c].planeId=0;
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
      planes[c].planeId = 0;
      planes[c++].threshold = luma_thresh;
    }
  }
  planes[c].planeId=0;
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
    return (int64_t)calculate_sad_sse2<false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#ifdef X86_32
  if ((pixelsize ==1 ) && (env->GetCPUFlags() & CPUF_INTEGER_SSE) && rowsize >= 8) {
    return (int64_t)calculate_sad_isse(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height);
  }
#endif
  // sse2 uint16_t
  if ((pixelsize == 2) && (env->GetCPUFlags() & CPUF_SSE2) && rowsize >= 16) {
    return calculate_sad_8_or_16_sse2<uint16_t, false>(cur_ptr, other_ptr, cur_pitch, other_pitch, rowsize, height) >> (bits_per_pixel-8);
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
  int c = 0;

  // Just skip if silly settings

  if ((!luma_threshold && !chroma_threshold) || !radius)
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

  do {
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
          int sad = (int)calculate_sad(c_plane, planeP[i], pitch, planePitch[i], CenterFrame->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
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
          int sad = (int)calculate_sad(c_plane, planeP[i], pitch, planePitch[i], CenterFrame->GetRowSize(planes[c].planeId), h, pixelsize, bits_per_pixel, env);
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
    c++;
  } while (planes[c].planeId);

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
