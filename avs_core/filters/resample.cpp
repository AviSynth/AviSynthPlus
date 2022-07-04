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

#include "resample.h"
#ifdef INTEL_INTRINSICS
#include "intel/resample_sse.h"
#include "intel/resample_avx2.h"
#include "intel/turn_sse.h"
#endif
#include <avs/config.h>

#include "transform.h"
#include "turn.h"
#include <avs/alignment.h>
#include <avs/minmax.h>
#include "../convert/convert_planar.h"
#include "../convert/convert_yuy2.h"

#include <type_traits>
#include <algorithm>

/***************************************
 ***** Vertical Resizer Assembly *******
 ***************************************/

template<typename pixel_t>
static void resize_v_planar_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(bits_per_pixel);
  AVS_UNUSED(storage);

  pixel_t* src0 = (pixel_t*)src;
  pixel_t* dst0 = (pixel_t*)dst;
  dst_pitch = dst_pitch / sizeof(pixel_t);

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const pixel_t* src_ptr = src0 + pitch_table[offset] / sizeof(pixel_t);

    memcpy(dst0, src_ptr, width * sizeof(pixel_t));

    dst0 += dst_pitch;
  }
}

template<typename pixel_t>
static void resize_v_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel, const int* pitch_table, const void* storage)
{
  AVS_UNUSED(src_pitch);
  AVS_UNUSED(storage);

  int filter_size = program->filter_size;

  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
  coeff_t* current_coeff;

  if (!std::is_floating_point<pixel_t>::value)
    current_coeff = (coeff_t*)program->pixel_coefficient;
  else
    current_coeff = (coeff_t*)program->pixel_coefficient_float;

  pixel_t* src0 = (pixel_t*)src;
  pixel_t* dst0 = (pixel_t*)dst;
  dst_pitch = dst_pitch / sizeof(pixel_t);

  pixel_t limit = 0;
  if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr (sizeof(pixel_t) == 1) limit = 255;
    else if constexpr (sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
  }

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const pixel_t* src_ptr = src0 + pitch_table[offset] / sizeof(pixel_t);

    for (int x = 0; x < width; x++) {
      // todo: check whether int result is enough for 16 bit samples (can an int overflow because of 16384 scale or really need int64_t?)
      typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, int64_t, float>::type >::type result;
      result = 0;
      for (int i = 0; i < filter_size; i++) {
        result += (src_ptr + pitch_table[i] / sizeof(pixel_t))[x] * current_coeff[i];
      }
      if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
        if constexpr (sizeof(pixel_t) == 1)
          result = (result + (1 << (FPScale8bits - 1))) >> FPScale8bits;
        else if constexpr (sizeof(pixel_t) == 2)
          result = (result + (1 << (FPScale16bits - 1))) >> FPScale16bits;
        result = clamp(result, decltype(result)(0), decltype(result)(limit));
      }
      dst0[x] = (pixel_t)result;
    }

    dst0 += dst_pitch;
    current_coeff += filter_size;
  }
}

AVS_FORCEINLINE static void resize_v_create_pitch_table(int* table, int pitch, int height) {
  table[0] = 0;
  for (int i = 1; i < height; i++) {
    table[i] = table[i - 1] + pitch;
  }
}



/***************************************
 ********* Horizontal Resizer** ********
 ***************************************/
#if 0
static void resize_h_pointresize(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  AVS_UNUSED(bits_per_pixel);

  int wMod4 = width / 4 * 4;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < wMod4; x += 4) {
#define pixel(a) src[program->pixel_offset[x+a]]
      unsigned int data = (pixel(3) << 24) + (pixel(2) << 16) + (pixel(1) << 8) + pixel(0);
#undef pixel
      * ((unsigned int*)(dst + x)) = data;
    }

    for (int x = wMod4; x < width; x++) {
      dst[x] = src[program->pixel_offset[x]];
    }

    dst += dst_pitch;
    src += src_pitch;
  }
}
#endif


template<typename pixel_t>
static void resize_h_c_planar(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  int filter_size = program->filter_size;

  typedef typename std::conditional < std::is_floating_point<pixel_t>::value, float, short>::type coeff_t;
  coeff_t* current_coeff;

  pixel_t limit = 0;
  if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
    if constexpr (sizeof(pixel_t) == 1) limit = 255;
    else if constexpr (sizeof(pixel_t) == 2) limit = pixel_t((1 << bits_per_pixel) - 1);
  }

  src_pitch = src_pitch / sizeof(pixel_t);
  dst_pitch = dst_pitch / sizeof(pixel_t);

  pixel_t* src0 = (pixel_t*)src;
  pixel_t* dst0 = (pixel_t*)dst;

  // external loop y is much faster
  for (int y = 0; y < height; y++) {
    if (!std::is_floating_point<pixel_t>::value)
      current_coeff = (coeff_t*)program->pixel_coefficient;
    else
      current_coeff = (coeff_t*)program->pixel_coefficient_float;
    for (int x = 0; x < width; x++) {
      int begin = program->pixel_offset[x];
      // todo: check whether int result is enough for 16 bit samples (can an int overflow because of 16384 scale or really need int64_t?)
      typename std::conditional < sizeof(pixel_t) == 1, int, typename std::conditional < sizeof(pixel_t) == 2, int64_t, float>::type >::type result;
      result = 0;
      for (int i = 0; i < filter_size; i++) {
        result += (src0 + y * src_pitch)[(begin + i)] * current_coeff[i];
      }
      if (!std::is_floating_point<pixel_t>::value) {  // floats are unscaled and uncapped
        if constexpr (sizeof(pixel_t) == 1)
          result = (result + (1 << (FPScale8bits - 1))) >> FPScale8bits;
        else if constexpr (sizeof(pixel_t) == 2)
          result = (result + (1 << (FPScale16bits - 1))) >> FPScale16bits;
        result = clamp(result, decltype(result)(0), decltype(result)(limit));
      }
      (dst0 + y * dst_pitch)[x] = (pixel_t)result;
      current_coeff += filter_size;
    }
  }
}

/********************************************************************
***** Declare index of new filters for Avisynth's filter engine *****
********************************************************************/

extern const AVSFunction Resample_filters[] = {
  { "PointResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_PointResize },
  { "BilinearResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_BilinearResize },
  { "BicubicResize",  BUILTIN_FUNC_PREFIX, "cii[b]f[c]f[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_BicubicResize },
  { "LanczosResize",  BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_LanczosResize},
  { "Lanczos4Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Lanczos4Resize},
  { "BlackmanResize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_BlackmanResize},
  { "Spline16Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline16Resize},
  { "Spline36Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline36Resize},
  { "Spline64Resize", BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f", FilteredResize::Create_Spline64Resize},
  { "GaussResize",    BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[p]f", FilteredResize::Create_GaussianResize},
  { "SincResize",     BUILTIN_FUNC_PREFIX, "cii[src_left]f[src_top]f[src_width]f[src_height]f[taps]i", FilteredResize::Create_SincResize},
  /**
    * Resize(PClip clip, dst_width, dst_height [src_left, src_top, src_width, int src_height,] )
    *
    * src_left et al.   =  when these optional arguments are given, the filter acts just like
    *                      a Crop was performed with those parameters before resizing, only faster
   **/

  { 0 }
};


FilteredResizeH::FilteredResizeH(PClip _child, double subrange_left, double subrange_width,
  int target_width, ResamplingFunction* func, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  resampling_program_luma(0), resampling_program_chroma(0),
  src_pitch_table_luma(0),
  filter_storage_luma(0), filter_storage_chroma(0)
{
  src_width = vi.width;
  src_height = vi.height;
  dst_width = target_width;
  dst_height = vi.height;

  pixelsize = vi.ComponentSize(); // AVS16
  bits_per_pixel = vi.BitsPerComponent();
  grey = vi.IsY();

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (target_width <= 0) {
    env->ThrowError("Resize: Width must be greater than 0.");
  }

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int mask = (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1;

    if (target_width & mask)
      env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
  }

  // Main resampling program
  resampling_program_luma = func->GetResamplingProgram(vi.width, subrange_left, subrange_width, target_width, bits_per_pixel, env);
  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
    const int div = 1 << shift;


    resampling_program_chroma = func->GetResamplingProgram(
      vi.width >> shift,
      subrange_left / div,
      subrange_width / div,
      target_width >> shift,
      bits_per_pixel,
      env);
  }

#ifdef INTEL_INTRINSICS
  // r2592+: no target_width mod4 check, (old avs needed for unaligned frames?)
  int cpu = env->GetCPUFlags();
  fast_resize = (cpu & CPUF_SSSE3) == CPUF_SSSE3 && vi.IsPlanar();
  bool has_sse2 = (cpu & CPUF_SSE2) != 0;
#else
  int cpu = 0;
  fast_resize = false;
#endif

#if 0
  if (false && resampling_program_luma->filter_size == 1 && vi.IsPlanar()) {
    // dead code?
    fast_resize = true;
    resampler_h_luma = resize_h_pointresize;
    resampler_h_chroma = resize_h_pointresize;
  }
  else
#endif
    if (!fast_resize) {

      // nonfast-resize: using V resizer for horizontal resizing between a turnleft/right

      // Create resampling program and pitch table
      src_pitch_table_luma = new int[vi.width];

      resampler_luma = FilteredResizeV::GetResampler(cpu, true, pixelsize, bits_per_pixel, filter_storage_luma, resampling_program_luma);
      if (vi.width < resampling_program_luma->filter_size) {
        env->ThrowError("Source width (%d) is too small for this resizing method, must be minimum of %d", vi.width, resampling_program_luma->filter_size);
      }
      if (vi.IsPlanar() && !grey && !isRGBPfamily) {
        resampler_chroma = FilteredResizeV::GetResampler(cpu, true, pixelsize, bits_per_pixel, filter_storage_chroma, resampling_program_chroma);
        const int width_UV = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
        if (width_UV < resampling_program_luma->filter_size) {
          env->ThrowError("Source chroma width (%d) is too small for this resizing method, must be minimum of %d", width_UV, resampling_program_chroma->filter_size);
        }
      }

      // Temporary buffer size
      temp_1_pitch = AlignNumber(vi.BytesFromPixels(src_height), FRAME_ALIGN);
      temp_2_pitch = AlignNumber(vi.BytesFromPixels(dst_height), FRAME_ALIGN);

      resize_v_create_pitch_table(src_pitch_table_luma, temp_1_pitch, src_width);

      // Initialize Turn function
      // see turn.cpp
      if (vi.IsRGB24()) {
#ifdef INTEL_INTRINSICS
        // no intel intentionally
#endif
        turn_left = turn_left_rgb24;
        turn_right = turn_right_rgb24;
      }
      else if (vi.IsRGB32()) {
#ifdef INTEL_INTRINSICS
        if (has_sse2) {
          turn_left = turn_left_rgb32_sse2;
          turn_right = turn_right_rgb32_sse2;
        }
        else
#endif
        {
          turn_left = turn_left_rgb32_c;
          turn_right = turn_right_rgb32_c;
        }
      }
      else if (vi.IsRGB48()) {
#ifdef INTEL_INTRINSICS
        // no intel intentionally
#endif
        turn_left = turn_left_rgb48_c;
        turn_right = turn_right_rgb48_c;
      }
      else if (vi.IsRGB64()) {
#ifdef INTEL_INTRINSICS
        if (has_sse2) {
          turn_left = turn_left_rgb64_sse2;
          turn_right = turn_right_rgb64_sse2;
        }
        else

#endif
        {
          turn_left = turn_left_rgb64_c;
          turn_right = turn_right_rgb64_c;
        }
      }
      else {
        switch (vi.ComponentSize()) {// AVS16
        case 1: // 8 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_8_sse2;
            turn_right = turn_right_plane_8_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_8_c;
            turn_right = turn_right_plane_8_c;
          }
          break;
        case 2: // 16 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_16_sse2;
            turn_right = turn_right_plane_16_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_16_c;
            turn_right = turn_right_plane_16_c;
          }
          break;
        default: // 32 bit
#ifdef INTEL_INTRINSICS
          if (has_sse2) {
            turn_left = turn_left_plane_32_sse2;
            turn_right = turn_right_plane_32_sse2;
          }
          else
#endif
          {
            turn_left = turn_left_plane_32_c;
            turn_right = turn_right_plane_32_c;
          }
        }
      }
    }
    else {
#ifdef INTEL_INTRINSICS
      // Planar + SSSE3 = use new horizontal resizer routines
      resampler_h_luma = GetResampler(cpu, true, pixelsize, bits_per_pixel, resampling_program_luma, env);

      if (!grey && !isRGBPfamily) {
        resampler_h_chroma = GetResampler(cpu, true, pixelsize, bits_per_pixel, resampling_program_chroma, env);
      }
#else
      assert(0);
#endif
    }
  // Change target video info size
  vi.width = target_width;
}

PVideoFrame __stdcall FilteredResizeH::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (!fast_resize) {
    // e.g. not aligned, not mod4
    // temp_1_pitch and temp_2_pitch is pixelsize-aware
    BYTE* temp_1 = static_cast<BYTE*>(env->Allocate(temp_1_pitch * src_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
    BYTE* temp_2 = static_cast<BYTE*>(env->Allocate(temp_2_pitch * dst_width, FRAME_ALIGN, AVS_POOLED_ALLOC));
    if (!temp_1 || !temp_2) {
      env->Free(temp_1);
      env->Free(temp_2);
      env->ThrowError("Could not reserve memory in a resampler.");
    }

    if (!vi.IsRGB() || isRGBPfamily) {
      // Y/G Plane
      turn_right(src->GetReadPtr(), temp_1, src_width * pixelsize, src_height, src->GetPitch(), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
      resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
      turn_left(temp_2, dst->GetWritePtr(), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch());

      if (isRGBPfamily)
      {
        turn_right(src->GetReadPtr(PLANAR_B), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_B), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_B), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_B));

        turn_right(src->GetReadPtr(PLANAR_R), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_R), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_R), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_R));
      }
      else if (!grey) {
        const int shift = vi.GetPlaneWidthSubsampling(PLANAR_U);
        const int shift_h = vi.GetPlaneHeightSubsampling(PLANAR_U);

        const int src_chroma_width = src_width >> shift;
        const int dst_chroma_width = dst_width >> shift;
        const int src_chroma_height = src_height >> shift_h;
        const int dst_chroma_height = dst_height >> shift_h;

        // turn_xxx: width * pixelsize: needs GetPlaneWidth-like full size
        // U Plane
        turn_right(src->GetReadPtr(PLANAR_U), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_U), temp_1_pitch);
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel, src_pitch_table_luma, filter_storage_chroma);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_U), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_U));

        // V Plane
        turn_right(src->GetReadPtr(PLANAR_V), temp_1, src_chroma_width * pixelsize, src_chroma_height, src->GetPitch(PLANAR_V), temp_1_pitch);
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_chroma, src_chroma_height, dst_chroma_width, bits_per_pixel, src_pitch_table_luma, filter_storage_chroma);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_V), dst_chroma_height * pixelsize, dst_chroma_width, temp_2_pitch, dst->GetPitch(PLANAR_V));
      }
      if (vi.IsYUVA() || vi.IsPlanarRGBA())
      {
        turn_right(src->GetReadPtr(PLANAR_A), temp_1, src_width * pixelsize, src_height, src->GetPitch(PLANAR_A), temp_1_pitch); // * pixelsize: turn_right needs GetPlaneWidth full size
        resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, src_height, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
        turn_left(temp_2, dst->GetWritePtr(PLANAR_A), dst_height * pixelsize, dst_width, temp_2_pitch, dst->GetPitch(PLANAR_A));
      }

    }
    else {
      // packed RGB
      // First left, then right. Reason: packed RGB bottom to top. Right+left shifts RGB24/RGB32 image to the opposite horizontal direction
      turn_left(src->GetReadPtr(), temp_1, vi.BytesFromPixels(src_width), src_height, src->GetPitch(), temp_1_pitch);
      resampler_luma(temp_2, temp_1, temp_2_pitch, temp_1_pitch, resampling_program_luma, vi.BytesFromPixels(src_height) / pixelsize, dst_width, bits_per_pixel, src_pitch_table_luma, filter_storage_luma);
      turn_right(temp_2, dst->GetWritePtr(), vi.BytesFromPixels(dst_height), dst_width, temp_2_pitch, dst->GetPitch());
    }

    env->Free(temp_1);
    env->Free(temp_2);
  }
  else {

    // Y Plane
    resampler_h_luma(dst->GetWritePtr(), src->GetReadPtr(), dst->GetPitch(), src->GetPitch(), resampling_program_luma, dst_width, dst_height, bits_per_pixel);

    if (isRGBPfamily) {
      resampler_h_luma(dst->GetWritePtr(PLANAR_B), src->GetReadPtr(PLANAR_B), dst->GetPitch(PLANAR_B), src->GetPitch(PLANAR_B), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
      resampler_h_luma(dst->GetWritePtr(PLANAR_R), src->GetReadPtr(PLANAR_R), dst->GetPitch(PLANAR_R), src->GetPitch(PLANAR_R), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
    }
    else if (!grey) {
      const int dst_chroma_width = dst_width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
      const int dst_chroma_height = dst_height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

      // U Plane
      resampler_h_chroma(dst->GetWritePtr(PLANAR_U), src->GetReadPtr(PLANAR_U), dst->GetPitch(PLANAR_U), src->GetPitch(PLANAR_U), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);

      // V Plane
      resampler_h_chroma(dst->GetWritePtr(PLANAR_V), src->GetReadPtr(PLANAR_V), dst->GetPitch(PLANAR_V), src->GetPitch(PLANAR_V), resampling_program_chroma, dst_chroma_width, dst_chroma_height, bits_per_pixel);
    }
    if (vi.IsYUVA() || vi.IsPlanarRGBA())
    {
      resampler_h_luma(dst->GetWritePtr(PLANAR_A), src->GetReadPtr(PLANAR_A), dst->GetPitch(PLANAR_A), src->GetPitch(PLANAR_A), resampling_program_luma, dst_width, dst_height, bits_per_pixel);
    }

  }

  return dst;
}

ResamplerH FilteredResizeH::GetResampler(int CPU, bool aligned, int pixelsize, int bits_per_pixel, ResamplingProgram* program, IScriptEnvironment* env)
{
  AVS_UNUSED(aligned);

  if (pixelsize == 1)
  {
#ifdef INTEL_INTRINSICS
    if (CPU & CPUF_SSSE3) {
      if (CPU & CPUF_AVX2) {
        // make the resampling coefficient array mod16 friendly for simd, padding non-used coeffs with zeros
        resize_h_prepare_coeff_8or16(program, env, 16);
        return resizer_h_avx2_generic_uint8_t;
      }
      else {
        // make the resampling coefficient array mod8 friendly for simd, padding non-used coeffs with zeros
        resize_h_prepare_coeff_8or16(program, env, 8);
        if (program->filter_size > 8)
          return resizer_h_ssse3_generic;
        else
          return resizer_h_ssse3_8; // no loop
      }
    }
    else // C version
#endif
    {
      return resize_h_c_planar<uint8_t>;
    }
  }
  else if (pixelsize == 2) {
#ifdef INTEL_INTRINSICS
    if (CPU & CPUF_SSSE3) {
      resize_h_prepare_coeff_8or16(program, env, 8); // alignment of 8 is enough for AVX2 uint16_t as well
      if (CPU & CPUF_AVX2) {
        if (bits_per_pixel < 16)
          return resizer_h_avx2_generic_uint16_t<true>;
        else
          return resizer_h_avx2_generic_uint16_t<false>;
      }
      else if (CPU & CPUF_SSE4_1) {
        if (bits_per_pixel < 16)
          return resizer_h_sse41_generic_uint16_t<true>;
        else
          return resizer_h_sse41_generic_uint16_t<false>;
      }
      else {
        // SSSE3 needed
        if (bits_per_pixel < 16)
          return resizer_h_ssse3_generic_uint16_t<true>;
        else
          return resizer_h_ssse3_generic_uint16_t<false>;
      }
    }
    else
#endif
    {
      return resize_h_c_planar<uint16_t>;
    }
  }
  else { //if (pixelsize == 4)
#ifdef INTEL_INTRINSICS
    if (CPU & CPUF_SSSE3) {
      resize_h_prepare_coeff_8or16(program, env, ALIGN_FLOAT_RESIZER_COEFF_SIZE); // alignment of 8 is enough for AVX2 float as well

      const int filtersizealign8 = AlignNumber(program->filter_size, 8);
      const int filtersizemod8 = program->filter_size & 7;

      if (CPU & CPUF_AVX2) {
        if (filtersizealign8 == 8) {
          switch (filtersizemod8) {
          case 0: return resizer_h_avx2_generic_float<1, 0>;
          case 1: return resizer_h_avx2_generic_float<1, 1>;
          case 2: return resizer_h_avx2_generic_float<1, 2>;
          case 3: return resizer_h_avx2_generic_float<1, 3>;
          case 4: return resizer_h_avx2_generic_float<1, 4>;
          case 5: return resizer_h_avx2_generic_float<1, 5>;
          case 6: return resizer_h_avx2_generic_float<1, 6>;
          case 7: return resizer_h_avx2_generic_float<1, 7>;
          }
        }
        else if (filtersizealign8 == 16) {
          switch (filtersizemod8) {
          case 0: return resizer_h_avx2_generic_float<2, 0>;
          case 1: return resizer_h_avx2_generic_float<2, 1>;
          case 2: return resizer_h_avx2_generic_float<2, 2>;
          case 3: return resizer_h_avx2_generic_float<2, 3>;
          case 4: return resizer_h_avx2_generic_float<2, 4>;
          case 5: return resizer_h_avx2_generic_float<2, 5>;
          case 6: return resizer_h_avx2_generic_float<2, 6>;
          case 7: return resizer_h_avx2_generic_float<2, 7>;
          }
        }
        else {
          switch (filtersizemod8) {
          case 0: return resizer_h_avx2_generic_float<-1, 0>;
          case 1: return resizer_h_avx2_generic_float<-1, 1>;
          case 2: return resizer_h_avx2_generic_float<-1, 2>;
          case 3: return resizer_h_avx2_generic_float<-1, 3>;
          case 4: return resizer_h_avx2_generic_float<-1, 4>;
          case 5: return resizer_h_avx2_generic_float<-1, 5>;
          case 6: return resizer_h_avx2_generic_float<-1, 6>;
          case 7: return resizer_h_avx2_generic_float<-1, 7>;
          }
        }
      }
      // SSSE3
      if (filtersizealign8 == 8) {
        switch (filtersizemod8) {
        case 0: return resizer_h_ssse3_generic_float<1, 0>;
        case 1: return resizer_h_ssse3_generic_float<1, 1>;
        case 2: return resizer_h_ssse3_generic_float<1, 2>;
        case 3: return resizer_h_ssse3_generic_float<1, 3>;
        case 4: return resizer_h_ssse3_generic_float<1, 4>;
        case 5: return resizer_h_ssse3_generic_float<1, 5>;
        case 6: return resizer_h_ssse3_generic_float<1, 6>;
        case 7: return resizer_h_ssse3_generic_float<1, 7>;
        }
      }
      else if (filtersizealign8 == 16) {
        switch (filtersizemod8) {
        case 0: return resizer_h_ssse3_generic_float<2, 0>;
        case 1: return resizer_h_ssse3_generic_float<2, 1>;
        case 2: return resizer_h_ssse3_generic_float<2, 2>;
        case 3: return resizer_h_ssse3_generic_float<2, 3>;
        case 4: return resizer_h_ssse3_generic_float<2, 4>;
        case 5: return resizer_h_ssse3_generic_float<2, 5>;
        case 6: return resizer_h_ssse3_generic_float<2, 6>;
        case 7: return resizer_h_ssse3_generic_float<2, 7>;
        }
      }
      else {
        switch (filtersizemod8) {
        case 0: return resizer_h_ssse3_generic_float<-1, 0>;
        case 1: return resizer_h_ssse3_generic_float<-1, 1>;
        case 2: return resizer_h_ssse3_generic_float<-1, 2>;
        case 3: return resizer_h_ssse3_generic_float<-1, 3>;
        case 4: return resizer_h_ssse3_generic_float<-1, 4>;
        case 5: return resizer_h_ssse3_generic_float<-1, 5>;
        case 6: return resizer_h_ssse3_generic_float<-1, 6>;
        case 7: return resizer_h_ssse3_generic_float<-1, 7>;
        }
      }
    }
#endif
    return resize_h_c_planar<float>;
  }
}

FilteredResizeH::~FilteredResizeH(void)
{
  if (resampling_program_luma) { delete resampling_program_luma; }
  if (resampling_program_chroma) { delete resampling_program_chroma; }
  if (src_pitch_table_luma) { delete[] src_pitch_table_luma; }
}

/***************************************
 ***** Filtered Resize - Vertical ******
 ***************************************/

FilteredResizeV::FilteredResizeV(PClip _child, double subrange_top, double subrange_height,
  int target_height, ResamplingFunction* func, IScriptEnvironment* env)
  : GenericVideoFilter(_child),
  resampling_program_luma(0), resampling_program_chroma(0),
  filter_storage_luma_aligned(0),
  filter_storage_chroma_aligned(0)
{
  if (target_height <= 0)
    env->ThrowError("Resize: Height must be greater than 0.");

  pixelsize = vi.ComponentSize(); // AVS16
  bits_per_pixel = vi.BitsPerComponent();
  grey = vi.IsY();
  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int mask = (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1;

    if (target_height & mask)
      env->ThrowError("Resize: Planar destination height must be a multiple of %d.", mask + 1);
  }

  if (vi.IsRGB() && !isRGBPfamily)
    subrange_top = vi.height - subrange_top - subrange_height; // packed RGB upside down

#ifdef INTEL_INTRINSICS
  int cpu = env->GetCPUFlags();
#else
  int cpu = 0;
#endif

  // Create resampling program and pitch table
  resampling_program_luma = func->GetResamplingProgram(vi.height, subrange_top, subrange_height, target_height, bits_per_pixel, env);
  resampler_luma_aligned = GetResampler(cpu, true, pixelsize, bits_per_pixel, filter_storage_luma_aligned, resampling_program_luma);
  if (vi.height < resampling_program_luma->filter_size) {
    env->ThrowError("Source height (%d) is too small for this resizing method, must be minimum of %d", vi.height, resampling_program_luma->filter_size);
  }

  if (vi.IsPlanar() && !grey && !isRGBPfamily) {
    const int shift = vi.GetPlaneHeightSubsampling(PLANAR_U);
    const int div = 1 << shift;

    resampling_program_chroma = func->GetResamplingProgram(
      vi.height >> shift,
      subrange_top / div,
      subrange_height / div,
      target_height >> shift,
      bits_per_pixel,
      env);

    resampler_chroma_aligned = GetResampler(cpu, true, pixelsize, bits_per_pixel, filter_storage_chroma_aligned, resampling_program_chroma);
    const int height_UV = vi.height >> shift;
    if (height_UV < resampling_program_chroma->filter_size) {
      env->ThrowError("Source chroma height (%d) is too small for this resizing method, must be minimum of %d", height_UV, resampling_program_chroma->filter_size);
    }
  }

  // Change target video info size
  vi.height = target_height;
}

PVideoFrame __stdcall FilteredResizeV::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  int src_pitch = src->GetPitch();
  int dst_pitch = dst->GetPitch();
  const BYTE* srcp = src->GetReadPtr();
  BYTE* dstp = dst->GetWritePtr();

  bool isRGBPfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();

  // Create pitch table
  int* src_pitch_table_luma = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(), 32, AVS_POOLED_ALLOC));
  if (!src_pitch_table_luma) {
    env->ThrowError("Could not reserve memory in a resampler.");
  }

  resize_v_create_pitch_table(src_pitch_table_luma, src->GetPitch(), src->GetHeight());

  int* src_pitch_table_chromaU = NULL;
  int* src_pitch_table_chromaV = NULL;
  if ((!grey && vi.IsPlanar() && !isRGBPfamily)) {
    src_pitch_table_chromaU = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(PLANAR_U), 32, AVS_POOLED_ALLOC));
    src_pitch_table_chromaV = static_cast<int*>(env->Allocate(sizeof(int) * src->GetHeight(PLANAR_V), 32, AVS_POOLED_ALLOC));
    if (!src_pitch_table_chromaU || !src_pitch_table_chromaV) {
      env->Free(src_pitch_table_luma);
      env->Free(src_pitch_table_chromaU);
      env->Free(src_pitch_table_chromaV);
      env->ThrowError("Could not reserve memory in a resampler.");
    }

    resize_v_create_pitch_table(src_pitch_table_chromaU, src->GetPitch(PLANAR_U), src->GetHeight(PLANAR_U));
    resize_v_create_pitch_table(src_pitch_table_chromaV, src->GetPitch(PLANAR_V), src->GetHeight(PLANAR_V));
  }

  // Do resizing
  int work_width = vi.IsPlanar() ? vi.width : vi.BytesFromPixels(vi.width) / pixelsize; // packed RGB: or vi.width * vi.NumComponent()
  // alignment to FRAME_ALIGN is guaranteed
  resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
  if (isRGBPfamily)
  {
    src_pitch = src->GetPitch(PLANAR_B);
    dst_pitch = dst->GetPitch(PLANAR_B);
    srcp = src->GetReadPtr(PLANAR_B);
    dstp = dst->GetWritePtr(PLANAR_B);
    // alignment to FRAME_ALIGN is guaranteed
    resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
    src_pitch = src->GetPitch(PLANAR_R);
    dst_pitch = dst->GetPitch(PLANAR_R);
    srcp = src->GetReadPtr(PLANAR_R);
    dstp = dst->GetWritePtr(PLANAR_R);
    // alignment to FRAME_ALIGN is guaranteed
    resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
  }
  else if (!grey && vi.IsPlanar()) {
    int width = vi.width >> vi.GetPlaneWidthSubsampling(PLANAR_U);
    int height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

    // Plane U resizing
    src_pitch = src->GetPitch(PLANAR_U);
    dst_pitch = dst->GetPitch(PLANAR_U);
    srcp = src->GetReadPtr(PLANAR_U);
    dstp = dst->GetWritePtr(PLANAR_U);

    // alignment to FRAME_ALIGN is guaranteed
    resampler_chroma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel, src_pitch_table_chromaU, filter_storage_chroma_aligned);

    // Plane V resizing
    src_pitch = src->GetPitch(PLANAR_V);
    dst_pitch = dst->GetPitch(PLANAR_V);
    srcp = src->GetReadPtr(PLANAR_V);
    dstp = dst->GetWritePtr(PLANAR_V);

    // alignment to FRAME_ALIGN is guaranteed
    resampler_chroma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_chroma, width, height, bits_per_pixel, src_pitch_table_chromaV, filter_storage_chroma_aligned);
  }

  if (vi.IsYUVA() || vi.IsPlanarRGBA()) {
    src_pitch = src->GetPitch(PLANAR_A);
    dst_pitch = dst->GetPitch(PLANAR_A);
    srcp = src->GetReadPtr(PLANAR_A);
    dstp = dst->GetWritePtr(PLANAR_A);
    // alignment to FRAME_ALIGN is guaranteed
    resampler_luma_aligned(dstp, srcp, dst_pitch, src_pitch, resampling_program_luma, work_width, vi.height, bits_per_pixel, src_pitch_table_luma, filter_storage_luma_aligned);
  }

  // Free pitch table
  env->Free(src_pitch_table_luma);
  env->Free(src_pitch_table_chromaU);
  env->Free(src_pitch_table_chromaV);

  return dst;
}

ResamplerV FilteredResizeV::GetResampler(int CPU, bool _aligned_not_used, int pixelsize, int bits_per_pixel, void*& storage, ResamplingProgram* program)
{
  AVS_UNUSED(storage);

  // no unaligned source possible

  if (program->filter_size == 1) {
    // Fast pointresize
    switch (pixelsize) // AVS16
    {
    case 1: return resize_v_planar_pointresize<uint8_t>;
    case 2: return resize_v_planar_pointresize<uint16_t>;
    default: // case 4:
      return resize_v_planar_pointresize<float>;
    }
  }
  else {
    // Other resizers
    if (pixelsize == 1)
    {
#ifdef INTEL_INTRINSICS
      // always aligned
      if (CPU & CPUF_AVX2)
        return resize_v_avx2_planar_uint8_t;
      else if (CPU & CPUF_SSE4_1)
        return resize_v_sse41_planar;
      else if (CPU & CPUF_SSSE3)
        return resize_v_ssse3_planar;
      else if (CPU & CPUF_SSE2)
        return resize_v_sse2_planar;
#ifdef X86_32
      else if (CPU & CPUF_MMX)
        return resize_v_mmx_planar;
#endif
      else // C version
#endif
      {
        return resize_v_c_planar<uint8_t>;
      }
    }
    else if (pixelsize == 2)
    {
#ifdef INTEL_INTRINSICS
      // always aligned
      if (CPU & CPUF_AVX2) {
        if (bits_per_pixel < 16)
          return resize_v_avx2_planar_uint16_t<true>;
        else
          return resize_v_avx2_planar_uint16_t<false>;
      }
      else if (CPU & CPUF_SSE4_1) {
        if (bits_per_pixel < 16)
          return resize_v_sse41_planar_uint16_t<true>;
        else
          return resize_v_sse41_planar_uint16_t<false>;
      }
      else if (CPU & CPUF_SSE2) {
        if (bits_per_pixel < 16)
          return resize_v_sse2_planar_uint16_t<true>;
        else
          return resize_v_sse2_planar_uint16_t<false>;
      }
      else // C version
#endif
      {
        return resize_v_c_planar<uint16_t>;
      }
    }
    else // pixelsize== 4
    {
#ifdef INTEL_INTRINSICS
      if (CPU & CPUF_AVX2) {
        return resize_v_avx2_planar_float;
      }
      else if (CPU & CPUF_SSE2) {
        return resize_v_sse2_planar_float;
      }
      else
#endif
      {
        return resize_v_c_planar<float>;
      }
    }
  }
}

FilteredResizeV::~FilteredResizeV(void)
{
  if (resampling_program_luma) { delete resampling_program_luma; }
  if (resampling_program_chroma) { delete resampling_program_chroma; }
}


/**********************************************
 *******   Resampling Factory Methods   *******
 **********************************************/

PClip FilteredResize::CreateResizeH(PClip clip, double subrange_left, double subrange_width, int target_width,
  ResamplingFunction* func, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  if (subrange_left == 0 && subrange_width == target_width && subrange_width == vi.width) {
    return clip;
  }
  /*
  // intentionally left here: don't use crop at special edge cases to avoid inconsistent results across params/color spaces
  if (subrange_left == int(subrange_left) && subrange_width == target_width
   && subrange_left >= 0 && subrange_left + subrange_width <= vi.width) {
    const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneWidthSubsampling(PLANAR_U)) - 1 : 0;

    if (((int(subrange_left) | int(subrange_width)) & mask) == 0)
      return new Crop(int(subrange_left), 0, int(subrange_width), vi.height, 0, clip, env);
  }
  */
  // Convert interleaved yuv to planar yuv
  PClip result = clip;
  if (vi.IsYUY2()) {
    result = new ConvertYUY2ToYV16(result, env);
  }
  result = new FilteredResizeH(result, subrange_left, subrange_width, target_width, func, env);
  if (vi.IsYUY2()) {
    result = new ConvertYV16ToYUY2(result, env);
  }

  return result;
}


PClip FilteredResize::CreateResizeV(PClip clip, double subrange_top, double subrange_height, int target_height,
  ResamplingFunction* func, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  if (subrange_top == 0 && subrange_height == target_height && subrange_height == vi.height) {
    return clip;
  }
  /*
  // intentionally left here: don't use crop at special edge cases to avoid inconsistent results across params/color spaces
  if (subrange_top == int(subrange_top) && subrange_height == target_height
   && subrange_top >= 0 && subrange_top + subrange_height <= vi.height) {
    const int mask = ((vi.IsYUV() || vi.IsYUVA()) && !vi.IsY()) ? (1 << vi.GetPlaneHeightSubsampling(PLANAR_U)) - 1 : 0;

    if (((int(subrange_top) | int(subrange_height)) & mask) == 0)
      return new Crop(0, int(subrange_top), vi.width, int(subrange_height), 0, clip, env);
  }
  */
  return new FilteredResizeV(clip, subrange_top, subrange_height, target_height, func, env);
}


PClip FilteredResize::CreateResize(PClip clip, int target_width, int target_height, const AVSValue* args,
  ResamplingFunction* f, IScriptEnvironment* env)
{
  const VideoInfo& vi = clip->GetVideoInfo();
  const double subrange_left = args[0].AsFloat(0), subrange_top = args[1].AsFloat(0);

  double subrange_width = args[2].AsDblDef(vi.width), subrange_height = args[3].AsDblDef(vi.height);
  // Crop style syntax
  if (subrange_width <= 0.0) subrange_width = vi.width - subrange_left + subrange_width;
  if (subrange_height <= 0.0) subrange_height = vi.height - subrange_top + subrange_height;

  PClip result;
  // ensure that the intermediate area is maximal

  const double area_FirstH = subrange_height * target_width;
  const double area_FirstV = subrange_width * target_height;

  // "minimal area" logic is not necessarily faster because H and V resizers are not the same speed.
  // so we keep the traditional max area logic.
  if (area_FirstH < area_FirstV)
  {
    result = CreateResizeV(clip, subrange_top, subrange_height, target_height, f, env);
    result = CreateResizeH(result, subrange_left, subrange_width, target_width, f, env);
  }
  else
  {
    result = CreateResizeH(clip, subrange_left, subrange_width, target_width, f, env);
    result = CreateResizeV(result, subrange_top, subrange_height, target_height, f, env);
  }
  return result;
}

AVSValue __cdecl FilteredResize::Create_PointResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = PointFilter();
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}


AVSValue __cdecl FilteredResize::Create_BilinearResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = TriangleFilter();
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}


AVSValue __cdecl FilteredResize::Create_BicubicResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = MitchellNetravaliFilter(args[3].AsDblDef(1. / 3.), args[4].AsDblDef(1. / 3.));
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[5], &f, env);
}

AVSValue __cdecl FilteredResize::Create_LanczosResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = LanczosFilter(args[7].AsInt(3));
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Lanczos4Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = LanczosFilter(4);
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_BlackmanResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = BlackmanFilter(args[7].AsInt(4));
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline16Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline16Filter();
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline36Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline36Filter();
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_Spline64Resize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = Spline64Filter();
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_GaussianResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = GaussianFilter(args[7].AsFloat(30.0f));
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

AVSValue __cdecl FilteredResize::Create_SincResize(AVSValue args, void*, IScriptEnvironment* env)
{
  auto f = SincFilter(args[7].AsInt(4));
  return CreateResize(args[0].AsClip(), args[1].AsInt(), args[2].AsInt(), &args[3], &f, env);
}

