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

// ConvertPlanar (c) 2005 by Klaus Post


#include "convert.h"
#include "convert_matrix.h"
#include "convert_helper.h"
#include "convert_planar.h"
#ifdef INTEL_INTRINSICS
#include "intel/convert_planar_sse.h"
#include "intel/convert_planar_avx2.h"
#endif
#include "convert_bits.h"
#include "../filters/resample.h"
#include "../filters/planeswap.h"
#include "../filters/field.h"

#ifdef AVS_WINDOWS
    #include <avs/win.h>
#else
    #include <avs/posix.h>
#endif

#include <avs/alignment.h>
#include <algorithm>
#include <string>
#include <vector>

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNREFERENCED_LOCAL_VARIABLE

template <typename pixel_t>
void fill_chroma(uint8_t * dstp_u, uint8_t * dstp_v, int height, int row_size, int pitch, pixel_t val)
{
  if (pitch == row_size) {
    size_t size = height * pitch / sizeof(pixel_t);
    std::fill_n(reinterpret_cast<pixel_t*>(dstp_u), size, val);
    std::fill_n(reinterpret_cast<pixel_t*>(dstp_v), size, val);
  }
  else {
    size_t size = row_size / sizeof(pixel_t);
    for (int i = 0; i < height; i++) {
      std::fill_n(reinterpret_cast<pixel_t*>(dstp_u), size, val);
      std::fill_n(reinterpret_cast<pixel_t*>(dstp_v), size, val);
      dstp_u += pitch;
      dstp_v += pitch;
    }
  }
}

template <typename pixel_t>
void fill_plane(uint8_t * dstp, int height, int row_size, int pitch, pixel_t val)
{
  if (pitch == row_size) {
    size_t size = height * pitch / sizeof(pixel_t);
    std::fill_n(reinterpret_cast<pixel_t*>(dstp), size, val);
  }
  else {
    size_t size = row_size / sizeof(pixel_t);
    for (int i = 0; i < height; i++) {
      std::fill_n(reinterpret_cast<pixel_t*>(dstp), size, val);
      dstp += pitch;
    }
  }
}

// specialize it
template void fill_plane<uint8_t>(uint8_t * dstp, int height, int row_size, int pitch, uint8_t val);
template void fill_plane<uint16_t>(uint8_t * dstp, int height, int row_size, int pitch, uint16_t val);
template void fill_plane<float>(uint8_t * dstp, int height, int row_size, int pitch, float val);
template void fill_chroma<uint8_t>(uint8_t * dstp_u, uint8_t * dstp_v, int height, int row_size, int pitch, uint8_t val);
template void fill_chroma<uint16_t>(uint8_t * dstp_u, uint8_t * dstp_v, int height, int row_size, int pitch, uint16_t val);
template void fill_chroma<float>(uint8_t * dstp_u, uint8_t * dstp_v, int height, int row_size, int pitch, float val);


// ConvertToY::ConvertToY moved into ConvertPlanarToGeneric, since the conversion logic and parameters overlap with -->444 Conversion, we just need only Y.
// Legacy formats (packed RGB/YUY2) are handled uniformly inside.
AVSValue __cdecl ConvertToPlanarGeneric::CreateY(AVSValue args, void* user_data, IScriptEnvironment* env) {
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  PClip clip = args[0].AsClip();

  // 0    1       2      3
  // c[matrix]s[bits]i[quality]b
  if (only_8bit) {
    // ConvertToY8() syntax
    if (args[2].Defined() && args[2].AsInt() != 8)
      env->ThrowError("ConvertToY8: only 8 bit output supported. "
        "Use ConvertToY for higher bit depth.");
  }

  // 1.) One single route: planar constructor, where YUY2 is converted to Y8 if needed
  // 2.) If Y, reason to not return it right now:
  // - possible bit-depth conversion,
  // - optional Y-Y future matrix (auto:full->auto:limited) syntax conversion)

  // We use the very same logic as for ConvertToYV24/YUV444, ConvertToYV16/YUV422, ConvertToYV12/YUV420
  if (only_8bit) {
    // ConvertToY8() syntax, force 8 bit output}
    AVSValue new_args[4] = {
      args[0], args[1], AVSValue(8), args[3]
    };
    AVSValue new_args_val(new_args, 4);
    return Create(new_args_val, "ConvertToY", only_8bit, false /*to_yuva*/, env);
  }
  else {
    return Create(args, "ConvertToY", only_8bit, false /*to_yuva*/, env);
  }
}

/*****************************************************
 * ConvertRGBToYUV444 or simply Y only (when to_y=true)
 ******************************************************/

ConvertRGBToYUV444::ConvertRGBToYUV444(PClip src, const char *matrix_name, bool keep_packedrgb_alpha, int _target_bit_depth, bool _quality, bool& bitdepthConverted, bool _to_y, IScriptEnvironment* env)
  : GenericVideoFilter(src), target_bit_depth(_target_bit_depth), quality(_quality), conv_function(nullptr), conv_function_chroma(nullptr), conv_function_a(nullptr), to_y(_to_y)
{
  if (!vi.IsRGB())
    env->ThrowError("ConvertRGBToYV24/YUV444: Only RGB data input accepted");

  if (vi.IsRGB48() || vi.IsRGB64())
    env->ThrowError("ConvertRGBToYV24/YUV444: internal error: packed 16 bit not supported, must be converted to RGBP(A)16");

  // ChromaInPlacement parameter exists, (default none/-1) + input frame properties; 'left'-ish _ChromaLocation is allowed, checked later
  auto frame0 = src->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);

  // input _ColorRange frame property can appear for RGB source (studio range limited rgb)
  matrix_parse_merge_with_props(true /*in rgb*/, false /*out yuv*/, matrix_name, props, theMatrix, theColorRange, theOutColorRange, env);

  const int shift = 15; // internally 15 bits precision, still no overflow in calculations
  int bits_per_pixel = vi.BitsPerComponent();
  source_bit_depth = bits_per_pixel;

  if (!do_BuildMatrix_Rgb2Yuv(theMatrix, theColorRange, theOutColorRange, shift, source_bit_depth, /*ref*/matrix))
    env->ThrowError("ConvertRGBToYV24/YUV444: Unknown matrix.");

  theColorRange = theOutColorRange; // final frame property

  // target_bit_depth == -1: no conversion required
  const bool need_bitdepth_conversion = (target_bit_depth != -1);

  // get alpha converter, GetFrame is using that if alpha must be depth-converterd and not only copied
  if (need_bitdepth_conversion)
  {
    // supported conversions:
    // 8-16-bits any range to 8-16, 32 bits any range
    bitdepthConverted = true;

    if (!to_y) { // single plane Y only has no alpha
#ifdef INTEL_INTRINSICS
      const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
      const bool sse4 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
      const bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
#endif
      // we really need only alpha conversion function
      const bool fulls = true; // alpha is always full range
      const bool fulld = true; // alpha is always full range
      const int dither_mode = -1;
#ifdef INTEL_INTRINSICS
      get_convert_any_bits_functions(dither_mode, source_bit_depth, target_bit_depth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
      get_convert_any_bits_functions(dither_mode, source_bit_depth, target_bit_depth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif
    }
  }
  else {
    bitdepthConverted = false;
    target_bit_depth = source_bit_depth;
  }

  isPlanarRGBfamily = vi.IsPlanarRGB() || vi.IsPlanarRGBA();
  targetHasAlpha = vi.IsPlanarRGBA() || (keep_packedrgb_alpha && (vi.IsRGB32() || vi.IsRGB64()));
  // for packed RGB it depends: ConvertToYUVAxxx() can force YUVA target option
  if (isPlanarRGBfamily)
  {
    if (to_y) {
      // --> Y
      pixel_step =  -1;
      switch (target_bit_depth)
      {
      case 8: vi.pixel_type = VideoInfo::CS_Y8; break;
      case 10: vi.pixel_type = VideoInfo::CS_Y10; break;
      case 12: vi.pixel_type = VideoInfo::CS_Y12; break;
      case 14: vi.pixel_type = VideoInfo::CS_Y14; break;
      case 16: vi.pixel_type = VideoInfo::CS_Y16; break;
      case 32: vi.pixel_type = VideoInfo::CS_Y32; break;
      }
    }
    else {
      // --> 444(4)
      pixel_step = targetHasAlpha ? -2 : -1;
      switch (target_bit_depth)
      {
      case 8: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444 : VideoInfo::CS_YV24; break;
      case 10: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444P10 : VideoInfo::CS_YUV444P10; break;
      case 12: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444P12 : VideoInfo::CS_YUV444P12; break;
      case 14: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444P14 : VideoInfo::CS_YUV444P14; break;
      case 16: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444P16 : VideoInfo::CS_YUV444P16; break;
      case 32: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444PS : VideoInfo::CS_YUV444PS; break;
      }
    }
  } else { // packed RGB24/32/48/64
    // for compatibility reasons ConvertToYUVxxx target is not YUVA even 
    // if original has alpha, such as RGB32.
    // Unlike ConvertToYUVA which can force keeping RGB32's or RGB64's alpha

    // must be the same, bit depth change is supported only from planar rgb
    if (bits_per_pixel != target_bit_depth)
      env->ThrowError("ConvertRGBToYV24/YUV444: bit depth change is only supported from planar RGB input");

    pixel_step = vi.BytesFromPixels(1); // 3,4 for packed 8 bit, 6,8 for
    switch(vi.ComponentSize())
    {
    case 1: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444 : VideoInfo::CS_YV24; break;
    case 2: vi.pixel_type = targetHasAlpha ? VideoInfo::CS_YUVA444P16 : VideoInfo::CS_YUV444P16; break;
    }
  }

}

#if 0
// Kept for reference
template<int pixel_step, bool copyalpha>
static void convert_rgb24or32_to_yuv444_c(BYTE* dstY, BYTE* dstU, BYTE* dstV, BYTE* dstA, const BYTE* srcp, size_t Ypitch, size_t UVpitch, size_t Apitch, size_t Spitch, size_t width, size_t height, const ConversionMatrix& m)
{
  const bool has_offset_rgb = 0 != m.offset_rgb;

  // RGB24 (pixel_step=3)
  // RGB32 (pixel_step=4) optional copyalpha
  srcp += Spitch * (height - 1);  // We start at last line
  const size_t Sstep = Spitch + width * pixel_step;
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      int b = srcp[0];
      int g = srcp[1];
      int r = srcp[2];
      if (has_offset_rgb) {
        b = b + m.offset_rgb;
        g = g + m.offset_rgb;
        r = r + m.offset_rgb;
      }
      int Y = m.offset_y + (((int)m.y_b * b + (int)m.y_g * g + (int)m.y_r * r + 16384) >> 15);
      int U = 128 + (((int)m.u_b * b + (int)m.u_g * g + (int)m.u_r * r + 16384) >> 15);
      int V = 128 + (((int)m.v_b * b + (int)m.v_g * g + (int)m.v_r * r + 16384) >> 15);
      dstY[x] = PixelClip(Y);  // All the safety we can wish for.
      dstU[x] = PixelClip(U);
      dstV[x] = PixelClip(V);
      if constexpr (copyalpha) {
        dstA[x] = srcp[3];
      }
      srcp += pixel_step;
    }
    srcp -= Sstep; // packed RGBs are upside down
    dstY += Ypitch;
    dstU += UVpitch;
    dstV += UVpitch;
    if constexpr (copyalpha) {
      dstA += Apitch;
    }
  }
}
#endif


PVideoFrame __stdcall ConvertRGBToYUV444::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  auto props = env->getFramePropsRW(dst);
  update_Matrix_and_ColorRange(props, theMatrix, theColorRange, env);

  BYTE* dstY = dst->GetWritePtr(PLANAR_Y);
  BYTE* dstU = to_y ? nullptr : dst->GetWritePtr(PLANAR_U);
  BYTE* dstV = to_y ? nullptr : dst->GetWritePtr(PLANAR_V);

  const int Ypitch = dst->GetPitch(PLANAR_Y);
  const int UVpitch = to_y ? 0 : dst->GetPitch(PLANAR_U);

  const bool packedRGBsource = pixel_step > 0;

  if (packedRGBsource) {
    env->ThrowError("Packed RGB is no more allowed here, already preconverted to planar RGB. This is a bug.");
  }

  // here we do planar RGB only
  const int bits_per_pixel = source_bit_depth;

  // copy or fill alpha
  // Two places, at YUV444toPlanarRGB as well
  BYTE* dstpA;
  if (targetHasAlpha) {
    dstpA = dst->GetWritePtr(PLANAR_A);
    int heightA = dst->GetHeight(PLANAR_A);
    int rowsizeA = dst->GetRowSize(PLANAR_A);
    int dst_pitchA = dst->GetPitch(PLANAR_A);
    // simple copy
    if (src->GetRowSize(PLANAR_A)) {

      if (source_bit_depth == target_bit_depth) {
        // vi.IsYUVA() no-no! vi is already the target video type
        env->BitBlt(dstpA, dst_pitchA, src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A_ALIGNED), src->GetHeight(PLANAR_A));
      }
      else {
        // RGBPA target + different bit-depth
        // Alpha bit depth conversion needed, normal planes are done in-process.
        // we need only conv_function_a, proper dispatching from ConvertBits
        conv_function_a(src->GetReadPtr(PLANAR_A), dstpA, src->GetRowSize(PLANAR_A), heightA, src->GetPitch(PLANAR_A), dst_pitchA, source_bit_depth, target_bit_depth, -1);
      }
    }
    else {
      // fill default transparency
      switch (target_bit_depth)
      {
      case 1:
        fill_plane<BYTE>(dstpA, heightA, rowsizeA, dst_pitchA, 255);
        break;
      case 2:
        fill_plane<uint16_t>(dstpA, heightA, rowsizeA, dst_pitchA, (1 << target_bit_depth) - 1);
        break;
      case 4:
        fill_plane<float>(dstpA, heightA, rowsizeA, dst_pitchA, 1.0f);
        break;
      }
    }
  }

  const BYTE* srcp[3] = { src->GetReadPtr(PLANAR_G), src->GetReadPtr(PLANAR_B), src->GetReadPtr(PLANAR_R) };
  const int srcPitch[3] = { src->GetPitch(PLANAR_G), src->GetPitch(PLANAR_B), src->GetPitch(PLANAR_R) };

  BYTE* dstp[3] = { dstY, dstU, dstV };
  int dstPitch[3] = { Ypitch, UVpitch, UVpitch };

  const bool force_float = quality;

  // to Y

  if (to_y) {
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_AVX2) {
      if (bits_per_pixel == 8)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel < 16)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 16)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 32)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_Y, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      return dst;
    }

    if (env->GetCPUFlags() & CPUF_SSE2) {
      if (bits_per_pixel == 8)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_Y, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel < 16)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_Y, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 16)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_Y, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 32)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_Y, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      return dst;
    }

#endif

    if (bits_per_pixel == 8)
      convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel < 16)
      convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 16)
      convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 32)
      convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    return dst;
  }

  // to YUV

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_AVX2) {
    if (bits_per_pixel == 8)
      convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel < 16)
      convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 16)
      convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 32)
      convert_yuv_to_planarrgb_avx2<ConversionDirection::RGB_TO_YUV, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    return dst;
  }

  if (env->GetCPUFlags() & CPUF_SSE2) {
    if (bits_per_pixel == 8)
      convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_YUV, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel < 16)
      convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_YUV, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 16)
      convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_YUV, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 32)
      convert_yuv_to_planarrgb_sse2<ConversionDirection::RGB_TO_YUV, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    return dst;
  }

#endif

  if (bits_per_pixel == 8)
    convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
  else if (bits_per_pixel < 16)
    convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
  else if (bits_per_pixel == 16)
    convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
  else if (bits_per_pixel == 32)
    convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
  return dst;

}



/*****************************************************
 * ConvertYV24ToRGB
 *
 * (c) Klaus Post, 2005
 * Generic 4:4:4(:4), 16 bit and Planar RGB(A) support 2016 by PF
 ******************************************************/


ConvertYUV444ToRGB::ConvertYUV444ToRGB(PClip src, const char *matrix_name, int _pixel_step, int _target_bit_depth, bool _quality, bool& bitdepthConverted, IScriptEnvironment* env)
  : GenericVideoFilter(src), pixel_step(_pixel_step), target_bit_depth(_target_bit_depth), quality(_quality), conv_function(nullptr), conv_function_chroma(nullptr), conv_function_a(nullptr)
{

  if (!vi.Is444())
    env->ThrowError("ConvertYUV444ToRGB: Only 4:4:4 data input accepted");

  auto frame0 = child->GetFrame(0, env);
  const AVSMap* props = env->getFramePropsRO(frame0);
  matrix_parse_merge_with_props(false /*in yuv*/, true /*out rgb*/, matrix_name, props, theMatrix, theColorRange, theOutColorRange, env);
  // Good to know:
  // - When frame property "_ColorRange" exists, it overrides the default output RGB color range.
  // - PC.xxx old-style matrix names leave the color range unchanged, whether they exist or not.
  // - When using PC.xxx and YUV is limited => output RGB is narrow range (limited)
  // - Other non PC.xxx old matrix names:
  //   - input range is defined by the matrix name, even if _ColorRange frame property exists
  //   - output range is default Full for RGB, unless _ColorRange exists, when its value defines the output RGB range (full or limited/narrow)
  // - new style matrix definitions: see docs. Preserve what is not set, distinct :f or :l suffixes may tell the color range, not part of matrix name.
  // - Matrix coefficients still need to know input and output color range (full or limited/narrow) so by this time we have these two.

  const int shift = 13; // for integer arithmetic, over 13 bits would overflow the matrix elements/internal calculation
  const int bits_per_pixel = vi.BitsPerComponent();
  source_bit_depth = bits_per_pixel;

  if (!do_BuildMatrix_Yuv2Rgb(theMatrix, theColorRange, theOutColorRange, shift, source_bit_depth, /*ref*/matrix))
    env->ThrowError("ConvertYUV444ToRGB: Unknown matrix.");

  // Old comment: what is not implemented in-process, we do the conversion with invoked ConvertBits()
  // Now all source-target combinations are supported.

  // target_bit_depth == -1: no conversion required
  const bool need_bitdepth_conversion = (target_bit_depth != -1);

  // get alpha converter, GetFrame is using that if alpha must be depth-converterd and not only copied
  if (need_bitdepth_conversion)
  {
    // supported conversions:
    // 8-16-bits any range to 8-16, 32 bits any range
    bitdepthConverted = true;
#ifdef INTEL_INTRINSICS
    const bool sse2 = !!(env->GetCPUFlags() & CPUF_SSE2);
    const bool sse4 = !!(env->GetCPUFlags() & CPUF_SSE4_1);
    const bool avx2 = !!(env->GetCPUFlags() & CPUF_AVX2);
    // we really need only alpha conversion function
    const bool fulls = true; // alpha is always full range
    const bool fulld = true; // alpha is always full range
    const int dither_mode = -1;
#ifdef INTEL_INTRINSICS
    get_convert_any_bits_functions(dither_mode, source_bit_depth, target_bit_depth, fulls, fulld, sse2, sse4, avx2, conv_function, conv_function_chroma, conv_function_a);
#else
    get_convert_any_bits_functions(dither_mode, source_bit_depth, target_bit_depth, fulls, fulld, conv_function, conv_function_chroma, conv_function_a);
#endif

#endif

  }
  else {
    bitdepthConverted = false;
    target_bit_depth = source_bit_depth;
  }

  theOutMatrix = Matrix_e::AVS_MATRIX_RGB;

  switch (pixel_step)
  {
  case -1: case -2: // target is planar RGB(A): all bit depths supported
    {
      const bool hasAlpha = (pixel_step == -2);
      switch (target_bit_depth)
      {
      case 8:  vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAP : VideoInfo::CS_RGBP; break;
      case 10: vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAP10 : VideoInfo::CS_RGBP10; break;
      case 12: vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAP12 : VideoInfo::CS_RGBP12; break;
      case 14: vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAP14 : VideoInfo::CS_RGBP14; break;
      case 16: vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAP16 : VideoInfo::CS_RGBP16; break;
      case 32: vi.pixel_type = hasAlpha ? VideoInfo::CS_RGBAPS : VideoInfo::CS_RGBPS; break;
      default:
        env->ThrowError("ConvertYUV444ToRGB: invalid vi.BitsPerComponent(): %d", vi.BitsPerComponent());
      }
    }
    break;
  case 3: vi.pixel_type = VideoInfo::CS_BGR24; break;
  case 4: vi.pixel_type = VideoInfo::CS_BGR32; break;
  case 6: vi.pixel_type = VideoInfo::CS_BGR48; break;
  case 8: vi.pixel_type = VideoInfo::CS_BGR64; break;
  default:
    env->ThrowError("ConvertYUV444ToRGB: invalid pixel step: %d", pixel_step);
  }

}

template<int pixel_step, bool source_has_alpha>
static void convert_yv24_to_rgb_c(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, int dst_pitch, int src_pitch_y, int src_pitch_uv, int src_pitch_a, int width, int height, const ConversionMatrix& matrix)
{
  dstp += dst_pitch * (height - 1);  // We start at last line. Packed RGB is bottom-up.
  auto round_mask_plus_rgb_offset_i = 4096 + (matrix.offset_rgb << 13);
  if constexpr (pixel_step == 4) { // RGB32
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        uint8_t a = source_has_alpha ? srcA[x] : 255; // YUVA aware
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + round_mask_plus_rgb_offset_i) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + round_mask_plus_rgb_offset_i) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + round_mask_plus_rgb_offset_i) >> 13);
        dstp[x * 4 + 0] = PixelClip(b);  // All the safety we can wish for.
        dstp[x * 4 + 1] = PixelClip(g);  // Probably needed here.
        dstp[x * 4 + 2] = PixelClip(r);
        dstp[x * 4 + 3] = a; // alpha
      }
      dstp -= dst_pitch;
      srcY += src_pitch_y;
      srcU += src_pitch_uv;
      srcV += src_pitch_uv;
      srcA += src_pitch_a;
    }
  }
  else if constexpr (pixel_step == 3) { // RGB24
    const int Dstep = dst_pitch + (width * pixel_step);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + round_mask_plus_rgb_offset_i) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + round_mask_plus_rgb_offset_i) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + round_mask_plus_rgb_offset_i) >> 13);
        dstp[0] = PixelClip(b);  // All the safety we can wish for.
        dstp[1] = PixelClip(g);  // Probably needed here.
        dstp[2] = PixelClip(r);
        dstp += pixel_step;
      }
      dstp -= Dstep;
      srcY += src_pitch_y;
      srcU += src_pitch_uv;
      srcV += src_pitch_uv;
    }
  }
}


// C reference implementation matching AVX2/SSE2 logic with all conversion directions
template<ConversionDirection direction, typename pixel_t, bool lessthan16bit, bool lessthan16bit_target, typename pixel_t_dst, YuvRgbConversionType conv_type>
static void convert_yuv_to_planarrgb_c_internal(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m,
  int bits_per_pixel, int bits_per_pixel_target)
{
  // int64_t needed for exact 16 bit pixels in integer workflow
  // float input always uses float workflow
  typedef typename std::conditional<sizeof(pixel_t) == 1 || lessthan16bit, int, int64_t>::type safe_int_t;

  // Remark: for RGB->YUV and RGB->Y 16 bit original, we later redirect the constexpr values to a full-float-workflow.
  // Though int64_t would work here in C just fine, but has serious performance issues at Intel SIMD,
  // so we make it identical to the SIMD path and use float path which is simple and much quicker in this case.)

  constexpr int INT_ARITH_SHIFT =
    (direction == ConversionDirection::YUV_TO_RGB) ? 13 :
    (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_Y) ? 15 :
    (direction == ConversionDirection::YUV_TO_YUV) ? 14 : 13;

  // Validate: float input requires FORCE_FLOAT conversion type
  static_assert(!(std::is_floating_point<pixel_t>::value && conv_type != YuvRgbConversionType::FORCE_FLOAT),
    "FORCE_FLOAT conversion type is required for float input pixel type");

  constexpr bool final_is_float = std::is_floating_point<pixel_t_dst>::value;
  constexpr bool would_need_64bit_v_patch = !lessthan16bit && !final_is_float && (INT_ARITH_SHIFT == 15);
  // RGB_TO_YUV, RGB_TO_Y 16 bit origin case needs 64 bit extension at one point to avoid overflow
  // but it makes it much slower than the full-float path, so we force float path for that case as well

  // effectively full-float-inside for int full range conversion, but with the same output rules as other float conversions
  constexpr bool force_float = (conv_type == YuvRgbConversionType::FORCE_FLOAT) || would_need_64bit_v_patch;

  constexpr bool need_int_conversion_narrow_range = conv_type == YuvRgbConversionType::BITCONV_INT_LIMITED;
  constexpr bool need_int_conversion_full_range = conv_type == YuvRgbConversionType::BITCONV_INT_FULL;
  constexpr bool need_int_conversion = conv_type == YuvRgbConversionType::BITCONV_INT_FULL ||
    conv_type == YuvRgbConversionType::BITCONV_INT_LIMITED ||
    (conv_type == YuvRgbConversionType::FORCE_FLOAT && !final_is_float);
  const bool float_matrix_workflow = force_float || need_int_conversion_full_range;

  // quasi-constexpr, may help optimizer
  if constexpr (std::is_same<pixel_t, uint8_t>::value) bits_per_pixel = 8;
  if constexpr (std::is_same<pixel_t_dst, uint8_t>::value) bits_per_pixel_target = 8;
  if constexpr (std::is_same<pixel_t, uint16_t>::value && !lessthan16bit) bits_per_pixel = 16;
  if constexpr (std::is_same<pixel_t_dst, uint16_t>::value && !lessthan16bit_target) bits_per_pixel_target = 16;
  if constexpr (conv_type == YuvRgbConversionType::NATIVE_INT) bits_per_pixel_target = bits_per_pixel;

  const int bit_diff = need_int_conversion ? bits_per_pixel_target - bits_per_pixel : 0;
  const int target_shift = need_int_conversion_narrow_range ? INT_ARITH_SHIFT - bit_diff : INT_ARITH_SHIFT;
  const int ROUNDER = (final_is_float || float_matrix_workflow) ? 0 : (1 << (target_shift - 1));

  const float out_offset_f = m.offset_out_f_32;
  const int half_pixel_offset = 1 << (bits_per_pixel - 1);
  const int half_pixel_offset_target = 1 << (bits_per_pixel_target - 1);
  const int max_pixel_value_target = (1 << bits_per_pixel_target) - 1;

  bits_conv_constants conversion_ranges;
  const bool full_scale_d = m.offset_out == 0;
  get_bits_conv_constants(conversion_ranges, false, full_scale_d, full_scale_d, bits_per_pixel, bits_per_pixel_target);

  constexpr int int_arithmetic_shift = 1 << INT_ARITH_SHIFT;
  float scale_f = conversion_ranges.mul_factor;
  if (final_is_float && !float_matrix_workflow)
    scale_f = scale_f / int_arithmetic_shift;

  const int offset_in_scalar = m.offset_in;
  const int offset_out_scalar = m.offset_out;

  int round_mask_plus_offset_out_i;
  int round_mask_plus_offset_out_chroma_i;

  if constexpr (!float_matrix_workflow) {
    round_mask_plus_offset_out_i = final_is_float ? 0 : ROUNDER + (offset_out_scalar << INT_ARITH_SHIFT);
    round_mask_plus_offset_out_chroma_i = final_is_float ? 0 : ROUNDER + (half_pixel_offset << INT_ARITH_SHIFT);
  }

  for (int y = 0; y < height; y++) {
    const pixel_t* src0 = reinterpret_cast<const pixel_t*>(srcp[0]); // Y or G
    const pixel_t* src1 = reinterpret_cast<const pixel_t*>(srcp[1]); // U or B
    const pixel_t* src2 = reinterpret_cast<const pixel_t*>(srcp[2]); // V or R
    pixel_t_dst* d0 = reinterpret_cast<pixel_t_dst*>(dstp[0]); // G or Y
    pixel_t_dst* d1 = reinterpret_cast<pixel_t_dst*>(dstp[1]); // B or U
    pixel_t_dst* d2 = reinterpret_cast<pixel_t_dst*>(dstp[2]); // R or V

    for (int x = 0; x < width; x++) {
      float in0_f, in1_f, in2_f;
      int in0_i, in1_i, in2_i;

      // Load inputs
      if constexpr (std::is_floating_point<pixel_t>::value) {
        in0_f = src0[x];
        in1_f = src1[x];
        in2_f = src2[x];
      }
      else {
        in0_i = static_cast<int>(src0[x]);
        in1_i = static_cast<int>(src1[x]);
        in2_i = static_cast<int>(src2[x]);
      }

      float out0_f, out1_f, out2_f;
      safe_int_t out0_raw, out1_raw, out2_raw; // Keep as int64_t for 16-bit (see would_need_64bit_v_patch remarks above)

      if constexpr (float_matrix_workflow) {
        // Float workflow
        if constexpr (!std::is_floating_point<pixel_t>::value) {
          // Convert integer to float and apply input offset
          if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
            // YUV input
            in0_f = static_cast<float>(in0_i + offset_in_scalar); // Y
            in1_f = static_cast<float>(in1_i - half_pixel_offset); // U
            in2_f = static_cast<float>(in2_i - half_pixel_offset); // V
          }
          else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_Y || direction == ConversionDirection::RGB_TO_RGB) {
            // RGB input
            in0_f = static_cast<float>(in0_i + offset_in_scalar); // G
            in1_f = static_cast<float>(in1_i + offset_in_scalar); // B
            in2_f = static_cast<float>(in2_i + offset_in_scalar); // R
          }
        }
        else {
          // Float input - apply offset
          if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
            in0_f += m.offset_in_f; // Y only
          }
          else {
            in0_f += m.offset_in_f; // G
            in1_f += m.offset_in_f; // B
            in2_f += m.offset_in_f; // R
          }
        }

        // Matrix multiply - coefficient order depends on direction
        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
          // YUV input: in0=Y, in1=U, in2=V
          out0_f = (m.y_g_f * in0_f + m.u_g_f * in1_f + m.v_g_f * in2_f) * scale_f; // G or Y
          out1_f = (m.y_b_f * in0_f + m.u_b_f * in1_f + m.v_b_f * in2_f) * scale_f; // B or U
          out2_f = (m.y_r_f * in0_f + m.u_r_f * in1_f + m.v_r_f * in2_f) * scale_f; // R or V
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB) {
          // RGB input: in0=G, in1=B, in2=R
          out0_f = (m.y_g_f * in0_f + m.y_b_f * in1_f + m.y_r_f * in2_f) * scale_f; // Y or G
          out1_f = (m.u_g_f * in0_f + m.u_b_f * in1_f + m.u_r_f * in2_f) * scale_f; // U or B
          out2_f = (m.v_g_f * in0_f + m.v_b_f * in1_f + m.v_r_f * in2_f) * scale_f; // V or R
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
          // RGB input: in0=G, in1=B, in2=R
          out0_f = (m.y_g_f * in0_f + m.y_b_f * in1_f + m.y_r_f * in2_f) * scale_f; // Y or G
        }

        // Add output offset
        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
          // RGB output: same offset for all
          float rgb_offset_scaled = m.offset_out_f * scale_f;
          out0_f += rgb_offset_scaled;
          out1_f += rgb_offset_scaled;
          out2_f += rgb_offset_scaled;
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {
          // YUV output: different offsets
          float y_offset_scaled = m.offset_out_f * scale_f;
          float uv_center = final_is_float ? 0.0f : (float)half_pixel_offset_target;
          out0_f += y_offset_scaled; // Y
          out1_f += uv_center; // U
          out2_f += uv_center; // V
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
          // Y output: single Y offsets
          float y_offset_scaled = m.offset_out_f * scale_f;
          out0_f += y_offset_scaled; // Y
        }

        if constexpr (final_is_float) {
          d0[x] = static_cast<pixel_t_dst>(out0_f);
          if constexpr (direction != ConversionDirection::RGB_TO_Y) {
            d1[x] = static_cast<pixel_t_dst>(out1_f);
            d2[x] = static_cast<pixel_t_dst>(out2_f);
          }
        }
        else {
          // Convert to integer with rounding
          int out0_i = static_cast<int>(out0_f + 0.5f);
          out0_i = std::clamp(out0_i, 0, max_pixel_value_target);
          d0[x] = static_cast<pixel_t_dst>(out0_i);

          if constexpr (direction != ConversionDirection::RGB_TO_Y) {
            int out1_i = static_cast<int>(out1_f + 0.5f);
            int out2_i = static_cast<int>(out2_f + 0.5f);
            out1_i = std::clamp(out1_i, 0, max_pixel_value_target);
            out2_i = std::clamp(out2_i, 0, max_pixel_value_target);
            d1[x] = static_cast<pixel_t_dst>(out1_i);
            d2[x] = static_cast<pixel_t_dst>(out2_i);
          }
        }
      }
      else {
        // Integer workflow
        // Apply input offset and centering
        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
          // YUV input
          in0_i = in0_i + offset_in_scalar; // Y
          in1_i = in1_i - half_pixel_offset; // U (centered)
          in2_i = in2_i - half_pixel_offset; // V (centered)
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB || direction == ConversionDirection::RGB_TO_Y) {
          // RGB input
          in0_i = in0_i + offset_in_scalar; // G
          in1_i = in1_i + offset_in_scalar; // B
          in2_i = in2_i + offset_in_scalar; // R
        }

        // Matrix multiply with appropriate rounding/offset
        int round_and_offset_0, round_and_offset_1, round_and_offset_2;

        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
          // RGB output: same offset for all
          round_and_offset_0 = round_mask_plus_offset_out_i;
          round_and_offset_1 = round_mask_plus_offset_out_i;
          round_and_offset_2 = round_mask_plus_offset_out_i;
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {
          // YUV output: different offsets
          round_and_offset_0 = round_mask_plus_offset_out_i; // Y
          round_and_offset_1 = round_mask_plus_offset_out_chroma_i; // U
          round_and_offset_2 = round_mask_plus_offset_out_chroma_i; // V
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
          // Y output: single Y offset
          round_and_offset_0 = round_mask_plus_offset_out_i; // Y
        }

        if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::YUV_TO_YUV) {
          // YUV input: in0=Y, in1=U, in2=V
          out0_raw = (safe_int_t)m.y_g * in0_i + (safe_int_t)m.u_g * in1_i + (safe_int_t)m.v_g * in2_i + round_and_offset_0;
          out1_raw = (safe_int_t)m.y_b * in0_i + (safe_int_t)m.u_b * in1_i + (safe_int_t)m.v_b * in2_i + round_and_offset_1;
          out2_raw = (safe_int_t)m.y_r * in0_i + (safe_int_t)m.u_r * in1_i + (safe_int_t)m.v_r * in2_i + round_and_offset_2;
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::RGB_TO_RGB) {
          // RGB input: in0=G, in1=B, in2=R
          out0_raw = (safe_int_t)m.y_g * in0_i + (safe_int_t)m.y_b * in1_i + (safe_int_t)m.y_r * in2_i + round_and_offset_0;
          out1_raw = (safe_int_t)m.u_g * in0_i + (safe_int_t)m.u_b * in1_i + (safe_int_t)m.u_r * in2_i + round_and_offset_1;
          out2_raw = (safe_int_t)m.v_g * in0_i + (safe_int_t)m.v_b * in1_i + (safe_int_t)m.v_r * in2_i + round_and_offset_2;
        }
        else if constexpr (direction == ConversionDirection::RGB_TO_Y) {
          // RGB input: in0=G, in1=B, in2=R
          out0_raw = (safe_int_t)m.y_g * in0_i + (safe_int_t)m.y_b * in1_i + (safe_int_t)m.y_r * in2_i + round_and_offset_0;
        }

        if constexpr (final_is_float) {
          // Integer workflow + float output
          out0_f = static_cast<float>(out0_raw) * scale_f + out_offset_f;
          if constexpr (direction == ConversionDirection::YUV_TO_RGB || direction == ConversionDirection::RGB_TO_RGB) {
            // offset for the rest two only RGB output
            out1_f = static_cast<float>(out1_raw) * scale_f + out_offset_f;
            out2_f = static_cast<float>(out2_raw) * scale_f + out_offset_f;
          }
          else if constexpr (direction == ConversionDirection::RGB_TO_YUV || direction == ConversionDirection::YUV_TO_YUV) {
            // for YUV the U and V float offset is different : 0.0f
            out1_f = static_cast<float>(out1_raw) * scale_f; // U
            out2_f = static_cast<float>(out2_raw) * scale_f; // V
          }

          d0[x] = static_cast<pixel_t_dst>(out0_f);
          if constexpr (direction != ConversionDirection::RGB_TO_Y) {
            d1[x] = static_cast<pixel_t_dst>(out1_f);
            d2[x] = static_cast<pixel_t_dst>(out2_f);
          }
        }
        else {
          // Integer workflow + integer output
          int out0_i = static_cast<int>(out0_raw >> target_shift);
          out0_i = std::clamp(out0_i, 0, max_pixel_value_target);
          d0[x] = static_cast<pixel_t_dst>(out0_i);

          if constexpr (direction != ConversionDirection::RGB_TO_Y) {
            int out1_i = static_cast<int>(out1_raw >> target_shift);
            int out2_i = static_cast<int>(out2_raw >> target_shift);

            // We are in integer domain, no convenient SIMD packus! We cannot spare min-max clamp to constexpr (lessthan16bit_target) {
            // We need to do it always
            out1_i = std::clamp(out1_i, 0, max_pixel_value_target);
            out2_i = std::clamp(out2_i, 0, max_pixel_value_target);

            d1[x] = static_cast<pixel_t_dst>(out1_i);
            d2[x] = static_cast<pixel_t_dst>(out2_i);
          }
        }
      }
    }

    dstp[0] += dstPitch[0];
    if constexpr (direction != ConversionDirection::RGB_TO_Y) {
      dstp[1] += dstPitch[1];
      dstp[2] += dstPitch[2];
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
  }
}

// Dispatcher matching AVX2/SSE2 structure
template<ConversionDirection direction, typename pixel_t_src, bool lessthan16bit>
void convert_yuv_to_planarrgb_c(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m,
  int bits_per_pixel, int bits_per_pixel_target, bool force_float)
{
  // Handle forced float or float input
  if (force_float || std::is_floating_point<pixel_t_src>::value) {
    if (bits_per_pixel_target == 8) {
      convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else if (bits_per_pixel_target < 16) {
      convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else if (bits_per_pixel_target == 16) {
      convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    else {
      convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, false, float, YuvRgbConversionType::FORCE_FLOAT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
    }
    return;
  }

  if constexpr (!std::is_floating_point<pixel_t_src>::value) {
    const bool need_conversion = bits_per_pixel_target != bits_per_pixel;
    if (!need_conversion) {
      convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, lessthan16bit, pixel_t_src, YuvRgbConversionType::NATIVE_INT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      return;
    }

    const bool full_d = m.offset_out == 0;
    if (bits_per_pixel_target >= 8 && bits_per_pixel <= 16) {
      if (bits_per_pixel_target == 8) {
        if (full_d)
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint8_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else if (bits_per_pixel_target < 16) {
        if (full_d)
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, true, uint16_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else if (bits_per_pixel_target == 16) {
        if (full_d)
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::BITCONV_INT_FULL>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
        else
          convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, false, uint16_t, YuvRgbConversionType::BITCONV_INT_LIMITED>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
      else {
        convert_yuv_to_planarrgb_c_internal<direction, pixel_t_src, lessthan16bit, false, float, YuvRgbConversionType::FLOAT_OUTPUT>(dstp, dstPitch, srcp, srcPitch, width, height, m, bits_per_pixel, bits_per_pixel_target);
      }
    }
  }
}

// Template instantiations for C version - all 4 directions
// YUV_TO_RGB
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint8_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint16_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint16_t, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, float, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// RGB_TO_YUV
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint8_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint16_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, uint16_t, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_YUV, float, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// RGB_TO_Y
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint8_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint16_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, uint16_t, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::RGB_TO_Y, float, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// YUV_TO_YUV
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_YUV, uint8_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_YUV, uint16_t, true>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_YUV, uint16_t, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);
template void convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_YUV, float, false>(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m, int bits_per_pixel, int bits_per_pixel_target, bool force_float);

PVideoFrame __stdcall ConvertYUV444ToRGB::GetFrame(int n, IScriptEnvironment* env)
{
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  auto props = env->getFramePropsRW(dst);
  update_Matrix_and_ColorRange(props, theOutMatrix, theOutColorRange, env);
  update_ChromaLocation(props, -1, env); // RGB target: delete _ChromaLocation

  const BYTE* srcY = src->GetReadPtr(PLANAR_Y);
  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);
  const BYTE* srcA = src->GetReadPtr(PLANAR_A);

  BYTE* dstp = dst->GetWritePtr();

  const int src_pitch_y = src->GetPitch(PLANAR_Y);
  const int src_pitch_uv = src->GetPitch(PLANAR_U);
  const int src_pitch_a = src->GetPitch(PLANAR_A); // zero if no Alpha

  const int dst_pitch = dst->GetPitch();

  // Legacy direct YV24 -> packed 8 bit RGB24/RGB32
  // quality=true float-inside or bit-depth changing transformation is handled through planar RGB intermediate.

  if (pixel_step == 3 || pixel_step == 4) {
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_AVX2) {
      if (pixel_step == 4) {
        // RGB32
        if (src_pitch_a) // move alpha channel from YUVA
          convert_yv24_to_rgb_avx2<4, true>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
        else
          convert_yv24_to_rgb_avx2<4, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
      }
      else {
        // RGB24
        convert_yv24_to_rgb_avx2<3, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
      }
      return dst;
    }
    else if (env->GetCPUFlags() & CPUF_SSE2) {
      if (pixel_step == 4) {
        if (src_pitch_a) // move alpha channel from YUVA
          convert_yv24_to_rgb_sse2<4, true>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
        else
          convert_yv24_to_rgb_sse2<4, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
      }
      else {
        if (env->GetCPUFlags() & CPUF_SSSE3) {
          convert_yv24_to_rgb_ssse3<3, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
        }
        else {
          convert_yv24_to_rgb_sse2<3, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
        }
      }
      return dst;
    }

#ifdef X86_32
    // packed RGB24 and RGB32
    if ((src_pitch_a == 0) && (env->GetCPUFlags() & CPUF_MMX)) {
      if (pixel_step == 4) {
        convert_yv24_to_rgb_mmx<4>(dstp, srcY, srcU, srcV, dst_pitch, src_pitch_y, src_pitch_uv, vi.width, vi.height, matrix);
      }
      else {
        convert_yv24_to_rgb_mmx<3>(dstp, srcY, srcU, srcV, dst_pitch, src_pitch_y, src_pitch_uv, vi.width, vi.height, matrix);
      }
      return dst;
    }
#endif
#endif

    // Unoptimized C-code.
    // rgb24 or rgb32, packed RGB
    if (pixel_step == 3 || pixel_step == 4) {
      if (pixel_step == 4) {
        if (src_pitch_a) // move alpha channel from YUVA
          convert_yv24_to_rgb_c<4, true>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
        else
          convert_yv24_to_rgb_c<4, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
      }
      else { // 3 RGB24
        convert_yv24_to_rgb_c<3, false>(dstp, srcY, srcU, srcV, srcA, dst_pitch, src_pitch_y, src_pitch_uv, src_pitch_a, vi.width, vi.height, matrix);
      }
      return dst;
    }
  } // YV24->RGB24/RGB32

  // rgb48 or rgb64, packed RGB
  if (pixel_step == 6 || pixel_step == 8) {
    // cannot happen, handled earlier, RGB48/64 are post converted from RGB(A)P16
    env->ThrowError("ConvertFrom 444: RGB48/64 conversion should have been handled earlier through planar RGB intermediate");
  }

  if (pixel_step < 0) {
    // -1: RGBP  -2:RGBAP
    // Common task: copy or fill alpha channel

    // YUV444 -> PlanarRGB
    // YUVA444 -> PlanarRGBA
    bool targetHasAlpha = pixel_step == -2;

    BYTE *dstpG = dst->GetWritePtr(PLANAR_G);
    BYTE *dstpB = dst->GetWritePtr(PLANAR_B);
    BYTE *dstpR = dst->GetWritePtr(PLANAR_R);

    // copy or fill alpha
    BYTE *dstpA;
    if (targetHasAlpha) {
      dstpA = dst->GetWritePtr(PLANAR_A);
      int heightA = dst->GetHeight(PLANAR_A);
      int rowsizeA = dst->GetRowSize(PLANAR_A);
      int dst_pitchA = dst->GetPitch(PLANAR_A);
        // simple copy
      if (src->GetRowSize(PLANAR_A)) {

        if (source_bit_depth == target_bit_depth) {
          // vi.IsYUVA() no-no! vi is already the target video type
          env->BitBlt(dstpA, dst_pitchA, src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A), src->GetRowSize(PLANAR_A_ALIGNED), src->GetHeight(PLANAR_A));
        }
        else {
          // RGBPA target + different bit-depth
          // Alpha bit depth conversion needed, normal planes are done in-process.
          // we need only conv_function_a, proper dispatching from ConvertBits
          conv_function_a(src->GetReadPtr(PLANAR_A), dstpA, src->GetRowSize(PLANAR_A), heightA, src->GetPitch(PLANAR_A), dst_pitchA, source_bit_depth, target_bit_depth, -1);
        }
      }
      else {
        // fill default transparency
        switch (target_bit_depth)
        {
        case 1:
          fill_plane<BYTE>(dstpA, heightA, rowsizeA, dst_pitchA, 255);
          break;
        case 2:
          fill_plane<uint16_t>(dstpA, heightA, rowsizeA, dst_pitchA, (1 << target_bit_depth) - 1);
          break;
        case 4:
          fill_plane<float>(dstpA, heightA, rowsizeA, dst_pitchA, 1.0f);
          break;
        }
      }
    }

    int dst_pitchG = dst->GetPitch(PLANAR_G);
    int dst_pitchB = dst->GetPitch(PLANAR_B);
    int dst_pitchR = dst->GetPitch(PLANAR_R);

    // always start from source bit depth
    int bits_per_pixel = source_bit_depth;

    const BYTE* srcp[3] = { src->GetReadPtr(PLANAR_Y), src->GetReadPtr(PLANAR_U), src->GetReadPtr(PLANAR_V) };
    const int srcPitch[3] = { src->GetPitch(PLANAR_Y), src->GetPitch(PLANAR_U), src->GetPitch(PLANAR_V) };

    BYTE* dstp[3] = { dstpG, dstpB, dstpR };
    int dstPitch[3] = { dst_pitchG, dst_pitchB, dst_pitchR };

    // alpha is already handled above

    const bool force_float = quality;

#ifdef INTEL_INTRINSICS

    if (env->GetCPUFlags() & CPUF_AVX2)
    {
      if (bits_per_pixel == 8)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel < 16)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 16)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 32)
        convert_yuv_to_planarrgb_avx2<ConversionDirection::YUV_TO_RGB, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      return dst;
    }

    if (env->GetCPUFlags() & CPUF_SSE2)
    {
      if (bits_per_pixel == 8)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::YUV_TO_RGB, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel < 16)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::YUV_TO_RGB, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 16)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::YUV_TO_RGB, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      else if (bits_per_pixel == 32)
        convert_yuv_to_planarrgb_sse2<ConversionDirection::YUV_TO_RGB, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
      return dst;
    }

#endif

    if (bits_per_pixel == 8)
      convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint8_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel < 16)
      convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint16_t, true>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 16)
      convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, uint16_t, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    else if (bits_per_pixel == 32)
      convert_yuv_to_planarrgb_c<ConversionDirection::YUV_TO_RGB, float, false>(dstp, dstPitch, srcp, srcPitch, vi.width, vi.height, matrix, bits_per_pixel, target_bit_depth, force_float);
    return dst;
  }
  return dst;
}

/************************************
 * YUY2 to YV16
 ************************************/

ConvertYUY2ToYV16_or_Y::ConvertYUY2ToYV16_or_Y(PClip src, bool _to_y, IScriptEnvironment* env) : GenericVideoFilter(src), to_y(_to_y) {

  if (!vi.IsYUY2())
    env->ThrowError("ConvertYUY2ToYV16/Y: Only YUY2 is allowed as input");

  vi.pixel_type = to_y ? VideoInfo::CS_Y8 : VideoInfo::CS_YV16;

}

static void convert_yuy2_to_y8_c(const BYTE* srcp, BYTE* dstp, size_t src_pitch, size_t dst_pitch, size_t width, size_t height)
{
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      dstp[x] = srcp[x * 2];
    }
    srcp += src_pitch;
    dstp += dst_pitch;
  }
}


static void convert_yuy2_to_yv16_c(const BYTE *srcp, BYTE *dstp_y, BYTE *dstp_u, BYTE *dstp_v, size_t src_pitch, size_t dst_pitch_y, size_t dst_pitch_uv, size_t width, size_t height)
{
  width /= 2;

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      dstp_y[x * 2]     = srcp[x * 4 + 0];
      dstp_y[x * 2 + 1] = srcp[x * 4 + 2];
      dstp_u[x]         = srcp[x * 4 + 1];
      dstp_v[x]         = srcp[x * 4 + 3];
    }
    srcp += src_pitch;
    dstp_y += dst_pitch_y;
    dstp_u += dst_pitch_uv;
    dstp_v += dst_pitch_uv;
  }
}


PVideoFrame __stdcall ConvertYUY2ToYV16_or_Y::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const BYTE* srcP = src->GetReadPtr();
  BYTE* dstY = dst->GetWritePtr(PLANAR_Y);

  if (to_y) {
    update_ChromaLocation(env->getFramePropsRW(dst), -1, env); // Y8 target: delete _ChromaLocation
    // YUY2 to Y8, luma only
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2) {
      convert_yuy2_to_y8_sse2(srcP, dstY, src->GetPitch(), dst->GetPitch(PLANAR_Y), vi.width, vi.height);
    }
    else
#endif
    {
      convert_yuy2_to_y8_c(srcP, dstY, src->GetPitch(), dst->GetPitch(PLANAR_Y), vi.width, vi.height);
    }
    return dst;
  }
  else {
    BYTE* dstU = dst->GetWritePtr(PLANAR_U);
    BYTE* dstV = dst->GetWritePtr(PLANAR_V);
    // YUY2 to YV16, separate luma and chroma
#ifdef INTEL_INTRINSICS
    if (env->GetCPUFlags() & CPUF_SSE2) {
      convert_yuy2_to_yv16_sse2(srcP, dstY, dstU, dstV, src->GetPitch(), dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), vi.width, vi.height);
    }
    else
#ifdef X86_32
      if (env->GetCPUFlags() & CPUF_MMX) {
        convert_yuy2_to_yv16_mmx(srcP, dstY, dstU, dstV, src->GetPitch(), dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), vi.width, vi.height);
      }
      else
#endif
#endif
      {
        convert_yuy2_to_yv16_c(srcP, dstY, dstU, dstV, src->GetPitch(), dst->GetPitch(PLANAR_Y), dst->GetPitch(PLANAR_U), vi.width, vi.height);
      }
  }

  return dst;
}


AVSValue __cdecl ConvertYUY2ToYV16_or_Y::Create(AVSValue args, void*, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsYV16())
    return clip;
  return new ConvertYUY2ToYV16_or_Y(clip, false /*to_y*/, env);
}

/************************************
 * YV16 to YUY2
 ************************************/

ConvertYV16ToYUY2::ConvertYV16ToYUY2(PClip src, IScriptEnvironment* env) : GenericVideoFilter(src) {

  if (!vi.IsYV16())
    env->ThrowError("ConvertYV16ToYUY2: Only YV16 is allowed as input");

  vi.pixel_type = VideoInfo::CS_YUY2;
}

static void convert_yv16_to_yuy2_c(const BYTE *srcp_y, const BYTE *srcp_u, const BYTE *srcp_v, BYTE *dstp, size_t src_pitch_y, size_t src_pitch_uv, size_t dst_pitch, size_t width, size_t height) {
  for (size_t y=0; y < height; y++) {
    for (size_t x=0; x < width / 2; x++) {
      dstp[x*4+0] = srcp_y[x*2];
      dstp[x*4+1] = srcp_u[x];
      dstp[x*4+2] = srcp_y[x*2+1];
      dstp[x*4+3] = srcp_v[x];
    }
    srcp_y += src_pitch_y;
    srcp_u += src_pitch_uv;
    srcp_v += src_pitch_uv;
    dstp += dst_pitch;
  }
}

PVideoFrame __stdcall ConvertYV16ToYUY2::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);
  PVideoFrame dst = env->NewVideoFrameP(vi, &src);

  const BYTE* srcY = src->GetReadPtr(PLANAR_Y);
  const BYTE* srcU = src->GetReadPtr(PLANAR_U);
  const BYTE* srcV = src->GetReadPtr(PLANAR_V);

  BYTE* dstp = dst->GetWritePtr();

#ifdef INTEL_INTRINSICS
  if (env->GetCPUFlags() & CPUF_SSE2) {
    //U and V don't have to be aligned since we user movq to read from those
    convert_yv16_to_yuy2_sse2(srcY, srcU, srcV, dstp, src->GetPitch(PLANAR_Y), src->GetPitch(PLANAR_U), dst->GetPitch(), vi.width, vi.height);
  } else
#ifdef X86_32
  if (env->GetCPUFlags() & CPUF_MMX) {
    convert_yv16_to_yuy2_mmx(srcY, srcU, srcV, dstp, src->GetPitch(PLANAR_Y), src->GetPitch(PLANAR_U), dst->GetPitch(), vi.width, vi.height);
  } else
#endif
#endif
  {
    convert_yv16_to_yuy2_c(srcY, srcU, srcV, dstp, src->GetPitch(PLANAR_Y), src->GetPitch(PLANAR_U), dst->GetPitch(), vi.width, vi.height);
  }

  return dst;
}


AVSValue __cdecl ConvertYV16ToYUY2::Create(AVSValue args, void*, IScriptEnvironment* env) {
  PClip clip = args[0].AsClip();
  if (clip->GetVideoInfo().IsYUY2())
    return clip;
  return new ConvertYV16ToYUY2(clip, env);
}

/**********************************************
 * Converter between arbitrary planar formats
 *
 * This uses plane copy for luma, and a custom
 * resizer for chroma
 *
 * (c) Klaus Post, 2005
 * (c) Ian Brabham, 2011
 **********************************************/

ConvertToPlanarGeneric::ConvertToPlanarGeneric(
  PClip src, int dst_space, bool interlaced,
  int _ChromaLocation_In, 
  const AVSValue& chromaResampler, const AVSValue& param1, const AVSValue& param2, const AVSValue& param3,
  int _ChromaLocation_Out,
  IScriptEnvironment* env) : 
  GenericVideoFilter(src), ChromaLocation_In(_ChromaLocation_In), ChromaLocation_Out(_ChromaLocation_Out)
{
  Yinput = vi.NumComponents() == 1;
  pixelsize = vi.ComponentSize();

  if (Yinput) {
    vi.pixel_type = dst_space;
    if (vi.ComponentSize() != pixelsize)
      env->ThrowError("Convert: Conversion from %d to %d-byte format not supported.", pixelsize, vi.ComponentSize());
    // no more preparation here, GetFrame simply returns Y
    return;
  }

  // Y only output from any YUV: no chroma involved, GetFrame will snatch Y plane
  if (dst_space == VideoInfo::CS_Y8 || dst_space == VideoInfo::CS_Y10 ||
    dst_space == VideoInfo::CS_Y12 || dst_space == VideoInfo::CS_Y14 || dst_space == VideoInfo::CS_Y16 ||
    dst_space == VideoInfo::CS_Y32)
  {
    vi.pixel_type = dst_space;
    return;
  }

  auto Is420 = [](int pix_type) {
    return pix_type == VideoInfo::CS_YV12 || pix_type == VideoInfo::CS_I420 ||
      pix_type == VideoInfo::CS_YUV420P10 || pix_type == VideoInfo::CS_YUV420P12 ||
      pix_type == VideoInfo::CS_YUV420P14 || pix_type == VideoInfo::CS_YUV420P16 ||
      pix_type == VideoInfo::CS_YUV420PS ||
        pix_type == VideoInfo::CS_YUVA420 ||
      pix_type == VideoInfo::CS_YUVA420P10 || pix_type == VideoInfo::CS_YUVA420P12 ||
      pix_type == VideoInfo::CS_YUVA420P14 || pix_type == VideoInfo::CS_YUVA420P16 ||
      pix_type == VideoInfo::CS_YUVA420PS;
  };

  if (!Is420(vi.pixel_type) && !Is420(dst_space))
    interlaced = false;  // Ignore, if YV12 is not involved.
  //if (interlaced) env->ThrowError("Convert: Interlaced only available with 4:2:0 color spaces.");

  // Describe input pixel positioning
  float xdInU = 0.0f, txdInU = 0.0f, bxdInU = 0.0f;
  float ydInU = 0.0f, tydInU = 0.0f, bydInU = 0.0f;
  float xdInV = 0.0f, txdInV = 0.0f, bxdInV = 0.0f;
  float ydInV = 0.0f, tydInV = 0.0f, bydInV = 0.0f;

  /*
    "mpeg1", "center"
      - 1-1 averaging kernel (4:2:0, 4:2:2 (horizontal only))
    "mpeg2", "left" (4:2:0, 4:2:2)
      - top-field samples are sited 1/4 sample below the luma samples (4:2:0)
      - bottom-field samples are sited 1/4 sample above the luma samples (4:2:0)
      - horizontal 1-2-1 kernel 
    "dv": (4:2:0)
      Chroma samples are sited on top of luma samples, but CB and CR samples are sited on alternate lines.
      - top: V
      - bottom: U
    "top_left" (4:2:0)
      - horizontal 1-2-1 vertical 1-2-1
  */

  if (Is420(vi.pixel_type)) {
    switch (ChromaLocation_In) {
      case ChromaLocation_e::AVS_CHROMA_DV: // spec. avisynth
        xdInU = 0.0f; ydInU = 1.0f; txdInU = 0.0f; tydInU = 1.0f; bxdInU = 0.0f; bydInU = 1.0f; // Cb
        xdInV = 0.0f; ydInV = 0.0f; txdInV = 0.0f; tydInV = 0.0f; bxdInV = 0.0f; bydInV = 0.0f; // Cr
        break;
      case ChromaLocation_e::AVS_CHROMA_TOP:
        xdInU = 0.5f, ydInU = 0.0f; txdInU = 0.5f; tydInU = 0.0f; bxdInU = 0.5f; bydInU = 0.5f;
        xdInV = 0.5f, ydInV = 0.0f; txdInV = 0.5f; tydInV = 0.0f; bxdInV = 0.5f; bydInV = 0.5f;
        break;
      case ChromaLocation_e::AVS_CHROMA_CENTER: // mpeg1, center
        xdInU = 0.5f, ydInU = 0.5f; txdInU = 0.5f; tydInU = 0.25f; bxdInU = 0.5f; bydInU = 0.75f;
        xdInV = 0.5f, ydInV = 0.5f; txdInV = 0.5f; tydInV = 0.25f; bxdInV = 0.5f; bydInV = 0.75f;
        break;
      case ChromaLocation_e::AVS_CHROMA_BOTTOM:
        xdInU = 0.5f, ydInU = 1.0f; txdInU = 0.5f; tydInU = 0.5f; bxdInU = 0.5f; bydInU = 1.0f;
        xdInV = 0.5f, ydInV = 1.0f; txdInV = 0.5f; tydInV = 0.5f; bxdInV = 0.5f; bydInV = 1.0f;
        break;
      case ChromaLocation_e::AVS_CHROMA_TOP_LEFT:
        xdInU = 0.0f; ydInU = 0.0f; txdInU = 0.0f; tydInU = 0.0f; bxdInU = 0.0f; bydInU = 0.5f;
        xdInV = 0.0f; ydInV = 0.0f; txdInV = 0.0f; tydInV = 0.0f; bxdInV = 0.0f; bydInV = 0.5f;
        break;
      case ChromaLocation_e::AVS_CHROMA_LEFT: // left, mpeg2
        xdInU = 0.0f; ydInU = 0.5f; txdInU = 0.0f; tydInU = 0.25f; bxdInU = 0.0f; bydInU = 0.75f;
        xdInV = 0.0f; ydInV = 0.5f; txdInV = 0.0f; tydInV = 0.25f; bxdInV = 0.0f; bydInV = 0.75f;
        break;
      case ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT:
        xdInU = 0.0f; ydInU = 1.0f; txdInU = 0.0f; tydInU = 0.5f; bxdInU = 0.0f; bydInU = 1.0f;
        xdInV = 0.0f; ydInV = 1.0f; txdInV = 0.0f; tydInV = 0.5f; bxdInV = 0.0f; bydInV = 1.0f;
        break;
      default:
        env->ThrowError("Convert: not supported ChromaPlacement for 4:2:0 input.");
    }
  }
  else if (vi.Is422()) {
    switch (ChromaLocation_In) {
    case ChromaLocation_e::AVS_CHROMA_CENTER: // center
      xdInU = 0.5f, ydInU = 0.0f; txdInU = 0.5f; tydInU = 0.0f; bxdInU = 0.5f; bydInU = 0.0f;
      xdInV = 0.5f, ydInV = 0.0f; txdInV = 0.5f; tydInV = 0.0f; bxdInV = 0.5f; bydInV = 0.0f;
      break;
    case ChromaLocation_e::AVS_CHROMA_TOP_LEFT: // treated as left
    case ChromaLocation_e::AVS_CHROMA_LEFT: // left, mpeg2
    case ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT: // treated as left
      xdInU = 0.0f; ydInU = 0.0f; txdInU = 0.0f; tydInU = 0.0f; bxdInU = 0.0f; bydInU = 0.0f;
      xdInV = 0.0f; ydInV = 0.0f; txdInV = 0.0f; tydInV = 0.0f; bxdInV = 0.0f; bydInV = 0.0f;
      break;
    default:
      env->ThrowError("Convert: not supported ChromaPlacement for 4:2:2 input.");
    }
  }
  else if (vi.IsYV411()) {
    if (ChromaLocation_In >= 0
      && ChromaLocation_In != ChromaLocation_e::AVS_CHROMA_TOP_LEFT
      && ChromaLocation_In != ChromaLocation_e::AVS_CHROMA_LEFT
      && ChromaLocation_In != ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT
      )
    {
      // if given, only the 'left'-ish versions are accepted, this is how it is treated
      env->ThrowError("Convert: not supported ChromaPlacement for 4:1:1 input. Only 'left'-style is allowed if given.");
    }
  }
  else if (ChromaLocation_In >= 0)
    env->ThrowError("Convert: Input ChromaPlacement is invalid for this format.");

  const int xsIn = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int ysIn = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);

  // change vi to the output format
  vi.pixel_type = dst_space;

  if (vi.ComponentSize() != pixelsize)
    env->ThrowError("Convert: Conversion from %d to %d-byte format not supported.", pixelsize, vi.ComponentSize());

  // Describe output pixel positioning
  float xdOutU = 0.0f, txdOutU = 0.0f, bxdOutU = 0.0f;
  float ydOutU = 0.0f, tydOutU = 0.0f, bydOutU = 0.0f;
  float xdOutV = 0.0f, txdOutV = 0.0f, bxdOutV = 0.0f;
  float ydOutV = 0.0f, tydOutV = 0.0f, bydOutV = 0.0f;

  if (Is420(vi.pixel_type)) {
    switch (ChromaLocation_Out) {
    case ChromaLocation_e::AVS_CHROMA_DV:
      xdOutU = 0.0f; ydOutU = 1.0f; txdOutU = 0.0f; tydOutU = 1.0f; bxdOutU = 0.0f; bydOutU = 1.0f; // Cb
      xdOutV = 0.0f; ydOutV = 0.0f; txdOutV = 0.0f; tydOutV = 0.0f; bxdOutV = 0.0f; bydOutV = 0.0f; // Cr
      break;
    case ChromaLocation_e::AVS_CHROMA_TOP:
      xdOutU = 0.5f, ydOutU = 0.0f; txdOutU = 0.5f; tydOutU = 0.0f; bxdOutU = 0.5f; bydOutU = 0.5f;
      xdOutV = 0.5f, ydOutV = 0.0f; txdOutV = 0.5f; tydOutV = 0.0f; bxdOutV = 0.5f; bydOutV = 0.5f;
      break;
    case ChromaLocation_e::AVS_CHROMA_CENTER: // mpeg1, center
      xdOutU = 0.5f, ydOutU = 0.5f; txdOutU = 0.5f; tydOutU = 0.25f; bxdOutU = 0.5f; bydOutU = 0.75f;
      xdOutV = 0.5f, ydOutV = 0.5f; txdOutV = 0.5f; tydOutV = 0.25f; bxdOutV = 0.5f; bydOutV = 0.75f;
      break;
    case ChromaLocation_e::AVS_CHROMA_BOTTOM:
      xdOutU = 0.5f, ydOutU = 1.0f; txdOutU = 0.5f; tydOutU = 0.5f; bxdOutU = 0.5f; bydOutU = 1.0f;
      xdOutV = 0.5f, ydOutV = 1.0f; txdOutV = 0.5f; tydOutV = 0.5f; bxdOutV = 0.5f; bydOutV = 1.0f;
      break;
    case ChromaLocation_e::AVS_CHROMA_TOP_LEFT:
      xdOutU = 0.0f; ydOutU = 0.0f; txdOutU = 0.0f; tydOutU = 0.0f; bxdOutU = 0.0f; bydOutU = 0.5f;
      xdOutV = 0.0f; ydOutV = 0.0f; txdOutV = 0.0f; tydOutV = 0.0f; bxdOutV = 0.0f; bydOutV = 0.5f;
      break;
    case ChromaLocation_e::AVS_CHROMA_LEFT: // left, mpeg2
      xdOutU = 0.0f; ydOutU = 0.5f; txdOutU = 0.0f; tydOutU = 0.25f; bxdOutU = 0.0f; bydOutU = 0.75f;
      xdOutV = 0.0f; ydOutV = 0.5f; txdOutV = 0.0f; tydOutV = 0.25f; bxdOutV = 0.0f; bydOutV = 0.75f;
      break;
    case ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT:
      xdOutU = 0.0f; ydOutU = 1.0f; txdOutU = 0.0f; tydOutU = 0.5f; bxdOutU = 0.0f; bydOutU = 1.0f;
      xdOutV = 0.0f; ydOutV = 1.0f; txdOutV = 0.0f; tydOutV = 0.5f; bxdOutV = 0.0f; bydOutV = 1.0f;
      break;
    default:
      env->ThrowError("Convert: not supported ChromaPlacement for 4:2:0 output.");
    }
  }
  else if (vi.Is422()) {
    switch (ChromaLocation_Out) {
    case ChromaLocation_e::AVS_CHROMA_CENTER: // mpeg1, center
      xdOutU = 0.5f, ydOutU = 0.0f; txdOutU = 0.5f; tydOutU = 0.0f; bxdOutU = 0.5f; bydOutU = 0.0f;
      xdOutV = 0.5f, ydOutV = 0.0f; txdOutV = 0.5f; tydOutV = 0.0f; bxdOutV = 0.5f; bydOutV = 0.0f;
      break;
    case ChromaLocation_e::AVS_CHROMA_TOP_LEFT: // treated as left
    case ChromaLocation_e::AVS_CHROMA_LEFT: // left, mpeg2
    case ChromaLocation_e::AVS_CHROMA_BOTTOM_LEFT: // treated as left
      xdOutU = 0.0f; ydOutU = 0.0f; txdOutU = 0.0f; tydOutU = 0.0f; bxdOutU = 0.0f; bydOutU = 0.0f;
      xdOutV = 0.0f; ydOutV = 0.0f; txdOutV = 0.0f; tydOutV = 0.0f; bxdOutV = 0.0f; bydOutV = 0.0f;
      break;
    default:
      env->ThrowError("Convert: not supported ChromaPlacement for 4:2:2 output.");
    }
  }
  else if (ChromaLocation_Out >= 0)
    env->ThrowError("Convert: Output ChromaPlacement only available with 4:2:0 or 4:2:2 output.");

  const int xsOut = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
  const int xmask = xsOut - 1;
  if (vi.width & xmask)
    env->ThrowError("Convert: Cannot convert if width isn't mod%d!", xsOut);

  const int ysOut = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
  const int ymask = ysOut - 1;
  if (vi.height & ymask)
    env->ThrowError("Convert: Cannot convert if height isn't mod%d!", ysOut);

  int uv_width  = vi.width  >> vi.GetPlaneWidthSubsampling(PLANAR_U);
  int uv_height = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

  ResamplingFunction *filter = getResampler(chromaResampler.AsString("bicubic"), param1, param2, param3, true, env);

  bool P = !lstrcmpi(chromaResampler.AsString(""), "point");

  auto ChrOffset = [P](int sIn, float dIn, int sOut, float dOut) {
    //     (1 - sOut/sIn)/2 + (dOut-dIn)/sIn; // Gavino Jan 2011
    return P ? (dOut - dIn) / sIn : 0.5f + (dOut - dIn - 0.5f*sOut) / sIn;
  };

  const int force = 0;
  const bool preserve_center = true;
  const char *placement_name_notused = nullptr; // n/a
  const int forced_chroma_placement = -1; // no force
  // chroma planes are extracted, behave like Y when resized, no chroma planes involved

  if (interlaced) {
    uv_height /=  2;

    AVSValue tUsubSampling[4] = { ChrOffset(xsIn, txdInU, xsOut, txdOutU), ChrOffset(ysIn, tydInU, ysOut, tydOutU), AVSValue(), AVSValue() };
    AVSValue bUsubSampling[4] = { ChrOffset(xsIn, bxdInU, xsOut, bxdOutU), ChrOffset(ysIn, bydInU, ysOut, bydOutU), AVSValue(), AVSValue() };
    AVSValue tVsubSampling[4] = { ChrOffset(xsIn, txdInV, xsOut, txdOutV), ChrOffset(ysIn, tydInV, ysOut, tydOutV), AVSValue(), AVSValue() };
    AVSValue bVsubSampling[4] = { ChrOffset(xsIn, bxdInV, xsOut, bxdOutV), ChrOffset(ysIn, bydInV, ysOut, bydOutV), AVSValue(), AVSValue() };

    Usource = new SeparateFields(new AssumeParity(new SwapUVToY(child, SwapUVToY::UToY8, env), true), env); // also works for Y16/Y32
    Vsource = new SeparateFields(new AssumeParity(new SwapUVToY(child, SwapUVToY::VToY8, env), true), env); // also works for Y16/Y32

    std::vector<PClip> tbUsource(2); // Interleave() will take ownership of these
    std::vector<PClip> tbVsource(2);

    tbUsource[0] = FilteredResize::CreateResize(new SelectEvery(Usource, 2, 0, env), uv_width, uv_height, tUsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);
    tbUsource[1] = FilteredResize::CreateResize(new SelectEvery(Usource, 2, 1, env), uv_width, uv_height, bUsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);
    tbVsource[0] = FilteredResize::CreateResize(new SelectEvery(Vsource, 2, 0, env), uv_width, uv_height, tVsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);
    tbVsource[1] = FilteredResize::CreateResize(new SelectEvery(Vsource, 2, 1, env), uv_width, uv_height, bVsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);

    Usource = new SelectEvery(new DoubleWeaveFields(new Interleave(std::move(tbUsource), env)), 2, 0, env);
    Vsource = new SelectEvery(new DoubleWeaveFields(new Interleave(std::move(tbVsource), env)), 2, 0, env);
  }
  else {
    AVSValue UsubSampling[4] = { ChrOffset(xsIn, xdInU, xsOut, xdOutU), ChrOffset(ysIn, ydInU, ysOut, ydOutU), AVSValue(), AVSValue() };
    AVSValue VsubSampling[4] = { ChrOffset(xsIn, xdInV, xsOut, xdOutV), ChrOffset(ysIn, ydInV, ysOut, ydOutV), AVSValue(), AVSValue() };

    Usource = FilteredResize::CreateResize(new SwapUVToY(child, SwapUVToY::UToY8, env), uv_width, uv_height, UsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);
    Vsource = FilteredResize::CreateResize(new SwapUVToY(child, SwapUVToY::VToY8, env), uv_width, uv_height, VsubSampling, force, filter, preserve_center, placement_name_notused, forced_chroma_placement, env);
  }
  delete filter;
}

PVideoFrame __stdcall ConvertToPlanarGeneric::GetFrame(int n, IScriptEnvironment* env) {
  PVideoFrame src = child->GetFrame(n, env);


  if (vi.IsY()) {
    // Abuse Subframe to snatch the Y plane
    PVideoFrame dst = env->Subframe(src, 0, src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y), src->GetHeight(PLANAR_Y));
    auto props = env->getFramePropsRW(dst);
    // ChromaLocation_Out is <0 : erased
    update_ChromaLocation(props, ChromaLocation_Out, env);
    return dst;
  }

  PVideoFrame dst = env->NewVideoFrameP(vi, &src);
  env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y),
    src->GetRowSize(PLANAR_Y_ALIGNED), src->GetHeight(PLANAR_Y));

  auto props = env->getFramePropsRW(dst);
  update_ChromaLocation(props, ChromaLocation_Out, env);

  // alpha. if pitch is zero -> no alpha channel
  const int rowsizeA = dst->GetRowSize(PLANAR_A);
  const int dst_pitchA = dst->GetPitch(PLANAR_A);
  BYTE* dstp_a = (dst_pitchA == 0) ? nullptr : dst->GetWritePtr(PLANAR_A);
  const int heightA = dst->GetHeight(PLANAR_A);

  if (dst_pitchA != 0)
  {
    if (src->GetPitch(PLANAR_A) != 0)
      env->BitBlt(dstp_a, dst_pitchA, src->GetReadPtr(PLANAR_A), src->GetPitch(PLANAR_A),
        src->GetRowSize(PLANAR_A_ALIGNED), src->GetHeight(PLANAR_A));
    else {
      // e.g. ConvertToYUVA() case from Alpha-less formats
      switch (vi.ComponentSize())
      {
      case 1:
        fill_plane<BYTE>(dstp_a, heightA, rowsizeA, dst_pitchA, 255);
        break;
      case 2:
        fill_plane<uint16_t>(dstp_a, heightA, rowsizeA, dst_pitchA, (1 << vi.BitsPerComponent()) - 1);
        break;
      case 4:
        fill_plane<float>(dstp_a, heightA, rowsizeA, dst_pitchA, 1.0f);
        break;
      }
    }
  }

  BYTE* dstp_u = dst->GetWritePtr(PLANAR_U);
  BYTE* dstp_v = dst->GetWritePtr(PLANAR_V);
  const int height = dst->GetHeight(PLANAR_U);
  const int rowsizeUV = dst->GetRowSize(PLANAR_U);
  const int dst_pitch = dst->GetPitch(PLANAR_U);

  if (Yinput) {
    switch (vi.ComponentSize())
    {
      case 1:
        fill_chroma<BYTE>(dstp_u, dstp_v, height, rowsizeUV, dst_pitch, 0x80);
        break;
      case 2:
        fill_chroma<uint16_t>(dstp_u, dstp_v, height, rowsizeUV, dst_pitch, 1 << (vi.BitsPerComponent() - 1));
        break;
      case 4:
        const float half = 0.0f;
        fill_chroma<float>(dstp_u, dstp_v, height, rowsizeUV, dst_pitch, half);
        break;
    }
  } else {
    src = Usource->GetFrame(n, env);
    env->BitBlt(dstp_u, dst_pitch, src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y_ALIGNED), height);
    src = Vsource->GetFrame(n, env);
    env->BitBlt(dstp_v, dst_pitch, src->GetReadPtr(PLANAR_Y), src->GetPitch(PLANAR_Y), src->GetRowSize(PLANAR_Y_ALIGNED), height);
  }

  return dst;
}
/*                                           0      1         2             3                  4            5       6         7       8       9
{ "ConvertToYUV411",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYV411, (void*)1 }, // alias for ConvertToYV411, 8 bit check later
{ "ConvertToYUV444",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV444, (void*)1 },
{ "ConvertToYUVA444", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV444, (void*)2 },
                                             0      1         2             3                   4            5                  6         7          8     9     10
{ "ConvertToYUV420",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV420, (void*)1 },
{ "ConvertToYUV422",  BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV422, (void*)1 },
{ "ConvertToYUVA420", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV420, (void*)2 },
{ "ConvertToYUVA422", BUILTIN_FUNC_PREFIX, "c[interlaced]b[matrix]s[ChromaInPlacement]s[chromaresample]s[ChromaOutPlacement]s[param1]f[param2]f[param3]f[bits]i[quality]b", ConvertToPlanarGeneric::CreateYUV422, (void*)2 },
                                            0   1        2       3 
{ "ConvertToY",       BUILTIN_FUNC_PREFIX, "c[matrix]s[bits]i[quality]b", ConvertToPlanarGeneric::CreateConvertToY, (void*)1 }, // user_data == 1 -> any bit depth sources

*/
AVSValue ConvertToPlanarGeneric::Create(AVSValue& args, const char* filter, bool strip_alpha_legacy_8bit, bool to_yuva, IScriptEnvironment* env) {
  bool converted = false;

  PClip clip = args[0].AsClip();
  VideoInfo vi = clip->GetVideoInfo();

  const bool to_y = strcmp(filter, "ConvertToY") == 0;
  const bool to_420 = strcmp(filter, "ConvertToYUV420") == 0;
  const bool to_422 = strcmp(filter, "ConvertToYUV422") == 0;
  const bool to_411 = strcmp(filter, "ConvertToYV411") == 0;
  const bool to_444 = strcmp(filter, "ConvertToYUV444") == 0;

  if (vi.IsYUY2()) {
    // use to_y: no need YV16 extra planes when our target is Y only
    clip = new ConvertYUY2ToYV16_or_Y(clip, to_y, env);
    vi = clip->GetVideoInfo();
  }

  AVSValue bits_arg;
  AVSValue quality_arg;
  AVSValue matrix_arg;
  if (to_y) {
    matrix_arg = args[1];
    bits_arg = args[2];
    quality_arg = args[3];
  }
  else if (to_420 || to_422) {
    matrix_arg = args[2];
    bits_arg = args[9];
    quality_arg = args[10];
  }
  else {
    matrix_arg = args[2];
    bits_arg = args[8];
    quality_arg = args[9];
  }

  const int target_bits_per_pixel = bits_arg.AsInt(vi.BitsPerComponent());
  const bool quality = quality_arg.AsBool(false); // float RGB YUV inside

  int bits_per_pixel = vi.BitsPerComponent();

  /*
  if (target_bits_per_pixel != bits_per_pixel && !vi.IsRGB()) {
    env->ThrowError("%s: on-thy-fly bit-depth conversion only supported from RGB formats.", filter);
  }
  */
  bool needConvertFinalBitdepth = false;

  if (target_bits_per_pixel != vi.BitsPerComponent()) {
    needConvertFinalBitdepth = true;
  }

  if (vi.IsRGB()) { // packed or planar source
    const bool keep_packedrgb_alpha = to_yuva && (vi.IsRGB32() || vi.IsRGB64());
    // 3.7.6: We convert ALL packed RGB formats to planar RGB!
    // The only case where a RGB32->YV24 is a bit slower is: SSE2-only but no SSSE3 capable CPU.
    // All other cases are quicker, when converting to planar first and then to YUV444, than doing packed RGB->YUV444 in one step.
    // On AVX2 capable CPUs using the PlanarRGB-YUV engine, we are +80% quicker
    // doing RGB32->PlanarRGB->YUV444 than doing RGB32->YUV444 in the old SSSE3 step.
    if (!vi.IsPlanar()) {
      // we convert to intermediate PlanarRGB, RGB48/64->YUV444 is slow C, planarRGB  is fast
      // Default ConvertToPlanarRGB call
      AVSValue new_args[10] = { clip, AVSValue(), AVSValue(), AVSValue(), AVSValue(), AVSValue(), AVSValue(), AVSValue(), AVSValue(), AVSValue() };
      // clip, matrix,  interlaced, chromainplacement, chromaresample, param1, param2, param3, bits, quality
      // convert to planar RGBA only if RGB64 and target is YUVA (need to keep alpha)
      const intptr_t planar_rgb_type = keep_packedrgb_alpha ? -2 : -1;
      clip = CreateConvertToRGB(AVSValue(new_args, 10), (void *)planar_rgb_type, env).AsClip();
      vi = clip->GetVideoInfo();
    }

    if (target_bits_per_pixel != vi.BitsPerComponent()) {
      needConvertFinalBitdepth = true;
    }

    bool bitdepthConverted = false;
    if (needConvertFinalBitdepth) {
      // Optional bit-depth conversion in PackedRGBtoPlanarRGB.
      // int->float full/narrow range, int-int full/narrow supported
      clip = new ConvertRGBToYUV444(clip, matrix_arg.AsString(0) /* matrix_name */, keep_packedrgb_alpha, target_bits_per_pixel, quality, /*ref*/bitdepthConverted, to_y, env);
      vi = clip->GetVideoInfo();
      if (bitdepthConverted) {
        needConvertFinalBitdepth = false; // done in-process
      }
    } else {
      // pass -1 as finalBitdepth, signing that no bit-depth conversion required
      // output bitdepthConverted is n/a
      clip = new ConvertRGBToYUV444(clip, matrix_arg.AsString(0) /* matrix_name */, keep_packedrgb_alpha, -1 /*no bit-depth conversion*/, quality, /*ref*/bitdepthConverted, to_y, env);
      vi = clip->GetVideoInfo();
    }

    vi = clip->GetVideoInfo();
    converted = true;
  }
  else if (!vi.IsPlanar())
    env->ThrowError("%s: Can only convert from Planar YUV.", filter);

  // YV411 has no 8+ bits variant.
  // A bit depth changing YV411->YUV4xx conversion must be reversed: first convert th format keeping 8 bits,
  // then do the bit-depth conversion.
  if (needConvertFinalBitdepth && !vi.IsYV411()) {
    // plain Invoke and no "new ConvertBits()": this detects and keeps source and target ranges
    AVSValue new_args[2] = { clip, target_bits_per_pixel };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
    vi = clip->GetVideoInfo();
    needConvertFinalBitdepth = false;
  }

  bits_per_pixel = vi.BitsPerComponent(); // rgb conversion or standalone ConvertBits() would alter it.

  int pixel_type = VideoInfo::CS_UNKNOWN;
  AVSValue outplacement = AVSValue(); // only for ConvertToYUV420 and ConvertToYUV422

  bool hasAlpha = vi.NumComponents() == 4 && !strip_alpha_legacy_8bit;
  bool shouldStripAlpha = vi.NumComponents() == 4 && strip_alpha_legacy_8bit;
  bool shouldAddAlpha = vi.NumComponents() != 4 && to_yuva;
  bool targethasAlpha = hasAlpha || shouldAddAlpha;

  int ChromaLocation_In = -1; // invalid. Chromalocation_e::AVS_CHROMALOCATION_UNUSED
  int ChromaLocation_Out = -1; // left as invalid for 444, and Y

  if (vi.IsYV411()) {
    // ChromaInPlacement parameter exists, (default none/-1) + input frame properties; 'left'-ish _ChromaLocation is allowed, checked later
    auto frame0 = clip->GetFrame(0, env);
    const AVSMap* props = env->getFramePropsRO(frame0);
    chromaloc_parse_merge_with_props(vi, args[3].AsString(nullptr), props, /* ref*/ChromaLocation_In, -1 /*default none chromaloc */, env);
  }
  else if (vi.Is420() || vi.Is422()) {
    // ChromaInPlacement parameter is valid + input frame properties
    auto frame0 = clip->GetFrame(0, env);
    const AVSMap* props = env->getFramePropsRO(frame0);
    chromaloc_parse_merge_with_props(vi, args[3].AsString(nullptr), props, /* ref*/ChromaLocation_In, ChromaLocation_e::AVS_CHROMA_LEFT /*default*/, env);
  }

  AVSValue param1;
  AVSValue param2;
  AVSValue param3;
  if (to_420 || to_422) {
    // ChromaOutPlacement parameter is valid
    chromaloc_parse_merge_with_props(vi, args[5].AsString(nullptr), nullptr, /* ref*/ChromaLocation_Out, ChromaLocation_e::AVS_CHROMA_LEFT /*default*/, env);
    param1 = args[6];
    param2 = args[7];
    param3 = args[8];
  }
  else if (to_444 || to_411) {
    // No ChromaOutPlacement parameter, indexes skip back accordingly
    param1 = args[5];
    param2 = args[6];
    param3 = args[7];
  }

  // FIXME: no-op or almost no-op shortcuts won't exist even at 420-420 conversion if YUV-YUV matrix conversion is added

  if (to_y) {
    if (vi.IsY()) {
      // After RGB->Y conversion or input as Y
      return clip;
    }
    // planar YUV original, GetFrame will return Y
    switch (bits_per_pixel)
    {
    case 8: pixel_type =  VideoInfo::CS_Y8; break;
    case 10: pixel_type = VideoInfo::CS_Y10; break;
    case 12: pixel_type = VideoInfo::CS_Y12; break;
    case 14: pixel_type = VideoInfo::CS_Y14; break;
    case 16: pixel_type = VideoInfo::CS_Y16; break;
    case 32: pixel_type = VideoInfo::CS_Y32; break;
    }
  }
  else if (to_420) {
    if (vi.Is420()) {
      // possible shortcut
      if (ChromaLocation_In == ChromaLocation_Out)
      {
        if (shouldStripAlpha)
          return new RemoveAlphaPlane(clip, env);
        if (shouldAddAlpha) {
          // create with default alpha
          clip = new AddAlphaPlane(clip, nullptr, 0.0f, false, env);
          vi = clip->GetVideoInfo();
        }
        return clip;
      }
    }

    outplacement = args[5];
    switch (vi.BitsPerComponent())
    {
    case 8 : pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420 : VideoInfo::CS_YV12; break;
    case 10: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420P10 : VideoInfo::CS_YUV420P10; break;
    case 12: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420P12 : VideoInfo::CS_YUV420P12; break;
    case 14: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420P14 : VideoInfo::CS_YUV420P14; break;
    case 16: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420P16 : VideoInfo::CS_YUV420P16; break;
    case 32: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA420PS  : VideoInfo::CS_YUV420PS; break;
    }
  }
  else if (to_422) {
    if (vi.Is422()) {
      // possible shortcut
      if (ChromaLocation_In == ChromaLocation_Out)
      {
        if (shouldStripAlpha)
          return new RemoveAlphaPlane(clip, env);
        if (shouldAddAlpha) {
          // create with default alpha
          clip = new AddAlphaPlane(clip, nullptr, 0.0f, false, env);
          vi = clip->GetVideoInfo();
        }
        return clip;
      }
    }

    outplacement = args[5];
    switch (bits_per_pixel)
    {
    case 8 : pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422 : VideoInfo::CS_YV16; break;
    case 10: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422P10 : VideoInfo::CS_YUV422P10; break;
    case 12: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422P12 : VideoInfo::CS_YUV422P12; break;
    case 14: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422P14 : VideoInfo::CS_YUV422P14; break;
    case 16: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422P16 : VideoInfo::CS_YUV422P16; break;
    case 32: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA422PS  : VideoInfo::CS_YUV422PS; break;
    }
  }
  else if (to_444) {
    if (vi.Is444()) {
      if (shouldStripAlpha)
        return new RemoveAlphaPlane(clip, env);
      if (shouldAddAlpha) {
        // create with default alpha
        clip = new AddAlphaPlane(clip, nullptr, 0.0f, false, env);
        vi = clip->GetVideoInfo();
      }
      return clip;
    }

    switch (bits_per_pixel)
    {
    case 8 : pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444 : VideoInfo::CS_YV24; break;
    case 10: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444P10 : VideoInfo::CS_YUV444P10; break;
    case 12: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444P12 : VideoInfo::CS_YUV444P12; break;
    case 14: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444P14 : VideoInfo::CS_YUV444P14; break;
    case 16: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444P16 : VideoInfo::CS_YUV444P16; break;
    case 32: pixel_type = targethasAlpha ? VideoInfo::CS_YUVA444PS  : VideoInfo::CS_YUV444PS; break;
    }
  }
  else if (to_411) {
    if (target_bits_per_pixel != 8)
      env->ThrowError("%s: only bits=8 supported for YV411", filter);
    if (vi.IsYV411()) return clip;
    if(vi.ComponentSize()!=1)
      env->ThrowError("%s: 8 bit input only", filter);

    pixel_type = VideoInfo::CS_YV411;
  }
  else env->ThrowError("Convert: unknown filter '%s'.", filter);

  if (pixel_type == VideoInfo::CS_UNKNOWN)
    env->ThrowError("%s: unsupported bit depth", filter);

  if (converted)
    clip = env->Invoke("Cache", AVSValue(clip)).AsClip();

  // ConvertToPlanarGeneric's GetFrame will recognize if alpha copy or fill-with-defaults needed
  AVSValue dummy_for_to_y;

  clip = new ConvertToPlanarGeneric(clip, pixel_type,
    to_y ? false : args[1].AsBool(false), // interlaced
    ChromaLocation_In, 
    to_y ? dummy_for_to_y : args[4], // chromaresample
    param1, param2, param3,
    ChromaLocation_Out,
    env);

  // If still not converted, do it now (YV411->YUVxxx 8+ bits)
  if (needConvertFinalBitdepth) {
    // plain Invoke and no "new ConvertBits()": this detects and keeps source and target ranges
    AVSValue new_args[2] = { clip, target_bits_per_pixel };
    clip = env->Invoke("ConvertBits", AVSValue(new_args, 2)).AsClip();
    vi = clip->GetVideoInfo();
    needConvertFinalBitdepth = false;
  }

  return clip;

}

AVSValue __cdecl ConvertToPlanarGeneric::CreateYUV420(AVSValue args, void* user_data, IScriptEnvironment* env) {
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  bool to_yuva = reinterpret_cast<intptr_t>(user_data) == 2;
  if (only_8bit) {
    // "ConvertToYV12" syntax
    if (args[9].Defined() && args[9].AsInt() != 8)
      env->ThrowError("ConvertToYV12: only 8 bit output supported. "
        "Use ConvertToYUV420 for higher bit depth.");
    AVSValue new_args[11] = {
      args[0], args[1], args[2], args[3], args[4],
      args[5], args[6], args[7], args[8],
      AVSValue(8), // bits forced to 8
      args[10]
    };
    AVSValue new_args_val(new_args, 11); // named, lives until end of scope
    return Create(new_args_val, "ConvertToYUV420", only_8bit, to_yuva, env);
  }
  return Create(args, "ConvertToYUV420", only_8bit, to_yuva, env);
}

AVSValue __cdecl ConvertToPlanarGeneric::CreateYUV422(AVSValue args, void* user_data, IScriptEnvironment* env) {
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  bool to_yuva = reinterpret_cast<intptr_t>(user_data) == 2;
  if (only_8bit) {
    // "ConvertToYV16" syntax
    if (args[9].Defined() && args[9].AsInt() != 8)
      env->ThrowError("ConvertToYV16: only 8 bit output supported. "
        "Use ConvertToYUV422 for higher bit depth.");
    AVSValue new_args[11] = {
      args[0], args[1], args[2], args[3], args[4],
      args[5], args[6], args[7], args[8],
      AVSValue(8),
      args[10]
    };
    AVSValue new_args_val(new_args, 11);
    return Create(new_args_val, "ConvertToYUV422", only_8bit, to_yuva, env);
  }
  return Create(args, "ConvertToYUV422", only_8bit, to_yuva, env);
}

AVSValue __cdecl ConvertToPlanarGeneric::CreateYUV444(AVSValue args, void* user_data, IScriptEnvironment* env) {
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  bool to_yuva = reinterpret_cast<intptr_t>(user_data) == 2;
  if (only_8bit) {
    // "ConvertToYV24" syntax
    // parameter index skip back by 1 compared to 420/422, no ChromaOutPlacement parameter, bits is at index 8 instead of 9
    if (args[8].Defined() && args[8].AsInt() != 8)
      env->ThrowError("ConvertToYV24: only 8 bit output supported. "
        "Use ConvertToYUV444 for higher bit depth.");
    AVSValue new_args[10] = {
      args[0], args[1], args[2], args[3], args[4],
      args[5], args[6], args[7],
      AVSValue(8),
      args[9]
    };
    AVSValue new_args_val(new_args, 10);
    return Create(new_args_val, "ConvertToYUV444", only_8bit, to_yuva, env);
  }
  return Create(args, "ConvertToYUV444", only_8bit, to_yuva, env);
}

AVSValue __cdecl ConvertToPlanarGeneric::CreateYV411(AVSValue args, void* user_data, IScriptEnvironment* env) {
  bool only_8bit = reinterpret_cast<intptr_t>(user_data) == 0;
  bool to_yuva = reinterpret_cast<intptr_t>(user_data) == 2;
  // parameter index skip back by 1 compared to 420/422, no ChromaOutPlacement parameter, bits is at index 8 instead of 9
  if (args[8].Defined() && args[8].AsInt() != 8)
    env->ThrowError("ConvertToYV411: only 8 bit output supported, "
      "no high bit depth YUV411 format exists.");
  AVSValue new_args[10] = {
    args[0], args[1], args[2], args[3], args[4],
    args[5], args[6], args[7],
    AVSValue(8),
    args[9]
  };
  AVSValue new_args_val(new_args, 10);
  return Create(new_args_val, "ConvertToYV411", only_8bit, to_yuva, env);
}

AVSValue __cdecl ConvertToPlanarGeneric::CreateConvertToYUY2(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip clip = args[0].AsClip();
  VideoInfo vi = clip->GetVideoInfo();

  // Fast no-op: YUY2 source, no parameters at all.
  // Do not touch frame properties - caller's responsibility.
  if (vi.IsYUY2()
    && !args[1].Defined()   // interlaced
    && !args[2].Defined()   // matrix
    && !args[3].Defined()   // ChromaInPlacement
    && !args[4].Defined()   // chromaresample
    && !args[5].Defined()   // ChromaOutPlacement
    && !args[6].Defined()   // param1
    && !args[7].Defined()   // param2
    && !args[8].Defined()   // param3
    && !args[9].Defined()   // bits
    && !args[10].Defined()) // quality
    return clip;

  // YUY2 is 8-bit only: bits= must be 8 if specified.
  // We check here to give a proper ConvertToYUY2-specific error message,
  // rather than letting CreateYUV422 complain about something unrelated.
  // Source can be any bit depth - the conversion chain will downconvert.
  if (args[9].Defined() && args[9].AsInt() != 8)
    env->ThrowError("ConvertToYUY2: bits must be 8, YUY2 is an 8-bit format");

  // Rebuild args with bits=8 forced, so CreateYUV422 sees an explicit
  // 8-bit target regardless of source bit depth.
  // This also means the void*1 path in CreateYUV422 won't reject
  // high bit depth sources - the bits=8 in args drives the downconvert.
  AVSValue new_args[11] = {
    args[0],     // clip
    args[1],     // interlaced
    args[2],     // matrix
    args[3],     // ChromaInPlacement
    args[4],     // chromaresample
    args[5],     // ChromaOutPlacement
    args[6],     // param1
    args[7],     // param2
    args[8],     // param3
    AVSValue(8), // bits: always 8 for YUY2, overrides source depth
    args[10]     // quality
  };

  // (void*)1: allow any source bit depth, bits=8 in args enforces 8-bit output.
  // The legacy (void*)0 source-depth check is not appropriate here since
  // we explicitly want to allow e.g. 10-bit YUV420 → 8-bit YUY2.
  PClip yv16 = CreateYUV422(AVSValue(new_args, 11), (void*)1, env).AsClip();

  // Lossless repack YV16->YUY2.
  // Frame properties preserved via NewVideoFrameP in GetFrame.
  return new ConvertYV16ToYUY2(yv16, env);
}

AVSValue __cdecl ConvertToPlanarGeneric::CreateConvertBackToYUY2(AVSValue args, void*, IScriptEnvironment* env)
{
  // ConvertBackToYUY2 was a pre-2.5 optimization for RGB roundtrip workflows.
  // It was never interlaced-aware (hardcoded interlaced=false).
  // Now obsolete - forward to ConvertToYUY2::Create with full parameter array.
  // ConvertBackToYUY2 signature: c[matrix]s
  // ConvertToYUY2 signature:     c[interlaced]b[matrix]s[ChromaInPlacement]s
  //                               [chromaresample]s[ChromaOutPlacement]s
  //                               [param1]f[param2]f[param3]f[bits]i[quality]b
  AVSValue new_args[11] = {
    args[0],        // clip
    AVSValue(false),// interlaced: was always false in ConvertBackToYUY2
    args[1],        // matrix: pass through
    AVSValue(),     // ChromaInPlacement: not defined
    AVSValue(),     // chromaresample: not defined
    AVSValue(),     // ChromaOutPlacement: not defined
    AVSValue(),     // param1: not defined
    AVSValue(),     // param2: not defined
    AVSValue(),     // param3: not defined
    AVSValue(),     // bits: not defined (will be forced to 8 inside)
    AVSValue()      // quality: not defined
  };
  return CreateConvertToYUY2(AVSValue(new_args, 11), nullptr, env);
}


/*
static int getPlacement(const AVSValue& _placement, IScriptEnvironment* env) {
  const char* placement = _placement.AsString(0);

  if (placement) {
    if (!lstrcmpi(placement, "mpeg2") || !lstrcmpi(placement, "left"))
      return PLACEMENT_MPEG2;

    if (!lstrcmpi(placement, "mpeg1") || !lstrcmpi(placement, "jpeg") || !lstrcmpi(placement, "center"))
      return PLACEMENT_MPEG1;

    if (!lstrcmpi(placement, "dv"))
      return PLACEMENT_DV;

    if (!lstrcmpi(placement, "top_left"))
      return PLACEMENT_TOP_LEFT;

    env->ThrowError("Convert: Unknown chromaplacement");
  }
  return PLACEMENT_MPEG2;
}
*/


ResamplingFunction* getResampler(const char* resampler, AVSValue param1, AVSValue param2, AVSValue param3, bool throw_on_error, IScriptEnvironment* env) {
  if (resampler) {
    if (!lstrcmpi(resampler, "point"))
      return new PointFilter();
    else if (!lstrcmpi(resampler, "bilinear"))
      return new TriangleFilter();
    else if (!lstrcmpi(resampler, "bicubic"))
      return new MitchellNetravaliFilter(param1.AsDblDef(1.0/3), param2.AsDblDef(1.0/3)); // optional B and C as param1 and param2
    else if (!lstrcmpi(resampler, "lanczos"))
      return new LanczosFilter((int)param1.AsFloat(3)); // optional Taps as param1
    else if (!lstrcmpi(resampler, "lanczos4"))
      return new LanczosFilter(4);
    else if (!lstrcmpi(resampler, "blackman"))
      return new BlackmanFilter((int)param1.AsFloat(4)); // optional Taps as param1
    else if (!lstrcmpi(resampler, "spline16"))
      return new Spline16Filter();
    else if (!lstrcmpi(resampler, "spline36"))
      return new Spline36Filter();
    else if (!lstrcmpi(resampler, "spline64"))
      return new Spline64Filter();
    else if (!lstrcmpi(resampler, "gauss"))
      return new GaussianFilter(param1.AsDblDef(30.0), param2.AsDblDef(2.0), param3.AsDblDef(4.0)); // optional P, B, S as param1, param2, param3
    else if (!lstrcmpi(resampler, "sinc"))
      return new SincFilter((int)param1.AsFloat(4)); // optional Taps as param1
    else if (!lstrcmpi(resampler, "sinpow"))
      return new SinPowerFilter(param1.AsDblDef(2.5)); // optional P as param1
    else if (!lstrcmpi(resampler, "sinclin2"))
      return new SincLin2Filter((int)param1.AsFloat(15)); // optional Taps= as param1
    else if (!lstrcmpi(resampler, "userdefined2"))
      return new UserDefined2Filter(param1.AsDblDef(121.0), param2.AsDblDef(19.0), param3.AsDblDef(2.3)); // optional B and C and S as param1, param2, param3
    else
      if (throw_on_error)
        env->ThrowError("Convert: Unknown chroma resampler, '%s'", resampler);
      else
        return nullptr; // e.g. from AddBorders
  }
  return new MitchellNetravaliFilter(param1.AsDblDef(1.0/3), param2.AsDblDef(1.0/3)); // Default colorspace conversion for AviSynth
}

DISABLE_WARNING_POP
