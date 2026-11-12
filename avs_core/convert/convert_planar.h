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

#ifndef __Convert_PLANAR_H__
#define __Convert_PLANAR_H__

#include <avisynth.h>
#include <stdint.h>
#include "convert.h"
#include "convert_matrix.h"
#include "../filters/resample.h"
#include "convert_helper.h"
#include "convert_bits.h"

enum YuvRgbConversionType {
  NATIVE_INT,           // Same bit depth, integer matrix arithmetic, no conversion
  FORCE_FLOAT,          // Float matrix workflow (required for float input and for float output, optional for accuracy)
  BITCONV_INT_LIMITED,  // Bit-depth change with limited range (shift-based scaling)
  BITCONV_INT_FULL      // Bit-depth change with full range (requires float coefficients)
};

// Dispatcher matching AVX2/SSE2 structure
template<ConversionDirection direction, typename pixel_t_src, bool lessthan16bit>
void convert_yuv_to_planarrgb_c(BYTE* dstp[3], int dstPitch[3], const BYTE* srcp[3], const int srcPitch[3], int width, int height, const ConversionMatrix& m,
  int bits_per_pixel, int bits_per_pixel_target, bool force_float);

// useful functions
template <typename pixel_t>
void fill_chroma(uint8_t * dstp_u, uint8_t * dstp_v, int height, int row_size, int pitch, pixel_t val);

template <typename pixel_t>
void fill_plane(uint8_t * dstp, int height, int row_size, int pitch, pixel_t val);

ResamplingFunction* getResampler(const char* resampler, AVSValue param1, AVSValue param2, AVSValue param3, bool throw_on_error, IScriptEnvironment* env);

class ConvertRGBToYUV444 : public GenericVideoFilter
{
public:
  ConvertRGBToYUV444(PClip src, const char *matrix_name, bool keep_packedrgb_alpha, int _target_bit_depth, bool _quality, bool& bitdepthConverted, bool _to_y, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  int theMatrix;
  int theColorRange;
  int theOutColorRange;
  ConversionMatrix matrix;
  int pixel_step;
  bool targetHasAlpha;
  bool isPlanarRGBfamily;
  int source_bit_depth;
  int target_bit_depth;
  bool quality;
  bool to_y;
  // primarily for alpha plane conversion
  BitDepthConvFuncPtr conv_function;
  BitDepthConvFuncPtr conv_function_chroma; // 32bit float YUV chroma
  BitDepthConvFuncPtr conv_function_a;
};

class ConvertYUY2ToYV16_or_Y : public GenericVideoFilter
{
public:
  ConvertYUY2ToYV16_or_Y(PClip src, bool _to_y, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  bool to_y;
};

class ConvertYUV444ToRGB : public GenericVideoFilter
{
public:
  ConvertYUV444ToRGB(PClip src, const char* matrix_name, int _pixel_step, int _target_bit_depth, bool quality, bool& bitdepthConverted, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  int theMatrix;
  int theColorRange;
  // separate out set for rgb target
  int theOutMatrix;
  int theOutColorRange;
  ConversionMatrix matrix;
  int pixel_step;
  int source_bit_depth;
  int target_bit_depth;
  bool quality; // true: forced float matrix multiplication
  // primarily for alpha plane conversion
  BitDepthConvFuncPtr conv_function;
  BitDepthConvFuncPtr conv_function_chroma; // 32bit float YUV chroma
  BitDepthConvFuncPtr conv_function_a;

};

class ConvertYV16ToYUY2 : public GenericVideoFilter
{
public:
  ConvertYV16ToYUY2(PClip src, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};

class ConvertToPlanarGeneric : public GenericVideoFilter
{
public:
  ConvertToPlanarGeneric(PClip src, int dst_space, bool interlaced,
                         int _ChromaLocation_In, 
                         const AVSValue& ChromaResampler, const AVSValue& param1, const AVSValue& param2, const AVSValue& param3,
                         int _ChromaLocation_Out,
                         IScriptEnvironment* env);
  ~ConvertToPlanarGeneric() {}
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl CreateY(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV411(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV440(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV410(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV420(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV422(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateYUV444(AVSValue args, void* user_data, IScriptEnvironment* env);
  // YUY2 targets: route through YV16, no GetFrame in this class needed.
  // Placed here as the natural hub for all YUV format conversion routing.
  static AVSValue __cdecl CreateConvertToYUY2(AVSValue args, void* user_data, IScriptEnvironment* env);
  static AVSValue __cdecl CreateConvertBackToYUY2(AVSValue args, void* user_data, IScriptEnvironment* env);


private:
  static AVSValue Create(AVSValue& args, const char* filter, bool strip_alpha_legacy_8bit, bool to_yuva, IScriptEnvironment* env);
  bool Yinput;
  int pixelsize;
  int ChromaLocation_In;
  int ChromaLocation_Out; // future _ChromaLocation
  PClip Usource;
  PClip Vsource;
};

class AddAlphaPlane : public GenericVideoFilter
{
public:
  AddAlphaPlane(PClip _child, PClip _maskClip, float _mask_f, bool isMaskDefined, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n,IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
  int mask;
  float mask_f;
  PClip alphaClip;
  int pixelsize;
  int bits_per_pixel;
};

class RemoveAlphaPlane : public GenericVideoFilter
{
public:
  RemoveAlphaPlane(PClip _child, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n,IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
};

#endif
