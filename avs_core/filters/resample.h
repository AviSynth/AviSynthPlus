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

#ifndef __Resample_H__
#define __Resample_H__

#include <avisynth.h>
#include "resample_functions.h"

void resize_prepare_coeffs(ResamplingProgram* p, IScriptEnvironment* env, int alignFilterSize8or16);

// Resizer function pointer
typedef void (*ResamplerV)(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
typedef void (*ResamplerH)(BYTE* dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// Turn function pointer -- copied from turn.h
typedef void (*TurnFuncPtr) (const BYTE* srcp, BYTE* dstp, int width, int height, int src_pitch, int dst_pitch);

/**
  * Class to resize in the horizontal direction using a specified sampling filter
  * Helper for resample functions
 **/
class FilteredResizeH : public GenericVideoFilter
{
public:
  FilteredResizeH(PClip _child, double subrange_left, double subrange_width, int target_width, ResamplingFunction* func, 
    bool preserve_center, int chroma_placement, IScriptEnvironment* env);
  virtual ~FilteredResizeH(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    if (cachehints == CACHE_GET_MTMODE) return MT_NICE_FILTER;
    if (cachehints == CACHE_INFORM_NUM_THREADS) {
      num_threads = frame_range;
    }
    return 0;
  }

  static ResamplerH GetResampler(int CPU, int pixelsize, int bits_per_pixel, ResamplingProgram* program, ResamplerH& resampler_h_alternative_for_mt, IScriptEnvironment* env);

private:
  // Resampling
  ResamplingProgram* resampling_program_luma;
  ResamplingProgram* resampling_program_chroma;

  int temp_1_pitch, temp_2_pitch;

  int src_width, src_height, dst_width, dst_height;
  bool grey;
  int pixelsize; // AVS16
  int bits_per_pixel;

  ResamplerH resampler_h_luma;
  ResamplerH resampler_h_chroma;
  // if the base version performs poorly, we decide to use MT version
  // Unfortunately the information about number of threads is not available at construction time
  // we get num_threads later via SetCacheHints callback with constants CACHE_INFORM_NUM_THREADS.
  ResamplerH resampler_h_luma_mt;
  ResamplerH resampler_h_chroma_mt;
  // when false, _mt fallback pointers are cleared and the primary resampler is always used.
  bool use_alternative_h_avx512_mt_fallback;
  bool fast_resize;

  ResamplerV resampler_luma;
  ResamplerV resampler_chroma;

  TurnFuncPtr turn_left, turn_right;

  int num_threads; // set by MTGuard by calling SetCacheHints(CACHE_INFORM_NUM_THREADS, n)
};


/**
  * Class to resize in the vertical direction using a specified sampling filter
  * Helper for resample functions
 **/
class FilteredResizeV : public GenericVideoFilter
{
public:
  FilteredResizeV(PClip _child, double subrange_top, double subrange_height, int target_height, ResamplingFunction* func, 
    bool preserve_center, int chroma_placement, IScriptEnvironment* env);
  virtual ~FilteredResizeV(void);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static ResamplerV GetResampler(int CPU, int pixelsize, int bits_per_pixel, ResamplingProgram* program, IScriptEnvironment* env);

private:
  bool grey;
  int pixelsize; // AVS16
  int bits_per_pixel;

  ResamplingProgram* resampling_program_luma;
  ResamplingProgram* resampling_program_chroma;
  ResamplingProgram* resampling_program_chroma_planeV; // nullptr unless V is sited differently than U ("dv")

  ResamplerV resampler_luma;
  ResamplerV resampler_chroma;
};


/*** Resample factory methods ***/

class FilteredResize
  /**
    * Helper for resample functions
   **/
{
public:
  static PClip CreateResizeH(PClip clip, double subrange_left, double subrange_width, int target_width, bool force,
    ResamplingFunction* func, 
    bool preserve_center, int chroma_placement,
    IScriptEnvironment* env);

  static PClip CreateResizeV(PClip clip, double subrange_top, double subrange_height, int target_height, bool force,
    ResamplingFunction* func, 
    bool preserve_center, int chroma_placement,
    IScriptEnvironment* env);

  static PClip CreateResize(PClip clip, int target_width, int target_height, const AVSValue* args, int force,
    ResamplingFunction* f, 
    bool preserve_center, const char *placement_name, int forced_chroma_placement,
    IScriptEnvironment* env);

  static AVSValue __cdecl Create_PointResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_BilinearResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_BicubicResize(AVSValue args, void*, IScriptEnvironment* env);

  // 09-14-2002 - Vlad59 - Lanczos3Resize -
  static AVSValue __cdecl Create_LanczosResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_Lanczos4Resize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_BlackmanResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_Spline16Resize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_Spline36Resize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_Spline64Resize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_GaussianResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_SincResize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_SinPowerResize(AVSValue args, void*, IScriptEnvironment* env);
  
  static AVSValue __cdecl Create_SincLin2Resize(AVSValue args, void*, IScriptEnvironment* env);

  static AVSValue __cdecl Create_UserDefined2Resize(AVSValue args, void*, IScriptEnvironment* env);
};



#endif // __Resample_H__
