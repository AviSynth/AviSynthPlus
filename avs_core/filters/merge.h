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


// Avisynth filter: YUV merge
// by Klaus Post
// adapted by Richard Berg (avisynth-dev@richardberg.net)


#ifndef __Merge_H__
#define __Merge_H__

#include <avisynth.h>


/****************************************************
****************************************************/

class MergeChroma : public GenericVideoFilter
/**
  * Merge the chroma planes of one clip into another, preserving luma
 **/
{
public:
  MergeChroma(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);

private:
  PClip clip;
  float weight;
  int pixelsize;
  int bits_per_pixel;
};


class MergeLuma : public GenericVideoFilter
/**
  * Merge the luma plane of one clip into another, preserving chroma
 **/
{
public:
  MergeLuma(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);

private:
  PClip clip;
  float weight;
  int pixelsize;
  int bits_per_pixel;
};


class MergeAll : public GenericVideoFilter
/**
  * Merge the planes of one clip into another
 **/
{
public:
  MergeAll(PClip _child, PClip _clip, float _weight, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env);

private:
  PClip clip;
  float weight;
  int pixelsize;
  int bits_per_pixel;
};

typedef void(*MergeFuncPtr) (BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);
#ifdef INTEL_INTRINSICS
MergeFuncPtr getMergeFunc(int bits_per_pixel, int cpuFlags, BYTE *srcp, const BYTE *otherp, float weight_f, int &weight_i, int &invweight_i);
#else
MergeFuncPtr getMergeFunc(int bits_per_pixel, BYTE *srcp, const BYTE *otherp, float weight_f, int &weight_i, int &invweight_i);
#endif

template<bool lessthan16bit>
void weighted_merge_planar_uint16_sse2(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);

template<typename pixel_t>
void weighted_merge_planar_c(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);
void weighted_merge_planar_c_float(BYTE *p1, const BYTE *p2, int p1_pitch, int p2_pitch, int rowsize, int height, float weight_f, int weight_i, int invweight_i);

#endif  // __Merge_H__
