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

#ifndef __Focus_H__
#define __Focus_H__

#include <avisynth.h>

template<bool packedRGB3264>
int calculate_sad_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);
template<typename pixel_t, bool packedRGB3264>
int64_t calculate_sad_8_or_16_sse2(const BYTE* cur_ptr, const BYTE* other_ptr, int cur_pitch, int other_pitch, size_t rowsize, size_t height);

class AdjustFocusV : public GenericVideoFilter
/**
  * Class to adjust focus in the vertical direction, helper for sharpen/blue
 **/
{
public:
  AdjustFocusV(double _amount, PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  const double amountd;
  int half_amount;
};


class AdjustFocusH : public GenericVideoFilter
/**
  * Class to adjust focus in the horizontal direction, helper for sharpen/blue
 **/
{
public:
  AdjustFocusH(double _amount, PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  const double amountd;
  int half_amount;
};

AVSValue __cdecl Create_Sharpen(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_Blur(AVSValue args, void*, IScriptEnvironment* env);


/*** Soften classes ***/

class TemporalSoften : public GenericVideoFilter
/**
  * Class to soften the focus on the temporal axis
 **/
{
public:
  TemporalSoften( PClip _child, unsigned radius, unsigned luma_thresh, unsigned chroma_thresh,int _scenechange, IScriptEnvironment* env );
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
    typedef struct {
      int planeId;
      int threshold;
    } planeInfo;
    planeInfo planes[4];
    int scenechange;
    int pixelsize;
    int bits_per_pixel;

// YUY2:
  const unsigned luma_threshold, chroma_threshold;
  const int kernel;

  enum { MAX_RADIUS=7 };
};


class SpatialSoften : public GenericVideoFilter
/**
  * Class to soften the focus on the spatial axis
 **/
{
public:
  SpatialSoften(PClip _child, int _radius, unsigned _luma_threshold, unsigned _chroma_threshold, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  const unsigned luma_threshold, chroma_threshold;
  const int diameter;

};

#endif  // __Focus_H__
