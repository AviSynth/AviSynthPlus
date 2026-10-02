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

#ifndef __Focus_H__
#define __Focus_H__

#include <avisynth.h>

// Blur/Sharpen kernel, identical in C and all SIMD paths.
//   3-tap kernel [wo, wc, wo] (left/upper, center, right/lower), wo: outer weight, wc: center weight
//   wc = amountd = 2^x (x: Sharpen parameter or -Blur parameter, half_amount = 32768 * amountd)
//   The kernel sums to 1: wc = 1 - 2*wo, wo = (1 - amountd)/2, so only wo is needed ("pivot" form,
//   the center pixel is the pivot, only its difference to the neighbours is weighted):
//     wc*center + wo*(left + right) = center + wo*(left + right - 2*center)
//   wo: 15 fractional bits (Q15: value = integer / 32768): outer_weight_15 = af_outer_weight_q15(half_amount)
//     result = center + (((left + right - 2*center) * outer_weight_15 + 0x4000) >> 15)   (0x4000: rounder; SIMD: pmulhrsw)
// amountd is 1/3..2 (Blur 1.58 .. Sharpen 1.0), so outer_weight_15 is -16384..10923: fits int16, and for
// 16 bit pixels |left + right - 2*center| * |outer_weight_15| + 0x4000 < int32_max (2^31).
static inline int af_outer_weight_q15(int64_t half_amount) { return (int)((32768 - half_amount + 1) >> 1); }
// The same as a linear kernel in the 16 bit fixed point used with ScaledPixelClip/Ex:
// (center * (65536 - 4*outer_weight_15) + (left + right) * 2*outer_weight_15 + 32768) >> 16 == the pivot form above, exactly
static inline int af_center_weight_int(int64_t half_amount) { return 65536 - 4 * af_outer_weight_q15(half_amount); }
static inline int af_outer_weight_int(int64_t half_amount) { return 2 * af_outer_weight_q15(half_amount); }

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
    int plane_count; // number of valid planes[] entries, 0: nothing to process
    int64_t scenechange; // full-frame SAD threshold ~8K needs int64
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
