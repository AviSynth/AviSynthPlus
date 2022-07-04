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




// Avisynth filter: Layer
// by "poptones" (poptones@myrealbox.com)


#ifndef __Layer_H__
#define __Layer_H__

#include <avisynth.h>
#include <stdint.h>


/********************************************************************
********************************************************************/

class Mask : public IClip
  /**
    * Class for overlaying a mask clip on a video clip
   **/
{
public:
  Mask(PClip _child1, PClip _child2, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  inline virtual void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override
  {
    child1->GetAudio(buf, start, count, env);
  }
  inline virtual const VideoInfo& __stdcall GetVideoInfo() override
  {
    return vi;
  }
  inline virtual bool __stdcall GetParity(int n) override
  {
    return child1->GetParity(n);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  const PClip child1, child2;
  VideoInfo vi;
  int mask_frames;
  int pixelsize;
  int bits_per_pixel;

};



class ColorKeyMask : public GenericVideoFilter
  /**
    * Class for setting a mask on a video clip based on a color key
  **/
{
public:
  ColorKeyMask(PClip _child, int _color, int _tolB, int _tolG, int _tolR, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const int color, tolB, tolG, tolR;
  uint64_t color64;
  int tolB16, tolG16, tolR16;
  int pixelsize;
  int bits_per_pixel;
  int max_pixel_value;

};



class ResetMask : public GenericVideoFilter
  /**
    * Class to set the mask to all-opaque
  **/
{
public:
  ResetMask(PClip _child, float _mask_f, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  float mask_f;
  int mask;
};



class Invert : public GenericVideoFilter
  /**
    * Class to invert selected RGBA channels
  **/
{
public:
  Invert(PClip _child, const char* _channels, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
  int mask;
  bool doB, doG, doR, doA;
  bool doY, doU, doV;

  uint64_t mask64;
  int pixelsize;
  int bits_per_pixel; // 8,10..16
};



class ShowChannel : public GenericVideoFilter
  /**
    * Class to set the RGB components to the alpha mask
  **/
{
public:
  ShowChannel(PClip _child, const char* _pixel_type, int _channel, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* channel, IScriptEnvironment* env);
private:
  int channel;
  const int input_type;
  const int pixelsize;
  const int bits_per_pixel;
  bool input_type_is_planar_rgb;
  bool input_type_is_planar_rgba;
  bool input_type_is_yuv;
  bool input_type_is_yuva;
  bool input_type_is_planar;
  bool input_type_is_packed_rgb;
  bool target_hasalpha;
  bool source_hasalpha;
};




class MergeRGB : public GenericVideoFilter
  /**
    * Class to load the RGB components from specified clips
  **/
{
public:
  MergeRGB(PClip _child, PClip _blue, PClip _green, PClip _red, PClip _alpha,
    const char* _pixel_type, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* mode, IScriptEnvironment* env);
private:
  const PClip blue, green, red, alpha;
  const VideoInfo& viB, & viG, & viR, & viA;
  const char* myname;
};


enum
{
  LIGHTEN = 0,
  DARKEN = 1
};

// 15 bit scaled constants used for calculating luma mask from RGB
// original constants (3736,19235,9798) cause int32 overfloat at 16 bits as sum()=32769
// modified constants (3736,19234,9798) O.K. at 16 bits as sum()=32768
// 32769 * 65535 + 16384 = 8000BFFF int32 overflow
// 32768 * 65535 + 16384 = 7FFFC000 OK
const int cyb = 3736;      // int(0.114 * 32768 + 0.5); // 3736
const int cyg = 19235 - 1; // int(0.587 * 32768 + 0.5); // 19235
const int cyr = 9798;      // int(0.299 * 32768 + 0.5); // 9798
// w/o correction: 32769
const float cyb_f = 0.114f;
const float cyg_f = 0.587f;
const float cyr_f = 0.299f;

class Layer : public IClip
  /**
    * Class for layering two clips on each other, combined by various functions
   **/
{
public:
  Layer(PClip _child1, PClip _child2, const char _op[], int _lev, int _x, int _y,
    int _t, bool _chroma, float _strength, int _placement, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  inline virtual void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override
  {
    child1->GetAudio(buf, start, count, env);
  }
  inline virtual const VideoInfo& __stdcall GetVideoInfo() override
  {
    return vi;
  }
  inline virtual bool __stdcall GetParity(int n) override
  {
    return child1->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const PClip child1, child2;
  VideoInfo vi;
  const  char* Op;
  int levelB, ThresholdParam;
  int ydest, xdest, ysrc, xsrc, ofsX, ofsY, ycount, xcount, overlay_frames;
  const bool chroma;
  bool hasAlpha;
  int bits_per_pixel;
  float opacity; // like in "Overlay"
  int placement; // PLACEMENT_MPEG1 or PLACEMENT_MPEG2
  float ThresholdParam_f;
};



class Subtract : public IClip
  /**
    * Class for subtracting one clip from another
   **/
{
public:
  Subtract(PClip _child1, PClip _child2, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  inline virtual void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override
  {
    child1->GetAudio(buf, start, count, env);
  }
  inline virtual const VideoInfo& __stdcall GetVideoInfo() override
  {
    return vi;
  }
  inline virtual bool __stdcall GetParity(int n) override
  {
    return child1->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const PClip child1, child2;
  VideoInfo vi;

  // Common to all instances
  static bool DiffFlag;
  static BYTE LUT_Diff8[513];
  int pixelsize;
  int bits_per_pixel;

};



#endif  // __Layer_H__
