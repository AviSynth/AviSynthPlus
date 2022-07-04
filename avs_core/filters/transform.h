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

#ifndef __Transform_H__
#define __Transform_H__

#include <avisynth.h>


/********************************************************************
********************************************************************/


class FillBorder : public GenericVideoFilter
{
public:
  FillBorder(PClip _clip);
  static PClip Create(PClip clip);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


class FlipVertical : public GenericVideoFilter
/**
  * Class to vertically flip a video
 **/
{
public:
  FlipVertical(PClip _child) : GenericVideoFilter(_child) {}
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class FlipHorizontal : public GenericVideoFilter
/**
  * Class to Horizontally flip a video
 **/
{
public:
  FlipHorizontal(PClip _child) : GenericVideoFilter(_child) {}
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class Crop : public GenericVideoFilter
/**
  * Class to crop a video
 **/
{
public:
  Crop(int _left, int _top, int _width, int _height, bool _align, PClip _child, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override;

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  /*const*/ int left_bytes, top, align;
  int xsub, ysub;
  bool isRGBPfamily;
  bool hasAlpha;
};



class AddBorders : public GenericVideoFilter
/**
  * Class to add borders to a video
 **/
{
public:
  AddBorders(int _left, int _top, int _right, int _bot, int _clr, bool _force_color_as_yuv, PClip _child, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  /*const*/ int left, top, right, bot, clr;
  int xsub, ysub;
  bool force_color_as_yuv;
};






/**** Factory methods ****/

AVSValue __cdecl Create_Letterbox(AVSValue args, void*, IScriptEnvironment* env);
AVSValue __cdecl Create_CropBottom(AVSValue args, void*, IScriptEnvironment* env);



#endif  // __Transform_H__
