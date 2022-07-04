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
#ifndef __FPS_H__
#define __FPS_H__

#include <stdint.h>
#include <avisynth.h>
#include "../core/internal.h"

/********************************************************************
********************************************************************/

AVSValue __cdecl ContinuedCreate(AVSValue args, void* key, IScriptEnvironment* env);

void FloatToFPS(const char *name, float n, unsigned &num, unsigned &den, IScriptEnvironment* env);

void PresetToFPS(const char *name, const char *p, unsigned &num, unsigned &den, IScriptEnvironment* env);

class AssumeScaledFPS : public NonCachedGenericVideoFilter
/**
  * Class to change the framerate without changing the frame count
 **/
{
public:
  AssumeScaledFPS(PClip _child, int multiplier, int divisor, bool sync_audio, IScriptEnvironment* env);
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class AssumeFPS : public NonCachedGenericVideoFilter
/**
  * Class to change the framerate without changing the frame count
 **/
{
public:
  AssumeFPS(PClip _child, unsigned numerator, unsigned denominator, bool sync_audio, IScriptEnvironment* env);
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFloat(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreatePreset(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFromClip(AVSValue args, void*, IScriptEnvironment* env);
};


class ChangeFPS : public GenericVideoFilter
/**
  * Class to change the framerate, deleting or adding frames as necessary
 **/
{
public:
  ChangeFPS(PClip _child, unsigned new_numerator, unsigned new_denominator, bool linear, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    //todo: not really sure if it has to be serialized or can do with multiple instances
    return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFloat(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreatePreset(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFromClip(AVSValue args, void*, IScriptEnvironment* env);

private:
  int64_t a, b;
  bool linear;
  int lastframe;
};



class ConvertFPS : public GenericVideoFilter
/**
  * Class to change the framerate, attempting to smooth the transitions
 **/
{
public:
  ConvertFPS( PClip _child, unsigned new_numerator, unsigned new_denominator, int _zone,
              int _vbi, IScriptEnvironment* env );
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFloat(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreatePreset(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateFromClip(AVSValue args, void*, IScriptEnvironment* env);

private:
  int64_t fa, fb;
  int zone;
//Variables used in switch mode only
  int vbi;    //Vertical Blanking Interval (lines)
  int lps;    //Lines per source frame
};


#endif  // __FPS_H_
