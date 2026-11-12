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

#ifndef __Combine_H__
#define __Combine_H__

#include <avisynth.h>
#include <vector>

/********************************************************************
********************************************************************/
class MultiOverlay : public IClip
  /**
    * Class to Overlay multiple clips in one pass
    **/
{
private:
  std::vector<PClip> children;
  std::vector<int> positions;
  VideoInfo vi;

public:
  MultiOverlay(const std::vector<PClip>& child_array, const std::vector<int>& position_array, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    children[0]->GetAudio(buf, start, count, env);
  }

  inline const VideoInfo& __stdcall GetVideoInfo() {
    return vi;
  }

  inline bool __stdcall GetParity(int n) {
    return children[0]->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};



class StackVertical : public IClip
  /**
    * Class to stack clips vertically
    **/
{
private:
  std::vector<PClip> children;
  VideoInfo vi;
  int firstchildindex;

public:
  StackVertical(const std::vector<PClip>& child_array, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    children[firstchildindex]->GetAudio(buf, start, count, env);
  }

  inline const VideoInfo& __stdcall GetVideoInfo() {
    return vi;
  }

  inline bool __stdcall GetParity(int n) {
    return children[firstchildindex]->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};



class StackHorizontal : public IClip
  /**
    * Class to stack clips vertically
    **/
{
private:
  std::vector<PClip> children;
  VideoInfo vi;

public:
  StackHorizontal(const std::vector<PClip>& child_array, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    children[0]->GetAudio(buf, start, count, env);
  }

  inline const VideoInfo& __stdcall GetVideoInfo() {
    return vi;
  }

  inline bool __stdcall GetParity(int n) {
    return children[0]->GetParity(n);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};




class ShowFiveVersions : public IClip
/**
  * Class to show every pulldown combination
 **/
{
public:
  ShowFiveVersions(PClip* children, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
    { child[0]->GetAudio(buf, start, count, env); }
  inline const VideoInfo& __stdcall GetVideoInfo()
    { return vi; }
  inline bool __stdcall GetParity(int n)
    { return child[0]->GetParity(n); }
  int __stdcall SetCacheHints(int cachehints,int frame_range) {
    AVS_UNUSED(cachehints);
    AVS_UNUSED(frame_range);
    return 0;
  };

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  PClip child[5];
  VideoInfo vi;
};



class Animate : public IClip
/**
  * Class to allow recursive animation of multiple clips (see docs)  *
 **/
{
  enum { cache_size = 3 };
  PClip cache[cache_size];
  int64_t cache_stage[cache_size]; // 0 .. last-first, which may exceed int range
  const int first, last;
  std::vector<AVSValue> args_before;
  AVSValue *args_after, *args_now;
  int num_args;
  const char* name;
  bool range_limit;
  AVSValue custom_fn;
  void InterpolateArgs(int64_t stage, int64_t scale, IScriptEnvironment* env);
public:
  Animate( PClip context, int _first, int _last, const char* _name, const AVSValue* _args_before,
           const AVSValue* _args_after, int _num_args, bool _range_limit, const AVSValue& _custom_fn, IScriptEnvironment* env );
  virtual ~Animate() { }
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);

  inline const VideoInfo& __stdcall GetVideoInfo()
    { return cache[0]->GetVideoInfo(); }
  bool __stdcall GetParity(int n);

  int __stdcall SetCacheHints(int cachehints,int frame_range) {
    AVS_UNUSED(frame_range);
    switch (cachehints)
    {
    case CACHE_GET_MTMODE:
      return MT_MULTI_INSTANCE;
    }
    return 0;  // We do not pass cache requests upwards.
  };

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl Create_Range(AVSValue args, void*, IScriptEnvironment* env);
};


#endif  // __Combine_H__
