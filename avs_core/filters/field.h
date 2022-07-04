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

#ifndef __Field_H__
#define __Field_H__

#include <avisynth.h>
#include "../core/internal.h"


/**********************************************************************
 * Reverse Engineering GetParity
 * -----------------------------
 * Notes for self - and hopefully useful to others.
 *
 * AviSynth is capable of dealing with both progressive and interlaced
 * material. The main problem is, that it often doesn't know what it
 * recieves from source filters.
 *
 * The fieldbased property in VideoInfo is made to give guides to
 * AviSynth on what to expect - unfortunately it is not that easy.
 *
 * GetParity is made to distinguish TFF and BFF. When a given frame is
 * returning true, it means that this frame should be considered
 * top field.
 * So an interlaced image always returning true must be interpreted as
 * TFF.
 *
 * SeparateFields splits out Top and Bottom frames. It returns true on
 * GetParity for Top fields and false for Bottom fields.
 *
 **********************************************************************/


class ComplementParity : public NonCachedGenericVideoFilter
  /**
    * Class to switch field precedence
    **/
{
public:
  ComplementParity(PClip _child) : NonCachedGenericVideoFilter(_child) {
    if (vi.IsBFF() && !vi.IsTFF()) {
      vi.Clear(VideoInfo::IT_BFF);
      vi.Set(VideoInfo::IT_TFF);
    } else if (!vi.IsBFF() && vi.IsTFF()) {
      vi.Set(VideoInfo::IT_BFF);
      vi.Clear(VideoInfo::IT_TFF);
    }
    //  else both were set (illegal state) or both were unset (parity unknown)
  }

  inline bool __stdcall GetParity(int n) {
    return !child->GetParity(n);
  }

  inline static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
    AVS_UNUSED(env);
    return new ComplementParity(args[0].AsClip());
  }

};


class AssumeParity : public NonCachedGenericVideoFilter
  /**
    * Class to assume field precedence, AssumeTFF() & AssumeBFF()
    **/
{
public:
  AssumeParity(PClip _child, bool _parity) : NonCachedGenericVideoFilter(_child), parity(_parity) {
    if (parity) {
      vi.Clear(VideoInfo::IT_BFF);
      vi.Set(VideoInfo::IT_TFF);
    } else {
      vi.Set(VideoInfo::IT_BFF);
      vi.Clear(VideoInfo::IT_TFF);
    }
  }

  inline bool __stdcall GetParity(int n) {
    return parity ^ (vi.IsFieldBased() && (n & 1));
  }

  inline static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env) {
    AVS_UNUSED(env);
    return new AssumeParity(args[0].AsClip(), user_data!=0);
  }

private:
  bool parity;
};

class AssumeFieldBased : public NonCachedGenericVideoFilter
  /**
    * Class to assume field-based video
    **/
{
public:
  AssumeFieldBased(PClip _child) : NonCachedGenericVideoFilter(_child)
  {
    vi.SetFieldBased(true); vi.Clear(VideoInfo::IT_BFF); vi.Clear(VideoInfo::IT_TFF);
  }

  inline bool __stdcall GetParity(int n) {
    return n&1;
  }

  inline static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
    AVS_UNUSED(env);
    return new AssumeFieldBased(args[0].AsClip());
  }

};


class AssumeFrameBased : public NonCachedGenericVideoFilter
  /**
    * Class to assume frame-based video
    **/
{
public:
  AssumeFrameBased(PClip _child) : NonCachedGenericVideoFilter(_child)
  {
    vi.SetFieldBased(false); vi.Clear(VideoInfo::IT_BFF); vi.Clear(VideoInfo::IT_TFF);
  }

  inline bool __stdcall GetParity(int n) {
    AVS_UNUSED(n);
    return false;
  }

  inline static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env) {
    AVS_UNUSED(env);
    return new AssumeFrameBased(args[0].AsClip());
  }
};


class SeparateColumns : public GenericVideoFilter
  /**
    * Class to separate columns of video
    **/
{
private:
  const int interval;

public:
  SeparateColumns(PClip _child, int _interval, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  inline bool __stdcall GetParity(int n) override {
    return child->GetParity(n/interval);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class WeaveColumns : public GenericVideoFilter
  /**
    * Class to weave columns of video
    **/
{
private:
  const int period;
  const int inframes;

public:
  WeaveColumns(PClip _child, int _period, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  inline bool __stdcall GetParity(int n) override {
    return child->GetParity(n*period);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class SeparateRows : public NonCachedGenericVideoFilter
  /**
    * Class to separate lines of video
    **/
{
private:
  const int interval;

public:
  SeparateRows(PClip _child, int _interval, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline bool __stdcall GetParity(int n) {
    return child->GetParity(n/interval);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class WeaveRows : public GenericVideoFilter
  /**
    * Class to weave lines of video
    **/
{
private:
  const int period;
  const int inframes;

public:
  WeaveRows(PClip _child, int _period, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  inline bool __stdcall GetParity(int n) override{
    return child->GetParity(n*period);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class SeparateFields : public NonCachedGenericVideoFilter
  /**
    * Class to separate fields of interlaced video
    **/
{
public:
  SeparateFields(PClip _child, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);

  inline bool __stdcall GetParity(int n) {
    return child->GetParity(n>>1) ^ (n&1);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};


class DoubleWeaveFields : public GenericVideoFilter
  /**
    * Class to weave fields into an equal number of frames
    **/
{
public:
  DoubleWeaveFields(PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }
};


class DoubleWeaveFrames : public GenericVideoFilter
  /**
    * Class to double-weave frames
    **/
{
public:
  DoubleWeaveFrames(PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;

  inline bool __stdcall GetParity(int n) override {
    return child->GetParity(n>>1) ^ (n&1);
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }
};

inline int congmod(int a, int b) {	// congruent modulus
  return ((a % b) + b) % b;
}

class Interleave : public IClip
  /**
    * Class to interleave several clips frame-by-frame
    **/
{
public:
  Interleave(int _num_children, const PClip* _child_array, IScriptEnvironment* env);

  inline const VideoInfo& __stdcall GetVideoInfo() {
    return vi;
  }

  inline PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
	return child_array[congmod(n, num_children)]->GetFrame(n / num_children, env);
  }

  inline void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) {
    child_array[0]->GetAudio(buf, start, count, env);
  }

  inline bool __stdcall GetParity(int n) {
    return child_array[n % num_children]->GetParity(n / num_children);
  }

  virtual ~Interleave() {
    delete[] child_array;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  int __stdcall SetCacheHints(int cachehints, int frame_range);

private:
  const int num_children;
  const PClip* child_array;
  VideoInfo vi;
  int child_devs;
};


class SelectEvery : public NonCachedGenericVideoFilter
  /**
    * Class to perform generalized pulldown (patterned frame removal)
    **/
{
public:
  SelectEvery(PClip _child, int _every, int _from, IScriptEnvironment* env);

  inline PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) {
    return child->GetFrame(n*every+from, env);
  }

  inline bool __stdcall GetParity(int n) {
    return child->GetParity(n*every+from);
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

  inline static AVSValue __cdecl Create_SelectEven(AVSValue args, void*, IScriptEnvironment* env) {
	  return new SelectEvery(args[0].AsClip(), 2, 0, env);
  }

  inline static AVSValue __cdecl Create_SelectOdd(AVSValue args, void*, IScriptEnvironment* env) {
	  return new SelectEvery(args[0].AsClip(), 2, 1, env);
  }

private:
  const int every, from;
};



class Fieldwise : public NonCachedGenericVideoFilter
/**
  * Helper class for Bob filter
 **/
{
public:
  Fieldwise(PClip _child1, PClip _child2);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  bool __stdcall GetParity(int n);

private:
  PClip child2;
};

class SelectRangeEvery : public NonCachedGenericVideoFilter {
  int every, length;
  bool audio;
  PClip achild;
public:
  SelectRangeEvery(PClip _child, int _every, int _length, int _offset, bool _audio, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  bool __stdcall GetParity(int n);
  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);

};


#endif  // __Field_H__
