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

#ifndef __Edit_H__
#define __Edit_H__

#include <avisynth.h>
#include "../core/internal.h"

/********************************************************************
********************************************************************/

class Trim : public GenericVideoFilter
/**
  * Class to select a range of frames from a longer clip
 **/
{
public:
  typedef enum { Invalid = 0, Default, Length, End } trim_mode_e;

  Trim(int _firstframe, int _lastframe, bool _padaudio, PClip _child, trim_mode_e mode, bool _cache, IScriptEnvironment* env);
  Trim(double starttime, double endtime, PClip _child, trim_mode_e mode, bool _cache, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    switch (cachehints) {
    case CACHE_DONT_CACHE_ME:
      return cache ? 0 : 1; // adaptively cache-able
    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;
    case CACHE_GET_DEV_TYPE:
      return (child->GetVersion() >= 5) ? child->SetCacheHints(CACHE_GET_DEV_TYPE, 0) : 0;
    default:
      return 0;
    }
  }

  static AVSValue __cdecl Create(AVSValue args, void* mode, IScriptEnvironment* env);
  static AVSValue __cdecl CreateA(AVSValue args, void* mode, IScriptEnvironment* env);

private:
  int firstframe;
  int64_t audio_offset;
  bool cache;
};




class FreezeFrame : public GenericVideoFilter
/**
  * Class to display a single frame for the duration of several
 **/
{
public:
  FreezeFrame(int _first, int _last, int _source, PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const int first, last, source;
};




class DeleteFrame : public GenericVideoFilter
/**
  * Class to delete a frame
 **/
{
public:
  DeleteFrame(int _frame, PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const int frame;
};




class DuplicateFrame : public GenericVideoFilter
/**
  * Class to duplicate a frame
 **/
{
public:
  DuplicateFrame(int _frame, PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  const int frame;
};




class Splice : public GenericVideoFilter
/**
  * Class to splice together video clips
 **/
{
public:
  Splice(PClip _child1, PClip _child2, bool realign_sound, bool passCache, IScriptEnvironment* env);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
  bool __stdcall GetParity(int n);
  int __stdcall SetCacheHints(int cachehints,int frame_range);

  static AVSValue __cdecl CreateUnaligned(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateAligned(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateUnalignedNoCache(AVSValue args, void*, IScriptEnvironment* env);
  static AVSValue __cdecl CreateAlignedNoCache(AVSValue args, void*, IScriptEnvironment* env);

private:
  PClip child2;
  int video_switchover_point;
  int64_t audio_switchover_point;
  const bool passCache;
  int child_devs;
};




class Dissolve : public GenericVideoFilter
/**
  * Class to smoothly transition from one video clip to another
 **/
{
public:
  Dissolve(PClip _child1, PClip _child2, int _overlap, double fps, IScriptEnvironment* env);
  virtual ~Dissolve();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
  bool __stdcall GetParity(int n);

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);

private:
  PClip child2;
  const int overlap;
  int video_fade_start, video_fade_end;
  int64_t audio_fade_start, audio_fade_end;
  int audio_overlap;
  BYTE* audbuffer;
  size_t audbufsize;
  int pixelsize;
  int bits_per_pixel;
  void EnsureBuffer(int minsize);
};




class AudioDub : public IClip {
/**
  * Class to mux the audio track of one clip with the video of another
 **/
public:
  AudioDub(PClip child1, PClip child2, int mode, IScriptEnvironment* env);
  const VideoInfo& __stdcall GetVideoInfo();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
  bool __stdcall GetParity(int n);
  int __stdcall SetCacheHints(int cachehints,int frame_range);

  static AVSValue __cdecl Create(AVSValue args, void* mode, IScriptEnvironment* env);

private:
  /*const*/ PClip vchild, achild;
  VideoInfo vi;
};




class Reverse : public GenericVideoFilter
/**
  * Class to play a clip backwards
 **/
{
public:
  Reverse(PClip _child);
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
  bool __stdcall GetParity(int n) override;
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
};




class Loop : public GenericVideoFilter {
/**
  * Class to loop over a range of frames
**/
public:
	Loop(PClip _child, int times, int _start, int _end, IScriptEnvironment* env);
	PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env) override;
	bool __stdcall GetParity(int n) override;
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env) override;

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    AVS_UNUSED(frame_range);
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

	static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
private:
	int   frames,    start,     end;
  int64_t aud_count, aud_start, aud_end;
	int convert(int n);
};




/**** A few factory methods ****/

AVSValue __cdecl Create_Fade(AVSValue args, void* user_data, IScriptEnvironment* env);
PClip new_Splice(PClip _child1, PClip _child2, bool realign_sound, IScriptEnvironment* env);


#endif  // __Edit_H__
