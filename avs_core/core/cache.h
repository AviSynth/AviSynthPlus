// Avisynth v2.6.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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

#ifndef __Cache_H__
#define __Cache_H__

#include <avisynth.h>
#include <mutex>
#include <vector>
#ifdef _DEBUG
#include <string>
#endif

struct CachePimpl;
class InternalEnvironment;

class Cache : public IClip
{
private:

  IScriptEnvironment* Env;
  CachePimpl* _pimpl;
  Device* device;
  void FillAudioZeros(void* buf, int start_offset, int count);

public:
#ifdef _DEBUG
  std::string FuncName = ""; // P.F. Invoked function's name whose queue owns the cache object
#endif
  Cache(const PClip& child, Device* device, InternalEnvironment* env);
  ~Cache();
  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
  const VideoInfo& __stdcall GetVideoInfo();
  bool __stdcall GetParity(int n);
  int __stdcall SetCacheHints(int cachehints,int frame_range);

  Device* GetDevice() const { return device; }

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static bool __stdcall IsCache(const PClip& c);

};

class CacheGuard : public IClip
{
private:
	struct CacheHints {
		size_t min, max;
		CachePolicyHint AudioPolicy;

		CacheHints() :
			min(0), max(std::numeric_limits<size_t>::max()), AudioPolicy(CACHE_AUDIO)
		{ }
	};

    PClip child;
    VideoInfo vi;
		IScriptEnvironment* globalEnv;

    std::vector<std::pair<Device*, PClip>> deviceCaches;
	CacheHints hints;
    mutable std::mutex mutex;

    PClip GetCache(IScriptEnvironment* env);

	void ApplyHints(int cachehints, int frame_range);

	int GetOrDefault(int cachehints, int frame_range, int def);

public:
    CacheGuard(const PClip& child, const char *name, IScriptEnvironment* env);
    ~CacheGuard();
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env);
    const VideoInfo& __stdcall GetVideoInfo();
    bool __stdcall GetParity(int n);
    int __stdcall SetCacheHints(int cachehints, int frame_range);

  static AVSValue __cdecl Create(AVSValue args, void*, IScriptEnvironment* env);
  static bool __stdcall IsCache(const PClip& c);

private:
  std::string name;
  enum {
    // Old 2.5 poorly defined cache hints.
    // Reserve values used by 2.5 API
    // Do not use in new filters
    CACHE_25_NOTHING=0,
    CACHE_25_RANGE=1,
    CACHE_25_ALL=2,
    CACHE_25_AUDIO=3,
    CACHE_25_AUDIO_NONE=4,
    CACHE_25_AUDIO_AUTO=5,
  };
};

#endif  // __Cache_H__
