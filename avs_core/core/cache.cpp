// Avisynth v2.6.  Copyright 2002-2009 Ben Rudiak-Gould et al.
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

#include "cache.h"
#include "internal.h"
#include "LruCache.h"
#include "InternalEnvironment.h"
#include "DeviceManager.h"
#include <cassert>
#include <chrono>
#include <cstdio>

#ifdef X86_32
#include <mmintrin.h>
#endif

#include <avs/config.h>
#ifdef AVS_WINDOWS
#include <avs/win.h>
#else
#include <avs/posix.h>
#endif


extern const AVSFunction Cache_filters[] = {
  { "Cache", BUILTIN_FUNC_PREFIX, "c[name]s", CacheGuard::Create },
  { "InternalCache", BUILTIN_FUNC_PREFIX, "c[name]s", CacheGuard::Create },
  { 0 }
};

class CacheStack
{
  InternalEnvironment* env;
  bool retSupressCaching;
public:
  CacheStack(InternalEnvironment* env)
    : env(env)
    , retSupressCaching(env->GetSupressCaching())
  { }
  ~CacheStack() {
    env->GetSupressCaching() = retSupressCaching;
  }
};

struct CachePimpl
{
  PClip child;
  VideoInfo vi;

  // Video cache
  std::shared_ptr<LruCache<size_t, PVideoFrame> > VideoCache;

  // Audio cache
  CachePolicyHint AudioPolicy;
  char* AudioCache;
  int SampleSize;
  int64_t MaxSampleCount;
  int64_t AudioCacheStart;
  int64_t CacheCount;
  int64_t ac_expected_next;
  int ac_too_small_count;
  long ac_currentscore;

  CachePimpl(const PClip& _child, CacheMode mode) :
    child(_child),
    vi(_child->GetVideoInfo()),
    VideoCache(std::make_shared<LruCache<size_t, PVideoFrame> >(0, mode)),
    AudioPolicy(vi.HasAudio() ? CACHE_AUDIO_AUTO_START_OFF : CACHE_AUDIO_NOTHING),  // Don't cache audio per default, auto mode.
    AudioCache(NULL),
    SampleSize(0),
    MaxSampleCount(0),
    AudioCacheStart(0),
    CacheCount(0),
    ac_expected_next(0),
    ac_too_small_count(0),
    ac_currentscore(20)
  {
    SampleSize = vi.BytesPerAudioSample();
  }
  ~CachePimpl()
  {
    if (AudioCache)
      free(AudioCache);
    AudioCache = NULL;
  }
};


AvsCache::AvsCache(const PClip& _child, Device* device, std::mutex& CacheGuardMutex, InternalEnvironment* env) :
  Env(env),
  _pimpl(NULL),
  device(device),
  CacheGuardMutex(CacheGuardMutex)
{
  _pimpl = new CachePimpl(_child, env->GetCacheMode());
  env->ManageCache(MC_RegisterCache, reinterpret_cast<void*>(this));
  _RPT5(0, "AvsCache::AvsCache registered. cache_id=%p child=%p w=%d h=%d VideoCacheSize=%Iu\n", (void*)this, (void*)_child, _pimpl->vi.width, _pimpl->vi.height, _pimpl->VideoCache->size());

  const int dummy = 0;
  // child is usually MTGuard, which forwards this request to its child filter - the real one
  const bool overAvs25 = (_child->GetVersion() >= 5);
  CachePolicyHint childAudioPolicy = (CachePolicyHint)(overAvs25 ? _child->SetCacheHints(CACHE_GETCHILD_AUDIO_MODE, dummy) : 0);
  // if not set by the plugin, get default (set in CachePimpl ctor)
  if (childAudioPolicy == 0)
    childAudioPolicy = (CachePolicyHint)this->SetCacheHints(CACHE_GET_AUDIO_POLICY, dummy);
  switch (childAudioPolicy) {
  case CachePolicyHint(0): break; // n/a
  case CACHE_AUDIO:          // Explicitly do cache audio, X byte cache.
  case CACHE_AUDIO_NOTHING:  // Explicitly do not cache audio.
  case CACHE_AUDIO_AUTO_START_OFF: // Audio cache off (auto mode), X byte initial cache.
  case CACHE_AUDIO_AUTO_START_ON:  // Audio cache on (auto mode), X byte initial cache.
    // ask child for desired audio cache size
    // e.g. EnsureVBRMP3Sync explicitely requests CACHE_AUDIO and 1024*1024
    // This will create cache only if clip has audio
    this->SetCacheHints(childAudioPolicy, overAvs25 ? _child->SetCacheHints(CACHE_GETCHILD_AUDIO_SIZE, dummy) : 0); // if size=0 remains the default
    break;
  default:
    env->ThrowError("Cache: Filter returned invalid response to CACHE_GETCHILD_AUDIO_MODE. %d", childAudioPolicy);
  }
}

AvsCache::~AvsCache()
{
  _RPT5(0, "AvsCache::AvsCache unregister. cache_id=%p child=%p w=%d h=%d VideoCacheSize=%Iu\n", (void *)this, (void *)_pimpl->child, _pimpl->vi.width, _pimpl->vi.height, _pimpl->VideoCache->size()); // P.F.
  Env->ManageCache(MC_UnRegisterCache, reinterpret_cast<void*>(this));
  delete _pimpl;
}

PVideoFrame __stdcall AvsCache::GetFrame(int n, IScriptEnvironment* env_)
{
#ifdef _DEBUG
  constexpr auto BUFSIZE = 255;
  std::unique_ptr<char[]> buf(new char[BUFSIZE+1]);

#endif
  InternalEnvironment* env = GetAndRevealCamouflagedEnv(env_);

  // Protect plugins that cannot handle out-of-bounds frame indices
  n = clamp(n, 0, GetVideoInfo().num_frames-1);

  if (_pimpl->VideoCache->requested_capacity() > _pimpl->VideoCache->capacity())
    env->ManageCache(MC_NodAndExpandCache, reinterpret_cast<void*>(this));
  else
    env->ManageCache(MC_NodCache, reinterpret_cast<void*>(this));

  PVideoFrame result;
  LruCache<size_t, PVideoFrame>::handle cache_handle;

  CacheStack cache_stack(env);

#ifdef _DEBUG
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;
  t_start = std::chrono::high_resolution_clock::now(); // t_start starts in the constructor. Used in logging

  LruLookupResult LruLookupRes = _pimpl->VideoCache->lookup(n, &cache_handle, true, result, &env->GetSupressCaching());
  /*
  std::string name = FuncName;
  snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame lookup follows: [%s] n=%6d Thread=%zu", name.c_str(), n, env->GetEnvProperty(AEP_THREAD_ID));
  _RPT0(0, buf.get());

  snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame lookup ready: [%s] n=%6d Thread=%zu res=%d", name.c_str(), n, env->GetEnvProperty(AEP_THREAD_ID), (int)LruLookupRes);
  _RPT0(0, buf.get());
  */

#ifdef _DEBUG
  std::string name = FuncName;
#endif

  switch (LruLookupRes)
#else
  // fill result in lookup before releasing cache handle lock
  switch(_pimpl->VideoCache->lookup(n, &cache_handle, true, result, &env->GetSupressCaching()))
#endif
  {
  case LRU_LOOKUP_NOT_FOUND:
    {
      try
      {
#ifdef _DEBUG
        snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame LRU_LOOKUP_NOT_FOUND: [%s] n=%6d child=%p\n", name.c_str(), n, (void*)_pimpl->child); // P.F.
        _RPT0(0, buf.get());
#endif
        //cache_handle.first->value = _pimpl->child->GetFrame(n, env);
        result = _pimpl->child->GetFrame(n, env); // P.F. fill result immediately

        // check device
        if (result->GetFrameBuffer()->device != device) {
            const char* error_msg = env->Sprintf("Frame device mismatch: Assumed: %s Actual: %s",
                device->GetName(), result->GetFrameBuffer()->device->GetName());
            result = env->NewVideoFrame(_pimpl->vi);
            env->ApplyMessage(&result, _pimpl->vi, error_msg, _pimpl->vi.width / 5, 0xa0a0a0, 0, 0);
        }

        cache_handle.first->value = result; // not after commit!
  #ifdef X86_32
        _mm_empty();
  #endif
        _pimpl->VideoCache->commit_value(&cache_handle);
      }
      catch (...)
      {
        _pimpl->VideoCache->rollback(&cache_handle);
        throw;
      }
#ifdef _DEBUG
      t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = t_end - t_start;
      std::string name = FuncName;
      if (NULL == cache_handle.first->value) {
          snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame LRU_LOOKUP_NOT_FOUND: HEY! got nulled! [%s] n=%6d child=%p frame=%p framebefore=%p SeekTimeWithGetFrame:%f\n", name.c_str(), n, (void *)_pimpl->child, (void *)cache_handle.first->value, (void *)result, elapsed_seconds.count()); // P.F.
          _RPT0(0, buf.get());
      } else {
          snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame LRU_LOOKUP_NOT_FOUND: [%s] n=%6d child=%p frame=%p framebefore=%p videoCacheSize=%zu SeekTimeWithGetFrame:%f\n", name.c_str(), n, (void *)_pimpl->child, (void *)cache_handle.first->value, (void *)result, _pimpl->VideoCache->size(), elapsed_seconds.count()); // P.F.
          _RPT0(0, buf.get());
      }
#endif
      // result = cache_handle.first->value; not here!
      // its content may change after commit when the last lock is released
      // (cache is being restructured by other threads, new frames, etc...)
      break;
    }
  case LRU_LOOKUP_FOUND_AND_READY:
    {
      // theoretically cache_handle here may point to wrong entry,
      // because the lock in lookup is released before this readout
      // solution:
      // when LRU_LOOKUP_FOUND_AND_READY, the cache_handle.first->value is copied and returned in result itself
      // result =  cache_handle.first->value; // old method not needed, result is filled already by lookup
#ifdef _DEBUG
      t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = t_end - t_start;
      std::string name = FuncName;
      snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame LRU_LOOKUP_FOUND_AND_READY: [%s] n=%6d child=%p frame=%p vfb=%p videoCacheSize=%zu SeekTime            :%f\n", name.c_str(), n, (void *)_pimpl->child, (void *)result, (void *)result->GetFrameBuffer(), _pimpl->VideoCache->size(), elapsed_seconds.count());
      _RPT0(0, buf.get());
      assert(result != NULL);
#endif
      break;
    }
  case LRU_LOOKUP_NO_CACHE:
    {
#ifdef _DEBUG
    snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame <Before GetFrame> LRU_LOOKUP_NO_CACHE: [%s] n=%6d child=%p\n", name.c_str(), n, (void*)_pimpl->child); // P.F.
    _RPT0(0, buf.get());
#endif
    result = _pimpl->child->GetFrame(n, env);
#ifdef _DEBUG
      t_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_seconds = t_end - t_start;
      snprintf(buf.get(), BUFSIZE, "AvsCache::GetFrame <After GetFrame> LRU_LOOKUP_NO_CACHE: [%s] n=%6d child=%p frame=%p vfb=%p videoCacheSize=%zu SeekTime            :%f\n", name.c_str(), n, (void *)_pimpl->child, (void *)result, (void *)result->GetFrameBuffer(), _pimpl->VideoCache->size(), elapsed_seconds.count()); // P.F.
      _RPT0(0, buf.get());
#endif
      break;
    }
  case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:    // Fall-through intentional
  default:
    {
      assert(0);
      break;
    }
  }

  return result;
}

void AvsCache::FillAudioZeros(void* buf, size_t start_offset, size_t count) {
    const int bps = _pimpl->vi.BytesPerAudioSample();
    unsigned char* byte_buf = (unsigned char*)buf;
    memset(byte_buf + start_offset * bps, 0, count * bps);
}

void __stdcall AvsCache::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{
  if (count <= 0)
    return;
  if (count > 0 && static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max())
    env->ThrowError("GetAudio error: Count exceeds platform maximum buffer size limit.");

  // -----------------------------------------------------------
  //          Enforce audio bounds
  // -----------------------------------------------------------

  VideoInfo * vi = &(_pimpl->vi);

  if ((!vi->HasAudio()) || (start + count <= 0) || (start >= vi->num_audio_samples)) {
    // Completely skip.
    FillAudioZeros(buf, 0, (size_t)count);
    count = 0;
    return;
  }

  if (start < 0) {  // Partial initial skip
    FillAudioZeros(buf, 0, static_cast<size_t>(-start));  // Fill all samples before 0 with silence.
    count += start;  // Subtract start bytes from count.
    buf = ((BYTE*)buf) - start * vi->BytesPerAudioSample();
    start = 0;
  }

  if (start + count > vi->num_audio_samples) {  // Partial ending skip
    FillAudioZeros(buf, (size_t)(vi->num_audio_samples - start), (size_t)(count - (vi->num_audio_samples - start)));  // Fill end samples
    count = (vi->num_audio_samples - start);
  }

  // -----------------------------------------------------------
  //          Caching
  // -----------------------------------------------------------

  long _cs = _pimpl->ac_currentscore;
  if (start < _pimpl->ac_expected_next)
    _cs = InterlockedExchangeAdd(&_pimpl->ac_currentscore, -5) - 5;  // revisiting old ground - a cache could help
  else if (start > _pimpl->ac_expected_next)
    _cs = InterlockedDecrement(&_pimpl->ac_currentscore);            // skipping chunks - a cache might not help
  else // (start == ac_expected_next)
    _cs = InterlockedIncrement(&_pimpl->ac_currentscore);            // strict linear reading - why bother with a cache

  if (_cs < -2000000)
    _pimpl->ac_currentscore = -2000000;
  else if (_cs > 90)
    _pimpl->ac_currentscore = 90;

  // Change from AUTO_OFF to AUTO_ON
  if (_pimpl->AudioPolicy == CACHE_AUDIO_AUTO_START_OFF && _pimpl->ac_currentscore <= 0) {
    // align size to 8192 byte boundary
    int64_t new_size = (_pimpl->vi.BytesFromAudioSamples(count) + 8191) & -8192;
    new_size = min((int64_t)4096 * 1024, new_size);
    _RPT2(0, "CA:%x: Automatically adding %d byte audiocache!\n", this, (int)new_size);
    SetCacheHints(CACHE_AUDIO_AUTO_START_ON, (int)new_size);
  }

  // Change from AUTO_ON to AUTO_OFF
  if (_pimpl->AudioPolicy == CACHE_AUDIO_AUTO_START_ON && (_pimpl->ac_currentscore > 80)) {
    _RPT1(0, "CA:%x: Automatically deleting cache!\n", this);
    SetCacheHints(CACHE_AUDIO_AUTO_START_OFF, 0);
  }

  _pimpl->ac_expected_next = start + count;

  if (_pimpl->AudioPolicy == CACHE_AUDIO_AUTO_START_OFF || _pimpl->AudioPolicy == CACHE_AUDIO_NOTHING) {
    _pimpl->child->GetAudio(buf, start, count, env);
    return;  // We are ok to return now!
  }

  std::unique_lock<std::mutex> global_lock(CacheGuardMutex);

  while (count > _pimpl->MaxSampleCount) {    //is cache big enough?
    _RPT1(0, "CA:%x:Cache too small->caching last audio\n", this);
    _pimpl->ac_too_small_count++;

    // But the current cache might have 99% of what was requested??

    if (_pimpl->ac_too_small_count > 2 && _pimpl->MaxSampleCount < _pimpl->vi.AudioSamplesFromBytes(8192 * 1024)) {  // Max size = 8MB!
      //automatically upsize cache!
      int64_t new_size = (_pimpl->vi.BytesFromAudioSamples(std::max(count, _pimpl->AudioCacheStart + _pimpl->CacheCount - start)) + 8192) & -8192; // Yes +1 to +8192 bytes
      new_size = std::min((int64_t)8192 * 1024, new_size);
      _RPT2(0, "CA:%x: Autoupsizing buffer to %d bytes!\n", this, (int)new_size);
      SetCacheHints(_pimpl->AudioPolicy, (int)new_size); // updates maxsamplecount!!
      _pimpl->ac_too_small_count = 0;
    }
    else {
      global_lock.unlock();
      _pimpl->child->GetAudio(buf, start, count, env);
      global_lock.lock();

      _pimpl->CacheCount = std::min(count, _pimpl->MaxSampleCount); // Remember maxsamplecount gets updated
      _pimpl->AudioCacheStart = start + count - _pimpl->CacheCount;
      BYTE* buff = (BYTE*)buf;
      buff += _pimpl->vi.BytesFromAudioSamples(_pimpl->AudioCacheStart - start);
      memcpy(_pimpl->AudioCache, buff, (size_t)_pimpl->vi.BytesFromAudioSamples(_pimpl->CacheCount));

      global_lock.unlock();

      return;
    }
  }

  if ((start < _pimpl->AudioCacheStart) || (start > _pimpl->AudioCacheStart + _pimpl->MaxSampleCount)) { //first sample is before cache or beyond linear reach -> restart cache
    _RPT1(0, "CA:%x: Restart\n", this);

    _pimpl->CacheCount = std::min(count, _pimpl->MaxSampleCount);
    _pimpl->AudioCacheStart = start;
    _pimpl->child->GetAudio(_pimpl->AudioCache, _pimpl->AudioCacheStart, _pimpl->CacheCount, env);
  }
  else {
    if (start + count > _pimpl->AudioCacheStart + _pimpl->CacheCount) { // Does the cache fail to cover the request?
      if (start + count > _pimpl->AudioCacheStart + _pimpl->MaxSampleCount) {  // Is cache shifting necessary?
        int shiftsamples = (int)((start + count) - (_pimpl->AudioCacheStart + _pimpl->MaxSampleCount)); // Align end of cache with end of request

        if ((start - _pimpl->AudioCacheStart) / 2 > shiftsamples) {  //shift half cache if possible
          shiftsamples = (int)((start - _pimpl->AudioCacheStart) / 2);
        }
        if (shiftsamples >= (int)_pimpl->CacheCount) { // Can we save any existing data
          _pimpl->AudioCacheStart = start + count - _pimpl->MaxSampleCount; // Maximise linear access
          _pimpl->CacheCount = 0;
        }
        else {
          memmove(_pimpl->AudioCache, _pimpl->AudioCache + shiftsamples * _pimpl->SampleSize, (size_t)((_pimpl->CacheCount - shiftsamples) * _pimpl->SampleSize));
          _pimpl->AudioCacheStart = _pimpl->AudioCacheStart + shiftsamples;
          _pimpl->CacheCount = _pimpl->CacheCount - shiftsamples;
        }
      }
      // Read just enough to complete the current request, append it to the cache
      _pimpl->child->GetAudio(_pimpl->AudioCache + _pimpl->CacheCount * _pimpl->SampleSize, _pimpl->AudioCacheStart + _pimpl->CacheCount, start + count - (_pimpl->AudioCacheStart + _pimpl->CacheCount), env);
      _pimpl->CacheCount += (size_t)(start + count - (_pimpl->AudioCacheStart + _pimpl->CacheCount));
    }
  }

  //copy cache to buf
  memcpy(buf, _pimpl->AudioCache + (size_t)(start - _pimpl->AudioCacheStart) * _pimpl->SampleSize, (size_t)(count * _pimpl->SampleSize));

}

const VideoInfo& __stdcall AvsCache::GetVideoInfo()
{
  return _pimpl->vi;
}

bool __stdcall AvsCache::GetParity(int n)
{
  return _pimpl->child->GetParity(n);
}

int __stdcall AvsCache::SetCacheHints(int cachehints, int frame_range)
{
  // _RPT3(0, "AvsCache::SetCacheHints called. cache=%p hint=%d frame_range=%d\n", (void *)this, cachehints, frame_range);
  switch(cachehints)
  {
    /*********************************************
        MISC
    *********************************************/

    // By returning IS_CACHE_ANS to IS_CACHE_REQ, we tell the caller we are a cache
    case CACHE_IS_CACHE_REQ:
      return CACHE_IS_CACHE_ANS;

    case CACHE_GET_POLICY: // Get the current policy.
      return CACHE_GENERIC;

    case CACHE_DONT_CACHE_ME:
      return 1;

    case CACHE_GET_MTMODE:
      return MT_NICE_FILTER;

    /*********************************************
        VIDEO
    *********************************************/

    case CACHE_SET_MIN_CAPACITY:
    { // This is not atomic, but frankly, we don't care
      size_t min, max;
      _pimpl->VideoCache->limits(&min, &max);
      min = frame_range;
      _pimpl->VideoCache->set_limits(min, max);
      break;
    }

    case CACHE_SET_MAX_CAPACITY:
    { // This is not atomic, but frankly, we don't care
      size_t min, max;
      _pimpl->VideoCache->limits(&min, &max);
      max = frame_range;
      _pimpl->VideoCache->set_limits(min, max);
      _RPT3(0, "AvsCache::SetCacheHints CACHE_SET_MAX_CAPACITY cache=%p hint=%d frame_range=%d\n", (void *)this, cachehints, frame_range);
      break;
    }

    case CACHE_GET_MIN_CAPACITY:
    {
      size_t min, max;
      _pimpl->VideoCache->limits(&min, &max);
      return (int)min;
    }

    case CACHE_GET_MAX_CAPACITY:
    {
      size_t min, max;
      _pimpl->VideoCache->limits(&min, &max);
      return (int)max;
    }

    case CACHE_GET_SIZE:
      return (int)_pimpl->VideoCache->size();

    case CACHE_GET_REQUESTED_CAP:
      return (int)_pimpl->VideoCache->requested_capacity();

    case CACHE_GET_CAPACITY:
      return (int)_pimpl->VideoCache->capacity();

    case CACHE_GET_WINDOW: // Get the current window h_span.
    case CACHE_GET_RANGE: // Get the current generic frame range.
      return 2;
      break;

    case CACHE_GENERIC:
    case CACHE_FORCE_GENERIC:
    case CACHE_NOTHING:
    case CACHE_WINDOW:
    case CACHE_PREFETCH_FRAME:          // Queue request to prefetch frame N.
    case CACHE_PREFETCH_GO:             // Action video prefetches.
      break;

    /*********************************************
        AUDIO
    *********************************************/

    case CACHE_AUDIO:
    case CACHE_AUDIO_AUTO_START_ON:
      if (!_pimpl->vi.HasAudio())
        break;

      // Range means for audio.
      // 0 == Create a default buffer (256kb).
      // Positive. Allocate X bytes for cache.
      if (frame_range == 0) {
        if (_pimpl->AudioPolicy != CACHE_AUDIO_AUTO_START_OFF)   // We already have a policy - no need for a default one.
          break;

        frame_range = 256 * 1024;
      }

      if (frame_range/_pimpl->SampleSize > _pimpl->MaxSampleCount) { // Only make bigger
        const bool audio_cache_existed = _pimpl->AudioCache != nullptr;
        // keep content, newly added bytes are undefined on increase
        // but CacheCount keeps track on that fact
        char * NewAudioCache = (char*)realloc(_pimpl->AudioCache, frame_range);
        if (NewAudioCache == NULL)
        {
          throw std::bad_alloc();
        }
        _pimpl->AudioCache = NewAudioCache;

        _pimpl->MaxSampleCount = frame_range/_pimpl->SampleSize;
        if(audio_cache_existed)
          _pimpl->CacheCount = std::min(_pimpl->CacheCount, _pimpl->MaxSampleCount);
        else {
          _pimpl->AudioCacheStart = 0;
          _pimpl->CacheCount = 0;
        }
      }

      _pimpl->AudioPolicy = (CachePolicyHint)cachehints;
      break;

    case CACHE_AUDIO_AUTO_START_OFF:
    case CACHE_AUDIO_NOTHING:
      free(_pimpl->AudioCache);
      _pimpl->AudioCache = NULL;
      _pimpl->MaxSampleCount = 0;
      _pimpl->AudioCacheStart = 0;
      _pimpl->CacheCount = 0;
      _pimpl->AudioPolicy = (CachePolicyHint)cachehints;
      break;

    case CACHE_GET_AUDIO_POLICY: // Get the current audio policy.
      return _pimpl->AudioPolicy;

    case CACHE_GET_AUDIO_SIZE: // Get the current audio cache size.
      return (int)(_pimpl->SampleSize * _pimpl->MaxSampleCount);

    // n/a ignore them, not implemented even in 2.6
    case CACHE_PREFETCH_AUDIO_BEGIN:    // Begin queue request to prefetch audio (take critical section).
    case CACHE_PREFETCH_AUDIO_STARTLO:  // Set low 32 bits of start.
    case CACHE_PREFETCH_AUDIO_STARTHI:  // Set high 32 bits of start.
    case CACHE_PREFETCH_AUDIO_COUNT:    // Set low 32 bits of length.
    case CACHE_PREFETCH_AUDIO_COMMIT:   // Enqueue request transaction to prefetch audio (release critical section).
    case CACHE_PREFETCH_AUDIO_GO:       // Action audio prefetch
      break;

    default:
      return 0;
  }

  return 0;
}

CacheGuard::CacheGuard(const PClip& child, const char *name, IScriptEnvironment* env) :
    child(child),
    vi(child->GetVideoInfo()),
    globalEnv(env)
{
  if (name)
    this->name = name;

  // other fields are set OK already
  hints.AudioPolicy = vi.HasAudio() ? CACHE_AUDIO_AUTO_START_OFF : CACHE_AUDIO_NOTHING;
  hints.default_AudioPolicy = hints.AudioPolicy;

}

CacheGuard::~CacheGuard()
{ }

PClip CacheGuard::GetCache(IScriptEnvironment* env_)
{
  std::unique_lock<std::mutex> global_lock(mutex);

  InternalEnvironment* env = GetAndRevealCamouflagedEnv(env_);

  Device* device = env->GetCurrentDevice();

  for (auto entry : deviceCaches) {
    if (entry.first == device) {
      return entry.second;
    }
  }

  // not found for current device, create it
  AvsCache* cache = new AvsCache(child, device, /*ref*/mutex, static_cast<InternalEnvironment*>(globalEnv));

  // apply cache hints if it is changed
  if (hints.min != hints.default_min)
    cache->SetCacheHints(CACHE_SET_MIN_CAPACITY, (int)hints.min);
  if (hints.max != hints.default_max)
    cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, (int)hints.max);

  if (hints.AudioPolicy != hints.default_AudioPolicy || hints.AudioSize != hints.default_AudioSize)
    cache->SetCacheHints(hints.AudioPolicy, hints.AudioSize);

  deviceCaches.emplace_back(device, cache);
  return deviceCaches.back().second;
}

void CacheGuard::ApplyHints(int cachehints, int frame_range)
{
  std::unique_lock<std::mutex> global_lock(mutex);
  for (auto entry : deviceCaches) {
    entry.second->SetCacheHints(cachehints, frame_range);
  }
}

int CacheGuard::GetOrDefault(int cachehints, int frame_range, int def)
{
  std::unique_lock<std::mutex> global_lock(mutex);
  for (auto entry : deviceCaches) {
    return entry.second->SetCacheHints(cachehints, frame_range);
  }
  return def;
}

PVideoFrame __stdcall CacheGuard::GetFrame(int n, IScriptEnvironment* env_)
{
  InternalEnvironment* IEnv = GetAndRevealCamouflagedEnv(env_);
  IScriptEnvironment* env = static_cast<IScriptEnvironment*>(IEnv);

  ScopedCounter getframe_counter(IEnv->GetFrameRecursiveCount());
  /*
  if (!name.empty())
    _RPT2(0, "CacheGuard::GetFrame call further GetFrame: %s %d\n", name.c_str(), n);
  */
  return GetCache(env)->GetFrame(n, env);
}

void __stdcall CacheGuard::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env_)
{
  InternalEnvironment* IEnv = GetAndRevealCamouflagedEnv(env_);
  IScriptEnvironment* env = static_cast<IScriptEnvironment*>(IEnv);

  ScopedCounter getframe_counter(IEnv->GetFrameRecursiveCount());
  return GetCache(env)->GetAudio(buf, start, count, env);
}

const VideoInfo& __stdcall CacheGuard::GetVideoInfo()
{
    return vi;
}

bool __stdcall CacheGuard::GetParity(int n)
{
    return child->GetParity(n);
}

int __stdcall CacheGuard::SetCacheHints(int cachehints, int frame_range)
{
  _RPT3(0, "CacheGuard::SetCacheHints called. cache=%p hint=%d frame_range=%d\n", (void *)this, cachehints, frame_range); // P.F.
  switch (cachehints)
  {
    /*********************************************
    MISC
    *********************************************/

    // By returning IS_CACHE_ANS to IS_CACHE_REQ, we tell the caller we are a cache
  case CACHE_IS_CACHE_REQ:
    return CACHE_IS_CACHE_ANS;

  case CACHE_GET_POLICY: // Get the current policy.
    return CACHE_GENERIC;

  case CACHE_DONT_CACHE_ME:
    return 1;

  case CACHE_GET_MTMODE:
    return MT_NICE_FILTER;

    /*********************************************
    AVS 2.5 TRANSLATION
    *********************************************/

    // Ignore 2.5 CACHE_NOTHING requests
  case CACHE_25_NOTHING:
    break;

    // Map 2.5 CACHE_RANGE calls to CACHE_WINDOW
    // force minimum range to 2
  case CACHE_25_RANGE:
    if (frame_range < 2) frame_range = 2;
    SetCacheHints(CACHE_WINDOW, frame_range);
    break;

    // Map 2.5 CACHE_ALL calls to CACHE_GENERIC
  case CACHE_25_ALL:
    SetCacheHints(CACHE_GENERIC, frame_range);
    break;

    // Map 2.5 CACHE_AUDIO calls to CACHE_AUDIO
  case CACHE_25_AUDIO:
    SetCacheHints(CACHE_AUDIO, frame_range);
    break;

    // Map 2.5 CACHE_AUDIO_NONE calls to CACHE_AUDIO_NONE
  case CACHE_25_AUDIO_NONE:
    SetCacheHints(CACHE_AUDIO_NONE, 0);
    break;

    // Map 2.5 CACHE_AUDIO_AUTO calls to CACHE_AUDIO_AUTO
  case CACHE_25_AUDIO_AUTO:
    SetCacheHints(CACHE_AUDIO_AUTO, frame_range);
    break;

    /*********************************************
    VIDEO
    *********************************************/

  case CACHE_SET_MIN_CAPACITY:
    hints.min = frame_range;
    ApplyHints(cachehints, frame_range);
    break;

  case CACHE_SET_MAX_CAPACITY:
    hints.max = frame_range;
    ApplyHints(cachehints, frame_range);
    break;

  case CACHE_GET_MIN_CAPACITY:
    return (int)hints.min;

  case CACHE_GET_MAX_CAPACITY:
    return (int)hints.max;

  case CACHE_GET_SIZE:
  case CACHE_GET_REQUESTED_CAP:
  case CACHE_GET_CAPACITY:
    return GetOrDefault(cachehints, frame_range, 0);

  case CACHE_GET_WINDOW: // Get the current window h_span.
  case CACHE_GET_RANGE: // Get the current generic frame range.
    return 2;
    break;

  case CACHE_GENERIC:
  case CACHE_FORCE_GENERIC:
  case CACHE_NOTHING:
  case CACHE_WINDOW:
  case CACHE_PREFETCH_FRAME:          // Queue request to prefetch frame N.
  case CACHE_PREFETCH_GO:             // Action video prefetches.
    break;

    /*********************************************
    AUDIO
    *********************************************/
  case CACHE_GETCHILD_AUDIO_MODE:
  case CACHE_GETCHILD_AUDIO_SIZE:
    return (child->GetVersion() >= 5) ? child->SetCacheHints(cachehints, 0) : 0;

  case CACHE_AUDIO:
  case CACHE_AUDIO_AUTO_START_ON: // auto mode, initially cache
  case CACHE_AUDIO_AUTO_START_OFF: // auto mode, initially don't cache
  case CACHE_AUDIO_NOTHING:
    hints.AudioPolicy = (CachePolicyHint)cachehints;
    hints.AudioSize = frame_range;
    ApplyHints(cachehints, frame_range);
    break;

  case CACHE_GET_AUDIO_POLICY: // Get the current audio policy.
    return hints.AudioPolicy;

  case CACHE_GET_AUDIO_SIZE: // Get the current audio cache size.
    return (child->GetVersion() >= 5) ? GetOrDefault(cachehints, frame_range, 0) : 0;

  case CACHE_PREFETCH_AUDIO_BEGIN:    // Begin queue request to prefetch audio (take critical section).
  case CACHE_PREFETCH_AUDIO_STARTLO:  // Set low 32 bits of start.
  case CACHE_PREFETCH_AUDIO_STARTHI:  // Set high 32 bits of start.
  case CACHE_PREFETCH_AUDIO_COUNT:    // Set low 32 bits of length.
  case CACHE_PREFETCH_AUDIO_COMMIT:   // Enqueue request transaction to prefetch audio (release critical section).
  case CACHE_PREFETCH_AUDIO_GO:       // Action audio prefetch
    break;

  case CACHE_GET_DEV_TYPE:
  case CACHE_GET_CHILD_DEV_TYPE:
    return (child->GetVersion() >= 5) ? child->SetCacheHints(cachehints, 0) : 0;

  default:
    return 0;
  }

  return 0;
}

AVSValue __cdecl CacheGuard::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  PClip p = 0;
  if (args.IsClip())
  {
    p = args.AsClip();
  }
  else if (args.IsArray() && args[0].IsClip())
  {
    p = args[0].AsClip();
  }
  const char* name = nullptr;
  if (args.IsArray() && args.ArraySize() >= 2 && args[1].IsString())
    name = args[1].AsString();

  if (p)  // If the child is a clip
  {
    if ( (p->GetVersion() >= 5)
      && (p->SetCacheHints(CACHE_DONT_CACHE_ME, 0) != 0) )
    {
      // Don't create cache instance if the child doesn't want to be cached
      // DONT_CACHE_ME is disabling audio cache as well, even if filter
      // would specify it by CACHE_GETCHILD_AUDIO_MODE
      return p; /* This is op, not args! */
    }
    else
    {
      return new CacheGuard(p, name, env);
    }
  }
  else
  {
    return args;
  }
}

bool __stdcall CacheGuard::IsCache(const PClip& p)
{
  return ((p->GetVersion() >= 5) && (p->SetCacheHints(CACHE_IS_CACHE_REQ, 0) == CACHE_IS_CACHE_ANS));
}
