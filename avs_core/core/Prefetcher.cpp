#include "Prefetcher.h"

#include <mutex>
#include <atomic>
#include <avisynth.h>
#include "ThreadPool.h"
#include "ObjectPool.h"
#include "LruCache.h"
#include "ScriptEnvironmentTLS.h"

struct PrefetcherJobParams
{
  int frame;
  Prefetcher* prefetcher;
  LruCache<size_t, PVideoFrame>::handle cache_handle;
};

struct PrefetcherPimpl
{
  PClip child;
  VideoInfo vi;

  // The number of threads to use for prefetching
  const size_t nThreads;

  // Maximum number of frames to prefetch
  const int nPrefetchFrames;

  ThreadPool ThreadPool;

  ObjectPool<PrefetcherJobParams> JobParamsPool;
  std::mutex params_pool_mutex;

  // Contains the pattern we are locked on to
  int LockedPattern;

  // The number of consecutive frames Pattern has repeated itself
  int PatternHits;

  // The number of consecutive frames LockedPattern was invalid
  int PatternMisses;

  // The current pattern that we are not locked on to
  int Pattern;

  // True if we have found a pattern to lock onto
  bool IsLocked;

  // The frame number that GetFrame() has been called with the last time
  int LastRequestedFrame;

  std::shared_ptr<LruCache<size_t, PVideoFrame> > VideoCache;
  std::atomic<int> running_workers;  
  std::mutex worker_exception_mutex;
  std::exception_ptr worker_exception;
  bool worker_exception_present;

  PrefetcherPimpl(const PClip& _child, size_t _nThreads) :
    child(_child),
    vi(_child->GetVideoInfo()),
    nThreads(_nThreads),
    nPrefetchFrames(_nThreads * 2),
    ThreadPool(_nThreads),
    LockedPattern(1),
    PatternHits(0),
    Pattern(1),
    LastRequestedFrame(0),
    VideoCache(NULL),
    running_workers(0),
    worker_exception_present(0),
    IsLocked(false),
    PatternMisses(0)
  {
  }
};


// The number of intervals a pattern has to repeat itself to become (un)locked
#define PATTERN_LOCK_LENGTH 3

AVSValue Prefetcher::ThreadWorker(IScriptEnvironment2* env, void* data)
{
  PrefetcherJobParams *ptr = (PrefetcherJobParams*)data;
  Prefetcher *prefetcher = ptr->prefetcher;
  int n = ptr->frame;
  LruCache<size_t, PVideoFrame>::handle cache_handle = ptr->cache_handle;

  {
    std::lock_guard<std::mutex> lock(prefetcher->_pimpl->params_pool_mutex);
    prefetcher->_pimpl->JobParamsPool.Destruct(ptr);
  }

  try
  {
    cache_handle.first->value = prefetcher->_pimpl->child->GetFrame(n, env);
    #ifdef X86_32
          _mm_empty();
    #endif

    prefetcher->_pimpl->VideoCache->commit_value(&cache_handle);
    --(prefetcher->_pimpl->running_workers);
  }
  catch(...)
  {
    prefetcher->_pimpl->VideoCache->rollback(&cache_handle);

    std::lock_guard<std::mutex> lock(prefetcher->_pimpl->worker_exception_mutex);
    prefetcher->_pimpl->worker_exception = std::current_exception();
    prefetcher->_pimpl->worker_exception_present = true;
    --(prefetcher->_pimpl->running_workers);
  }

  return AVSValue();
}

Prefetcher::Prefetcher(const PClip& _child, size_t _nThreads, IScriptEnvironment2 *env) :
  _pimpl(NULL)
{
  _pimpl = new PrefetcherPimpl(_child, _nThreads);
  _pimpl->VideoCache = std::make_shared<LruCache<size_t, PVideoFrame> >(_pimpl->nPrefetchFrames*2);
}

Prefetcher::~Prefetcher()
{
  while(_pimpl->running_workers > 0);
  delete _pimpl;
}

size_t Prefetcher::NumPrefetchThreads() const
{
  return _pimpl->nThreads;
}

int __stdcall Prefetcher::SchedulePrefetch(int current_n, int prefetch_start, IScriptEnvironment2* env)
{
  int n = prefetch_start;
  while ((_pimpl->running_workers < _pimpl->nPrefetchFrames) && (std::abs(n - current_n) < _pimpl->nPrefetchFrames) )
  {
    n += _pimpl->IsLocked ? _pimpl->LockedPattern : 1;
    if (n >= _pimpl->vi.num_frames)
      break;

    PVideoFrame result;
    LruCache<size_t, PVideoFrame>::handle cache_handle;
    switch(_pimpl->VideoCache->lookup(n, &cache_handle, false))
    {
    case LRU_LOOKUP_NOT_FOUND:
      {
        PrefetcherJobParams *p = NULL;
        {
          std::lock_guard<std::mutex> lock(_pimpl->params_pool_mutex);
          p = _pimpl->JobParamsPool.Construct();
        }
        p->frame = n;
        p->prefetcher = this;
        p->cache_handle = cache_handle;
        ++_pimpl->running_workers;
        _pimpl->ThreadPool.QueueJob(ThreadWorker, p, env, NULL);
        break;
      }
    case LRU_LOOKUP_FOUND_AND_READY:      // Fall-through intentional
    case LRU_LOOKUP_NO_CACHE:             // Fall-through intentional
    case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:
      {
        break;
      }
    default:
      {
        assert(0);
        break;
      }
    }
  } // switch

  return n;
}

PVideoFrame __stdcall Prefetcher::GetFrame(int n, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);

  int pattern = n - _pimpl->LastRequestedFrame;
  _pimpl->LastRequestedFrame = n;
  if (pattern == 0)
    pattern = 1;

  if (_pimpl->IsLocked)
  {
    if (_pimpl->LockedPattern == pattern)
    {
      _pimpl->PatternHits = 0;    // Tracks Pattern
      _pimpl->PatternMisses = 0;  // Tracks LockedPattern
    }
    else if (_pimpl->Pattern == pattern)
    {
      _pimpl->PatternHits++;    // Tracks Pattern
      _pimpl->PatternMisses++;  // Tracks LockedPattern
    }
    else
    {
      _pimpl->PatternHits = 0;  // Tracks Pattern
      _pimpl->PatternMisses++;  // Tracks LockedPattern
    }
    _pimpl->Pattern = pattern;

    if ((_pimpl->PatternMisses >= PATTERN_LOCK_LENGTH) && (_pimpl->PatternHits >= PATTERN_LOCK_LENGTH))
    {
      _pimpl->LockedPattern = _pimpl->Pattern;
      _pimpl->PatternHits = 0;    // Tracks Pattern
      _pimpl->PatternMisses = 0;  // Tracks LockedPattern
    }
    else if ((_pimpl->PatternMisses >= PATTERN_LOCK_LENGTH) && (_pimpl->PatternHits < PATTERN_LOCK_LENGTH))
    {
      _pimpl->IsLocked = false;
    }
  }
  else
  {
    if (_pimpl->Pattern == pattern)
    {
      _pimpl->PatternHits++;      // Tracks Pattern
      _pimpl->PatternMisses = 0;  // Tracks Pattern
    }
    else
    {
      _pimpl->PatternHits = 0;    // Tracks Pattern
      _pimpl->PatternMisses++;    // Tracks Pattern
    }

    if (_pimpl->PatternHits >= PATTERN_LOCK_LENGTH)
    {
      _pimpl->LockedPattern = pattern;
      _pimpl->PatternMisses = 0;  // Tracks Pattern
      _pimpl->IsLocked = true;
    }
  }


  {
    std::lock_guard<std::mutex> lock(_pimpl->worker_exception_mutex);
    if (_pimpl->worker_exception_present)
    {
      std::rethrow_exception(_pimpl->worker_exception);
    }
  }


  // Prefetch 1
  size_t scheduled_Frames = 0;
  int prefetch_pos = SchedulePrefetch(n, n, env2);

  // Get requested frame
  PVideoFrame result;
  LruCache<size_t, PVideoFrame>::handle cache_handle;
  switch(_pimpl->VideoCache->lookup(n, &cache_handle, true))
  {
  case LRU_LOOKUP_NOT_FOUND:
    {
      try
      {
        cache_handle.first->value = _pimpl->child->GetFrame(n, env);
  #ifdef X86_32
        _mm_empty();
  #endif
        _pimpl->VideoCache->commit_value(&cache_handle);
      }
      catch(...)
      {
        _pimpl->VideoCache->rollback(&cache_handle);
        throw;
      }
      result = cache_handle.first->value;
      break;
    }
  case LRU_LOOKUP_FOUND_AND_READY:
    {
      result = cache_handle.first->value;
      break;
    }
  case LRU_LOOKUP_NO_CACHE:
    {
      result = _pimpl->child->GetFrame(n, env);
      break;
    }
  case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:    // Fall-through intentional
  default:
    {
      assert(0);
      break;
    }
  }

  // Prefetch 2
  SchedulePrefetch(n, prefetch_pos, env2);

  return result;
}

bool __stdcall Prefetcher::GetParity(int n)
{
  return _pimpl->child->GetParity(n);
}

void __stdcall Prefetcher::GetAudio(void* buf, __int64 start, __int64 count, IScriptEnvironment* env)
{
  _pimpl->child->GetAudio(buf, start, count, env);
}

int __stdcall Prefetcher::SetCacheHints(int cachehints, int frame_range)
{
  return _pimpl->child->SetCacheHints(cachehints, frame_range);
}

const VideoInfo& __stdcall Prefetcher::GetVideoInfo()
{
  return _pimpl->vi;
}

AVSValue Prefetcher::Create(AVSValue args, void*, IScriptEnvironment* env)
{
  IScriptEnvironment2 *env2 = static_cast<IScriptEnvironment2*>(env);
  PClip child = args[0].AsClip();

  int PrefetchThreads = args[1].AsInt(env2->GetProperty(AEP_PHYSICAL_CPUS)+1);
  
  if (PrefetchThreads > 0)
  {
    Prefetcher* prefetcher = new Prefetcher(child, PrefetchThreads, env2);
    try
    {
      env2->SetPrefetcher(prefetcher);
      return prefetcher;
    }
    catch(...)
    {
      delete prefetcher;
      throw;
    }
  }
  else
    return child;
}