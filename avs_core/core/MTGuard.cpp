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

#include "MTGuard.h"
#include "cache.h"
#include "internal.h"
#include "FilterConstructor.h"
#include "InternalEnvironment.h"
#include <cassert>
#include <mutex>

#ifdef X86_32
#include <mmintrin.h>
#endif

struct MTGuardChildFilter {
  PClip filter;
  std::mutex mutex;
};

MTGuard::MTGuard(PClip firstChild, MtMode mtmode, std::unique_ptr<const FilterConstructor>&& funcCtor, InternalEnvironment* env) :
  Env(env),
  nThreads(1),
#ifndef OLD_PREFETCH
  mt_enabled(false),
#endif
  FilterCtor(std::move(funcCtor)),
  MTMode(mtmode)
{
  assert( ((int)mtmode > (int)MT_INVALID) && ((int)mtmode < (int)MT_MODE_COUNT) );

  ChildFilters = std::unique_ptr<MTGuardChildFilter[]>(new MTGuardChildFilter[1]);
  ChildFilters[0].filter = firstChild;
  vi = ChildFilters[0].filter->GetVideoInfo();

  Env->ManageCache(MC_RegisterMTGuard, reinterpret_cast<void*>(this));
}

MTGuard::~MTGuard()
{
  Env->ManageCache(MC_UnRegisterMTGuard, reinterpret_cast<void*>(this));
}

void MTGuard::EnableMT(size_t nThreads)
{
  assert(nThreads >= 1);
#ifdef OLD_PREFETCH
  assert((nThreads & (nThreads - 1)) == 0); // must be 2^n
  // 2^N: needed because of directly accessing a masked array
  // PVideoFrame __stdcall MTGuard::GetFrame
  // envI->GetThreadId() & (nThreads - 1)
#endif

  if (nThreads > 1)
  {
    switch (MTMode)
    {
    case MT_NICE_FILTER:
    {
      // Nothing to do
      break;
    }
    case MT_MULTI_INSTANCE:
    {
      // creates the extra filter instances needed for the actual thread count
#ifdef OLD_PREFETCH
      if (this->nThreads < nThreads) {
#else
      // set only when unset
      if (!this->mt_enabled) {
#endif
        auto newchilds = std::unique_ptr<MTGuardChildFilter[]>(new MTGuardChildFilter[nThreads]);
        // copy existing
        for (size_t i = 0; i < this->nThreads; ++i) {
          newchilds[i].filter = ChildFilters[i].filter;
        }
        // create the rest
        for (size_t i = this->nThreads; i < nThreads; ++i) {
          newchilds[i].filter = FilterCtor->InstantiateFilter().AsClip();
        }
        ChildFilters = std::move(newchilds);
      }
      break;
    }
    case MT_SERIALIZED:
    {
      // Nothing to do
      break;
    }
    default:
    {
      assert(0);
      break;
    }
    }
  }

#ifdef OLD_PREFETCH
  this->nThreads = std::max(this->nThreads, nThreads);
#else
  if (!this->mt_enabled) {
    this->nThreads = std::max(this->nThreads, nThreads);
    this->mt_enabled = true;
  }
#endif

  // We don't need the stored parameters any more,
  // free their memory.
  //FilterCtor.reset();
}

PVideoFrame __stdcall MTGuard::GetFrame(int n, IScriptEnvironment* env)
{
  assert(nThreads > 0);
  /*
  // We can't call child filter without mutex guards even when nThreads is 1,
  // because a lately invoked filter may call GetFrame again, see
  // MT_SERIALIZED considerations below.
  // Code left here intentionally but commented out.
  if (nThreads == 1)
    return ChildFilters[0].filter->GetFrame(n, env);
  */
  InternalEnvironment* envI = static_cast<InternalEnvironment*>(env);
  PVideoFrame frame = NULL;

  switch (MTMode)
  {
  case MT_NICE_FILTER:
  {
    frame = ChildFilters[0].filter->GetFrame(n, env);
    break;
  }
  case MT_MULTI_INSTANCE:
  {
    auto myThreadID = envI->GetThreadId();
    // When called from thread pool then thread IDs are one-to-one mapped to ChildFilters array.
    // 'binary and' mask (old method) or 'modulo' method ensures that no over-addressing can happen.
    // The number of filter instances are created in EnableMT as such.
    // Thread pool thread IDs are consecutive numbers, starting with 1.
    // Prefetch threads are consecutive numbers, starting with 0.
    // Here we cannot differentiate from where was the filter called.
    // Note: 
    //   when called from Prefetcher (which has 'num_of_logical_processors' threads instead of nThreads)
    //   the mapping is not ideal. It can be less or more than the actual instance count.
    //   E.g. when mapping a thread on a CPU with 8 logical cores 
    //   to a instance count of 3 (nThread=3) the mapping is uneven (modulo example): [0,3,6]->[0] [1,4,7]->[1] [2,5]->[2]
#ifdef OLD_PREFETCH
    assert((nThreads & (nThreads - 1)) == 0); // must be 2^n
    size_t clipIndex = myThreadID & (nThreads - 1);
#else
    size_t clipIndex = envI->GetThreadId() % nThreads;
#endif
    auto& child = ChildFilters[clipIndex];
    std::lock_guard<std::mutex> lock(child.mutex);
    frame = child.filter->GetFrame(n, env);
    break;
  }
  case MT_SERIALIZED:
  {
    std::lock_guard<std::mutex> lock(ChildFilters[0].mutex);
    /*
    // Debug lines left here intentionally: allow maximum of 4000ms for obtaining the lock.
    // Invoke-GetFrame deadlock situation detection.
    // When GetFrame(0) was called from an Invoke (more exactly from a filter constructor to read
    // frame properties of 0th frame), which operated on a Clip AVSValue variable
    // resulted by a previous Eval operation (which had Prefetch inside).
    // Invoke put a lock a memory_mutex and called a GetFrame(0).
    // But the existing Prefetch of the Eval'd script had already running a GetFrame(0) from
    // another thread and thus would also like to obtain a lock on the common memory_mutex
    // (for expanding the frame registry).
    // This was solved in 3.7.2 by separating memory_mutex from invoke_mutex.
    // Solving part#2: mutex may be needed even for single threaded mode, see above.
    std::lock_guard<std::timed mutex> lock(ChildFilters[0].mutex);
    while(!ChildFilters[0].mutex.try_lock_for(std::chrono::milliseconds(4000)))
    {
      _RPT3(0, "MTGuard::GetFrame lock DEADLOCK %d w=%d p=%p\n", n, ChildFilters[0].filter->GetVideoInfo().width, (void*)&ChildFilters[0].mutex);
      envI->ThrowError("Deadlock!");
      break;
    }
    // possible frame registry expansion thus memory mutex requiration here:
    frame = ChildFilters[0].filter->GetFrame(n, env);
    ChildFilters[0].mutex.unlock();
    */
    frame = ChildFilters[0].filter->GetFrame(n, env);
    break;
  }
  default:
  {
    assert(0);
    envI->ThrowError("Invalid Avisynth logic.");
    break;
  }
  } // switch

#ifdef X86_32
  _mm_empty();
#endif

  return frame;
}

void __stdcall MTGuard::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{
  assert(nThreads > 0);

  if (nThreads == 1)
  {
    ChildFilters[0].filter->GetAudio(buf, start, count, env);
    return;
  }

  InternalEnvironment *envI = static_cast<InternalEnvironment*>(env);

  switch (MTMode)
  {
  case MT_NICE_FILTER:
    {
      ChildFilters[0].filter->GetAudio(buf, start, count, env);
      break;
    }
  case MT_MULTI_INSTANCE:
    {
      auto& child = ChildFilters[envI->GetThreadId() & (nThreads - 1)];
      std::lock_guard<std::mutex> lock(child.mutex);
      child.filter->GetAudio(buf, start, count, env);
      break;
    }
  case MT_SERIALIZED:
    {
      std::lock_guard<std::mutex> lock(ChildFilters[0].mutex);
      ChildFilters[0].filter->GetAudio(buf, start, count, env);
      break;
    }
  default:
    {
      assert(0);
      envI->ThrowError("Invalid Avisynth logic.");
      break;
    }
  } // switch

#ifdef X86_32
  _mm_empty();
#endif
}

const VideoInfo& __stdcall MTGuard::GetVideoInfo()
{
  return vi;
}

bool __stdcall MTGuard::GetParity(int n)
{
  return ChildFilters[0].filter->GetParity(n);
}

int __stdcall MTGuard::SetCacheHints(int cachehints, int frame_range)
{
  AVS_UNUSED(frame_range);
  if (CACHE_GET_MTMODE == cachehints) {
    return MT_NICE_FILTER;
  }
  if (CACHE_IS_MTGUARD_REQ == cachehints) {
    return CACHE_IS_MTGUARD_ANS;
  }
  if (CACHE_GET_DEV_TYPE == cachehints || CACHE_GET_CHILD_DEV_TYPE == cachehints) {
    return (ChildFilters[0].filter->GetVersion() >= 5) ? ChildFilters[0].filter->SetCacheHints(cachehints, 0) : 0;
  }

  return 0;
}

bool __stdcall MTGuard::IsMTGuard(const PClip& p)
{
  return ((p->GetVersion() >= 5) && (p->SetCacheHints(CACHE_IS_MTGUARD_REQ, 0) == CACHE_IS_MTGUARD_ANS));
}

PClip MTGuard::Create(MtMode mode, PClip filterInstance, std::unique_ptr<const FilterConstructor> funcCtor, InternalEnvironment* env)
{
    switch (mode)
    {
    case MT_NICE_FILTER:
    {
        // No need to wrap and protect this filter
        return filterInstance;
    }
    case MT_MULTI_INSTANCE: // Fall-through intentional
    {
        return new MTGuard(filterInstance, mode, std::move(funcCtor), env);
        // args2 and args3 are not valid after this point anymore
    }
    case MT_SERIALIZED:
    {
        // FIXME 2021: probably MT_SERIALIZED do not need this one, after USE_MT_GUARDEXIT concept failed...
        return new MTGuard(filterInstance, mode, NULL, env);
        // args2 and args3 are not valid after this point anymore
    }
    default:
        // There are broken plugins out there in the wild that have (GetVersion() >= 5), but still
        // return garbage for SetCacheHints(). However, this case should be recognized and
        // handled earlier, so we can never get to this default-branch. If we do, assume the worst.
        assert(0);
        return new MTGuard(filterInstance, MT_SERIALIZED, NULL, env);
    }
}

#ifdef USE_MT_GUARDEXIT
// 170531 Optimizing concept introduced in r2069 temporarily disabled by this define

// ---------------------------------------------------------------------
//                      MTGuardExit
// ---------------------------------------------------------------------

template <class T>
class reverse_lock {
public:
    reverse_lock(T *mutex) : mutex_(mutex) {
        if (mutex_) {
            mutex_->unlock();
        }
    }

    ~reverse_lock() {
        if (mutex_) {
            mutex_->lock();
        }
    }

    reverse_lock(const reverse_lock&) = delete;
    reverse_lock& operator=(const reverse_lock&) = delete;

private:
    T *mutex_;
};

MTGuardExit::MTGuardExit(const PClip &clip, const char *_name) :
    NonCachedGenericVideoFilter(clip),
  name(_name)
{}

void MTGuardExit::Activate(PClip &with_guard)
{
    assert(MTGuard::IsMTGuard(with_guard));
    this->guard = (MTGuard*)((void*)with_guard);
}

PVideoFrame __stdcall MTGuardExit::GetFrame(int n, IScriptEnvironment* env)
{
    std::mutex *m = (nullptr == guard) ? nullptr : guard->GetMutex();
#ifdef DEBUG
    if(nullptr != guard)
      _RPT3(0, "MTGuardExit::GetFrame %d name=%s (before unlock ) thread=%d\n", n, name.c_str(), GetCurrentThreadId());
#endif
    reverse_lock<std::mutex> unlock_guard(m);
#ifdef DEBUG
    if (nullptr != guard)
      _RPT3(0, "MTGuardExit::GetFrame %d name=%s (unlock ok     ) thread=%d\n", n, name.c_str(), GetCurrentThreadId());
#endif
    PVideoFrame result = child->GetFrame(n, env);
#ifdef DEBUG
    if (nullptr != guard)
      _RPT3(0, "MTGuardExit::GetFrame %d name=%s (lock again    ) thread=%d\n", n, name.c_str(), GetCurrentThreadId());
#endif
    // 170531. problem: in real life MTGuardExit unlocks and allows MT_SERIALIZED filters to be called again
    // even if they are still in work, make them to be called in a reentant way like in NICE_FILTER mode
    return result;
}

void __stdcall MTGuardExit::GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env)
{
    std::mutex *m = (nullptr == guard) ? nullptr : guard->GetMutex();
    reverse_lock<std::mutex> unlock_guard(m);
    return child->GetAudio(buf, start, count, env);
}
#endif
