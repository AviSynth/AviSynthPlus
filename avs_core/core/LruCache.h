#ifndef AVS_LRUCACHE_H
#define AVS_LRUCACHE_H

#include <mutex>
#include <condition_variable>
#include <memory>
#include <cassert>
#include <list>
#include <unordered_map>
#include <algorithm>
#include "ObjectPool.h"
#include "SimpleLruCache.h"
#include "InternalEnvironment.h"

enum LruLookupResult
{
  LRU_LOOKUP_NOT_FOUND,          // Item has not been found, storage is reserved to be used by caller
  LRU_LOOKUP_FOUND_AND_READY,    // Item has been found and returned
  LRU_LOOKUP_FOUND_BUT_NOTAVAIL, // Item has been found, but is waiting for completion and is not yet ready
  LRU_LOOKUP_NO_CACHE            // Item will not be cached, no storage is returned
};

// ---------------------------------------------------------------------------------------------
// LruCache: the frame cache behind every AvsCache, plus the fixed size caches of the Prefetcher
// and the device transfer filters (CACHE_NO_RESIZE).
//
// A cache starts with capacity 0 and finds its own size from the requests it sees. Nobody tells
// it how many frames a temporal filter or a GUI will need again.
//
// Ghosts
//   Keys evicted from the cache are remembered (key only, no frame) in a ghost list, with a
//   counter 'ghosted' = how many times the key was evicted. A miss on a key that is still among
//   the ghosts means: one more slot would have turned this miss into a hit.
//   The ghost list is an LRU list with a hash index, capacity max(1024, 4 * capacity), so that a
//   cycle of a few hundred frames (AvsPmod bookmarks, a temporal filter with a large radius on a
//   few outputs) is still remembered while lookups stay O(1).
//
// Growth: limited by proven benefit
//   - A ghost hit on a key evicted at least once (ghosted >= 1) adds one slot, but at most
//     UNPROVEN_GROWTH slots in a row may be added this way ('unproven' counts them).
//   - A ghost hit on a key evicted at least twice (ghosted >= 2) proves steady reuse: the key keeps
//     coming back at about the same distance (bookmark cycles, ranges played again). It resets
//     'unproven', so growth can go on until the whole cycle fits.
//   - Patterns whose keys come back once, at an ever growing distance, never prove anything and stop
//     after UNPROVEN_GROWTH slots: stepping back after playing forward (#379), Bob/Merge mixes
//     (#270). Before, a per-key threshold had to choose between these and the bookmark case
//     (ghosted > 0: unbounded growth in #270/#379; ghosted > 1 (3.7.6): bookmarks never cached).
//
// Shrink
//   Over a window of max(SHRINK_WINDOW_MIN, 4 * capacity) misses the deepest hit position is
//   recorded. If every hit came from the front (deepest + SHRINK_SLACK < capacity), up to a quarter
//   of the slots is given back. The window counts misses, not lookups: only new keys age the cache,
//   hits in the front (a player re-reading its read-ahead) do not.
//   Evicted frames go back to the frame registry and are reused by the next NewVideoFrame; their
//   memory is only freed under memory pressure (ScriptEnvironment::ShrinkCache).
//
// Growth suppression for the caches below
//   When a cache grows for a ghost hit, it will keep that frame itself, so the caches below it in
//   the same GetFrame chain (same thread, env->GetSupressCaching(), restored by CacheStack) must not
//   grow for this request too; else one missing frame at the top makes the whole chain grow.
//   - CACHE_FAST_START (default): the caches below still store the frame at their current capacity
//     (a temporal filter below keeps its neighbour frames), they only do not grow.
//   - CACHE_OPTIMAL_SIZE (SetCacheMode): the caches below do not store it at all (Nekopanda's
//     original meaning, less memory, more recalculation).
//
// Memory limit
//   Independently of the above, ScriptEnvironment lowers the maximum capacity of the least recently
//   used caches when frame allocation hits SetMemoryMax, and raises it again by one per request
//   (MC_NodAndExpandCache) when a cache's requested capacity is above its allowed one.
// ---------------------------------------------------------------------------------------------
template<typename K, typename V>
class LruCache : public std::enable_shared_from_this<LruCache<K, V> >
{
private:
  enum LruEntryState
  {
    LRU_ENTRY_EMPTY,
    LRU_ENTRY_AVAILABLE,
    LRU_ENTRY_ROLLED_BACK
  };

  struct LruGhostEntry
  {
    K key;
    size_t ghosted;

    LruGhostEntry() :
      key(0), ghosted(0)
    {
    }

    LruGhostEntry(K key, size_t ghosted) :
      key(key), ghosted(ghosted)
    {
    }
  };

  struct LruEntry
  {
    K key;
    V value;
    size_t locks;      // the number of threads waiting on this entry. used to prevent eviction when readers are waiting on it
    size_t ghosted;    // the number of times this entry has entered the ghost list
    std::condition_variable ready_cond;
    enum LruEntryState state;

    LruEntry(const K& key)
    {
      reset(key, V());
    }

    void reset(const K& k, const V& v)
    {
      key = k;
      value = v;
      locks = 0;
      ghosted = 0;
      state = LRU_ENTRY_EMPTY;
    }

  private:
    LruEntry(const LruEntry&);
    LruEntry& operator=(const LruEntry&);
  };

  typedef LruEntry entry_type;
  typedef entry_type* entry_ptr;
  typedef SimpleLruCache<K, entry_ptr> CacheType;

  typedef size_t  size_type;

  static constexpr size_t GHOST_MIN_CAPACITY = 1024;
  static constexpr size_t UNPROVEN_GROWTH = 32;
  static constexpr size_t SHRINK_WINDOW_MIN = 64;
  static constexpr size_t SHRINK_SLACK = 8;

  CacheMode mode;
  CacheType MainCache;
  ObjectPool<entry_type> EntryPool;
  mutable std::mutex mutex;

  // ghost list: LRU order in the list, index for O(1) lookup
  std::list<LruGhostEntry> GhostList;
  std::unordered_map<K, typename std::list<LruGhostEntry>::iterator> GhostIndex;
  size_t GhostCapacity = GHOST_MIN_CAPACITY;

  size_t unproven = 0;          // slots added since the last proof of steady reuse
  size_t window_misses = 0;     // misses in the current shrink window
  size_t window_max_depth = 0;  // deepest hit position + 1 in the current shrink window

#ifdef AVS_CACHE_EXPERIMENT
public:
  // helper/cache_harness statistics, updated under 'mutex', read unlocked (approximate is fine)
  size_t stat_ghost_hits = 0; // miss in main cache, found among the ghosts
  size_t stat_grows = 0;      // capacity +1 steps
  size_t stat_evictions = 0;
  size_t stat_shrinks = 0;    // shrink steps
private:
#endif

  // like SimpleLruCache::lookup: found -> moved to front; not found -> new front entry (oldest dropped)
  LruGhostEntry* GhostLookup(const K& key, bool* found)
  {
    auto it = GhostIndex.find(key);
    if (it != GhostIndex.end()) {
      *found = true;
      GhostList.splice(GhostList.begin(), GhostList, it->second);
      return &GhostList.front();
    }
    *found = false;
    while (GhostList.size() >= GhostCapacity && !GhostList.empty()) {
      GhostIndex.erase(GhostList.back().key);
      GhostList.pop_back();
    }
    GhostList.emplace_front(key, 0);
    GhostIndex[key] = GhostList.begin();
    return &GhostList.front();
  }

  // follows the main capacity, called after every resize
  void GhostFitCapacity()
  {
    GhostCapacity = std::max(GHOST_MIN_CAPACITY, MainCache.capacity() * 4);
    while (GhostList.size() > GhostCapacity) {
      GhostIndex.erase(GhostList.back().key);
      GhostList.pop_back();
    }
  }

  // Called at the start of every non-suppressed lookup, under 'mutex'.
  // Closes the shrink window when it is full, see the class comment.
  void ShrinkTick()
  {
    const size_t cap = MainCache.capacity();
    if (window_misses < std::max<size_t>(SHRINK_WINDOW_MIN, cap * 4))
      return;
    if (mode != CACHE_NO_RESIZE) {
      const size_t need = window_max_depth + SHRINK_SLACK;
      if (need < cap) {
        const size_t target = std::max(need, cap - (cap + 3) / 4);
        MainCache.resize(target);
        GhostFitCapacity();
#ifdef AVS_CACHE_EXPERIMENT
        stat_shrinks++;
#endif
      }
    }
    window_misses = 0;
    window_max_depth = 0;
  }

  static bool MainEvictEvent(CacheType* cache, const typename CacheType::Entry& entry, void* userData)
  {
    if (entry.value->locks > 0)
      return false;

    LruCache* me = reinterpret_cast<LruCache*>(userData);
#ifdef AVS_CACHE_EXPERIMENT
    me->stat_evictions++;
#endif

    bool ghost_found;
    auto *g = me->GhostLookup(entry.key, &ghost_found);
    if (!ghost_found)
    {
      *g = LruGhostEntry(entry.key, entry.value->ghosted + 1);
    }
    else
    {
      g->ghosted++;
    }

    entry.value->reset(0, NULL);
    me->EntryPool.Destruct(entry.value);
    return true;
  }

public:

  typedef std::pair<entry_ptr, std::shared_ptr<LruCache> > handle;

  LruCache(size_type capacity, CacheMode mode) :
    mode(mode),
    MainCache(capacity, &MainEvictEvent, reinterpret_cast<void*>(this))
  {
  }

  size_type size() const
  {
    return MainCache.size();
  }

  size_t requested_capacity() const
  {
    return MainCache.requested_capacity();
  }

  size_t capacity() const
  {
    return MainCache.capacity();
  }

#ifdef AVS_CACHE_EXPERIMENT
  size_t ghost_capacity() const
  {
    return GhostCapacity;
  }
#endif

  void limits(size_t* min, size_t* max) const
  {
    std::unique_lock<std::mutex> global_lock(mutex);

    MainCache.limits(min, max);
  }

  void set_limits(size_t min, size_t max)
  {
    std::unique_lock<std::mutex> global_lock(mutex);

    MainCache.set_limits(min, max);
  }

  LruLookupResult lookup(const K& key, handle *hndl, bool block_for_completion, V& foundItem, bool* suppressCaching = nullptr)
  {
    // suppress: a cache above us in this GetFrame chain has just grown to keep this frame itself
    const bool suppress = (suppressCaching != nullptr) && *suppressCaching;
    hndl->first = nullptr; // clear handle

    std::unique_lock<std::mutex> global_lock(mutex);

    if (!suppress)
      ShrinkTick();

    bool found;
    size_t hit_pos = 0;
    // CACHE_OPTIMAL_SIZE: a suppressed request is not stored at all
    const bool lookup_only = suppress && mode == CACHE_OPTIMAL_SIZE;
    entry_ptr* entryp = MainCache.lookup(key, &found, lookup_only, &hit_pos);

    if (found)
    {
      if (!suppress)
        window_max_depth = std::max(window_max_depth, hit_pos + 1);
      entry_ptr entry = *entryp;

      if (!block_for_completion && (entry->state != LRU_ENTRY_AVAILABLE))
      {
        return LRU_LOOKUP_FOUND_BUT_NOTAVAIL;
      }

      // wait until data becomes available
      ++(entry->locks);
      while (entry->state == LRU_ENTRY_EMPTY)
      {
        entry->ready_cond.wait(global_lock);

        switch (entry->state)
        {
        case LRU_ENTRY_EMPTY:           // do nothing, spurious wakeup
          break;
        case LRU_ENTRY_AVAILABLE:       // finally, data available
          break;
        case LRU_ENTRY_ROLLED_BACK:     // whoever we were waiting for decided to step back. we take over his place.
          entry->state = LRU_ENTRY_EMPTY;
          *hndl = handle(entry, this->shared_from_this());
          return LRU_LOOKUP_NOT_FOUND;
        default:
          assert(0);
        }
      }
      // copy and return entry->value before releasing lock
      foundItem = entry->value;
      --(entry->locks);
      return LRU_LOOKUP_FOUND_AND_READY;
    }
    else if (suppress && entryp != NULL)
    {
      // CACHE_FAST_START, suppressed: store at the current capacity, no ghost bookkeeping, no growth
      *entryp = EntryPool.Construct(key);
      entry_ptr entry = *entryp;
      *hndl = handle(entry, this->shared_from_this());
      entry->locks = 1;
      entry->ghosted = 0;
      entry->value = NULL;
      return LRU_LOOKUP_NOT_FOUND;
    }
    else if (suppress == false)
    {
      // ghost: self-tuning caching algorithm, see the class comment
      bool ghost_found;
      auto *g = GhostLookup(key, &ghost_found);
      assert(g != NULL);
      window_misses++;
      if (!ghost_found)
      {
        *g = LruGhostEntry(key, 0);
      }
      else if (g->ghosted >= 1)
      {
#ifdef AVS_CACHE_EXPERIMENT
        stat_ghost_hits++;
#endif
        if (g->ghosted >= 2)
          unproven = 0; // steady reuse proven
        if (mode != CACHE_NO_RESIZE)
        {
          if (unproven < UNPROVEN_GROWTH)
          {
            MainCache.resize(MainCache.capacity() + 1);
            GhostFitCapacity();
            unproven++;
#ifdef AVS_CACHE_EXPERIMENT
            stat_grows++;
#endif
          }
          // this cache keeps the frame: the caches below must not grow for it as well
          if (suppressCaching != nullptr)
            *suppressCaching = true;
        }
      }

      if (entryp != NULL)
      {
        *entryp = EntryPool.Construct(key);
        entry_ptr entry = *entryp;
        *hndl = handle(entry, this->shared_from_this());
        entry->locks = 1;
        entry->ghosted = g->ghosted;
        entry->value = NULL;
        return LRU_LOOKUP_NOT_FOUND;
      }
      else
      {
        g->ghosted++;
        return LRU_LOOKUP_NO_CACHE;
      }
    } // if
    else {
      assert(entryp == nullptr);
      return LRU_LOOKUP_NO_CACHE;
    }
  }

  void commit_value(handle *hndl)
  {
    std::unique_lock<std::mutex> global_lock(mutex);

    // mark data as ready
    entry_ptr e = hndl->first;
    e->state = LRU_ENTRY_AVAILABLE;
    --(e->locks);

    // notify waiters
    global_lock.unlock();
    e->ready_cond.notify_all();

    hndl->second.reset();
  }

  void rollback(handle *hndl)
  {
    std::unique_lock<std::mutex> global_lock(mutex);

    entry_ptr e = hndl->first;
    assert(e->locks > 0);

    if (e->locks == 1)
    {
      MainCache.remove(e->key);
    }
    else
    {
      // others have started waiting for this data, so another one will have to take over
      --(e->locks);
      e->state = LRU_ENTRY_ROLLED_BACK;

      // notify one waiter
      global_lock.unlock();
      e->ready_cond.notify_one();
    }

    hndl->second.reset();
  }
};

#endif  // AVS_LRUCACHE_H
