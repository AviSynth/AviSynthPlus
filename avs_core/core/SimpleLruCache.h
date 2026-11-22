#ifndef AVS_SIMPLELRUCACHE_H
#define AVS_SIMPLELRUCACHE_H

#include <list>
#include <functional>
#include <limits>
#include <avs/minmax.h>

template<typename K, typename V>
class SimpleLruCache
{
public:
  struct Entry
  {
    K key;
    V value;

    Entry(const K& k) :
      key(k)
    {}
  };

  typedef V value_type;
  typedef K key_type;
  typedef typename std::list<Entry>::iterator entry_type;

  typedef std::function<bool(SimpleLruCache*, const Entry&, void*)> EvictEventType;

private:
  size_t MinCapacity;
  size_t MaxCapacity;
  size_t RequestedCapacity;
  size_t RealCapacity;
  std::list<Entry> Cache;
  std::list<Entry> Pool;

  void* EventUserData;
  const EvictEventType EvictEvent;

public:
  SimpleLruCache(size_t capacity, const EvictEventType&& evict, void* evData) :
    MinCapacity(0),
    MaxCapacity(std::numeric_limits<size_t>::max()),
    RequestedCapacity(capacity),
    RealCapacity(capacity),
    EventUserData(evData),
    EvictEvent(evict)
  {
  }

  size_t size() const
  {
    return Cache.size();
  }

  size_t requested_capacity() const
  {
    return RequestedCapacity;
  }

  size_t capacity() const
  {
    return RealCapacity;
  }

  void limits(size_t* min, size_t* max) const
  {
    *min = MinCapacity;
    *max = MaxCapacity;
  }

  void set_limits(size_t min, size_t max)
  {
    MinCapacity = min;
    MaxCapacity = max;
    resize(RequestedCapacity);
  }

  // hit_pos (optional): position of the found entry before moving it to the front, 0 = most recent
  V* lookup(const K& key, bool *found, bool lookuponly = false, size_t* hit_pos = nullptr)
  {
    // Look for an existing cache entry,
    // and return it when found
    const entry_type it_end =  Cache.end();
    size_t pos = 0;
    for (
      entry_type it = Cache.begin();
      it != it_end;
      ++it, ++pos
    )
    {
      if (it->key == key)
      {
        if (hit_pos)
          *hit_pos = pos;
        // Move found element to the front of the list
        if (it != Cache.begin())
          Cache.splice(Cache.begin(), Cache, it);

        *found = true;
        return &(Cache.front().value);
      }
    }

    // Nothing found
    *found = false;

		if (lookuponly) {
			return NULL;
		}

    // Evict an old element if the cache is full
    trim();

    if (RealCapacity != 0)
    {
      // See if we can take one from our pool
      if (!Pool.empty())
      {
        Cache.splice(Cache.begin(), Pool, Pool.begin());
        Cache.front().key = key;
      }
      else
      {
        Cache.emplace_front(key);
      }

      return &(Cache.front().value);
    }

    // Return NULL-storage if we cannot store anything
    return NULL;
  }

  void remove(const K& key)
  {
    const entry_type it_end =  Cache.end();
    for (
      entry_type it = Cache.begin();
      it != it_end;
      ++it
    )
    {
      if (it->key == key)
      {
        Pool.splice(Pool.begin(), Cache, it);
        break;
      }
    }
  }

  void trim()
  {
    if (Cache.size() > RealCapacity)
    {
      size_t nItemsToDelete = Cache.size() - RealCapacity;
      auto it = --Cache.end();
      for (size_t i = 0; i < nItemsToDelete; ++i)
      {
        auto prev_it = it;
        bool end = (it == Cache.begin());
        if (!end)
        {
          prev_it = it;
          --prev_it;
        }

        if (EvictEvent != NULL)
        {
          if (EvictEvent(this, *it, EventUserData))
          {
            Pool.splice(Pool.begin(), Cache, it);
          }
        }
        else
        {
          // TODO: Do we want the consumer to always define EvictItem?
          Pool.splice(Pool.begin(), Cache, it);
        }

        if (!end)
          it = prev_it;
      }
    }
  }

  void resize(size_t new_cap)
  {
    RequestedCapacity = new_cap;
    RealCapacity = clamp(RequestedCapacity, MinCapacity, MaxCapacity);
    trim();
  }
};

#endif // AVS_SIMPLELRUCACHE_H
