#ifndef _AVS_BUFFERPOOL_H
#define _AVS_BUFFERPOOL_H

#include <map>
#include <avs/types.h>

class InternalEnvironment;

class BufferPool
{
private:

  struct BufferDesc;
  typedef std::multimap<size_t, BufferDesc*> MapType;

  InternalEnvironment* Env;
  MapType Map;

  void* PrivateAlloc(size_t nBytes, size_t alignment, void* user);
  void PrivateFree(void* buffer);

public:

  BufferPool(InternalEnvironment* env);
  ~BufferPool();

  void* Allocate(size_t nBytes, size_t alignment, bool pool);
  void Free(void* ptr);

};

#endif  // _AVS_BUFFERPOOL_H
