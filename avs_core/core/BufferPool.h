#ifndef _AVS_BUFFERPOOL_H
#define _AVS_BUFFERPOOL_H

#include <map>

class InternalEnvironment;

class BufferPool
{
private:

  struct BufferDesc;
  typedef std::multimap<std::size_t, BufferDesc*> MapType;

  InternalEnvironment* Env;
  MapType Map;

  void* PrivateAlloc(std::size_t nBytes, std::size_t alignment, void* user);
  void PrivateFree(void* buffer);

public:

  BufferPool(InternalEnvironment* env);
  ~BufferPool();

  void* Allocate(std::size_t nBytes, std::size_t alignment, bool pool);
  void Free(void* ptr);

};

#endif  // _AVS_BUFFERPOOL_H
