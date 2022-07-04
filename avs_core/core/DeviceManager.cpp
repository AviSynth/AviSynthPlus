
#include "DeviceManager.h"
#include "internal.h"
#include "InternalEnvironment.h"
#include <avs/minmax.h>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include "LruCache.h"
#include "ThreadPool.h"
#include "AVSMap.h"
#include "parser/scriptparser.h"

#ifndef MINGW_HAS_SECURE_API
#define sprintf_s sprintf
#endif

#define ENABLE_CUDA_COMPUTE_STREAM 0

#ifdef ENABLE_CUDA

#include <cuda_runtime_api.h>

#endif // #ifdef ENABLE_CUDA

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err__ = call; \
    if (err__ != cudaSuccess) { \
      env->ThrowError("[CUDA Error] %d: %s @%d", err__, cudaGetErrorString(err__), __LINE__); \
    } \
  } while (0)

int GetDeviceTypes(const PClip& child)
{
  if (child->GetVersion() < 5) {
    return DEV_TYPE_CPU;
  }
  int deviceflags = child->SetCacheHints(CACHE_GET_DEV_TYPE, 0);
  if (deviceflags == 0) {
    // if not implement CACHE_GET_DEVICE_TYPE, we assume CPU only filter.
    deviceflags = DEV_TYPE_CPU;
  }
  return deviceflags;
}

int GetTargetDeviceTypes(const PClip& clip)
{
  if (clip->GetVersion() < 5) {
    return DEV_TYPE_CPU;
  }
  int deviceflags = clip->SetCacheHints(CACHE_GET_CHILD_DEV_TYPE, 0);
  if (deviceflags == 0) {
    deviceflags = clip->SetCacheHints(CACHE_GET_DEV_TYPE, 0);
    if (deviceflags == 0) {
      // if not implement CACHE_GET_DEVICE_TYPE, we assume CPU only filter.
      deviceflags = DEV_TYPE_CPU;
    }
  }
  return deviceflags;
}

std::string DeviceTypesString(int devicetypes)
{
  std::vector<const char*> typesstr;
  if (devicetypes & DEV_TYPE_CPU) {
    typesstr.push_back("CPU");
  }
  if (devicetypes & DEV_TYPE_CUDA) {
    typesstr.push_back("CUDA");
  }
  std::ostringstream oss;
  for (int i = 0; i < (int)typesstr.size(); ++i) {
    if (i > 0) oss << ",";
    oss << typesstr[i];
  }
  return oss.str();
}

static void CheckDeviceTypes(const char* name, int devicetypes, const AVSValue& last, const AVSValue& arr, InternalEnvironment* env)
{
  for (int i = -1; i < arr.ArraySize(); ++i) {
    const AVSValue& val = (i == -1) ? last : arr[i];
    if (val.IsClip()) {
      int childtypes = GetDeviceTypes(val.AsClip());
      if ((devicetypes & childtypes) == 0) {
        std::string parentdevstr = DeviceTypesString(devicetypes);
        std::string childdevstr = DeviceTypesString(childtypes);
        // e.g.. Device unmatch: XYfilter[CPU] does not support [CUDA] frame
        env->ThrowError(
          "Device unmatch: %s[%s] does not support [%s] frame",
          name, parentdevstr.c_str(), childdevstr.c_str());
      }
    }
    else if (val.IsArray()) {
      CheckDeviceTypes(name, devicetypes, AVSValue(), val, env);
    }
  }
}

void CheckChildDeviceTypes(const PClip& clip, const char* name, const AVSValue& last, const AVSValue& args, const char* const* argnames, InternalEnvironment* env)
{
  int deviceflags = GetTargetDeviceTypes(clip);
  if (args.IsArray()) {
    CheckDeviceTypes(name, deviceflags, last, args, env);
  }
  else {
    CheckDeviceTypes(name, deviceflags, last, AVSValue(&args, 1), env);
  }
}

size_t GetFrameHead(const PVideoFrame& vf)
{
  int head = vf->GetOffset();
  if (vf->GetPitch(PLANAR_U)) {
    head = min(head, vf->GetOffset(PLANAR_U));
  }
  if (vf->GetPitch(PLANAR_V)) {
    head = min(head, vf->GetOffset(PLANAR_V));
  }
  if (vf->GetPitch(PLANAR_A)) {
    head = min(head, vf->GetOffset(PLANAR_A));
  }
  return head;
}

size_t GetFrameTail(const PVideoFrame& vf)
{
  int tail = vf->GetOffset() + vf->GetPitch() * vf->GetHeight();
  if (vf->GetPitch(PLANAR_U)) {
    tail = max(tail, vf->GetOffset(PLANAR_U) + vf->GetPitch(PLANAR_U) * vf->GetHeight(PLANAR_U));
  }
  if (vf->GetPitch(PLANAR_V)) {
    tail = max(tail, vf->GetOffset(PLANAR_V) + vf->GetPitch(PLANAR_V) * vf->GetHeight(PLANAR_V));
  }
  if (vf->GetPitch(PLANAR_A)) {
    tail = max(tail, vf->GetOffset(PLANAR_A) + vf->GetPitch(PLANAR_A) * vf->GetHeight(PLANAR_A));
  }
  return min(tail, vf->GetFrameBuffer()->GetDataSize());
}

class CPUDevice : public Device {
public:
  CPUDevice(InternalEnvironment* env)
    : Device(DEV_TYPE_CPU, 0, 0, env)
  { }

  virtual int SetMemoryMax(int mem)
  {
    // memory_max for CPU device is not implemented here.
    env->ThrowError("Not implemented ...");
    return 0;
  }

  virtual BYTE* Allocate(size_t size, int margin)
  {
    size += margin;
#ifdef _DEBUG
    BYTE* data = new BYTE[size + 16];
    int *pInt = (int *)(data + size);
    pInt[0] = 0xDEADBEEF;
    pInt[1] = 0xDEADBEEF;
    pInt[2] = 0xDEADBEEF;
    pInt[3] = 0xDEADBEEF;

    static const BYTE filler[] = { 0x0A, 0x11, 0x0C, 0xA7, 0xED };
    BYTE* pByte = data;
    BYTE* q = pByte + size / 5 * 5;
    for (; pByte < q; pByte += 5)
    {
      pByte[0] = filler[0];
      pByte[1] = filler[1];
      pByte[2] = filler[2];
      pByte[3] = filler[3];
      pByte[4] = filler[4];
    }
    return data;
#else
    return new BYTE[size + 16];
#endif
  }

  virtual void Free(BYTE* ptr)
  {
    if (ptr != nullptr) {
      delete[] ptr;
    }
  }

  virtual const char* GetName() const { return "CPU"; }

  virtual void AddCompleteCallback(DeviceCompleteCallbackData cbdata)
  {
    // no need to delay, call immediately
    cbdata.cb(cbdata.user_data);
  }

  virtual std::unique_ptr<std::vector<DeviceCompleteCallbackData>> GetAndClearCallbacks()
  {
    return nullptr;
  }

  virtual void SetActiveToCurrentThread(InternalEnvironment* env)
  {
    // do nothing
  }

  virtual void* GetComputeStream()
  {
    return nullptr;
  }

  virtual void SetDeviceOpt(DeviceOpt opt, int val, InternalEnvironment* env)
  {
    // do nothing
  }

  virtual void GetAlignmentRequirement(int* memoryAlignment, int* pitchAlignment)
  {
    *memoryAlignment = FRAME_ALIGN;
    *pitchAlignment = FRAME_ALIGN;
  }
};

#ifdef ENABLE_CUDA
class CUDACPUDevice : public CPUDevice {
  int num_cuda_devices;
  bool allocated;
  bool enable_pinned;
public:
  CUDACPUDevice(InternalEnvironment* env, int num_cuda_devices)
    : CPUDevice(env)
    , num_cuda_devices(num_cuda_devices)
    , allocated()
    , enable_pinned()
  { }

  virtual BYTE* Allocate(size_t size, int margin)
  {
    allocated = true;
    if (!enable_pinned) {
      return CPUDevice::Allocate(size, margin);
    }
    //unsigned int flags = (prop.canMapHostMemory && prop.unifiedAddressing)
    //  ? cudaHostAllocMapped : cudaHostAllocDefault;
    // Without portable flag, allocated memory is associated with one cuda context,
    // i.e. one GPU. portable flag is required to use this memory with all GPUs.
    unsigned int flags = (num_cuda_devices > 1)
      ? cudaHostAllocPortable : cudaHostAllocDefault;
    BYTE* data = nullptr;
    size += margin;
#ifdef _DEBUG
    CUDA_CHECK(cudaHostAlloc((void**)&data, size + 16, flags));
    int *pInt = (int *)(data + size);
    pInt[0] = 0xDEADBEEF;
    pInt[1] = 0xDEADBEEF;
    pInt[2] = 0xDEADBEEF;
    pInt[3] = 0xDEADBEEF;

    static const BYTE filler[] = { 0x0A, 0x11, 0x0C, 0xA7, 0xED };
    BYTE* pByte = data;
    BYTE* q = pByte + size / 5 * 5;
    for (; pByte < q; pByte += 5)
    {
      pByte[0] = filler[0];
      pByte[1] = filler[1];
      pByte[2] = filler[2];
      pByte[3] = filler[3];
      pByte[4] = filler[4];
    }
#else
    CUDA_CHECK(cudaHostAlloc((void**)&data, size + 16, flags));
#endif
    return data;
  }

  virtual void Free(BYTE* ptr)
  {
    if (!enable_pinned) {
      CPUDevice::Free(ptr);
    }
    else {
      if (ptr != nullptr) {
        CUDA_CHECK(cudaFreeHost(ptr));
      }
    }
  }

  virtual const char* GetName() const { return "CPU(CUDAAware)"; }

  void SetDeviceOpt(DeviceOpt opt, int val, InternalEnvironment* env)
  {
    if (opt == DEV_CUDA_PINNED_HOST) {
      if (!enable_pinned) {
        if (allocated) {
          env->ThrowError("SetDeviceOpt: Allocation mode change must be done before any frame allocation.");
        }
        enable_pinned = true;
      }
    }
    if (opt == DEV_FREE_THRESHOLD) {
      free_thresh = val;
    }
  }

  virtual void GetAlignmentRequirement(int* memoryAlignment, int* pitchAlignment)
  {
    *memoryAlignment = FRAME_ALIGN;
    *pitchAlignment = FRAME_ALIGN;
  }
};
#endif

#ifdef ENABLE_CUDA
class CUDADevice : public Device {
  class ScopedCUDADevice
  {
    int old_device;
    int tgt_device;
  public:
    ScopedCUDADevice(int device_index, IScriptEnvironment* env)
      : tgt_device(device_index)
    {
      CUDA_CHECK(cudaGetDevice(&old_device));
      if (tgt_device != old_device) {
        CUDA_CHECK(cudaSetDevice(tgt_device));
      }
    }
    ~ScopedCUDADevice()
    {
      if (tgt_device != old_device) {
        cudaSetDevice(old_device);
      }
    }
  };

  char name[32];

  cudaDeviceProp prop;

  std::mutex mutex;
  std::vector<DeviceCompleteCallbackData> callbacks;

#if ENABLE_CUDA_COMPUTE_STREAM
  cudaStream_t computeStream;
  cudaEvent_t computeEvent;
#endif

public:
  CUDADevice(int id, int n, InternalEnvironment* env) :
    Device(DEV_TYPE_CUDA, id, n, env)
  {
    sprintf_s(name, "CUDA %d", n);

    ScopedCUDADevice d(device_index, env);

    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));

    SetMemoryMax(768); // start with 768MB
#if ENABLE_CUDA_COMPUTE_STREAM
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaEventCreate(&computeEvent));
#endif
  }

  ~CUDADevice()
  {
#if ENABLE_CUDA_COMPUTE_STREAM
    cudaStreamDestroy(computeStream);
    cudaEventDestroy(computeEvent);
#endif
  }

  virtual int SetMemoryMax(int mem)
  {
    if (mem > 0) {
      uint64_t requested = mem * 1048576ull;
      uint64_t mem_limit = prop.totalGlobalMem;
      memory_max = clamp(requested, (uint64_t)(64 * 1024 * 1024ull), (uint64_t)(mem_limit - 128 * 1024 * 1024ull));
    }
    return (int)(memory_max / 1048576ull);
  }

  virtual BYTE* Allocate(size_t size, int margin)
  {
    BYTE* data = nullptr;
    ScopedCUDADevice d(device_index, env);
    CUDA_CHECK(cudaMalloc((void**)&data, size));
    return data;
  }

  virtual void Free(BYTE* ptr)
  {
    if (ptr != NULL) {
      ScopedCUDADevice d(device_index, env);
      CUDA_CHECK(cudaFree(ptr));
    }
  }

  virtual const char* GetName() const { return name; }

  virtual void AddCompleteCallback(DeviceCompleteCallbackData cbdata)
  {
    std::lock_guard<std::mutex> lock(mutex);

    callbacks.push_back(cbdata);
  }

  virtual std::unique_ptr<std::vector<DeviceCompleteCallbackData>> GetAndClearCallbacks()
  {
    std::lock_guard<std::mutex> lock(mutex);

    if (callbacks.size() > 0) {
      auto *ret = new std::vector<DeviceCompleteCallbackData>(std::move(callbacks));
      callbacks.clear();
      return std::unique_ptr<std::vector<DeviceCompleteCallbackData>>(ret);
    }

    return nullptr;
  }

  virtual void SetActiveToCurrentThread(InternalEnvironment* env)
  {
    CUDA_CHECK(cudaSetDevice(device_index));
  }

  virtual void* GetComputeStream() {
#if ENABLE_CUDA_COMPUTE_STREAM
    return computeStream;
#else
    return nullptr;
#endif
  }

  void MakeStreamWaitCompute(cudaStream_t stream, InternalEnvironment* env)
  {
#if ENABLE_CUDA_COMPUTE_STREAM
    std::lock_guard<std::mutex> lock(mutex);

    CUDA_CHECK(cudaEventRecord(computeEvent, computeStream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, computeEvent, 0));
#endif
  }

  void SetDeviceOpt(DeviceOpt opt, int val, InternalEnvironment* env) {
    if (opt == DEV_FREE_THRESHOLD) {
      free_thresh = val;
    }
  }

  virtual void GetAlignmentRequirement(int* memoryAlignment, int* pitchAlignment)
  {
    *memoryAlignment = prop.textureAlignment;
    *pitchAlignment = prop.texturePitchAlignment;
  }
};
#endif

DeviceManager::DeviceManager(InternalEnvironment* env) :
  env(env)
{
  // 0 is CPU device, start from 1 for other devices.
  int next_device_id = 1;

#ifdef ENABLE_CUDA
  int cuda_device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&cuda_device_count);
  if (status == cudaSuccess) {
    // _RPT0(0, "cudaGetDeviceCount = %d\r\n", cuda_device_count);
    for (int i = 0; i < cuda_device_count; ++i) {
      cudaDevices.emplace_back(new CUDADevice(next_device_id++, i, env));
    }
  }
  else {
    if (status == cudaErrorInitializationError) {
      // We probably get this error because of current Nvidia driver version is lower than
      // the minimum required version by the used CUDA SDK, or a Computing Capability is too low.
      // Example: GTX460: Latest driver is 391.35, latest supporting CUDA SDK is 9.1.85
      // Consequence: since SDK 9 is unsupported in VS2019, no GTX460 support
      // As of Jan.2021 Avisynth+ is is using CUDA Toolkit version 11.2.
      int version;
      status = cudaRuntimeGetVersion(&version);
      if(status == cudaSuccess)
        _RPT1(0, "cudaGetDeviceCount: cudaErrorInitializationError!\r\nMaybe CUDA Runtime version (%d) does not support old drivers. \r\n", version);
      else
        _RPT0(0, "cudaGetDeviceCount: cudaErrorInitializationError! Runtime version request failed\r\n");
    }
    else {
      _RPT1(0, "cudaGetDeviceCount failed (%d)\r\n", (int)status);
    }
  }
  // do not modify CUDADevices after this since it causes pointer change

  cpuDevice = std::unique_ptr<Device>((cuda_device_count > 0)
    ? new CUDACPUDevice(env, cuda_device_count)
    : new CPUDevice(env));
#else // ENABLE_CUDA
  cpuDevice = std::unique_ptr<CPUDevice>(new CPUDevice(env));
#endif // ENABLE_CUDA

  numDevices = next_device_id;
}

Device* DeviceManager::GetDevice(AvsDeviceType device_type, int device_index) const
{
  switch (device_type) {

  case DEV_TYPE_CPU:
    return cpuDevice.get();

  case DEV_TYPE_CUDA:
#ifdef ENABLE_CUDA
    if (device_index < 0) {
      env->ThrowError("Invalid device index %d", device_index);
    }
    if (cudaDevices.size() == 0) {
      env->ThrowError("No CUDA devices ...");
    }
    // wrap index
    device_index %= (int)cudaDevices.size();
    return cudaDevices[device_index].get();
#else
    env->ThrowError("This Avisynth does not support memory type %d (CUDA)", device_type);
#endif

  default:
    env->ThrowError("Not supported memory type %d", device_type);
  }
  return nullptr;
}

int DeviceManager::GetNumDevices(AvsDeviceType device_type) const
{
  switch (device_type) {

  case DEV_TYPE_CPU:
    return 1;

  case DEV_TYPE_CUDA:
    return (int)cudaDevices.size();

  default:
    env->ThrowError("Not supported memory type %d", device_type);
  }
  return 0;
}

void DeviceManager::SetDeviceOpt(DeviceOpt opt, int val, InternalEnvironment* env)
{
  cpuDevice->SetDeviceOpt(opt, val, env);
#ifdef ENABLE_CUDA
  for (auto& dev : cudaDevices) {
    dev->SetDeviceOpt(opt, val, env);
  }
#endif // #ifdef ENABLE_CUDA
}

DeviceSetter::DeviceSetter(InternalEnvironment* env, Device* upstreamDevice)
  : env(env)
{
  downstreamDevice = env->SetCurrentDevice(upstreamDevice);
  if (downstreamDevice == nullptr) {
    env->ThrowError("This thread is not created by AviSynth. It is not allowed to invoke script on this thread ...");
  }
}

DeviceSetter::~DeviceSetter()
{
  env->SetCurrentDevice(downstreamDevice);
}

class QueuePrefetcher
{
  PClip child;
  VideoInfo vi;

  int prefetchFrames;
  int numThreads;

  ThreadPool* threadPool;
  Device* device;

  typedef LruCache<size_t, PVideoFrame> CacheType;

  std::shared_ptr<CacheType> videoCache;

  std::mutex mutex; // do not acceess to videoCache during locked by this mutex
  std::deque<std::pair<size_t, CacheType::handle>> prefetchQueue;
  int numWorkers;
  std::exception_ptr workerException;
  bool workerExceptionPresent;

  static AVSValue ThreadWorker_(IScriptEnvironment2* env, void* data)
  {
    return static_cast<QueuePrefetcher*>(data)->ThreadWorker(
      static_cast<InternalEnvironment*>(env));
  }

  AVSValue ThreadWorker(InternalEnvironment* env)
  {
    device->SetActiveToCurrentThread(env);

    while (true) {
      std::pair<size_t, CacheType::handle> work;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (prefetchQueue.size() == 0) {
          // there are no prefetch work
          --numWorkers;
          break;
        }
        work = prefetchQueue.front();
        prefetchQueue.pop_front();
      }

      try
      {
        work.second.first->value = child->GetFrame((int)work.first, env);
        videoCache->commit_value(&work.second);
      }
      catch (...)
      {
        {
          std::lock_guard<std::mutex> lock(mutex);
          workerException = std::current_exception();
          workerExceptionPresent = true;
        }
        videoCache->rollback(&work.second);
      }
    }

    return AVSValue();
  }

  void SchedulePrefetch(int currentN, InternalEnvironment* env)
  {
    int numQueued = 0;
    for (int n = currentN + 1;
      (n < currentN + prefetchFrames) && (n < vi.num_frames);
      ++n)
    {
      PVideoFrame result;
      CacheType::handle cacheHandle;
      switch (videoCache->lookup(n, &cacheHandle, false, result))
      {
      case LRU_LOOKUP_NOT_FOUND:
      {
        std::lock_guard<std::mutex> lock(mutex);
        prefetchQueue.emplace_back(n, cacheHandle);
        ++numQueued;
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
        env->ThrowError("Invalid Program");
        break;
      }
      }
    }
    // wake up sleeping thread
    if (numQueued > 0) {
      std::lock_guard<std::mutex> lock(mutex);
      for (; numWorkers < numThreads && numQueued > 0; ++numWorkers, --numQueued) {
        threadPool->QueueJob(ThreadWorker_, this, env, nullptr);
      }
    }
  }

public:
  QueuePrefetcher(PClip child, int prefetchFrames, int numThreads, Device* device, InternalEnvironment* env) :
    child(child),
    vi(child->GetVideoInfo()),
    prefetchFrames(prefetchFrames),
    numThreads(numThreads),
    threadPool(NULL),
    device(device),
    videoCache(NULL),
    numWorkers(0),
    workerExceptionPresent(false)
  {
    if (numThreads > 0 && prefetchFrames > 0) {
      threadPool = env->NewThreadPool(numThreads);
      videoCache = std::shared_ptr<CacheType>(new CacheType(prefetchFrames - 1, CACHE_NO_RESIZE));
    }
    else {
      numThreads = prefetchFrames = 0;
    }
  }

  ~QueuePrefetcher()
  {
    if (numThreads > 0) {
      // finish threadpool
      threadPool->Finish();

      // cancel queue
      while (prefetchQueue.size() > 0) {
        videoCache->rollback(&prefetchQueue.front().second);
        prefetchQueue.pop_front();
      }
    }
  }

  VideoInfo GetVideoInfo() const { return vi; }

  PVideoFrame GetFrame(int n, InternalEnvironment* env)
  {
    // do not use thread when invoke running
    if (prefetchFrames == 0 || env->GetSuppressThreadCount() > 0) {
      return child->GetFrame(n, env);
    }

    PVideoFrame result;
    CacheType::handle cacheHandle;
    switch (videoCache->lookup(n, &cacheHandle, true, result)) {
    case LRU_LOOKUP_FOUND_AND_READY:
      break;
    case LRU_LOOKUP_NOT_FOUND:
    {
      try
      {
        {
          std::lock_guard<std::mutex> lock(mutex);
          // check error
          if (workerExceptionPresent)
          {
            std::rethrow_exception(workerException);
          }
        }
        cacheHandle.first->value = child->GetFrame(n, env);
        result = cacheHandle.first->value;
        videoCache->commit_value(&cacheHandle);
      }
      catch (...)
      {
        videoCache->rollback(&cacheHandle);
        throw;
      }
      break;
    }
    case LRU_LOOKUP_NO_CACHE:
      result = child->GetFrame(n, env);
      break;
    case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:   // Fall-through intentional
    default:
      env->ThrowError("Invalid Program");
      break;
    }

    SchedulePrefetch(n, env);

    return result;
  }
};

class FrameTransferEngine
{
public:
  QueuePrefetcher& child;
  VideoInfo vi;

  Device* upstreamDevice;
  Device* downstreamDevice;

  FrameTransferEngine(QueuePrefetcher& child, Device* upstreamDevice, Device* downstreamDevice) :
    child(child),
    vi(child.GetVideoInfo()),
    upstreamDevice(upstreamDevice),
    downstreamDevice(downstreamDevice)
  { }

  virtual ~FrameTransferEngine() { }

  virtual PVideoFrame GetFrame(int n, InternalEnvironment* env) = 0;
};

#ifdef ENABLE_CUDA
class CUDAFrameTransferEngine : public FrameTransferEngine
{
  typedef LruCache<size_t, PVideoFrame> CacheType;

  struct QueueItem {
    size_t n;
    PVideoFrame src;
    CacheType::handle cacheHandle;
    cudaEvent_t completeEvent;
    std::unique_ptr<std::vector<DeviceCompleteCallbackData>> completeCallbacks;
  };

  int prefetchFrames;

  std::shared_ptr<CacheType> videoCache;

  std::mutex mutex;
  cudaStream_t stream;
  std::deque<QueueItem> prefetchQueue;

  cudaMemcpyKind GetMemcpyKind()
  {
    if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Host to Device
      return cudaMemcpyHostToDevice;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
      // Device to Host
      return cudaMemcpyDeviceToHost;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Device to Device
      return cudaMemcpyDeviceToDevice;
    }
    _ASSERT(false);
    return cudaMemcpyDefault;
  }

  void ExecuteCallbacks(const std::vector<DeviceCompleteCallbackData>* callbacks)
  {
    if (callbacks != nullptr) {
      for (auto cbdata : *callbacks) {
        cbdata.cb(cbdata.user_data);
      }
    }
  }

  void TransferFrameData(PVideoFrame& dst, PVideoFrame& src, bool async, InternalEnvironment* env)
  {
    size_t srchead = GetFrameHead(src);
    size_t sz = GetFrameTail(src) - srchead;
    const BYTE* srcptr = src->GetFrameBuffer()->GetReadPtr() + srchead;
    BYTE* dstptr = dst->GetFrameBuffer()->GetWritePtr() + GetFrameHead(dst);
    cudaMemcpyKind kind = GetMemcpyKind();

    if (async) {
      CUDA_CHECK(cudaMemcpyAsync(dstptr, srcptr, sz, kind, stream));
    }
    else {
      CUDA_CHECK(cudaMemcpy(dstptr, srcptr, sz, kind));
    }
  }

  PVideoFrame GetFrameImmediate(int n, InternalEnvironment* env)
  {
    PVideoFrame src = child.GetFrame(n, env);
    PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
    TransferFrameData(dst, src, false, env);

#if 1
    AVSMap* mapv = env->getFramePropsRW(dst);
    const int numKeys = env->propNumKeys(mapv);
    for (int i = 0; i < numKeys; i++) {
      const char* key = env->propGetKey(mapv, i);
      if (env->propGetType(mapv, key) == AVSPropTypes::PROPTYPE_FRAME) {
        // isFrame true
        const int numElements = env->propNumElements(mapv, key);

        std::vector<PVideoFrame> frameset;
        int error;
        // avs+: can be more frames in a frame property array
        for (int index = 0; index < numElements; index++) {
          const PVideoFrame srcframe = env->propGetFrame(mapv, key, index, &error);
          frameset.push_back(srcframe);
        }

        env->propDeleteKey(mapv, key);

        for (int index = 0; index < numElements; index++) {
          PVideoFrame src = frameset[index];
          PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
          TransferFrameData(dst, src, false, env);
          env->propSetFrame(mapv, key, dst, AVSPropAppendMode::PROPAPPENDMODE_APPEND);
        }
      }
    }
#else
    // kept for reference from neo fork, a single frame in frame properties. No array of frames here
    AVSMap* mapv = env->GetAVSMap(dst);
    for (auto it = mapv->data.begin(), end = mapv->data.end(); it != end; ++it) {
      if (it->second.IsFrame()) {
        PVideoFrame src = it->second.GetFrame();
        PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
        TransferFrameData(dst, src, false, env);
        it->second = dst;
      }
    }
#endif
    ExecuteCallbacks(downstreamDevice->GetAndClearCallbacks().get());

    return dst;
  }

  QueueItem SetupTransfer(int n, CacheType::handle& cacheHandle, InternalEnvironment* env)
  {
    QueueItem item = { (size_t)n, child.GetFrame(n, env), cacheHandle, nullptr, nullptr };
    cacheHandle.first->value = env->GetOnDeviceFrame(item.src, downstreamDevice);
    CUDA_CHECK(cudaEventCreate(&item.completeEvent));

    item.completeCallbacks = upstreamDevice->GetAndClearCallbacks();

    if (upstreamDevice->device_type == DEV_TYPE_CUDA) {
      static_cast<CUDADevice*>(upstreamDevice)->MakeStreamWaitCompute(stream, env);
    }

    TransferFrameData(cacheHandle.first->value, item.src, true, env);

#if 1
    AVSMap* mapv = env->getFramePropsRW(cacheHandle.first->value);
    const int numKeys = env->propNumKeys(mapv);
    for (int i = 0; i < numKeys; i++) {
      const char* key = env->propGetKey(mapv, i);
      if (env->propGetType(mapv, key) == AVSPropTypes::PROPTYPE_FRAME) {
        // isFrame true
        const int numElements = env->propNumElements(mapv, key);

        std::vector<PVideoFrame> frameset;
        int error;
        // avs+: can be more frames in a frame property array
        for (int index = 0; index < numElements; index++) {
          const PVideoFrame srcframe = env->propGetFrame(mapv, key, index, &error);
          frameset.push_back(srcframe);
        }

        env->propDeleteKey(mapv, key);

        for (int index = 0; index < numElements; index++) {
          PVideoFrame src = frameset[index];
          PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
          TransferFrameData(dst, src, true, env);
          env->propSetFrame(mapv, key, dst, AVSPropAppendMode::PROPAPPENDMODE_APPEND);
        }
      }
    }
#else
    // kept for reference from neo fork, a single frame in frame properties. No frame arrays here
    AVSMap* mapv = env->GetAVSMap(cacheHandle.first->value);
    for (auto it = mapv->data.begin(), end = mapv->data.end(); it != end; ++it) {
      if (it->second.IsFrame()) {
        PVideoFrame src = it->second.GetFrame();
        PVideoFrame dst = env->GetOnDeviceFrame(src, downstreamDevice);
        TransferFrameData(dst, src, true, env);
        it->second = dst;
      }
    }
#endif

    CUDA_CHECK(cudaEventRecord(item.completeEvent, stream));

    return std::move(item);
  }

  int SchedulePrefetch(int currentN, int prefetchStart, InternalEnvironment* env)
  {
    int numQueued = 0;
    int n = prefetchStart;
    for (; n < currentN + prefetchFrames && n < vi.num_frames; ++n)
    {
      PVideoFrame result;
      CacheType::handle cacheHandle;
      switch (videoCache->lookup(n, &cacheHandle, false, result))
      {
      case LRU_LOOKUP_NOT_FOUND:
      {
        try {
          prefetchQueue.emplace_back(SetupTransfer(n, cacheHandle, env));
        }
        catch (...) {
          videoCache->rollback(&cacheHandle);
          throw;
        }
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
        env->ThrowError("Invalid Program");
        break;
      }
      }
    }
    return n;
  }

  void FinishCompleted(InternalEnvironment* env)
  {
    while (prefetchQueue.size() > 0) {
      QueueItem& item = prefetchQueue.front();
      cudaError_t err = cudaEventQuery(item.completeEvent);
      if (err == cudaErrorNotReady) {
        break;
      }
      try {
        CUDA_CHECK(err);

        // transfer is complete
        CUDA_CHECK(cudaEventDestroy(item.completeEvent));
        ExecuteCallbacks(item.completeCallbacks.get());

        videoCache->commit_value(&item.cacheHandle);
      }
      catch (...) {
        videoCache->rollback(&item.cacheHandle);
        throw;
      }
      prefetchQueue.pop_front();
    }
  }

  PVideoFrame WaitUntil(int n, InternalEnvironment* env)
  {
    while (prefetchQueue.size() > 0) {
      QueueItem& item = prefetchQueue.front();
      try {
        CUDA_CHECK(cudaEventSynchronize(item.completeEvent));
        CUDA_CHECK(cudaEventDestroy(item.completeEvent));
        ExecuteCallbacks(item.completeCallbacks.get());

        PVideoFrame frame = item.cacheHandle.first->value; // fill before Commit !!!

        videoCache->commit_value(&item.cacheHandle);

        if (item.n == n) {
          prefetchQueue.pop_front();
          return frame;
        }

        prefetchQueue.pop_front();
      }
      catch (...) {
        videoCache->rollback(&item.cacheHandle);
        throw;
      }
    }
    env->ThrowError("invalid program");
    return PVideoFrame();
  }

  void CheckDevicePair(InternalEnvironment* env)
  {
    if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Host to Device
      return;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
      // Device to Host
      return;
    }
    if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
      // Device to Device
      return;
    }
    env->ThrowError("[CUDAFrameTransferEngine] invalid device pair up:%s down:%d",
      upstreamDevice->GetName(), downstreamDevice->GetName());
  }

public:
  CUDAFrameTransferEngine(QueuePrefetcher& child, Device* upstreamDevice, Device* downstreamDevice, int prefetchFrames, InternalEnvironment* env) :
    FrameTransferEngine(child, upstreamDevice, downstreamDevice),
    prefetchFrames(prefetchFrames),
    videoCache(new CacheType(std::max(0, prefetchFrames - 1), CACHE_NO_RESIZE)),
    stream(nullptr)
  {
    CheckDevicePair(env);

    // note: stream belongs to a device
    upstreamDevice->SetActiveToCurrentThread(env);
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  ~CUDAFrameTransferEngine()
  {
    for (auto& item : prefetchQueue) {
      cudaEventSynchronize(item.completeEvent);
      cudaEventDestroy(item.completeEvent);
      ExecuteCallbacks(item.completeCallbacks.get());
      videoCache->commit_value(&item.cacheHandle);
    }
    prefetchQueue.clear();

    cudaStreamDestroy(stream);
  }

  virtual PVideoFrame GetFrame(int n, InternalEnvironment* env)
  {
    // Giant lock. This is OK because all transfer is done asynchronously
    std::lock_guard<std::mutex> lock(mutex);

    // set upstream device
    upstreamDevice->SetActiveToCurrentThread(env);

    // do not use thread when invoke running
    if (prefetchFrames == 0 || env->GetSuppressThreadCount() > 0) {
      return GetFrameImmediate(n, env);
    }

    FinishCompleted(env);

    SchedulePrefetch(n, n, env);

    // Get requested frame
    PVideoFrame result;
    CacheType::handle cacheHandle;
    // fill result if LRU_LOOKUP_FOUND_AND_READY
    switch (videoCache->lookup(n, &cacheHandle, false, result))
    {
    case LRU_LOOKUP_FOUND_AND_READY:
    {
      break;
    }
    case LRU_LOOKUP_NO_CACHE:
    {
      result = GetFrameImmediate(n, env);
      break;
    }
    case LRU_LOOKUP_FOUND_BUT_NOTAVAIL:
    {
      // now transferring, wait until completion
      result = WaitUntil(n, env);
      break;
    }
    case LRU_LOOKUP_NOT_FOUND:
    {
      env->ThrowError("Oh.. maybe cache size is too small ...");
      break;
    }
    default:
    {
      env->ThrowError("Invalid Program");
      break;
    }
    }

    // set downstreamdevice
    downstreamDevice->SetActiveToCurrentThread(env);

    return result;
  }
};
#endif

FrameTransferEngine* CreateTransferEngine(QueuePrefetcher& child,
  Device* upstreamDevice, Device* downstreamDevice, int prefetchFrames, InternalEnvironment* env)
{
#ifdef ENABLE_CUDA
  if (upstreamDevice->device_type == DEV_TYPE_CPU && downstreamDevice->device_type == DEV_TYPE_CUDA) {
    // CPU to CUDA
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
  if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CPU) {
    // CUDA to CPU
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
  if (upstreamDevice->device_type == DEV_TYPE_CUDA && downstreamDevice->device_type == DEV_TYPE_CUDA) {
    // CUDA to CUDA
    return new CUDAFrameTransferEngine(child, upstreamDevice, downstreamDevice, prefetchFrames, env);
  }
#endif
  env->ThrowError("Not supported frame data transfer. up:%s down:%d",
    upstreamDevice->GetName(), downstreamDevice->GetName());
  return nullptr;
}

class OnDevice : public GenericVideoFilter
{
  Device* upstreamDevice;
  int prefetchFrames;

  QueuePrefetcher prefetcher;

  std::mutex mutex;
  std::map<Device*, std::unique_ptr<FrameTransferEngine>> transferEngines;

  FrameTransferEngine* GetOrCreateTransferEngine(Device* downstreamDevice, InternalEnvironment* env)
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = transferEngines.find(downstreamDevice);
    if (it != transferEngines.end()) {
      return it->second.get();
    }
    int transferPrefetch = (prefetchFrames == 1) ? 2 : prefetchFrames;
    auto pEngine = CreateTransferEngine(prefetcher, upstreamDevice, downstreamDevice, transferPrefetch, env);
    transferEngines[downstreamDevice] = std::unique_ptr<FrameTransferEngine>(pEngine);
    return pEngine;
  }

public:
  OnDevice(PClip child, int prefetchFrames, Device* upstreamDevice, InternalEnvironment* env) :
    GenericVideoFilter(child),
    upstreamDevice(upstreamDevice),
    prefetchFrames(prefetchFrames),
    prefetcher(child, (prefetchFrames >= 2) ? 2 : 0, (prefetchFrames >= 2) ? 1 : 0, upstreamDevice, env)
  { }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env_)
  {
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);
    Device* downstreamDevice = env->SetCurrentDevice(upstreamDevice);

    if (downstreamDevice == nullptr) {
      env->ThrowError("This thread is not created by AviSynth. It is not allowed to call GetFrame on this thread ...");
    }

    if (downstreamDevice == upstreamDevice) {
      // shortcut
      return child->GetFrame(n, env);
    }

    // Get frame via transfer engine
    PVideoFrame frame = GetOrCreateTransferEngine(downstreamDevice, env)->GetFrame(n, env);

    env->SetCurrentDevice(downstreamDevice);
    return frame;
  }

  void __stdcall GetAudio(void* buf, int64_t start, int64_t count, IScriptEnvironment* env_)
  {
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);
    Device* downstreamDevice = env->SetCurrentDevice(upstreamDevice);
    try {
      child->GetAudio(buf, start, count, env);
      env->SetCurrentDevice(downstreamDevice);
    }
    catch (...) {
      env->SetCurrentDevice(downstreamDevice);
      throw;
    }
  }

  int __stdcall SetCacheHints(int cachehints, int frame_range)
  {
    if (cachehints == CACHE_GET_MTMODE) {
      return MT_NICE_FILTER;
    }
    if (cachehints == CACHE_GET_DEV_TYPE) {
      return DEV_TYPE_ANY;
    }
    if (cachehints == CACHE_GET_CHILD_DEV_TYPE) {
      return upstreamDevice->device_type;
    }
    return 0;
  }

  static AVSValue __cdecl Create(AVSValue args, void* user_data, IScriptEnvironment* env_)
  {
    AvsDeviceType upstreamType = (AvsDeviceType)(size_t)user_data;
    InternalEnvironment* env = static_cast<InternalEnvironment*>(env_);

    if (args[0].IsClip()) {
      PClip clip = args[0].AsClip();
      int numPrefetch = args[1].Defined() ? args[1].AsInt() : 1;
      int upstreamIndex = (args.ArraySize() >= 3 && args[2].Defined()) ? args[2].AsInt() : 0;

      if (numPrefetch < 0) {
        numPrefetch = 0;
      }

      switch (upstreamType) {
      case DEV_TYPE_CPU:
        return new OnDevice(clip, numPrefetch, (Device*)(void*)env->GetDevice(DEV_TYPE_CPU, 0), env);
      case DEV_TYPE_CUDA:
        return new OnDevice(clip, numPrefetch, (Device*)(void*)env->GetDevice(DEV_TYPE_CUDA, upstreamIndex), env);
      default:
        env->ThrowError("Not supported device ...");
        return AVSValue();
      }

    }
    else {
      assert(args[0].IsFunction());
      PFunction func = args[0].AsFunction();
      int upstreamIndex = (args.ArraySize() >= 2 && args[1].Defined()) ? args[1].AsInt() : 0;

      Device* upstreamDevice = nullptr;
      switch (upstreamType) {
      case DEV_TYPE_CPU:
        upstreamDevice = (Device*)(void*)env->GetDevice(DEV_TYPE_CPU, 0);
        break;
      case DEV_TYPE_CUDA:
        upstreamDevice = (Device*)(void*)env->GetDevice(DEV_TYPE_CUDA, upstreamIndex);
        break;
      default:
        env->ThrowError("Not supported device ...");
        break;
      }

      DeviceSetter setter(env, upstreamDevice);

      try {
        AVSValue ret = env->Invoke3(AVSValue(), func, AVSValue(nullptr, 0));
        return ret;
      }
      catch (IScriptEnvironment::NotFound) {
        const char* name = (upstreamType == DEV_TYPE_CPU) ? "OnCPU" : "OnCUDA";
        env->ThrowError(
          "%s: Invalid function parameter type '%s'(%s)\n"
          "Function should have no argument",
          name, func->GetDefinition()->param_types, func->ToString(env));
      }

      return AVSValue();
    }
  }
};

void CopyCUDAFrame(const PVideoFrame& dst, const PVideoFrame& src, InternalEnvironment* env, bool sync)
{
#ifdef ENABLE_CUDA
  size_t srchead = GetFrameHead(src);
  size_t sz = GetFrameTail(src) - srchead;
  const BYTE* srcptr = src->GetFrameBuffer()->GetReadPtr() + srchead;
  BYTE* dstptr = dst->GetFrameBuffer()->GetWritePtr() + GetFrameHead(dst);

  AvsDeviceType srcDevice = src->GetDevice()->device_type;
  AvsDeviceType dstDevice = dst->GetDevice()->device_type;
  cudaMemcpyKind kind = cudaMemcpyHostToHost;

  if (srcDevice == DEV_TYPE_CPU && dstDevice == DEV_TYPE_CUDA) {
    kind = cudaMemcpyHostToDevice;
  }
  if (srcDevice == DEV_TYPE_CUDA && dstDevice == DEV_TYPE_CPU) {
    kind = cudaMemcpyDeviceToHost;
  }
  if (srcDevice == DEV_TYPE_CUDA && dstDevice == DEV_TYPE_CUDA) {
    kind = cudaMemcpyDeviceToDevice;
  }

  if (sync) {
    CUDA_CHECK(cudaMemcpy(dstptr, srcptr, sz, kind));
  }
  else {
    CUDA_CHECK(cudaMemcpyAsync(dstptr, srcptr, sz, kind));
  }
#else
  env->ThrowError("CopyCUDAFrame: CUDA support is disabled ...");
#endif
}

PVideoFrame GetFrameOnDevice(PClip& c, int n, const PDevice& device, InternalEnvironment* env)
{
  DeviceSetter setter(env, (Device*)(void*)device);
  return c->GetFrame(n, env);
}

extern const AVSFunction Device_filters[] = {
  { "OnCPU", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i", OnDevice::Create, (void*)DEV_TYPE_CPU },
  { "OnCUDA", BUILTIN_FUNC_PREFIX, "c[num_prefetch]i[device_index]i", OnDevice::Create, (void*)DEV_TYPE_CUDA },
  { "OnCPU", BUILTIN_FUNC_PREFIX, "n", OnDevice::Create, (void*)DEV_TYPE_CPU },
  { "OnCUDA", BUILTIN_FUNC_PREFIX, "n[device_index]i", OnDevice::Create, (void*)DEV_TYPE_CUDA },
  { 0 }
};
