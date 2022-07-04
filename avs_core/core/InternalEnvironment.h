#ifndef _AVS_SCRIPTENVIRONMENT_H_INCLUDED
#define _AVS_SCRIPTENVIRONMENT_H_INCLUDED

#include <avisynth.h>
#include <algorithm>
#include <string>
#include <memory>
#include "function.h"

class ClipDataStore;

typedef enum _ELogLevel
{
    LOGLEVEL_NONE = 0,
    LOGLEVEL_ERROR = 1,
    LOGLEVEL_WARNING = 2,
    LOGLEVEL_INFO = 3,
    LOGLEVEL_DEBUG = 4
} ELogLevel;

typedef enum _ELogTicketType
{
    LOGTICKET_W1000 = 1000, // leaks during shutdown
    LOGTICKET_W1001 = 1001, // source plugin with no mt-mode
    LOGTICKET_W1002 = 1002, // buggy SetCacheHints()
    LOGTICKET_W1003 = 1003, // too stringent memory limit
    LOGTICKET_W1004 = 1004, // filter completely without mt-mode
    LOGTICKET_W1005 = 1005, // filter with inconsequent MT-modes
    LOGTICKET_W1006 = 1006, // filter with redundant MT-modes
    LOGTICKET_W1007 = 1007, // user should try 64-bit AVS for more memory
    LOGTICKET_W1008 = 1008, // multiple plugins define the same function
    LOGTICKET_W1009 = 1009, // a filter is using forced alignment
    LOGTICKET_W1010 = 1010, // MT-mode specified for script function
    LOGTICKET_W1100 = 1100, // memory reallocation occurs
} ELogTicketType;

enum CacheMode {
	CACHE_FAST_START,    // start up time and size balanced mode
	CACHE_OPTIMAL_SIZE,  // slow start up but optimal speed and cache size
	CACHE_NO_RESIZE,     // internal use only

	CACHE_DEFAULT = CACHE_FAST_START,
};

enum DeviceOpt: int {
    DEV_CUDA_PINNED_HOST, // allocate CPU frame with CUDA pinned host memory
    DEV_FREE_THRESHOLD,   // free request count threshold to free frame
};

class OneTimeLogTicket
{
public:
    ELogTicketType _type;
    const Function *_function = nullptr;
    const std::string _string;

    OneTimeLogTicket(ELogTicketType type);
    OneTimeLogTicket(ELogTicketType type, const Function *func);
    OneTimeLogTicket(ELogTicketType type, const std::string &str);
    bool operator==(const OneTimeLogTicket &other) const;
};

class Device;
class ThreadPool;
class ConcurrentVarStringFrame;
class FilterGraphNode;

class ScopedCounter {
	int& counter;
public:
	ScopedCounter(int& counter) : counter(counter) {
		++counter;
	}
	~ScopedCounter() {
		--counter;
	}
};

// Strictly for Avisynth core only.
// Neither host applications nor plugins should use
// these interfaces.
class InternalEnvironment :
  public IScriptEnvironment2,
  public IScriptEnvironment_Avs25,
  public INeoEnv {
protected:
  virtual ~InternalEnvironment() {}
public:
  // define commons to fix ambiguous error

  typedef IScriptEnvironment::NotFound NotFound;
  typedef IScriptEnvironment::ApplyFunc ApplyFunc;

  // IScriptEnvironment
  virtual int __stdcall GetCPUFlags() = 0;
  virtual char* __stdcall SaveString(const char* s, int length = -1) = 0;
  virtual char* Sprintf(const char* fmt, ...) = 0;
  virtual char* __stdcall VSprintf(const char* fmt, va_list val) = 0;
  __declspec(noreturn) virtual void ThrowError(const char* fmt, ...) = 0;
  virtual void __stdcall AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data) = 0;
  virtual bool __stdcall FunctionExists(const char* name) = 0;
  virtual AVSValue __stdcall Invoke(const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;
  virtual AVSValue __stdcall GetVar(const char* name) = 0;
  virtual bool __stdcall SetVar(const char* name, const AVSValue& val) = 0;
  virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) = 0;
  virtual void __stdcall PushContext(int level = 0) = 0;
  virtual void __stdcall PopContext() = 0;
  // NewVideoFrame is replaced with new one
  virtual bool __stdcall MakeWritable(PVideoFrame* pvf) = 0;
  virtual void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) = 0;
  virtual void __stdcall AtExit(INeoEnv::ShutdownFunc function, void* user_data) = 0;
  virtual void __stdcall CheckVersion(int version = AVISYNTH_INTERFACE_VERSION) = 0;
  virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) = 0;
  virtual int __stdcall SetMemoryMax(int mem) = 0;
  virtual int __stdcall SetWorkingDir(const char * newdir) = 0;
  virtual void* __stdcall ManageCache(int key, void* data) = 0;
  virtual bool __stdcall PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key) = 0;
  virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) = 0;
  // AVISYNTH_INTERFACE_VERSION 5
  virtual void __stdcall DeleteScriptEnvironment() = 0;
  virtual void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
    int textcolor, int halocolor, int bgcolor) = 0;
  virtual const AVS_Linkage* __stdcall GetAVSLinkage() = 0;

  // AVISYNTH_INTERFACE_VERSION 6
  virtual AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def = AVSValue()) = 0;

  // AVISYNTH_INTERFACE_VERSION 8
  virtual PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) = 0;

  // frame properties support
  virtual void __stdcall copyFrameProps(const PVideoFrame& src, PVideoFrame& dst) = 0;

  virtual const AVSMap* __stdcall getFramePropsRO(const PVideoFrame& frame) = 0;
  virtual AVSMap* __stdcall getFramePropsRW(PVideoFrame &frame) = 0;

  virtual int __stdcall propNumKeys(const AVSMap* map) = 0;
  virtual const char* __stdcall propGetKey(const AVSMap* map, int index) = 0;
  virtual int __stdcall propDeleteKey(AVSMap* map, const char* key) = 0;
  virtual int __stdcall propNumElements(const AVSMap* map, const char* key) = 0;
  virtual char __stdcall propGetType(const AVSMap* map, const char* key) = 0;

  virtual int64_t __stdcall propGetInt(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual double __stdcall propGetFloat(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual const char* __stdcall propGetData(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual int __stdcall propGetDataSize(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual PClip __stdcall propGetClip(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual const PVideoFrame __stdcall propGetFrame(const AVSMap* map, const char* key, int index, int* error) = 0;
  virtual int __stdcall propSetInt(AVSMap* map, const char* key, int64_t i, int append) = 0;
  virtual int __stdcall propSetFloat(AVSMap* map, const char* key, double d, int append) = 0;
  virtual int __stdcall propSetData(AVSMap* map, const char* key, const char* d, int length, int append) = 0;
  virtual int __stdcall propSetClip(AVSMap* map, const char* key, PClip& clip, int append) = 0;
  virtual int __stdcall propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append) = 0;

  virtual const int64_t* __stdcall propGetIntArray(const AVSMap* map, const char* key, int* error) = 0;
  virtual const double* __stdcall propGetFloatArray(const AVSMap* map, const char* key, int* error) = 0;
  virtual int __stdcall propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size) = 0;
  virtual int __stdcall propSetFloatArray(AVSMap* map, const char* key, const double* d, int size) = 0;

  virtual AVSMap* __stdcall createMap() = 0;
  virtual void __stdcall freeMap(AVSMap* map) = 0;
  virtual void __stdcall clearMap(AVSMap* map) = 0;

  // NewVideoFrame with frame prop source is replaced with new one

  virtual size_t  __stdcall GetEnvProperty(AvsEnvProperty prop) = 0;
  virtual void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type) = 0;
  virtual void __stdcall Free(void* ptr) = 0;

  // Returns TRUE and the requested variable. If the method fails, returns FALSE and does not touch 'val'.
  virtual bool __stdcall GetVarTry(const char* name, AVSValue* val) const = 0; // ex virtual bool  __stdcall GetVar(const char* name, AVSValue* val) const = 0;
  virtual bool __stdcall GetVarBool(const char* name, bool def) const = 0; // ex: virtual bool __stdcall GetVar(const char* name, bool def) const = 0;
  virtual int __stdcall GetVarInt(const char* name, int def) const = 0; // ex: int  __stdcall GetVar(const char* name, int def) const = 0;
  virtual double __stdcall GetVarDouble(const char* name, double def) const = 0; // ex: virtual double  __stdcall GetVar(const char* name, double def) const = 0;
  virtual const char* __stdcall GetVarString(const char* name, const char* def) const = 0; // ex: virtual const char* __stdcall GetVar(const char* name, const char* def) const = 0;
  virtual int64_t __stdcall GetVarLong(const char* name, int64_t def) const = 0; // brand new in v8 - though no real int64 support yet

  // Invoke functions renamed for keeping vtable order in IS
  // moved from IS2
  virtual bool __stdcall InvokeTry(AVSValue* result, const char* name, const AVSValue& args, const char* const* arg_names = 0) = 0;
  // Since V8
  virtual AVSValue __stdcall Invoke2(const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;
  // moved from INeo
  virtual bool __stdcall Invoke2Try(AVSValue* result, const AVSValue& implicit_last, const char* name, const AVSValue args, const char* const* arg_names = 0) = 0;
  virtual AVSValue __stdcall Invoke3(const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0) = 0;
  virtual bool __stdcall Invoke3Try(AVSValue* result, const AVSValue& implicit_last, const PFunction& func, const AVSValue args, const char* const* arg_names = 0) = 0;

  virtual bool __stdcall MakePropertyWritable(PVideoFrame* pvf) = 0; // V9

  // IScriptEnvironment2
  virtual bool __stdcall LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result) = 0;
  virtual void __stdcall AddAutoloadDir(const char* dirPath, bool toFront) = 0;
  virtual void __stdcall ClearAutoloadDirs() = 0;
  virtual void __stdcall AutoloadPlugins() = 0;
  virtual void __stdcall AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data, const char *exportVar) = 0;
  virtual bool __stdcall InternalFunctionExists(const char* name) = 0;
  virtual void __stdcall SetFilterMTMode(const char* filter, MtMode mode, bool force) = 0;
  virtual IJobCompletion* __stdcall NewCompletion(size_t capacity) = 0;
  virtual void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion) = 0;
  // InternalEnvironment
  virtual int __stdcall IncrImportDepth() = 0;
  virtual int __stdcall DecrImportDepth() = 0;
  virtual void __stdcall AdjustMemoryConsumption(size_t amount, bool minus) = 0;
  virtual bool __stdcall FilterHasMtMode(const Function* filter) const = 0;
  virtual MtMode __stdcall GetFilterMTMode(const Function* filter, bool* is_forced) const = 0; // If filter is "", gets the default MT mode
  virtual ClipDataStore* __stdcall ClipData(IClip *clip) = 0;
  virtual MtMode __stdcall GetDefaultMtMode() const = 0;
  virtual void __stdcall SetLogParams(const char *target, int level) = 0;
  virtual void LogMsg(int level, const char* fmt, ...) = 0;
  virtual void __stdcall LogMsg_valist(int level, const char* fmt, va_list va) = 0;
  virtual void LogMsgOnce(const OneTimeLogTicket &ticket, int level, const char* fmt, ...) = 0;
  virtual void __stdcall LogMsgOnce_valist(const OneTimeLogTicket &ticket, int level, const char* fmt, va_list va) = 0;
  virtual void __stdcall VThrowError(const char* fmt, va_list va) = 0;
  virtual void __stdcall SetMaxCPU(const char *feature) = 0;

  virtual IScriptEnvironment2* __stdcall GetEnv2() final { return this; }
  virtual IScriptEnvironment_Avs25* __stdcall GetEnv25() final { return this; }


  virtual void __stdcall SetGraphAnalysis(bool enable) = 0;

  virtual Device* __stdcall SetCurrentDevice(Device* device) = 0;
  virtual Device* __stdcall GetCurrentDevice() const = 0;
  // replacement of NewVideoFrame
  virtual PVideoFrame __stdcall NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device) = 0;
  virtual PVideoFrame __stdcall NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device, PVideoFrame *propSrc) = 0;
  virtual PVideoFrame __stdcall GetOnDeviceFrame(const PVideoFrame& src, Device* device) = 0;

  using INeoEnv::SetMemoryMax;
  using INeoEnv::Invoke;
  using INeoEnv::NewVideoFrame;
  using INeoEnv::SaveString;

  // Nekopanda: support multiple prefetcher //
  // to allow thread to submit with their env
  virtual void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment *env) = 0;
  virtual ThreadPool* __stdcall NewThreadPool(size_t nThreads) = 0;
  virtual void __stdcall AddRef() = 0;
  virtual void __stdcall Release() = 0;

  virtual ConcurrentVarStringFrame* __stdcall GetTopFrame() = 0;

  // Nekopanda: new cache control mechanism
  virtual void __stdcall SetCacheMode(CacheMode mode) = 0;
  virtual CacheMode __stdcall GetCacheMode() = 0;
	virtual bool& __stdcall GetSupressCaching() = 0;

  virtual void __stdcall SetDeviceOpt(DeviceOpt mode, int val) = 0;

  virtual void __stdcall UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar) = 0;
  virtual bool __stdcall Invoke_(AVSValue *result, const AVSValue& implicit_last,
    const char* name, const Function *f, const AVSValue& args, const char* const* arg_names) = 0;

  virtual InternalEnvironment* __stdcall NewThreadScriptEnvironment(int thread_id) = 0;

	// per thread data access
	virtual int __stdcall GetThreadId() = 0;
	virtual int& __stdcall GetFrameRecursiveCount() = 0;
	virtual int& __stdcall GetSuppressThreadCount() = 0;
	virtual FilterGraphNode*& GetCurrentGraphNode() = 0;
};

struct InternalEnvironmentDeleter {
	void operator()(InternalEnvironment* ptr) const {
		ptr->Release();
	}
};
typedef std::unique_ptr<InternalEnvironment, InternalEnvironmentDeleter> PInternalEnvironment;

#endif // _AVS_SCRIPTENVIRONMENT_H_INCLUDED
