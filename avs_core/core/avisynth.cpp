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

#include <avisynth.h>
#include "../core/internal.h"
#include "InternalEnvironment.h"
#include "./parser/script.h"
#include <avs/minmax.h>
#include <avs/alignment.h>
#include "strings.h"
#include <avs/cpuid.h>
#include <unordered_set>
#include "bitblt.h"
#include "FilterConstructor.h"
#include "PluginManager.h"
#include "MappedList.h"
#include <vector>
#include <iostream>
#include <fstream>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#ifdef AVS_WINDOWS
    #include <avs/win.h>
    #include <objbase.h>
#else
    #include "avisynth_conf.h"
#if defined(AVS_MACOS)
    #include <mach/host_info.h>
    #include <mach/mach_host.h>
    #include <mach/mach_init.h>
    #include <sys/sysctl.h>
#elif defined(AVS_BSD)
    #include <sys/sysctl.h>
#else
    #include <avs/filesystem.h>
    #include <set>
#if defined(AVS_HAIKU)
    #include <OS.h>
#else
    #include <sys/sysinfo.h>
#endif
#endif
    #include <avs/posix.h>
#endif

#include <string>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include "MTGuard.h"
#include "cache.h"
#include <clocale>

#include "FilterGraph.h"
#include "DeviceManager.h"
#include "AVSMap.h"

#ifndef YieldProcessor // low power spin idle
  #define YieldProcessor() __nop(void)
#endif

extern const AVSFunction Audio_filters[],
                         Combine_filters[],
                         Convert_filters[],
                         Convolution_filters[],
                         Edit_filters[],
                         Field_filters[],
                         Focus_filters[],
                         Fps_filters[],
                         Histogram_filters[],
                         Layer_filters[],
                         Levels_filters[],
                         Misc_filters[],
                         Plugin_functions[],
                         Resample_filters[],
                         Resize_filters[],
                         Script_functions[],
                         Source_filters[],
                         Text_filters[],
                         Transform_filters[],
                         Merge_filters[],
                         Color_filters[],
                         Debug_filters[],
                         Turn_filters[],
                         Conditional_filters[],
                         Conditional_funtions_filters[],
                         Cache_filters[],
                         Greyscale_filters[],
                         Swap_filters[],
                         Overlay_filters[],
                         Exprfilter_filters[],
                         FilterGraph_filters[],
                         Device_filters[]
;


const AVSFunction* const builtin_functions[] = {
                         Audio_filters,
                         Combine_filters,
                         Convert_filters,
                         Convolution_filters,
                         Edit_filters,
                         Field_filters,
                         Focus_filters,
                         Fps_filters,
                         Histogram_filters,
                         Layer_filters,
                         Levels_filters,
                         Misc_filters,
                         Resample_filters,
                         Resize_filters,
                         Script_functions,
                         Source_filters,
                         Text_filters,
                         Transform_filters,
                         Merge_filters,
                         Color_filters,
                         Debug_filters,
                         Turn_filters,
                         Conditional_filters,
                         Conditional_funtions_filters,
                         Plugin_functions,
                         Cache_filters,
                         Overlay_filters,
                         Greyscale_filters,
                         Swap_filters,
                         FilterGraph_filters,
                         Device_filters,
                         Exprfilter_filters
};

#if 0
// Global statistics counters
struct {
  unsigned int CleanUps;
  unsigned int Losses;
  unsigned int PlanA1;
  unsigned int PlanA2;
  unsigned int PlanB;
  unsigned int PlanC;
  unsigned int PlanD;
  char tag[36];
} g_Mem_stats = { 0, 0, 0, 0, 0, 0, 0, "CleanUps, Losses, Plan[A1,A2,B,C,D]" };
#endif

const _PixelClip PixelClip;


#ifdef MSVC
// Helper function to count set bits in the processor mask.
template<typename T>
static uint32_t CountSetBits(T bitMask)
{
  uint32_t LSHIFT = sizeof(T) * 8 - 1;
  uint32_t bitSetCount = 0;
  T bitTest = (T)1 << LSHIFT;
  uint32_t i;

  for (i = 0; i <= LSHIFT; ++i)
  {
    bitSetCount += ((bitMask & bitTest) ? 1 : 0);
    bitTest /= 2;
  }

  return bitSetCount;
}
#endif

static size_t GetNumPhysicalCPUs()
{
#if defined(AVS_WINDOWS)
#ifdef MSVC
  typedef BOOL(WINAPI* LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD);
  LPFN_GLPI glpi;
  BOOL done = FALSE;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = NULL;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION ptr = NULL;
  DWORD returnLength = 0;
  DWORD logicalProcessorCount = 0;
  DWORD numaNodeCount = 0;
  DWORD processorCoreCount = 0;
  DWORD processorL1CacheCount = 0;
  DWORD processorL2CacheCount = 0;
  DWORD processorL3CacheCount = 0;
  DWORD processorPackageCount = 0;
  DWORD byteOffset = 0;
  PCACHE_DESCRIPTOR Cache;

  glpi = (LPFN_GLPI)GetProcAddress(
    GetModuleHandle(TEXT("kernel32")),
    "GetLogicalProcessorInformation");
  if (NULL == glpi)
  {
    //    _tprintf(TEXT("\nGetLogicalProcessorInformation is not supported.\n"));
    return (0);
  }

  while (!done)
  {
    BOOL rc = glpi(buffer, &returnLength);

    if (FALSE == rc)
    {
      if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
      {
        if (buffer)
          free(buffer);

        buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
          returnLength);

        if (NULL == buffer)
        {
          //          _tprintf(TEXT("\nError: Allocation failure\n"));
          return (0);
        }
      }
      else
      {
        //        _tprintf(TEXT("\nError %d\n"), GetLastError());
        return (0);
      }
    }
    else
    {
      done = TRUE;
    }
  }

  ptr = buffer;

  while (byteOffset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= returnLength)
  {
    switch (ptr->Relationship)
    {
    case RelationNumaNode:
      // Non-NUMA systems report a single record of this type.
      numaNodeCount++;
      break;

    case RelationProcessorCore:
      processorCoreCount++;

      // A hyperthreaded core supplies more than one logical processor.
      logicalProcessorCount += CountSetBits<ULONG_PTR>(ptr->ProcessorMask);
      break;

    case RelationCache:
      // Cache data is in ptr->Cache, one CACHE_DESCRIPTOR structure for each cache.
      Cache = &ptr->Cache;
      if (Cache->Level == 1)
      {
        processorL1CacheCount++;
      }
      else if (Cache->Level == 2)
      {
        processorL2CacheCount++;
      }
      else if (Cache->Level == 3)
      {
        processorL3CacheCount++;
      }
      break;

    case RelationProcessorPackage:
      // Logical processors share a physical package.
      processorPackageCount++;
      break;

    default:
      //      _tprintf(TEXT("\nError: Unsupported LOGICAL_PROCESSOR_RELATIONSHIP value.\n"));
      return (0);
    }
    byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    ptr++;
  }

  /*
  _tprintf(TEXT("\nGetLogicalProcessorInformation results:\n"));
  _tprintf(TEXT("Number of NUMA nodes: %d\n"),
    numaNodeCount);
  _tprintf(TEXT("Number of physical processor packages: %d\n"),
    processorPackageCount);
  _tprintf(TEXT("Number of processor cores: %d\n"),
    processorCoreCount);
  _tprintf(TEXT("Number of logical processors: %d\n"),
    logicalProcessorCount);
  _tprintf(TEXT("Number of processor L1/L2/L3 caches: %d/%d/%d\n"),
    processorL1CacheCount,
    processorL2CacheCount,
    processorL3CacheCount);
  */

  free(buffer);

  return processorCoreCount;
#else
  return 4; // TODO: GCC on Windows?
#endif
#elif defined(AVS_LINUX)
  std::set<int> core_ids;
  for (auto& p : fs::directory_iterator("/sys/devices/system/cpu")) {
    if (!p.path().filename().string().rfind("cpu", 0)) {
      std::ifstream ifs(p.path() / "topology/core_id");
      int core_id;
      if (ifs) {
        ifs >> core_id;
        if (ifs)
          core_ids.insert(core_id);
      }
    }
  }
  return core_ids.size();
#elif defined(AVS_MACOS)
  int cpu_cnt = 0;
  size_t cpu_cnt_size = sizeof(cpu_cnt);
  return sysctlbyname("hw.physicalcpu", &cpu_cnt, &cpu_cnt_size, NULL, 0) ? 0 : cpu_cnt;
#else
  return 4; // AVS_BSD TODO
#endif
}

#ifdef MSVC
static std::string FormatString(const char* fmt, va_list args)
{
  va_list args2;
  va_copy(args2, args);
  _locale_t locale = _create_locale(LC_NUMERIC, "C"); // decimal point: dot

  int count = _vscprintf_l(fmt, locale, args2);
  // don't use _vsnprintf_l(NULL, 0, fmt, locale, args) here,
  // it returns -1 instead of the buffer size under Wine (February, 2017)
  std::vector<char> buf(count + 1);
  _vsnprintf_l(buf.data(), buf.size(), fmt, locale, args2);

  _free_locale(locale);
  va_end(args2);

  return std::string(buf.data());
}
#else
static std::string FormatString(const char* fmt, va_list args)
{
  va_list args2;
  va_copy(args2, args);

  int count = vsnprintf(NULL, 0, fmt, args);
  std::vector<char> buf(count + 1);
  vsnprintf(buf.data(), buf.size(), fmt, args2);

  va_end(args2);

  return std::string(buf.data());
}
#endif

void* VideoFrame::operator new(size_t size) {
  return ::operator new(size);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offset), offsetV(_offset), pitchUV(0), row_sizeUV(0), heightUV(0)  // PitchUV=0 so this doesn't take up additional space
  , offsetA(0), pitchA(0), row_sizeA(0), properties(avsmap)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height,
  int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
  , offsetA(0), pitchA(0), row_sizeA(0), properties(avsmap)
{
  InterlockedIncrement(&vfb->refcount);
}

VideoFrame::VideoFrame(VideoFrameBuffer* _vfb, AVSMap* avsmap, int _offset, int _pitch, int _row_size, int _height,
  int _offsetU, int _offsetV, int _pitchUV, int _row_sizeUV, int _heightUV, int _offsetA)
  : refcount(0), vfb(_vfb), offset(_offset), pitch(_pitch), row_size(_row_size), height(_height),
  offsetU(_offsetU), offsetV(_offsetV), pitchUV(_pitchUV), row_sizeUV(_row_sizeUV), heightUV(_heightUV)
  , offsetA(_offsetA), pitchA(_pitch), row_sizeA(_row_size), properties(avsmap)
{
  InterlockedIncrement(&vfb->refcount);
}
// Hack note :- Use of SubFrame will require an "InterlockedDecrement(&retval->refcount);" after
// assignement to a PVideoFrame, the same as for a "New VideoFrame" to keep the refcount consistant.
// P.F. ?? so far it works automatically

VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height) const {
  return new VideoFrame(vfb, new AVSMap(), offset + rel_offset, new_pitch, new_row_size, new_height);
}


VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height,
                                 int rel_offsetU, int rel_offsetV, int new_pitchUV) const {
    // Maintain plane size relationship
    const int new_row_sizeUV = !row_size ? 0 : MulDiv(new_row_size, row_sizeUV, row_size);
    const int new_heightUV   = !height   ? 0 : MulDiv(new_height,   heightUV,   height);

    return new VideoFrame(vfb, new AVSMap(), offset + rel_offset, new_pitch, new_row_size, new_height,
      rel_offsetU + offsetU, rel_offsetV + offsetV, new_pitchUV, new_row_sizeUV, new_heightUV);
}

// alpha support
VideoFrame* VideoFrame::Subframe(int rel_offset, int new_pitch, int new_row_size, int new_height,
  int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) const {
  // Maintain plane size relationship
  const int new_row_sizeUV = !row_size ? 0 : MulDiv(new_row_size, row_sizeUV, row_size);
  const int new_heightUV = !height ? 0 : MulDiv(new_height, heightUV, height);

  return new VideoFrame(vfb, new AVSMap(), offset + rel_offset, new_pitch, new_row_size, new_height,
    rel_offsetU + offsetU, rel_offsetV + offsetV, new_pitchUV, new_row_sizeUV, new_heightUV, rel_offsetA + offsetA);
}

VideoFrameBuffer::VideoFrameBuffer() : data(NULL), data_size(0), sequence_number(0), refcount(1) {}


VideoFrameBuffer::VideoFrameBuffer(int size, int margin, Device* device) :
  data(device->Allocate(size, margin)),
  data_size(size),
  sequence_number(0),
  refcount(0),
  device(device)
{
}

VideoFrameBuffer::~VideoFrameBuffer() {
  //  _ASSERTE(refcount == 0);
  InterlockedIncrement(&sequence_number); // HACK : Notify any children with a pointer, this buffer has changed!!!
  if (data) device->Free(data);
  data = nullptr; // and mark it invalid!!
  data_size = 0;   // and don't forget to set the size to 0 as well!
  device = nullptr; // no longer related to a device
}


class AtExiter {
  struct AtExitRec {
    const IScriptEnvironment::ShutdownFunc func;
    void* const user_data;
    AtExitRec* const next;
    AtExitRec(IScriptEnvironment::ShutdownFunc _func, void* _user_data, AtExitRec* _next)
      : func(_func), user_data(_user_data), next(_next) {}
  };
  AtExitRec* atexit_list;

public:
  AtExiter() {
    atexit_list = 0;
  }

  void Add(IScriptEnvironment::ShutdownFunc f, void* d) {
    atexit_list = new AtExitRec(f, d, atexit_list);
  }

  void Execute(IScriptEnvironment* env) {
    while (atexit_list) {
      AtExitRec* next = atexit_list->next;
      atexit_list->func(atexit_list->user_data, env);
      delete atexit_list;
      atexit_list = next;
    }
  }
};


static std::string NormalizeString(const std::string& str)
{
  // lowercase
  std::string ret = str;
  for (size_t i = 0; i < ret.size(); ++i)
    ret[i] = tolower(ret[i]);

  // trim trailing spaces
  size_t endpos = ret.find_last_not_of(" \t");
  if (std::string::npos != endpos)
    ret = ret.substr(0, endpos + 1);

  // trim leading spaces
  size_t startpos = ret.find_first_not_of(" \t");
  if (std::string::npos != startpos)
    ret = ret.substr(startpos);

  return ret;
}

typedef enum class _MtWeight
{
  MT_WEIGHT_0_DEFAULT,
  MT_WEIGHT_1_USERSPEC,
  MT_WEIGHT_2_USERFORCE,
  MT_WEIGHT_MAX
} MtWeight;

class ClipDataStore
{
public:

  // The clip instance that we hold data for.
  IClip* Clip = nullptr;

  // Clip was created directly by an Invoke() call
  bool CreatedByInvoke = false;

  ClipDataStore(IClip* clip) : Clip(clip) {};
};

class MtModeEvaluator
{
public:
  int NumChainedNice = 0;
  int NumChainedMulti = 0;
  int NumChainedSerial = 0;

  MtMode GetFinalMode(MtMode topInvokeMode)
  {
    if (NumChainedSerial > 0)
    {
      return MT_SERIALIZED;
    }
    else if (NumChainedMulti > 0)
    {
      if (MT_SERIALIZED == topInvokeMode)
      {
        return MT_SERIALIZED;
      }
      else
      {
        return MT_MULTI_INSTANCE;
      }
    }
    else
    {
      return topInvokeMode;
    }
  }

  void Accumulate(const MtModeEvaluator& other)
  {
    NumChainedNice += other.NumChainedNice;
    NumChainedMulti += other.NumChainedMulti;
    NumChainedSerial += other.NumChainedSerial;
  }

  void Accumulate(MtMode mode)
  {
    switch (mode)
    {
    case MT_NICE_FILTER:
      ++NumChainedNice;
      break;
    case MT_MULTI_INSTANCE:
      ++NumChainedMulti;
      break;
    case MT_SERIALIZED:
      ++NumChainedSerial;
      break;
    default:
      assert(0);
      break;
    }
  }

  static bool ClipSpecifiesMtMode(const PClip& clip)
  {
    int val = clip->SetCacheHints(CACHE_GET_MTMODE, 0);
    return (clip->GetVersion() >= 5) && (val > MT_INVALID) && (val < MT_MODE_COUNT);
  }

  static MtMode GetInstanceMode(const PClip& clip, MtMode defaultMode)
  {
    return ClipSpecifiesMtMode(clip) ? (MtMode)clip->SetCacheHints(CACHE_GET_MTMODE, 0) : defaultMode;
  }

  static MtMode GetInstanceMode(const PClip& clip)
  {
    return (MtMode)clip->SetCacheHints(CACHE_GET_MTMODE, 0);
  }

  static MtMode GetMtMode(const PClip& clip, const Function* invokeCall, const InternalEnvironment* env)
  {
    bool invokeModeForced;

    MtMode invokeMode = env->GetFilterMTMode(invokeCall, &invokeModeForced);
    if (invokeModeForced) {
      return invokeMode;
    }

    bool hasInstanceMode = ClipSpecifiesMtMode(clip);
    if (hasInstanceMode) {
      return GetInstanceMode(clip);
    }
    else {
      return invokeMode;
    }
  }

  static bool UsesDefaultMtMode(const PClip& clip, const Function* invokeCall, const InternalEnvironment* env)
  {
    return !ClipSpecifiesMtMode(clip) && !env->FilterHasMtMode(invokeCall);
  }

  void AddChainedFilter(const PClip& clip, MtMode defaultMode)
  {
    MtMode mode = GetInstanceMode(clip, defaultMode);
    Accumulate(mode);
  }
};


OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type)
  : _type(type)
{}

OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type, const Function* func)
  : _type(type), _function(func)
{}

OneTimeLogTicket::OneTimeLogTicket(ELogTicketType type, const std::string& str)
  : _type(type), _string(str)
{}

bool OneTimeLogTicket::operator==(const OneTimeLogTicket& other) const
{
  return (_type == other._type)
    && (_function == other._function)
    && (_string.compare(other._string) == 0);
}

namespace std
{
  template <>
  struct hash<OneTimeLogTicket>
  {
    std::size_t operator()(const OneTimeLogTicket& k) const
    {
      // TODO: This is a pretty poor combination function for hashes.
      // Find something better than a simple XOR.
      return hash<int>()(k._type)
        ^ hash<void*>()((void*)k._function)
        ^ hash<std::string>()((std::string)k._string);
    }
  };
}

#include "vartable.h"
#include "ThreadPool.h"
#include <map>
#include <unordered_set>
#include <atomic>
#include <stack>
#include "Prefetcher.h"
#include "BufferPool.h"
#include "ScriptEnvironmentTLS.h"

class ThreadScriptEnvironment;
class ScriptEnvironment {
public:
  ScriptEnvironment();
  void CheckVersion(int version);
  int GetCPUFlags();
  void AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data = 0);
  void AddFunction25(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data = 0);
  bool FunctionExists(const char* name);
  PVideoFrame NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device);
  PVideoFrame NewVideoFrameOnDevice(int row_size, int height, int align, Device* device);
  PVideoFrame NewVideoFrame(const VideoInfo& vi, const PDevice& device);
  // variant #3, with frame property source
  PVideoFrame NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device, PVideoFrame *propSrc);
  // variant #1, with frame property source
  PVideoFrame NewVideoFrameOnDevice(int row_size, int height, int align, Device* device, PVideoFrame* propSrc);
  // variant #2, with frame property source
  PVideoFrame NewVideoFrame(const VideoInfo& vi, const PDevice& device, PVideoFrame* propSrc);

  PVideoFrame NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, Device* device);

  bool MakeWritable(PVideoFrame* pvf);
  void BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height);
  void AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data);
  PVideoFrame Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height);
  int SetMemoryMax(int mem);
  int SetWorkingDir(const char* newdir);
  AVSC_CC ~ScriptEnvironment();
  void* ManageCache(int key, void* data);
  bool PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key);
  PVideoFrame SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV);
  void DeleteScriptEnvironment();
  void ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor);
  const AVS_Linkage* GetAVSLinkage();

  // alpha support
  PVideoFrame NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, bool alpha, Device* device);
  PVideoFrame SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA);
  PVideoFrame SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA);

  void copyFrameProps(const PVideoFrame& src, PVideoFrame& dst);
  const AVSMap* getFramePropsRO(const PVideoFrame& frame) AVS_NOEXCEPT;
  AVSMap* getFramePropsRW(PVideoFrame& frame) AVS_NOEXCEPT;
  int propNumKeys(const AVSMap* map) AVS_NOEXCEPT;
  const char* propGetKey(const AVSMap* map, int index) AVS_NOEXCEPT;
  int propNumElements(const AVSMap* map, const char* key) AVS_NOEXCEPT;
  char propGetType(const AVSMap* map, const char* key) AVS_NOEXCEPT;
  int propDeleteKey(AVSMap* map, const char* key) AVS_NOEXCEPT;
  int64_t propGetInt(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  double propGetFloat(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  const char* propGetData(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  int propGetDataSize(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  PClip propGetClip(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  const PVideoFrame propGetFrame(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT;
  int propSetInt(AVSMap* map, const char* key, int64_t i, int append) AVS_NOEXCEPT;
  int propSetFloat(AVSMap* map, const char* key, double d, int append) AVS_NOEXCEPT;
  int propSetData(AVSMap* map, const char* key, const char* d, int length, int append) AVS_NOEXCEPT;
  int propSetClip(AVSMap* map, const char* key, PClip& clip, int append) AVS_NOEXCEPT;
  int propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append) AVS_NOEXCEPT;

  const int64_t* propGetIntArray(const AVSMap* map, const char* key, int* error) AVS_NOEXCEPT;
  const double* propGetFloatArray(const AVSMap* map, const char* key, int* error) AVS_NOEXCEPT;
  int propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size) AVS_NOEXCEPT;
  int propSetFloatArray(AVSMap* map, const char* key, const double* d, int size) AVS_NOEXCEPT;

  AVSMap* createMap() AVS_NOEXCEPT;
  void freeMap(AVSMap* map) AVS_NOEXCEPT;
  void clearMap(AVSMap* map) AVS_NOEXCEPT;

  PVideoFrame NewVideoFrame(const VideoInfo& vi, PVideoFrame* propSrc, int align = FRAME_ALIGN);

  bool MakePropertyWritable(PVideoFrame* pvf); // V9

  /* IScriptEnvironment2 */
  bool LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result);
  void AddAutoloadDir(const char* dirPath, bool toFront);
  void ClearAutoloadDirs();
  void AutoloadPlugins();
  void AddFunction(const char* name, const char* params, INeoEnv::ApplyFunc apply, void* user_data, const char *exportVar);
  bool InternalFunctionExists(const char* name);
  void AdjustMemoryConsumption(size_t amount, bool minus);
  void SetFilterMTMode(const char* filter, MtMode mode, bool force);
  void SetFilterMTMode(const char* filter, MtMode mode, MtWeight weight);
  MtMode GetFilterMTMode(const Function* filter, bool* is_forced) const;
  void ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion);
  IJobCompletion* NewCompletion(size_t capacity);
  size_t  GetEnvProperty(AvsEnvProperty prop);
  ClipDataStore* ClipData(IClip* clip);
  MtMode GetDefaultMtMode() const;
  bool FilterHasMtMode(const Function* filter) const;
  void SetLogParams(const char* target, int level);
  void LogMsg(int level, const char* fmt, ...);
  void LogMsg_valist(int level, const char* fmt, va_list va);
  void LogMsgOnce(const OneTimeLogTicket& ticket, int level, const char* fmt, ...);
  void LogMsgOnce_valist(const OneTimeLogTicket& ticket, int level, const char* fmt, va_list va);

  void SetMaxCPU(const char *features); // fixme: why is here InternalEnvironment?

  /* INeoEnv */
  bool Invoke_(AVSValue *result, const AVSValue& implicit_last,
    const char* name, const Function *f, const AVSValue& args, const char* const* arg_names,
    InternalEnvironment* env_thread, bool is_runtime);

  PDevice GetDevice(AvsDeviceType device_type, int device_index) const;
  int SetMemoryMax(AvsDeviceType type, int index, int mem);

  PVideoFrame GetOnDeviceFrame(const PVideoFrame& src, Device* device);
  void ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment *env);
  ThreadPool* NewThreadPool(size_t nThreads);
  void SetGraphAnalysis(bool enable) { graphAnalysisEnable = enable; }

  void IncEnvCount() { InterlockedIncrement(&EnvCount); }
  void DecEnvCount() { InterlockedDecrement(&EnvCount); }

  ConcurrentVarStringFrame* GetTopFrame() { return &top_frame; }
  void SetCacheMode(CacheMode mode) { cacheMode = mode; }
  CacheMode GetCacheMode() { return cacheMode; }
  void SetDeviceOpt(DeviceOpt opt, int val);

  void UpdateFunctionExports(const char* funcName, const char* funcParams, const char* exportVar);

  ThreadScriptEnvironment* GetMainThreadEnv() { return threadEnv.get(); }

private:
  typedef IScriptEnvironment::NotFound NotFound;
  typedef IScriptEnvironment::ApplyFunc ApplyFunc;

  // Tritical May 2005
  // Note order here!!
  // AtExiter has functions which
  // rely on StringDump elements.
  ConcurrentVarStringFrame top_frame;
  std::unique_ptr<ThreadScriptEnvironment> threadEnv;
  std::mutex string_mutex;

  AtExiter at_exit;
  ThreadPool* thread_pool;

  PluginManager* plugin_manager;
  std::recursive_mutex plugin_mutex;

  long EnvCount; // for ThreadScriptEnvironment leak detection

  void ThrowError(const char* fmt, ...);
  void VThrowError(const char* fmt, va_list va);

  const Function* Lookup(const char* search_name, const AVSValue* args, size_t num_args,
    bool &pstrict, size_t args_names_count, const char* const* arg_names, IScriptEnvironment2* env_thread);
  bool CheckArguments(const Function* f, const AVSValue* args, size_t num_args,
    bool &pstrict, size_t args_names_count, const char* const* arg_names);
  std::unordered_map<IClip*, ClipDataStore> clip_data;

  void ExportBuiltinFilters();

  bool PlanarChromaAlignmentState;

  long hrfromcoinit;
  uint32_t coinitThreadId;

  struct DebugTimestampedFrame
  {
    VideoFrame* frame;
    AVSMap* properties;

#ifdef _DEBUG
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
#endif

    DebugTimestampedFrame(VideoFrame* _frame,
      AVSMap* _properties
    )
      : frame(_frame)
      , properties(_properties)

#ifdef _DEBUG
      , timestamp(std::chrono::high_resolution_clock::now())
#endif
    {}
  };
  class VFBStorage : public VideoFrameBuffer {
  public:
    int free_count;
    int margin;
    PGraphMemoryNode memory_node;
    VFBStorage()
      : VideoFrameBuffer(),
      free_count(0)
    { }
    VFBStorage(int size, int margin, Device* device)
      : VideoFrameBuffer(size, margin, device),
      free_count(0),
      margin(margin)
    { }
    void Attach(FilterGraphNode* node) {
      if (memory_node) {
        memory_node->OnFree(data_size, device);
        memory_node = nullptr;
      }
      if (node != nullptr) {
        memory_node = node->GetMemoryNode();
        memory_node->OnAllocate(data_size, device);
      }
    }
    ~VFBStorage() {
      if (memory_node) {
        memory_node->OnFree(data_size, device);
        memory_node = nullptr;
      }
#ifdef _DEBUG
      if (data && device->device_type == DEV_TYPE_CPU) {
        // check buffer overrun
        int *pInt = (int *)(data + margin + data_size);
        if (pInt[0] != 0xDEADBEEF ||
          pInt[1] != 0xDEADBEEF ||
          pInt[2] != 0xDEADBEEF ||
          pInt[3] != 0xDEADBEEF)
        {
          printf("Buffer overrun!!!\n");
        }
      }
#endif
    }
  };
  typedef std::vector<DebugTimestampedFrame> VideoFrameArrayType;
  typedef std::map<VideoFrameBuffer *, VideoFrameArrayType> FrameBufferRegistryType;
  typedef std::map<size_t, FrameBufferRegistryType> FrameRegistryType2; // post r1825 P.F.
  typedef mapped_list<Cache*> CacheRegistryType;


  FrameRegistryType2 FrameRegistry2; // P.F.
#ifdef _DEBUG
  void ListFrameRegistry(size_t min_size, size_t max_size, bool someframes);
#endif

  std::unique_ptr<DeviceManager> Devices;
  CacheRegistryType CacheRegistry;
  Cache* FrontCache;
  VideoFrame* GetNewFrame(size_t vfb_size, size_t margin, Device* device);
  VideoFrame* GetFrameFromRegistry(size_t vfb_size, Device* device);
  void ShrinkCache(Device* device);
  VideoFrame* AllocateFrame(size_t vfb_size, size_t margin, Device* device);
  std::recursive_mutex memory_mutex;

  int frame_align;
  int plane_align;

  //BufferPool BufferPool;

  typedef std::vector<MTGuard*> MTGuardRegistryType;
  MTGuardRegistryType MTGuardRegistry;

  std::vector <std::unique_ptr<ThreadPool>> ThreadPoolRegistry;
  size_t nTotalThreads;
  size_t nMaxFilterInstances;

  // Members used to reconstruct Association between Invoke() calls and filter instances
  std::stack<MtModeEvaluator*> invoke_stack;

  // MT mode specifications
  std::unordered_map<std::string, std::pair<MtMode, MtWeight>> MtMap;
  MtMode DefaultMtMode = MtMode::MT_MULTI_INSTANCE;
  static const std::string DEFAULT_MODE_SPECIFIER;

  // Logging-related members
  int LogLevel;
  std::string LogTarget;
  std::ofstream LogFileStream;
  std::unordered_set<OneTimeLogTicket> LogTickets;

  // filter graph
  bool graphAnalysisEnable;

  typedef std::vector<FilterGraphNode*> GraphNodeRegistryType;
  GraphNodeRegistryType GraphNodeRegistry;

  CacheMode cacheMode;

  void InitMT();
};
const std::string ScriptEnvironment::DEFAULT_MODE_SPECIFIER = "DEFAULT_MT_MODE";

// Only ThrowError and Sprintf is implemented(Used by destructor)
class MinimumScriptEnvironment : public IScriptEnvironment {
  VarTable var_table;
public:
  MinimumScriptEnvironment(ConcurrentVarStringFrame* top_frame) : var_table(top_frame) { }
  virtual ~MinimumScriptEnvironment() {}
  virtual int __stdcall GetCPUFlags() {
    throw AvisynthError("Not Implemented");
  }
  virtual char* __stdcall SaveString(const char* s, int length = -1) {
    return var_table.SaveString(s, length);
  }
  virtual char* Sprintf(const char* fmt, ...) {
    va_list val;
    va_start(val, fmt);
    char* result = VSprintf(fmt, val);
    va_end(val);
    return result;
  }
  virtual char* __stdcall VSprintf(const char* fmt, va_list val) {
    try {
      std::string str = FormatString(fmt, val);
      return var_table.SaveString(str.c_str(), int(str.size())); // SaveString will add the NULL in len mode.
    }
    catch (...) {
      return NULL;
    }
  }
  __declspec(noreturn) virtual void ThrowError(const char* fmt, ...) {
    va_list val;
    va_start(val, fmt);
    VThrowError(fmt, val);
    va_end(val);
  }
  void __stdcall VThrowError(const char* fmt, va_list va)
  {
    std::string msg;
    try {
      msg = FormatString(fmt, va);
    }
    catch (...) {
      msg = "Exception while processing ScriptEnvironment::ThrowError().";
    }
    // Throw...
    throw AvisynthError(var_table.SaveString(msg.c_str()));
  }
  virtual void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data) {
    throw AvisynthError("Not Implemented");
  }
  virtual bool __stdcall FunctionExists(const char* name) {
    throw AvisynthError("Not Implemented");
  }
  virtual AVSValue __stdcall Invoke(const char* name, const AVSValue args, const char* const* arg_names = 0) {
    throw AvisynthError("Not Implemented");
  }
  virtual AVSValue __stdcall GetVar(const char* name) {
    throw AvisynthError("Not Implemented");
  }
  virtual bool __stdcall SetVar(const char* name, const AVSValue& val) {
    throw AvisynthError("Not Implemented");
  }
  virtual bool __stdcall SetGlobalVar(const char* name, const AVSValue& val) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall PushContext(int level = 0) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall PopContext() {
    throw AvisynthError("Not Implemented");
  }
  virtual PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align = FRAME_ALIGN) {
    throw AvisynthError("Not Implemented");
  }
  virtual bool __stdcall MakeWritable(PVideoFrame* pvf) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall AtExit(ShutdownFunc function, void* user_data) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall CheckVersion(int version = AVISYNTH_INTERFACE_VERSION) {
    throw AvisynthError("Not Implemented");
  }
  virtual PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) {
    throw AvisynthError("Not Implemented");
  }
  virtual int __stdcall SetMemoryMax(int mem) {
    throw AvisynthError("Not Implemented");
  }
  virtual int __stdcall SetWorkingDir(const char * newdir) {
    throw AvisynthError("Not Implemented");
  }
  virtual void* __stdcall ManageCache(int key, void* data) {
    throw AvisynthError("Not Implemented");
  }
  virtual bool __stdcall PlanarChromaAlignment(PlanarChromaAlignmentMode key) {
    throw AvisynthError("Not Implemented");
  }
  virtual PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
    int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall DeleteScriptEnvironment() {
    throw AvisynthError("Not Implemented");
  }
  virtual void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size,
    int textcolor, int halocolor, int bgcolor) {
    throw AvisynthError("Not Implemented");
  }
  virtual const AVS_Linkage* __stdcall GetAVSLinkage() {
    throw AvisynthError("Not Implemented");
  }
  virtual AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def = AVSValue()) {
    throw AvisynthError("Not Implemented");
  }
};

/* ---------------------------------------------------------------------------------
*  Per thread data
* ---------------------------------------------------------------------------------
*/
struct ScriptEnvironmentTLS
{
  const int thread_id;
  // PF 161223 why do we need thread-local global variables?
  // comment remains here until it gets cleared, anyway, I make it of no use
  VarTable var_table;
  BufferPool buffer_pool;
  Device* currentDevice;
  bool closing;                 // Used to avoid deadlock, if vartable is being accessed while shutting down (Popcontext)
  bool supressCaching;
  int ImportDepth;
  int getFrameRecursiveCount;

  // Concurrent GetFrame with Invoke causes deadlock.
  // Increment this variable when Invoke running
  // to prevent submitting job to threadpool
  int suppressThreadCount;

  FilterGraphNode* currentGraphNode;
  volatile long refcount;

  ScriptEnvironmentTLS(int thread_id, InternalEnvironment* core)
    : thread_id(thread_id)
    , var_table(core->GetTopFrame())
    , buffer_pool(core)
    , currentDevice(NULL)
    , closing(false)
    , supressCaching(false)
    , ImportDepth(0)
    , getFrameRecursiveCount(0)
    , suppressThreadCount(0)
    , currentGraphNode(nullptr)
    , refcount(1)
  {
  }
};

// per thread data is bound to a thread (not ThreadScriptEnvironment)
// since some filter (e.g. svpflow1) ignores env given for GetFrame, and always use main thread's env.
// this is a work-around for that.
#if defined(AVS_WINDOWS) && !defined(__GNUC__)
#  ifdef XP_TLS
extern DWORD dwTlsIndex;
#  else
// does not work on XP when DLL is dynamic loaded. see dwTlsIndex instead
__declspec(thread) ScriptEnvironmentTLS* g_TLS = nullptr;
#  endif
#else
__thread ScriptEnvironmentTLS* g_TLS = nullptr;
#endif

class ThreadScriptEnvironment : public InternalEnvironment
{
  ScriptEnvironment* core;
  ScriptEnvironmentTLS* coreTLS;
  ScriptEnvironmentTLS myTLS;
public:

  ThreadScriptEnvironment(int thread_id, ScriptEnvironment* core, ScriptEnvironmentTLS* coreTLS)
    : core(core)
    , coreTLS(coreTLS)
    , myTLS(thread_id, this)
  {
    if (coreTLS == nullptr) {
      // when this is main thread TLS
      this->coreTLS = &myTLS;
    }
    if (thread_id != 0) {
      // thread pool thread
#ifdef XP_TLS
      ScriptEnvironmentTLS* g_TLS = (ScriptEnvironmentTLS*)(TlsGetValue(dwTlsIndex));
#endif
      if (g_TLS != nullptr) {
        ThrowError("Detected multiple ScriptEnvironmentTLSs for a single thread");
      }
      g_TLS = &myTLS;
#ifdef XP_TLS
      if (!TlsSetValue(dwTlsIndex, g_TLS)) {
        ThrowError("Could not store thread local value for ScriptEnvironmentTLS");
      }
#endif
    }
    core->IncEnvCount(); // for leak detection
  }

  ~ThreadScriptEnvironment() {
    core->DecEnvCount(); // for leak detection
  }

  ScriptEnvironmentTLS* GetTLS() { return &myTLS; }

#ifdef XP_TLS
  // a ? : b, evaluate 'a' only once
#define IFNULL(a, b) ([&](){ auto val = (a); return ((val) == nullptr ? (b) : (val)); }())
#define DISPATCH(name) IFNULL((ScriptEnvironmentTLS*)(TlsGetValue(dwTlsIndex)), coreTLS)->name
#else
#define DISPATCH(name) (g_TLS ? g_TLS : coreTLS)->name
#endif


  AVSValue __stdcall GetVar(const char* name)
  {
    if (DISPATCH(closing)) return AVSValue();  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (DISPATCH(var_table).Get(name, &val))
      return val;
    else
      throw IScriptEnvironment::NotFound();
  }

  bool __stdcall SetVar(const char* name, const AVSValue& val)
  {
    if (DISPATCH(closing)) return true;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).Set(name, val);
  }

  bool __stdcall SetGlobalVar(const char* name, const AVSValue& val)
  {
    if (DISPATCH(closing)) return true;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).SetGlobal(name, val);
  }

  void __stdcall PushContext(int level = 0)
  {
    DISPATCH(var_table).Push();
  }

  void __stdcall PopContext()
  {
    DISPATCH(var_table).Pop();
  }

  void __stdcall PushContextGlobal()
  {
    DISPATCH(var_table).PushGlobal();
  }

  void __stdcall PopContextGlobal()
  {
    DISPATCH(var_table).PopGlobal();
  }

  bool __stdcall GetVarTry(const char* name, AVSValue* val) const
  {
    if (DISPATCH(closing)) return false;  // We easily risk  being inside the critical section below, while deleting variables.
    return DISPATCH(var_table).Get(name, val);
  }

  AVSValue __stdcall GetVarDef(const char* name, const AVSValue& def)
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return val;
    else
      return def;
  }

  bool __stdcall GetVarBool(const char* name, bool def) const
  {
    if (DISPATCH(closing)) return false;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return val.AsBool(def);
    else
      return def;
  }

  int __stdcall GetVarInt(const char* name, int def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return val.AsInt(def);
    else
      return def;
  }

  double __stdcall GetVarDouble(const char* name, double def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return val.AsDblDef(def);
    else
      return def;
  }

  const char* __stdcall GetVarString(const char* name, const char* def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return val.AsString(def);
    else
      return def;
  }

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4244) // conversion from __int64, possible loss of data
#endif
  int64_t __stdcall GetVarLong(const char* name, int64_t def) const
  {
    if (DISPATCH(closing)) return def;  // We easily risk  being inside the critical section below, while deleting variables.
    AVSValue val;
    if (this->GetVarTry(name, &val))
      return (int)(val.AsInt(def)); // until we have int64
    else
      return def;
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

  void* __stdcall Allocate(size_t nBytes, size_t alignment, AvsAllocType type)
  {
    if ((type != AVS_NORMAL_ALLOC) && (type != AVS_POOLED_ALLOC))
      return NULL;
    return DISPATCH(buffer_pool).Allocate(nBytes, alignment, type == AVS_POOLED_ALLOC);
  }

  void __stdcall Free(void* ptr)
  {
    DISPATCH(buffer_pool).Free(ptr);
  }

  Device* __stdcall GetCurrentDevice() const
  {
    return DISPATCH(currentDevice);
  }

  Device* __stdcall SetCurrentDevice(Device* device)
  {
    Device* old = DISPATCH(currentDevice);
    DISPATCH(currentDevice) = device;
    return old;
  }

  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, int align)
  {
    return core->NewVideoFrameOnDevice(vi, align, DISPATCH(currentDevice));
  }

  PVideoFrame __stdcall NewVideoFrameP(const VideoInfo& vi, PVideoFrame *propSrc, int align)
  {
    return core->NewVideoFrameOnDevice(vi, align, DISPATCH(currentDevice), propSrc);
  }

  void* __stdcall GetDeviceStream()
  {
    return DISPATCH(currentDevice)->GetComputeStream();
  }

  void __stdcall DeviceAddCallback(void(*cb)(void*), void* user_data)
  {
    DeviceCompleteCallbackData cbdata = { cb, user_data };
    DISPATCH(currentDevice)->AddCompleteCallback(cbdata);
  }

  PVideoFrame __stdcall GetFrame(PClip c, int n, const PDevice& device)
  {
    DeviceSetter setter(this, (Device*)(void*)device);
    return c->GetFrame(n, this);
  }


  /* ---------------------------------------------------------------------------------
  *             S T U B S
  * ---------------------------------------------------------------------------------
  */

  bool __stdcall InternalFunctionExists(const char* name)
  {
    return core->InternalFunctionExists(name);
  }

  void __stdcall AdjustMemoryConsumption(size_t amount, bool minus)
  {
    core->AdjustMemoryConsumption(amount, minus);
  }

  void __stdcall CheckVersion(int version)
  {
    core->CheckVersion(version);
  }

  int __stdcall GetCPUFlags()
  {
    return core->GetCPUFlags();
  }

  char* __stdcall SaveString(const char* s, int length = -1)
  {
    return DISPATCH(var_table).SaveString(s, length);
  }

  char* __stdcall SaveString(const char* s, int length, bool escape)
  {
    return DISPATCH(var_table).SaveString(s, length, escape);
  }

  char* Sprintf(const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    // do not call core->Sprintf, because cannot pass ... further
    char* result = VSprintf(fmt, val);
    va_end(val);
    return result;
  }

  char* __stdcall VSprintf(const char* fmt, va_list val)
  {
    try {
      std::string str = FormatString(fmt, val);
      return DISPATCH(var_table).SaveString(str.c_str(), int(str.size())); // SaveString will add the NULL in len mode.
    }
    catch (...) {
      return NULL;
    }
  }

  void ThrowError(const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    VThrowError(fmt, val);
    va_end(val);
  }

  void __stdcall VThrowError(const char* fmt, va_list va)
  {
    std::string msg;
    try {
      msg = FormatString(fmt, va);
    }
    catch (...) {
      msg = "Exception while processing ScriptEnvironment::ThrowError().";
    }

    // Also log the error before throwing
    this->LogMsg(LOGLEVEL_ERROR, msg.c_str());

    // Throw...
    throw AvisynthError(DISPATCH(var_table).SaveString(msg.c_str()));
  }

  PVideoFrame __stdcall SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA)
  {
    return core->SubframePlanarA(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
  }

  bool __stdcall MakePropertyWritable(PVideoFrame* pvf)
  {
    return core->MakePropertyWritable(pvf);
  }

  void __stdcall copyFrameProps(const PVideoFrame& src, PVideoFrame& dst)
  {
    core->copyFrameProps(src, dst);
  }

  const AVSMap* __stdcall getFramePropsRO(const PVideoFrame& frame)
  {
    return core->getFramePropsRO(frame);
  }
  AVSMap* __stdcall getFramePropsRW(PVideoFrame& frame)
  {
    return core->getFramePropsRW(frame);
  }
  int __stdcall propNumKeys(const AVSMap* map)
  {
    return core->propNumKeys(map);
  }
  const char* __stdcall propGetKey(const AVSMap* map, int index)
  {
    return core->propGetKey(map, index);
  }
  int __stdcall propNumElements(const AVSMap* map, const char* key)
  {
    return core->propNumElements(map, key);
  }
  char __stdcall propGetType(const AVSMap* map, const char* key)
  {
    return core->propGetType(map, key);
  }
  int __stdcall propDeleteKey(AVSMap* map, const char* key)
  {
    return core->propDeleteKey(map, key);
  }
  int64_t __stdcall propGetInt(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetInt(map, key, index, error);
  }
  double __stdcall propGetFloat(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetFloat(map, key, index, error);
  }
  const char* __stdcall propGetData(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetData(map, key, index, error);
  }
  int __stdcall propGetDataSize(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetDataSize(map, key, index, error);
  }
  PClip __stdcall propGetClip(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetClip(map, key, index, error);
  }
  const PVideoFrame __stdcall propGetFrame(const AVSMap* map, const char* key, int index, int* error)
  {
    return core->propGetFrame(map, key, index, error);
  }
  int __stdcall propSetInt(AVSMap* map, const char* key, int64_t i, int append)
  {
    return core->propSetInt(map, key, i, append);
  }
  int __stdcall propSetFloat(AVSMap* map, const char* key, double d, int append)
  {
    return core->propSetFloat(map, key, d, append);
  }
  int __stdcall propSetData(AVSMap* map, const char* key, const char* d, int length, int append)
  {
    return core->propSetData(map, key, d, length, append);
  }
  int __stdcall propSetClip(AVSMap* map, const char* key, PClip& clip, int append)
  {
    return core->propSetClip(map, key, clip, append);
  }
  int __stdcall propSetFrame(AVSMap* map, const char* key, const PVideoFrame& frame, int append)
  {
    return core->propSetFrame(map, key, frame, append);
  }

  const int64_t* __stdcall propGetIntArray(const AVSMap* map, const char* key, int* error)
  {
    return core->propGetIntArray(map, key, error);
  }

  const double* __stdcall propGetFloatArray(const AVSMap* map, const char* key, int* error)
  {
    return core->propGetFloatArray(map, key, error);
  }

  int __stdcall propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size)
  {
    return core->propSetIntArray(map, key, i, size);
  }

  int __stdcall propSetFloatArray(AVSMap* map, const char* key, const double* d, int size)
  {
    return core->propSetFloatArray(map, key, d, size);
  }

  AVSMap* __stdcall createMap()
  {
    return core->createMap();
  }

  void __stdcall freeMap(AVSMap* map)
  {
    core->freeMap(map);
  }

  void __stdcall clearMap(AVSMap* map)
  {
    core->clearMap(map);
  }

  void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data = 0)
  {
    core->AddFunction(name, params, apply, user_data);
  }

  void __stdcall AddFunction25(const char* name, const char* params, ApplyFunc apply, void* user_data = 0)
  {
    core->AddFunction25(name, params, apply, user_data);
  }

  bool __stdcall FunctionExists(const char* name)
  {
    return core->FunctionExists(name);
  }

  bool IsRuntime() {
    // When invoked from GetFrame/GetAudio, skip all cache and mt mecanism
    bool is_runtime = true;

#ifdef XP_TLS
    ScriptEnvironmentTLS* g_TLS = (ScriptEnvironmentTLS*)(TlsGetValue(dwTlsIndex));
#endif
    if (g_TLS == nullptr) { // not called by thread
      if (GetFrameRecursiveCount() == 0) { // not called by GetFrame
        is_runtime = false;
      }
    }

    return is_runtime;
  }

  // thrower Invoke, IScriptEnvironment
  AVSValue __stdcall Invoke(const char* name,
    const AVSValue args, const char* const* arg_names)
  {
    AVSValue result;
    if (!core->Invoke_(&result, AVSValue(), name, nullptr, args, arg_names, this, IsRuntime()))
    {
      throw NotFound();
    }
    return result;
  }

  // thrower Invoke, IScriptEnvironment_Avs25
  AVSValue __stdcall Invoke25(const char* name,
    const AVSValue args, const char* const* arg_names)
  {
#ifndef NEW_AVSVALUE
    return Invoke(name, args, arg_names);
#else
    AVSValue result;
    // MarkArrayAsC: signing for destructor: don't free array elements.
    // Reason: CPP 2.5 plugins have "baked code" in their avisynth.h and do not know 
    // about that arrays (used for parameters) are now deep-copied and deep-free'd.
    // Normally the elements of 'args' parameter array content would be freed up on 
    // Invoke25's function exit. But the caller would free them up again-> crash.
    // So we are assume that the free is on the other (cpp 2.5 plugin) side.
    const bool success = core->Invoke_(&result, AVSValue(), name, nullptr, args, arg_names, this, IsRuntime());

    if (args.IsArray())
      ((AVSValue*)&args)->MarkArrayAsC();

    if (!success)
      throw NotFound();

    return result;
#endif
  }


  //  no-throw Invoke, IScriptEnvironment, Ex-IS2
  bool __stdcall InvokeTry(AVSValue* result,
    const char* name, const AVSValue& args, const char* const* arg_names)
  {
    return core->Invoke_(result, AVSValue(), name, nullptr, args, arg_names, this, IsRuntime());
  }

  // thrower Invoke + implicit last, since IS V8
  // created to have throw and non-throw (xxxxTry) versions from all Invokes
  AVSValue __stdcall Invoke2(const AVSValue& implicit_last,
    const char* name, const AVSValue args, const char* const* arg_names)
  {
    AVSValue result;
    if (!core->Invoke_(&result, implicit_last,
      name, nullptr, args, arg_names, this, IsRuntime()))
    {
      throw NotFound();
    }
    return result;
  }

  // no-throw Invoke + implicit last, Ex-INeo
  bool __stdcall Invoke2Try(AVSValue* result, const AVSValue& implicit_last,
    const char* name, const AVSValue args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last,
      name, nullptr, args, arg_names, this, IsRuntime());
  }

  // thrower Invoke + implicit last + PFunction
  AVSValue __stdcall Invoke3(const AVSValue& implicit_last,
    const PFunction& func, const AVSValue args, const char* const* arg_names)
  {
    AVSValue result;
    if (!core->Invoke_(&result, implicit_last,
      func->GetLegacyName(), func->GetDefinition(), args, arg_names, this, IsRuntime()))
    {
      throw NotFound();
    }
    return result;
  }

  // no-throw Invoke + implicit last + PFunction
  bool __stdcall Invoke3Try(AVSValue *result, const AVSValue& implicit_last,
    const PFunction& func, const AVSValue args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last,
      func->GetLegacyName(), func->GetDefinition(), args, arg_names, this, IsRuntime());
  }

  // King of all Invoke versions: no-throw Invoke + implicit last + funtion name + function definition
  bool __stdcall Invoke_(AVSValue *result, const AVSValue& implicit_last,
    const char* name, const Function *f, const AVSValue& args, const char* const* arg_names)
  {
    return core->Invoke_(result, implicit_last, name, f, args, arg_names, this, IsRuntime());
  }

  bool __stdcall MakeWritable(PVideoFrame* pvf)
  {
    return core->MakeWritable(pvf);
  }

  void __stdcall BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height)
  {
    core->BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
  }

  void __stdcall AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data)
  {
    core->AtExit(function, user_data);
  }

  PVideoFrame __stdcall Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height)
  {
    return core->Subframe(src, rel_offset, new_pitch, new_row_size, new_height);
  }

  int __stdcall SetMemoryMax(int mem)
  {
    return core->SetMemoryMax(mem);
  }

  int __stdcall SetWorkingDir(const char* newdir)
  {
    return core->SetWorkingDir(newdir);
  }

  void* __stdcall ManageCache(int key, void* data)
  {
    if ((MANAGE_CACHE_KEYS)key == MC_QueryAvs25)
      return (intptr_t*)0;
    return core->ManageCache(key, data);
  }

  void* __stdcall ManageCache25(int key, void* data)
  {
    // We use a v2.5-special ManageCache call with special key to query if
    // env ptr is v2.5 even if casted to IScriptEnvironment 
    if ((MANAGE_CACHE_KEYS)key == MC_QueryAvs25)
      return (intptr_t *)1;
    return ManageCache(key, data);
  }

  bool __stdcall PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key)
  {
    return core->PlanarChromaAlignment(key);
  }

  PVideoFrame __stdcall SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV)
  {
    return core->SubframePlanar(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV);
  }

  void __stdcall DeleteScriptEnvironment()
  {
#ifdef XP_TLS
    ScriptEnvironmentTLS* g_TLS = (ScriptEnvironmentTLS*)(TlsGetValue(dwTlsIndex));
#endif
    if (g_TLS != nullptr) {
      ThrowError("Cannot delete environment from a TLS proxy.");
    }
    core->DeleteScriptEnvironment();
  }

  void __stdcall ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor)
  {
    core->ApplyMessage(frame, vi, message, size, textcolor, halocolor, bgcolor);
  }

  const AVS_Linkage* __stdcall GetAVSLinkage()
  {
    return core->GetAVSLinkage();
  }

  /* IScriptEnvironment2 */
  bool __stdcall LoadPlugin(const char* filePath, bool throwOnError, AVSValue* result)
  {
    return core->LoadPlugin(filePath, throwOnError, result);
  }

  void __stdcall AddAutoloadDir(const char* dirPath, bool toFront)
  {
    core->AddAutoloadDir(dirPath, toFront);
  }

  void __stdcall ClearAutoloadDirs()
  {
    core->ClearAutoloadDirs();
  }

  void __stdcall AutoloadPlugins()
  {
    core->AutoloadPlugins();
  }

  void __stdcall AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data, const char* exportVar)
  {
    core->AddFunction(name, params, apply, user_data, exportVar);
  }

  int __stdcall IncrImportDepth()
  {
    return ++DISPATCH(ImportDepth);
  }

  int __stdcall DecrImportDepth()
  {
    return --DISPATCH(ImportDepth);
  }

  size_t  __stdcall GetEnvProperty(AvsEnvProperty prop)
  {
    switch (prop)
    {
    case AEP_THREAD_ID:
      return DISPATCH(thread_id);
    case AEP_SUPPRESS_THREAD:
      return DISPATCH(suppressThreadCount);
    case AEP_GETFRAME_RECURSIVE:
      return DISPATCH(getFrameRecursiveCount);
    default:
      return core->GetEnvProperty(prop);
    }
  }

  void __stdcall SetFilterMTMode(const char* filter, MtMode mode, bool force)
  {
    core->SetFilterMTMode(filter, mode, force);
  }

  MtMode __stdcall GetFilterMTMode(const Function* filter, bool* is_forced) const
  {
    return core->GetFilterMTMode(filter, is_forced);
  }

  bool __stdcall FilterHasMtMode(const Function* filter) const
  {
    return core->FilterHasMtMode(filter);
  }

  IJobCompletion* __stdcall NewCompletion(size_t capacity)
  {
    return core->NewCompletion(capacity);
  }

  void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion)
  {
    core->ParallelJob(jobFunc, jobData, completion, this);
  }

  void __stdcall ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment* env)
  {
    core->ParallelJob(jobFunc, jobData, completion, env);
  }

  ClipDataStore* __stdcall ClipData(IClip* clip)
  {
    return core->ClipData(clip);
  }

  MtMode __stdcall GetDefaultMtMode() const
  {
    return core->GetDefaultMtMode();
  }

  void __stdcall SetLogParams(const char* target, int level)
  {
    core->SetLogParams(target, level);
  }

  // stdcall calling convention is not supported on variadic function
  void LogMsg(int level, const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    core->LogMsg_valist(level, fmt, val);
    va_end(val);
  }

  void __stdcall LogMsg_valist(int level, const char* fmt, va_list va)
  {
    core->LogMsg_valist(level, fmt, va);
  }

  // stdcall calling convention is not supported on variadic function
  void LogMsgOnce(const OneTimeLogTicket& ticket, int level, const char* fmt, ...)
  {
    va_list val;
    va_start(val, fmt);
    core->LogMsgOnce_valist(ticket, level, fmt, val);
    va_end(val);
  }

  void __stdcall LogMsgOnce_valist(const OneTimeLogTicket& ticket, int level, const char* fmt, va_list va)
  {
    core->LogMsgOnce_valist(ticket, level, fmt, va);
  }

  void __stdcall SetMaxCPU(const char *features)
  {
    core->SetMaxCPU(features);
  }

  void __stdcall SetGraphAnalysis(bool enable)
  {
    core->SetGraphAnalysis(enable);
  }

  int __stdcall SetMemoryMax(AvsDeviceType type, int index, int mem)
  {
    return core->SetMemoryMax(type, index, mem);
  }

  PDevice __stdcall GetDevice(AvsDeviceType device_type, int device_index) const
  {
    return core->GetDevice(device_type, device_index);
  }

  PDevice __stdcall GetDevice() const
  {
    return DISPATCH(currentDevice);
  }

  AvsDeviceType __stdcall GetDeviceType() const
  {
    return DISPATCH(currentDevice)->device_type;
  }

  int __stdcall GetDeviceId() const
  {
    return DISPATCH(currentDevice)->device_id;
  }

  int __stdcall GetDeviceIndex() const
  {
    return DISPATCH(currentDevice)->device_index;
  }

  void* __stdcall GetDeviceStream() const
  {
    return DISPATCH(currentDevice)->GetComputeStream();;
  }

  PVideoFrame __stdcall NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device)
  {
    return core->NewVideoFrameOnDevice(vi, align, device);
  }

  // shortcut to the above
  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, DISPATCH(currentDevice));
  }

  // shortcut to the above
  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, const PDevice& device)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, (Device*)(void*)device);
  }

  // variants with frame property source
  PVideoFrame __stdcall NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device, PVideoFrame *propSrc)
  {
    return core->NewVideoFrameOnDevice(vi, align, device, propSrc);
  }

  // shortcut to the above
  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, PVideoFrame* propSrc)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, DISPATCH(currentDevice), propSrc);
  }

  // shortcut to the above
  PVideoFrame __stdcall NewVideoFrame(const VideoInfo& vi, const PDevice& device, PVideoFrame* propSrc)
  {
    return NewVideoFrameOnDevice(vi, FRAME_ALIGN, (Device*)(void*)device, propSrc);
  }

  PVideoFrame __stdcall GetOnDeviceFrame(const PVideoFrame& src, Device* device)
  {
    return core->GetOnDeviceFrame(src, device);
  }


  ThreadPool* __stdcall NewThreadPool(size_t nThreads)
  {
    return core->NewThreadPool(nThreads);
  }


  void __stdcall AddRef() {
    InterlockedIncrement(&DISPATCH(refcount));
  }

  void __stdcall Release() {
    if (InterlockedDecrement(&DISPATCH(refcount)) == 0) {
      delete this;
    }
  }

  void __stdcall IncEnvCount() {
    core->IncEnvCount();
  }

  void __stdcall DecEnvCount() {
    core->DecEnvCount();
  }

  ConcurrentVarStringFrame* __stdcall GetTopFrame()
  {
    return core->GetTopFrame();
  }

  void __stdcall SetCacheMode(CacheMode mode)
  {
    core->SetCacheMode(mode);
  }

  CacheMode __stdcall GetCacheMode()
  {
    return core->GetCacheMode();
  }

  bool& __stdcall GetSupressCaching()
  {
    return DISPATCH(supressCaching);
  }

  void __stdcall SetDeviceOpt(DeviceOpt opt, int val)
  {
    core->SetDeviceOpt(opt, val);
  }

  void __stdcall UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar)
  {
    if (GetThreadId() != 0 || GetFrameRecursiveCount() != 0) {
      // no need to export function at runtime
      return;
    }
    core->UpdateFunctionExports(funcName, funcParams, exportVar);
  }


  InternalEnvironment* __stdcall NewThreadScriptEnvironment(int thread_id)
  {
    return new ThreadScriptEnvironment(thread_id, core, coreTLS);
  }

  int __stdcall GetThreadId() {
    return DISPATCH(thread_id);
  }

  int& __stdcall GetFrameRecursiveCount() {
    return DISPATCH(getFrameRecursiveCount);
  }

  int& __stdcall GetSuppressThreadCount() {
    return DISPATCH(suppressThreadCount);
  }

  FilterGraphNode*& GetCurrentGraphNode() {
    return DISPATCH(currentGraphNode);
  }

#undef DISPATCH
#undef IFNULL
};

#ifdef AVS_POSIX
static uint64_t posix_get_physical_memory() {
  uint64_t ullTotalPhys;
#if defined(AVS_MACOS)
  size_t len;
  sysctlbyname("hw.memsize", nullptr, &len, nullptr, 0);
  int64_t memsize;
  sysctlbyname("hw.memsize", (void*)&memsize, &len, nullptr, 0);
  ullTotalPhys = memsize;
#elif defined(AVS_BSD)
  size_t len;
  int64_t memsize;
#if !defined(__OpenBSD__)
// OpenBSD doesn't have sysctlbyname at all, so this needs to be
// ported to plain sysctl.
  sysctlbyname("hw.physmem", nullptr, &len, nullptr, 0);
  sysctlbyname("hw.physmem", (void*)&memsize, &len, nullptr, 0);
#else
  sysctl((const int *)HW_PHYSMEM, 4, nullptr, &len, nullptr, 0);
  sysctl((const int *)HW_PHYSMEM, 4, (void*)&memsize, &len, nullptr, 0);
#endif
  ullTotalPhys = memsize;
#elif defined(AVS_HAIKU)
  system_info sysinf;
  get_system_info(&sysinf);
  ullTotalPhys = PAGESIZE * sysinf.max_pages;
#else
  // linux
  struct sysinfo info;
  if (sysinfo(&info) != 0) {
    throw AvisynthError("sysinfo: error reading system statistics");
  }
  ullTotalPhys = (uint64_t)info.totalram * info.mem_unit;
#endif
  return ullTotalPhys;
}

static int64_t posix_get_available_memory() {
  int64_t memory;

  long nPageSize = sysconf(_SC_PAGE_SIZE);
  int64_t nAvailablePhysicalPages;

#if defined(AVS_MACOS)
  vm_statistics64_data_t vmstats;
  mach_msg_type_number_t vmstatsz = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info_t)&vmstats, &vmstatsz);
  nAvailablePhysicalPages = vmstats.free_count;
#elif defined(AVS_BSD)
#if !defined(__OpenBSD__)
// OpenBSD does not have sysctlbyname
  size_t nAvailablePhysicalPagesLen = sizeof(nAvailablePhysicalPages);
  sysctlbyname("vm.stats.vm.v_free_count", &nAvailablePhysicalPages, &nAvailablePhysicalPagesLen, NULL, 0);
#endif
#elif defined(AVS_HAIKU)
  system_info sysinf;
  get_system_info(&sysinf);
  nAvailablePhysicalPages = sysinf.free_memory;
#else // Linux
  nAvailablePhysicalPages = sysconf(_SC_AVPHYS_PAGES);
#endif

  memory = nPageSize * nAvailablePhysicalPages;

  return memory;
}
#endif

static uint64_t ConstrainMemoryRequest(uint64_t requested)
{
  // Get system memory information
#ifdef AVS_WINDOWS // needs linux alternative
  MEMORYSTATUSEX memstatus;
  memstatus.dwLength = sizeof(memstatus);
  GlobalMemoryStatusEx(&memstatus);

  // mem_limit is the largest amount of memory that makes sense to use.
  // We don't want to use more than the virtual address space,
  // and we also don't want to start paging to disk.
  uint64_t mem_limit = min(memstatus.ullTotalVirtual, memstatus.ullTotalPhys);

  uint64_t mem_sysreserve = 0;
  if (memstatus.ullTotalPhys > memstatus.ullTotalVirtual)
  {
    // We are probably running on a 32bit OS system where the virtual space is capped to
    // much less than what the system can use, so it is enough to reserve only a small amount.
    mem_sysreserve = 128 * 1024 * 1024ull;
  }
  else
  {
    // We could probably use up all the RAM in our single application,
    // so reserve more to leave some RAM for other apps and the OS too.
    mem_sysreserve = 1024 * 1024 * 1024ull;
  }

  // Cap memory_max to at most mem_sysreserve less than total, but at least to 64MB.
  return clamp(requested, (uint64_t)64 * 1024 * 1024, mem_limit - mem_sysreserve);
#else
  // copied over from AvxSynth, check against current code!!!

  // Check#1
  // AvxSynth returned simply the actual total_available memory
  // this part is trying to fine tune it, considering that
  // - total_available may contain swap area which we do not want to use FIXME: check it!
  // - leave some memory for other processes (1 GB for x64, 128MB for 32 bit)

  uint64_t physical_memory = posix_get_physical_memory();
  uint64_t total_available = posix_get_available_memory();
  // We don't want to use more than the virtual address space,
  // and we also don't want to start paging to disk.
  uint64_t mem_limit = min(total_available, physical_memory);

  // We could probably use up all the RAM in our single application,
  // so reserve more to leave some RAM for other apps and the OS too.
  const bool isX64 = sizeof(void*) == 8;
  uint64_t mem_sysreserve = isX64 ? (uint64_t)1024 * 1024 * 1024 : (uint64_t)128 * 1024 * 1024;

  // Cap memory_max to at most mem_sysreserve less than total, but at least to 64MB.
  uint64_t allowed_memory = clamp(requested, (uint64_t)64 * 1024 * 1024, mem_limit - mem_sysreserve);
#if 0
  const int DIV = 1024 * 1024;
  fprintf(stdout, "requested= %" PRIu64 " MB\r\n", requested / DIV);
  fprintf(stdout, "physical_memory= %" PRIu64 " MB\r\n", physical_memory / DIV);
  fprintf(stdout, "total_available= %" PRIu64 " MB\r\n", total_available / DIV);
  fprintf(stdout, "mem_limit= %" PRIu64 " MB\r\n", mem_limit / DIV);
  fprintf(stdout, "mem_sysreserve= %" PRIu64 " MB\r\n", mem_sysreserve / DIV);
  fprintf(stdout, "allowed_memory= %" PRIu64 " MB\r\n", allowed_memory / DIV);
  /*For a computer with 16GB RAM, 64 bit OS
    No SetMemoryMax, where default max request is 4GB on x64
      requested= 4072 MB
      physical_memory= 16291 MB
      total_available= 7640 MB
      mem_limit= 7640 MB
      mem_sysreserve= 1024 MB
      allowed_memory= 4072 MB
    Using SetmemoryMax(10000)
      requested= 10000 MB
      physical_memory= 16291 MB
      total_available= 7667 MB
      mem_limit= 7667 MB
      mem_sysreserve= 1024 MB
      allowed_memory= 6643 MB
  */
#endif
  return allowed_memory;
#endif

}

IJobCompletion* ScriptEnvironment::NewCompletion(size_t capacity)
{
  return new JobCompletion(capacity);
}

ScriptEnvironment::ScriptEnvironment()
  : threadEnv(),
  at_exit(),
  thread_pool(NULL),
  plugin_manager(NULL),
  EnvCount(0),
  PlanarChromaAlignmentState(true),   // Change to "true" for 2.5.7
  hrfromcoinit(E_FAIL), coinitThreadId(0),
  Devices(),
  FrontCache(NULL),
  nTotalThreads(1),
  nMaxFilterInstances(1),
  graphAnalysisEnable(false)
{
#ifdef XP_TLS
    if(dwTlsIndex == 0)
      throw("ScriptEnvironment: TlsAlloc failed on DLL load");
#endif

  try {
#ifdef AVS_WINDOWS
    // Make sure COM is initialised
    hrfromcoinit = CoInitialize(NULL);
    // If it was already init'd then decrement
    // the use count and leave it alone!
    if (hrfromcoinit == S_FALSE) {
      hrfromcoinit = E_FAIL;
      CoUninitialize();
    }
    // Remember our threadId.
    coinitThreadId = GetCurrentThreadId();
#endif

    threadEnv = std::unique_ptr<ThreadScriptEnvironment>(new ThreadScriptEnvironment(0, this, nullptr));
    Devices = std::unique_ptr<DeviceManager>(new DeviceManager(threadEnv.get()));

    // calc frame align
    frame_align = plane_align = FRAME_ALIGN;
#ifdef ENABLE_CUDA
    for (int i = 0, end = Devices->GetNumDevices(DEV_TYPE_CUDA); i < end; ++i) {
      int align, pitchAlign;
      Devices->GetDevice(DEV_TYPE_CUDA, i)->GetAlignmentRequirement(&align, &pitchAlign);
      frame_align = max(frame_align, pitchAlign);
      plane_align = max(plane_align, align);
    }
#endif



    auto cpuDevice = Devices->GetCPUDevice();
    threadEnv->GetTLS()->currentDevice = cpuDevice;

#ifdef AVS_WINDOWS
    MEMORYSTATUSEX memstatus;
    memstatus.dwLength = sizeof(memstatus);
    GlobalMemoryStatusEx(&memstatus);
    cpuDevice->memory_max = ConstrainMemoryRequest(memstatus.ullTotalPhys / 4);
#endif
#ifdef AVS_POSIX
    uint64_t ullTotalPhys = posix_get_physical_memory();
    cpuDevice->memory_max = ConstrainMemoryRequest(ullTotalPhys / 4);
    // fprintf(stdout, "Total physical memory= %" PRIu64 ", after constraint=%" PRIu64 "\r\n", ullTotalPhys, memory_max);
    // Total physical memory = 17083355136, after constraint = 7274700800
#endif
    const bool isX64 = sizeof(void*) == 8;
    cpuDevice->memory_max = min(cpuDevice->memory_max, (uint64_t)((isX64 ? 4096 : 1024) * (1024 * 1024ull)));  // at start, cap memory usage to 1GB(x86)/4GB (x64)
    cpuDevice->memory_used = 0ull;

    top_frame.Set("true", true);
    top_frame.Set("false", false);
    top_frame.Set("yes", true);
    top_frame.Set("no", false);
    top_frame.Set("last", AVSValue());

    top_frame.Set("$ScriptName$", AVSValue());
    top_frame.Set("$ScriptFile$", AVSValue());
    top_frame.Set("$ScriptDir$", AVSValue());
    top_frame.Set("$ScriptNameUtf8$", AVSValue());
    top_frame.Set("$ScriptFileUtf8$", AVSValue());
    top_frame.Set("$ScriptDirUtf8$", AVSValue());

    plugin_manager = new PluginManager(threadEnv.get());
#ifdef AVS_WINDOWS
    plugin_manager->AddAutoloadDir("USER_PLUS_PLUGINS", false);
    plugin_manager->AddAutoloadDir("MACHINE_PLUS_PLUGINS", false);
    plugin_manager->AddAutoloadDir("USER_CLASSIC_PLUGINS", false);
    plugin_manager->AddAutoloadDir("MACHINE_CLASSIC_PLUGINS", false);
#else
    // system_avs_plugindir relies on install path, it and user_avs_plugindir_configurable get
    // defined in avisynth_conf.h.in when configuring.

    std::string user_avs_plugindir = std::getenv("HOME");
    std::string user_avs_dirname = "/.avisynth";
    user_avs_plugindir.append(user_avs_dirname);

    plugin_manager->AddAutoloadDir(user_avs_plugindir, false);
    plugin_manager->AddAutoloadDir(user_avs_plugindir_configurable, false);
    plugin_manager->AddAutoloadDir(system_avs_plugindir, false);
#endif

    top_frame.Set("LOG_ERROR", (int)LOGLEVEL_ERROR);
    top_frame.Set("LOG_WARNING", (int)LOGLEVEL_WARNING);
    top_frame.Set("LOG_INFO",    (int)LOGLEVEL_INFO);
    top_frame.Set("LOG_DEBUG",   (int)LOGLEVEL_DEBUG);

    top_frame.Set("DEV_TYPE_CPU", (int)DEV_TYPE_CPU);
    top_frame.Set("DEV_TYPE_CUDA", (int)DEV_TYPE_CUDA);

    top_frame.Set("CACHE_FAST_START", (int)CACHE_FAST_START);
    top_frame.Set("CACHE_OPTIMAL_SIZE", (int)CACHE_OPTIMAL_SIZE);
    top_frame.Set("DEV_CUDA_PINNED_HOST", (int)DEV_CUDA_PINNED_HOST);
    top_frame.Set("DEV_FREE_THRESHOLD", (int)DEV_FREE_THRESHOLD);

    InitMT();
    thread_pool = new ThreadPool(std::thread::hardware_concurrency(), 1, threadEnv.get());

    ExportBuiltinFilters();

    clip_data.max_load_factor(0.8f);
    LogTickets.max_load_factor(0.8f);
  }
  catch (const AvisynthError& err) {
#ifdef AVS_WINDOWS
    if (SUCCEEDED(hrfromcoinit)) {
      hrfromcoinit = E_FAIL;
      CoUninitialize();
    }
#endif
    // Needs must, to not loose the text we
    // must leak a little memory.
    throw AvisynthError(_strdup(err.msg));
  }
}

MtMode ScriptEnvironment::GetDefaultMtMode() const
{
  return DefaultMtMode;
}

void ScriptEnvironment::InitMT()
{
  top_frame.Set("MT_NICE_FILTER", (int)MT_NICE_FILTER);
  top_frame.Set("MT_MULTI_INSTANCE", (int)MT_MULTI_INSTANCE);
  top_frame.Set("MT_SERIALIZED", (int)MT_SERIALIZED);
  top_frame.Set("MT_SPECIAL_MT", (int)MT_SPECIAL_MT);
}

ScriptEnvironment::~ScriptEnvironment() {

  _RPT0(0, "~ScriptEnvironment() called.\n");

  auto tls = threadEnv->GetTLS();
  tls->closing = true;

  // Before we start to pull the world apart
  // give every one their last wish.
  at_exit.Execute(threadEnv.get());

  delete thread_pool;

  tls->var_table.Clear();
  top_frame.Clear();

  // There can be a circular reference between the Prefetcher and the
  // TLS PopContext() variables of the threads started by it. Normally
  // this doesn't happen, but it can for example when somebody
  // sets 'last' in a TLS (see ScriptClip for a specific example).
  // This circular reference causes leaks, so we call
  // Destroy() on the prefetcher, which will in turn terminate all
  // its TLS stuff and break the chain.
  for (auto& pool : ThreadPoolRegistry) {
    pool->Join();
  }
  ThreadPoolRegistry.clear();

  // delete ThreadScriptEnvironment
  threadEnv = nullptr;

  // check ThreadScriptEnvironment leaks
  if (EnvCount > 0) {
    LogMsg(LOGLEVEL_WARNING, "ThreadScriptEnvironment leaks.");
  }

#if 0
  // check clip leaks DoDumpGraph
  if (std::find_if(GraphNodeRegistry.begin(), GraphNodeRegistry.end(),
    [](FilterGraphNode* node) { return node != nullptr; }) != GraphNodeRegistry.end())
  {
    // This is dangerous operation because thread's string is destroyed
    // and may be there are dangling string pointer which results in access violation.
    MinimumScriptEnvironment env(&top_frame);
    DoDumpGraph(GraphNodeRegistry, "clip_leaks.txt", &env);
  }
#endif

  // delete avsmap
  for (auto &it: FrameRegistry2)
  {
    for (auto &it2: it.second)
  {
      for (auto &it3: it2.second)
      {
        delete it3.properties;
        it3.properties = 0;
        it3.frame->properties = 0; // fixme ??
      }
    }
  }

#ifdef _DEBUG
  // LogMsg(LOGLEVEL_DEBUG, "We are before FrameRegistryCleanup");
  // ListFrameRegistry(0,10000000000000ull, true, device); // list all
#endif
  // and deleting the frame buffer from FrameRegistry2 as well
  bool somethingLeaks = false;
  for (auto &it: FrameRegistry2)
  {
    for (auto &it2: it.second)
  {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2.first);
      delete vfb;
      // iterate through frames belonging to this vfb
      for (auto &it3: it2.second)
      {
        VideoFrame *frame = it3.frame;

        frame->vfb = 0;

        //assert(0 == frame->refcount);
        if (0 == frame->refcount)
        {
          delete frame;
        }
        else
        {
          somethingLeaks = true;
        }
      } // it3
    } // it2
  } // it

  if (somethingLeaks) {
    LogMsg(LOGLEVEL_WARNING, "A plugin or the host application might be causing memory leaks.");
  }

  delete plugin_manager;

#ifdef AVS_WINDOWS // COM is Win32-specific
  // If we init'd COM and this is the right thread then release it
  // If it's the wrong threadId then tuff, nothing we can do.
  if (SUCCEEDED(hrfromcoinit) && (coinitThreadId == GetCurrentThreadId())) {
    hrfromcoinit = E_FAIL;
    CoUninitialize();
  }
#endif

}

void ScriptEnvironment::SetLogParams(const char* target, int level)
{
  if (nullptr == target) {
    target = "stderr";
  }

  if (-1 == level) {
    level = LOGLEVEL_INFO;
  }

  if (LogFileStream.is_open()) {
    LogFileStream.close();
  }

  LogLevel = LOGLEVEL_NONE;

  if (!streqi(target, "stderr") && !streqi(target, "stdout")) {
    LogFileStream.open(target, std::ofstream::out | std::ofstream::app);
    if (LogFileStream.fail()) {
      this->ThrowError("SetLogParams: Could not open file \"%s\" for writing.", target);
      return;
    }
  }

  LogLevel = level;
  LogTarget = target;
}

void ScriptEnvironment::LogMsg(int level, const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  LogMsg_valist(level, fmt, val);
  va_end(val);
}

void ScriptEnvironment::LogMsg_valist(int level, const char* fmt, va_list va)
{
  // Don't output message if our logging level is not high enough
  if (level > LogLevel) {
    return;
  }

  // Setup string prefixes for output messages
  const char* levelStr = nullptr;
  uint16_t levelAttr;
  switch (level)
  {
  case LOGLEVEL_ERROR:
    levelStr = "ERROR: ";
#ifdef AVS_WINDOWS // FOREGROUND_* is Windows-specific
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_RED;
#endif
    break;
  case LOGLEVEL_WARNING:
    levelStr = "WARNING: ";
#ifdef AVS_WINDOWS // FOREGROUND_* is Windows-specific
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED;
#endif
    break;
  case LOGLEVEL_INFO:
    levelStr = "INFO: ";
#ifdef AVS_WINDOWS // FOREGROUND_* is Windows-specific
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE;
#endif
    break;
  case LOGLEVEL_DEBUG:
    levelStr = "DEBUG: ";
#ifdef AVS_WINDOWS // FOREGROUND_* is Windows-specific
    levelAttr = FOREGROUND_INTENSITY | FOREGROUND_BLUE | FOREGROUND_RED;
#endif
    break;
  default:
    this->ThrowError("LogMsg: level argument must be between 1 and 4.");
    break;
  }

  // Prepare message output target
  std::ostream* targetStream = nullptr;
#ifdef AVS_WINDOWS
  void* hConsole = GetStdHandle(STD_ERROR_HANDLE);
#else
  void* hConsole = stderr;
#endif

  if (streqi("stderr", LogTarget.c_str()))
  {
#ifdef AVS_WINDOWS
    hConsole = GetStdHandle(STD_ERROR_HANDLE);
#else
    hConsole = stderr;
#endif
    targetStream = &std::cerr;
  }
  else if (streqi("stdout", LogTarget.c_str()))
  {
#ifdef AVS_WINDOWS // linux alternative?
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
#else
    hConsole = stdout;
#endif
    targetStream = &std::cout;
  }
  else if (LogFileStream.is_open())
  {
    targetStream = &LogFileStream;
  }
  else
  {
    // Logging not yet set up (SetLogParams() not yet called).
    // Do nothing.
    return;
  }

  // Format our message string
  std::string msg = FormatString(fmt, va);

#ifdef AVS_WINDOWS
  // Save current console attributes so that we can restore them later
  CONSOLE_SCREEN_BUFFER_INFO Info;
  GetConsoleScreenBufferInfo(hConsole, &Info);
#endif

  // Do the output
  std::lock_guard<std::mutex> lock(string_mutex);
  *targetStream << "---------------------------------------------------------------------" << std::endl;
#ifdef AVS_WINDOWS
  SetConsoleTextAttribute(hConsole, levelAttr);
#endif
  *targetStream << levelStr;
#ifdef AVS_WINDOWS
  SetConsoleTextAttribute(hConsole, Info.wAttributes);
#endif
  *targetStream << msg << std::endl;
  targetStream->flush();
}

void ScriptEnvironment::LogMsgOnce(const OneTimeLogTicket& ticket, int level, const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  LogMsgOnce_valist(ticket, level, fmt, val);
  va_end(val);
}

void ScriptEnvironment::LogMsgOnce_valist(const OneTimeLogTicket& ticket, int level, const char* fmt, va_list va)
{
  if (LogTickets.end() == LogTickets.find(ticket))
  {
    LogMsg_valist(level, fmt, va);
    LogTickets.insert(ticket);
  }
}

void ScriptEnvironment::SetMaxCPU(const char* features)
{
  enum CPUlevel {
    CL_NONE,
    CL_MMX,
    CL_SSE,
    CL_SSE2,
    CL_SSE3,
    CL_SSSE3,
    CL_SSE4_1,
    CL_SSE4_2,
    CL_AVX,
    CL_AVX2
  };

  std::string s;
  const int len = (int)strlen(features);
  s.resize(len);
  for (int i = 0; i < len; i++)
    s[i] = tolower(features[i]);

  int cpu_flags = GetCPUFlags();

  std::vector<std::string> tokens;
  std::size_t start = 0, end = 0;
  while ((end = s.find(',', start)) != std::string::npos) {
    if (end != start) {
      tokens.push_back(s.substr(start, end - start));
    }
    start = end + 1;
  }
  if (end != start) {
    tokens.push_back(s.substr(start));
  }

  for (auto token : tokens)
  {
    token = trim(token);
    if (token.empty()) continue;

    int mode = 0; // limit

    char ch = token[token.size() - 1];
    if (ch == '-') mode = -1; // remove
    else if (ch == '+') mode = 1; // add

    if (mode != 0)
      token.resize(token.size() - 1);

    CPUlevel cpulevel;

    const char* t = token.c_str();

    if (streqi(t, "") || streqi(t, "none")) cpulevel = CL_NONE;
    else if (streqi(t, "mmx")) cpulevel = CL_MMX;
    else if (streqi(t, "sse")) cpulevel = CL_SSE;
    else if (streqi(t, "sse2")) cpulevel = CL_SSE2;
    else if (streqi(t, "sse3")) cpulevel = CL_SSE3;
    else if (streqi(t, "ssse3")) cpulevel = CL_SSSE3;
    else if (streqi(t, "sse4") || streqi(t, "sse4.1")) cpulevel = CL_SSE4_1;
    else if (streqi(t, "sse4.2")) cpulevel = CL_SSE4_2;
    else if (streqi(t, "avx")) cpulevel = CL_AVX;
    else if (streqi(t, "avx2")) cpulevel = CL_AVX2;
    else ThrowError("SetMaxCPU error: cpu level must be empty or none, mmx, sse, sse2, sse3, ssse3, sse4 or sse4.1, sse4.2, avx or avx2! (%s)", t);

    if (0 == mode) { // limit
      if (cpulevel <= CL_AVX2)
        cpu_flags &= ~(CPUF_AVX512BW | CPUF_AVX512CD | CPUF_AVX512DQ |
          CPUF_AVX512ER | CPUF_AVX512F | CPUF_AVX512IFMA | CPUF_AVX512PF | CPUF_AVX512VBMI | CPUF_AVX512VL);
      if (cpulevel <= CL_AVX)
        cpu_flags &= ~(CPUF_AVX2 | CPUF_FMA3 | CPUF_FMA4 | CPUF_F16C);
      if (cpulevel <= CL_SSE4_2)
        cpu_flags &= ~(CPUF_AVX);
      if (cpulevel <= CL_SSE4_1)
        cpu_flags &= ~(CPUF_SSE4_2);
      if (cpulevel <= CL_SSSE3)
        cpu_flags &= ~(CPUF_SSE4_1);
      if (cpulevel <= CL_SSE3)
        cpu_flags &= ~(CPUF_SSSE3);
      if (cpulevel <= CL_SSE2)
        cpu_flags &= ~(CPUF_SSE3);
      if (cpulevel <= CL_SSE)
        cpu_flags &= ~(CPUF_SSE2);
      if (cpulevel <= CL_MMX)
        cpu_flags &= ~(CPUF_SSE | CPUF_INTEGER_SSE); // ?
      if (cpulevel <= CL_NONE)
        cpu_flags &= ~(CPUF_MMX);
    }
    else {
      int current_flag;
      switch (cpulevel) {
      case CL_AVX2: current_flag = CPUF_AVX2 | CPUF_FMA3; break;
      case CL_AVX: current_flag = CPUF_AVX; break;
      case CL_SSE4_2: current_flag = CPUF_SSE4_2; break;
      case CL_SSE4_1: current_flag = CPUF_SSE4_1; break;
      case CL_SSSE3: current_flag = CPUF_SSSE3; break;
      case CL_SSE3: current_flag = CPUF_SSE3; break;
      case CL_SSE2: current_flag = CPUF_SSE2; break;
      case CL_SSE: current_flag = CPUF_SSE; break;
      case CL_MMX: current_flag = CPUF_MMX; break;
      default:
        current_flag = 0;
      }
      if (mode < 0) {
        if (0 != current_flag)
          cpu_flags &= ~current_flag; // sse2-: removes sse2
      }
      else
        cpu_flags |= current_flag; // avx2+: adds avx2 and fma3
      // limit to sse2 and avx2: "sse2,avx2+"
    }
  }

  ::SetMaxCPU(cpu_flags);
}

ClipDataStore* ScriptEnvironment::ClipData(IClip *clip)
{
#if ( !defined(_MSC_VER) || (_MSC_VER < 1900) )
  return &(clip_data.emplace(clip, clip).first->second);
#else
  return &(clip_data.try_emplace(clip, clip).first->second);
#endif
}

void ScriptEnvironment::AdjustMemoryConsumption(size_t amount, bool minus)
{
  if (minus)
    Devices->GetCPUDevice()->memory_used -= amount;
  else
    Devices->GetCPUDevice()->memory_used += amount;
}

void ScriptEnvironment::ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion)
{
  thread_pool->QueueJob(jobFunc, jobData, threadEnv.get(), static_cast<JobCompletion*>(completion));
}

void ScriptEnvironment::ParallelJob(ThreadWorkerFuncPtr jobFunc, void* jobData, IJobCompletion* completion, InternalEnvironment* env)
{
  thread_pool->QueueJob(jobFunc, jobData, env, static_cast<JobCompletion*>(completion));
}

void ScriptEnvironment::SetFilterMTMode(const char* filter, MtMode mode, bool force)
{
  this->SetFilterMTMode(filter, mode, force ? MtWeight::MT_WEIGHT_2_USERFORCE : MtWeight::MT_WEIGHT_1_USERSPEC);
}

void ScriptEnvironment::SetFilterMTMode(const char* filter, MtMode mode, MtWeight weight)
{
  assert(NULL != filter);
  assert(strcmp("", filter) != 0);

  if (((int)mode <= (int)MT_INVALID)
    || ((int)mode >= (int)MT_MODE_COUNT))
  {
    throw AvisynthError("Invalid MT mode specified.");
  }

  if (streqi(filter, DEFAULT_MODE_SPECIFIER.c_str()))
  {
    DefaultMtMode = mode;
    return;
  }

  std::string name_to_register;
  std::string loading;
  {
    std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
    loading = plugin_manager->PluginLoading();
  }
  if (loading.empty())
    name_to_register = filter;
  else
    name_to_register = loading.append("_").append(filter);

  name_to_register = NormalizeString(name_to_register);

  auto it = MtMap.find(name_to_register);
  if (it != MtMap.end())
  {
    if ((int)weight >= (int)(it->second.second))
    {
      it->second.first = mode;
      it->second.second = weight;
    }
  }
  else
  {
    MtMap.emplace(name_to_register, std::make_pair(mode, weight));
  }
}

bool ScriptEnvironment::FilterHasMtMode(const Function* filter) const
{
  if (filter->name == nullptr) { // no named function
    return false;
  }
  const auto& end = MtMap.end();
  return (end != MtMap.find(NormalizeString(filter->canon_name)))
    || (end != MtMap.find(NormalizeString(filter->name)));
}

MtMode ScriptEnvironment::GetFilterMTMode(const Function* filter, bool* is_forced) const
{
  assert(NULL != filter);
  if (filter->name == nullptr) { // no named function
    *is_forced = false;
    return DefaultMtMode;
  }

  assert(NULL != filter->name);
  assert(NULL != filter->canon_name);

  auto it = MtMap.find(NormalizeString(filter->canon_name));
  if (it != MtMap.end())
  {
    *is_forced = it->second.second == MtWeight::MT_WEIGHT_2_USERFORCE;
    return it->second.first;
  }

  it = MtMap.find(NormalizeString(filter->name));
  if (it != MtMap.end())
  {
    *is_forced = it->second.second == MtWeight::MT_WEIGHT_2_USERFORCE;
    return it->second.first;
  }

  *is_forced = false;
  return DefaultMtMode;
}

/* This function adds information about builtin functions into global variables.
 * External utilities (like AvsPmod) can parse these variables and use them
 * to learn about supported functions and their syntax.
 */
void ScriptEnvironment::ExportBuiltinFilters()
{
  std::string FunctionList;
  FunctionList.reserve(512);
  const size_t NumFunctionArrays = sizeof(builtin_functions) / sizeof(builtin_functions[0]);
  for (size_t i = 0; i < NumFunctionArrays; ++i)
  {
    for (const AVSFunction* f = builtin_functions[i]; !f->empty(); ++f)
    {
      // This builds the $InternalFunctions$ variable, which is a list of space-delimited
      // function names. Utilities can learn the names of the builtin function from this.
      FunctionList.append(f->name);
      FunctionList.push_back(' ');

      // For each supported function, a global variable is added with <param_var_name> as the name,
      // and the list of parameters to that function as the value.
      std::string param_var_name;
      param_var_name.reserve(128);
      param_var_name.append("$Plugin!");
      param_var_name.append(f->name);
      param_var_name.append("!Param$");
      threadEnv->SetGlobalVar(threadEnv->SaveString(param_var_name.c_str(), (int)param_var_name.size()), AVSValue(f->param_types));
    }
  }

  // Save $InternalFunctions$
  threadEnv->SetGlobalVar("$InternalFunctions$", AVSValue(threadEnv->SaveString(FunctionList.c_str(), (int)FunctionList.size())));
}

size_t  ScriptEnvironment::GetEnvProperty(AvsEnvProperty prop)
{
  switch (prop)
  {
  case AEP_NUM_DEVICES:
    return Devices->GetNumDevices();
  case AEP_FRAME_ALIGN:
    return frame_align;
  case AEP_PLANE_ALIGN:
    return plane_align;
  case AEP_FILTERCHAIN_THREADS:
    return nMaxFilterInstances;
  case AEP_PHYSICAL_CPUS:
    return GetNumPhysicalCPUs();
  case AEP_LOGICAL_CPUS:
    return std::thread::hardware_concurrency();
  case AEP_THREAD_ID:
    return 0;
  case AEP_THREADPOOL_THREADS:
    return thread_pool->NumThreads();
  case AEP_VERSION:
#ifdef RELEASE_TARBALL
    return 0;
#else
    return AVS_SEQREV;
#endif
  case AEP_HOST_SYSTEM_ENDIANNESS:
    return (uint64_t)AVS_ENDIANNESS;
  default:
    this->ThrowError("Invalid property request.");
    return std::numeric_limits<size_t>::max();
  }

  assert(0);
}

bool ScriptEnvironment::LoadPlugin(const char* filePath, bool throwOnError, AVSValue *result)
{
  // Autoload needed to ensure that manual LoadPlugin() calls always override autoloaded plugins.
  // For that, autoloading must happen before any LoadPlugin(), so we force an
  // autoload operation before any LoadPlugin().
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  this->AutoloadPlugins();
  return plugin_manager->LoadPlugin(filePath, throwOnError, result);
}

void ScriptEnvironment::AddAutoloadDir(const char* dirPath, bool toFront)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AddAutoloadDir(dirPath, toFront);
}

void ScriptEnvironment::ClearAutoloadDirs()
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->ClearAutoloadDirs();
}

void ScriptEnvironment::AutoloadPlugins()
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AutoloadPlugins();
}

int ScriptEnvironment::SetMemoryMax(int mem) {

  Device* cpuDevice = Devices->GetCPUDevice();
  if (mem > 0)  /* If mem is zero, we should just return current setting */
    cpuDevice->memory_max = ConstrainMemoryRequest(mem * 1048576ull);

  return (int)(cpuDevice->memory_max / 1048576ull);
}

int ScriptEnvironment::SetWorkingDir(const char* newdir) {
  return SetCurrentDirectory(newdir) ? 0 : 1;
}

void ScriptEnvironment::CheckVersion(int version) {
  if (version > AVISYNTH_INTERFACE_VERSION)
    ThrowError("Plugin was designed for a later version of Avisynth (%d)", version);
}

int ScriptEnvironment::GetCPUFlags() { return ::GetCPUFlags(); }

void ScriptEnvironment::AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data) {
  this->AddFunction(name, params, apply, user_data, NULL);
}

// called from IScriptEnvironment_Avs25
void ScriptEnvironment::AddFunction25(const char* name, const char* params, ApplyFunc apply, void* user_data) {
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AddFunction(name, params, apply, user_data, NULL, true);
}

void ScriptEnvironment::AddFunction(const char* name, const char* params, ApplyFunc apply, void* user_data, const char *exportVar) {
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->AddFunction(name, params, apply, user_data, exportVar, false);
}

VideoFrame* ScriptEnvironment::AllocateFrame(size_t vfb_size, size_t margin, Device* device)
{
  if (vfb_size > (size_t)std::numeric_limits<int>::max())
  {
    throw AvisynthError(threadEnv->Sprintf("Requested buffer size of %zu is too large", vfb_size));
  }

  VFBStorage* vfb = NULL;
  try
  {
    vfb = new VFBStorage((int)vfb_size, (int)margin, device);
  }
  catch(const std::bad_alloc&)
  {
    return NULL;
  }

  VideoFrame *newFrame = NULL;
  try
  {
    newFrame = new VideoFrame(vfb, new AVSMap(), 0, 0, 0, 0);
  }
  catch(const std::bad_alloc&)
  {
    delete vfb;
    return NULL;
  }

  device->memory_used += vfb_size;
  vfb->Attach(threadEnv->GetCurrentGraphNode());

  // automatically inserts keys if they not exist!
  // no locking here, calling method have done it already
  FrameRegistry2[vfb_size][vfb].push_back(DebugTimestampedFrame(newFrame,
    newFrame->properties
  ));

  //_RPT1(0, "ScriptEnvironment::AllocateFrame %zu frame=%p vfb=%p %" PRIu64 "\n", vfb_size, newFrame, newFrame->vfb, memory_used);

  return newFrame;
}

#ifdef _DEBUG

static void DebugOut(char* s)
{
#ifdef AVS_POSIX
  LogMsg(LOGLEVEL_DEBUG, s);
#else
  _RPT0(0, s);
#endif
}

void ScriptEnvironment::ListFrameRegistry(size_t min_size, size_t max_size, bool someframes)
{
  char buf[1024];
  //#define FULL_LIST_OF_VFBs
  //#define LIST_ALSO_SOME_FRAMES
  int size1 = 0;
  int size2 = 0;
  int size3 = 0;
  snprintf(buf, 1023, "******** %p <= FrameRegistry2 Address. Buffer list for size between %7zu and %7zu\n", &FrameRegistry2, min_size, max_size);
  DebugOut(buf);
  snprintf(buf, 1023, ">> IterateLevel #1: Different vfb sizes: FrameRegistry2.size=%zu \n", FrameRegistry2.size());
  DebugOut(buf);
  size_t total_vfb_size = 0;
  auto t_end = std::chrono::high_resolution_clock::now();

  // list to debugview: all frames up-to vfb_size size
  for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(min_size), end_it = FrameRegistry2.upper_bound(max_size);
    it != end_it;
    ++it)
  {
    size1++;
    _RPT3(0, ">>>> IterateLevel #2 [%3d]: Vfb count for size %7zu is %7zu\n", size1, it->first, it->second.size());
    for (auto &it2: it->second)
    {
      size2++;
      VideoFrameBuffer* vfb = it2.first;
      total_vfb_size += vfb->GetDataSize();
      size_t inner_frame_count_size = it2.second.size();
      snprintf(buf, 1023, ">>>> IterateLevel #3 %5zu frames in [%3d,%5d] --> vfb=%p vfb_refcount=%3ld seqNum=%d\n", inner_frame_count_size, size1, size2, vfb, vfb->refcount, vfb->GetSequenceNumber());
      DebugOut(buf);
      // iterate the frame list of this vfb
      int inner_frame_count = 0;
      int inner_frame_count_for_frame_refcount_nonzero = 0;
      for (auto &it3: it2.second)
      {
        size3++;
        inner_frame_count++;
#ifdef _DEBUG
        VideoFrame* frame = it3.frame;
        std::chrono::time_point<std::chrono::high_resolution_clock> frame_entry_timestamp = it3.timestamp;
#else
        VideoFrame* frame = it3;
#endif
        if (0 != frame->refcount)
          inner_frame_count_for_frame_refcount_nonzero++;
        if (someframes)
        {
          std::chrono::duration<double> elapsed_seconds = t_end - frame_entry_timestamp;
          if (inner_frame_count <= 2) // list only the first 2. There can be even many thousand of frames!
          {
            // log only if frame creation timestamp is too old!
            // e.g. 100 secs, it must be a stuck frame (but can also be a valid static frame from ColorBars)
            // if (elapsed_seconds.count() > 100.0f && frame->refcount > 0)
            if (frame->refcount > 0)
            {
              snprintf(buf, 1023, "  >> Frame#%6d: vfb=%p frame=%p frame_refcount=%3ld timestamp=%f ago\n", inner_frame_count, vfb, frame, frame->refcount, elapsed_seconds.count());
              DebugOut(buf);
            }
          }
          else if (inner_frame_count == inner_frame_count_size - 1)
          {
            // log the last one
            if (frame->refcount > 0)
            {
              snprintf(buf, 1023, "  ...Frame#%6d: vfb=%p frame=%p frame_refcount=%3ld \n", inner_frame_count, vfb, frame, frame->refcount);
              DebugOut(buf);
            }
            _RPT2(0, "  == TOTAL of %d frames. Number of nonzero refcount=%d \n", inner_frame_count, inner_frame_count_for_frame_refcount_nonzero);
          }
          if (0 == vfb->refcount && 0 != frame->refcount)
          {
            snprintf(buf, 1023, "  ########## VFB=0 FRAME!=0 ####### VFB: %p Frame:%p frame_refcount=%3ld \n", vfb, frame, frame->refcount);
            DebugOut(buf);
          }
        }
      }
    }
  }
  snprintf(buf, 1023, ">> >> >> array sizes %d %d %d Total VFB size=%zu\n", size1, size2, size3, total_vfb_size);
  DebugOut(buf);
  snprintf(buf, 1023, "  ----------------------------\n");
  DebugOut(buf);
}
#endif

VideoFrame* ScriptEnvironment::GetFrameFromRegistry(size_t vfb_size, Device* device)
{
#ifdef _DEBUG
  std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end; // std::chrono::time_point<std::chrono::system_clock> t_start, t_end;
  t_start = std::chrono::high_resolution_clock::now();
#endif

  // FrameRegistry2 is like: map<key1, map<key2, vector<VideoFrame *>> >
  // typedef std::vector<VideoFrame*> VideoFrameArrayType;
  // typedef std::map<VideoFrameBuffer *, VideoFrameArrayType> FrameBufferRegistryType;
  // typedef std::map<size_t, FrameBufferRegistryType> FrameRegistryType2;
  // [vfb_size = 10000][vfb = 0x111111111] [frame = 0x129837192(,timestamp=xxx)]
  //                                       [frame = 0x012312122(,timestamp=xxx)]
  //                                       [frame = 0x232323232(,timestamp=xxx)]
  //                   [vfb = 0x222222222] [frame = 0x333333333(,timestamp=xxx)]
  //                                       [frame = 0x444444444(,timestamp=xxx)]
  // Which is better?
  // - found exact vfb_size or
  // - allow reusing existing vfb's with size up to size_to_find*1.5 THIS ONE!
  // - allow to occupy any buffer that is bigger than the requested size
  //for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.upper_bound(vfb_size); // exact! no-go. special service clips can fragment it
  //for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.end(); // vfb_size or bigger, so a 100K size would claim a 1.5M space.
  for (FrameRegistryType2::iterator it = FrameRegistry2.lower_bound(vfb_size), end_it = FrameRegistry2.upper_bound(vfb_size * 3 / 2); // vfb_size or at most 1.5* bigger
    it != end_it;
    ++it)
  {
    for (auto &it2: it->second)
    {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2.first); // same for all map content, the key is vfb pointer
      if (device == vfb->device && 0 == vfb->refcount) // vfb device and refcount check
      {
        size_t videoFrameListSize = it2.second.size();
        VideoFrame *frame_found;
        AVSMap* properties_found;
        bool found = false;
        for (VideoFrameArrayType::iterator it3 = it2.second.begin(), end_it3 = it2.second.end();
          it3 != end_it3;
          /*++it3 not here, because of the delete*/)
        {
          VideoFrame *frame = it3->frame;

          // sanity check if its refcount is zero
          // because when a vfb is free (refcount==0) then all its parent frames should also be free
          assert(0 == frame->refcount);
          assert(nullptr != frame->properties);

          if (!found)
          {
            InterlockedIncrement(&(frame->vfb->refcount)); // same as &(vfb->refcount)
            vfb->free_count = 0; // reset free count
            vfb->Attach(threadEnv->GetCurrentGraphNode());
#ifdef _DEBUG
            char buf[256];
            t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = t_end - t_start;
            snprintf(buf, 255, "ScriptEnvironment::GetNewFrame NEW METHOD EXACT hit! VideoFrameListSize=%7zu GotSize=%7zu FrReg.Size=%6zu vfb=%p frame=%p SeekTime:%f\n", videoFrameListSize, vfb_size, FrameRegistry2.size(), vfb, frame, elapsed_seconds.count());
            _RPT0(0, buf);
            _RPT5(0, "                                          frame %p RowSize=%d Height=%d Pitch=%d Offset=%d\n", frame, frame->GetRowSize(), frame->GetHeight(), frame->GetPitch(), frame->GetOffset());
#endif
            // only 1 frame in list -> no delete
            if (videoFrameListSize <= 1)
            {
              _RPT1(0, "ScriptEnvironment::GetNewFrame returning frame. VideoFrameListSize was 1\n", videoFrameListSize);
#ifdef _DEBUG
              it3->timestamp = std::chrono::high_resolution_clock::now(); // refresh timestamp!
#endif
              return frame; // return immediately
            }
            // more than X: just registered the frame found, and erase all other frames from list plus delete frame objects also
            frame_found = frame;
            properties_found = it3->properties;
            found = true;
            ++it3;
          }
          else {
            // if the first frame to this vfb was already found, then we free all others and delete it from the list
            // Benefit: no 4-5k frame list count per a single vfb.
            //_RPT4(0, "ScriptEnvironment::GetNewFrame Delete one frame %p RowSize=%d Height=%d Pitch=%d Offset=%d\n", frame, frame->GetRowSize(), frame->GetHeight(), frame->GetPitch(), frame->GetOffset());
            delete frame;
            delete it3->properties;
            ++it3;
          }
        } // for it3
        if (found)
        {
          _RPT1(0, "ScriptEnvironment::GetNewFrame returning frame_found. clearing frames. List count: it2->second.size(): %7zu \n", it2.second.size());
          it2.second.clear();
          it2.second.reserve(16); // initial capacity set to 16, avoid reallocation when 1st, 2nd, etc.. elements pushed later (possible speedup)
          it2.second.push_back(DebugTimestampedFrame(frame_found,
            properties_found
          )); // keep only the first
          return frame_found;
        }
      }
    } // for it2
  } // for it
  _RPT3(0, "ScriptEnvironment::GetNewFrame, no free entry in FrameRegistry. Requested vfb size=%zu memused=%" PRIu64 " memmax=%" PRIu64 "\n", vfb_size, device->memory_used.load(), device->memory_max);

#ifdef _DEBUG
  //ListFrameRegistry(vfb_size, vfb_size, true); // for chasing stuck frames
  //ListFrameRegistry(0, vfb_size, true); // for chasing stuck frames
#endif

  return NULL;
}

VideoFrame* ScriptEnvironment::GetNewFrame(size_t vfb_size, size_t margin, Device* device)
{
  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex);

  // prevent fragmentation of vfb buffer list many different small-sized vfb's
  if (vfb_size < 64) vfb_size = 64;
  else if (vfb_size < 256) vfb_size = 256;
  else if (vfb_size < 512) vfb_size = 512;
  else if (vfb_size < 1024) vfb_size = 1024;
  else if (vfb_size < 2048) vfb_size = 2048;
  else if (vfb_size < 4096) vfb_size = 4096;

  /* -----------------------------------------------------------
  *   Try to return an unused but already allocated instance
  * -----------------------------------------------------------
  */
  VideoFrame* frame = GetFrameFromRegistry(vfb_size, device);
  if (frame != NULL)
    return frame;

  /* -----------------------------------------------------------
   *   No unused instance was found, try to allocate a new one
   * -----------------------------------------------------------
   */
   // We reserve 15% for unaccounted stuff
  if (device->memory_used + vfb_size < device->memory_max * 0.85f) {
    frame = AllocateFrame(vfb_size, margin, device);
  }
  if (frame != NULL)
    return frame;

#ifdef _DEBUG
  // #define LIST_CACHES
  // list all cache_entries
#ifdef LIST_CACHES
  int cache_counter = 0;
  for (auto &cit: CacheRegistry)
  {
    cache_counter++;
    Cache* cache = cit;
    int cache_size = cache->SetCacheHints(CACHE_GET_SIZE, 0);
    _RPT4(0, "  cache#%d cache_ptr=%p cache_size=%d \n", cache_counter, (void*)cache, cache_size); // let's see what's in the cache
  }
#endif
#endif

  /* -----------------------------------------------------------
  * Couldn't allocate, shrink cache and get more unused frames
  * -----------------------------------------------------------
  */
  ShrinkCache(device);

  /* -----------------------------------------------------------
  *   Try to return an unused frame again
  * -----------------------------------------------------------
  */
  frame = GetFrameFromRegistry(vfb_size, device);
  if (frame != NULL)
    return frame;

  /* -----------------------------------------------------------
  *   Try to allocate again
  * -----------------------------------------------------------
  */
  frame = AllocateFrame(vfb_size, margin, device);
  if (frame != NULL)
    return frame;

  OneTimeLogTicket ticket(LOGTICKET_W1100);
  LogMsgOnce(ticket, LOGLEVEL_WARNING, "Memory reallocation occurs. This will probably degrade performance. You can try increasing the limit using SetMemoryMax().");

  /* -----------------------------------------------------------
   * No frame found, free all the unused frames!!!
   * -----------------------------------------------------------
   */
  _RPT1(0, "Allocate failed. GC start memory_used=%" PRIu64 "\n", device->memory_used.load());
  // unfortunately if we reach here, only 0 or 1 vfbs or frames can be freed, from lower vfb sizes
  // usually it's not enough
  // yet it is true that it's meaningful only to free up smaller vfb sizes here
  for (FrameRegistryType2::iterator it = FrameRegistry2.begin(), end_it = FrameRegistry2.upper_bound(vfb_size);
    it != end_it;
    ++it)
  {
    for (FrameBufferRegistryType::iterator it2 = (it->second).begin(), end_it2 = (it->second).end();
      it2 != end_it2;
      /*++it2: not here: may delete iterator position */)
    {
      VFBStorage *vfb = static_cast<VFBStorage*>(it2->first);
      if (device == vfb->device && 0 == vfb->refcount) // vfb refcount check
      {
        vfb->device->memory_used -= vfb->GetDataSize(); // frame->vfb->GetDataSize();
        delete vfb;
        for (auto &it3: it2->second)
        {
          VideoFrame *currentframe = it3.frame;
          assert(0 == currentframe->refcount);
          delete currentframe;
          delete it3.properties;

        }
        // delete array belonging to this vfb in one step
        it2->second.clear(); // clear frame list
        it2 = (it->second).erase(it2); // clear current vfb
      }
      else ++it2;
    }
  }
  _RPT1(0, "End of garbage collection A memused=%" PRIu64 "\n", device->memory_used.load());
#if 0
  static int counter = 0;
  char buf[200]; sprintf(buf, "Re allocation %d\r\n", counter++);
  OutputDebugStringA(buf);
#endif
  /* -----------------------------------------------------------
   *   Try to allocate again
   * -----------------------------------------------------------
   */
  frame = AllocateFrame(vfb_size, margin, device);
  if ( frame != NULL)
    return frame;


  /* -----------------------------------------------------------
   *   Oh boy...
   * -----------------------------------------------------------
   */

   // See if we could benefit from 64-bit Avisynth
  if (sizeof(void*) == 4 && device == Devices->GetCPUDevice())
  {
#ifdef AVS_WINDOWS
    // Get system memory information
    MEMORYSTATUSEX memstatus;
    memstatus.dwLength = sizeof(memstatus);
    GlobalMemoryStatusEx(&memstatus);

    BOOL using_wow64;
    if (IsWow64Process(GetCurrentProcess(), &using_wow64)     // if running 32-bits on a 64-bit OS
      && (memstatus.ullAvailPhys > 1024ull * 1024 * 1024))    // if at least 1GB of system memory is still free
    {
      OneTimeLogTicket ticket(LOGTICKET_W1007);
      LogMsgOnce(ticket, LOGLEVEL_INFO, "We have run out of memory, but your system still has some free RAM left. You might benefit from a 64-bit build of Avisynth+.");
    }
#endif
  }

  ThrowError("Could not allocate video frame. Out of memory. memory_max = %" PRIu64 ", memory_used = %" PRIu64 " Request=%zu", device->memory_max, device->memory_used.load(), vfb_size);
  return NULL;
}

void ScriptEnvironment::ShrinkCache(Device *device)
{
  /* -----------------------------------------------------------
  *   Shrink cache to keep memory limit
  * -----------------------------------------------------------
  */
  int shrinkcount = 0;

  for (auto &cit: CacheRegistry)
  {
    // Oh darn. We'd need more memory than we are allowed to use.
    // Let's reduce the amount of caching.

    // We try to shrink least recently used caches first.

    Cache* cache = cit;
    if (cache->GetDevice() != device) {
      continue;
    }
    int cache_size = cache->SetCacheHints(CACHE_GET_SIZE, 0);
    if (cache_size != 0)
    {
      _RPT2(0, "ScriptEnvironment::EnsureMemoryLimit shrink cache. cache=%p new size=%d\n", (void*)cache, cache_size - 1);
      cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, cache_size - 1);
      shrinkcount++;
    } // if
  } // for cit

  if (shrinkcount != 0)
  {
    OneTimeLogTicket ticket(LOGTICKET_W1003);
    LogMsgOnce(ticket, LOGLEVEL_WARNING, "Caches have been shrunk due to low memory limit. This will probably degrade performance. You can try increasing the limit using SetMemoryMax().");
  }

  /* -----------------------------------------------------------
  * Count up free_count and free if it exceeds the threshold
  * -----------------------------------------------------------
  */
  // Free up in one pass in FrameRegistry2
  if (shrinkcount)
  {
    _RPT1(0, "EnsureMemoryLimit GC start: memused=%" PRIu64 "\n", device->memory_used.load());
    int freed_vfb_count = 0;
    int freed_frame_count = 0;
    int unfreed_frame_count = 0;
    for (auto &it: FrameRegistry2)
    {
      for (FrameBufferRegistryType::iterator it2 = (it.second).begin(), end_it2 = (it.second).end();
        it2 != end_it2;
        /*++it2: not here: may delete iterator position */)
      {
        VFBStorage *vfb = static_cast<VFBStorage*>(it2->first);
        // vfb device and refcount check and free count exceeds the threshold
        if (device == vfb->device && 0 == vfb->refcount && vfb->free_count++ >= device->free_thresh)
        {
#if 0
          static int counter = 0;
          char buf[200]; sprintf(buf, "Free frame !!! %d\r\n", counter++);
          OutputDebugStringA(buf);
#endif
          device->memory_used -= vfb->GetDataSize();
#ifdef _DEBUG
          VFBStorage *_vfb = vfb;
#endif
          delete vfb;
          ++freed_vfb_count;
          for (auto &it3: it2->second)
          {
            VideoFrame *frame = it3.frame;
            assert(0 == frame->refcount);
            if (0 == frame->refcount)
            {
              delete frame;
              delete it3.properties;
              ++freed_frame_count;
            }
            else {
              // there should not be such case: vfb.refcount=0 and frame.refcount!=0
              ++unfreed_frame_count;
#ifdef _DEBUG
              _RPT3(0, "  ?????? frame refcount error!!! _vfb=%p frame=%p framerefcount=%d \n", _vfb, frame, frame->refcount);
#endif
            }
          }
          // delete array belonging to this vfb in one step
          it2->second.clear(); // clear frame list
          it2 = (it.second).erase(it2); // clear vfb entry
        }
        else ++it2;
      }
    }
    _RPT4(0, "End of garbage collection B: freed_vfb=%d frame=%d unfreed=%d memused=%" PRIu64 "\n", freed_vfb_count, freed_frame_count, unfreed_frame_count, device->memory_used.load());
  }
}

// no alpha
PVideoFrame ScriptEnvironment::NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, Device* device)
{
  return NewPlanarVideoFrame(row_size, height, row_sizeUV, heightUV, align, U_first, false, device); // no alpha
}

// with alpha support
PVideoFrame ScriptEnvironment::NewPlanarVideoFrame(int row_size, int height, int row_sizeUV, int heightUV, int align, bool U_first, bool alpha, Device* device)
{
  if (align < 0)
  {
    align = -align;
    OneTimeLogTicket ticket(LOGTICKET_W1009);
    LogMsgOnce(ticket, LOGLEVEL_WARNING, "A filter is using forced frame alignment, a feature that is deprecated and disabled. The filter will likely behave erroneously.");
  }
  align = max(align, frame_align);

  int pitchUV;
  const int pitchY = AlignNumber(row_size, align);
  if (!PlanarChromaAlignmentState && (row_size == row_sizeUV*2) && (height == heightUV*2)) { // Meet old 2.5 series API expectations for YV12
    // Legacy alignment - pack Y as specified, pack UV half that
    pitchUV = (pitchY+1)>>1;  // UV plane, width = 1/2 byte per pixel - don't align UV planes seperately.
  }
  else {
    // Align planes separately
    pitchUV = AlignNumber(row_sizeUV, align);
  }

  size_t sizeY = AlignNumber(pitchY * height, plane_align);
  size_t sizeUV = AlignNumber(pitchUV * heightUV, plane_align);
  size_t size = sizeY + 2 * sizeUV + (alpha ? sizeY : 0);

  VideoFrame *res = GetNewFrame(size, align - 1, device);

  int  offsetU, offsetV, offsetA;
  const int offsetY = (int)(AlignPointer(res->vfb->GetWritePtr(), align) - res->vfb->GetWritePtr()); // first line offset for proper alignment
  if (U_first) {
    offsetU = offsetY + (int)sizeY;
    offsetV = offsetY + (int)sizeY + (int)sizeUV;
    offsetA = alpha ? offsetV + (int)sizeUV : 0;
  } else {
    offsetV = offsetY + (int)sizeY;
    offsetU = offsetY + (int)sizeY + (int)sizeUV;
    offsetA = alpha ? offsetU + (int)sizeUV : 0;
  }

  res->offset = offsetY;
  res->pitch = pitchY;
  res->row_size = row_size;
  res->height = height;
  res->offsetU = offsetU;
  res->offsetV = offsetV;
  res->pitchUV = pitchUV;
  res->row_sizeUV = row_sizeUV;
  res->heightUV = heightUV;
  // alpha support
  res->offsetA = offsetA;
  res->row_sizeA = alpha ? row_size : 0;
  res->pitchA = alpha ? pitchY : 0;

  return PVideoFrame(res);
}

// Variant #1.no frame property source
PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(int row_size, int height, int align, Device* device)
{
  if (align < 0)
  {
    align = -align;
    OneTimeLogTicket ticket(LOGTICKET_W1009);
    this->LogMsgOnce(ticket, LOGLEVEL_WARNING, "A filter is using forced frame alignment, a feature that is deprecated and disabled. The filter will likely behave erroneously.");
  }
  align = max(align, frame_align);

  const int pitch = AlignNumber(row_size, align);
  size_t size = pitch * height;

  VideoFrame *res = GetNewFrame(size, align - 1, device);

  const int offset = (int)(AlignPointer(res->vfb->GetWritePtr(), align) - res->vfb->GetWritePtr()); // first line offset for proper alignment

  res->offset = offset;
  res->pitch = pitch;
  res->row_size = row_size;
  res->height = height;
  res->offsetU = offset;
  res->offsetV = offset;
  res->pitchUV = 0;
  res->row_sizeUV = 0;
  res->heightUV = 0;
  // alpha support
  res->offsetA = 0;
  res->row_sizeA = 0;
  res->pitchA = 0;

  return PVideoFrame(res);
}

// Variant #1. with frame property source
PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(int row_size, int height, int align, Device* device, PVideoFrame* propSrc)
{
  PVideoFrame result = NewVideoFrameOnDevice(row_size, height, align, device);

  if (propSrc)
    copyFrameProps(*propSrc, result);

  return result;
}

// Variant #2. without frame property source
PVideoFrame ScriptEnvironment::NewVideoFrame(const VideoInfo& vi, const PDevice& device) {
  return NewVideoFrameOnDevice(vi, frame_align, (Device*)(void*)device);
}

// Variant #2. with frame property source
PVideoFrame ScriptEnvironment::NewVideoFrame(const VideoInfo& vi, const PDevice& device, PVideoFrame* propSrc)
{
  PVideoFrame result = NewVideoFrame(vi, device);

  if (propSrc)
    copyFrameProps(*propSrc, result);

  return result;
}

// Variant #3. without frame property source
PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(const VideoInfo & vi, int align, Device * device) {
  // todo: high bit-depth: we have too many types now. Do we need really check?
  // Check requested pixel_type:
  switch (vi.pixel_type) {
    case VideoInfo::CS_BGR24:
    case VideoInfo::CS_BGR32:
    case VideoInfo::CS_YUY2:
    case VideoInfo::CS_Y8:
    case VideoInfo::CS_YV12:
    case VideoInfo::CS_YV16:
    case VideoInfo::CS_YV24:
    case VideoInfo::CS_YV411:
    case VideoInfo::CS_I420:
    // AVS16 do not reject when a filter requests it
        // planar YUV 10-32 bit
    case VideoInfo::CS_YUV420P10:
    case VideoInfo::CS_YUV422P10:
    case VideoInfo::CS_YUV444P10:
    case VideoInfo::CS_Y10:
    case VideoInfo::CS_YUV420P12:
    case VideoInfo::CS_YUV422P12:
    case VideoInfo::CS_YUV444P12:
    case VideoInfo::CS_Y12:
    case VideoInfo::CS_YUV420P14:
    case VideoInfo::CS_YUV422P14:
    case VideoInfo::CS_YUV444P14:
    case VideoInfo::CS_Y14:
    case VideoInfo::CS_YUV420P16:
    case VideoInfo::CS_YUV422P16:
    case VideoInfo::CS_YUV444P16:
    case VideoInfo::CS_Y16:
    case VideoInfo::CS_YUV420PS:
    case VideoInfo::CS_YUV422PS:
    case VideoInfo::CS_YUV444PS:
    case VideoInfo::CS_Y32:
        // 16 bit/sample packed RGB
    case VideoInfo::CS_BGR48:
    case VideoInfo::CS_BGR64:
        // planar RGB
    case VideoInfo::CS_RGBP:
    case VideoInfo::CS_RGBP10:
    case VideoInfo::CS_RGBP12:
    case VideoInfo::CS_RGBP14:
    case VideoInfo::CS_RGBP16:
    case VideoInfo::CS_RGBPS:
        // planar RGBA
    case VideoInfo::CS_RGBAP:
    case VideoInfo::CS_RGBAP10:
    case VideoInfo::CS_RGBAP12:
    case VideoInfo::CS_RGBAP14:
    case VideoInfo::CS_RGBAP16:
    case VideoInfo::CS_RGBAPS:
        // planar YUVA 8-32 bit
    case VideoInfo::CS_YUVA420:
    case VideoInfo::CS_YUVA422:
    case VideoInfo::CS_YUVA444:
    case VideoInfo::CS_YUVA420P10:
    case VideoInfo::CS_YUVA422P10:
    case VideoInfo::CS_YUVA444P10:
    case VideoInfo::CS_YUVA420P12:
    case VideoInfo::CS_YUVA422P12:
    case VideoInfo::CS_YUVA444P12:
    case VideoInfo::CS_YUVA420P14:
    case VideoInfo::CS_YUVA422P14:
    case VideoInfo::CS_YUVA444P14:
    case VideoInfo::CS_YUVA420P16:
    case VideoInfo::CS_YUVA422P16:
    case VideoInfo::CS_YUVA444P16:
    case VideoInfo::CS_YUVA420PS:
    case VideoInfo::CS_YUVA422PS:
    case VideoInfo::CS_YUVA444PS:
        break;
    default:
      ThrowError("Filter Error: Filter attempted to create VideoFrame with invalid pixel_type.");
  }

  PVideoFrame retval;

  if (vi.IsPlanar() && (vi.NumComponents() > 1)) {
    if (vi.IsYUV() || vi.IsYUVA()) {
        // Planar requires different math ;)
        const int xmod  = 1 << vi.GetPlaneWidthSubsampling(PLANAR_U);
        const int xmask = xmod - 1;
        if (vi.width & xmask)
          ThrowError("Filter Error: Attempted to request a planar frame that wasn't mod%d in width!", xmod);

        const int ymod  = 1 << vi.GetPlaneHeightSubsampling(PLANAR_U);
        const int ymask = ymod - 1;
        if (vi.height & ymask)
          ThrowError("Filter Error: Attempted to request a planar frame that wasn't mod%d in height!", ymod);

        const int heightUV = vi.height >> vi.GetPlaneHeightSubsampling(PLANAR_U);

      retval = NewPlanarVideoFrame(vi.RowSize(PLANAR_Y), vi.height, vi.RowSize(PLANAR_U), heightUV, align, !vi.IsVPlaneFirst(), vi.IsYUVA(), device);
    } else {
      // plane order: G,B,R
      retval = NewPlanarVideoFrame(vi.RowSize(PLANAR_G), vi.height, vi.RowSize(PLANAR_G), vi.height, align, !vi.IsVPlaneFirst(), vi.IsPlanarRGBA(), device);
    }
  }
  else {
    if ((vi.width&1)&&(vi.IsYUY2()))
      ThrowError("Filter Error: Attempted to request an YUY2 frame that wasn't mod2 in width.");

    retval= NewVideoFrameOnDevice(vi.RowSize(), vi.height, align, device);
  }

  return retval;
}


// Variant #3. with frame property source
PVideoFrame ScriptEnvironment::NewVideoFrameOnDevice(const VideoInfo& vi, int align, Device* device, PVideoFrame* propSrc)
{
  PVideoFrame result = NewVideoFrameOnDevice(vi, align, device);

  if (propSrc)
    copyFrameProps(*propSrc, result);

  return result;
}


bool ScriptEnvironment::MakeWritable(PVideoFrame* pvf) {
  const PVideoFrame& vf = *pvf;

  // If the frame is already writable, do nothing.
  if (vf->IsWritable())
    return false;

  // Otherwise, allocate a new frame (using NewVideoFrame) and
  // copy the data into it.  Then modify the passed PVideoFrame
  // to point to the new buffer.
  Device* device = vf->GetFrameBuffer()->device;
  PVideoFrame dst;

#ifdef ENABLE_CUDA
  if (device->device_type == DEV_TYPE_CUDA) {
    // copy whole frame
    dst = GetOnDeviceFrame(vf, device);
    CopyCUDAFrame(dst, vf, threadEnv.get());
  }
  else
#endif
  {
    const int row_size = vf->GetRowSize();
    const int height = vf->GetHeight();

    bool alpha = 0 != vf->GetPitch(PLANAR_A);
    if (vf->GetPitch(PLANAR_U)) {  // we have no videoinfo, so we assume that it is Planar if it has a U plane.
      const int row_sizeUV = vf->GetRowSize(PLANAR_U); // for Planar RGB this returns row_sizeUV which is the same for all planes
      const int heightUV = vf->GetHeight(PLANAR_U);
      dst = NewPlanarVideoFrame(row_size, height, row_sizeUV, heightUV, frame_align, false, alpha, device);  // Always V first on internal images
    }
    else {
      dst = NewVideoFrameOnDevice(row_size, height, frame_align, device);
    }

    BitBlt(dst->GetWritePtr(), dst->GetPitch(), vf->GetReadPtr(), vf->GetPitch(), row_size, height);
    // Blit More planes (pitch, rowsize and height should be 0, if none is present)
    BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), vf->GetReadPtr(PLANAR_V),
      vf->GetPitch(PLANAR_V), vf->GetRowSize(PLANAR_V), vf->GetHeight(PLANAR_V));
    BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), vf->GetReadPtr(PLANAR_U),
      vf->GetPitch(PLANAR_U), vf->GetRowSize(PLANAR_U), vf->GetHeight(PLANAR_U));
    if (alpha)
      BitBlt(dst->GetWritePtr(PLANAR_A), dst->GetPitch(PLANAR_A), vf->GetReadPtr(PLANAR_A),
        vf->GetPitch(PLANAR_A), vf->GetRowSize(PLANAR_A), vf->GetHeight(PLANAR_A));
  }

  copyFrameProps(vf, dst);

  *pvf = dst;
  return true;
}


void ScriptEnvironment::AtExit(IScriptEnvironment::ShutdownFunc function, void* user_data) {
  at_exit.Add(function, user_data);
}

PVideoFrame ScriptEnvironment::Subframe(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height) {

  if (src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((new_pitch | rel_offset) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");

  VideoFrame* subframe;
  subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height);

  subframe->setProperties(src->getConstProperties());

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
  // automatically inserts if not exists!
  assert(NULL != subframe);

  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe,
    subframe->properties
  )); // insert with timestamp!

  return subframe;
}

//tsp June 2005 new function compliments the above function
PVideoFrame ScriptEnvironment::SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
  int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV) {
  if(src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((rel_offset | new_pitch | rel_offsetU | rel_offsetV | new_pitchUV) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");

  VideoFrame *subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV);

  subframe->setProperties(src->getConstProperties());

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
  // automatically inserts if not exists!
  assert(subframe != NULL);

  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe,
    subframe->properties
  )); // insert with timestamp!

  return subframe;
}

// alpha aware version
PVideoFrame ScriptEnvironment::SubframePlanar(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size,
  int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA) {
  if (src->GetFrameBuffer()->device->device_type == DEV_TYPE_CPU)
    if ((rel_offset | new_pitch | rel_offsetU | rel_offsetV | new_pitchUV | rel_offsetA) & (frame_align - 1))
      ThrowError("Filter Error: Filter attempted to break alignment of VideoFrame.");
  VideoFrame* subframe;
  subframe = src->Subframe(rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);

  subframe->setProperties(src->getConstProperties());

  size_t vfb_size = src->GetFrameBuffer()->GetDataSize();

  std::unique_lock<std::recursive_mutex> env_lock(memory_mutex); // vector needs locking!
                                                       // automatically inserts if not exists!
  assert(subframe != NULL);

  FrameRegistry2[vfb_size][src->GetFrameBuffer()].push_back(DebugTimestampedFrame(subframe,
    subframe->properties
  )); // insert with timestamp!

  return subframe;
}

void* ScriptEnvironment::ManageCache(int key, void* data) {
// An extensible interface for providing system or user access to the
// ScriptEnvironment class without extending the IScriptEnvironment
// definition.

  std::lock_guard<std::recursive_mutex> env_lock(memory_mutex);
  switch ((MANAGE_CACHE_KEYS)key)
  {
    // Called by Cache instances upon creation
  case MC_RegisterCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (FrontCache != NULL)
      CacheRegistry.push_back(FrontCache);
    FrontCache = cache;
    break;
  }
  // Called by Cache instances upon destruction
  case MC_UnRegisterCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (FrontCache == cache)
      FrontCache = NULL;
    else
      CacheRegistry.remove(cache);
    break;
  }
  // Called by Cache instances when they want to expand their limit
  case MC_NodAndExpandCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);

    // Nod
    if (cache != FrontCache)
    {
      CacheRegistry.move_to_back(cache);
    }

    // Given that we are within our memory limits,
    // try to expand the limit of those caches that
    // need it.
    // We try to expand most recently used caches first.

    int cache_cap = cache->SetCacheHints(CACHE_GET_CAPACITY, 0);
    int cache_reqcap = cache->SetCacheHints(CACHE_GET_REQUESTED_CAP, 0);
    if (cache_reqcap <= cache_cap)
      return 0;

    Device* device = cache->GetDevice();
    if ((device->memory_used > device->memory_max) || (device->memory_max - device->memory_used < device->memory_max*0.1f))
    {
      // If we don't have enough free reserves, take away a cache slot from
      // a cache instance that hasn't been used since long.

      for (Cache* old_cache : CacheRegistry)
      {
        if (old_cache->GetDevice() != device) {
          continue;
        }
        int osize = cache->SetCacheHints(CACHE_GET_SIZE, 0);
        if (osize != 0)
        {
          old_cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, osize - 1);
          break;
        }
      } // for cit
    }
#ifdef _DEBUG
    _RPT2(0, "ScriptEnvironment::ManageCache increase capacity to %d cache_id=%s\n", cache_cap + 1, cache->FuncName.c_str());
#endif
    cache->SetCacheHints(CACHE_SET_MAX_CAPACITY, cache_cap + 1);

    break;
  }
  // Called by Cache instances when they are accessed
  case MC_NodCache:
  {
    Cache* cache = reinterpret_cast<Cache*>(data);
    if (cache == FrontCache) {
      return 0;
    }

    CacheRegistry.move_to_back(cache);
    break;
  } // case
  case MC_RegisterMTGuard:
  {
    MTGuard* guard = reinterpret_cast<MTGuard*>(data);

    // If we already have a prefetcher, enable MT on the guard
    if (ThreadPoolRegistry.size() > 0)
    {
      guard->EnableMT(nMaxFilterInstances);
    }

    MTGuardRegistry.push_back(guard);

    break;
  }
  case MC_UnRegisterMTGuard:
  {
    MTGuard* guard = reinterpret_cast<MTGuard*>(data);
    for (auto& item : MTGuardRegistry)
    {
      if (item == guard)
      {
        item = NULL;
        break;
      }
    }
    break;
  }
  case MC_RegisterGraphNode:
  {
    FilterGraphNode* node = reinterpret_cast<FilterGraphNode*>(data);
    GraphNodeRegistry.push_back(node);
    break;
  }
  case MC_UnRegisterGraphNode:
  {
    FilterGraphNode* node = reinterpret_cast<FilterGraphNode*>(data);
    for (auto& item : GraphNodeRegistry)
    {
      if (item == node)
      {
        item = NULL;
        break;
      }
    }
    break;
  }
  case MC_QueryAvs25:
  {
    break; // not cache related
  }
  } // switch
  return 0;
}


bool ScriptEnvironment::PlanarChromaAlignment(IScriptEnvironment::PlanarChromaAlignmentMode key){
  bool oldPlanarChromaAlignmentState = PlanarChromaAlignmentState;

  switch (key)
  {
  case IScriptEnvironment::PlanarChromaAlignmentOff:
  {
    PlanarChromaAlignmentState = false;
    break;
  }
  case IScriptEnvironment::PlanarChromaAlignmentOn:
  {
    PlanarChromaAlignmentState = true;
    break;
  }
  default:
    break;
  }
  return oldPlanarChromaAlignmentState;
}

/* A helper for Invoke.
   Copy a nested array of 'src' into a flat array 'dst'.
   Returns the number of elements that have been written to 'dst'.
   If 'dst' is NULL, will still return the number of elements
   that would have been written to 'dst', but will not actually write to 'dst'.
*/
static size_t Flatten(const AVSValue& src, AVSValue* dst, size_t index, int level, const char* const* arg_names = NULL) {
  // level is starting from zero
  if (src.IsArray()
#ifdef NEW_AVSVALUE
    && level == 0
#endif
    ) { // flatten for the first arg level
    const int array_size = src.ArraySize();
    for (int i=0; i<array_size; ++i) {
      if (!arg_names || arg_names[i] == 0)
        index = Flatten(src[i], dst, index, level+1);
    }
  } else {
    if (dst != NULL)
      dst[index] = src;
    ++index;
  }
  return index;
}

const Function* ScriptEnvironment::Lookup(const char* search_name, const AVSValue* args, size_t num_args,
  bool& pstrict, size_t args_names_count, const char* const* arg_names, IScriptEnvironment2* ctx)
{
  AVSValue avsv;
  if (ctx->GetVarTry(search_name, &avsv) && avsv.IsFunction()) {
    //auto& funcv = avsv.AsFunction(); // c++ strict conformance: cannot Convert PFunction to PFunction&
    const PFunction& funcv = avsv.AsFunction();
    const char* name = funcv->GetLegacyName();
    const Function* func = funcv->GetDefinition();
    if (name != nullptr) {
      // wrapped function
      search_name = name;
    }
    else if (AVSFunction::TypeMatch(func->param_types, args, num_args, false, threadEnv.get()) &&
      AVSFunction::ArgNameMatch(func->param_types, args_names_count, arg_names))
    {
      pstrict = AVSFunction::TypeMatch(func->param_types, args, num_args, true, threadEnv.get());
      return func;
    }
  }

  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  const Function *result = NULL;

  size_t oanc;
  do {
    for (int strict = 1; strict >= 0; --strict) {
      pstrict = strict & 1;
      // first, look in loaded plugins or user defined functions
      result = plugin_manager->Lookup(search_name, args, num_args, pstrict, args_names_count, arg_names);
      if (result)
        return result;

      // then, look for a built-in function
      for (int i = 0; i < sizeof(builtin_functions)/sizeof(builtin_functions[0]); ++i)
        for (const AVSFunction* j = builtin_functions[i]; !j->empty(); ++j)
        {
          if (streqi(j->name, search_name)) {
            if (AVSFunction::TypeMatch(j->param_types, args, num_args, pstrict, ctx) &&
              AVSFunction::ArgNameMatch(j->param_types, args_names_count, arg_names))
              return j;
          }
        }
    }
    // Try again without arg name matching
    oanc = args_names_count;
    args_names_count = 0;
  } while (oanc);

  // If we got here it means the function has not been found.
  // If we haven't done so yet, load the plugins in the autoload folders
  // and try again.
  if (!plugin_manager->HasAutoloadExecuted())
  {
    plugin_manager->AutoloadPlugins();
    return Lookup(search_name, args, num_args, pstrict, args_names_count, arg_names, ctx);
  }

  return NULL;
}

bool ScriptEnvironment::CheckArguments(const Function* func, const AVSValue* args, size_t num_args,
  bool &pstrict, size_t args_names_count, const char* const* arg_names)
{
  if (AVSFunction::TypeMatch(func->param_types, args, num_args, false, threadEnv.get()) &&
    AVSFunction::ArgNameMatch(func->param_types, args_names_count, arg_names))
  {
    pstrict = AVSFunction::TypeMatch(func->param_types, args, num_args, true, threadEnv.get());
    return true;
  }
  return false;
}

#ifdef LISTARGUMENTS
// debug
static void ListArguments(const char *name, const AVSValue& args, int &level, bool flattened) {
  if (!strcmp(name, "Import"))
    return;
  if (!strcmp(name, "Eval"))
    return;
  if (level == 0)
    fprintf(stdout, "------- %s (%s)\r\n", name, flattened ? "flattened" : "orig");
  level++;
  if (args.IsArray()) {
    const int as = args.ArraySize();
    fprintf(stdout, "Array, size=%d {\r\n", as);
    for (int i = 0; i < as; i++) {
      fprintf(stdout, "Element#%d\r\n", i);
      ListArguments(name, args[i], level, flattened);
    }
    fprintf(stdout, "}\r\n");
  }
  else {
    if (!args.Defined())
      fprintf(stdout, "Undefined\r\n");
    else if(args.IsBool())
      fprintf(stdout, "Bool %s\r\n", args.AsBool() ? "true" : "false");
    else if (args.IsInt())
        fprintf(stdout, "Int %d\r\n", args.AsInt());
    else if (args.IsString())
      fprintf(stdout, "String %s\r\n", args.AsString());
    else if (args.IsFloat())
      fprintf(stdout, "Float %f\r\n", args.AsFloatf());
    else if (args.IsFunction())
      fprintf(stdout, "Function\r\n");
    else if (args.IsClip())
      fprintf(stdout, "Clip\r\n");
    else
      fprintf(stdout, "Unknown type\r\n");
  }
  level--;
}

static void ListArguments2(const char* name, const AVSValue* args, int& level, bool flattened, int len) {
  if (!strcmp(name, "Import"))
    return;
  if (!strcmp(name, "Eval"))
    return;
  fprintf(stdout, "------- %s (%s)\r\n", name, flattened ? "flattened" : "orig");
  level++;
  for (int i = 0; i < len; i++) {
    ListArguments(name, *(args +i), level, flattened);
  }
}
#endif

bool ScriptEnvironment::Invoke_(AVSValue *result, const AVSValue& implicit_last,
  const char* name, const Function *f, const AVSValue& args, const char* const* arg_names,
  InternalEnvironment* env_thread, bool is_runtime)
{

  const int args_names_count = (arg_names && args.IsArray()) ? args.ArraySize() : 0;

  if (name == nullptr) {
    // for debug printing
    name = "<anonymous function>";
  }

  // Step #1: Flattening
  // arrays received in the place of unnamed parameters are flattened back
  // to have them as a list again, in order to be compatible with Avisynth's "array elements as comma
  // delimited parameters" definition style.

  // get how many args we will need to store
  size_t args2_count = Flatten(args, NULL, 0, 0, arg_names);
  if (args2_count > ScriptParser::max_args)
    ThrowError("Too many arguments passed to function (max. is %d)", ScriptParser::max_args);

  // flatten unnamed args
  std::vector<AVSValue> args2(args2_count + 1, AVSValue());
  args2[0] = implicit_last;
  Flatten(args, args2.data() + 1, 0, 0, arg_names);

#ifdef LISTARGUMENTS
  // debug list of Invoke arguments before-after flattening
  int level = 0;
  ListArguments(name, args, level, false); // unflattened remark
  ListArguments2(name, args2.data()+1, level, true, args2_count); // flattened remark
#endif

  // Step #2: Arguments check and function detection by matching signature.

  bool strict = false;
  int argbase = 1;
  if (f != nullptr) {
    // check arguments
    if (!this->CheckArguments(f, args2.data() + 1, args2_count, strict, args_names_count, arg_names)) {
      if (!implicit_last.Defined() ||
        !this->CheckArguments(f, args2.data(), args2_count + 1, strict, args_names_count, arg_names))
        return false;
      argbase = 0;
      args2_count += 1;
    }
  }
  else {
    // Because only one level is flattened, for 2+ Dimension arrays result are
    // at least two parameters which are still arrays)
    // [3,4,5] is flattened as 3,4,5
    // clip, [[2,3], [3,4,5]] is flattened as clip, [2,3], [3,4,5]

    // find matching function
    f = this->Lookup(name, args2.data() + 1, args2_count, strict, args_names_count, arg_names, env_thread);
    if (!f)
    {
      if (!implicit_last.Defined())
        return false;
      // search function definitions with implicite "last" given
      f = this->Lookup(name, args2.data(), args2_count + 1, strict, args_names_count, arg_names, env_thread);
      if (!f)
        return false;
      argbase = 0;
      args2_count += 1;
    }
  }

  // Problem: Animate has parameter signature both "iis.*" and "ciis.*"
  //   ColorBars()
  //   Animate(0, 100, "blur", 0.1, 1.5)
  // Here we find "iis.*" but it turnes out that its given function parameter "Blur" requires a clip
  // Thus we got an exception later during the filter instantiation (really it is "Blur" who throws the exception)
  // (see comment Issue20200818 later).
  // Expression evaluator would catch NotFound and reissue _Invoke with a forced implicit_last in args.

  // Step #3: Unnamed arguments

  // combine unnamed args into arrays
  size_t src_index = 0;
  const char* p = f->param_types;

  std::vector<AVSValue> args3;
  std::vector<bool> args3_really_filled;

  bool last_was_named = false;
  while (*p) {
    if (*p == '[') {
      // named parameter check: between brackets (no name validity check, empty is valid as well)
      const char* pstart = p + 1;
      p = strchr(pstart, ']');
      if (!p) break;
      last_was_named = true;
      p++; // skip closing bracket
    }
    else if (((p[1] == '*') || (p[1] == '+'))) {
      // Case of unnamed arrays or named arrays
      // Special named array: [] with no real name
      // Pre 3.6: filling up named arrays with names from script level was not possible.
      // Memo: array function signature types: * means "zero or more". + means "one or more"
      // Example fake-named array: the first clip array parameter of BlankClip "[]c*[length]i etc.
      // Some functions with array signatures:
      //   Select i.+
      //   SelectEvery: cii*
      //   Format: s.*
      //   MPP_SharedMemoryServer: csi[aux_clips]c*[max_cache_frames]i
      //   BlankClip []c*[length]i (fake named array)
      //   ArrayGet .i+
      //   ArraySize .
      //   Array/ArrayCreate .*: [[2,3,4], [1,2]] -> [2,3,4], [1,2]
      //   user_script_function(clip, int_array) c[]i*
      // Pre v3.6 script-level array definition: autodetect comma separated list elements
      // The parser detects the end of array: the next argument type is different.
      // Since the end of free-typed (".+" or ".*") arrays cannot be recognized, they are allowed to appear
      // only at the very end of the parameter signature
      // Option since 3.6: bracket-style array definition syntax
      size_t start = src_index;
      const char arg_type = *p;


      if (src_index < args2_count &&
        args2[argbase + src_index].IsArray() 
        // After flattening this only happens when an explicite array was passed.
        // typed array recognition is easy and unambigous
        && arg_type != '.' // to avoid .+ case when one can pass array of arrays like [[1,2,3],[4,5]] flattened to [1,2,3],[4,5]
        )
      {
        if (arg_type != '.') {
          if (!AVSFunction::SingleTypeMatchArray(arg_type, args2[argbase + src_index], strict))
            ThrowError("Array elements do not match with the specified type in function %s", name);
        }

        if(p[1] == '+' && args2[argbase + src_index].ArraySize() == 0) // one or more
          ThrowError("An array with zero element size was given to function %s which expects a 'one-or-more' style array argument", name);
        args3.push_back(args2[argbase + src_index]); // can't delete args2 early because of this
        args3_really_filled.push_back(true); // valid array content
        src_index++;
      }
      else
      {
        // Collect consecutive similary-typed parameters into an array by automatic detection of its end:
        // The end of a simple-typed array:
        // - when the comma separated parameter list is changing type
        // - no more parameters (list ends)
        // E.g. collect values into an integer array (i*) until values in the list are still integers.
        // The array collection in parameter sequence 1, 2, 3, "hello1", "hello2" will stop before "hello1"
        // because type is changed from i to s.
        // This is why the end of an 'any-type' array (.*) cannot be detected from a comma delimited list:
        // no stopping condition (no type), only the end of list can stop collecting.
        while ((src_index < args2_count)) {
          const bool match = AVSFunction::SingleTypeMatch(arg_type, args2[argbase + src_index], strict);
          if (!match)
            break;
          src_index++;
        }
        size_t size = src_index - start; // so we have size number of items detected. Put them back into an array.
        assert(args2_count >= size);

        // Even if the AVSValue below is an array of zero size, we can't skip adding it to args3,
        // because filters like BlankClip []c* or MPP_SharedMemoryServer might still be expecting it.
        // 3.7.1.: This statement is no longer true. BlankClip was modified to handle Undefined AVSValue instead of array[] properly

/*
  Considerations on resolving parameter handling for "array of anything" parameter when array(s) would be passed directly.
  Memo:
  - Avisynth signature: .* or .+
  - Script function specifier val_array or val_array_nz

  When parameter signature is array of anything (.* or .+) and the
  parameter is passed unnamed (even if it is a named parameter) then
  there is an ambiguos situation which 

  Giving a parameter list of 1,2,3 will be detected as [1,2,3] (compatibility: list is grouped together into an array internally).
  E.g. 1 will be detected as [1]
  Passing nothing from an Avisynth script in the place of the parameter will be detected as [] (array size is zero, value is defined)
  initially, then later in the named parameter processing procedure it can be overridden directly by a named value.
  
  This rule means that a directly given script array e.g. [1,2,3] will be detected as [[1,2,3]], because unnamed and untyped parameters are
  put together into an array, which has the size of the list. This is a list of 1 element which happens to be an array.
  Avisynth cannot 'guess' whether we want to define a single array directly or this array is the only one part of the list.
  So an unnamedly passed [1,2,3] will appear as [ [1,2,3] ] in the unnamed "array of anything" parameter value.

  Syntax hint:
  When someone would like to pass a directly specified array (e.g. [1,2,3] instead of 1,2,3) to a .+ or .* parameter
  the parameter must be passed by name to avoid ambiguity!

  Example script function definition:
    function foo(val_array "n")

  Value seen inside the function body:
      Call                          Value of n
      foo()                         Undefined
      foo(1)                        [1] (*)
      foo(1,2,3)                    [1,2,3] (*)
      foo([1,2,3])            !     [[1,2,3]] (*)
      foo([1,2,3],[4,5])      !     [[1,2,3],[4,5]] (*)
      foo(n=[1,2,3])                [1,2,3]
      foo(n=[[1,2,3],[4,5]])        [[1,2,3],[4,5]]
      foo(n=[])                     []
      foo(n=1)                      [1] (**)
      foo(n="hello")                ["hello"]  (**)

    *  compatible Avisynth way: comma delimited consecutive values form an array
    ** simple-type value passed to a named array parameter will be created as a 1-element real array

      // unnamed signature
    function foo(val_array n)
      Call                          n
      foo()                         [] (defined and array size is zero) Avisynth compatible behaviour
*/

        if (size == 0 && p[1] == '+') // '+': one or more
        {
          if(!last_was_named)
            ThrowError("A zero-sized array (or empty parameter list) appeared in the place of an unnamed parameter to function %s which expects a 'one-or-more' style array argument", name);
          args3.push_back(AVSValue()); // push undefined
          args3_really_filled.push_back(false); // zero sized (not found) array can be specified later with argname
        }
        else {
          if (size == 0) {
            if (last_was_named)
              args3.push_back(AVSValue());
            // Named array is left Undefined here. In a later section it can be overridden with named argument
            // This ensures that passing empty [] is different than not passing anything (Defined() vs. Undefined())
            // because of this (undefined instead of zero-sized array) BlankClip was changed to handle to allow the parameter as undefined.
            else {
              args3.push_back(AVSValue(NULL, 0));
              // Unnamed parameter: array of zero size.
              // Fixme cosmetics: Maybe this one even cannot be reached, because an unnamed compulsory parameter cannot be 'zero or more' array
            }
          }
          else
            // create a proper array from the list of elements
            args3.push_back(AVSValue(args2.data() + argbase + start, (int)size)); // can't delete args2 early because of this

          if (last_was_named && size > 0)
            args3_really_filled.push_back(true); // valid array content
          else
            args3_really_filled.push_back(false); // zero sized (not found) array can be specified later with argname
        }
      }

      p += 2; // skip type-char and '*' or '+'
      last_was_named = false;
    }
    else {
      if (src_index < args2_count) {
        // At this point we could still have an array in AVSValue because a directly given (e.g. [1,2,3]) array is accepted at the place of a "." (any) type
        // The signature checker recognizes and reports a signature match in AVSFunction::TypeMatch
        // Example call: fn([1,2,3]) where fn has a signature of "c[n]."
        // Allowing it does not do any harm
        // FIXME, When removed, we'd better remove the limitation in (look for: QWERTZUIOP)
#ifdef DISABLE_ARRAYS_WHEN_DOT_ALONE
        if (args2[argbase + src_index].IsArray() && p[0] != 'a') {
          ThrowError((std::string("An array is passed to a non-array parameter type in function ") + std::string(name)).c_str());
        }
        else 
#endif
        {
          args3.push_back(args2[argbase + src_index]);
        }
        args3_really_filled.push_back(true);
      }
      else {
        args3.push_back(AVSValue());
        args3_really_filled.push_back(false);
      }
      src_index++;
      p++;
      // Skip possible array marker for named arrays like [colors]f+
      // Note: this declaration style is recognized only for versions >= 3.5.3
      // so plugins using this syntax won't load/work with older AviSynth versions
      // Older versions without this check will report various errors like "named parameter was given more than once"
      // but only when given with names. Providing a named parameter as unnamed was possible from 
      // direct plugin use formerly as well.
      if ((p[0] == '*') || (p[0] == '+'))
        p++;

      last_was_named = false;
    }
  }
  if (src_index < args2_count)
    ThrowError("Too many arguments to function %s", name);

  // Step #4: Named arguments

  // copy named args
  for (int i = 0; i<args_names_count; ++i) {
    if (arg_names[i]) {
      size_t named_arg_index = 0;
      for (const char* p = f->param_types; *p; ++p) {
        if (*p == '*' || *p == '+') {
          continue;   // without incrementing named_arg_index
        }
        else if (*p == '[') {
          p += 1;
          const char* q = strchr(p, ']');
          if (!q) break;
          if (strlen(arg_names[i]) == size_t(q - p) && !_strnicmp(arg_names[i], p, q - p)) {
            // we have a match
            if (args3[named_arg_index].Defined() && args3_really_filled[named_arg_index]) {
              // when a parameter like named array was filled as an empty array
              // from the unnamed section we don't throw error for the first time
              ThrowError("Script error: the named argument \"%s\" was passed more than once (twice as named or first unnamed then named) to %s", arg_names[i], name);
            }
#ifndef NEW_AVSVALUE
            //PF 161028 AVS+ arrays as named arguments
            else if (args[i].IsArray()) {
              ThrowError("Script error: can't pass an array as a named argument");
            }
#endif
            else if (args[i].Defined() && args[i].IsArray() && ((q[2] == '*' || q[2] == '+')) && !AVSFunction::SingleTypeMatchArray(q[1], args[i], false))
            {
              // e.g. passing colors=[235, 128, "Hello"] to [colors]f+
              ThrowError("Script error: the named array argument \"%s\" to %s had a wrong element type", arg_names[i], name);
            }
            else if (args[i].Defined() && !args[i].IsArray() && !AVSFunction::SingleTypeMatch(q[1], args[i], false))
            {
              ThrowError("Script error: the named argument \"%s\" to %s had the wrong type", arg_names[i], name);
            }
            else {
              if (args[i].Defined() && args[i].IsArray()) {
                // e.g. foo(sigma=[1.0, 1.1]) to [sigma]f is invalid 
                if (!(q[2] == '*' || q[2] == '+') && q[1] != '.' && q[1] != 'a')
                  ThrowError("Script error: the named argument \"%s\" to %s had the wrong type (passed an array to a non-array and not-any parameter)", arg_names[i], name);
                if (q[2] == '+' && args[i].ArraySize() == 0)
                  ThrowError("Script error: the named argument \"%s\" to %s is a 'one-or-more' element style array but had zero element count.", arg_names[i], name);
              }
              // Note (after having array type parameters in script functions)
              // When passing a simple value to an array parameter, we should really make an array of it
              // or else script function parameters of array types won't see this parameter as an array inside the function body.
              // Note: Unlike AVS scripts a real plugin - through Avisynth interface - is 
              // - allowed to index an AVSValue which is not even an array. Index 0 returns the simple-type value itself.
              // - calling AVSValue ArraySize() returns 1
              // But an AVS script syntax indexing and ArraySize() requires a real array, simple base type is not allowed there.
              if(args[i].Defined() && !args[i].IsArray() && (q[2] == '*' || q[2] == '+' || q[1] == 'a'))
                args3[named_arg_index] = AVSValue(&args[i],1); // really create an array with one element
              else
                args3[named_arg_index] = args[i];
              args3_really_filled[named_arg_index] = true;
              goto success;
            }
          }
          else {
            p = q + 1;
          }
        }
        named_arg_index++;
      }
      // failure
      ThrowError("Script error: %s does not have a named argument \"%s\"", name, arg_names[i]);
    success:;
    }
  }

  std::vector<AVSValue>(args3).shrink_to_fit();

  // end of parameter matching and parsing

  if(is_runtime) {
    // Invoked by a thread or GetFrame
    AVSValue funcArgs(args3.data(), (int)args3.size());

    if(f->isAvs25) // like GRunT's AverageLuma wrapper
      *result = f->apply(funcArgs, f->user_data, (IScriptEnvironment *)((IScriptEnvironment_Avs25 *)env_thread));
    else
    *result = f->apply(funcArgs, f->user_data, env_thread);
    return true;
  }

  std::lock_guard<std::recursive_mutex> env_lock(memory_mutex);

  ScopedCounter suppressThreadCount_(threadEnv->GetSuppressThreadCount());

  // chainedCtor is true if we are being constructed inside/by the
  // constructor of another filter. In that case we want MT protections
  // applied not here, but by the Invoke() call of that filter.
  const bool chainedCtor = invoke_stack.size() > 0;

  MtModeEvaluator mthelper;
#ifdef USE_MT_GUARDEXIT
  std::vector<MTGuardExit*> GuardExits;
#endif

  bool foundClipArgument = false;
  for (int i = argbase; i < (int)args2.size(); ++i)
  {
    auto& argx = args2[i];
#ifndef NEW_AVSVALUE
    assert(!argx.IsArray()); // todo: we can have arrays 161106
#endif
    // todo PF 161112 new arrays: recursive look into arrays whether they contain clips
    if (argx.IsClip())
    {
      foundClipArgument = true;

      const PClip &clip = argx.AsClip();
      IClip *clip_raw = (IClip*)((void*)clip);
      ClipDataStore *data = this->ClipData(clip_raw);

      if (!data->CreatedByInvoke)
      {
#ifdef _DEBUG
        _RPT3(0, "ScriptEnvironment::Invoke.AddChainedFilter %s thread %d this->DefaultMtMode=%d\n", name, GetCurrentThreadId(), (int)this->DefaultMtMode);
#endif
        mthelper.AddChainedFilter(clip, this->DefaultMtMode);
      }

#ifdef USE_MT_GUARDEXIT
      // Wrap this input parameter into a guard exit, which is used when
      // the new clip created later below is MT_SERIALIZED.
      MTGuardExit *ge = new MTGuardExit(argx.AsClip(), name);
      GuardExits.push_back(ge);
      argx = ge;
#endif
    }
  }
  bool isSourceFilter = !foundClipArgument;

  auto call_env = f->isAvs25 ? nullptr : threadEnv.get();
  auto call_env25 = f->isAvs25 ? threadEnv.get()->GetEnv25() : nullptr;
  // ... and we're finally ready to make the call
  std::unique_ptr<const FilterConstructor> funcCtor =
    std::make_unique<const FilterConstructor>(call_env, call_env25 , f, &args2, &args3);
  _RPT1(0, "ScriptEnvironment::Invoke after funcCtor make unique %s\r\n", name);

  // args2 and args3 are not valid after this point anymore

  bool is_mtmode_forced;
  bool filterHasSpecialMT = this->GetFilterMTMode(f, &is_mtmode_forced) == MT_SPECIAL_MT;

  if (filterHasSpecialMT) // pre-avs 3.6 workaround for MP_Pipeline
  {
    *result = funcCtor->InstantiateFilter();
#ifdef _DEBUG
    _RPT1(0, "ScriptEnvironment::Invoke done funcCtor->InstantiateFilter %s\r\n", name);
#endif
  }
  else if (funcCtor->IsScriptFunction())
  {
    // Eval, EvalOop, Import and user defined script functions
    // Warn user if he set an MT-mode for a script function
    if (this->FilterHasMtMode(f))
    {
      OneTimeLogTicket ticket(LOGTICKET_W1010, f);
      LogMsgOnce(ticket, LOGLEVEL_WARNING, "An MT-mode is set for %s() but it is a script function. You can only set the MT-mode for binary filters, for scripted functions it will be ignored.", f->name);
    }

    *result = funcCtor->InstantiateFilter();
#ifdef _DEBUG
    _RPT1(0, "ScriptEnvironment::Invoke done funcCtor->InstantiateFilter %s\r\n", name);
#endif
  }
  else
  {
#ifdef _DEBUG
    Cache *PrevFrontCache = FrontCache;
#endif

    AVSValue fret;

    invoke_stack.push(&mthelper);
    try
    {
      fret = funcCtor->InstantiateFilter();
      invoke_stack.pop();
    }
    catch (...)
    {
      // comment: Issue20200818
      invoke_stack.pop();
      throw;
    }

    // Determine MT-mode, as if this instance had not called Invoke()
    // in its constructor. Note that this is not necessary the final
    // MT-mode.
    // PF 161012 hack(?) don't call if prefetch. If effective mt mode is MT_MULTI, then
    // Prefetch create gets called again
    // Prefetch is activated above in: fret = funcCtor->InstantiateFilter();
    if (fret.IsClip() && (f->name == nullptr || strcmp(f->name, "Prefetch")))
    {
      const PClip &clip = fret.AsClip();

      bool is_mtmode_forced;
      this->GetFilterMTMode(f, &is_mtmode_forced);
      MtMode mtmode = MtModeEvaluator::GetMtMode(clip, f, threadEnv.get());

      if (chainedCtor)
      {
        // Propagate information about our children's MT-safety
          // to our parent.
        invoke_stack.top()->Accumulate(mthelper);

        // Add our own MT-mode's information to the parent.
        invoke_stack.top()->Accumulate(mtmode);

        *result = fret;
      }
      else
      {
        if (!is_mtmode_forced) {
          mtmode = mthelper.GetFinalMode(mtmode);
        }

        // Special handling for source filters
        if (isSourceFilter
          && MtModeEvaluator::UsesDefaultMtMode(clip, f, threadEnv.get())
          && (MT_SERIALIZED != mtmode))
        {
          mtmode = MT_SERIALIZED;
          OneTimeLogTicket ticket(LOGTICKET_W1001, f);
          LogMsgOnce(ticket, LOGLEVEL_INFO, "%s() does not have any MT-mode specification. Because it is a source filter, it will use MT_SERIALIZED instead of the default MT mode.", f->canon_name);
        }


        *result = MTGuard::Create(mtmode, clip, std::move(funcCtor), threadEnv.get());

#ifdef USE_MT_GUARDEXIT
        // 170531: concept introduced in r2069 is not working
        // Mutex of serialized filters are unlocked and allow to call
        // such filters as MT_NICE_FILTER in a reentrant way
        // Kept for reference, but put in USE_MT_GUARDEXIT define.

        // Activate the guard exists. This allows us to exit the critical
        // section encompassing the filter when execution leaves its routines
        // to call other filters.
        if (MT_SERIALIZED == mtmode)
        {
          for (auto &ge : GuardExits)
          {
            _RPT3(0, "ScriptEnvironment::Invoke.ActivateGuard %s thread %d\n", name, GetCurrentThreadId());
            ge->Activate(guard);
          }
        }
#endif

        IClip *clip_raw = (IClip*)((void*)clip);
        ClipDataStore *data = this->ClipData(clip_raw);
        data->CreatedByInvoke = true;
      } // if (chainedCtor)

      // Nekopanda: moved here from above.
      // some filters invoke complex filters in its constructor, and they need cache.
      *result = CacheGuard::Create(*result, NULL, threadEnv.get());

      // Check that the filter returns zero for unknown queries in SetCacheHints().
      // This is actually something we rely upon.
      if ((clip->GetVersion() >= 5) && (0 != clip->SetCacheHints(CACHE_USER_CONSTANTS, 0)))
      {
        OneTimeLogTicket ticket(LOGTICKET_W1002, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() violates semantic contracts and may cause undefined behavior. Please inform the author of the plugin.", f->canon_name);
      }

      // Warn user if the MT-mode of this filter is unknown
      if (MtModeEvaluator::UsesDefaultMtMode(clip, f, threadEnv.get()) && !isSourceFilter)
      {
        OneTimeLogTicket ticket(LOGTICKET_W1004, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() has no MT-mode set and will use the default MT-mode. This might be dangerous.", f->canon_name);
      }

      // Warn user if he forced an MT-mode that differs from the one specified by the filter itself
      if (is_mtmode_forced
        && MtModeEvaluator::ClipSpecifiesMtMode(clip)
        && MtModeEvaluator::GetInstanceMode(clip) != mtmode)
      {
        OneTimeLogTicket ticket(LOGTICKET_W1005, f);
        LogMsgOnce(ticket, LOGLEVEL_WARNING, "%s() specifies an MT-mode for itself, but a script forced a different one. Either the plugin or the script is erronous.", f->canon_name);
      }

      // Inform user if a script unnecessarily specifies an MT-mode for this filter
      if (!is_mtmode_forced
        && this->FilterHasMtMode(f)
        && MtModeEvaluator::ClipSpecifiesMtMode(clip))
      {
        OneTimeLogTicket ticket(LOGTICKET_W1006, f);
        LogMsgOnce(ticket, LOGLEVEL_INFO, "Ignoring unnecessary MT-mode specification for %s() by script.", f->canon_name);
      }

    } // if (fret.IsClip())
    else
    {
      *result = fret;
    }

    // static device check
    // this is not enough to check all dependencies but much helpful to users
    if ((*result).IsClip()) {
      auto last = (argbase == 0) ? implicit_last : AVSValue();
      CheckChildDeviceTypes((*result).AsClip(), name, last, args, arg_names, threadEnv.get());
    }

    // filter graph
    if (graphAnalysisEnable && (*result).IsClip()) {
      auto last = (argbase == 0) ? implicit_last : AVSValue();
      *result = new FilterGraphNode((*result).AsClip(), f->name, last, args, arg_names, threadEnv.get());
    }

#ifdef _DEBUG
    if (PrevFrontCache != FrontCache && FrontCache != NULL) // cache registering swaps frontcache to the current
    {
      _RPT2(0, "ScriptEnvironment::Invoke done Cache::Create %s  cache_id=%p\r\n", name, (void*)FrontCache);
      FrontCache->FuncName = name; // helps debugging. See also in cache.cpp
    }
    else {
      _RPT1(0, "ScriptEnvironment::Invoke done Cache::Create %s\r\n", name);
    }
#endif
  }

  return true;
}


bool ScriptEnvironment::FunctionExists(const char* name)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);

  // Look among variable table
  AVSValue result;
  if (threadEnv->GetVarTry(name, &result)) {
    if (result.IsFunction()) {
      return true;
    }
  }

  // Look among internal functions
  if (InternalFunctionExists(name))
    return true;

  // Look among plugin functions
  if (plugin_manager->FunctionExists(name))
    return true;

  // Uhh... maybe if we load the plugins we'll have the function
  if (!plugin_manager->HasAutoloadExecuted())
  {
    plugin_manager->AutoloadPlugins();
    return this->FunctionExists(name);
  }

  return false;
}

bool ScriptEnvironment::InternalFunctionExists(const char* name)
{
  for (int i = 0; i < sizeof(builtin_functions)/sizeof(builtin_functions[0]); ++i)
    for (const AVSFunction* j = builtin_functions[i]; !j->empty(); ++j)
      if (streqi(j->name, name))
        return true;

  return false;
}

void ScriptEnvironment::BitBlt(BYTE* dstp, int dst_pitch, const BYTE* srcp, int src_pitch, int row_size, int height) {
  if (height<0)
    ThrowError("Filter Error: Attempting to blit an image with negative height.");
  if (row_size<0)
    ThrowError("Filter Error: Attempting to blit an image with negative row size.");
  ::BitBlt(dstp, dst_pitch, srcp, src_pitch, row_size, height);
}

void ScriptEnvironment::ThrowError(const char* fmt, ...)
{
  va_list val;
  va_start(val, fmt);
  threadEnv->VThrowError(fmt, val);
  va_end(val);
}

void ScriptEnvironment::VThrowError(const char* fmt, va_list va)
{
  threadEnv->VThrowError(fmt, va);
}

// since IF V8 it moved from IScriptEnvironment2 to IScriptEnvironment
PVideoFrame ScriptEnvironment::SubframePlanarA(PVideoFrame src, int rel_offset, int new_pitch, int new_row_size, int new_height, int rel_offsetU, int rel_offsetV, int new_pitchUV, int rel_offsetA)
{
  return SubframePlanar(src, rel_offset, new_pitch, new_row_size, new_height, rel_offsetU, rel_offsetV, new_pitchUV, rel_offsetA);
}

bool ScriptEnvironment::MakePropertyWritable(PVideoFrame* pvf)
{
  const PVideoFrame& vf = *pvf;

  // If the frame is already writable, do nothing.
  if (vf->IsPropertyWritable())
    return false;

  // Otherwise, allocate a new frame (using Subframe)
  // Thus we avoid the frame-content copy overhead and still get a new frame with its unique frameprop
  PVideoFrame dst;
  if (vf->GetPitch(PLANAR_A)) {
    // planar + alpha
    dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight(), 0, 0, vf->GetPitch(PLANAR_U), 0);
  }
  else if (vf->GetPitch(PLANAR_U)) {
    // planar
    dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight(), 0, 0, vf->GetPitch(PLANAR_U));
  }
  else {
    // single plane
    dst = vf->Subframe(0, vf->GetPitch(), vf->GetRowSize(), vf->GetHeight());
  }

  copyFrameProps(vf, dst);
  *pvf = dst;
  return true;
}

// since IF V8 frame property helpers are part of IScriptEnvironment
void ScriptEnvironment::copyFrameProps(const PVideoFrame& src, PVideoFrame& dst)
{
  dst->setProperties(src->getProperties());
}

// frame properties support
// core imported from VapourSynth
// from vsapi.cpp
const AVSMap* ScriptEnvironment::getFramePropsRO(const PVideoFrame& frame) AVS_NOEXCEPT {
  assert(frame);
  return &(frame->getConstProperties());
}

AVSMap* ScriptEnvironment::getFramePropsRW(PVideoFrame &frame) AVS_NOEXCEPT {
  assert(frame);
  return &(frame->getProperties());
}

int ScriptEnvironment::propNumKeys(const AVSMap* map) AVS_NOEXCEPT {
  assert(map);
  return static_cast<int>(map->size());
}

const char* ScriptEnvironment::propGetKey(const AVSMap* map, int index) AVS_NOEXCEPT {
  assert(map);
  if (index < 0 || static_cast<size_t>(index) >= map->size())
    ThrowError(("propGetKey: Out of bounds index " + std::to_string(index) + " passed. Valid range: [0," + std::to_string(map->size() - 1) + "]").c_str());

  return map->key(index);
}

static int propNumElementsInternal(const AVSMap* map, const std::string& key) AVS_NOEXCEPT {
  FramePropVariant* val = map->find(key);
  return val ? (int)val->size() : -1;
}


int ScriptEnvironment::propNumElements(const AVSMap* map, const char* key) AVS_NOEXCEPT {
  assert(map && key);
  return propNumElementsInternal(map, key);
}

char ScriptEnvironment::propGetType(const AVSMap* map, const char* key) AVS_NOEXCEPT {
  assert(map && key);
  const char a[] = { 'u', 'i', 'f', 's', 'c', 'v', 'm' };
  FramePropVariant* val = map->find(key);
  return val ? a[val->getType()] : 'u';
}

int ScriptEnvironment::propDeleteKey(AVSMap* map, const char* key) AVS_NOEXCEPT {
  assert(map && key);
  return map->erase(key);
}

#define PROP_GET_SHARED(vt, retexpr) \
    assert(map && key); \
    if (map->hasError()) \
        ThrowError("Attempted to read key '%s' from a map with error set: %s", key, map->getErrorMessage().c_str()); \
    int err = 0; \
    FramePropVariant *l = map->find(key); \
    if (l && l->getType() == (vt)) { \
        if (index >= 0 && static_cast<size_t>(index) < l->size()) { \
            if (error) \
                *error = 0; \
            return (retexpr); \
        } else { \
            err |= AVSGetPropErrors::GETPROPERROR_INDEX; \
        } \
    } else if (l) { \
        err |= AVSGetPropErrors::GETPROPERROR_TYPE; \
    } else { \
        err = AVSGetPropErrors::GETPROPERROR_UNSET; \
    } \
    if (!error) \
        ThrowError("Property read unsuccessful but no error output: %s", key); \
    *error = err; \
    return 0;

int64_t ScriptEnvironment::propGetInt(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  PROP_GET_SHARED(FramePropVariant::vInt, l->getValue<int64_t>(index))
}

double ScriptEnvironment::propGetFloat(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  PROP_GET_SHARED(FramePropVariant::vFloat, l->getValue<double>(index))
}

const char* ScriptEnvironment::propGetData(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  PROP_GET_SHARED(FramePropVariant::vData, l->getValue<VSMapData>(index)->c_str())
}

int ScriptEnvironment::propGetDataSize(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  PROP_GET_SHARED(FramePropVariant::vData, static_cast<int>(l->getValue<VSMapData>(index)->size()))
}

PClip ScriptEnvironment::propGetClip(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  PROP_GET_SHARED(FramePropVariant::vClip, l->getValue<PClip>(index))
}

const PVideoFrame ScriptEnvironment::propGetFrame(const AVSMap* map, const char* key, int index, int* error) AVS_NOEXCEPT {
  // PVideoFrame itself is reference counted
  PROP_GET_SHARED(FramePropVariant::vFrame, l->getValue<PVideoFrame>(index))
}

static inline bool isAlphaUnderscore(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static inline bool isAlphaNumUnderscore(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
}

static bool isValidVSMapKey(const std::string& s) {
  size_t len = s.length();
  if (!len)
    return false;

  if (!isAlphaUnderscore(s[0]))
    return false;
  for (size_t i = 1; i < len; i++)
    if (!isAlphaNumUnderscore(s[i]))
      return false;
  return true;
}

// insert and append are guarded and make new copy of actual storage before the modification
#define PROP_SET_SHARED(vv, appendexpr) \
    assert(map && key); \
    if (append != AVSPropAppendMode::PROPAPPENDMODE_REPLACE && \
        append != AVSPropAppendMode::PROPAPPENDMODE_APPEND && \
        append != AVSPropAppendMode::PROPAPPENDMODE_TOUCH) \
        ThrowError("Invalid prop append mode given when setting key '%s'", key); \
    std::string skey = key; \
    if (!isValidVSMapKey(skey)) \
        return 1; \
    if (append != AVSPropAppendMode::PROPAPPENDMODE_REPLACE && map->contains(skey)) { \
        FramePropVariant &l = map->at(skey); \
        if (l.getType() != (vv)) \
            return 1; \
        else if (append == AVSPropAppendMode::PROPAPPENDMODE_APPEND) \
            map->append(skey, appendexpr); \
    } else { \
        FramePropVariant l((vv)); \
        if (append != AVSPropAppendMode::PROPAPPENDMODE_TOUCH) \
            l.append(appendexpr); \
        map->insert(skey, std::move(l)); \
    } \
    return 0;


int ScriptEnvironment::propSetInt(AVSMap* map, const char* key, int64_t i, int append) AVS_NOEXCEPT {
  PROP_SET_SHARED(FramePropVariant::vInt, i)
}

int ScriptEnvironment::propSetFloat(AVSMap* map, const char* key, double d, int append) AVS_NOEXCEPT {
  PROP_SET_SHARED(FramePropVariant::vFloat, d)
}

int ScriptEnvironment::propSetData(AVSMap* map, const char* key, const char* d, int length, int append) AVS_NOEXCEPT {
  PROP_SET_SHARED(FramePropVariant::vData, length >= 0 ? std::string(d, length) : std::string(d))
}

int ScriptEnvironment::propSetClip(AVSMap* map, const char* key, PClip& clip, int append) AVS_NOEXCEPT {
  PROP_SET_SHARED(FramePropVariant::vClip, clip)
}

int ScriptEnvironment::propSetFrame(AVSMap* map, const char* key, const PVideoFrame &frame, int append) AVS_NOEXCEPT {
  PROP_SET_SHARED(FramePropVariant::vFrame, frame)
}

const int64_t* ScriptEnvironment::propGetIntArray(const AVSMap* map, const char* key, int* error) AVS_NOEXCEPT {
  int index = 0;
  PROP_GET_SHARED(FramePropVariant::vInt, l->getArray<int64_t>())
}

const double* ScriptEnvironment::propGetFloatArray(const AVSMap* map, const char* key, int* error) AVS_NOEXCEPT {
  int index = 0;
  PROP_GET_SHARED(FramePropVariant::vFloat, l->getArray<double>())
}

int ScriptEnvironment::propSetIntArray(AVSMap* map, const char* key, const int64_t* i, int size) AVS_NOEXCEPT {
  assert(map && key && size >= 0);
  if (size < 0)
    return 1;
  std::string skey = key;
  if (!isValidVSMapKey(skey))
    return 1;
  FramePropVariant l(FramePropVariant::vInt);
  l.setArray(i, size);
  map->insert(skey, std::move(l));
  return 0;
}

 int ScriptEnvironment::propSetFloatArray(AVSMap* map, const char* key, const double* d, int size) AVS_NOEXCEPT {
  assert(map && key && size >= 0);
  if (size < 0)
    return 1;
  std::string skey = key;
  if (!isValidVSMapKey(skey))
    return 1;
  FramePropVariant l(FramePropVariant::vFloat);
  l.setArray(d, size);
  map->insert(skey, std::move(l));
  return 0;
}

AVSMap* ScriptEnvironment::createMap() AVS_NOEXCEPT {
  return new AVSMap();
}

void ScriptEnvironment::freeMap(AVSMap* map) AVS_NOEXCEPT {
  delete map;
}

void ScriptEnvironment::clearMap(AVSMap* map) AVS_NOEXCEPT {
  assert(map);
  map->clear();
}
// end of frame prop support functions

PDevice ScriptEnvironment::GetDevice(AvsDeviceType device_type, int device_index) const
{
  return Devices->GetDevice(device_type, device_index);
}

int ScriptEnvironment::SetMemoryMax(AvsDeviceType type, int index, int mem)
{
  return Devices->GetDevice(type, index)->SetMemoryMax(mem);
}

PVideoFrame ScriptEnvironment::GetOnDeviceFrame(const PVideoFrame& src, Device* device)
{
  typedef int diff_t;

  size_t srchead = GetFrameHead(src);

  // make space for alignment
  size_t size = GetFrameTail(src) - srchead;

  VideoFrame *res = GetNewFrame(size, frame_align - 1, device);

  const diff_t offset = (diff_t)(AlignPointer(res->vfb->GetWritePtr(), frame_align) - res->vfb->GetWritePtr()); // first line offset for proper alignment
  const diff_t diff = offset - (diff_t)srchead;

  res->offset = src->offset + diff;
  res->pitch = src->pitch;
  res->row_size = src->row_size;
  res->height = src->height;
  res->offsetU = src->pitchUV ? (src->offsetU + diff) : res->offset;
  res->offsetV = src->pitchUV ? (src->offsetV + diff) : res->offset;
  res->pitchUV = src->pitchUV;
  res->row_sizeUV = src->row_sizeUV;
  res->heightUV = src->heightUV;
  res->offsetA = src->pitchA ? (src->offsetA + diff) : 0;
  res->pitchA = src->pitchA;
  res->row_sizeA = src->row_sizeA;
  *res->properties = *src->properties;
  return PVideoFrame(res);
}


ThreadPool* ScriptEnvironment::NewThreadPool(size_t nThreads)
{
  ThreadPool* pool = new ThreadPool(nThreads, nTotalThreads, threadEnv.get());
  ThreadPoolRegistry.emplace_back(pool);

  nTotalThreads += nThreads;

  if (nMaxFilterInstances < nThreads + 1) {
    // make 2^n
    nMaxFilterInstances = 1;
    while (nThreads + 1 > (nMaxFilterInstances <<= 1));
  }

  // Since this method basically enables MT operation,
  // upgrade all MTGuards to MT-mode.
  for (MTGuard* guard : MTGuardRegistry)
  {
    if (guard != NULL)
      guard->EnableMT(nMaxFilterInstances);
  }

  return pool;
}

void ScriptEnvironment::SetDeviceOpt(DeviceOpt opt, int val)
{
  Devices->SetDeviceOpt(opt, val, threadEnv.get());
}

void ScriptEnvironment::UpdateFunctionExports(const char* funcName, const char* funcParams, const char *exportVar)
{
  std::unique_lock<std::recursive_mutex> env_lock(plugin_mutex);
  plugin_manager->UpdateFunctionExports(funcName, funcParams, exportVar);
}

extern void ApplyMessage(PVideoFrame* frame, const VideoInfo& vi,
  const char* message, int size, int textcolor, int halocolor, int bgcolor,
  IScriptEnvironment* env);


const AVS_Linkage* ScriptEnvironment::GetAVSLinkage() {
  extern const AVS_Linkage* const AVS_linkage; // In interface.cpp

  return AVS_linkage;
}


void ScriptEnvironment::ApplyMessage(PVideoFrame* frame, const VideoInfo& vi, const char* message, int size, int textcolor, int halocolor, int bgcolor)
{
#ifdef ENABLE_CUDA
  if ((*frame)->GetDevice()->device_type == DEV_TYPE_CUDA) {
    // if frame is CUDA frame, copy to CPU and apply
    PVideoFrame copy = GetOnDeviceFrame(*frame, Devices->GetCPUDevice());
    CopyCUDAFrame(copy, *frame, threadEnv.get(), true);
    ::ApplyMessage(&copy, vi, message, size, textcolor, halocolor, bgcolor, threadEnv.get());
    CopyCUDAFrame(*frame, copy, threadEnv.get(), true);
  }
  else
#endif
  {
    ::ApplyMessage(frame, vi, message, size, textcolor, halocolor, bgcolor, threadEnv.get());
  }
}


void ScriptEnvironment::DeleteScriptEnvironment() {
  // Provide a method to delete this ScriptEnvironment in
  // the same malloc context in which it was created below.
  delete this;
}


AVSC_API(IScriptEnvironment*, CreateScriptEnvironment)(int version) {
  return CreateScriptEnvironment2(version);
}

AVSC_API(IScriptEnvironment2*, CreateScriptEnvironment2)(int version)
{
  /* Some plugins use OpenMP. But OMP threads do not exit immediately
  * after all work is exhausted, and keep spinning for a small amount
  * of time waiting for new jobs. If we unload the OMP DLL (indirectly
  * by unloading its plugin that started it) while its threads are
  * running, the sky comes crashing down. This results in crashes
  * from OMP plugins if the IScriptEnvironment is destructed shortly
  * after a GetFrame() call.
  *
  * OMP_WAIT_POLICY=passive changes the behavior of OMP thread pools
  * to shut down immediately instead of continuing to spin.
  * This solves our problem at the cost of some performance.
  */
#ifdef AVS_WINDOWS
  _putenv("OMP_WAIT_POLICY=passive");
#endif

  // When a CPP plugin explicitely requests avs2.5 interface
  if (version <= AVISYNTH_CLASSIC_INTERFACE_VERSION_25) {
    auto IEnv25 = (new ScriptEnvironment())->GetMainThreadEnv()->GetEnv25();
    // return a disguised IScriptEnvironment_Avs25
    return reinterpret_cast<IScriptEnvironment2 *>(IEnv25);
  }
  else if (version <= AVISYNTH_INTERFACE_VERSION)
    return (new ScriptEnvironment())->GetMainThreadEnv();
  else
    return NULL;
}

FramePropVariant::FramePropVariant(FramePropVType vtype) : vtype(vtype), internalSize(0), storage(nullptr) {
}

FramePropVariant::FramePropVariant(const FramePropVariant& v) : vtype(v.vtype), internalSize(v.internalSize), storage(nullptr) {
  if (internalSize) {
    switch (vtype) {
    case FramePropVariant::vInt:
      storage = new IntList(*reinterpret_cast<IntList*>(v.storage)); break;
    case FramePropVariant::vFloat:
      storage = new FloatList(*reinterpret_cast<FloatList*>(v.storage)); break;
    case FramePropVariant::vData:
      storage = new DataList(*reinterpret_cast<DataList*>(v.storage)); break;
    case FramePropVariant::vClip:
      storage = new ClipList(*reinterpret_cast<ClipList*>(v.storage)); break;
    case FramePropVariant::vFrame:
      storage = new FrameList(*reinterpret_cast<FrameList*>(v.storage)); break;
/*    case FramePropVariant::vMethod:
      storage = new FuncList(*reinterpret_cast<FuncList*>(v.storage)); break;*/
    default:;
    }
  }
}

FramePropVariant::FramePropVariant(FramePropVariant&& v) : vtype(v.vtype), internalSize(v.internalSize), storage(v.storage) {
  v.vtype = vUnset;
  v.storage = nullptr;
  v.internalSize = 0;
}

FramePropVariant::~FramePropVariant() {
  if (storage) {
    switch (vtype) {
    case FramePropVariant::vInt:
      delete reinterpret_cast<IntList*>(storage); break;
    case FramePropVariant::vFloat:
      delete reinterpret_cast<FloatList*>(storage); break;
    case FramePropVariant::vData:
      delete reinterpret_cast<DataList*>(storage); break;
    case FramePropVariant::vClip:
      delete reinterpret_cast<ClipList*>(storage); break;
    case FramePropVariant::vFrame:
      delete reinterpret_cast<FrameList*>(storage); break;
/*    case FramePropVariant::vMethod:
      delete reinterpret_cast<FuncList*>(storage); break;*/
    default:;
    }
  }
}

size_t FramePropVariant::size() const {
  return internalSize;
}

FramePropVariant::FramePropVType FramePropVariant::getType() const {
  return vtype;
}

void FramePropVariant::append(int64_t val) {
  initStorage(vInt);
  reinterpret_cast<IntList*>(storage)->push_back(val);
  internalSize++;
}

void FramePropVariant::append(double val) {
  initStorage(vFloat);
  reinterpret_cast<FloatList*>(storage)->push_back(val);
  internalSize++;
}

void FramePropVariant::append(const std::string& val) {
  initStorage(vData);
  reinterpret_cast<DataList*>(storage)->push_back(std::make_shared<std::string>(val));
  internalSize++;
}

void FramePropVariant::append(const PClip& val) {
  initStorage(vClip);
  reinterpret_cast<ClipList*>(storage)->push_back(val);
  internalSize++;
}

void FramePropVariant::append(const PVideoFrame& val) {
  initStorage(vFrame);
  reinterpret_cast<FrameList*>(storage)->push_back(val);
  internalSize++;
}

/*
void FramePropVariant::append(const PExtFunction& val) {
  initStorage(vMethod);
  reinterpret_cast<FuncList*>(storage)->push_back(val);
  internalSize++;
}
*/

void FramePropVariant::initStorage(FramePropVType t) {
  assert(vtype == vUnset || vtype == t);
  vtype = t;
  if (!storage) {
    switch (t) {
    case FramePropVariant::vInt:
      storage = new IntList(); break;
    case FramePropVariant::vFloat:
      storage = new FloatList(); break;
    case FramePropVariant::vData:
      storage = new DataList(); break;
    case FramePropVariant::vClip:
      storage = new ClipList(); break;
    case FramePropVariant::vFrame:
      storage = new FrameList(); break;
/*    case FramePropVariant::vMethod:
      storage = new FuncList(); break;*/
    default:;
    }
  }
}

///////////////

