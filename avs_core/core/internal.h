// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
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


#ifndef __Internal_H__
#define __Internal_H__

#include <avs/config.h>
#include <avs/minmax.h>
#include <stdint.h>
#include <string.h>
#include "version.h"
#include <memory>
#include <string>
#ifdef AVS_POSIX
#include <limits.h>
#endif
#include "InternalEnvironment.h"
#ifdef INTEL_INTRINSICS
#ifdef AVS_WINDOWS
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#if AVS_DEVNEXT_REV > 0
#define AVS_DEVELOPMENT_BUILD "\nNext Version Development Build: " AVS_PPSTR(AVS_DEV_REVDATE)
#define AVS_DEVELOPMENT_BUILD_GIT "\nGit: " AVS_PPSTR(AVS_DEV_GITHASH)
#else
#define AVS_DEVELOPMENT_BUILD
#define AVS_DEVELOPMENT_BUILD_GIT
#endif

#define AVS_CLASSIC_VERSION 2.60  // Note: Used by VersionNumber() script function
#define AVS_COPYRIGHT "\n\xA9 2000-2015 Ben Rudiak-Gould, et al.\nhttp://avisynth.nl\n\xA9 2013-2023 AviSynth+ Project"
#define AVS_COPYRIGHT_UTF8 u8"\n\u00A9 2000-2015 Ben Rudiak-Gould, et al.\nhttp://avisynth.nl\n\u00A9 2013-2023 AviSynth+ Project"
#define BUILTIN_FUNC_PREFIX "AviSynth"

enum MANAGE_CACHE_KEYS
{
  MC_RegisterCache     = (int)0xFFFF0004,
  MC_UnRegisterCache   = (int)0xFFFF0006,
  MC_NodCache          = (int)0xFFFF0007,
  MC_NodAndExpandCache = (int)0xFFFF0008,
  MC_RegisterMTGuard,
  MC_UnRegisterMTGuard,

	MC_RegisterGraphNode = (int)0xFFFF0100,
	MC_UnRegisterGraphNode,

  MC_QueryAvs25        = (int)0xFFFF0200,
};

#include <avisynth.h>
#ifdef INTEL_INTRINSICS
#include <emmintrin.h>
#endif
#include <string>
#include "function.h"


const char *GetPixelTypeName(const int pixel_type); // in script.c
int GetPixelTypeFromName(const char *pixeltypename); // in script.c
const char* GetAVSTypeName(AVSValue value); // in script.c
int GetDeviceTypes(const PClip& child); // in DeviceManager.cpp
size_t GetFrameHead(const PVideoFrame& vf); // in DeviceManager.cpp
size_t GetFrameTail(const PVideoFrame& vf); // in DeviceManager.cpp

PClip Create_MessageClip(const char* message, int width, int height,
  int pixel_type, bool shrink, int textcolor, int halocolor, int bgcolor,
  int fps_numerator, int fps_denominator, int num_frames,
  IScriptEnvironment* env);

PClip new_Splice(PClip _child1, PClip _child2, bool realign_sound, IScriptEnvironment* env);
PClip new_SeparateFields(PClip _child, IScriptEnvironment* env);
PClip new_AssumeFrameBased(PClip _child);


/* Used to clip/clamp a byte to the 0-255 range.
   Uses a look-up table internally for performance.
*/
class _PixelClip {
  enum { buffer=320 };
  BYTE lut[256+buffer*2];
public:
  _PixelClip() {
    memset(lut, 0, buffer);
    for (int i=0; i<256; ++i) lut[i+buffer] = (BYTE)i;
    memset(lut+buffer+256, 255, buffer);
  }
  BYTE operator()(int i) const { return lut[i+buffer]; }
};

extern const _PixelClip PixelClip;


template<class ListNode>
static __inline void Relink(ListNode* newprev, ListNode* me, ListNode* newnext) {
  if (me == newprev || me == newnext) return;
  me->next->prev = me->prev;
  me->prev->next = me->next;
  me->prev = newprev;
  me->next = newnext;
  me->prev->next = me->next->prev = me;
}

class CWDChanger
/**
  * Class to change the current working directory
 **/
{
public:
  CWDChanger(const char* new_cwd);
  CWDChanger(const wchar_t* new_cwd);
  ~CWDChanger(void);

private:
  void Init(const wchar_t* new_cwd);
#ifdef AVS_WINDOWS
  std::unique_ptr<wchar_t[]> old_working_directory;
#else
  char old_working_directory[PATH_MAX];
#endif
  bool restore;
};

class DllDirChanger
{
public:
  DllDirChanger(const char* new_cwd);
  ~DllDirChanger(void);

private:
  std::unique_ptr<char[]> old_directory;
  bool restore;
};

class NonCachedGenericVideoFilter : public GenericVideoFilter
/**
  * Class to select a range of frames from a longer clip
 **/
{
public:
  NonCachedGenericVideoFilter(PClip _child);
  int __stdcall SetCacheHints(int cachehints, int frame_range);
};



/*** Inline helper methods ***/

// 8 bit uv to float
// 16-128-240 -> -112-0-112 -> 1..255: +/-127 -> +/-0.5
[[maybe_unused]] static AVS_FORCEINLINE float uv8tof(int color) {
  return (color - 128) / 255.f; // consistent with convert_uintN_to_float_c
}

// 8 bit fullscale to float
[[maybe_unused]] static AVS_FORCEINLINE float c8tof(int color) {
  return color / 255.0f;
}

[[maybe_unused]] static AVS_FORCEINLINE uint8_t Scaled15bitPixelClip(int i) {
  return (uint8_t)clamp((i + 16384) >> 15, 0, 255);
}

[[maybe_unused]] static AVS_FORCEINLINE uint8_t ScaledPixelClip(int i) {
  // return PixelClip((i+32768) >> 16);
  // PF: clamp is faster than lut
  return (uint8_t)clamp((i + 32768) >> 16, 0, 255);
}

[[maybe_unused]] static AVS_FORCEINLINE uint16_t ScaledPixelClip(int64_t i) {
    return (uint16_t)clamp((i + 32768) >> 16, (int64_t)0, (int64_t)65535);
}

[[maybe_unused]] static AVS_FORCEINLINE uint16_t ScaledPixelClipEx(int64_t i, int max_value) {
  return (uint16_t)clamp((int)((i + 32768) >> 16), 0, max_value);
}

[[maybe_unused]] static AVS_FORCEINLINE bool IsClose(int a, int b, unsigned threshold)
  { return (unsigned(a-b+threshold) <= threshold*2); }

[[maybe_unused]] static AVS_FORCEINLINE bool IsCloseFloat(float a, float b, float threshold)
{ return (a-b+threshold <= threshold*2); }

#ifdef INTEL_INTRINSICS
// useful SIMD helpers

// sse2 replacement of _mm_mullo_epi32 in SSE4.1
// use it after speed test, may have too much overhead and C is faster
[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_MULLO_EPI32(const __m128i &a, const __m128i &b)
{
  // for SSE 4.1: return _mm_mullo_epi32(a, b);
  __m128i tmp1 = _mm_mul_epu32(a,b); // mul 2,0
  __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); // mul 3,1
  // shuffle results to [63..0] and pack. a2->a1, a0->a0
  return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0)));
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4309)
#endif
// fake _mm_packus_epi32 (orig is SSE4.1 only)
[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_PACKUS_EPI32( __m128i a, __m128i b )
{
  const __m128i val_32 = _mm_set1_epi32(0x8000);
  const __m128i val_16 = _mm_set1_epi16(0x8000);

  a = _mm_sub_epi32(a, val_32);
  b = _mm_sub_epi32(b, val_32);
  a = _mm_packs_epi32(a, b);
  a = _mm_add_epi16(a, val_16);
  return a;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
// fake _mm_packus_epi32 (orig is SSE4.1 only)
// only for packing 00000000..0000FFFF range integers, does not clamp properly above that, e.g. 00010001
[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_PACKUS_EPI32_SRC_TRUEWORD(__m128i a, __m128i b)
{
  a = _mm_slli_epi32 (a, 16);
  a = _mm_srai_epi32 (a, 16);
  b = _mm_slli_epi32 (b, 16);
  b = _mm_srai_epi32 (b, 16);
  a = _mm_packs_epi32 (a, b);
  return a;
}

[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_CMPLE_EPU16(__m128i x, __m128i y)
{
  // Returns 0xFFFF where x <= y:
  return _mm_cmpeq_epi16(_mm_subs_epu16(x, y), _mm_setzero_si128());
}

[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_BLENDV_SI128(__m128i x, __m128i y, __m128i mask)
{
  // Replace bit in x with bit in y when matching bit in mask is set:
  return _mm_or_si128(_mm_andnot_si128(mask, x), _mm_and_si128(mask, y));
}

// sse2 simulation of SSE4's _mm_min_epu16
[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_MIN_EPU16(__m128i x, __m128i y)
{
  // Returns x where x <= y, else y:
  return _MM_BLENDV_SI128(y, x, _MM_CMPLE_EPU16(x, y));
}

// sse2 simulation of SSE4's _mm_max_epu16
[[maybe_unused]] static AVS_FORCEINLINE __m128i _MM_MAX_EPU16(__m128i x, __m128i y)
{
  // Returns x where x >= y, else y:
  return _MM_BLENDV_SI128(x, y, _MM_CMPLE_EPU16(x, y));
}
#endif

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3)                              \
                ((DWORD)(BYTE)(ch0) | ((DWORD)(BYTE)(ch1) << 8) |   \
                ((DWORD)(BYTE)(ch2) << 16) | ((DWORD)(BYTE)(ch3) << 24 ))
#endif

class GlobalVarFrame
{
   InternalEnvironment* env;
public:
   GlobalVarFrame(InternalEnvironment* env) : env(env) {
      env->PushContextGlobal();
   }
   ~GlobalVarFrame() {
      env->PopContextGlobal();
   }
};

#endif  // __Internal_H__
