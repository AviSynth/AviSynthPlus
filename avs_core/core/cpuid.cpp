// Avisynth v1.0 beta.  Copyright 2000 Ben Rudiak-Gould.
// http://www.math.berkeley.edu/~benrg/avisynth.html

//	VirtualDub - Video processing and capture application
//	Copyright (C) 1998-2000 Avery Lee
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#include <avs/cpuid.h>
#include <avs/config.h>
#include <stdint.h>
#ifdef AVS_WINDOWS
#include <intrin.h>
#else
#if defined(X86_32) || defined(X86_64)
#include <x86intrin.h>
#include <cpuid.h>
#undef __cpuid

static inline void __cpuid(int cpuinfo[4], int leaf) {
  unsigned int eax, ebx, ecx, edx;
  // for deeper leaves __get_cpuid is not enough
  __get_cpuid_count(leaf, 0, &eax, &ebx, &ecx, &edx);
  cpuinfo[0] = eax;
  cpuinfo[1] = ebx;
  cpuinfo[2] = ecx;
  cpuinfo[3] = edx;
}
#endif
#endif

#define IS_BIT_SET(bitfield, bit) ((bitfield) & (1<<(bit)) ? true : false)

#if defined(X86_32) || defined(X86_64)
static uint32_t get_xcr0()
{
    uint32_t xcr0;
    // _XCR_XFEATURE_ENABLED_MASK: 0
#if defined(GCC) || defined(CLANG)
    __asm__("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx");
#else
    xcr0 = (uint32_t)_xgetbv(0);
#endif
    return xcr0;
}
#endif

static int CPUCheckForExtensions()
{
  int result = 0;
  int cpuinfo[4];

#if defined(X86_32) || defined(X86_64)
  __cpuid(cpuinfo, 1);
  if (IS_BIT_SET(cpuinfo[3], 0))
    result |= CPUF_FPU;
  if (IS_BIT_SET(cpuinfo[3], 23))
    result |= CPUF_MMX;
  if (IS_BIT_SET(cpuinfo[3], 25))
    result |= CPUF_SSE | CPUF_INTEGER_SSE;
  if (IS_BIT_SET(cpuinfo[3], 26))
    result |= CPUF_SSE2;
  if (IS_BIT_SET(cpuinfo[2], 0))
    result |= CPUF_SSE3;
  if (IS_BIT_SET(cpuinfo[2], 9))
    result |= CPUF_SSSE3;
  if (IS_BIT_SET(cpuinfo[2], 19))
    result |= CPUF_SSE4_1;
  if (IS_BIT_SET(cpuinfo[2], 20))
    result |= CPUF_SSE4_2;
  if (IS_BIT_SET(cpuinfo[2], 22))
    result |= CPUF_MOVBE;
  if (IS_BIT_SET(cpuinfo[2], 23))
    result |= CPUF_POPCNT;
  if (IS_BIT_SET(cpuinfo[2], 25))
    result |= CPUF_AES;
  if (IS_BIT_SET(cpuinfo[2], 29))
    result |= CPUF_F16C;
  // AVX
  bool xgetbv_supported = IS_BIT_SET(cpuinfo[2], 27);
  bool avx_supported = IS_BIT_SET(cpuinfo[2], 28);
  if (xgetbv_supported && avx_supported)
  {
    uint32_t xgetbv0_32 = get_xcr0();
    if ((xgetbv0_32 & 0x6u) == 0x6u) {
      result |= CPUF_AVX;
      if (IS_BIT_SET(cpuinfo[2], 12))
        result |= CPUF_FMA3;
      __cpuid(cpuinfo, 7);
      if (IS_BIT_SET(cpuinfo[1], 5))
        result |= CPUF_AVX2;
    }
    if((xgetbv0_32 & (0x7u << 5)) && // OPMASK: upper-256 enabled by OS
       (xgetbv0_32 & (0x3u << 1))) { // XMM/YMM enabled by OS
      // Verify that XCR0[7:5] = ‘111b’ (OPMASK state, upper 256-bit of ZMM0-ZMM15 and
      // ZMM16-ZMM31 state are enabled by OS)
      /// and that XCR0[2:1] = ‘11b’ (XMM state and YMM state are enabled by OS).
      __cpuid(cpuinfo, 7);
      if (IS_BIT_SET(cpuinfo[1], 16))
        result |= CPUF_AVX512F;
      if (IS_BIT_SET(cpuinfo[1], 17))
        result |= CPUF_AVX512DQ;
      if (IS_BIT_SET(cpuinfo[1], 21))
        result |= CPUF_AVX512IFMA;
      if (IS_BIT_SET(cpuinfo[1], 26))
        result |= CPUF_AVX512PF;
      if (IS_BIT_SET(cpuinfo[1], 27))
        result |= CPUF_AVX512ER;
      if (IS_BIT_SET(cpuinfo[1], 28))
        result |= CPUF_AVX512CD;
      if (IS_BIT_SET(cpuinfo[1], 30))
        result |= CPUF_AVX512BW;
      if (IS_BIT_SET(cpuinfo[1], 31))
        result |= CPUF_AVX512VL;
      if (IS_BIT_SET(cpuinfo[2], 1)) // [2]!
        result |= CPUF_AVX512VBMI;
    }
#else
    result |= CPUF_FORCE;

    return result;
#endif
  }

#if defined(X86_32) || defined(X86_64)
  // 3DNow!, 3DNow!, ISSE, FMA4
  __cpuid(cpuinfo, (int)0x80000000);
  if (cpuinfo[0] >= (int)0x80000001)
  {
    __cpuid(cpuinfo, (int)0x80000001);

    if (IS_BIT_SET(cpuinfo[3], 31))
      result |= CPUF_3DNOW;

    if (IS_BIT_SET(cpuinfo[3], 30))
      result |= CPUF_3DNOW_EXT;

    if (IS_BIT_SET(cpuinfo[3], 22))
      result |= CPUF_INTEGER_SSE;

    if (result & CPUF_AVX) {
      if (IS_BIT_SET(cpuinfo[2], 16))
        result |= CPUF_FMA4;
    }
  }

  return result;
}
#endif

class _CPUFlags
{
private:
  int lCPUExtensionsAvailable;
  _CPUFlags() { lCPUExtensionsAvailable = CPUCheckForExtensions(); }

public:
  static _CPUFlags& getInstance() {
    static _CPUFlags theInstance;
    return theInstance;
  }

  int GetCPUFlags() {
    return lCPUExtensionsAvailable;
  }

  void SetCPUFlags(int new_flags) {
    lCPUExtensionsAvailable = new_flags;
  }
};

int GetCPUFlags() {
  return _CPUFlags::getInstance().GetCPUFlags();
}

void SetMaxCPU(int new_flags) {
  _CPUFlags::getInstance().SetCPUFlags(new_flags);
}
